"""Search one recording's DSS parameters, one stage at a time.

Addressed by array index so a failed task can be requeued with the same index
and no task writes another subject's file.

The stages are searched in the order ``clean_dss`` runs them, and each stage's
grid is scored on the input that stage actually receives: the cardiac grid on
the trunk, the ocular grid on the trunk with production cardiac removed, the
muscle grid with production cardiac *and* ocular removed. Those two upstream
states cost nothing extra -- every stage's grid contains its production
configuration, so the row that is the reference is also the row that becomes
the next stage's input.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import mne
import numpy as np

from meg_tokens.meg.meg_bids import raw_meg_path, staged_subjects
from meg_tokens.meg.preprocessing import (
    ECG_CHANNEL,
    OCULAR_BAND,
    VEOG_CHANNEL,
    artifact_events,
    build_trunk,
    dss_subtract,
    ocular_events,
    read_staged_raw,
)

from scripts.qc.dss_param_search import grid as grid_module
from scripts.qc.dss_param_search.metrics import (
    ALPHA_BAND,
    artifact_presence,
    band_power_db,
    evoked_rms,
    reference_correlation,
)

MIN_EVENTS = 10
MUSCLE_SCORING_BAND = (110.0, 140.0)


def _trim_to_complete_windows(
    samples: np.ndarray, window: tuple[float, float], sfreq: float, n_times: int
) -> np.ndarray:
    """CycleAverageBias requires a complete window for every event it averages."""
    first, last = int(round(window[0] * sfreq)), int(round(window[1] * sfreq))
    samples = np.asarray(samples, dtype=int)
    return samples[(samples + first >= 0) & (samples + last <= n_times)]


def _build_bias(config: grid_module.Config, events: dict[str, np.ndarray], sfreq: float, n_times: int):
    """The bias for one configuration, and how many events it averaged."""
    from mne_denoise.dss.denoisers import BandpassBias, CycleAverageBias

    if config.stage == "muscle":
        return BandpassBias(
            freq_band=config.params["freq_band"],
            sfreq=sfreq,
            order=config.params["order"],
        ), 0

    key = "ecg" if config.stage == "qrs" else "blink"
    window = config.params["window"]
    samples = _trim_to_complete_windows(events[key], window, sfreq, n_times)
    if samples.size < MIN_EVENTS:
        raise ValueError(f"{key}: {samples.size} usable events, too few to average")
    return CycleAverageBias(
        event_samples=samples, window=window, window_unit="seconds", sfreq=sfreq
    ), int(samples.size)


def _score(
    stage: str,
    before: mne.io.BaseRaw,
    after: mne.io.BaseRaw,
    picks: np.ndarray,
    scoring_events: np.ndarray | None,
    reference: np.ndarray | None,
    baseline: dict[str, float],
) -> dict[str, float]:
    """Removal, reference correlation and alpha cost for one configuration.

    ``scoring_events`` and ``reference`` are the *production* event set and
    reference channel, never the candidate's own. A configuration that changes
    the event set would otherwise be scored against the events it chose, which
    is how a detector that finds three large blinks scores perfectly.
    """
    removed = before.get_data(picks=picks) - after.get_data(picks=picks)
    scores = {
        "alpha_db": band_power_db(after, picks, ALPHA_BAND),
        "muscle_band_db": band_power_db(after, picks, MUSCLE_SCORING_BAND),
    }
    scores["alpha_delta_db"] = scores["alpha_db"] - baseline["alpha_db"]
    scores["muscle_reduction_db"] = baseline["muscle_band_db"] - scores["muscle_band_db"]
    if stage == "muscle":
        scores["removal_percent"] = float("nan")
        scores["reference_r"] = float("nan")
    else:
        after_rms = evoked_rms(after, picks, scoring_events)
        scores["removal_percent"] = 100.0 * (1.0 - after_rms / baseline["evoked_rms"])
        scores["reference_r"] = reference_correlation(removed, reference)
    del removed
    return scores


def _reference_signal(trunk: mne.io.BaseRaw, stage: str) -> np.ndarray | None:
    """The channel a stage's removed subspace is checked against."""
    if stage == "qrs":
        return trunk.copy().pick([ECG_CHANNEL]).get_data()[0]
    if stage == "blink":
        return (
            trunk.copy()
            .pick([VEOG_CHANNEL])
            .filter(*OCULAR_BAND, fir_design="firwin", verbose=False)
            .get_data()[0]
        )
    return None


def search_recording(
    path: Path,
    *,
    stages: tuple[str, ...] = grid_module.STAGES,
) -> dict:
    """Every configuration for every requested stage, on one recording."""
    raw, source_grade = read_staged_raw(path)
    trunk, trunk_report = build_trunk(raw, source_grade)
    del raw
    events = artifact_events(trunk)
    picks = mne.pick_types(trunk.info, meg=True, ref_meg=False, eeg=False, exclude=[])
    sfreq = float(trunk.info["sfreq"])

    presence = artifact_presence(trunk, events, picks)
    blink_cache: dict[float, np.ndarray] = {}
    rows: list[dict] = []
    current = trunk

    for stage in grid_module.STAGES:
        production = grid_module.production_config(stage)
        reference = _reference_signal(trunk, stage)
        scoring_events = None if stage == "muscle" else events["ecg" if stage == "qrs" else "blink"]
        baseline = {
            "alpha_db": band_power_db(current, picks, ALPHA_BAND),
            "muscle_band_db": band_power_db(current, picks, MUSCLE_SCORING_BAND),
            "evoked_rms": (
                float("nan") if stage == "muscle" else evoked_rms(current, picks, scoring_events)
            ),
        }
        # The next stage's input, whether or not this stage was searched.
        next_input: mne.io.BaseRaw | None = None
        configs = grid_module.stage_grid(stage) if stage in stages else [production]

        for config in configs:
            stage_events = dict(events)
            if stage == "blink":
                multiple = config.params["scale_multiple"]
                if multiple not in blink_cache:
                    blink_cache[multiple] = ocular_events(
                        trunk, VEOG_CHANNEL, scale_multiple=multiple
                    )
                stage_events["blink"] = blink_cache[multiple]

            started = time.monotonic()
            try:
                bias, n_events = _build_bias(config, stage_events, sfreq, current.n_times)
                cleaned, n_selected = dss_subtract(current, bias, **config.selection_params)
            except Exception as error:  # a named failure, not a silent skip
                rows.append({
                    "stage": stage, "label": config.label, "params": config.params,
                    "is_production": config.is_production, "failed": repr(error),
                    "seconds": round(time.monotonic() - started, 1),
                })
                continue

            row = {
                "stage": stage,
                "label": config.label,
                "params": config.params,
                "is_production": config.is_production,
                "n_selected": int(n_selected),
                "n_events": n_events,
                "seconds": round(time.monotonic() - started, 1),
            }
            row.update(_score(stage, current, cleaned, picks, scoring_events, reference, baseline))
            rows.append(row)

            if config.is_production:
                next_input = cleaned
            else:
                del cleaned

        if next_input is None:
            raise RuntimeError(f"{stage}: production configuration produced no output")
        if current is not trunk:
            del current
        current = next_input

    del current
    return {
        "recording": str(path),
        "presence": presence,
        "trunk": {
            "compensation_grade_source": trunk_report.compensation_grade_source,
            "compensation_applied": trunk_report.compensation_applied,
            "sfreq": trunk_report.sfreq,
            "duration_seconds": trunk_report.duration_seconds,
            "n_meg_channels": trunk_report.n_meg_channels,
            "n_bad_channels": len(trunk_report.bad_channels),
            "annotated_seconds": trunk_report.annotated_seconds,
        },
        "rows": rows,
    }


def _commit() -> str:
    """Provenance, including for a checkout without a ``.git``."""
    recorded = os.environ.get("DSS_SEARCH_SOURCE_COMMIT")
    if recorded:
        return recorded
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    """The runner's command line, reachable so the launcher can be tested against it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", type=int, help="array index into the staged subjects")
    group.add_argument("--subject")
    parser.add_argument("--condition", default="Fast")
    parser.add_argument("--run", default=1, type=int)
    parser.add_argument("--stages", nargs="+", default=list(grid_module.STAGES),
                        choices=list(grid_module.STAGES))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    subjects = staged_subjects(args.bids_root)
    if args.index is not None:
        if args.index >= len(subjects):
            print(f"array index {args.index} outside {len(subjects)} staged subjects")
            return 0
        subject = subjects[args.index]
    else:
        subject = args.subject

    path = raw_meg_path(args.bids_root, subject, args.condition, args.run)
    if not path.is_dir():
        raise SystemExit(f"not staged: {path}")

    mne.set_log_level("ERROR")
    started = time.monotonic()
    result = search_recording(path, stages=tuple(args.stages))
    result.update({
        "subject": subject,
        "condition": args.condition,
        "run": args.run,
        "fold": grid_module.folds(subjects)[subject],
        "commit": _commit(),
        "total_seconds": round(time.monotonic() - started, 1),
    })

    args.output_root.mkdir(parents=True, exist_ok=True)
    out = args.output_root / f"sub-{subject}_dsssearch.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"{subject}: {len(result['rows'])} rows in {result['total_seconds']}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
