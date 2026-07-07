"""
Stage 4 source-space frequency-band power extraction.

This utility consumes the Stage 3 source-estimate manifest and writes
BIDS-derivatives-style array tensors for downstream decoding, PCA, and
trajectory analyses.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Optional, Sequence

import mne
import numpy as np
import pandas as pd

from meg_tokens.io import derivative_path, require_file, save_array
from meg_tokens.meg.sources import source_derivative_path
from meg_tokens.meg.time_frequency import (
    DEFAULT_BANDS,
    compute_band_power,
    compute_window_times,
    rescale_baseline,
)
from meg_tokens.utils.batch_processor import normalize_subject_id
from meg_tokens.utils.epochs_builder import parse_run_label


def parse_frequency_bands(items: Optional[Sequence[str]]) -> dict[str, tuple[float, float]]:
    """Parse band selectors such as ``alpha`` or ``alpha=8,15``."""
    if not items:
        return dict(DEFAULT_BANDS)

    bands: dict[str, tuple[float, float]] = {}
    for item in items:
        if "=" not in item:
            if item not in DEFAULT_BANDS:
                valid = ", ".join(sorted(DEFAULT_BANDS))
                raise ValueError(f"Unknown frequency band '{item}'. Known bands: {valid}")
            bands[item] = DEFAULT_BANDS[item]
            continue

        name, bounds = item.split("=", 1)
        parts = [part.strip() for part in bounds.replace(":", ",").split(",") if part.strip()]
        if len(parts) != 2:
            raise ValueError(f"Band '{item}' must be formatted as name=fmin,fmax")
        fmin, fmax = float(parts[0]), float(parts[1])
        if fmin <= 0 or fmax <= fmin:
            raise ValueError(f"Band '{item}' must satisfy 0 < fmin < fmax")
        bands[name.strip()] = (fmin, fmax)

    return bands


def find_stc_manifest(
    source_dir: str,
    subject: str,
    run: str,
    condition: Optional[str],
    align_to: str,
    source_method: str,
) -> Path:
    """Return the unique Stage 3 source-estimate manifest for one run."""
    subject = normalize_subject_id(subject)
    run_number, inferred_condition = parse_run_label(run)
    condition = condition or inferred_condition
    expected = source_derivative_path(
        source_dir,
        subject,
        suffix="stcmanifest",
        extension=".tsv",
        run_id=run_number,
        condition=condition,
        description=f"{align_to}-{source_method}",
    )
    if expected.is_file():
        return expected

    condition_part = f"{condition.lower()}-" if condition else ""
    pattern = (
        f"**/sub-{subject}_task-tokens_run-{run_number}_"
        f"desc-{condition_part}{align_to}-{source_method}_stcmanifest.tsv"
    )
    matches = sorted(path for path in Path(source_dir).glob(pattern) if path.is_file())
    if not matches:
        raise FileNotFoundError(
            "No Stage 3 source-estimate manifest found for "
            f"subject={subject}, run={run_number}, condition={condition}, "
            f"alignment={align_to}, method={source_method}. Expected: {expected}"
        )
    if len(matches) > 1:
        raise ValueError(f"Multiple source-estimate manifests matched: {matches}")
    return matches[0]


def power_derivative_path(
    output_root: str,
    *,
    subject: str,
    run: str,
    condition: Optional[str],
    align_to: str,
    source_method: str,
    power_method: str,
    band: str,
) -> Path:
    """Build the Stage 4 power derivative path for one band."""
    subject = normalize_subject_id(subject)
    run_number, inferred_condition = parse_run_label(run)
    condition = condition or inferred_condition
    desc_parts = []
    if condition:
        desc_parts.append(condition.lower())
    desc_parts.extend([align_to, source_method, power_method, band.replace("_", "-")])
    return derivative_path(
        output_root,
        subject=subject,
        datatype="meg",
        task="tokens",
        run=run_number,
        description="-".join(desc_parts),
        suffix="power",
        extension=".npy",
    )


def _read_manifest(manifest_path: str | Path) -> pd.DataFrame:
    path = require_file(manifest_path, purpose="Stage 3 source-estimate manifest")
    table = pd.read_csv(path, sep="\t")
    required = {"trial", "stc_base"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Manifest {path} is missing required columns: {missing}")
    if table.empty:
        raise ValueError(f"Manifest {path} does not contain any trials")
    return table


def _source_files_for_base(base_path: str | Path) -> list[Path]:
    base = Path(base_path)
    candidates = [
        base,
        base.with_name(f"{base.name}-lh.stc"),
        base.with_name(f"{base.name}-rh.stc"),
        base.with_name(f"{base.name}-vl.stc"),
        base.with_name(f"{base.name}-vol.stc"),
        base.with_suffix(".stc"),
    ]
    return [path for path in candidates if path.is_file()]


def _read_source_estimate(base_path: str | Path):
    if not _source_files_for_base(base_path):
        raise FileNotFoundError(f"Source-estimate files do not exist for base path: {base_path}")
    return mne.read_source_estimate(str(base_path))


def _vertices_equal(left, right) -> bool:
    if len(left) != len(right):
        return False
    return all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(left, right))


def _validate_stc(stc, reference, row_index: int) -> None:
    if stc.__class__ is not reference.__class__:
        raise ValueError(
            f"Source estimate at manifest row {row_index} has type "
            f"{stc.__class__.__name__}; expected {reference.__class__.__name__}"
        )
    if stc.data.shape != reference.data.shape:
        raise ValueError(
            f"Source estimate at manifest row {row_index} has shape "
            f"{stc.data.shape}; expected {reference.data.shape}"
        )
    if not np.isclose(stc.tmin, reference.tmin) or not np.isclose(stc.tstep, reference.tstep):
        raise ValueError(f"Source estimate at manifest row {row_index} has inconsistent timing")
    if not _vertices_equal(stc.vertices, reference.vertices):
        raise ValueError(f"Source estimate at manifest row {row_index} has inconsistent vertices")


def _metadata_value(table: pd.DataFrame, key: str, default=None):
    if key not in table.columns:
        return default
    values = table[key].dropna().unique()
    if len(values) == 0:
        return default
    if len(values) > 1:
        raise ValueError(f"Manifest column '{key}' contains multiple values: {list(values)}")
    return values[0]


def _power_dims(power: np.ndarray) -> tuple[str, ...]:
    if power.ndim == 3:
        return ("trial", "source", "time")
    if power.ndim == 4:
        return ("trial", "source", "orientation", "time")
    raise ValueError(f"Unexpected stacked power shape: {power.shape}")


def _power_coords(power: np.ndarray, trials: Sequence[int], times: np.ndarray) -> dict[str, object]:
    coords: dict[str, object] = {
        "trial": list(trials),
        "time_sec": times,
    }
    if power.ndim == 4:
        coords["orientation"] = ["x", "y", "z"] if power.shape[2] == 3 else list(range(power.shape[2]))
    return coords


def extract_power_from_manifest(
    manifest_path: str | Path,
    out_dir: str,
    *,
    freq_bands: Optional[Mapping[str, tuple[float, float]]] = None,
    method: str = "hilbert",
    width: int = 400,
    step: int = 110,
    n_jobs: int = 1,
    baseline: Optional[tuple[Optional[float], Optional[float]]] = None,
    baseline_method: str = "percent",
) -> dict[str, Path]:
    """Compute and save source-space band-power tensors for one manifest."""
    manifest_path = Path(require_file(manifest_path, purpose="Stage 3 source-estimate manifest"))
    manifest = _read_manifest(manifest_path)
    freq_bands = dict(freq_bands or DEFAULT_BANDS)
    trials = manifest["trial"].astype(int).tolist()

    reference = None
    per_band: dict[str, list[np.ndarray]] = {band: [] for band in freq_bands}

    for row_index, row in manifest.reset_index(drop=True).iterrows():
        stc = _read_source_estimate(row["stc_base"])
        if reference is None:
            reference = stc
        else:
            _validate_stc(stc, reference, row_index)

        sfreq = 1.0 / float(stc.tstep)
        power_results = compute_band_power(
            stc,
            sfreq=sfreq,
            freq_bands=freq_bands,
            method=method,
            width=width,
            step=step,
            n_jobs=n_jobs,
            return_mne=False,
        )
        for band, power in power_results.items():
            per_band[band].append(np.asarray(power))

    if reference is None:
        raise ValueError(f"Manifest {manifest_path} did not yield any source estimates")

    sfreq = 1.0 / float(reference.tstep)
    times = compute_window_times(reference.tmin, sfreq, reference.data.shape[-1], width, step)
    subject = str(_metadata_value(manifest, "subject", reference.subject or ""))
    run = str(_metadata_value(manifest, "run"))
    condition = _metadata_value(manifest, "condition")
    align_to = str(_metadata_value(manifest, "alignment"))
    source_method = str(_metadata_value(manifest, "method"))
    if not subject or run == "None" or align_to == "None" or source_method == "None":
        raise ValueError(f"Manifest {manifest_path} is missing subject, run, alignment, or method metadata")

    saved: dict[str, Path] = {}
    for band, trial_arrays in per_band.items():
        stacked = np.stack(trial_arrays, axis=0)
        if baseline is not None:
            stacked = rescale_baseline(stacked, times, baseline, method=baseline_method)

        out_path = power_derivative_path(
            out_dir,
            subject=subject,
            run=run,
            condition=None if pd.isna(condition) else str(condition),
            align_to=align_to,
            source_method=source_method,
            power_method=method,
            band=band,
        )
        metadata = {
            "stage": "time_frequency_power",
            "kind": "source_band_power",
            "subject": normalize_subject_id(subject),
            "run": parse_run_label(run)[0],
            "condition": None if pd.isna(condition) else str(condition),
            "alignment": align_to,
            "source_method": source_method,
            "power_method": method,
            "band": band,
            "fmin_hz": float(freq_bands[band][0]),
            "fmax_hz": float(freq_bands[band][1]),
            "sfreq_hz": float(sfreq),
            "window_width_samples": int(width),
            "window_step_samples": int(step),
            "baseline_sec": list(baseline) if baseline is not None else None,
            "baseline_method": baseline_method if baseline is not None else None,
            "input_manifest": str(manifest_path),
            "source_estimate_type": reference.__class__.__name__,
            "source_vertices": [np.asarray(vertices).tolist() for vertices in reference.vertices],
        }
        saved[band] = save_array(
            out_path,
            stacked,
            dims=_power_dims(stacked),
            coords=_power_coords(stacked, trials, times),
            metadata=metadata,
        )
        print(f"Saved {band} power: {saved[band]}")

    return saved


def run_time_frequency_pipeline(
    subjects_list: Sequence[str],
    source_dir: str,
    out_dir: str,
    *,
    run: str,
    condition: Optional[str] = None,
    align_to: str = "go",
    source_method: str = "dSPM",
    freq_bands: Optional[Mapping[str, tuple[float, float]]] = None,
    method: str = "hilbert",
    width: int = 400,
    step: int = 110,
    n_jobs: int = 1,
    baseline: Optional[tuple[Optional[float], Optional[float]]] = None,
    baseline_method: str = "percent",
) -> dict[str, dict[str, Path]]:
    """Run Stage 4 power extraction for subjects with Stage 3 manifests."""
    outputs: dict[str, dict[str, Path]] = {}
    for subject in subjects_list:
        subject = normalize_subject_id(subject)
        manifest = find_stc_manifest(source_dir, subject, run, condition, align_to, source_method)
        print(f"=== Extracting source power for {subject}: {manifest} ===")
        outputs[subject] = extract_power_from_manifest(
            manifest,
            out_dir,
            freq_bands=freq_bands,
            method=method,
            width=width,
            step=step,
            n_jobs=n_jobs,
            baseline=baseline,
            baseline_method=baseline_method,
        )
    return outputs


def _parse_baseline(values: Optional[Sequence[float]]) -> Optional[tuple[Optional[float], Optional[float]]]:
    if values is None:
        return None
    if len(values) != 2:
        raise ValueError("--baseline requires two values: start end")
    return (values[0], values[1])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract source-space frequency-band power from Stage 3 manifests.")
    parser.add_argument("--manifest", type=str, nargs="+", default=None,
                        help="One or more Stage 3 stcmanifest TSV files. If omitted, manifests are found from subject/run metadata.")
    parser.add_argument("--subjects", type=str, nargs="+", default=None,
                        help="Subject IDs to process when --manifest is omitted.")
    parser.add_argument("--source_dir", type=str, default=None,
                        help="BIDS derivatives root containing Stage 3 source manifests.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="BIDS derivatives root for Stage 4 power arrays.")
    parser.add_argument("--run", type=str, default=None,
                        help="Run label, for example Slow1 or 1. Required when --manifest is omitted.")
    parser.add_argument("--condition", type=str, default=None,
                        help="Condition label if the run label does not include it.")
    parser.add_argument("--align_to", type=str, default="go", choices=["go", "enter", "feedback"])
    parser.add_argument("--source_method", type=str, default="dSPM",
                        help="Source reconstruction method used in Stage 3.")
    parser.add_argument("--method", type=str, default="hilbert", choices=["hilbert", "morlet", "multitaper"],
                        help="Power computation method.")
    parser.add_argument("--bands", type=str, nargs="+", default=None,
                        help="Bands to compute, using known names or name=fmin,fmax.")
    parser.add_argument("--width", type=int, default=400,
                        help="Sliding window width in samples.")
    parser.add_argument("--step", type=int, default=110,
                        help="Sliding window step in samples.")
    parser.add_argument("--baseline", type=float, nargs=2, default=None,
                        help="Optional baseline interval in seconds.")
    parser.add_argument("--baseline_method", type=str, default="percent",
                        choices=["percent", "ratio", "logratio", "zscore", "difference"])
    parser.add_argument("--n_jobs", type=int, default=1)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    bands = parse_frequency_bands(args.bands)
    baseline = _parse_baseline(args.baseline)

    if args.manifest:
        for manifest in args.manifest:
            extract_power_from_manifest(
                manifest,
                args.out_dir,
                freq_bands=bands,
                method=args.method,
                width=args.width,
                step=args.step,
                n_jobs=args.n_jobs,
                baseline=baseline,
                baseline_method=args.baseline_method,
            )
        return

    if not args.subjects:
        parser.error("--subjects is required when --manifest is omitted")
    if args.source_dir is None:
        parser.error("--source_dir is required when --manifest is omitted")
    if args.run is None:
        parser.error("--run is required when --manifest is omitted")

    run_time_frequency_pipeline(
        args.subjects,
        args.source_dir,
        args.out_dir,
        run=args.run,
        condition=args.condition,
        align_to=args.align_to,
        source_method=args.source_method,
        freq_bands=bands,
        method=args.method,
        width=args.width,
        step=args.step,
        n_jobs=args.n_jobs,
        baseline=baseline,
        baseline_method=args.baseline_method,
    )


if __name__ == "__main__":
    main()
