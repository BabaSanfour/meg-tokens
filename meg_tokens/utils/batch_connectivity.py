"""Stage 10 functional connectivity over staged ERP/parcellation derivatives."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from meg_tokens.io import derivative_path, ensure_dir, load_array, save_array
from meg_tokens.meg.connectivity import compute_spectral_connectivity, infer_sfreq_from_times, select_time_window
from meg_tokens.utils.batch_decoding import _load_feature_array, find_feature_arrays
from meg_tokens.utils.batch_processor import normalize_subject_id
from meg_tokens.utils.epochs_builder import parse_run_label


DEFAULT_FMIN = (2.0, 4.0, 8.0, 15.0)
DEFAULT_FMAX = (4.0, 8.0, 15.0, 30.0)
DEFAULT_BANDS = ("delta", "theta", "alpha", "beta")


def connectivity_derivative_path(
    output_root: str,
    *,
    subject: str,
    run: str,
    condition: str,
    align_to: str,
    source_method: str,
    parc: str,
    method: str,
    suffix: str = "connectivity",
    extension: str = ".npy",
) -> Path:
    """Build the Stage 10 connectivity derivative path."""
    return derivative_path(
        output_root,
        subject=normalize_subject_id(subject),
        datatype="meg",
        task="tokens",
        run=parse_run_label(run)[0],
        description="-".join([condition.lower(), align_to, source_method, parc, method]),
        suffix=suffix,
        extension=extension,
    )


def find_roi_timeseries(data_dir: str, subject: str, condition: str) -> Path:
    """Find a real ROI time-course derivative for a subject and condition.

    This helper is kept for transitional loaders. New connectivity runs should
    pass Stage 5 ERP derivatives through ``--feature_dir``.
    """
    base = Path(data_dir)
    candidates = [
        base / subject / f"{subject}_{condition}_roi_timeseries.npy",
        base / f"{subject}_{condition}_roi_timeseries.npy",
    ]
    candidates.extend(base.glob(f"**/*{subject}*{condition}*roi*timeseries*.npy"))
    existing = sorted({path for path in candidates if path.is_file()})
    if not existing:
        raise FileNotFoundError(
            f"No ROI time-course derivative found for subject={subject}, condition={condition} under {data_dir}"
        )
    if len(existing) > 1:
        raise ValueError(f"Multiple ROI time-course derivatives matched subject={subject}, condition={condition}: {existing}")
    return existing[0]


def _stage_metadata(metadata: Mapping[str, object]) -> Mapping[str, object]:
    stage_meta = metadata.get("metadata", {})
    return stage_meta if isinstance(stage_meta, Mapping) else {}


def _load_timeseries(path: Path, labels: Optional[Sequence[str]]) -> tuple[np.ndarray, np.ndarray, list[str], Mapping[str, object]]:
    X, times, feature_names, metadata = _load_feature_array(path, labels=labels, lateralize=False)
    if X.ndim != 3:
        raise ValueError(f"Connectivity input must be trial x node x time, got {X.shape} in {path}")
    if X.shape[1] < 2:
        raise ValueError("Connectivity requires at least two nodes")
    return X, np.asarray(times, dtype=float), feature_names, metadata


def _metadata_value(metadata: Mapping[str, object], key: str, fallback):
    stage_meta = _stage_metadata(metadata)
    value = stage_meta.get(key)
    return fallback if value is None else value


def _parse_bands(bands: Optional[Sequence[str]]) -> tuple[list[str], list[float], list[float]]:
    if not bands:
        return list(DEFAULT_BANDS), list(DEFAULT_FMIN), list(DEFAULT_FMAX)

    names = []
    fmin = []
    fmax = []
    known = {name: (lo, hi) for name, lo, hi in zip(DEFAULT_BANDS, DEFAULT_FMIN, DEFAULT_FMAX)}
    for item in bands:
        if "=" not in item:
            if item not in known:
                raise ValueError(f"Unknown connectivity band '{item}'. Use name=fmin,fmax for custom bands.")
            lo, hi = known[item]
            names.append(item)
            fmin.append(lo)
            fmax.append(hi)
            continue
        name, bounds = item.split("=", 1)
        parts = [part.strip() for part in bounds.replace(":", ",").split(",") if part.strip()]
        if len(parts) != 2:
            raise ValueError(f"Band '{item}' must be formatted as name=fmin,fmax")
        lo, hi = float(parts[0]), float(parts[1])
        if lo <= 0 or hi <= lo:
            raise ValueError(f"Band '{item}' must satisfy 0 < fmin < fmax")
        names.append(name.strip())
        fmin.append(lo)
        fmax.append(hi)
    return names, fmin, fmax


def run_batch_connectivity(
    feature_dir: str,
    output_dir: str,
    subjects: Sequence[str],
    conditions: Sequence[str],
    *,
    align_to: str = "enter",
    source_method: str = "dSPM",
    parc: str = "HCPMMP1",
    labels: Optional[Sequence[str]] = None,
    runs_by_condition: Optional[Mapping[str, Sequence[str]]] = None,
    fmin: Sequence[float] = DEFAULT_FMIN,
    fmax: Sequence[float] = DEFAULT_FMAX,
    freq_names: Sequence[str] = DEFAULT_BANDS,
    method: str = "imcoh",
    mode: str = "fourier",
    sfreq: Optional[float] = None,
    before_window: tuple[float, float] = (0.7, 1.4),
    after_window: tuple[float, float] = (1.6, 2.3),
    n_jobs: int = 1,
) -> dict[str, list[Path]]:
    """Compute before/after spectral connectivity for staged ERP arrays."""
    if len(fmin) != len(fmax) or len(fmin) != len(freq_names):
        raise ValueError("fmin, fmax, and freq_names must have the same length")

    print(f"=== Starting Functional Connectivity Pipeline ({method}) ===")
    ensure_dir(output_dir)
    outputs: dict[str, list[Path]] = {}
    runs_by_condition = runs_by_condition or {}

    for subject in subjects:
        subject = normalize_subject_id(subject)
        outputs[subject] = []
        for condition in conditions:
            paths = find_feature_arrays(
                feature_dir,
                subject,
                condition,
                feature_source="erp",
                align_to=align_to,
                source_method=source_method,
                parc=parc,
                runs=runs_by_condition.get(condition),
            )
            for path in paths:
                print(f"  -> Loading ERP ROI time courses: {path}")
                data, times, node_names, metadata = _load_timeseries(path, labels)
                sfreq_hz = float(sfreq) if sfreq is not None else infer_sfreq_from_times(times)
                before_data = select_time_window(data, times, before_window)
                after_data = select_time_window(data, times, after_window)

                print(f"     Calculating {method} connectivity: before={before_data.shape}, after={after_data.shape}")
                before = compute_spectral_connectivity(
                    before_data,
                    method=method,
                    sfreq=sfreq_hz,
                    fmin=fmin,
                    fmax=fmax,
                    mode=mode,
                    n_jobs=n_jobs,
                )
                after = compute_spectral_connectivity(
                    after_data,
                    method=method,
                    sfreq=sfreq_hz,
                    fmin=fmin,
                    fmax=fmax,
                    mode=mode,
                    n_jobs=n_jobs,
                )

                run = str(_metadata_value(metadata, "run", "1"))
                condition_meta = str(_metadata_value(metadata, "condition", condition))
                out_path = connectivity_derivative_path(
                    output_dir,
                    subject=subject,
                    run=run,
                    condition=condition_meta,
                    align_to=align_to,
                    source_method=source_method,
                    parc=parc,
                    method=method,
                )
                saved = save_array(
                    out_path,
                    np.stack([before, after], axis=0),
                    dims=("window", "band", "node_from", "node_to"),
                    coords={
                        "window": ["before", "after"],
                        "band": list(freq_names),
                        "node_from": node_names,
                        "node_to": node_names,
                    },
                    metadata={
                        "stage": "connectivity",
                        "kind": "spectral_connectivity_before_after",
                        "subject": subject,
                        "run": parse_run_label(run)[0],
                        "condition": condition_meta,
                        "alignment": align_to,
                        "source_method": source_method,
                        "parcellation": parc,
                        "selected_labels": list(labels) if labels else None,
                        "method": method,
                        "mode": mode,
                        "sfreq_hz": sfreq_hz,
                        "fmin_hz": list(fmin),
                        "fmax_hz": list(fmax),
                        "before_window_sec": list(before_window),
                        "after_window_sec": list(after_window),
                        "n_trials": int(data.shape[0]),
                        "input_feature": str(path),
                    },
                )
                outputs[subject].append(saved)
                print(f"     Saved connectivity derivative: {saved}")

    print("=== Connectivity Pipeline Complete ===")
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute ROI functional connectivity from Stage 5 ERP derivatives.")
    parser.add_argument("--feature_dir", type=str, required=True,
                        help="BIDS derivatives root containing Stage 5 ERP arrays.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="BIDS derivatives root for connectivity outputs.")
    parser.add_argument("--subjects", type=str, nargs="+", required=True,
                        help="Subject IDs to process.")
    parser.add_argument("--conditions", type=str, nargs="+", required=True,
                        help="Condition labels to process.")
    parser.add_argument("--align_to", type=str, default="enter", choices=["go", "enter", "feedback"])
    parser.add_argument("--source_method", type=str, default="dSPM")
    parser.add_argument("--parc", type=str, default="HCPMMP1")
    parser.add_argument("--labels", type=str, nargs="+", default=None,
                        help="Optional labels/ROIs to select before connectivity.")
    parser.add_argument("--method", type=str, default="imcoh",
                        help="Connectivity metric, e.g. imcoh, wpli2_debiased, pli.")
    parser.add_argument("--mode", type=str, default="fourier")
    parser.add_argument("--bands", type=str, nargs="+", default=None,
                        help="Bands to compute, using known names or name=fmin,fmax.")
    parser.add_argument("--sfreq", type=float, default=None,
                        help="Sampling rate in Hz. If omitted, inferred from time coordinates.")
    parser.add_argument("--before_window", type=float, nargs=2, default=(0.7, 1.4),
                        help="Before/baseline window in seconds on the derivative time coordinate.")
    parser.add_argument("--after_window", type=float, nargs=2, default=(1.6, 2.3),
                        help="After/active window in seconds on the derivative time coordinate.")
    parser.add_argument("--n_jobs", type=int, default=1)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    freq_names, fmin, fmax = _parse_bands(args.bands)
    run_batch_connectivity(
        args.feature_dir,
        args.out_dir,
        args.subjects,
        args.conditions,
        align_to=args.align_to,
        source_method=args.source_method,
        parc=args.parc,
        labels=args.labels,
        fmin=fmin,
        fmax=fmax,
        freq_names=freq_names,
        method=args.method,
        mode=args.mode,
        sfreq=args.sfreq,
        before_window=tuple(args.before_window),
        after_window=tuple(args.after_window),
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
