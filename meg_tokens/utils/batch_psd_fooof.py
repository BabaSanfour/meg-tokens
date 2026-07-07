"""Stage 5b PSD and specparam modeling over staged Epochs derivatives."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Mapping, Optional, Sequence

import mne
import numpy as np
import pandas as pd

from meg_tokens.io import derivative_path, require_file, save_array, save_table, sidecar_path
from meg_tokens.meg.time_frequency import compute_psd, fit_specparam
from meg_tokens.utils.batch_processor import normalize_subject_id
from meg_tokens.utils.epochs_builder import parse_run_label


def _format_freq(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def psd_description(
    *,
    condition: Optional[str],
    align_to: str,
    method: str,
    fmin: float,
    fmax: float,
) -> str:
    parts = []
    if condition:
        parts.append(condition.lower())
    parts.extend([align_to, method, f"{_format_freq(fmin)}to{_format_freq(fmax)}hz"])
    return "-".join(parts)


def psd_derivative_path(
    output_root: str | Path,
    *,
    subject: str,
    run: str,
    condition: Optional[str],
    align_to: str,
    method: str,
    fmin: float,
    fmax: float,
    suffix: str,
    extension: str,
) -> Path:
    return derivative_path(
        output_root,
        subject=normalize_subject_id(subject),
        datatype="meg",
        task="tokens",
        run=parse_run_label(run)[0],
        description=psd_description(condition=condition, align_to=align_to, method=method, fmin=fmin, fmax=fmax),
        suffix=suffix,
        extension=extension,
    )


def _metadata_from_epochs_path(path: Path) -> dict[str, object]:
    meta_path = sidecar_path(path)
    if meta_path.is_file():
        with meta_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        return {
            "subject": metadata.get("subject"),
            "run": metadata.get("run"),
            "condition": metadata.get("condition"),
            "alignment": metadata.get("alignment"),
        }

    match = re.match(
        r"sub-(?P<subject>[^_]+)_task-[^_]+_run-(?P<run>[^_]+)_desc-(?P<desc>.+)_epo\.fif$",
        path.name,
    )
    if not match:
        raise ValueError(f"Epochs file is missing a sidecar and does not match the derivative pattern: {path}")
    desc = match.group("desc")
    parts = desc.split("-")
    if len(parts) == 1:
        condition = None
        alignment = parts[0]
    else:
        condition = parts[0].capitalize()
        alignment = parts[-1]
    return {
        "subject": match.group("subject"),
        "run": match.group("run"),
        "condition": condition,
        "alignment": alignment,
    }


def find_epoch_derivatives(
    epochs_dir: str | Path,
    subject: str,
    *,
    run: Optional[str] = None,
    condition: Optional[str] = None,
    align_to: Optional[str] = None,
) -> list[Path]:
    """Find staged Epochs FIF derivatives for one subject."""
    subject = normalize_subject_id(subject)
    root = Path(epochs_dir)
    pattern = f"**/sub-{subject}_task-tokens_run-*_desc-*_epo.fif"
    candidates = sorted(path for path in root.glob(pattern) if path.is_file())
    if not candidates:
        raise FileNotFoundError(f"No staged Epochs FIF derivatives found for {subject} under {epochs_dir}")

    run_filter = parse_run_label(run)[0] if run else None
    condition_filter = condition.lower() if condition else None
    align_filter = align_to.lower() if align_to else None
    selected: list[Path] = []
    for path in candidates:
        metadata = _metadata_from_epochs_path(path)
        if run_filter and str(metadata.get("run")) != run_filter:
            continue
        if condition_filter and str(metadata.get("condition", "")).lower() != condition_filter:
            continue
        if align_filter and str(metadata.get("alignment", "")).lower() != align_filter:
            continue
        selected.append(path)

    if not selected:
        raise FileNotFoundError(
            "No staged Epochs FIF derivatives matched "
            f"subject={subject}, run={run}, condition={condition}, alignment={align_to}"
        )
    return selected


def _channel_names(epochs: mne.BaseEpochs) -> list[str]:
    return [str(name) for name in epochs.ch_names]


def _safe_positive_spectra(spectra: np.ndarray) -> np.ndarray:
    values = np.asarray(spectra, dtype=float)
    finite_positive = values[np.isfinite(values) & (values > 0)]
    if finite_positive.size == 0:
        raise ValueError("PSD contains no positive finite values for specparam fitting")
    floor = float(np.min(finite_positive)) * 1e-6
    return np.where(np.isfinite(values) & (values > 0), values, floor)


def _specparam_tables(model, channels: Sequence[str], base_metadata: Mapping[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = model.to_df().copy()
    if len(params) != len(channels):
        raise ValueError(f"specparam returned {len(params)} rows for {len(channels)} channels")

    params.insert(0, "channel", list(channels))
    for key in ("subject", "run", "condition", "alignment", "psd_method", "fmin_hz", "fmax_hz"):
        params.insert(0, key, base_metadata.get(key))
    params["n_peaks"] = 0

    peak_rows = []
    group_results = getattr(getattr(model, "results", None), "group_results", [])
    for channel, result in zip(channels, group_results):
        peaks = np.asarray(getattr(result, "peak_converted", np.empty((0, 3))), dtype=float)
        if peaks.size == 0:
            continue
        if peaks.ndim == 1:
            peaks = peaks.reshape(1, -1)
        params.loc[params["channel"] == channel, "n_peaks"] = int(peaks.shape[0])
        for peak_index, row in enumerate(peaks, start=1):
            peak_rows.append({
                **base_metadata,
                "channel": channel,
                "peak_index": peak_index,
                "center_frequency_hz": float(row[0]),
                "power": float(row[1]),
                "bandwidth_hz": float(row[2]),
            })

    peaks = pd.DataFrame(
        peak_rows,
        columns=[
            "subject",
            "run",
            "condition",
            "alignment",
            "psd_method",
            "fmin_hz",
            "fmax_hz",
            "channel",
            "peak_index",
            "center_frequency_hz",
            "power",
            "bandwidth_hz",
        ],
    )
    return params, peaks


def process_epochs_psd(
    epochs_path: str | Path,
    output_dir: str | Path,
    *,
    fmin: float = 1.0,
    fmax: float = 100.0,
    method: str = "welch",
    n_fft: int = 2048,
    n_overlap: int = 150,
    n_jobs: int = 1,
    fit_model: bool = True,
    peak_width_limits: tuple[float, float] = (1.0, 10.0),
    min_peak_height: float = 0.1,
    max_n_peaks: int = 6,
    peak_threshold: float = 2.0,
    aperiodic_mode: str = "fixed",
) -> dict[str, Path]:
    """Compute PSD and optional specparam tables for one Epochs derivative."""
    epochs_file = require_file(epochs_path, purpose="staged Epochs FIF derivative")
    metadata = _metadata_from_epochs_path(epochs_file)
    subject = normalize_subject_id(str(metadata["subject"]))
    run = str(metadata["run"])
    condition = None if metadata.get("condition") in (None, "None") else str(metadata["condition"])
    align_to = str(metadata["alignment"])

    epochs = mne.read_epochs(str(epochs_file), preload=True, verbose=False)
    psds, freqs = compute_psd(
        epochs,
        fmin=fmin,
        fmax=fmax,
        method=method,
        n_fft=n_fft,
        n_overlap=n_overlap,
        n_jobs=n_jobs,
    )
    mean_psd = np.nanmean(psds, axis=0)
    channels = _channel_names(epochs)
    if mean_psd.shape[0] != len(channels):
        raise ValueError(f"PSD channel axis {mean_psd.shape[0]} does not match Epochs channels {len(channels)}")

    psd_path = psd_derivative_path(
        output_dir,
        subject=subject,
        run=run,
        condition=condition,
        align_to=align_to,
        method=method,
        fmin=fmin,
        fmax=fmax,
        suffix="psd",
        extension=".npy",
    )
    saved = {
        "psd": save_array(
            psd_path,
            mean_psd,
            dims=("channel", "frequency"),
            coords={"channel": channels, "frequency_hz": freqs},
            metadata={
                "stage": "psd_specparam",
                "kind": "epoch_mean_psd",
                "subject": subject,
                "run": parse_run_label(run)[0],
                "condition": condition,
                "alignment": align_to,
                "psd_method": method,
                "fmin_hz": float(fmin),
                "fmax_hz": float(fmax),
                "n_fft": int(min(n_fft, len(epochs.times))) if method == "welch" else None,
                "n_overlap": int(min(n_overlap, max(0, min(n_fft, len(epochs.times)) - 1))) if method == "welch" else None,
                "n_epochs": int(len(epochs)),
                "input_epochs": str(epochs_file),
            },
        )
    }

    if fit_model:
        positive_psd = _safe_positive_spectra(mean_psd)
        model = fit_specparam(
            freqs,
            positive_psd,
            peak_width_limits=peak_width_limits,
            min_peak_height=min_peak_height,
            max_n_peaks=max_n_peaks,
            peak_threshold=peak_threshold,
            freq_range=(fmin, fmax),
            aperiodic_mode=aperiodic_mode,
            n_jobs=n_jobs,
            verbose=False,
        )
        base_metadata = {
            "subject": subject,
            "run": parse_run_label(run)[0],
            "condition": condition,
            "alignment": align_to,
            "psd_method": method,
            "fmin_hz": float(fmin),
            "fmax_hz": float(fmax),
        }
        params, peaks = _specparam_tables(model, channels, base_metadata)
        params_path = psd_derivative_path(
            output_dir,
            subject=subject,
            run=run,
            condition=condition,
            align_to=align_to,
            method=method,
            fmin=fmin,
            fmax=fmax,
            suffix="specparam",
            extension=".tsv",
        )
        peaks_path = psd_derivative_path(
            output_dir,
            subject=subject,
            run=run,
            condition=condition,
            align_to=align_to,
            method=method,
            fmin=fmin,
            fmax=fmax,
            suffix="specparampeaks",
            extension=".tsv",
        )
        saved["specparam"] = save_table(
            params_path,
            params,
            metadata={
                "stage": "psd_specparam",
                "kind": "specparam_channel_parameters",
                "input_psd": str(saved["psd"]),
                "input_epochs": str(epochs_file),
                "aperiodic_mode": aperiodic_mode,
                "peak_width_limits_hz": list(peak_width_limits),
                "min_peak_height": float(min_peak_height),
                "max_n_peaks": int(max_n_peaks),
                "peak_threshold": float(peak_threshold),
            },
        )
        saved["specparam_peaks"] = save_table(
            peaks_path,
            peaks,
            metadata={
                "stage": "psd_specparam",
                "kind": "specparam_periodic_peaks",
                "input_psd": str(saved["psd"]),
                "input_epochs": str(epochs_file),
            },
        )

    return saved


def run_psd_specparam(
    epochs_dir: str | Path,
    output_dir: str | Path,
    subjects: Sequence[str],
    *,
    run: Optional[str] = None,
    condition: Optional[str] = None,
    align_to: Optional[str] = None,
    fmin: float = 1.0,
    fmax: float = 100.0,
    method: str = "welch",
    n_fft: int = 2048,
    n_overlap: int = 150,
    n_jobs: int = 1,
    fit_model: bool = True,
) -> dict[str, list[dict[str, Path]]]:
    """Run PSD/specparam extraction for staged Epochs derivatives."""
    outputs: dict[str, list[dict[str, Path]]] = {}
    for subject in subjects:
        subject = normalize_subject_id(subject)
        outputs[subject] = []
        epoch_files = find_epoch_derivatives(
            epochs_dir,
            subject,
            run=run,
            condition=condition,
            align_to=align_to,
        )
        for epochs_path in epoch_files:
            print(f"Computing PSD/specparam for {epochs_path}")
            outputs[subject].append(
                process_epochs_psd(
                    epochs_path,
                    output_dir,
                    fmin=fmin,
                    fmax=fmax,
                    method=method,
                    n_fft=n_fft,
                    n_overlap=n_overlap,
                    n_jobs=n_jobs,
                    fit_model=fit_model,
                )
            )
    return outputs


def run_psd_fooof(*args, **kwargs):
    """Compatibility alias for the modern PSD/specparam runner."""
    return run_psd_specparam(*args, **kwargs)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute PSD and specparam models from Stage 2 Epochs derivatives.")
    parser.add_argument("--epochs_dir", type=str, required=True,
                        help="BIDS derivatives root containing Stage 2 Epochs FIF files.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="BIDS derivatives root for PSD/specparam outputs.")
    parser.add_argument("--subjects", type=str, nargs="+", required=True,
                        help="Subject IDs to process.")
    parser.add_argument("--run", type=str, default=None,
                        help="Optional run label filter, for example Slow1 or 1.")
    parser.add_argument("--condition", type=str, default=None,
                        help="Optional condition filter.")
    parser.add_argument("--align_to", type=str, default=None, choices=["go", "enter", "feedback"],
                        help="Optional alignment filter.")
    parser.add_argument("--method", type=str, default="welch", choices=["welch", "multitaper"],
                        help="PSD method.")
    parser.add_argument("--fmin", type=float, default=1.0)
    parser.add_argument("--fmax", type=float, default=100.0)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--n_overlap", type=int, default=150)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--no_specparam", action="store_true",
                        help="Only save PSD arrays; skip specparam fitting.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_psd_specparam(
        epochs_dir=args.epochs_dir,
        output_dir=args.out_dir,
        subjects=args.subjects,
        run=args.run,
        condition=args.condition,
        align_to=args.align_to,
        fmin=args.fmin,
        fmax=args.fmax,
        method=args.method,
        n_fft=args.n_fft,
        n_overlap=args.n_overlap,
        n_jobs=args.n_jobs,
        fit_model=not args.no_specparam,
    )


if __name__ == "__main__":
    main()
