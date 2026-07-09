"""
Stage 4 source-space frequency-band power extraction.

This utility consumes the Stage 3 source-estimate manifest and writes
BIDS-derivatives-style array tensors for downstream decoding, PCA, and
trajectory analyses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import mne
import numpy as np
import pandas as pd

from meg_tokens.core import (
    PowerConfig,
    ProjectConfig,
    WorkflowResult,
    normalize_subject_id,
    parse_run_label,
)
from meg_tokens.features.time_frequency import (
    DEFAULT_BANDS,
    compute_band_power,
    compute_window_times,
    rescale_baseline,
)
from meg_tokens.io import DerivativeLayout, require_file, save_array


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
    return DerivativeLayout(source_dir).find_source_manifest(
        subject=subject,
        run=run,
        condition=condition,
        alignment=align_to,
        source_method=source_method,
    )


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
    return DerivativeLayout(output_root).power(
        subject=subject,
        run=run,
        condition=condition,
        alignment=align_to,
        source_method=source_method,
        power_method=power_method,
        band=band,
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


def extract_power_features(
    project: ProjectConfig,
    *,
    subjects: Sequence[str],
    settings: PowerConfig,
    source_root: Optional[str | Path] = None,
    output_root: Optional[str | Path] = None,
) -> WorkflowResult:
    """Extract configured source-power bands for selected subjects."""
    bands = (
        {name: (fmin, fmax) for name, fmin, fmax in settings.bands}
        if settings.bands
        else None
    )
    source_root = Path(source_root or project.bids_root)
    outputs = run_time_frequency_pipeline(
        subjects,
        source_dir=str(source_root),
        out_dir=str(output_root or project.bids_root),
        run=settings.run,
        condition=settings.condition,
        align_to=settings.alignment,
        source_method=settings.source_method,
        freq_bands=bands,
        method=settings.method,
        width=settings.width,
        step=settings.step,
        n_jobs=settings.n_jobs,
        baseline=settings.baseline,
        baseline_method=settings.baseline_method,
    )
    inputs = tuple(
        DerivativeLayout(source_root).find_source_manifest(
            subject=subject,
            run=settings.run,
            condition=settings.condition,
            alignment=settings.alignment,
            source_method=settings.source_method,
        )
        for subject in subjects
    )
    output_paths = tuple(
        path
        for subject_outputs in outputs.values()
        for path in subject_outputs.values()
    )
    return WorkflowResult(
        stage="source_power",
        inputs=inputs,
        outputs=output_paths,
        settings={
            "subjects": [normalize_subject_id(subject) for subject in subjects],
            **settings.__dict__,
        },
    )


def _parse_baseline(values: Optional[Sequence[float]]) -> Optional[tuple[Optional[float], Optional[float]]]:
    if values is None:
        return None
    if len(values) != 2:
        raise ValueError("--baseline requires two values: start end")
    return (values[0], values[1])

