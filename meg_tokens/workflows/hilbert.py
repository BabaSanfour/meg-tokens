"""Hilbert feature workflow for PAC/CFC-ready derivatives."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from meg_tokens.core import (
    HilbertConfig,
    ProjectConfig,
    WorkflowResult,
    normalize_subject_id,
    parse_run_label,
)
from meg_tokens.features.connectivity import infer_sfreq_from_times
from meg_tokens.features.dataset import find_feature_arrays, load_feature_array
from meg_tokens.features.time_frequency import (
    DEFAULT_BANDS,
    compute_hilbert_band_features,
)
from meg_tokens.io import DerivativeLayout, ensure_dir, save_array


DEFAULT_FEATURES = ("amplitude", "power", "phase", "sigfilt")


def hilbert_feature_derivative_path(
    output_root: str,
    *,
    subject: str,
    run: str,
    condition: str,
    align_to: str,
    source_method: str,
    parc: str,
    band: str,
    feature: str,
) -> Path:
    """Build a Stage 11 Hilbert-feature derivative path."""
    return DerivativeLayout(output_root).hilbert_feature(
        subject=subject,
        run=run,
        condition=condition,
        alignment=align_to,
        source_method=source_method,
        parc=parc,
        band=band,
        feature=feature,
    )


def _stage_metadata(metadata: Mapping[str, object]) -> Mapping[str, object]:
    stage_meta = metadata.get("metadata", {})
    return stage_meta if isinstance(stage_meta, Mapping) else {}


def _metadata_value(metadata: Mapping[str, object], key: str, fallback):
    value = _stage_metadata(metadata).get(key)
    return fallback if value is None else value


def _coord_values(metadata: Mapping[str, object], key: str, length: int) -> list[object]:
    coords = metadata.get("coords", {})
    if not isinstance(coords, Mapping):
        return list(range(length))
    values = coords.get(key)
    if values is None:
        if key == "trial":
            return list(range(1, length + 1))
        return list(range(length))
    return list(values)


def _canonical_features(features: Sequence[str]) -> tuple[str, ...]:
    canonical = tuple("sigfilt" if item == "filtered" else item for item in features)
    valid = set(DEFAULT_FEATURES)
    unknown = sorted(set(canonical) - valid)
    if unknown:
        raise ValueError(f"Unknown Hilbert feature(s): {unknown}. Valid features: {sorted(valid)}")
    return canonical


def run_batch_hilbert_features(
    feature_dir: str,
    output_dir: str,
    subjects: Sequence[str],
    conditions: Sequence[str],
    *,
    align_to: str = "go",
    source_method: str = "dSPM",
    parc: str = "HCPMMP1",
    labels: Optional[Sequence[str]] = None,
    runs_by_condition: Optional[Mapping[str, Sequence[str]]] = None,
    freq_bands: Optional[Mapping[str, tuple[float, float]]] = None,
    features: Sequence[str] = DEFAULT_FEATURES,
    sfreq: Optional[float] = None,
    n_jobs: int = 1,
) -> dict[str, list[Path]]:
    """Extract band-filtered signal, amplitude, power, and phase from ERP tensors."""
    ensure_dir(output_dir)
    requested = _canonical_features(features)
    bands = dict(freq_bands or DEFAULT_BANDS)
    runs_by_condition = runs_by_condition or {}
    outputs: dict[str, list[Path]] = {}

    for subject in subjects:
        subject = normalize_subject_id(subject)
        outputs[subject] = []
        for condition in conditions:
            paths = find_feature_arrays(
                feature_dir,
                subject,
                condition,
                feature_source="erp",
                alignment=align_to,
                source_method=source_method,
                parc=parc,
                runs=runs_by_condition.get(condition),
            )
            for path in paths:
                data, times, feature_names, metadata = load_feature_array(path, labels=labels, lateralize=False)
                sfreq_hz = float(sfreq) if sfreq is not None else infer_sfreq_from_times(times)
                features_by_band = compute_hilbert_band_features(
                    data,
                    sfreq=sfreq_hz,
                    freq_bands=bands,
                    features=requested,
                    n_jobs=n_jobs,
                )

                run = str(_metadata_value(metadata, "run", "1"))
                condition_meta = str(_metadata_value(metadata, "condition", condition))
                trial_coords = _coord_values(metadata, "trial", data.shape[0])

                for band, band_features in features_by_band.items():
                    bounds = bands[band]
                    for feature, values in band_features.items():
                        out_path = hilbert_feature_derivative_path(
                            output_dir,
                            subject=subject,
                            run=run,
                            condition=condition_meta,
                            align_to=align_to,
                            source_method=source_method,
                            parc=parc,
                            band=band,
                            feature=feature,
                        )
                        saved = save_array(
                            out_path,
                            values,
                            dims=("trial", "feature", "time"),
                            coords={
                                "trial": trial_coords,
                                "feature": feature_names,
                                "time_sec": times,
                            },
                            metadata={
                                "stage": "hilbert_features",
                                "kind": "erp_hilbert_feature",
                                "subject": subject,
                                "run": parse_run_label(run)[0],
                                "condition": condition_meta,
                                "alignment": align_to,
                                "source_method": source_method,
                                "parcellation": parc,
                                "selected_labels": list(labels) if labels else None,
                                "band": band,
                                "fmin_hz": float(bounds[0]),
                                "fmax_hz": float(bounds[1]),
                                "feature": feature,
                                "sfreq_hz": sfreq_hz,
                                "input_feature": str(path),
                            },
                        )
                        outputs[subject].append(saved)
                        print(f"Saved {feature} {band} Hilbert feature: {saved}")

    return outputs


def extract_hilbert_features(
    project: ProjectConfig,
    *,
    subjects: Sequence[str],
    settings: HilbertConfig,
    feature_root: Optional[str | Path] = None,
    output_root: Optional[str | Path] = None,
) -> WorkflowResult:
    """Extract configured analytic-signal features from ERP arrays."""
    bands = (
        {name: (fmin, fmax) for name, fmin, fmax in settings.bands}
        if settings.bands
        else None
    )
    feature_root = Path(feature_root or project.bids_root)
    outputs = run_batch_hilbert_features(
        str(feature_root),
        str(output_root or project.bids_root),
        subjects,
        settings.conditions,
        align_to=settings.alignment,
        source_method=settings.source_method,
        parc=settings.parc,
        labels=settings.labels,
        freq_bands=bands,
        features=settings.features,
        sfreq=settings.sfreq,
        n_jobs=settings.n_jobs,
    )
    inputs = tuple(
        path
        for subject in subjects
        for condition in settings.conditions
        for path in find_feature_arrays(
            feature_root,
            subject,
            condition,
            feature_source="erp",
            alignment=settings.alignment,
            source_method=settings.source_method,
            parc=settings.parc,
        )
    )
    output_paths = tuple(path for paths in outputs.values() for path in paths)
    return WorkflowResult(
        stage="hilbert_features",
        inputs=inputs,
        outputs=output_paths,
        settings={
            "subjects": [normalize_subject_id(subject) for subject in subjects],
            **settings.__dict__,
        },
    )
