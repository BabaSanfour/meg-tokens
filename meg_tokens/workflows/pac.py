"""Phase-amplitude coupling workflow over Hilbert feature derivatives."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from meg_tokens.core import (
    PACConfig,
    ProjectConfig,
    WorkflowResult,
    normalize_subject_id,
    parse_run_label,
)
from meg_tokens.features.pac import modulation_index, select_time_window
from meg_tokens.io import DerivativeLayout, ensure_dir, load_array, save_array
from meg_tokens.workflows.hilbert import hilbert_feature_derivative_path


def pac_derivative_path(
    output_root: str | Path,
    *,
    subject: str,
    run: str,
    condition: str,
    align_to: str,
    source_method: str,
    parc: str,
    phase_bands: Sequence[str],
    amplitude_bands: Sequence[str],
    method: str,
) -> Path:
    """Build a Stage 12 PAC/CFC derivative path."""
    return DerivativeLayout(output_root).pac(
        subject=subject,
        run=run,
        condition=condition,
        alignment=align_to,
        source_method=source_method,
        parc=parc,
        phase_bands=phase_bands,
        amplitude_bands=amplitude_bands,
        method=method,
    )


def find_hilbert_feature_arrays(
    data_dir: str | Path,
    subject: str,
    condition: str,
    *,
    align_to: str,
    source_method: str,
    parc: str,
    band: str,
    feature: str,
    runs: Optional[Sequence[str]] = None,
) -> list[Path]:
    """Find Stage 11 Hilbert feature derivatives for one band/feature."""
    subject = normalize_subject_id(subject)
    if runs:
        candidates = [
            hilbert_feature_derivative_path(
                str(data_dir),
                subject=subject,
                run=parse_run_label(run)[0],
                condition=condition,
                align_to=align_to,
                source_method=source_method,
                parc=parc,
                band=band,
                feature=feature,
            )
            for run in runs
        ]
        existing = [path for path in candidates if path.is_file()]
    else:
        pattern = (
            f"**/sub-{subject}_task-tokens_run-*_desc-{condition.lower()}-"
            f"{align_to}-{source_method}-{parc}-{band.replace('_', '-')}-{feature}_hilbertfeature.npy"
        )
        existing = sorted(path for path in Path(data_dir).glob(pattern) if path.is_file())

    if not existing:
        raise FileNotFoundError(
            f"No Hilbert {feature} derivatives found for subject={subject}, condition={condition}, "
            f"alignment={align_to}, band={band}"
        )
    return sorted(existing)


def _stage_metadata(metadata: Mapping[str, object]) -> Mapping[str, object]:
    stage_meta = metadata.get("metadata", {})
    return stage_meta if isinstance(stage_meta, Mapping) else {}


def _metadata_value(metadata: Mapping[str, object], key: str, fallback):
    value = _stage_metadata(metadata).get(key)
    return fallback if value is None else value


def _load_hilbert_feature(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str], Mapping[str, object]]:
    loaded = load_array(path, expected_ndim=3, require_sidecar=True)
    data = np.asarray(loaded.data, dtype=float)
    dims = tuple(loaded.metadata.get("dims", []))
    if dims != ("trial", "feature", "time"):
        raise ValueError(f"Expected trial x feature x time in {path}, got {dims}")
    coords = loaded.metadata.get("coords", {})
    if not isinstance(coords, Mapping):
        coords = {}
    times = np.asarray(coords.get("time_sec", np.arange(data.shape[-1])), dtype=float)
    if times.size != data.shape[-1]:
        raise ValueError(f"time coordinate length does not match data in {path}")
    features = [str(value) for value in coords.get("feature", list(range(data.shape[1])))]
    if len(features) != data.shape[1]:
        raise ValueError(f"feature coordinate length does not match data in {path}")
    return data, times, features, loaded.metadata


def _validate_pair(
    phase_path: Path,
    amplitude_path: Path,
    phase_data: np.ndarray,
    amplitude_data: np.ndarray,
    phase_times: np.ndarray,
    amplitude_times: np.ndarray,
    phase_features: Sequence[str],
    amplitude_features: Sequence[str],
) -> None:
    if phase_data.shape != amplitude_data.shape:
        raise ValueError(f"Phase/amplitude shapes differ for {phase_path} and {amplitude_path}")
    if not np.allclose(phase_times, amplitude_times):
        raise ValueError(f"Phase/amplitude time coordinates differ for {phase_path} and {amplitude_path}")
    if list(phase_features) != list(amplitude_features):
        raise ValueError(f"Phase/amplitude feature coordinates differ for {phase_path} and {amplitude_path}")


def run_batch_pac_cfc(
    feature_dir: str | Path,
    output_dir: str | Path,
    subjects: Sequence[str],
    conditions: Sequence[str],
    *,
    phase_bands: Sequence[str],
    amplitude_bands: Sequence[str],
    align_to: str = "go",
    source_method: str = "dSPM",
    parc: str = "HCPMMP1",
    runs_by_condition: Optional[Mapping[str, Sequence[str]]] = None,
    method: str = "modulation_index",
    n_bins: int = 18,
    time_window: Optional[tuple[float, float]] = None,
) -> dict[str, list[Path]]:
    """Compute PAC/CFC arrays from Stage 11 phase and amplitude derivatives."""
    if method != "modulation_index":
        raise ValueError("Only method='modulation_index' is currently supported")
    if not phase_bands:
        raise ValueError("At least one phase band is required")
    if not amplitude_bands:
        raise ValueError("At least one amplitude band is required")

    ensure_dir(output_dir)
    runs_by_condition = runs_by_condition or {}
    outputs: dict[str, list[Path]] = {}

    for subject in subjects:
        subject = normalize_subject_id(subject)
        outputs[subject] = []
        for condition in conditions:
            reference_paths = find_hilbert_feature_arrays(
                feature_dir,
                subject,
                condition,
                align_to=align_to,
                source_method=source_method,
                parc=parc,
                band=phase_bands[0],
                feature="phase",
                runs=runs_by_condition.get(condition),
            )
            for reference_path in reference_paths:
                _, _, _, reference_meta = _load_hilbert_feature(reference_path)
                run = str(_metadata_value(reference_meta, "run", "1"))
                condition_meta = str(_metadata_value(reference_meta, "condition", condition))

                pair_values = []
                input_phase_paths = []
                input_amplitude_paths = []
                feature_names = None
                time_values = None

                for phase_band in phase_bands:
                    phase_path = hilbert_feature_derivative_path(
                        str(feature_dir),
                        subject=subject,
                        run=run,
                        condition=condition_meta,
                        align_to=align_to,
                        source_method=source_method,
                        parc=parc,
                        band=phase_band,
                        feature="phase",
                    )
                    phase_data, phase_times, phase_features, _ = _load_hilbert_feature(phase_path)
                    amp_values = []
                    for amplitude_band in amplitude_bands:
                        amplitude_path = hilbert_feature_derivative_path(
                            str(feature_dir),
                            subject=subject,
                            run=run,
                            condition=condition_meta,
                            align_to=align_to,
                            source_method=source_method,
                            parc=parc,
                            band=amplitude_band,
                            feature="amplitude",
                        )
                        amplitude_data, amplitude_times, amplitude_features, _ = _load_hilbert_feature(amplitude_path)
                        _validate_pair(
                            phase_path,
                            amplitude_path,
                            phase_data,
                            amplitude_data,
                            phase_times,
                            amplitude_times,
                            phase_features,
                            amplitude_features,
                        )
                        phase_selected = select_time_window(phase_data, phase_times, time_window)
                        amplitude_selected = select_time_window(amplitude_data, amplitude_times, time_window)
                        amp_values.append(modulation_index(phase_selected, amplitude_selected, n_bins=n_bins))
                        input_amplitude_paths.append(str(amplitude_path))

                    pair_values.append(np.stack(amp_values, axis=0))
                    input_phase_paths.append(str(phase_path))
                    feature_names = phase_features
                    time_values = phase_times

                pac = np.stack(pair_values, axis=0)
                out_path = pac_derivative_path(
                    output_dir,
                    subject=subject,
                    run=run,
                    condition=condition_meta,
                    align_to=align_to,
                    source_method=source_method,
                    parc=parc,
                    phase_bands=phase_bands,
                    amplitude_bands=amplitude_bands,
                    method=method,
                )
                saved = save_array(
                    out_path,
                    pac,
                    dims=("phase_band", "amplitude_band", "feature"),
                    coords={
                        "phase_band": list(phase_bands),
                        "amplitude_band": list(amplitude_bands),
                        "feature": feature_names or [],
                    },
                    metadata={
                        "stage": "pac_cfc",
                        "kind": "phase_amplitude_modulation_index",
                        "subject": subject,
                        "run": parse_run_label(run)[0],
                        "condition": condition_meta,
                        "alignment": align_to,
                        "source_method": source_method,
                        "parcellation": parc,
                        "method": method,
                        "n_bins": int(n_bins),
                        "time_window_sec": list(time_window) if time_window else None,
                        "time_start_sec": float(np.asarray(time_values)[0]) if time_values is not None else None,
                        "time_stop_sec": float(np.asarray(time_values)[-1]) if time_values is not None else None,
                        "input_phase_features": input_phase_paths,
                        "input_amplitude_features": sorted(set(input_amplitude_paths)),
                    },
                )
                outputs[subject].append(saved)
                print(f"Saved PAC/CFC derivative: {saved}")

    return outputs


def extract_pac_features(
    project: ProjectConfig,
    *,
    subjects: Sequence[str],
    settings: PACConfig,
    feature_root: Optional[str | Path] = None,
    output_root: Optional[str | Path] = None,
) -> WorkflowResult:
    """Compute PAC modulation indices from staged Hilbert derivatives."""
    feature_root = Path(feature_root or project.bids_root)
    outputs = run_batch_pac_cfc(
        feature_root,
        output_root or project.bids_root,
        subjects,
        settings.conditions,
        phase_bands=settings.phase_bands,
        amplitude_bands=settings.amplitude_bands,
        align_to=settings.alignment,
        source_method=settings.source_method,
        parc=settings.parc,
        method=settings.method,
        n_bins=settings.n_bins,
        time_window=settings.time_window,
    )
    inputs = set()
    for subject in subjects:
        for condition in settings.conditions:
            for band, feature in (
                *((band, "phase") for band in settings.phase_bands),
                *((band, "amplitude") for band in settings.amplitude_bands),
            ):
                inputs.update(
                    find_hilbert_feature_arrays(
                        feature_root,
                        subject,
                        condition,
                        align_to=settings.alignment,
                        source_method=settings.source_method,
                        parc=settings.parc,
                        band=band,
                        feature=feature,
                    )
                )
    return WorkflowResult(
        stage="pac_features",
        inputs=tuple(sorted(inputs)),
        outputs=tuple(path for paths in outputs.values() for path in paths),
        settings={
            "subjects": [normalize_subject_id(subject) for subject in subjects],
            **settings.__dict__,
        },
    )

