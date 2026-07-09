"""Discovery and loading of staged trial-by-feature arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from meg_tokens.core import normalize_subject_id
from meg_tokens.io import DerivativeLayout, load_array, require_file


def find_feature_arrays(
    feature_root: str | Path,
    subject: str,
    condition: str,
    *,
    feature_source: str,
    alignment: str,
    source_method: str,
    parc: Optional[str] = "HCPMMP1",
    band: Optional[str] = None,
    power_method: str = "hilbert",
    runs: Optional[Sequence[str]] = None,
) -> list[Path]:
    """Find ERP or source-power arrays for one subject-condition cell."""
    subject = normalize_subject_id(subject)
    layout = DerivativeLayout(feature_root)
    if feature_source == "erp":
        if parc is None:
            raise ValueError("parc is required for ERP feature discovery")
        if runs:
            candidates = [
                layout.erp(
                    subject=subject,
                    run=run,
                    condition=condition,
                    alignment=alignment,
                    source_method=source_method,
                    parc=parc,
                    suffix="erp",
                    extension=".npy",
                )
                for run in runs
            ]
            existing = [path for path in candidates if path.is_file()]
        else:
            pattern = (
                f"**/sub-{subject}_task-tokens_run-*_desc-{condition.lower()}-"
                f"{alignment}-{source_method}-{parc}_erp.npy"
            )
            existing = sorted(path for path in Path(feature_root).glob(pattern) if path.is_file())
    elif feature_source == "power":
        if band is None:
            raise ValueError("band is required for power feature discovery")
        if runs:
            candidates = [
                layout.power(
                    subject=subject,
                    run=run,
                    condition=condition,
                    alignment=alignment,
                    source_method=source_method,
                    power_method=power_method,
                    band=band,
                )
                for run in runs
            ]
            existing = [path for path in candidates if path.is_file()]
        else:
            pattern = (
                f"**/sub-{subject}_task-tokens_run-*_desc-{condition.lower()}-"
                f"{alignment}-{source_method}-{power_method}-{band.replace('_', '-')}_power.npy"
            )
            existing = sorted(path for path in Path(feature_root).glob(pattern) if path.is_file())
    else:
        raise ValueError("feature_source must be 'erp' or 'power'")

    if not existing:
        raise FileNotFoundError(
            f"No {feature_source} derivatives found for subject={subject}, "
            f"condition={condition}, alignment={alignment}"
        )
    return sorted(existing)


def _coord_values(coords: Mapping[str, object], key: str, length: int) -> list:
    values = coords.get(key)
    return list(range(length)) if values is None else list(values)


def _select_label_indices(
    label_names: Sequence[object],
    labels: Optional[Sequence[str]],
) -> list[int]:
    if not labels:
        return list(range(len(label_names)))
    lookup = {str(label): index for index, label in enumerate(label_names)}
    selected = []
    for label in labels:
        if str(label).isdigit():
            index = int(label)
            if index < 0 or index >= len(label_names):
                raise ValueError(f"Label index {index} is out of range")
            selected.append(index)
        else:
            if label not in lookup:
                raise ValueError(
                    f"Requested label is absent from feature coordinates: {label}"
                )
            selected.append(lookup[label])
    return selected


def _label_pair_key(label: object) -> tuple[str, str]:
    text = str(label)
    lower = text.lower()
    if lower.endswith("-lh"):
        hemisphere = "lh"
        base = text[:-3]
    elif lower.endswith("-rh"):
        hemisphere = "rh"
        base = text[:-3]
    else:
        raise ValueError(f"Cannot infer hemisphere from label name: {text}")
    if base.startswith(("L_", "R_", "L-", "R-")):
        base = base[2:]
    return base, hemisphere


def lateralize_labels(
    data: np.ndarray,
    label_names: Sequence[object],
    *,
    label_axis: int,
) -> tuple[np.ndarray, list[str]]:
    """Subtract right homologues from left labels along one named axis."""
    if data.shape[label_axis] != len(label_names):
        raise ValueError(
            f"Label coordinate has {len(label_names)} values for axis length "
            f"{data.shape[label_axis]}"
        )
    pairs: dict[str, dict[str, int]] = {}
    for index, label in enumerate(label_names):
        base, hemisphere = _label_pair_key(label)
        pairs.setdefault(base, {})[hemisphere] = index
    complete = [
        (base, sides["lh"], sides["rh"])
        for base, sides in sorted(pairs.items())
        if {"lh", "rh"} <= set(sides)
    ]
    if not complete:
        raise ValueError("No left/right label pairs were found")

    names = []
    differences = []
    for base, left_index, right_index in complete:
        differences.append(
            np.take(data, left_index, axis=label_axis)
            - np.take(data, right_index, axis=label_axis)
        )
        names.append(base)
    return np.stack(differences, axis=label_axis), names


def load_feature_array(
    path: str | Path,
    *,
    labels: Optional[Sequence[str]] = None,
    lateralize: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, object]]:
    """Load and flatten one trial-by-feature derivative using named dimensions."""
    path = Path(path)
    loaded = load_array(path, require_sidecar=True)
    data = np.asarray(loaded.data, dtype=float)
    dims = tuple(loaded.metadata.get("dims", []))
    coords = loaded.metadata.get("coords", {})
    if not isinstance(coords, Mapping):
        raise ValueError(f"Feature coordinates must be a mapping in {path}")
    if not dims or dims[0] != "trial":
        raise ValueError(f"Expected first dimension to be trial in {path}, got {dims}")
    if data.ndim not in (3, 4):
        raise ValueError(
            f"Expected feature array to be 3D or 4D, got shape {data.shape} in {path}"
        )

    time = coords.get("time_sec", coords.get("time_ms"))
    time_values = (
        np.arange(data.shape[-1], dtype=float)
        if time is None
        else np.asarray(time, dtype=float)
    )
    if len(time_values) != data.shape[-1]:
        raise ValueError(f"Time coordinate length does not match data shape in {path}")

    if "label" in dims:
        label_axis = dims.index("label")
        label_names = _coord_values(coords, "label", data.shape[label_axis])
        if lateralize:
            data, pair_names = lateralize_labels(
                data,
                label_names,
                label_axis=label_axis,
            )
            if data.ndim == 3:
                feature_names = pair_names
            else:
                feature_names = [
                    f"{component}:{pair}"
                    for component in _coord_values(coords, "component", data.shape[1])
                    for pair in pair_names
                ]
        else:
            selected = _select_label_indices(label_names, labels)
            data = np.take(data, selected, axis=label_axis)
            selected_labels = [str(label_names[index]) for index in selected]
            if data.ndim == 3:
                feature_names = selected_labels
            else:
                components = [
                    str(component)
                    for component in _coord_values(coords, "component", data.shape[1])
                ]
                feature_names = [
                    f"{component}:{label}"
                    for component in components
                    for label in selected_labels
                ]
    elif labels:
        raise ValueError("Label selection requires a label coordinate")
    elif lateralize:
        raise ValueError("Lateralization requires a label coordinate")
    elif "source" in dims:
        source_axis = dims.index("source")
        source_names = [
            str(value)
            for value in _coord_values(coords, "source", data.shape[source_axis])
        ]
        if data.ndim == 3:
            feature_names = source_names
        else:
            orientation_axis = dims.index("orientation") if "orientation" in dims else 2
            orientations = [
                str(value)
                for value in _coord_values(
                    coords,
                    "orientation",
                    data.shape[orientation_axis],
                )
            ]
            feature_names = [
                f"{source}:{orientation}"
                for source in source_names
                for orientation in orientations
            ]
    else:
        n_features = int(np.prod(data.shape[1:-1]))
        feature_names = [f"feature_{index}" for index in range(n_features)]

    flattened = data.reshape(
        data.shape[0],
        int(np.prod(data.shape[1:-1])),
        data.shape[-1],
    )
    return flattened, time_values, feature_names, loaded.metadata


def trial_table_for_array(
    path: str | Path,
    metadata: Mapping[str, object],
) -> Optional[Path]:
    """Resolve the trial metadata table associated with an ERP array."""
    path = Path(path)
    stage_metadata = metadata.get("metadata", {})
    trial_table = (
        stage_metadata.get("trial_table")
        if isinstance(stage_metadata, Mapping)
        else None
    )
    if trial_table:
        return require_file(trial_table, purpose="ERP trial metadata")
    candidate = path.with_name(path.name.replace("_erp.npy", "_erptrials.tsv"))
    return candidate if candidate.is_file() else None
