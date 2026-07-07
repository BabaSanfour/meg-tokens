"""
Stage 8 time-resolved decoding over staged ERP or power derivatives.

The command builds a real feature matrix from Stage 6 ERP arrays or Stage 4
source-power arrays and writes BIDS-derivatives-style decoding outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from meg_tokens.io import derivative_path, load_array, require_file, save_array, save_sidecar
from meg_tokens.meg.decoding import compute_decoding_permutations, compute_time_resolved_decoding
from meg_tokens.utils.batch_erp_parcellation import erp_derivative_path
from meg_tokens.utils.batch_processor import normalize_subject_id
from meg_tokens.utils.batch_time_frequency import power_derivative_path
from meg_tokens.utils.epochs_builder import parse_run_label


def load_decoding_inputs(data_dir: str, conditions: list[str]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Load prebuilt real decoding tensors from a staged derivative directory.

    Expected files:
    - `X.npy`: epochs x features x time
    - `y.npy`: epoch labels, either strings matching `conditions` or numeric labels
    - `groups.npy`: optional subject/group labels
    - `times_ms.npy`: optional time coordinate in milliseconds
    """
    base = Path(data_dir)
    X = load_array(base / "X.npy", expected_ndim=3).data
    y = load_array(base / "y.npy", expected_ndim=1, allow_pickle=True).data

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y disagree on epochs: {X.shape[0]} != {y.shape[0]}")

    groups_path = base / "groups.npy"
    groups = load_array(groups_path, expected_ndim=1, allow_pickle=True).data if groups_path.exists() else None
    if groups is not None and groups.shape[0] != X.shape[0]:
        raise ValueError(f"X and groups disagree on epochs: {X.shape[0]} != {groups.shape[0]}")

    if y.dtype.kind in {"U", "S", "O"}:
        keep = np.isin(y, conditions)
        missing = sorted(set(conditions) - set(y[keep].astype(str)))
        if missing:
            raise ValueError(f"Requested conditions absent from y.npy: {missing}")
        X = X[keep]
        y = y[keep].astype(str)
        if groups is not None:
            groups = groups[keep]
        label_to_int = {label: idx for idx, label in enumerate(conditions)}
        y = np.array([label_to_int[label] for label in y], dtype=int)

    times_path = base / "times_ms.npy"
    if times_path.exists():
        times = load_array(times_path, expected_ndim=1).data
        if times.shape[0] != X.shape[-1]:
            raise ValueError(f"times_ms length {times.shape[0]} does not match X time dimension {X.shape[-1]}")
    else:
        times = np.arange(X.shape[-1], dtype=float)

    return X, y, groups, times


def decoding_derivative_path(
    output_root: str,
    *,
    conditions: Sequence[str],
    feature_source: str,
    align_to: str,
    source_method: str,
    parc: Optional[str],
    band: Optional[str],
    suffix: str,
    extension: str,
    roi: Optional[str] = None,
    lateralized: bool = False,
) -> Path:
    """Build a group-level decoding derivative path."""
    desc_parts = [
        conditions[0].lower(),
        "vs",
        *[condition.lower() for condition in conditions[1:]],
        feature_source,
        align_to,
        source_method,
    ]
    if parc:
        desc_parts.append(parc)
    if band:
        desc_parts.append(band.replace("_", "-"))
    if roi:
        desc_parts.append(roi)
    if lateralized:
        desc_parts.append("lateralized")
    return derivative_path(
        output_root,
        subject="group",
        datatype="meg",
        task="tokens",
        description="-".join(desc_parts),
        suffix=suffix,
        extension=extension,
    )


def decoding_figure_path(
    output_root: str,
    *,
    conditions: Sequence[str],
    feature_source: str,
    align_to: str,
    source_method: str,
    parc: Optional[str],
    band: Optional[str],
    roi: Optional[str] = None,
    lateralized: bool = False,
) -> Path:
    return decoding_derivative_path(
        output_root,
        conditions=conditions,
        feature_source=feature_source,
        align_to=align_to,
        source_method=source_method,
        parc=parc,
        band=band,
        roi=roi,
        lateralized=lateralized,
        suffix="decodingplot",
        extension=".png",
    ).parent.parent / "fig" / decoding_derivative_path(
        output_root,
        conditions=conditions,
        feature_source=feature_source,
        align_to=align_to,
        source_method=source_method,
        parc=parc,
        band=band,
        roi=roi,
        lateralized=lateralized,
        suffix="decodingplot",
        extension=".png",
    ).name


def discover_subjects(
    feature_dir: str,
    *,
    conditions: Sequence[str],
    feature_source: str,
    align_to: str,
    source_method: str,
    parc: Optional[str],
    band: Optional[str],
    power_method: str,
) -> list[str]:
    """Discover subjects with feature derivatives for every requested condition."""
    subject_sets = []
    for condition in conditions:
        if feature_source == "erp":
            if parc is None:
                raise ValueError("parc is required for ERP feature discovery")
            pattern = (
                f"**/sub-*_task-tokens_run-*_desc-{condition.lower()}-"
                f"{align_to}-{source_method}-{parc}_erp.npy"
            )
        elif feature_source == "power":
            if band is None:
                raise ValueError("band is required for power feature discovery")
            pattern = (
                f"**/sub-*_task-tokens_run-*_desc-{condition.lower()}-"
                f"{align_to}-{source_method}-{power_method}-{band.replace('_', '-')}_power.npy"
            )
        else:
            raise ValueError("feature_source must be 'erp' or 'power'")

        subject_sets.append({
            path.name.split("_", 1)[0].replace("sub-", "")
            for path in Path(feature_dir).glob(pattern)
            if path.is_file()
        })

    common = set.intersection(*subject_sets) if subject_sets else set()
    if not common:
        raise FileNotFoundError(f"No subjects have {feature_source} derivatives for all conditions: {conditions}")
    return sorted(common)


def find_feature_arrays(
    feature_dir: str,
    subject: str,
    condition: str,
    *,
    feature_source: str,
    align_to: str,
    source_method: str,
    parc: Optional[str] = "HCPMMP1",
    band: Optional[str] = None,
    power_method: str = "hilbert",
    runs: Optional[Sequence[str]] = None,
) -> list[Path]:
    """Find Stage 4/Stage 6 feature arrays for one subject-condition cell."""
    subject = normalize_subject_id(subject)
    if feature_source == "erp":
        if parc is None:
            raise ValueError("parc is required for ERP decoding")
        if runs:
            candidates = [
                erp_derivative_path(
                    feature_dir,
                    subject=subject,
                    run=parse_run_label(run)[0],
                    condition=condition,
                    align_to=align_to,
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
                f"{align_to}-{source_method}-{parc}_erp.npy"
            )
            existing = sorted(path for path in Path(feature_dir).glob(pattern) if path.is_file())
    elif feature_source == "power":
        if band is None:
            raise ValueError("band is required for power decoding")
        if runs:
            candidates = [
                power_derivative_path(
                    feature_dir,
                    subject=subject,
                    run=parse_run_label(run)[0],
                    condition=condition,
                    align_to=align_to,
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
                f"{align_to}-{source_method}-{power_method}-{band.replace('_', '-')}_power.npy"
            )
            existing = sorted(path for path in Path(feature_dir).glob(pattern) if path.is_file())
    else:
        raise ValueError("feature_source must be 'erp' or 'power'")

    if not existing:
        raise FileNotFoundError(
            f"No {feature_source} derivatives found for subject={subject}, "
            f"condition={condition}, alignment={align_to}"
        )
    return sorted(existing)


def _coord_values(coords: Mapping[str, object], key: str, length: int) -> list:
    values = coords.get(key)
    if values is None:
        return list(range(length))
    return list(values)


def _select_label_indices(label_names: Sequence[object], labels: Optional[Sequence[str]]) -> list[int]:
    if not labels:
        return list(range(len(label_names)))
    lookup = {str(label): idx for idx, label in enumerate(label_names)}
    selected = []
    for label in labels:
        if str(label).isdigit():
            idx = int(label)
            if idx < 0 or idx >= len(label_names):
                raise ValueError(f"Label index {idx} is out of range")
            selected.append(idx)
        else:
            if label not in lookup:
                raise ValueError(f"Requested label is absent from feature coordinates: {label}")
            selected.append(lookup[label])
    return selected


def _label_pair_key(label: object) -> tuple[str, str]:
    text = str(label)
    lower = text.lower()
    if lower.endswith("-lh"):
        hemi = "lh"
        base = text[:-3]
    elif lower.endswith("-rh"):
        hemi = "rh"
        base = text[:-3]
    else:
        raise ValueError(f"Cannot infer hemisphere from label name: {text}")
    if base.startswith("L_") or base.startswith("R_"):
        base = base[2:]
    if base.startswith("L-") or base.startswith("R-"):
        base = base[2:]
    return base, hemi


def _lateralize_data(data: np.ndarray, label_names: Sequence[object]) -> tuple[np.ndarray, list[str]]:
    pairs: dict[str, dict[str, int]] = {}
    for idx, label in enumerate(label_names):
        base, hemi = _label_pair_key(label)
        pairs.setdefault(base, {})[hemi] = idx

    complete = [(base, sides["lh"], sides["rh"]) for base, sides in sorted(pairs.items()) if {"lh", "rh"} <= set(sides)]
    if not complete:
        raise ValueError("No left/right label pairs were found for lateralized decoding")

    pair_names = []
    diffs = []
    for base, lh_idx, rh_idx in complete:
        if data.ndim == 3:
            diffs.append(data[:, lh_idx, :] - data[:, rh_idx, :])
        elif data.ndim == 4:
            diffs.append(data[:, :, lh_idx, :] - data[:, :, rh_idx, :])
        else:
            raise ValueError(f"Unexpected labeled data shape for lateralization: {data.shape}")
        pair_names.append(base)

    if data.ndim == 3:
        return np.stack(diffs, axis=1), pair_names
    return np.stack(diffs, axis=2), pair_names


def _load_feature_array(
    path: Path,
    *,
    labels: Optional[Sequence[str]] = None,
    lateralize: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, object]]:
    loaded = load_array(path, require_sidecar=True)
    data = np.asarray(loaded.data, dtype=float)
    dims = tuple(loaded.metadata.get("dims", []))
    coords = loaded.metadata.get("coords", {})
    if not dims or dims[0] != "trial":
        raise ValueError(f"Expected first dimension to be trial in {path}, got {dims}")
    if data.ndim not in (3, 4):
        raise ValueError(f"Expected feature array to be 3D or 4D, got shape {data.shape} in {path}")

    time = coords.get("time_sec", coords.get("time_ms"))
    if time is None:
        time_values = np.arange(data.shape[-1], dtype=float)
    else:
        time_values = np.asarray(time, dtype=float)
    if len(time_values) != data.shape[-1]:
        raise ValueError(f"Time coordinate length does not match data shape in {path}")

    feature_names: list[str]
    if "label" in dims:
        label_axis = dims.index("label")
        label_names = _coord_values(coords, "label", data.shape[label_axis])
        if lateralize:
            data, pair_names = _lateralize_data(data, label_names)
            feature_names = pair_names if data.ndim == 3 else [
                f"{component}:{pair}"
                for component in _coord_values(coords, "component", data.shape[1])
                for pair in pair_names
            ]
        else:
            selected = _select_label_indices(label_names, labels)
            data = np.take(data, selected, axis=label_axis)
            selected_labels = [str(label_names[idx]) for idx in selected]
            if data.ndim == 3:
                feature_names = selected_labels
            else:
                components = [str(component) for component in _coord_values(coords, "component", data.shape[1])]
                feature_names = [f"{component}:{label}" for component in components for label in selected_labels]
    elif labels:
        raise ValueError("Label selection requires feature arrays with a label coordinate")
    elif lateralize:
        raise ValueError("Lateralized decoding requires feature arrays with a label coordinate")
    elif "source" in dims:
        source_axis = dims.index("source")
        source_names = [str(value) for value in _coord_values(coords, "source", data.shape[source_axis])]
        if data.ndim == 3:
            feature_names = source_names
        else:
            orientation_axis = dims.index("orientation") if "orientation" in dims else 2
            orientations = [
                str(value)
                for value in _coord_values(coords, "orientation", data.shape[orientation_axis])
            ]
            feature_names = [
                f"{source}:{orientation}"
                for source in source_names
                for orientation in orientations
            ]
    else:
        n_features = int(np.prod(data.shape[1:-1]))
        feature_names = [f"feature_{idx}" for idx in range(n_features)]

    X = data.reshape(data.shape[0], int(np.prod(data.shape[1:-1])), data.shape[-1])
    return X, time_values, feature_names, loaded.metadata


def _trial_table_for_array(path: Path, metadata: Mapping[str, object]) -> Optional[Path]:
    stage_meta = metadata.get("metadata", {})
    trial_table = stage_meta.get("trial_table") if isinstance(stage_meta, Mapping) else None
    if trial_table:
        return require_file(trial_table, purpose="ERP trial metadata")
    candidate = path.with_name(path.name.replace("_erp.npy", "_erptrials.tsv"))
    return candidate if candidate.is_file() else None


def _indices_from_class_column(
    path: Path,
    metadata: Mapping[str, object],
    *,
    class_column: str,
    class_values: Sequence[str],
) -> dict[int, np.ndarray]:
    trial_table = _trial_table_for_array(path, metadata)
    if trial_table is None:
        raise ValueError(f"class_column decoding requires an ERP trial table for {path}")
    table = pd.read_csv(trial_table, sep="\t")
    if class_column not in table.columns:
        raise ValueError(f"Trial table {trial_table} is missing class column '{class_column}'")
    values = table[class_column].astype(str)
    out = {}
    for class_idx, value in enumerate(class_values):
        out[class_idx] = np.where(values == str(value))[0]
    return out


def build_decoding_dataset(
    feature_dir: str,
    *,
    conditions: Sequence[str],
    feature_source: str = "erp",
    subjects: Optional[Sequence[str]] = None,
    input_conditions: Optional[Sequence[str]] = None,
    align_to: str = "go",
    source_method: str = "dSPM",
    parc: Optional[str] = "HCPMMP1",
    band: Optional[str] = None,
    power_method: str = "hilbert",
    runs_by_condition: Optional[Mapping[str, Sequence[str]]] = None,
    labels: Optional[Sequence[str]] = None,
    lateralize: bool = False,
    class_column: Optional[str] = None,
    class_values: Optional[Sequence[str]] = None,
) -> dict[str, object]:
    """Build X/y/groups from staged derivative arrays."""
    if len(conditions) < 2:
        raise ValueError("At least two decoding conditions are required")
    conditions = [str(condition) for condition in conditions]
    runs_by_condition = runs_by_condition or {}

    if class_column:
        if feature_source != "erp":
            raise ValueError("class_column decoding is only supported for ERP derivatives")
        if not class_values:
            class_values = conditions
        if len(class_values) != len(conditions):
            raise ValueError("class_values must have the same length as conditions")
        search_conditions = list(input_conditions or conditions)
    else:
        class_values = conditions
        search_conditions = conditions

    if subjects is None:
        subjects = discover_subjects(
            feature_dir,
            conditions=search_conditions,
            feature_source=feature_source,
            align_to=align_to,
            source_method=source_method,
            parc=parc,
            band=band,
            power_method=power_method,
        )
    else:
        subjects = [normalize_subject_id(subject) for subject in subjects]

    X_parts = []
    y_parts = []
    group_parts = []
    input_paths = []
    reference_time = None
    reference_features = None

    for subject in subjects:
        for search_condition in search_conditions:
            paths = find_feature_arrays(
                feature_dir,
                subject,
                search_condition,
                feature_source=feature_source,
                align_to=align_to,
                source_method=source_method,
                parc=parc,
                band=band,
                power_method=power_method,
                runs=runs_by_condition.get(search_condition),
            )
            for path in paths:
                X_path, time_values, feature_names, metadata = _load_feature_array(
                    path,
                    labels=labels,
                    lateralize=lateralize,
                )
                if reference_time is None:
                    reference_time = time_values
                    reference_features = feature_names
                else:
                    if not np.allclose(reference_time, time_values, equal_nan=True):
                        raise ValueError(f"Time coordinates changed across feature arrays at {path}")
                    if list(reference_features) != list(feature_names):
                        raise ValueError(f"Feature coordinates changed across feature arrays at {path}")

                if class_column:
                    class_indices = _indices_from_class_column(
                        path,
                        metadata,
                        class_column=class_column,
                        class_values=class_values,
                    )
                    for class_idx, indices in class_indices.items():
                        if len(indices) == 0:
                            continue
                        X_parts.append(X_path[indices])
                        y_parts.append(np.full(len(indices), class_idx, dtype=int))
                        group_parts.append(np.array([subject] * len(indices), dtype=object))
                else:
                    class_idx = conditions.index(search_condition)
                    X_parts.append(X_path)
                    y_parts.append(np.full(X_path.shape[0], class_idx, dtype=int))
                    group_parts.append(np.array([subject] * X_path.shape[0], dtype=object))
                input_paths.append(str(path))

    if not X_parts:
        raise ValueError("No trials were available for decoding after loading/filtering derivatives")

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    groups = np.concatenate(group_parts, axis=0)
    present = sorted(set(y.tolist()))
    expected = list(range(len(conditions)))
    if present != expected:
        missing = [conditions[idx] for idx in expected if idx not in present]
        raise ValueError(f"No decoding trials were found for conditions: {missing}")

    return {
        "X": X,
        "y": y,
        "groups": groups,
        "time_sec": reference_time,
        "feature_names": reference_features,
        "subjects": list(subjects),
        "input_paths": input_paths,
        "conditions": conditions,
        "class_values": list(class_values),
    }


def _full_time_array(values: np.ndarray, valid_time_mask: np.ndarray) -> np.ndarray:
    out_shape = values.shape[:-1] + (len(valid_time_mask),)
    out = np.full(out_shape, np.nan, dtype=float)
    out[..., valid_time_mask] = values
    return out


def run_batch_decoding(
    feature_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    conditions: Sequence[str] = ("Fast", "Slow"),
    *,
    feature_source: str = "erp",
    subjects: Optional[Sequence[str]] = None,
    input_conditions: Optional[Sequence[str]] = None,
    align_to: str = "go",
    source_method: str = "dSPM",
    parc: Optional[str] = "HCPMMP1",
    band: Optional[str] = None,
    power_method: str = "hilbert",
    runs_by_condition: Optional[Mapping[str, Sequence[str]]] = None,
    labels: Optional[Sequence[str]] = None,
    lateralize: bool = False,
    class_column: Optional[str] = None,
    class_values: Optional[Sequence[str]] = None,
    data_dir: Optional[str] = None,
    permutations: int = 0,
    n_jobs: int = 4,
) -> dict[str, Path]:
    """Run time-resolved decoding and save derivative outputs."""
    conditions = [str(condition) for condition in conditions]
    if data_dir is not None:
        X, y, groups, times = load_decoding_inputs(data_dir, conditions)
        dataset = {
            "X": X,
            "y": y,
            "groups": groups,
            "time_sec": np.asarray(times, dtype=float) / 1000.0,
            "feature_names": [f"feature_{idx}" for idx in range(X.shape[1])],
            "subjects": sorted(set(groups.tolist())) if groups is not None else [],
            "input_paths": [str(Path(data_dir) / "X.npy")],
            "conditions": conditions,
            "class_values": conditions,
        }
        if output_dir is None:
            raise ValueError("output_dir is required")
    else:
        if feature_dir is None:
            raise ValueError("feature_dir is required unless data_dir is supplied")
        if output_dir is None:
            raise ValueError("output_dir is required")
        dataset = build_decoding_dataset(
            feature_dir,
            conditions=conditions,
            feature_source=feature_source,
            subjects=subjects,
            input_conditions=input_conditions,
            align_to=align_to,
            source_method=source_method,
            parc=parc,
            band=band,
            power_method=power_method,
            runs_by_condition=runs_by_condition,
            labels=labels,
            lateralize=lateralize,
            class_column=class_column,
            class_values=class_values,
        )

    X = np.asarray(dataset["X"], dtype=float)
    y = np.asarray(dataset["y"], dtype=int)
    groups = dataset["groups"]
    times = np.asarray(dataset["time_sec"], dtype=float)
    valid_time_mask = np.all(np.isfinite(X), axis=(0, 1))
    if not np.any(valid_time_mask):
        raise ValueError("No finite time points are available for decoding")
    X_valid = X[:, :, valid_time_mask]

    print(f"=== Time-resolved decoding: {' vs '.join(conditions)} ===")
    print(f"Feature source: {feature_source}; X={X.shape}; finite times={int(np.sum(valid_time_mask))}/{X.shape[-1]}")

    if permutations > 0:
        split_scores, perm_scores, threshold = compute_decoding_permutations(
            X=X_valid,
            y=y,
            groups=groups,
            balance=True,
            n_permutations=permutations,
            n_jobs=n_jobs,
        )
        perm_scores_full = _full_time_array(np.asarray(perm_scores), valid_time_mask)
    else:
        split_scores = compute_time_resolved_decoding(
            X=X_valid,
            y=y,
            groups=groups,
            balance=True,
            n_jobs=n_jobs,
        )
        threshold = None
        perm_scores_full = None

    split_scores_full = _full_time_array(np.asarray(split_scores), valid_time_mask)
    mean_scores = np.full(split_scores_full.shape[-1], np.nan, dtype=float)
    mean_scores[valid_time_mask] = np.mean(np.asarray(split_scores), axis=0)
    roi_name = None if not labels or len(labels) != 1 else str(labels[0])
    metadata = {
        "stage": "decoding",
        "kind": "time_resolved_classification",
        "conditions": conditions,
        "class_values": dataset["class_values"],
        "feature_source": feature_source,
        "alignment": align_to,
        "source_method": source_method,
        "parcellation": parc,
        "band": band,
        "power_method": power_method if feature_source == "power" else None,
        "labels": list(labels) if labels else None,
        "lateralized": bool(lateralize),
        "class_column": class_column,
        "classifier": "LinearDiscriminantAnalysis",
        "cross_validation": "LeaveOneGroupOut" if groups is not None else "StratifiedKFold",
        "subjects": dataset["subjects"],
        "n_trials": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_times": int(X.shape[-1]),
        "valid_time_mask": valid_time_mask.tolist(),
        "input_paths": dataset["input_paths"],
        "chance_level": 1.0 / len(conditions),
        "n_permutations": int(permutations),
    }

    paths = {
        "score": decoding_derivative_path(
            output_dir,
            conditions=conditions,
            feature_source=feature_source,
            align_to=align_to,
            source_method=source_method,
            parc=parc,
            band=band,
            roi=roi_name,
            lateralized=lateralize,
            suffix="decoding",
            extension=".npy",
        ),
        "splits": decoding_derivative_path(
            output_dir,
            conditions=conditions,
            feature_source=feature_source,
            align_to=align_to,
            source_method=source_method,
            parc=parc,
            band=band,
            roi=roi_name,
            lateralized=lateralize,
            suffix="decodingsplits",
            extension=".npy",
        ),
        "plot": decoding_figure_path(
            output_dir,
            conditions=conditions,
            feature_source=feature_source,
            align_to=align_to,
            source_method=source_method,
            parc=parc,
            band=band,
            roi=roi_name,
            lateralized=lateralize,
        ),
    }

    coords = {"time_sec": times}
    save_array(paths["score"], mean_scores, dims=("time",), coords=coords, metadata={**metadata, "statistic": "mean_accuracy"})
    save_array(
        paths["splits"],
        split_scores_full,
        dims=("split", "time"),
        coords={"split": list(range(split_scores_full.shape[0])), **coords},
        metadata={**metadata, "statistic": "cross_validation_accuracy"},
    )

    if threshold is not None:
        paths["threshold"] = decoding_derivative_path(
            output_dir,
            conditions=conditions,
            feature_source=feature_source,
            align_to=align_to,
            source_method=source_method,
            parc=parc,
            band=band,
            roi=roi_name,
            lateralized=lateralize,
            suffix="decodingthreshold",
            extension=".npy",
        )
        paths["permutations"] = decoding_derivative_path(
            output_dir,
            conditions=conditions,
            feature_source=feature_source,
            align_to=align_to,
            source_method=source_method,
            parc=parc,
            band=band,
            roi=roi_name,
            lateralized=lateralize,
            suffix="decodingpermutations",
            extension=".npy",
        )
        save_array(
            paths["threshold"],
            np.array([threshold], dtype=float),
            dims=("threshold",),
            metadata={**metadata, "statistic": "fwe_accuracy_threshold_95"},
        )
        save_array(
            paths["permutations"],
            perm_scores_full,
            dims=("permutation", "time"),
            coords={"permutation": list(range(perm_scores_full.shape[0])), **coords},
            metadata={**metadata, "statistic": "permutation_accuracy"},
        )

    _save_decoding_plot(
        paths["plot"],
        times=times,
        mean_scores=mean_scores,
        conditions=conditions,
        threshold=threshold,
        chance=1.0 / len(conditions),
    )
    save_sidecar(paths["plot"], {**metadata, "kind": "decoding_timecourse_figure"})
    print(f"Saved decoding scores: {paths['score']}")
    return paths


def _save_decoding_plot(
    path: Path,
    *,
    times: np.ndarray,
    mean_scores: np.ndarray,
    conditions: Sequence[str],
    threshold: Optional[float],
    chance: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(times, mean_scores, label="LDA accuracy", lw=2, color="#2f5d62")
    ax.axhline(chance, color="#333333", linestyle="--", lw=1, label="chance")
    if threshold is not None:
        ax.axhline(threshold, color="#b85c38", linestyle=":", lw=1.5, label="FWE 95%")
        significant = mean_scores > threshold
        if np.any(significant):
            ax.fill_between(times, mean_scores, threshold, where=significant, color="#b85c38", alpha=0.22)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Accuracy")
    ax.set_title(" vs ".join(conditions))
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _runs_by_condition(conditions: Sequence[str], runs: Optional[Sequence[str]]) -> dict[str, Sequence[str]]:
    if not runs:
        return {}
    if len(conditions) == 1:
        return {conditions[0]: list(runs)}
    if len(runs) % len(conditions) != 0:
        raise ValueError("--runs must contain an equal number of run labels per condition")
    per_condition = len(runs) // len(conditions)
    return {
        condition: list(runs[idx * per_condition:(idx + 1) * per_condition])
        for idx, condition in enumerate(conditions)
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run time-resolved decoding from staged derivatives.")
    parser.add_argument("--feature_dir", type=str, default=None,
                        help="BIDS derivatives root containing Stage 4 power or Stage 6 ERP arrays.")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Compatibility path containing X.npy/y.npy/groups.npy/times_ms.npy.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="BIDS derivatives root for decoding outputs.")
    parser.add_argument("--feature_source", type=str, default="erp", choices=["erp", "power"])
    parser.add_argument("--conditions", type=str, nargs="+", default=["Fast", "Slow"])
    parser.add_argument("--input_conditions", type=str, nargs="+", default=None,
                        help="Run/file conditions to load when class labels come from --class_column.")
    parser.add_argument("--subjects", type=str, nargs="+", default=None)
    parser.add_argument("--align_to", "--alignment", dest="align_to", type=str, default="go",
                        choices=["go", "enter", "feedback"])
    parser.add_argument("--source_method", type=str, default="dSPM")
    parser.add_argument("--parc", type=str, default="HCPMMP1")
    parser.add_argument("--band", type=str, default=None,
                        help="Power band for --feature_source power.")
    parser.add_argument("--power_method", type=str, default="hilbert")
    parser.add_argument("--runs", type=str, nargs="+", default=None,
                        help="Optional run labels. If multiple conditions are supplied, provide equal-sized run blocks per condition.")
    parser.add_argument("--labels", "--roi", dest="labels", type=str, nargs="+", default=None,
                        help="Optional ERP label names or indices to decode.")
    parser.add_argument("--lateralize", action="store_true",
                        help="Use left-minus-right paired label features for ERP arrays.")
    parser.add_argument("--class_column", type=str, default=None,
                        help="Trial metadata column to use as class labels for ERP derivatives.")
    parser.add_argument("--class_values", type=str, nargs="+", default=None,
                        help="Values in --class_column corresponding to --conditions.")
    parser.add_argument("--permutations", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=4)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.feature_dir is None and args.data_dir is None:
        parser.error("--feature_dir is required unless --data_dir is supplied")
    if args.feature_source == "power" and args.band is None and args.data_dir is None:
        parser.error("--band is required when --feature_source power")
    run_batch_decoding(
        feature_dir=args.feature_dir,
        output_dir=args.out_dir,
        conditions=args.conditions,
        feature_source=args.feature_source,
        subjects=args.subjects,
        input_conditions=args.input_conditions,
        align_to=args.align_to,
        source_method=args.source_method,
        parc=args.parc,
        band=args.band,
        power_method=args.power_method,
        runs_by_condition=_runs_by_condition(args.conditions if not args.class_column else (args.input_conditions or args.conditions), args.runs),
        labels=args.labels,
        lateralize=args.lateralize,
        class_column=args.class_column,
        class_values=args.class_values,
        data_dir=args.data_dir,
        permutations=args.permutations,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
