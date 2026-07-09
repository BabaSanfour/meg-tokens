"""
Stage 6 group-level statistics over parcellated ERP derivatives.

This stage consumes Stage 5 `*_erp.npy` array derivatives, builds paired
subject-level condition contrasts, and writes statistics as `.npy` arrays with
JSON sidecars plus a table of significant time windows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from meg_tokens.core import (
    LateralizedStatisticsConfig,
    ProjectConfig,
    StatisticsConfig,
    WorkflowResult,
    normalize_subject_id,
)
from meg_tokens.analysis.statistics import compute_permutation_t_test, get_significance_windows
from meg_tokens.features.dataset import lateralize_labels
from meg_tokens.io import DerivativeLayout, load_array, save_array, save_table


def stats_derivative_path(
    output_root: str,
    *,
    conditions: Sequence[str],
    align_to: str,
    source_method: str,
    parc: str,
    suffix: str,
    extension: str,
) -> Path:
    """Build a group-level stats derivative path."""
    return DerivativeLayout(output_root).group_stats(
        conditions=conditions,
        alignment=align_to,
        source_method=source_method,
        parc=parc,
        suffix=suffix,
        extension=extension,
    )


def find_erp_arrays(
    erp_dir: str,
    subject: str,
    condition: str,
    *,
    align_to: str,
    source_method: str,
    parc: str,
    runs: Optional[Sequence[str]] = None,
) -> list[Path]:
    """Find Stage 5 ERP array derivatives for one subject and condition."""
    return DerivativeLayout(erp_dir).find_erp(
        subject=subject,
        condition=condition,
        alignment=align_to,
        source_method=source_method,
        parc=parc,
        runs=runs,
    )


def discover_subjects(
    erp_dir: str,
    conditions: Sequence[str],
    *,
    align_to: str,
    source_method: str,
    parc: str,
) -> list[str]:
    """Discover subjects that have at least one ERP derivative for both conditions."""
    return DerivativeLayout(erp_dir).discover_erp_subjects(
        conditions=conditions,
        alignment=align_to,
        source_method=source_method,
        parc=parc,
    )


def _coords_close(left, right) -> bool:
    try:
        return np.allclose(np.asarray(left, dtype=float), np.asarray(right, dtype=float), equal_nan=True)
    except (TypeError, ValueError):
        return list(left) == list(right)


def _load_subject_condition_mean(paths: Sequence[Path]) -> tuple[np.ndarray, tuple[str, ...], dict[str, object], list[str]]:
    """Average trials, then runs, for one subject-condition cell."""
    run_means = []
    reference_dims = None
    reference_coords = None
    source_paths = []

    for path in paths:
        loaded = load_array(path, require_sidecar=True)
        data = loaded.data
        sidecar = loaded.metadata
        dims = tuple(sidecar.get("dims", []))
        if not dims or dims[0] != "trial":
            raise ValueError(f"Expected first dimension to be trial in {path}, got {dims}")
        if data.ndim not in (3, 4):
            raise ValueError(f"Expected ERP array to be 3D or 4D, got shape {data.shape} in {path}")

        feature_dims = dims[1:]
        coords = sidecar.get("coords", {})
        feature_coords = {key: value for key, value in coords.items() if key != "trial"}
        trial_mean = np.nanmean(data, axis=0)

        if reference_dims is None:
            reference_dims = feature_dims
            reference_coords = feature_coords
        else:
            if feature_dims != reference_dims:
                raise ValueError(f"ERP feature dims changed across runs: {feature_dims} != {reference_dims}")
            for key, value in reference_coords.items():
                if key in feature_coords and not _coords_close(value, feature_coords[key]):
                    raise ValueError(f"ERP coordinate '{key}' changed across runs")

        run_means.append(trial_mean)
        source_paths.append(str(path))

    if reference_dims is None or reference_coords is None:
        raise ValueError("No ERP paths were provided")
    return np.nanmean(np.stack(run_means, axis=0), axis=0), reference_dims, reference_coords, source_paths


def _load_condition_matrix(
    erp_dir: str,
    subjects: Sequence[str],
    condition: str,
    *,
    align_to: str,
    source_method: str,
    parc: str,
    runs: Optional[Sequence[str]],
) -> tuple[np.ndarray, tuple[str, ...], dict[str, object], dict[str, list[str]]]:
    subject_means = []
    reference_dims = None
    reference_coords = None
    inputs = {}

    for subject in subjects:
        paths = find_erp_arrays(
            erp_dir,
            subject,
            condition,
            align_to=align_to,
            source_method=source_method,
            parc=parc,
            runs=runs,
        )
        subject_mean, dims, coords, source_paths = _load_subject_condition_mean(paths)
        if reference_dims is None:
            reference_dims = dims
            reference_coords = coords
        else:
            if dims != reference_dims:
                raise ValueError(f"ERP feature dims changed across subjects: {dims} != {reference_dims}")
            for key, value in reference_coords.items():
                if key in coords and not _coords_close(value, coords[key]):
                    raise ValueError(f"ERP coordinate '{key}' changed across subjects")
        subject_means.append(subject_mean)
        inputs[normalize_subject_id(subject)] = source_paths

    if reference_dims is None or reference_coords is None:
        raise ValueError(f"No ERP data loaded for condition {condition}")
    return np.stack(subject_means, axis=0), reference_dims, reference_coords, inputs


def _window_table(
    p_values: np.ndarray,
    dims: Sequence[str],
    coords: Mapping[str, object],
    *,
    alpha: float,
) -> pd.DataFrame:
    time_values = coords.get("time_sec", list(range(p_values.shape[-1])))
    rows = []

    if tuple(dims) == ("label", "time"):
        labels = coords.get("label", list(range(p_values.shape[0])))
        for label_idx, label in enumerate(labels):
            for start, end in get_significance_windows(p_values[label_idx], alpha=alpha):
                rows.append({
                    "label": label,
                    "label_index": label_idx,
                    "start_index": int(start),
                    "end_index_exclusive": int(end),
                    "start_time_sec": time_values[start] if start < len(time_values) else np.nan,
                    "end_time_sec": time_values[end - 1] if end - 1 < len(time_values) else np.nan,
                    "alpha": alpha,
                })
    elif tuple(dims) == ("component", "label", "time"):
        components = coords.get("component", list(range(p_values.shape[0])))
        labels = coords.get("label", list(range(p_values.shape[1])))
        for comp_idx, component in enumerate(components):
            for label_idx, label in enumerate(labels):
                for start, end in get_significance_windows(p_values[comp_idx, label_idx], alpha=alpha):
                    rows.append({
                        "component": component,
                        "component_index": comp_idx,
                        "label": label,
                        "label_index": label_idx,
                        "start_index": int(start),
                        "end_index_exclusive": int(end),
                        "start_time_sec": time_values[start] if start < len(time_values) else np.nan,
                        "end_time_sec": time_values[end - 1] if end - 1 < len(time_values) else np.nan,
                        "alpha": alpha,
                    })
    else:
        return pd.DataFrame(columns=["feature_index", "start_index", "end_index_exclusive", "alpha"])

    if rows:
        return pd.DataFrame(rows)
    columns = ["label", "label_index", "start_index", "end_index_exclusive", "start_time_sec", "end_time_sec", "alpha"]
    if tuple(dims) == ("component", "label", "time"):
        columns = ["component", "component_index", *columns]
    return pd.DataFrame(columns=columns)


def _permutation_test_subject_matrix(
    subject_matrix: np.ndarray,
    *,
    n_permutations: int,
    tail: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flat = subject_matrix.reshape(subject_matrix.shape[0], -1)
    valid_features = np.all(np.isfinite(flat), axis=0)
    if not np.any(valid_features):
        raise ValueError("No finite subject-level features are available for statistics")
    observed, p_values, h0 = compute_permutation_t_test(
        flat[:, valid_features],
        n_permutations=n_permutations,
        tail=tail,
        n_jobs=n_jobs,
    )
    feature_shape = subject_matrix.shape[1:]
    observed_full = np.full(flat.shape[1], np.nan, dtype=float)
    p_full = np.full(flat.shape[1], np.nan, dtype=float)
    observed_full[valid_features] = np.asarray(observed)
    p_full[valid_features] = np.asarray(p_values)
    return (
        observed_full.reshape(feature_shape),
        p_full.reshape(feature_shape),
        np.asarray(h0),
        valid_features,
    )


def run_group_statistics_contrast(
    erp_dir: str,
    out_dir: str,
    conditions: Sequence[str] = ("Fast", "Slow"),
    n_permutations: int = 1000,
    subjects_list: Optional[Sequence[str]] = None,
    *,
    align_to: str = "go",
    source_method: str = "dSPM",
    parc: str = "HCPMMP1",
    runs_by_condition: Optional[Mapping[str, Sequence[str]]] = None,
    alpha: float = 0.05,
    tail: int = 0,
    n_jobs: int = 1,
) -> dict[str, Path]:
    """Run a paired group-level permutation t-test for two ERP conditions."""
    if len(conditions) != 2:
        raise ValueError("Group statistics currently requires exactly two conditions")
    conditions = [str(condition) for condition in conditions]
    runs_by_condition = runs_by_condition or {}

    if subjects_list is None:
        subjects = discover_subjects(
            erp_dir,
            conditions,
            align_to=align_to,
            source_method=source_method,
            parc=parc,
        )
    else:
        subjects = [normalize_subject_id(subject) for subject in subjects_list]

    if len(subjects) < 2:
        raise ValueError(f"At least two subjects are required for group statistics, got {len(subjects)}")

    print(f"=== Group statistics: {conditions[0]} vs {conditions[1]} ===")
    print(f"Subjects: {', '.join(subjects)}")

    cond1, feature_dims, feature_coords, cond1_inputs = _load_condition_matrix(
        erp_dir,
        subjects,
        conditions[0],
        align_to=align_to,
        source_method=source_method,
        parc=parc,
        runs=runs_by_condition.get(conditions[0]),
    )
    cond2, dims2, coords2, cond2_inputs = _load_condition_matrix(
        erp_dir,
        subjects,
        conditions[1],
        align_to=align_to,
        source_method=source_method,
        parc=parc,
        runs=runs_by_condition.get(conditions[1]),
    )
    if dims2 != feature_dims:
        raise ValueError(f"Condition feature dims differ: {feature_dims} != {dims2}")
    for key, value in feature_coords.items():
        if key in coords2 and not _coords_close(value, coords2[key]):
            raise ValueError(f"Condition coordinate '{key}' differs")

    contrast = cond1 - cond2
    t_obs, p_values, h0, valid_features = _permutation_test_subject_matrix(
        contrast,
        n_permutations=n_permutations,
        tail=tail,
        n_jobs=n_jobs,
    )

    metadata = {
        "stage": "group_statistics",
        "kind": "paired_condition_contrast",
        "conditions": conditions,
        "contrast": f"{conditions[0]}-{conditions[1]}",
        "subjects": subjects,
        "alignment": align_to,
        "source_method": source_method,
        "parcellation": parc,
        "n_subjects": len(subjects),
        "n_permutations": int(n_permutations),
        "valid_feature_count": int(np.sum(valid_features)),
        "total_feature_count": int(valid_features.size),
        "tail": int(tail),
        "alpha": float(alpha),
        "condition_inputs": {
            conditions[0]: cond1_inputs,
            conditions[1]: cond2_inputs,
        },
    }
    coords = dict(feature_coords)

    paths = {
        "tstat": stats_derivative_path(
            out_dir,
            conditions=conditions,
            align_to=align_to,
            source_method=source_method,
            parc=parc,
            suffix="tstat",
            extension=".npy",
        ),
        "pvalue": stats_derivative_path(
            out_dir,
            conditions=conditions,
            align_to=align_to,
            source_method=source_method,
            parc=parc,
            suffix="pvalue",
            extension=".npy",
        ),
        "contrast": stats_derivative_path(
            out_dir,
            conditions=conditions,
            align_to=align_to,
            source_method=source_method,
            parc=parc,
            suffix="contrast",
            extension=".npy",
        ),
        "h0": stats_derivative_path(
            out_dir,
            conditions=conditions,
            align_to=align_to,
            source_method=source_method,
            parc=parc,
            suffix="h0",
            extension=".npy",
        ),
        "windows": stats_derivative_path(
            out_dir,
            conditions=conditions,
            align_to=align_to,
            source_method=source_method,
            parc=parc,
            suffix="sigwindows",
            extension=".tsv",
        ),
    }

    save_array(paths["tstat"], t_obs, dims=feature_dims, coords=coords, metadata={**metadata, "statistic": "t"})
    save_array(paths["pvalue"], p_values, dims=feature_dims, coords=coords, metadata={**metadata, "statistic": "p"})
    save_array(
        paths["contrast"],
        contrast,
        dims=("subject", *feature_dims),
        coords={"subject": subjects, **coords},
        metadata={**metadata, "statistic": "subject_contrast"},
    )
    save_array(
        paths["h0"],
        h0,
        dims=("permutation",),
        metadata={**metadata, "statistic": "max_statistic_null"},
    )
    save_table(
        paths["windows"],
        _window_table(p_values, feature_dims, coords, alpha=alpha),
        metadata={**metadata, "kind": "significant_time_windows"},
    )
    print(f"Saved group statistics to {paths['tstat'].parent}")
    return paths


def run_lateralized_statistics_test(
    erp_dir: str,
    out_dir: str,
    condition: str,
    *,
    subjects_list: Optional[Sequence[str]] = None,
    align_to: str = "go",
    source_method: str = "dSPM",
    parc: str = "HCPMMP1",
    runs: Optional[Sequence[str]] = None,
    n_permutations: int = 1000,
    alpha: float = 0.05,
    tail: int = 0,
    n_jobs: int = 1,
) -> dict[str, Path]:
    """Test homologous left-minus-right ERP labels against zero."""
    subjects = (
        discover_subjects(
            erp_dir,
            (condition,),
            align_to=align_to,
            source_method=source_method,
            parc=parc,
        )
        if subjects_list is None
        else [normalize_subject_id(subject) for subject in subjects_list]
    )
    if len(subjects) < 2:
        raise ValueError(
            f"At least two subjects are required for lateralized statistics, got {len(subjects)}"
        )

    condition_data, feature_dims, feature_coords, inputs = _load_condition_matrix(
        erp_dir,
        subjects,
        condition,
        align_to=align_to,
        source_method=source_method,
        parc=parc,
        runs=runs,
    )
    if "label" not in feature_dims:
        raise ValueError("Lateralized statistics require ERP arrays with a label dimension")
    label_axis = 1 + feature_dims.index("label")
    labels = feature_coords.get("label")
    if labels is None:
        raise ValueError("Lateralized statistics require named label coordinates")
    lateralized, pair_names = lateralize_labels(
        condition_data,
        labels,
        label_axis=label_axis,
    )
    coords = dict(feature_coords)
    coords["label"] = pair_names
    t_obs, p_values, h0, valid_features = _permutation_test_subject_matrix(
        lateralized,
        n_permutations=n_permutations,
        tail=tail,
        n_jobs=n_jobs,
    )

    layout = DerivativeLayout(out_dir)
    path_options = {
        "condition": condition,
        "alignment": align_to,
        "source_method": source_method,
        "parc": parc,
    }
    paths = {
        "tstat": layout.lateralized_stats(
            **path_options, suffix="tstat", extension=".npy"
        ),
        "pvalue": layout.lateralized_stats(
            **path_options, suffix="pvalue", extension=".npy"
        ),
        "lateralization": layout.lateralized_stats(
            **path_options, suffix="lateralization", extension=".npy"
        ),
        "h0": layout.lateralized_stats(
            **path_options, suffix="h0", extension=".npy"
        ),
        "windows": layout.lateralized_stats(
            **path_options, suffix="sigwindows", extension=".tsv"
        ),
    }
    metadata = {
        "stage": "group_statistics",
        "kind": "hemisphere_lateralization",
        "condition": condition,
        "contrast": "left-right",
        "subjects": subjects,
        "alignment": align_to,
        "source_method": source_method,
        "parcellation": parc,
        "n_subjects": len(subjects),
        "n_permutations": int(n_permutations),
        "valid_feature_count": int(np.sum(valid_features)),
        "total_feature_count": int(valid_features.size),
        "tail": int(tail),
        "alpha": float(alpha),
        "condition_inputs": inputs,
    }
    save_array(
        paths["tstat"],
        t_obs,
        dims=feature_dims,
        coords=coords,
        metadata={**metadata, "statistic": "t"},
    )
    save_array(
        paths["pvalue"],
        p_values,
        dims=feature_dims,
        coords=coords,
        metadata={**metadata, "statistic": "p"},
    )
    save_array(
        paths["lateralization"],
        lateralized,
        dims=("subject", *feature_dims),
        coords={"subject": subjects, **coords},
        metadata={**metadata, "statistic": "subject_lateralization"},
    )
    save_array(
        paths["h0"],
        h0,
        dims=("permutation",),
        metadata={**metadata, "statistic": "max_statistic_null"},
    )
    save_table(
        paths["windows"],
        _window_table(p_values, feature_dims, coords, alpha=alpha),
        metadata={**metadata, "kind": "significant_time_windows"},
    )
    return paths


def run_lateralized_statistics(
    project: ProjectConfig,
    *,
    settings: LateralizedStatisticsConfig,
    subjects: Optional[Sequence[str]] = None,
    feature_root: Optional[str | Path] = None,
    output_root: Optional[str | Path] = None,
) -> WorkflowResult:
    """Run the configured hemisphere contrast and declare its derivatives."""
    feature_root = Path(feature_root or project.bids_root)
    outputs = run_lateralized_statistics_test(
        str(feature_root),
        str(output_root or project.bids_root),
        settings.condition,
        subjects_list=subjects,
        align_to=settings.alignment,
        source_method=settings.source_method,
        parc=settings.parc,
        runs=settings.runs,
        n_permutations=settings.permutations,
        alpha=settings.alpha,
        tail=settings.tail,
        n_jobs=settings.n_jobs,
    )
    selected_subjects = (
        [normalize_subject_id(subject) for subject in subjects]
        if subjects
        else discover_subjects(
            str(feature_root),
            (settings.condition,),
            align_to=settings.alignment,
            source_method=settings.source_method,
            parc=settings.parc,
        )
    )
    inputs = tuple(
        path
        for subject in selected_subjects
        for path in find_erp_arrays(
            str(feature_root),
            subject,
            settings.condition,
            align_to=settings.alignment,
            source_method=settings.source_method,
            parc=settings.parc,
            runs=settings.runs,
        )
    )
    return WorkflowResult(
        stage="lateralized_statistics",
        inputs=inputs,
        outputs=tuple(outputs.values()),
        settings={"subjects": selected_subjects, **settings.__dict__},
    )


def run_group_statistics(
    project: ProjectConfig,
    *,
    settings: StatisticsConfig,
    subjects: Optional[Sequence[str]] = None,
    feature_root: Optional[str | Path] = None,
    output_root: Optional[str | Path] = None,
) -> WorkflowResult:
    """Run one paired group contrast and declare all consumed ERP arrays."""
    feature_root = Path(feature_root or project.bids_root)
    runs_by_condition = {
        settings.conditions[0]: settings.runs_condition_1,
        settings.conditions[1]: settings.runs_condition_2,
    }
    runs_by_condition = {
        condition: runs
        for condition, runs in runs_by_condition.items()
        if runs
    }
    outputs = run_group_statistics_contrast(
        str(feature_root),
        str(output_root or project.bids_root),
        conditions=settings.conditions,
        n_permutations=settings.permutations,
        subjects_list=subjects,
        align_to=settings.alignment,
        source_method=settings.source_method,
        parc=settings.parc,
        runs_by_condition=runs_by_condition,
        alpha=settings.alpha,
        tail=settings.tail,
        n_jobs=settings.n_jobs,
    )
    selected_subjects = (
        [normalize_subject_id(subject) for subject in subjects]
        if subjects
        else discover_subjects(
            str(feature_root),
            settings.conditions,
            align_to=settings.alignment,
            source_method=settings.source_method,
            parc=settings.parc,
        )
    )
    inputs = tuple(
        path
        for subject in selected_subjects
        for condition in settings.conditions
        for path in DerivativeLayout(feature_root).find_erp(
            subject=subject,
            condition=condition,
            alignment=settings.alignment,
            source_method=settings.source_method,
            parc=settings.parc,
            runs=runs_by_condition.get(condition),
        )
    )
    return WorkflowResult(
        stage="group_statistics",
        inputs=inputs,
        outputs=tuple(outputs.values()),
        settings={
            "subjects": selected_subjects,
            **settings.__dict__,
        },
    )
