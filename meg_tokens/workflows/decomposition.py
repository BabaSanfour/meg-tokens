"""
Stage 9 PCA/dPCA trajectory analysis over staged MEG derivatives.

The default PCA path replicates the legacy nmData trajectory behavior at the
analysis level: condition means are built from real staged ERP/power arrays,
PCA is fit over condition-by-time samples, and every condition trajectory is
projected onto the shared loading axes.
"""

from __future__ import annotations

import itertools
import string
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from meg_tokens.analysis.decomposition import apply_neural_transform, fit_condition_pca, nanmean_no_warning
from meg_tokens.core import (
    DecompositionConfig,
    ProjectConfig,
    WorkflowResult,
    normalize_subject_id,
)
from meg_tokens.io import DerivativeLayout, save_array, save_table
from meg_tokens.features.dataset import (
    load_feature_array as _load_feature_array,
    trial_table_for_array as _trial_table_for_array,
)
from meg_tokens.workflows.decoding import (
    discover_subjects,
    find_feature_arrays,
)


def _condition_desc(conditions: Sequence[str]) -> str:
    if len(conditions) == 1:
        return conditions[0].lower()
    return "-".join([conditions[0].lower(), "vs", *[condition.lower() for condition in conditions[1:]]])


def _selection_desc(labels: Optional[Sequence[str]], lateralize: bool) -> Optional[str]:
    if lateralize:
        return "lateralized"
    if not labels:
        return None
    if len(labels) == 1:
        return str(labels[0])
    return f"{len(labels)}labels"


def pca_derivative_path(
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
    labels: Optional[Sequence[str]] = None,
    lateralize: bool = False,
) -> Path:
    """Build a group-level PCA derivative path."""
    return DerivativeLayout(output_root).pca(
        conditions=conditions,
        feature_source=feature_source,
        alignment=align_to,
        source_method=source_method,
        parc=parc,
        band=band,
        suffix=suffix,
        extension=extension,
        labels=labels,
        lateralized=lateralize,
    )


def _validate_reference(
    *,
    path: Path,
    time_values: np.ndarray,
    feature_names: Sequence[str],
    reference_time: Optional[np.ndarray],
    reference_features: Optional[Sequence[str]],
) -> tuple[np.ndarray, list[str]]:
    if reference_time is None:
        return np.asarray(time_values, dtype=float), list(feature_names)
    if not np.allclose(reference_time, time_values, equal_nan=True):
        raise ValueError(f"Time coordinates changed across feature arrays at {path}")
    if list(reference_features or []) != list(feature_names):
        raise ValueError(f"Feature coordinates changed across feature arrays at {path}")
    return reference_time, list(reference_features or feature_names)


def _metadata_source_vertices(metadata: Mapping[str, object]) -> Optional[list]:
    stage_meta = metadata.get("metadata", {})
    if isinstance(stage_meta, Mapping):
        vertices = stage_meta.get("source_vertices")
        if vertices is not None:
            return vertices
    return None


def build_pca_trajectory_dataset(
    feature_dir: str,
    *,
    conditions: Sequence[str],
    feature_source: str = "erp",
    subjects: Optional[Sequence[str]] = None,
    align_to: str = "go",
    source_method: str = "dSPM",
    parc: Optional[str] = "HCPMMP1",
    band: Optional[str] = None,
    power_method: str = "hilbert",
    runs_by_condition: Optional[Mapping[str, Sequence[str]]] = None,
    labels: Optional[Sequence[str]] = None,
    lateralize: bool = False,
    average_unit: str = "subject",
    transform: Optional[str] = None,
) -> dict[str, object]:
    """Load staged derivatives and build condition mean matrices."""
    conditions = [str(condition) for condition in conditions]
    if len(conditions) < 1:
        raise ValueError("At least one condition is required")
    if average_unit not in {"subject", "trial"}:
        raise ValueError("average_unit must be 'subject' or 'trial'")
    runs_by_condition = runs_by_condition or {}

    if subjects is None:
        subjects = discover_subjects(
            feature_dir,
            conditions=conditions,
            feature_source=feature_source,
            align_to=align_to,
            source_method=source_method,
            parc=parc,
            band=band,
            power_method=power_method,
        )

    condition_observations: dict[str, list[np.ndarray]] = {condition: [] for condition in conditions}
    observation_rows = []
    input_paths = []
    reference_time: Optional[np.ndarray] = None
    reference_features: Optional[list[str]] = None
    source_vertices = None

    for subject in subjects:
        subject = str(subject)
        for condition in conditions:
            paths = find_feature_arrays(
                feature_dir,
                subject,
                condition,
                feature_source=feature_source,
                align_to=align_to,
                source_method=source_method,
                parc=parc,
                band=band,
                power_method=power_method,
                runs=runs_by_condition.get(condition),
            )
            subject_trials = []
            for path in paths:
                X, time_values, feature_names, metadata = _load_feature_array(
                    path,
                    labels=labels,
                    lateralize=lateralize,
                )
                X = apply_neural_transform(X, transform)
                reference_time, reference_features = _validate_reference(
                    path=path,
                    time_values=time_values,
                    feature_names=feature_names,
                    reference_time=reference_time,
                    reference_features=reference_features,
                )
                if source_vertices is None:
                    source_vertices = _metadata_source_vertices(metadata)
                subject_trials.append(X)
                input_paths.append(str(path))

            if not subject_trials:
                continue
            stacked_trials = np.concatenate(subject_trials, axis=0)
            if average_unit == "subject":
                condition_observations[condition].append(nanmean_no_warning(stacked_trials, axis=0))
                observation_rows.append({
                    "condition": condition,
                    "subject": subject,
                    "average_unit": "subject",
                    "n_trials": int(stacked_trials.shape[0]),
                    "input_paths": ";".join(str(path) for path in paths),
                })
            else:
                for trial_idx, trial in enumerate(stacked_trials):
                    condition_observations[condition].append(trial)
                    observation_rows.append({
                        "condition": condition,
                        "subject": subject,
                        "average_unit": "trial",
                        "n_trials": 1,
                        "trial_index_within_subject_condition": trial_idx + 1,
                        "input_paths": ";".join(str(path) for path in paths),
                    })

    missing = [condition for condition, observations in condition_observations.items() if not observations]
    if missing:
        raise ValueError(f"No observations were available for conditions: {missing}")
    if reference_time is None or reference_features is None:
        raise ValueError("No feature arrays were loaded")

    condition_means = np.stack([
        nanmean_no_warning(np.stack(condition_observations[condition], axis=0), axis=0)
        for condition in conditions
    ], axis=0)

    return {
        "condition_means": condition_means,
        "conditions": conditions,
        "time_sec": reference_time,
        "feature_names": reference_features,
        "subjects": list(subjects),
        "observation_table": pd.DataFrame(observation_rows),
        "input_paths": input_paths,
        "source_vertices": source_vertices,
        "average_unit": average_unit,
        "transform": transform or "none",
    }
def run_pca_trajectory(
    feature_dir: str,
    output_dir: str,
    conditions: Sequence[str],
    *,
    feature_source: str = "erp",
    subjects: Optional[Sequence[str]] = None,
    align_to: str = "go",
    source_method: str = "dSPM",
    parc: Optional[str] = "HCPMMP1",
    band: Optional[str] = None,
    power_method: str = "hilbert",
    runs_by_condition: Optional[Mapping[str, Sequence[str]]] = None,
    labels: Optional[Sequence[str]] = None,
    lateralize: bool = False,
    average_unit: str = "subject",
    transform: Optional[str] = None,
    n_components: int = 20,
    min_variance: Optional[float] = None,
    fit_time_range: Optional[tuple[float, float]] = None,
    project_centered: bool = False,
) -> dict[str, Path]:
    """Run PCA trajectory analysis and save BIDS-style derivatives."""
    dataset = build_pca_trajectory_dataset(
        feature_dir,
        conditions=conditions,
        feature_source=feature_source,
        subjects=subjects,
        align_to=align_to,
        source_method=source_method,
        parc=parc,
        band=band,
        power_method=power_method,
        runs_by_condition=runs_by_condition,
        labels=labels,
        lateralize=lateralize,
        average_unit=average_unit,
        transform=transform,
    )
    result = fit_condition_pca(
        dataset["condition_means"],
        time=dataset["time_sec"],
        n_components=n_components,
        min_variance=min_variance,
        fit_time_range=fit_time_range,
        project_centered=project_centered,
    )

    conditions = list(dataset["conditions"])
    feature_names = list(dataset["feature_names"])
    components = list(range(1, result.loadings.shape[1] + 1))
    time_sec = np.asarray(dataset["time_sec"], dtype=float)
    base_kwargs = {
        "conditions": conditions,
        "feature_source": feature_source,
        "align_to": align_to,
        "source_method": source_method,
        "parc": parc,
        "band": band,
        "labels": labels,
        "lateralize": lateralize,
    }

    common_metadata = {
        "stage": "pca_trajectory",
        "conditions": conditions,
        "feature_source": feature_source,
        "alignment": align_to,
        "source_method": source_method,
        "parcellation": parc,
        "band": band,
        "power_method": power_method if feature_source == "power" else None,
        "labels": list(labels) if labels else None,
        "lateralized": bool(lateralize),
        "average_unit": dataset["average_unit"],
        "transform": dataset["transform"],
        "subjects": list(dataset["subjects"]),
        "input_paths": list(dataset["input_paths"]),
        "fit_time_range_sec": list(fit_time_range) if fit_time_range else None,
        "project_centered": bool(project_centered),
        "legacy_reference": {
            "matlab_functions": ["nmCalcPCANoDB", "nmGetPCsNoDB", "nmPlotPCSpaceNoDB"],
            "fit_matrix": "condition x time observations by feature",
            "projection": "raw condition means projected onto centered PCA loading axes",
        },
        "source_vertices": dataset["source_vertices"],
    }

    outputs: dict[str, Path] = {}
    outputs["condition_means"] = save_array(
        pca_derivative_path(output_dir, suffix="pcacondmeans", extension=".npy", **base_kwargs),
        result.condition_means,
        dims=("condition", "feature", "time"),
        coords={"condition": conditions, "feature": feature_names, "time_sec": time_sec},
        metadata={**common_metadata, "kind": "condition_feature_time_means"},
    )
    outputs["trajectory"] = save_array(
        pca_derivative_path(output_dir, suffix="pcatrajectory", extension=".npy", **base_kwargs),
        result.trajectory,
        dims=("condition", "component", "time"),
        coords={"condition": conditions, "component": components, "time_sec": time_sec},
        metadata={**common_metadata, "kind": "condition_component_trajectory"},
    )
    outputs["loadings"] = save_array(
        pca_derivative_path(output_dir, suffix="pcaloadings", extension=".npy", **base_kwargs),
        result.loadings,
        dims=("feature", "component"),
        coords={"feature": feature_names, "component": components},
        metadata={
            **common_metadata,
            "kind": "feature_component_loadings",
            "feature_mask": result.feature_mask.tolist(),
            "pca_mean": result.pca_mean.tolist(),
        },
    )
    outputs["variance"] = save_array(
        pca_derivative_path(output_dir, suffix="pcavariance", extension=".npy", **base_kwargs),
        result.variance_ratio,
        dims=("component",),
        coords={"component": components},
        metadata={**common_metadata, "kind": "explained_variance_ratio"},
    )
    outputs["fit_scores"] = save_array(
        pca_derivative_path(output_dir, suffix="pcafitscores", extension=".npy", **base_kwargs),
        result.fit_scores,
        dims=("sample", "component"),
        coords={"component": components},
        metadata={**common_metadata, "kind": "finite_fit_sample_scores"},
    )

    fit_samples = pd.DataFrame({
        "sample": np.arange(1, result.fit_scores.shape[0] + 1),
        "condition": [conditions[idx] for idx in result.fit_sample_condition],
        "time_index": result.fit_sample_time_index.astype(int),
        "time_sec": time_sec[result.fit_sample_time_index],
    })
    outputs["fit_samples"] = save_table(
        pca_derivative_path(output_dir, suffix="pcafitsamples", extension=".tsv", **base_kwargs),
        fit_samples,
        metadata={**common_metadata, "kind": "finite_fit_sample_metadata"},
    )
    outputs["observations"] = save_table(
        pca_derivative_path(output_dir, suffix="pcaobservations", extension=".tsv", **base_kwargs),
        dataset["observation_table"],
        metadata={**common_metadata, "kind": "condition_mean_observations"},
    )

    for key, path in outputs.items():
        print(f"Saved PCA {key}: {path}")
    return outputs


def _load_dpca_trials(
    feature_dir: str,
    *,
    input_conditions: Sequence[str],
    feature_source: str,
    subjects: Optional[Sequence[str]],
    align_to: str,
    source_method: str,
    parc: Optional[str],
    band: Optional[str],
    power_method: str,
    labels: Optional[Sequence[str]],
    lateralize: bool,
    transform: Optional[str],
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, list[str], list[str]]:
    if feature_source != "erp":
        raise ValueError("dPCA trial metadata loading currently requires ERP derivatives with erptrials TSV sidecars")
    if subjects is None:
        subjects = discover_subjects(
            feature_dir,
            conditions=input_conditions,
            feature_source=feature_source,
            align_to=align_to,
            source_method=source_method,
            parc=parc,
            band=band,
            power_method=power_method,
        )

    X_parts = []
    meta_parts = []
    input_paths = []
    reference_time = None
    reference_features = None

    for subject in subjects:
        for condition in input_conditions:
            for path in find_feature_arrays(
                feature_dir,
                subject,
                condition,
                feature_source=feature_source,
                align_to=align_to,
                source_method=source_method,
                parc=parc,
                band=band,
                power_method=power_method,
            ):
                X, time_values, feature_names, metadata = _load_feature_array(
                    path,
                    labels=labels,
                    lateralize=lateralize,
                )
                X = apply_neural_transform(X, transform)
                reference_time, reference_features = _validate_reference(
                    path=path,
                    time_values=time_values,
                    feature_names=feature_names,
                    reference_time=reference_time,
                    reference_features=reference_features,
                )
                trial_table = _trial_table_for_array(path, metadata)
                if trial_table is None:
                    raise ValueError(f"dPCA requires a trial metadata TSV for {path}")
                table = pd.read_csv(trial_table, sep="\t")
                if len(table) != X.shape[0]:
                    raise ValueError(f"Trial table {trial_table} has {len(table)} rows but array has {X.shape[0]} trials")
                table = table.copy()
                table["subject"] = subject
                table["input_condition"] = condition
                table["input_path"] = str(path)
                X_parts.append(X)
                meta_parts.append(table)
                input_paths.append(str(path))

    if not X_parts or reference_time is None or reference_features is None:
        raise ValueError("No ERP trials were loaded for dPCA")
    return (
        np.concatenate(X_parts, axis=0),
        pd.concat(meta_parts, axis=0).reset_index(drop=True),
        reference_time,
        reference_features,
        input_paths,
    )


def build_dpca_tensor(
    X: np.ndarray,
    metadata: pd.DataFrame,
    marginalize_cols: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, list], pd.DataFrame]:
    """Build trial and mean tensors for demixed PCA."""
    if not marginalize_cols:
        raise ValueError("At least one marginalization column is required for dPCA")
    missing = [column for column in marginalize_cols if column not in metadata.columns]
    if missing:
        raise ValueError(f"dPCA metadata is missing marginalization columns: {missing}")

    unique_values = {
        column: sorted(metadata[column].dropna().astype(str).unique().tolist())
        for column in marginalize_cols
    }
    empty = [column for column, values in unique_values.items() if not values]
    if empty:
        raise ValueError(f"dPCA marginalization columns have no non-null values: {empty}")

    dims = [len(unique_values[column]) for column in marginalize_cols]
    counts = metadata.groupby(list(marginalize_cols)).size()
    max_trials = int(counts.max())
    trial_tensor = np.full((max_trials, X.shape[1], *dims, X.shape[2]), np.nan, dtype=float)
    count_rows = []

    string_meta = metadata.copy()
    for column in marginalize_cols:
        string_meta[column] = string_meta[column].astype(str)

    for idxs in itertools.product(*[range(dim) for dim in dims]):
        values = {column: unique_values[column][idx] for column, idx in zip(marginalize_cols, idxs)}
        mask = np.ones(len(string_meta), dtype=bool)
        for column, value in values.items():
            mask &= string_meta[column].to_numpy() == value
        trial_indices = np.where(mask)[0]
        if len(trial_indices) == 0:
            raise ValueError(f"No dPCA trials for marginalization cell: {values}")
        target = (slice(0, len(trial_indices)), slice(None), *idxs, slice(None))
        trial_tensor[target] = X[trial_indices]
        count_rows.append({**values, "n_trials": int(len(trial_indices))})

    mean_tensor = nanmean_no_warning(trial_tensor, axis=0)
    return trial_tensor, mean_tensor, unique_values, pd.DataFrame(count_rows)


def run_demixed_pca(
    feature_dir: str,
    output_dir: str,
    *,
    input_conditions: Sequence[str],
    marginalize_cols: Sequence[str],
    feature_source: str = "erp",
    subjects: Optional[Sequence[str]] = None,
    align_to: str = "go",
    source_method: str = "dSPM",
    parc: Optional[str] = "HCPMMP1",
    band: Optional[str] = None,
    power_method: str = "hilbert",
    labels: Optional[Sequence[str]] = None,
    lateralize: bool = False,
    transform: Optional[str] = None,
    dpca_labels: Optional[str] = None,
    n_components: int = 20,
) -> dict[str, Path]:
    """Run optional dPCA from staged trial derivatives."""
    if not marginalize_cols:
        raise ValueError("marginalize_cols is required for dPCA")
    try:
        from dPCA import dPCA  # type: ignore
    except ImportError as exc:
        raise ImportError("dPCA analysis requires the optional Python package 'dPCA'") from exc

    X, metadata, time_sec, feature_names, input_paths = _load_dpca_trials(
        feature_dir,
        input_conditions=input_conditions,
        feature_source=feature_source,
        subjects=subjects,
        align_to=align_to,
        source_method=source_method,
        parc=parc,
        band=band,
        power_method=power_method,
        labels=labels,
        lateralize=lateralize,
        transform=transform,
    )
    trial_tensor, mean_tensor, unique_values, counts = build_dpca_tensor(X, metadata, marginalize_cols)
    if dpca_labels is None:
        dpca_labels = "".join(string.ascii_lowercase[: len(marginalize_cols)]) + "t"

    model = dPCA(labels=dpca_labels, regularizer=0, n_components=n_components)
    model.protect = ["t"]
    components_by_marginalization = model.fit_transform(mean_tensor, trialX=trial_tensor)

    desc_conditions = list(input_conditions)
    base_kwargs = {
        "conditions": desc_conditions,
        "feature_source": feature_source,
        "align_to": align_to,
        "source_method": source_method,
        "parc": parc,
        "band": band,
        "labels": labels,
        "lateralize": lateralize,
    }
    metadata_common = {
        "stage": "demixed_pca",
        "input_conditions": list(input_conditions),
        "marginalize_cols": list(marginalize_cols),
        "marginalize_values": unique_values,
        "dpca_labels": dpca_labels,
        "feature_source": feature_source,
        "alignment": align_to,
        "source_method": source_method,
        "parcellation": parc,
        "labels": list(labels) if labels else None,
        "lateralized": bool(lateralize),
        "transform": transform or "none",
        "input_paths": input_paths,
    }

    outputs: dict[str, Path] = {}
    mean_dims = ("feature", *marginalize_cols, "time")
    mean_coords = {"feature": feature_names, "time_sec": time_sec, **unique_values}
    outputs["mean_tensor"] = save_array(
        pca_derivative_path(output_dir, suffix="dpcamean", extension=".npy", **base_kwargs),
        mean_tensor,
        dims=mean_dims,
        coords=mean_coords,
        metadata={**metadata_common, "kind": "mean_tensor"},
    )
    outputs["trial_counts"] = save_table(
        pca_derivative_path(output_dir, suffix="dpcatrialcounts", extension=".tsv", **base_kwargs),
        counts,
        metadata={**metadata_common, "kind": "marginalization_cell_trial_counts"},
    )

    for key, values in components_by_marginalization.items():
        suffix = f"dpca{key}components"
        array = np.asarray(values)
        dims = ("component", *marginalize_cols, "time") if array.ndim > 2 else ("component", "time")
        coords = {"component": list(range(1, array.shape[0] + 1)), "time_sec": time_sec, **unique_values}
        outputs[f"components_{key}"] = save_array(
            pca_derivative_path(output_dir, suffix=suffix, extension=".npy", **base_kwargs),
            array,
            dims=dims,
            coords=coords,
            metadata={**metadata_common, "kind": "components", "marginalization": key},
        )

    for key, path in outputs.items():
        print(f"Saved dPCA {key}: {path}")
    return outputs


def run_batch_dpca(
    feature_dir: str,
    output_dir: str,
    conditions: Sequence[str] = ("Fast", "Slow"),
    *,
    analysis: str = "pca",
    marginalize_cols: Optional[Sequence[str]] = None,
    input_conditions: Optional[Sequence[str]] = None,
    **kwargs,
) -> dict[str, Path]:
    """Compatibility wrapper for PCA trajectory or optional dPCA analysis."""
    if analysis == "pca":
        return run_pca_trajectory(feature_dir, output_dir, conditions, **kwargs)
    if analysis == "dpca":
        if not marginalize_cols:
            raise ValueError("marginalize_cols is required for dPCA")
        return run_demixed_pca(
            feature_dir,
            output_dir,
            input_conditions=list(input_conditions or conditions),
            marginalize_cols=marginalize_cols,
            **kwargs,
        )
    raise ValueError("analysis must be 'pca' or 'dpca'")


def run_decomposition(
    project: ProjectConfig,
    *,
    settings: DecompositionConfig,
    subjects: Optional[Sequence[str]] = None,
    feature_root: Optional[str | Path] = None,
    output_root: Optional[str | Path] = None,
) -> WorkflowResult:
    """Run PCA trajectories or dPCA with declared staged feature inputs."""
    feature_root = Path(feature_root or project.bids_root)
    common = {
        "feature_source": settings.feature_source,
        "subjects": subjects,
        "align_to": settings.alignment,
        "source_method": settings.source_method,
        "parc": settings.parc,
        "band": settings.band,
        "power_method": settings.power_method,
        "labels": settings.labels,
        "lateralize": settings.lateralize,
        "transform": settings.transform,
        "n_components": settings.n_components,
    }
    if settings.analysis == "pca":
        outputs = run_pca_trajectory(
            str(feature_root),
            str(output_root or project.bids_root),
            settings.conditions,
            average_unit=settings.average_unit,
            min_variance=settings.min_variance,
            fit_time_range=settings.fit_time_range,
            project_centered=settings.project_centered,
            **common,
        )
    else:
        outputs = run_demixed_pca(
            str(feature_root),
            str(output_root or project.bids_root),
            input_conditions=settings.conditions,
            marginalize_cols=settings.marginalize_cols or (),
            dpca_labels=settings.dpca_labels,
            **common,
        )

    selected_subjects = (
        [normalize_subject_id(subject) for subject in subjects]
        if subjects
        else discover_subjects(
            str(feature_root),
            settings.conditions,
            feature_source=settings.feature_source,
            align_to=settings.alignment,
            source_method=settings.source_method,
            parc=settings.parc,
            band=settings.band,
            power_method=settings.power_method,
        )
    )
    inputs = tuple(
        path
        for subject in selected_subjects
        for condition in settings.conditions
        for path in find_feature_arrays(
            str(feature_root),
            subject,
            condition,
            feature_source=settings.feature_source,
            align_to=settings.alignment,
            source_method=settings.source_method,
            parc=settings.parc,
            band=settings.band,
            power_method=settings.power_method,
        )
    )
    return WorkflowResult(
        stage=f"{settings.analysis}_decomposition",
        inputs=inputs,
        outputs=tuple(outputs.values()),
        settings={
            "subjects": selected_subjects,
            **settings.__dict__,
        },
    )
