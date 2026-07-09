"""Sensor-space decoding workflow over staged trial-by-sensor arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from meg_tokens.analysis.decoding import (
    compute_spatial_decoding,
    compute_spatial_decoding_permutations,
)
from meg_tokens.core import ProjectConfig, SpatialDecodingConfig, WorkflowResult
from meg_tokens.io import derivative_path, load_array, save_array


def load_spatial_decoding_inputs(
    data_dir: str | Path,
    conditions: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], list[str], tuple[Path, ...]]:
    """Load labeled trial-by-sensor arrays and encode requested classes."""
    base = Path(data_dir)
    x_path = base / "X.npy"
    y_path = base / "y.npy"
    loaded_x = load_array(x_path, expected_ndim=2, require_sidecar=True)
    loaded_y = load_array(
        y_path,
        expected_ndim=1,
        allow_pickle=True,
        require_sidecar=True,
    )
    X = np.asarray(loaded_x.data, dtype=float)
    y = np.asarray(loaded_y.data)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y disagree on trials: {X.shape[0]} != {y.shape[0]}")
    dims = tuple(loaded_x.metadata.get("dims", ()))
    if dims not in {("trial", "sensor"), ("trial", "channel")}:
        raise ValueError(f"X.npy must have trial-by-sensor dimensions, got {dims}")
    coords = loaded_x.metadata.get("coords", {})
    channels = list(coords.get("channel", coords.get("sensor", ())))
    if len(channels) != X.shape[1]:
        raise ValueError("X.npy must provide one channel coordinate per sensor")

    inputs = [x_path, y_path]
    groups_path = base / "groups.npy"
    groups = None
    if groups_path.exists():
        groups = np.asarray(
            load_array(
                groups_path,
                expected_ndim=1,
                allow_pickle=True,
                require_sidecar=True,
            ).data
        )
        if groups.shape[0] != X.shape[0]:
            raise ValueError(
                f"X and groups disagree on trials: {X.shape[0]} != {groups.shape[0]}"
            )
        inputs.append(groups_path)

    if y.dtype.kind in {"U", "S", "O"}:
        conditions = [str(condition) for condition in conditions]
        labels = y.astype(str)
        keep = np.isin(labels, conditions)
        missing = sorted(set(conditions) - set(labels[keep]))
        if missing:
            raise ValueError(f"Requested conditions absent from y.npy: {missing}")
        X = X[keep]
        labels = labels[keep]
        if groups is not None:
            groups = groups[keep]
        mapping = {label: index for index, label in enumerate(conditions)}
        y = np.asarray([mapping[label] for label in labels], dtype=int)

    return X, y, groups, [str(channel) for channel in channels], tuple(inputs)


def spatial_decoding_path(
    output_root: str | Path,
    conditions: Sequence[str],
    *,
    suffix: str,
) -> Path:
    description = "-".join(
        [
            str(conditions[0]).lower(),
            "vs",
            *[str(item).lower() for item in conditions[1:]],
            "sensor",
        ]
    )
    return derivative_path(
        output_root,
        subject="group",
        datatype="meg",
        task="tokens",
        description=description,
        suffix=suffix,
        extension=".npy",
    )


def run_spatial_decoding_arrays(
    data_dir: str | Path,
    output_root: str | Path,
    conditions: Sequence[str],
    *,
    permutations: int = 0,
    n_jobs: int = 4,
) -> dict[str, Path]:
    X, y, groups, channels, inputs = load_spatial_decoding_inputs(
        data_dir, conditions
    )
    metadata = {
        "stage": "spatial_decoding",
        "conditions": list(conditions),
        "classifier": "LinearDiscriminantAnalysis",
        "inputs": [str(path) for path in inputs],
    }
    outputs = {
        "scores": spatial_decoding_path(
            output_root, conditions, suffix="spatialdecoding"
        )
    }
    if permutations:
        scores, permutation_scores, threshold = (
            compute_spatial_decoding_permutations(
                X=X,
                y=y,
                groups=groups,
                balance=True,
                n_permutations=permutations,
                n_jobs=n_jobs,
            )
        )
        outputs["permutations"] = spatial_decoding_path(
            output_root, conditions, suffix="spatialdecodingpermutations"
        )
        outputs["threshold"] = spatial_decoding_path(
            output_root, conditions, suffix="spatialdecodingthreshold"
        )
        save_array(
            outputs["permutations"],
            permutation_scores,
            dims=("permutation", "sensor"),
            coords={"sensor": channels},
            metadata={**metadata, "statistic": "permutation_accuracy"},
        )
        save_array(
            outputs["threshold"],
            np.asarray([threshold], dtype=float),
            dims=("threshold",),
            metadata={**metadata, "statistic": "fwe_accuracy_threshold_95"},
        )
    else:
        scores = compute_spatial_decoding(
            X=X,
            y=y,
            groups=groups,
            balance=True,
            n_jobs=n_jobs,
        )
    save_array(
        outputs["scores"],
        scores,
        dims=("sensor",),
        coords={"sensor": channels},
        metadata={**metadata, "statistic": "accuracy"},
    )
    return outputs


def run_spatial_decoding(
    project: ProjectConfig,
    *,
    settings: SpatialDecodingConfig,
    data_dir: str | Path,
    output_root: str | Path | None = None,
) -> WorkflowResult:
    outputs = run_spatial_decoding_arrays(
        data_dir,
        output_root or project.bids_root,
        settings.conditions,
        permutations=settings.permutations,
        n_jobs=settings.n_jobs,
    )
    inputs = [Path(data_dir) / "X.npy", Path(data_dir) / "y.npy"]
    groups = Path(data_dir) / "groups.npy"
    if groups.exists():
        inputs.append(groups)
    return WorkflowResult(
        stage="spatial_decoding",
        inputs=tuple(inputs),
        outputs=tuple(outputs.values()),
        settings=settings.__dict__,
    )
