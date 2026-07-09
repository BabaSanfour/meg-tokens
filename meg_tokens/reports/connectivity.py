"""Group-level circular plots for connectivity derivatives."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from mne.stats import permutation_t_test

from meg_tokens.io import derivative_path, ensure_dir, load_array, require_file, save_array, save_sidecar


def _get_plot_connectivity_circle():
    try:
        from mne_connectivity.viz import plot_connectivity_circle
    except ImportError as exc:
        raise ImportError(
            "mne-connectivity is required to plot connectivity circles. "
            "Install the project dependencies before running this plotting command."
        ) from exc
    return plot_connectivity_circle


def _stage_metadata(metadata: dict) -> dict:
    stage_meta = metadata.get("metadata", {})
    return stage_meta if isinstance(stage_meta, dict) else {}


def _load_modern_connectivity_file(path: Path, condition: str, band: str):
    loaded = load_array(path, require_sidecar=True)
    dims = tuple(loaded.metadata.get("dims", []))
    coords = loaded.metadata.get("coords", {})
    stage_meta = _stage_metadata(loaded.metadata)
    if dims != ("window", "band", "node_from", "node_to"):
        return None
    if str(stage_meta.get("condition", "")).lower() != condition.lower():
        return None
    bands = [str(item) for item in coords.get("band", [])]
    windows = [str(item) for item in coords.get("window", [])]
    if band not in bands or "before" not in windows or "after" not in windows:
        return None
    band_idx = bands.index(band)
    before_idx = windows.index("before")
    after_idx = windows.index("after")
    node_names = [str(item) for item in coords.get("node_from", [])]
    if not node_names:
        node_names = [f"node_{idx:03d}" for idx in range(loaded.data.shape[-1])]
    return {
        "subject": str(stage_meta.get("subject", path.name.split("_", 1)[0].replace("sub-", ""))),
        "before": np.asarray(loaded.data[before_idx, band_idx], dtype=float),
        "after": np.asarray(loaded.data[after_idx, band_idx], dtype=float),
        "node_names": node_names,
        "path": str(path),
        "metadata": stage_meta,
    }


def load_connectivity_pairs_with_metadata(
    data_dir: str,
    condition: str,
    band: str,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    """Load before/after connectivity matrices and average runs by subject."""
    modern_rows = []
    for path in sorted(Path(data_dir).glob("**/*_connectivity.npy")):
        row = _load_modern_connectivity_file(path, condition, band)
        if row is not None:
            modern_rows.append(row)

    if not modern_rows:
        raise FileNotFoundError(
            f"No staged connectivity derivatives found for "
            f"condition={condition}, band={band} under {data_dir}"
        )

    by_subject: dict[str, list[dict]] = {}
    node_names = modern_rows[0]["node_names"]
    for row in modern_rows:
        if row["node_names"] != node_names:
            raise ValueError(f"Node names changed across connectivity derivatives at {row['path']}")
        by_subject.setdefault(row["subject"], []).append(row)

    subjects = sorted(by_subject)
    before = []
    after = []
    input_paths = []
    for subject in subjects:
        rows = by_subject[subject]
        before.append(np.nanmean(np.stack([row["before"] for row in rows], axis=0), axis=0))
        after.append(np.nanmean(np.stack([row["after"] for row in rows], axis=0), axis=0))
        input_paths.extend(row["path"] for row in rows)
    return np.stack(before, axis=0), np.stack(after, axis=0), node_names, subjects, input_paths


def load_connectivity_pairs(data_dir: str, condition: str, band: str) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible loader returning only before/after arrays."""
    before, after, _, _, _ = load_connectivity_pairs_with_metadata(data_dir, condition, band)
    return before, after


def _connectivity_stat_path(output_root: str, condition: str, band: str, suffix: str, extension: str) -> Path:
    return derivative_path(
        output_root,
        subject="group",
        datatype="meg",
        task="tokens",
        description=f"{condition.lower()}-{band}-connectivity",
        suffix=suffix,
        extension=extension,
    )


def run_batch_plot_connectivity_circle(
    data_dir: str,
    output_dir: str,
    condition: str,
    band: str = "alpha",
    p_threshold: float = 0.05,
    n_permutations: int = 1000,
    node_names_path: str = None,
) -> dict[str, Path]:
    """Run active-baseline edge tests and plot a circular chord diagram."""
    print("=== Starting Connectivity Circle Plotting ===")
    output_path = ensure_dir(output_dir)

    con_before_group, con_after_group, node_names, subjects, input_paths = load_connectivity_pairs_with_metadata(
        data_dir, condition, band
    )
    n_subjects, n_rois, _ = con_before_group.shape
    print(f"Loaded connectivity matrices for {n_subjects} subjects and {n_rois} nodes.")

    if node_names_path:
        node_names = [line.strip() for line in require_file(node_names_path, purpose="ROI node names").read_text().splitlines() if line.strip()]
        if len(node_names) != n_rois:
            raise ValueError(f"Node-name file has {len(node_names)} rows but matrix has {n_rois} nodes")

    diff_group = con_after_group - con_before_group
    diff_flat = diff_group.reshape(n_subjects, -1)
    print(f"Running permutation t-test (permutations={n_permutations})...")
    t_vals, p_vals, h0 = permutation_t_test(diff_flat, n_permutations=n_permutations, n_jobs=1)
    t_vals = t_vals.reshape(n_rois, n_rois)
    p_vals = p_vals.reshape(n_rois, n_rois)

    adjacency = np.nanmean(diff_group, axis=0)
    adjacency[p_vals > p_threshold] = 0.0
    np.fill_diagonal(adjacency, 0.0)
    adjacency = np.maximum(adjacency, adjacency.T)
    n_significant_edges = int(np.count_nonzero(adjacency) // 2)
    print(f"Found {n_significant_edges} significant edges at p < {p_threshold}")

    coords = {"node_from": node_names, "node_to": node_names}
    metadata = {
        "stage": "connectivity_plot",
        "kind": "active_minus_baseline_connectivity_stats",
        "condition": condition,
        "band": band,
        "subjects": subjects,
        "input_paths": input_paths,
        "p_threshold": float(p_threshold),
        "n_permutations": int(n_permutations),
    }
    outputs = {
        "adjacency": save_array(
            _connectivity_stat_path(output_dir, condition, band, "connectivityadjacency", ".npy"),
            adjacency,
            dims=("node_from", "node_to"),
            coords=coords,
            metadata={**metadata, "kind": "thresholded_adjacency"},
        ),
        "tstat": save_array(
            _connectivity_stat_path(output_dir, condition, band, "connectivitytstat", ".npy"),
            t_vals,
            dims=("node_from", "node_to"),
            coords=coords,
            metadata={**metadata, "kind": "t_statistic"},
        ),
        "pvalue": save_array(
            _connectivity_stat_path(output_dir, condition, band, "connectivitypvalue", ".npy"),
            p_vals,
            dims=("node_from", "node_to"),
            coords=coords,
            metadata={**metadata, "kind": "p_value"},
        ),
        "h0": save_array(
            _connectivity_stat_path(output_dir, condition, band, "connectivityh0", ".npy"),
            np.asarray(h0, dtype=float),
            dims=("permutation",),
            metadata={**metadata, "kind": "permutation_null_distribution"},
        ),
    }

    plot_connectivity_circle = _get_plot_connectivity_circle()
    fig, _ = plot_connectivity_circle(
        adjacency,
        node_names,
        n_lines=300,
        node_angles=None,
        node_colors=None,
        title=f"{condition} ({band}): active vs baseline",
        fontsize_names=6,
        show=False,
    )
    fig_path = output_path / f"circle_{condition}_{band}.png"
    fig.savefig(fig_path, facecolor="black", dpi=300)
    plt.close(fig)
    save_sidecar(fig_path, {**metadata, "kind": "connectivity_circle_figure", "adjacency": str(outputs["adjacency"])})
    outputs["figure"] = fig_path
    print(f"Saved circular connectivity plot: {fig_path}")
    return outputs
