"""Seed-based summaries from Stage 10 connectivity derivatives."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from mne.stats import permutation_t_test

from meg_tokens.io import derivative_path, ensure_dir, require_file, save_array
from meg_tokens.utils.batch_plot_connectivity_circle import load_connectivity_pairs_with_metadata


def _seed_output_path(output_root: str, condition: str, band: str, seed_roi: str, suffix: str) -> Path:
    return derivative_path(
        output_root,
        subject="group",
        datatype="meg",
        task="tokens",
        description=f"{condition.lower()}-{band}-seed-{seed_roi}-connectivity",
        suffix=suffix,
        extension=".npy",
    )


def _resolve_seed(seed_roi: str, node_names: Sequence[str], node_names_path: Optional[str]) -> tuple[int, list[str]]:
    names = list(node_names)
    if node_names_path:
        names = [line.strip() for line in require_file(node_names_path, purpose="ROI node names").read_text().splitlines() if line.strip()]
        if len(names) != len(node_names):
            raise ValueError(f"Node-name file has {len(names)} rows but matrix has {len(node_names)} nodes")

    if seed_roi in names:
        return names.index(seed_roi), names
    try:
        seed_idx = int(seed_roi)
    except ValueError as exc:
        raise ValueError("Seed must be a node name present in the derivative sidecar, or an integer index") from exc
    if seed_idx < 0 or seed_idx >= len(names):
        raise ValueError(f"Seed index {seed_idx} is outside matrix range 0..{len(names) - 1}")
    return seed_idx, names


def run_batch_plot_seed_connectivity(
    data_dir: str,
    output_dir: str,
    condition: str,
    seed_roi: str,
    band: str = "alpha",
    p_threshold: float = 0.05,
    n_permutations: int = 1000,
    node_names_path: str = None,
) -> dict[str, Path]:
    """Extract and threshold a seed-to-all active-baseline connectivity vector."""
    print("=== Starting Seed-Based Connectivity Extraction ===")
    ensure_dir(output_dir)
    con_before_group, con_after_group, node_names, subjects, input_paths = load_connectivity_pairs_with_metadata(
        data_dir, condition, band
    )
    _, n_rois, _ = con_before_group.shape
    seed_idx, node_names = _resolve_seed(seed_roi, node_names, node_names_path)

    print(f"Extracting Seed -> All vector for seed index {seed_idx}")
    before = con_before_group[:, seed_idx, :]
    after = con_after_group[:, seed_idx, :]
    diff = after - before

    print(f"Running permutation t-test (permutations={n_permutations})...")
    t_vals, p_vals, h0 = permutation_t_test(diff, n_permutations=n_permutations, n_jobs=1)
    spatial_map = np.nanmean(diff, axis=0)
    spatial_map[p_vals > p_threshold] = 0.0
    spatial_map[seed_idx] = 1.0
    n_significant = int(np.count_nonzero(spatial_map) - 1)
    print(f"Found {n_significant} significantly connected nodes at p < {p_threshold}")

    coords = {"node": node_names}
    metadata = {
        "stage": "seed_connectivity",
        "condition": condition,
        "band": band,
        "seed_roi": seed_roi,
        "seed_index": int(seed_idx),
        "subjects": subjects,
        "input_paths": input_paths,
        "p_threshold": float(p_threshold),
        "n_permutations": int(n_permutations),
    }
    outputs = {
        "seed_map": save_array(
            _seed_output_path(output_dir, condition, band, seed_roi, "seedconnectivity"),
            spatial_map,
            dims=("node",),
            coords=coords,
            metadata={**metadata, "kind": "thresholded_seed_to_all_map"},
        ),
        "tstat": save_array(
            _seed_output_path(output_dir, condition, band, seed_roi, "seedconnectivitytstat"),
            np.asarray(t_vals, dtype=float),
            dims=("node",),
            coords=coords,
            metadata={**metadata, "kind": "seed_to_all_t_statistic"},
        ),
        "pvalue": save_array(
            _seed_output_path(output_dir, condition, band, seed_roi, "seedconnectivitypvalue"),
            np.asarray(p_vals, dtype=float),
            dims=("node",),
            coords=coords,
            metadata={**metadata, "kind": "seed_to_all_p_value"},
        ),
        "h0": save_array(
            _seed_output_path(output_dir, condition, band, seed_roi, "seedconnectivityh0"),
            np.asarray(h0, dtype=float),
            dims=("permutation",),
            metadata={**metadata, "kind": "seed_to_all_permutation_null_distribution"},
        ),
    }

    print(f"Saved thresholded seed map: {outputs['seed_map']}")
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract and threshold seed-based connectivity maps.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing Stage 10 connectivity derivatives.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory to save seed vectors.")
    parser.add_argument("--condition", type=str, required=True)
    parser.add_argument("--seed_roi", type=str, required=True,
                        help="Seed node name or integer index.")
    parser.add_argument("--band", type=str, default="alpha")
    parser.add_argument("--p_threshold", type=float, default=0.05)
    parser.add_argument("--perms", type=int, default=1000)
    parser.add_argument("--node_names", type=str, default=None,
                        help="Optional text file with one node name per line.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_batch_plot_seed_connectivity(
        args.data_dir,
        args.out_dir,
        args.condition,
        args.seed_roi,
        band=args.band,
        p_threshold=args.p_threshold,
        n_permutations=args.perms,
        node_names_path=args.node_names,
    )


if __name__ == "__main__":
    main()
