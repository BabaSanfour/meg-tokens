"""Plot dPCA component derivatives saved by ``batch_dpca --analysis dpca``."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from meg_tokens.io import ensure_dir, load_array


def _component_paths(component_paths: Optional[Sequence[str]], dpca_dir: Optional[str]) -> list[Path]:
    if component_paths:
        return [Path(path) for path in component_paths]
    if dpca_dir is None:
        raise ValueError("Either component_paths or dpca_dir is required")
    paths = sorted(Path(dpca_dir).glob("**/*_dpca*components.npy"))
    if not paths:
        raise FileNotFoundError(f"No dPCA component derivatives found under {dpca_dir}")
    return paths


def run_batch_plot_dpca(
    component_paths: Optional[Sequence[str]] = None,
    output_dir: str = "figures/dpca",
    *,
    dpca_dir: Optional[str] = None,
    n_components_to_plot: int = 3,
) -> list[Path]:
    """Plot one or more sidecar-backed dPCA component arrays."""
    print("=== Starting dPCA Visualization ===")
    output_path = ensure_dir(output_dir)
    sns.set_theme(style="whitegrid", context="paper", palette="deep")

    saved = []
    for component_path in _component_paths(component_paths, dpca_dir):
        loaded = load_array(component_path, require_sidecar=True)
        values = np.asarray(loaded.data, dtype=float)
        dims = list(loaded.metadata.get("dims", []))
        coords = loaded.metadata.get("coords", {})
        stage_meta = loaded.metadata.get("metadata", {})
        if not dims or dims[0] != "component" or dims[-1] != "time":
            raise ValueError(f"Expected component x ... x time dPCA array, got dims={dims} in {component_path}")

        times = np.asarray(coords.get("time_sec", np.arange(values.shape[-1])), dtype=float)
        if times.shape[0] != values.shape[-1]:
            raise ValueError(f"time coordinate length does not match dPCA data in {component_path}")
        time_label = "Time (s)" if "time_sec" in coords else "Time sample"
        marginalization = stage_meta.get("marginalization", component_path.stem)

        condition_shape = values.shape[1:-1]
        condition_dims = dims[1:-1]
        condition_coords = [coords.get(dim, list(range(size))) for dim, size in zip(condition_dims, condition_shape)]

        for comp_idx in range(min(n_components_to_plot, values.shape[0])):
            fig, ax = plt.subplots(figsize=(10, 6))
            if condition_shape:
                for idxs in itertools.product(*[range(size) for size in condition_shape]):
                    selector = (comp_idx, *idxs, slice(None))
                    label_bits = [
                        f"{dim}={condition_coords[pos][idx]}"
                        for pos, (dim, idx) in enumerate(zip(condition_dims, idxs))
                    ]
                    ax.plot(times, values[selector], label=", ".join(label_bits), lw=2, alpha=0.85)
            else:
                ax.plot(times, values[comp_idx], label="Main effect", lw=2, color="black")

            ax.axvline(0, color="black", linestyle="--", alpha=0.5)
            ax.set_title(f"dPCA {marginalization} component {comp_idx + 1}")
            ax.set_xlabel(time_label)
            ax.set_ylabel("Component amplitude")
            if condition_shape and int(np.prod(condition_shape)) <= 12:
                ax.legend(loc="best")
            sns.despine(ax=ax)

            save_path = output_path / f"{component_path.stem}_component-{comp_idx + 1}.png"
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            saved.append(save_path)
            print(f"Saved dPCA component plot: {save_path}")

    return saved


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot dPCA component derivatives.")
    parser.add_argument("--component_paths", type=str, nargs="+", default=None,
                        help="One or more *_dpca*components.npy files written by batch_dpca.")
    parser.add_argument("--dpca_dir", type=str, default=None,
                        help="Directory to search recursively for dPCA component derivatives.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory to save plots.")
    parser.add_argument("--n_components", type=int, default=3,
                        help="Number of components to plot per marginalization.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_batch_plot_dpca(
        component_paths=args.component_paths,
        dpca_dir=args.dpca_dir,
        output_dir=args.out_dir,
        n_components_to_plot=args.n_components,
    )


if __name__ == "__main__":
    main()
