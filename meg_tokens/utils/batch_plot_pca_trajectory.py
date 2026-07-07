import argparse
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import matplotlib.pyplot as plt
from meg_tokens.io import ensure_dir, load_array


def run_batch_plot_pca_trajectory(
    timecourse_path: str,
    output_dir: str,
    conditions: Optional[Sequence[str]] = None,
    components: Sequence[int] = (1, 2, 3),
) -> str:
    print(f"=== Starting PCA 3D Trajectory Plotting ===")
    
    output_path = ensure_dir(output_dir)
    loaded = load_array(timecourse_path, require_sidecar=True)
    data = loaded.data
    print(f"Loaded timecourse shape: {data.shape}")

    coords = loaded.metadata.get("coords", {})
    if conditions is None:
        conditions = [str(item) for item in coords.get("condition", [])]
    if not conditions:
        conditions = [f"Condition {idx + 1}" for idx in range(data.shape[0])]

    if data.ndim == 4:
        mean_data = np.nanmean(data, axis=1)
    elif data.ndim == 3:
        mean_data = np.asarray(data, dtype=float)
    else:
        raise ValueError(f"Expected trajectory shape condition x component x time or condition x subject x component x time, got {data.shape}")
    if mean_data.shape[1] < 2:
        raise ValueError("At least two PCA components are required for trajectory plotting")
    if len(components) < 2:
        raise ValueError("At least two component indices are required for trajectory plotting")

    component_indices = [component - 1 for component in components]
    if any(component < 0 or component >= mean_data.shape[1] for component in component_indices):
        raise ValueError(f"Requested components {components} are out of range for {mean_data.shape[1]} components")
    while len(component_indices) < 3:
        component_indices.append(component_indices[-1])
        
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(conditions)))
    
    for c, cond_name in enumerate(conditions):
        if c >= mean_data.shape[0]:
            break
            
        pc1 = mean_data[c, component_indices[0], :]
        pc2 = mean_data[c, component_indices[1], :]
        pc3 = mean_data[c, component_indices[2], :] if len(set(component_indices[:3])) > 2 else np.zeros_like(pc1)
        finite = np.isfinite(pc1) & np.isfinite(pc2) & np.isfinite(pc3)
        if not np.any(finite):
            continue
        pc1, pc2, pc3 = pc1[finite], pc2[finite], pc3[finite]
        
        ax.plot(pc1, pc2, pc3, label=cond_name, color=colors[c], linewidth=2)
        
        ax.scatter(pc1[0], pc2[0], pc3[0], color='green', marker='o', s=50, label='Start' if c == 0 else "")
        ax.scatter(pc1[-1], pc2[-1], pc3[-1], color='red', marker='x', s=50, label='End' if c == 0 else "")

    ax.set_xlabel(f'Principal Component {components[0]}')
    ax.set_ylabel(f'Principal Component {components[1]}')
    z_component = components[2] if len(components) > 2 else components[-1]
    ax.set_zlabel(f'Principal Component {z_component}')
    ax.set_title('3D State Space Trajectory of PCA Components')
    ax.legend(loc='upper right')
    
    save_path = output_path / "pca_3d_trajectory.png"
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved 3D Trajectory Plot to {save_path}")
    return str(save_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot 3D State Space Trajectories of PCA Components")
    parser.add_argument("--timecourse_path", type=str, required=True,
                        help="Path to .npy file containing component timecourses")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory to save the 3D plot")
    parser.add_argument("--conditions", type=str, nargs='+', default=None,
                        help="Names of the conditions being compared")
    parser.add_argument("--components", type=int, nargs="+", default=[1, 2, 3],
                        help="One-based PCA component indices to plot.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_batch_plot_pca_trajectory(args.timecourse_path, args.out_dir, args.conditions, args.components)


if __name__ == "__main__":
    main()
