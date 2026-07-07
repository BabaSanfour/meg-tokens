import argparse
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import matplotlib.pyplot as plt
from meg_tokens.meg.plotting import plot_roi_timecourse
from meg_tokens.io import ensure_dir, load_array


def run_batch_plot_component_timecourse(
    timecourse_path: str,
    output_dir: str,
    components_to_plot: int = 3,
    conditions: Optional[Sequence[str]] = None,
) -> list[str]:
    print(f"=== Starting Component Timecourse Plotting ===")
    print(f"Loading timecourses from: {timecourse_path}")
    
    output_path = ensure_dir(output_dir)

    loaded = load_array(timecourse_path, require_sidecar=True)
    data = loaded.data
    print(f"Loaded timecourses shape: {data.shape}")
    if data.ndim == 3:
        data = data[:, np.newaxis, :, :]
    elif data.ndim != 4:
        raise ValueError("Expected shape condition x component x time or condition x subject x component x time")

    coords = loaded.metadata.get("coords", {})
    if conditions is None:
        conditions = [str(item) for item in coords.get("condition", [])]
    if not conditions:
        conditions = [f"Condition {idx + 1}" for idx in range(data.shape[0])]
    if len(conditions) < 2:
        raise ValueError("At least two condition labels are required for component timecourse comparison")
    if data.shape[0] < len(conditions):
        raise ValueError(f"Timecourse data has {data.shape[0]} conditions but {len(conditions)} condition labels were provided")
        
    n_times = data.shape[-1]
    times = np.asarray(coords.get("time_sec", coords.get("time_ms", np.arange(n_times))), dtype=float)
    if times.shape[0] != n_times:
        raise ValueError("time coordinate length does not match the time dimension")

    saved = []
    
    for comp_idx in range(min(components_to_plot, data.shape[2])):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data for this component: shape (n_subjects, n_times)
        data_cond1 = data[0, :, comp_idx, :]
        data_cond2 = data[1, :, comp_idx, :]
        
        ax = plot_roi_timecourse(
            times=times,
            data_cond1=data_cond1,
            data_cond2=data_cond2,
            label_cond1=conditions[0],
            label_cond2=conditions[1] if len(conditions) > 1 else 'None',
            roi_name=f'Principal Component {comp_idx + 1}',
            ax=ax
        )
        
        save_path = output_path / f"component_{comp_idx+1}_timecourse.png"
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Saved Component {comp_idx+1} timecourse to {save_path}")
        saved.append(str(save_path))
    return saved


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot 2D line graphs of PCA/LDA component timecourses")
    parser.add_argument("--timecourse_path", type=str, required=True,
                        help="Path to .npy file containing component timecourses")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory to save 2D line plots")
    parser.add_argument("--components", type=int, default=3,
                        help="Number of components to plot")
    parser.add_argument("--conditions", type=str, nargs='+', default=None,
                        help="Names of the conditions being compared")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_batch_plot_component_timecourse(args.timecourse_path, args.out_dir, args.components, args.conditions)


if __name__ == "__main__":
    main()
