import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_batch_plot_dpca(
    dpca_prefix: str,
    output_dir: str,
    time_step: float = 50.0,
    time_offset: float = -1000.0,
    n_components_to_plot: int = 3
):
    """
    Plots the top Demixed Principal Components (dPCA) and their significance masks.
    
    This replaces the legacy '092_Mixed_PCA_PLOT.ipynb' scratchpad with a formal 
    automated tool that dynamically iterates over all marginalized dimensions.
    """
    print(f"=== Starting dPCA Visualization ===")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    sns.set_theme(style="whitegrid", context="paper", palette="deep")
    
    print(f"Loading dPCA results from prefix: {dpca_prefix}")
    try:
        components = np.load(f"{dpca_prefix}_components.npy", allow_pickle=True).item()
        masks = np.load(f"{dpca_prefix}_signif_masks.npy", allow_pickle=True).item()
    except FileNotFoundError:
        print(f"Error: Could not find dPCA result files matching prefix '{dpca_prefix}'")
        return
        
    for key, Z in components.items():
        print(f"Plotting marginalization '{key}'...")
        
        # Z shape is (n_components, dim1, dim2, ..., n_times)
        # We want to plot the first N components.
        n_times = Z.shape[-1]
        times = np.arange(n_times) * time_step + time_offset
        
        n_dims = len(Z.shape) - 2 # Subtract components and time axes
        mask_array = masks.get(key, None)
        
        for comp_idx in range(min(n_components_to_plot, Z.shape[0])):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Collapse all condition dimensions into a single list of 1D arrays for plotting
            import itertools
            condition_shapes = Z.shape[1:-1]
            if n_dims > 0:
                for idxs in itertools.product(*[range(d) for d in condition_shapes]):
                    slice_idx = (comp_idx,) + idxs + (slice(None),)
                    time_course = Z[slice_idx]
                    
                    label = f"Condition {idxs}"
                    ax.plot(times, time_course, label=label, lw=2, alpha=0.8)
            else:
                # E.g., for pure time 't' component
                time_course = Z[comp_idx, :]
                ax.plot(times, time_course, label="Main Effect", lw=2, color='black')
                
            # Plot significance mask if available
            if mask_array is not None:
                comp_mask = mask_array[comp_idx] # boolean array over time
                
                # Extract contiguous blocks of True
                sig_indices = np.where(comp_mask)[0]
                if len(sig_indices) > 0:
                    # Find jumps to define blocks
                    breaks = np.where(np.diff(sig_indices) > 1)[0]
                    splits = np.split(sig_indices, breaks + 1)
                    
                    for block in splits:
                        start_time = times[block[0]]
                        end_time = times[block[-1]]
                        ax.axvspan(start_time, end_time, color='gray', alpha=0.2)
                        
                    ax.plot([], [], color='gray', alpha=0.2, linewidth=10, label='Significant (p<0.05)')
                    
            ax.set_title(f"dPCA Marginalization '{key}' - Component {comp_idx+1}", fontsize=14)
            ax.set_xlabel("Time (ms)", fontsize=12)
            ax.set_ylabel("Component Amplitude (A.U.)", fontsize=12)
            ax.axvline(0, color='black', linestyle='--', alpha=0.5)
            
            # Only show legend if not too many conditions
            if n_dims > 0 and np.prod(condition_shapes) <= 12:
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc='best')
                
            sns.despine(ax=ax)
            
            out_file = os.path.join(output_dir, f"dpca_{key}_comp{comp_idx+1}.png")
            plt.savefig(out_file, dpi=300, bbox_inches='tight')
            plt.close()
            
    print(f"Saved all dPCA component plots to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Demixed PCA (dPCA) Components")
    parser.add_argument("--dpca_prefix", type=str, required=True,
                        help="Path prefix to the saved dPCA .npy files")
    parser.add_argument("--out_dir", type=str, default='./figures/dpca/',
                        help="Directory to save the plots")
    parser.add_argument("--n_components", type=int, default=3,
                        help="Number of components to plot per marginalization")
    parser.add_argument("--time_step", type=float, default=50.0,
                        help="Time step between samples in ms")
    parser.add_argument("--time_offset", type=float, default=-1000.0,
                        help="Time offset for the first sample in ms")
    
    args = parser.parse_args()
    run_batch_plot_dpca(
        args.dpca_prefix, args.out_dir, args.time_step, args.time_offset, args.n_components
    )
