import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from meg_tokens.meg.plotting import plot_roi_timecourse
import seaborn as sns

def run_batch_plot_component_timecourse(
    timecourse_path: str,
    output_dir: str,
    components_to_plot: int = 3,
    conditions: list = ['Condition 1', 'Condition 2']
):
    print(f"=== Starting Component Timecourse Plotting ===")
    print(f"Loading timecourses from: {timecourse_path}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Expected shape: (n_conditions, n_subjects, n_components, n_times)
    # OR for a mock execution if the file doesn't exist, we generate dummy data
    try:
        data = np.load(timecourse_path)
        print(f"Loaded timecourses shape: {data.shape}")
    except FileNotFoundError:
        print(f"Warning: {timecourse_path} not found. Generating mock data for demonstration.")
        n_conds = len(conditions)
        n_subjects = 28
        n_components = 5
        n_times = 100
        # Shape: (conditions, subjects, components, time)
        data = np.random.normal(0, 1, (n_conds, n_subjects, n_components, n_times))
        # Add a mock signal to condition 1, component 0
        data[0, :, 0, 40:60] += np.sin(np.linspace(0, np.pi, 20)) * 2
        
    n_times = data.shape[-1]
    times = np.linspace(-500, 1000, n_times) # Mock time axis in ms
    
    for comp_idx in range(min(components_to_plot, data.shape[2])):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data for this component: shape (n_subjects, n_times)
        data_cond1 = data[0, :, comp_idx, :]
        data_cond2 = data[1, :, comp_idx, :] if data.shape[0] > 1 else np.zeros_like(data_cond1)
        
        ax = plot_roi_timecourse(
            times=times,
            data_cond1=data_cond1,
            data_cond2=data_cond2,
            label_cond1=conditions[0],
            label_cond2=conditions[1] if len(conditions) > 1 else 'None',
            roi_name=f'Principal Component {comp_idx + 1}',
            ax=ax
        )
        
        save_path = os.path.join(output_dir, f"component_{comp_idx+1}_timecourse.png")
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Saved Component {comp_idx+1} timecourse to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot 2D line graphs of PCA/LDA component timecourses")
    parser.add_argument("--timecourse_path", type=str, required=True,
                        help="Path to .npy file containing component timecourses")
    parser.add_argument("--out_dir", type=str, default='./figures/component_timecourses/',
                        help="Directory to save 2D line plots")
    parser.add_argument("--components", type=int, default=3,
                        help="Number of components to plot")
    parser.add_argument("--conditions", type=str, nargs='+', default=['Fast', 'Slow'],
                        help="Names of the conditions being compared")
    
    args = parser.parse_args()
    run_batch_plot_component_timecourse(args.timecourse_path, args.out_dir, args.components, args.conditions)
