import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def run_batch_plot_pca_trajectory(
    timecourse_path: str,
    output_dir: str,
    conditions: list = ['Condition 1', 'Condition 2']
):
    print(f"=== Starting PCA 3D Trajectory Plotting ===")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    try:
        data = np.load(timecourse_path)
        print(f"Loaded timecourse shape: {data.shape}")
    except FileNotFoundError:
        print(f"Warning: {timecourse_path} not found. Generating mock data for demonstration.")
        # Shape: (conditions, subjects, components, time)
        n_conds = len(conditions)
        n_subjects = 28
        n_components = 3
        n_times = 100
        
        # Create a mock spiral trajectory in 3D
        t = np.linspace(0, 4 * np.pi, n_times)
        data = np.zeros((n_conds, n_subjects, n_components, n_times))
        for c in range(n_conds):
            phase = c * np.pi
            data[c, :, 0, :] = np.sin(t + phase) * t
            data[c, :, 1, :] = np.cos(t + phase) * t
            data[c, :, 2, :] = t
        
        # Add noise
        data += np.random.normal(0, 1, data.shape)

    # Average across subjects to get the mean trajectory for each condition
    # Resulting shape: (conditions, components, time)
    if data.ndim == 4:
        mean_data = np.mean(data, axis=1)
    else:
        mean_data = data
        
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(conditions)))
    
    for c, cond_name in enumerate(conditions):
        if c >= mean_data.shape[0]:
            break
            
        pc1 = mean_data[c, 0, :]
        pc2 = mean_data[c, 1, :]
        pc3 = mean_data[c, 2, :] if mean_data.shape[1] > 2 else np.zeros_like(pc1)
        
        ax.plot(pc1, pc2, pc3, label=cond_name, color=colors[c], linewidth=2)
        
        # Mark the start and end of the trajectory
        ax.scatter(pc1[0], pc2[0], pc3[0], color='green', marker='o', s=50, label='Start' if c == 0 else "")
        ax.scatter(pc1[-1], pc2[-1], pc3[-1], color='red', marker='x', s=50, label='End' if c == 0 else "")

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('3D State Space Trajectory of PCA Components')
    ax.legend(loc='upper right')
    
    save_path = os.path.join(output_dir, "pca_3d_trajectory.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved 3D Trajectory Plot to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot 3D State Space Trajectories of PCA Components")
    parser.add_argument("--timecourse_path", type=str, required=True,
                        help="Path to .npy file containing component timecourses")
    parser.add_argument("--out_dir", type=str, default='./figures/pca_trajectories/',
                        help="Directory to save the 3D plot")
    parser.add_argument("--conditions", type=str, nargs='+', default=['Fast', 'Slow'],
                        help="Names of the conditions being compared")
    
    args = parser.parse_args()
    run_batch_plot_pca_trajectory(args.timecourse_path, args.out_dir, args.conditions)
