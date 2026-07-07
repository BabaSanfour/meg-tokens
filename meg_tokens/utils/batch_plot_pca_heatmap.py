import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_batch_plot_pca_heatmap(
    data_path: str,
    output_dir: str,
    rois: list = None
):
    print(f"=== Starting PCA ROI Heatmap Plotting ===")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if rois is None or len(rois) == 0:
        rois = ['Pallidum', 'Caudate', 'Putamen', 'Amygdala', 'Thalamus-Proper', 'Cerebellum-Cortex', 'Brain-Stem']
        
    try:
        data = np.load(data_path)
        print(f"Loaded heatmap data shape: {data.shape}")
    except FileNotFoundError:
        print(f"Warning: {data_path} not found. Generating mock data for demonstration.")
        # Mock data: shape (n_components, n_rois)
        n_components = 5
        data = np.random.uniform(10, 50, (n_components, len(rois)))
        # Make the first component explain the most variance
        data[0, :] = np.random.uniform(40, 70, len(rois))
        
    n_components = data.shape[0]
    
    # Construct DataFrame for seaborn pivoting
    records = []
    for pc_idx in range(n_components):
        for roi_idx, roi in enumerate(rois):
            records.append({
                'PC': f'PC{pc_idx + 1}',
                'ROI': roi,
                'nVarExpl': data[pc_idx, roi_idx]
            })
            
    df = pd.DataFrame(records)
    df_piv = df.pivot(index='PC', columns='ROI', values='nVarExpl')
    
    # Plotting
    f, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        df_piv, 
        linewidths=0.5, 
        linecolor='silver', 
        annot=True, 
        fmt=".1f",
        ax=ax, 
        cmap="YlGnBu_r",
        cbar_kws={'label': 'Variance Explained (%)'}
    )
    
    ax.set_title('PCA Variance Explained by ROI', fontsize=14, pad=15)
    ax.set_ylabel('Principal Component', fontsize=12)
    ax.set_xlabel('Region of Interest (ROI)', fontsize=12)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "pca_roi_heatmap.png")
    plt.savefig(save_path, dpi=300)
    plt.close(f)
    print(f"Saved PCA Heatmap to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot PCA Variance Explained Heatmap (PCs vs ROIs)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to .npy file containing 2D array of variance (n_components, n_rois)")
    parser.add_argument("--out_dir", type=str, default='./figures/pca_heatmaps/',
                        help="Directory to save the heatmap")
    parser.add_argument("--rois", type=str, nargs='+', default=None,
                        help="List of ROI names corresponding to the columns of the .npy matrix")
    
    args = parser.parse_args()
    run_batch_plot_pca_heatmap(args.data_path, args.out_dir, args.rois)
