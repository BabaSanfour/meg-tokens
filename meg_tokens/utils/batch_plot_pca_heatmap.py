import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from meg_tokens.io import ensure_dir, load_array

def run_batch_plot_pca_heatmap(
    data_path: str,
    output_dir: str,
    rois: list = None
):
    print(f"=== Starting PCA ROI Heatmap Plotting ===")
    
    output_path = ensure_dir(output_dir)
        
    if rois is None or len(rois) == 0:
        rois = ['Pallidum', 'Caudate', 'Putamen', 'Amygdala', 'Thalamus-Proper', 'Cerebellum-Cortex', 'Brain-Stem']
        
    data = load_array(data_path, expected_ndim=2).data
    print(f"Loaded heatmap data shape: {data.shape}")
    if data.shape[1] != len(rois):
        raise ValueError(f"Heatmap data has {data.shape[1]} ROI columns but {len(rois)} ROI labels were provided")
        
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
    
    save_path = output_path / "pca_roi_heatmap.png"
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
