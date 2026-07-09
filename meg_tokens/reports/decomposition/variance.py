import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from meg_tokens.io import ensure_dir, load_array

def run_batch_plot_pca_variance(
    variance_path: str,
    output_dir: str
) -> str:
    print(f"=== Starting PCA Variance Explained Plotting ===")
    print(f"Loading variance data from: {variance_path}")
    
    output_path = ensure_dir(output_dir)
    nVarExpl = load_array(variance_path, expected_ndim=1).data
        
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot cumulative sum
    cum_var = np.cumsum(nVarExpl) * 100
    
    ax.plot(cum_var, 'o', color=sns.color_palette("dark")[3], markersize=8)
    ax.plot(cum_var, '-', color='black', lw=2)
    
    ax.set_ylabel('Cumulative Variance Explained (%)', fontsize=12)
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_xticks(np.arange(0, len(cum_var), max(1, len(cum_var)//10)))
    ax.set_title('PCA Scree Plot', fontsize=14)
    
    sns.despine(ax=ax)
    
    save_path = output_path / "pca_variance_explained.png"
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved PCA Variance plot to {save_path}")
    return str(save_path)
