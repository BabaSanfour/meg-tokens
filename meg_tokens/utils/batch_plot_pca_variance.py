import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_batch_plot_pca_variance(
    variance_path: str,
    output_dir: str
):
    print(f"=== Starting PCA Variance Explained Plotting ===")
    print(f"Loading variance data from: {variance_path}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    try:
        nVarExpl = np.load(variance_path)
    except FileNotFoundError:
        print(f"Warning: {variance_path} not found. Generating mock data for demonstration.")
        # Mock variance explained that drops off logarithmically
        nVarExpl = np.exp(-np.arange(1, 21)/4)
        nVarExpl /= np.sum(nVarExpl)
        
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
    
    save_path = os.path.join(output_dir, "pca_variance_explained.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved PCA Variance plot to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot PCA Variance Explained (Scree Plot)")
    parser.add_argument("--variance_path", type=str, required=True,
                        help="Path to .npy file containing 1D array of variance explained ratios")
    parser.add_argument("--out_dir", type=str, default='./figures/pca_loadings/',
                        help="Directory to save the scree plot")
    
    args = parser.parse_args()
    run_batch_plot_pca_variance(args.variance_path, args.out_dir)
