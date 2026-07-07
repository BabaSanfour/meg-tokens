import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def extract_first_significant_moment(scores, threshold, times):
    """
    Finds the first timepoint where the score exceeds the permutation threshold.
    Returns the time in milliseconds, or None if never significant.
    """
    significant_indices = np.where(scores > threshold)[0]
    if len(significant_indices) > 0:
        return times[significant_indices[0]]
    return None

def run_batch_plot_decoding_onset(
    score_files: list,
    threshold_files: list,
    condition_names: list,
    output_dir: str,
    time_offset: int = 0,
    time_step: int = 50
):
    """
    Automatically extracts the first significant decoding moment for multiple conditions
    and plots them as a comparative bar chart (replicating 09_1st_moment... notebooks natively).
    """
    print("=== Extracting Decoding Onsets ===")
    
    if len(score_files) != len(condition_names) or len(threshold_files) != len(condition_names):
        print("Error: You must provide an equal number of score files, threshold files, and condition names.")
        return
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    onsets = []
    valid_names = []
    colors = ['#F5D3C8', '#FF8484', '#AA4465', '#B9C0DA', '#83BCFF', '#0471A6']
    
    for i, (sf, tf, name) in enumerate(zip(score_files, threshold_files, condition_names)):
        if not os.path.exists(sf) or not os.path.exists(tf):
            print(f"Warning: Could not find files for {name}. Skipping.")
            continue
            
        scores = np.load(sf)
        threshold = np.load(tf)[0]
        
        # Reconstruct time axis based on array length
        times = np.arange(len(scores)) * time_step + time_offset
        
        onset = extract_first_significant_moment(scores, threshold, times)
        if onset is not None:
            print(f"[{name}] First significant decoding moment: {onset} ms")
            onsets.append(onset)
            valid_names.append(name)
        else:
            print(f"[{name}] No significant decoding found.")
            onsets.append(0)
            valid_names.append(name)
            
    if len(onsets) == 0:
        print("No significant onsets extracted.")
        return
        
    # --- Plot the Bar Chart ---
    plt.figure(figsize=(8, 10))
    x_pos = np.arange(len(valid_names)) * 1.5
    
    # Assign colors dynamically up to the list length, wrap around if needed
    plot_colors = [colors[i % len(colors)] for i in range(len(valid_names))]
    
    plt.bar(x_pos, onsets, color=plot_colors, width=1.0)
    plt.axhline(y=0, color='k', linestyle='-', linewidth=1.5)
    
    plt.xticks(x_pos, valid_names, fontweight='bold')
    plt.xlabel("Conditions")
    plt.ylabel("First signif. moment in time (ms)")
    plt.title("Decoding Onset Latencies")
    
    out_png = os.path.join(output_dir, f"Bar_1st_signif_time_{'_vs_'.join(valid_names)}.png")
    plt.savefig(out_png, dpi=300)
    plt.close()
    
    print(f"Saved Onset Bar Chart -> {out_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and plot decoding onset latencies")
    parser.add_argument("--scores", type=str, nargs='+', required=True,
                        help="List of path to decoding_scores.npy files")
    parser.add_argument("--thresholds", type=str, nargs='+', required=True,
                        help="List of path to decoding_threshold.npy files")
    parser.add_argument("--names", type=str, nargs='+', required=True,
                        help="Names of the conditions corresponding to the files")
    parser.add_argument("--out_dir", type=str, default='./decoding_results/figures/',
                        help="Directory to save the bar chart")
    parser.add_argument("--time_offset", type=float, default=-1000,
                        help="Start time of the epoch in ms (e.g., -1000 for 1s baseline)")
    parser.add_argument("--time_step", type=float, default=50,
                        help="Time resolution between data points in ms")
                        
    args = parser.parse_args()
    run_batch_plot_decoding_onset(
        args.scores, args.thresholds, args.names, args.out_dir, 
        time_offset=args.time_offset, time_step=args.time_step
    )
