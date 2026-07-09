from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from meg_tokens.io import DerivativeLayout, load_array, save_sidecar

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
    time_step: int = 50,
) -> Path:
    """
    Automatically extracts the first significant decoding moment for multiple conditions
    and plots them as a comparative bar chart (replicating 09_1st_moment... notebooks natively).
    """
    print("=== Extracting Decoding Onsets ===")
    
    if len(score_files) != len(condition_names) or len(threshold_files) != len(condition_names):
        raise ValueError(
            "score_files, threshold_files, and condition_names must have equal lengths"
        )
        
    onsets = []
    valid_names = []
    colors = ['#F5D3C8', '#FF8484', '#AA4465', '#B9C0DA', '#83BCFF', '#0471A6']
    
    for sf, tf, name in zip(score_files, threshold_files, condition_names):
        score_array = load_array(sf, expected_ndim=1, require_sidecar=True)
        threshold_array = load_array(tf, expected_ndim=1, require_sidecar=True)
        scores = np.asarray(score_array.data, dtype=float)
        threshold = float(threshold_array.data[0])
        coords = score_array.metadata.get("coords", {})
        if "time_sec" in coords:
            times = np.asarray(coords["time_sec"], dtype=float) * 1000.0
        elif "time_ms" in coords:
            times = np.asarray(coords["time_ms"], dtype=float)
        else:
            times = np.arange(len(scores)) * time_step + time_offset
        if times.shape != scores.shape:
            raise ValueError(f"Decoding time coordinates do not match scores in {sf}")
        
        onset = extract_first_significant_moment(scores, threshold, times)
        if onset is not None:
            print(f"[{name}] First significant decoding moment: {onset} ms")
            onsets.append(onset)
            valid_names.append(name)
        else:
            print(f"[{name}] No significant decoding found.")
            onsets.append(0)
            valid_names.append(name)
            
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
    
    out_png = DerivativeLayout(output_dir).path(
        subject="group",
        datatype="fig",
        description="-vs-".join(name.lower() for name in valid_names),
        suffix="decodingonset",
        extension=".png",
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()
    save_sidecar(
        out_png,
        {
            "stage": "decoding_report",
            "kind": "first_significant_moment",
            "conditions": list(valid_names),
            "score_files": [str(path) for path in score_files],
            "threshold_files": [str(path) for path in threshold_files],
            "onsets_ms": [float(value) for value in onsets],
        },
    )
    print(f"Saved Onset Bar Chart -> {out_png}")
    return out_png
