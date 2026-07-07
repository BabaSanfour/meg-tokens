"""
Pipeline execution script for visualizing Stage 3 Behavioral Distributions and
Stage 8 Brain-Behavior Correlations (e.g. Success Probability & Latencies).
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from meg_tokens.io import ensure_dir, require_file
from meg_tokens.behavior.plotting import (
    plot_fast_slow_distributions,
    plot_trial_class_distributions,
    plot_comparison_bars
)
from meg_tokens.meg.plotting import plot_correlation

def run_behavior_plotting(
    behavior_dir: str,
    output_figures_dir: str,
    subjects_list: list = None,
    neural_metrics_csv: str = None
):
    print("=== Generating Behavioral & Correlation Plots ===")

    output_path = ensure_dir(output_figures_dir)
        
    if subjects_list is None:
        subjects_list = sorted([d for d in os.listdir(behavior_dir) if d.startswith('H')])
        
    all_fast_dt = []
    all_slow_dt = []
    
    # 1. Load behavioral data to generate Decision Time Distributions
    for subject in subjects_list:
        subj_dir = os.path.join(behavior_dir, subject)
        if not os.path.isdir(subj_dir):
            continue
            
        fast_files = glob.glob(os.path.join(subj_dir, "*Fast*.csv"))
        slow_files = glob.glob(os.path.join(subj_dir, "*Slow*.csv"))
        
        for f in fast_files:
            df = pd.read_csv(f)
            if 'tDecisionTime' in df.columns:
                all_fast_dt.extend(df['tDecisionTime'].dropna().values)
                
        for f in slow_files:
            df = pd.read_csv(f)
            if 'tDecisionTime' in df.columns:
                all_slow_dt.extend(df['tDecisionTime'].dropna().values)

    if all_fast_dt and all_slow_dt:
        # Scale to ms if necessary (assumes data is in ms or samples)
        # Using the plotting module
        fig = plot_fast_slow_distributions(
            dt_fast=np.array(all_fast_dt),
            dt_slow=np.array(all_slow_dt),
            title="Decision Time Distribution: Fast vs Slow",
            save_path=os.path.join(output_figures_dir, "dt_fast_vs_slow_kde.png")
        )
        plt.close(fig)
        
        print("Generated Decision Time KDE plots.")
        
    if neural_metrics_csv:
        metrics = pd.read_csv(require_file(neural_metrics_csv, purpose="neural metrics for behavior correlation"))
        required = {"subject", "neural_peak_ms"}
        missing = required - set(metrics.columns)
        if missing:
            raise ValueError(f"Neural metrics table is missing columns: {sorted(missing)}")

        mean_subj_dts = []
        neural_peaks = []
        for subject in subjects_list:
            subj_dir = os.path.join(behavior_dir, subject)
            paths = glob.glob(os.path.join(subj_dir, "*.csv"))
            values = []
            for path in paths:
                df = pd.read_csv(path)
                if "tDecisionTime" in df.columns:
                    values.extend(df["tDecisionTime"].dropna().values)
            metric_rows = metrics.loc[metrics["subject"] == subject, "neural_peak_ms"]
            if values and not metric_rows.empty:
                mean_subj_dts.append(float(np.mean(values)))
                neural_peaks.append(float(metric_rows.iloc[0]))

        if len(mean_subj_dts) < 3:
            raise ValueError("At least three subjects with behavior and neural metrics are required for correlation")

        fig, ax = plt.subplots(figsize=(8, 8))
        plot_correlation(
            x_data=np.asarray(mean_subj_dts),
            y_data=np.asarray(neural_peaks),
            x_label='Behavioral Decision Time (ms)',
            y_label='Neural Peak Commitment (ms)',
            title='Correlation: Behavior vs. Neural Peak',
            ax=ax
        )
        fig.savefig(output_path / "correlation_dt_vs_neural_peak.png", dpi=300)
        plt.close(fig)
        print("Generated Brain-Behavior correlation scatterplots.")
    else:
        print("No neural metrics table provided; skipped brain-behavior correlation plotting.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run behavioral plotting and correlations.")
    parser.add_argument("--subjects", type=str, nargs='+', default=None)
    parser.add_argument("--behavior_dir", type=str, default='/media/external/DDM/dataframes/')
    parser.add_argument("--out_dir", type=str, default='/media/external/DDM/figures/behavior/')
    parser.add_argument("--neural_metrics_csv", type=str, default=None,
                        help="CSV with columns subject and neural_peak_ms for correlation plotting")
    args = parser.parse_args()
    
    run_behavior_plotting(
        behavior_dir=args.behavior_dir,
        output_figures_dir=args.out_dir,
        subjects_list=args.subjects,
        neural_metrics_csv=args.neural_metrics_csv
    )
