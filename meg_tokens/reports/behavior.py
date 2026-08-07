import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_fast_slow_distributions(
    dt_fast: np.ndarray,
    dt_slow: np.ndarray,
    title: str = "Raw Response Time: Fast vs. Slow",
    save_path: str = None
) -> plt.Figure:
    """
    Plot raw response-time distributions for Fast and Slow trials.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use histplot/kdeplot instead of deprecated distplot
    sns.kdeplot(dt_slow, label='Slow', color='orange', fill=True, alpha=0.3, ax=ax)
    sns.kdeplot(dt_fast, label='Fast', color='blue', fill=True, alpha=0.3, ax=ax)
    
    ax.legend(fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Raw response time (ms)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig

def plot_trial_class_distributions(
    easy_dt: np.ndarray,
    ambiguous_dt: np.ndarray,
    misleading_dt: np.ndarray,
    title: str = "Decision Time by Trial Difficulty",
    save_path: str = None
) -> plt.Figure:
    """
    Plots the probability density distributions of Decision Times for Easy, Ambiguous, and Misleading trials.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.kdeplot(easy_dt, label='Easy', color='green', fill=True, alpha=0.2, ax=ax)
    sns.kdeplot(ambiguous_dt, label='Ambiguous', color='blue', fill=True, alpha=0.2, ax=ax)
    sns.kdeplot(misleading_dt, label='Misleading', color='red', fill=True, alpha=0.2, ax=ax)
    
    ax.legend(fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Decision Time (ms)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig

def plot_comparison_bars(
    mean_a: float,
    sem_a: float,
    mean_b: float,
    sem_b: float,
    raw_a: np.ndarray,
    raw_b: np.ndarray,
    label_a: str = "Condition A",
    label_b: str = "Condition B",
    title: str = "Decision Time Comparison",
    save_path: str = None
) -> plt.Figure:
    """
    Plots a bar chart comparing two conditions with SEM error bars, and overlays 
    individual subject data points with horizontal jitter.
    """
    means = np.array([mean_a, mean_b])
    sems = np.array([sem_a, sem_b])
    
    fig, ax = plt.subplots(figsize=(5, 8))
    
    # Plot bars
    indices = np.arange(2)
    ax.bar(indices, means, yerr=sems, color=['lime', 'red'], alpha=0.6, capsize=8, zorder=1)
    
    # Add deterministic horizontal offsets for subject points.
    jitter_a = indices[0] + np.linspace(-0.08, 0.08, len(raw_a)) if len(raw_a) else np.array([])
    jitter_b = indices[1] + np.linspace(-0.08, 0.08, len(raw_b)) if len(raw_b) else np.array([])
    
    # Curated colormap for subjects
    subject_colors = plt.cm.tab20(np.linspace(0, 1, max(len(raw_a), len(raw_b))))
    
    for i in range(len(raw_a)):
        ax.scatter(jitter_a[i], raw_a[i], marker='^', color=subject_colors[i % len(subject_colors)], s=50, zorder=2)
    for i in range(len(raw_b)):
        ax.scatter(jitter_b[i], raw_b[i], marker='^', color=subject_colors[i % len(subject_colors)], s=50, zorder=2)
        
    ax.set_xticks(indices)
    ax.set_xticklabels([label_a, label_b], fontsize=12)
    ax.set_ylabel('Decision Time (ms)', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Set reasonable y limits
    all_vals = np.concatenate([raw_a, raw_b])
    if len(all_vals) > 0:
        y_min = max(0, np.min(all_vals) - 200)
        y_max = np.max(all_vals) + 200
        ax.set_ylim(y_min, y_max)
        
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig
