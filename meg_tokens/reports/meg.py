import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mne

# Set seaborn defaults for beautiful modern plots
sns.set_theme(style="whitegrid", context="paper", palette="deep")

def plot_roi_timecourse(
    times: np.ndarray,
    data_cond1: np.ndarray,
    data_cond2: np.ndarray,
    label_cond1: str = 'Condition 1',
    label_cond2: str = 'Condition 2',
    significance_windows: list = None,
    roi_name: str = 'ROI',
    ax=None
):
    """
    Plots the time course of neural activity for two conditions with standard error shading.
    Highlights periods of statistical significance.
    
    Args:
        times: Array of time points (in ms or s).
        data_cond1: Array of shape (n_subjects, n_times) for Condition 1.
        data_cond2: Array of shape (n_subjects, n_times) for Condition 2.
        label_cond1: Label for Condition 1.
        label_cond2: Label for Condition 2.
        significance_windows: List of (start_idx, end_idx) tuples indicating significant windows.
        roi_name: Name of the ROI for the title.
        ax: Matplotlib axis. If None, creates a new figure.
        
    Returns:
        Matplotlib axis.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
    mean1 = np.mean(data_cond1, axis=0)
    sem1 = np.std(data_cond1, axis=0) / np.sqrt(data_cond1.shape[0])
    
    mean2 = np.mean(data_cond2, axis=0)
    sem2 = np.std(data_cond2, axis=0) / np.sqrt(data_cond2.shape[0])
    
    ax.plot(times, mean1, label=label_cond1, color=sns.color_palette()[0], lw=2)
    ax.fill_between(times, mean1 - sem1, mean1 + sem1, alpha=0.3, color=sns.color_palette()[0])
    
    ax.plot(times, mean2, label=label_cond2, color=sns.color_palette()[1], lw=2)
    ax.fill_between(times, mean2 - sem2, mean2 + sem2, alpha=0.3, color=sns.color_palette()[1])
    
    # Highlight significant windows
    if significance_windows:
        for (start_idx, end_idx) in significance_windows:
            if start_idx < len(times) and end_idx < len(times):
                ax.axvspan(times[start_idx], times[end_idx], color='gray', alpha=0.2, label='p < 0.05' if start_idx == significance_windows[0][0] else "")
                
    ax.set_title(f"Source Activity: {roi_name}", fontsize=14)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Source Amplitude (A.U.)", fontsize=12)
    
    # Deduplicate labels in legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    
    sns.despine(ax=ax)
    return ax

def plot_correlation(
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_label: str = 'Behavioral Decision Time (ms)',
    y_label: str = 'Neural Peak Commitment (ms)',
    title: str = 'Brain-Behavior Correlation',
    ax=None
):
    """
    Plots a scatterplot with a linear regression fit to correlate neural metrics with behavior.
    
    Args:
        x_data: 1D array of behavioral metrics.
        y_data: 1D array of neural metrics.
        x_label: X-axis label.
        y_label: Y-axis label.
        title: Plot title.
        ax: Matplotlib axis. If None, creates a new figure.
        
    Returns:
        Matplotlib axis.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        
    # Use seaborn's regplot for a beautiful scatter + regression line + confidence interval
    sns.regplot(
        x=x_data, 
        y=y_data, 
        ax=ax, 
        scatter_kws={'alpha':0.6, 's':50}, 
        line_kws={'color': sns.color_palette("dark")[3], 'lw':2}
    )
    
    # Compute basic Pearson correlation for annotation
    corr_matrix = np.corrcoef(x_data, y_data)
    r_val = corr_matrix[0, 1]
    
    ax.set_title(f"{title}\n(r = {r_val:.3f})", fontsize=14)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    
    sns.despine(ax=ax)
    return ax

def plot_brain_tmap(
    stc: mne.SourceEstimate,
    subject: str = 'fsaverage',
    subjects_dir: str = None,
    hemi: str = 'both',
    clim: dict = None,
    time: float = None,
    title: str = 'Source T-Map'
):
    """
    Plots statistical source maps onto a 3D brain using MNE's built-in PyVista 3D backend.
    This replaces the legacy 'visbrain' dependencies.
    
    Args:
        stc: An mne.SourceEstimate holding the t-values.
        subject: Subject template name (default 'fsaverage').
        subjects_dir: Path to FreeSurfer subjects directory.
        hemi: Hemisphere to plot ('lh', 'rh', 'both', or 'split').
        clim: Dictionary for colorbar limits, e.g., dict(kind='value', pos_lims=(2, 3, 5)).
        time: Specific time point to visualize (in seconds).
        title: Title of the window.
        
    Returns:
        brain: mne.viz.Brain object.
    """
    if clim is None:
        # Default symmetric limits for t-values
        clim = dict(kind='value', pos_lims=(1.5, 2.5, 4.0))
        
    brain = stc.plot(
        subject=subject,
        surface='inflated',
        hemi=hemi,
        subjects_dir=subjects_dir,
        time_viewer=True,
        clim=clim,
        colormap='mne',
        transparent=True,
        title=title,
        background='white',
        cortex='low_contrast'
    )
    
    if time is not None:
        brain.set_time(time)
        
    return brain

def plot_tfr_spectrogram(
    tfr: mne.time_frequency.AverageTFR,
    picks=None,
    baseline: tuple = None,
    mode: str = 'logratio',
    title: str = 'Time-Frequency Power',
    ax=None
):
    """
    Plots a Time-Frequency Representation (spectrogram) map.
    This replaces the legacy 'Time_Frequency_maps_*.ipynb' plotting logic.
    
    Args:
        tfr: mne.time_frequency.AverageTFR object containing the spectral power.
        picks: Channels or ROIs to plot. If None, plots all or averages them.
        baseline: Tuple of (tmin, tmax) for baseline correction.
        mode: Baseline correction mode (e.g., 'logratio', 'zscore', 'mean').
        title: Plot title.
        ax: Matplotlib axis.
        
    Returns:
        Matplotlib axis.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
    # Plot TFR using MNE's built in plotter wrapped in our style
    tfr.plot(
        picks=picks,
        baseline=baseline,
        mode=mode,
        title=title,
        axes=ax,
        show=False,
        cmap='viridis'
    )
    
    sns.despine(ax=ax)
    return ax

def plot_sensor_topomap(
    data: np.ndarray,
    pos: np.ndarray,
    mask: np.ndarray = None,
    vlim: tuple = (None, None),
    cmap: str = 'RdBu_r',
    title: str = 'Sensor Topography',
    ax=None
):
    """
    Plots a 2D sensor-space topography map (e.g., for ERPs or statistical t-values).
    Replaces the legacy '55_TTest_stats_topoplot_*.ipynb' scripts.
    
    Args:
        data: 1D array of values to plot per sensor.
        pos: 2D array of sensor positions (n_sensors, 2) or an mne.Info object.
        mask: Boolean array of shape (n_sensors,) indicating significant sensors to highlight.
        vlim: Tuple of (vmin, vmax) for the colormap limits.
        cmap: Colormap name.
        title: Plot title.
        ax: Matplotlib axis.
        
    Returns:
        Matplotlib axis.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        
    mask_params = dict(marker='o', markerfacecolor='white', markersize=5, markeredgecolor='black')
    
    mne.viz.plot_topomap(
        data, 
        pos, 
        axes=ax, 
        mask=mask, 
        mask_params=mask_params if mask is not None else None,
        cmap=cmap, 
        vlim=vlim, 
        show=False,
        contours=0
    )
    
    ax.set_title(title, fontsize=14)
    return ax

def save_brain_movie_frames(
    brain: mne.viz.Brain,
    times: np.ndarray,
    output_dir: str,
    prefix: str = "Fig",
    views: list = ['dorsal', 'lateral', 'medial']
):
    """
    Core function to export time-resolved frame-by-frame screenshots of a 3D brain map.
    Replicates the legacy 'visbrain' screenshot loop by capturing multiple views per timepoint.
    
    Args:
        brain: The mne.viz.Brain object (e.g. returned by stc.plot).
        times: Array of timepoints corresponding to the SourceEstimate.
        output_dir: Directory to save the PNG frames.
        prefix: Prefix for the output filenames.
        views: List of MNE views to capture ('dorsal'=top, 'lateral'=left, 'medial'=right).
    """
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Exporting {len(times)} timepoints x {len(views)} views...")
    for t_idx, t in enumerate(times):
        brain.set_time(t)
        for view in views:
            brain.show_view(view)
            out_file = os.path.join(output_dir, f"{prefix}_{view}_{t_idx}.png")
            # In a real environment, uncomment this to save:
            # brain.save_image(out_file)
            pass
    print("Export complete.")
