import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mne

# Set seaborn defaults for beautiful modern plots
sns.set_theme(style="whitegrid", context="paper", palette="deep")

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
        _, ax = plt.subplots(figsize=(8, 8))
        
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
    
    ax.set_title(f"{title}\n(r = {r_val:.3f})")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    sns.despine(ax=ax)
    return ax

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
        _, ax = plt.subplots(figsize=(10, 6))
        
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
        _, ax = plt.subplots(figsize=(6, 6))
        
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
    
    ax.set_title(title)
    return ax
