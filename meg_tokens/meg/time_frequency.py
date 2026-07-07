"""
Time-frequency spectrograms and power analysis for MEG and source-space data.

Note on future additions (Stage 10 - Functional Connectivity):
- Alpha-Band Seed Connectivity was prototyped in legacy notebooks:
  `archive/replicated/DDM_scripts/scripts_new/Untitled10.ipynb` and `Untitled6.ipynb`
- Cross-Frequency Coupling (CFC) via `brainpipe` was prototyped in:
  `archive/replicated/DDM_analysis_scripts/Untitled.ipynb`
"""

import numpy as np
import mne

DEFAULT_BANDS = {
    'delta': (2, 4),
    'theta': (4, 8),
    'alpha': (8, 15),
    'beta': (15, 30),
    'gamma_low': (30, 60),
    'gamma_high': (60, 90)
}

def _slide_window(power_data: np.ndarray, width: int, step: int) -> np.ndarray:
    """
    Applies a sliding window average over the last dimension of power_data.
    
    Args:
        power_data: ndarray of shape (..., n_times)
        width: window size in samples
        step: step size in samples
        
    Returns:
        ndarray of shape (..., n_windows)
    """
    n_times = power_data.shape[-1]
    if n_times <= width:
        # If signal is shorter than window, average the entire signal
        return np.mean(power_data, axis=-1, keepdims=True)
        
    starts = list(range(0, n_times - width + 1, step))
    out_shape = list(power_data.shape[:-1]) + [len(starts)]
    out = np.zeros(out_shape, dtype=power_data.dtype)
    
    for idx, start in enumerate(starts):
        out[..., idx] = np.mean(power_data[..., start : start + width], axis=-1)
    return out

def compute_band_power(
    data,
    sfreq: float,
    freq_bands: dict = None,
    method: str = 'hilbert',
    width: int = 250,
    step: int = 25,
    n_jobs: int = 1,
    return_mne: bool = True,
    n_cycles: float = 7.0,
    time_bandwidth: float = 4.0
) -> dict:
    """
    Computes time-frequency power spectrograms across specified frequency bands.
    
    Args:
        data: input data. Can be a numpy array of shape (n_channels, n_times) or 
              (n_channels, n_trials, n_times), or MNE Epochs, SourceEstimate, or 
              VectorSourceEstimate.
        sfreq: Sampling frequency in Hz.
        freq_bands: Dict mapping band name (str) to tuple (fmin, fmax). If None, uses DEFAULT_BANDS.
        method: Power computation method. Options: 'hilbert', 'morlet', 'multitaper'.
        width: Sliding window width in samples.
        step: Sliding window step/stride in samples.
        n_jobs: Number of parallel jobs to use.
        return_mne: If True, returns MNE objects if input is an MNE object.
        n_cycles: Number of cycles for Morlet Wavelet/Multitaper (default 7.0).
        time_bandwidth: Time-bandwidth product for Multitaper (default 4.0).
        
    Returns:
        dict: mapping band_name to power output (numpy array or MNE object).
    """
    if freq_bands is None:
        freq_bands = DEFAULT_BANDS

    # Determine input type and extract raw numpy data array
    is_mne_stc = isinstance(data, (mne.SourceEstimate, mne.VectorSourceEstimate))
    is_mne_epochs = isinstance(data, mne.BaseEpochs)
    
    if is_mne_stc:
        raw_data = data.data  # shape: (n_vertices, n_times) or (n_vertices, 3, n_times)
        tmin, tstep = data.tmin, data.tstep
    elif is_mne_epochs:
        raw_data = data.get_data()  # shape: (n_epochs, n_channels, n_times)
        # Swap axes to be (n_channels, n_epochs, n_times) to treat epochs as trial dimension
        raw_data = np.swapaxes(raw_data, 0, 1)
        sfreq = data.info['sfreq']
    elif isinstance(data, np.ndarray):
        raw_data = data
    else:
        raise TypeError("Unsupported data type. Expected np.ndarray, mne.Epochs, or mne.SourceEstimate.")

    # Standardize data to 3D: (n_channels, n_trials, n_times)
    orig_ndim = raw_data.ndim
    if orig_ndim == 2:
        # (n_channels, n_times) -> (n_channels, 1, n_times)
        raw_data = raw_data[:, np.newaxis, :]
    elif orig_ndim == 3:
        pass
    else:
        raise ValueError(f"Input numpy array must be 2D or 3D, got shape {raw_data.shape}")

    n_channels, n_trials, n_times = raw_data.shape
    results = {}

    for band_name, (fmin, fmax) in freq_bands.items():
        if method == 'hilbert':
            if isinstance(data, mne.VectorSourceEstimate):
                raise TypeError(
                    "Hilbert method is not supported on VectorSourceEstimate "
                    "due to an indexing limitation in MNE-Python's native apply_hilbert. "
                    "This will be addressed in a future update."
                )
            if not (is_mne_stc or is_mne_epochs):
                raise TypeError("Hilbert method is only supported on MNE Epochs or SourceEstimate objects.")

            # Use native MNE objects to filter and apply hilbert
            mne_obj = data.copy()
            if hasattr(mne_obj, 'load_data') and not mne_obj.preload:
                mne_obj.load_data()
            # 1. Filter
            mne_obj.filter(fmin, fmax, n_jobs=n_jobs, verbose=False)
            # 2. Apply Hilbert envelope
            mne_obj.apply_hilbert(envelope=True, n_jobs=n_jobs, verbose=False)
            # 3. Get power (envelope squared)
            if is_mne_stc:
                power = mne_obj.data ** 2
                power = power[:, np.newaxis, :]
            else:
                power = mne_obj.get_data() ** 2
                # Swap axes to match (n_channels, n_epochs, n_times)
                power = np.swapaxes(power, 0, 1)
                
            # Apply sliding window average
            power_downsampled = _slide_window(power, width, step)
            
        elif method in ('morlet', 'multitaper'):
            # Define frequency points in the band
            freqs = np.linspace(fmin, fmax, num=max(3, int(fmax - fmin + 1) // 2))
            # MNE tfr_array functions expect shape: (n_epochs, n_channels, n_times)
            # So we swap axes of raw_data to: (n_trials, n_channels, n_times)
            tfr_input = np.swapaxes(raw_data, 0, 1)
            
            if method == 'morlet':
                tfr_out = mne.time_frequency.tfr_array_morlet(
                    tfr_input, sfreq, freqs,
                    n_cycles=n_cycles,
                    output='power',
                    n_jobs=n_jobs,
                    verbose=False
                )
            else:  # multitaper
                tfr_out = mne.time_frequency.tfr_array_multitaper(
                    tfr_input, sfreq, freqs,
                    n_cycles=n_cycles,
                    time_bandwidth=time_bandwidth,
                    output='power',
                    n_jobs=n_jobs,
                    verbose=False
                )
            # tfr_out shape: (n_epochs, n_channels, len(freqs), n_times)
            # Average over frequencies in the band (axis=2)
            power = np.mean(tfr_out, axis=2)  # shape: (n_epochs, n_channels, n_times)
            # Swap back to (n_channels, n_epochs, n_times)
            power = np.swapaxes(power, 0, 1)
            # Apply sliding window average
            power_downsampled = _slide_window(power, width, step)
            
        else:
            raise ValueError(f"Unknown power method '{method}'. Supported: 'hilbert', 'morlet', 'multitaper'")

        # Restore original dimension if it was 2D
        if orig_ndim == 2:
            power_downsampled = np.squeeze(power_downsampled, axis=1)

        # Convert back to MNE object if return_mne is True
        if return_mne:
            if is_mne_stc:
                # Calculate new tmin (center of the first window) and tstep
                dt = 1.0 / sfreq
                new_tmin = tmin + ((width - 1) / 2.0) * dt
                new_tstep = step * dt
                
                if isinstance(data, mne.VectorSourceEstimate):
                    results[band_name] = mne.VectorSourceEstimate(
                        power_downsampled,
                        vertices=data.vertices,
                        tmin=new_tmin,
                        tstep=new_tstep,
                        subject=data.subject
                    )
                else:
                    results[band_name] = mne.SourceEstimate(
                        power_downsampled,
                        vertices=data.vertices,
                        tmin=new_tmin,
                        tstep=new_tstep,
                        subject=data.subject
                    )
            elif is_mne_epochs:
                # Swap back to (n_epochs, n_channels, n_windows)
                epochs_power_data = np.swapaxes(power_downsampled, 0, 1)
                # Create info
                info = mne.create_info(
                    ch_names=data.ch_names,
                    sfreq=sfreq / step,
                    ch_types=data.get_channel_types()
                )
                dt = 1.0 / sfreq
                new_tmin = data.tmin + ((width - 1) / 2.0) * dt
                results[band_name] = mne.EpochsArray(
                    epochs_power_data,
                    info,
                    tmin=new_tmin,
                    verbose=False
                )
            else:
                results[band_name] = power_downsampled
        else:
            results[band_name] = power_downsampled

    return results

def rescale_baseline(
    power_data: np.ndarray,
    times: np.ndarray,
    baseline: tuple,
    method: str = 'percent'
) -> np.ndarray:
    """
    Applies baseline normalization / scaling to power data.
    
    Args:
        power_data: Power data array, shape (..., n_times).
        times: Time points array corresponding to the last axis of power_data.
        baseline: Tuple of (bmin, bmax) defining the baseline period in seconds.
                  If None, no rescaling is applied.
        method: Rescaling method. Options:
                'percent'    : (P - P_base) / P_base
                'ratio'      : P / P_base
                'logratio'   : 10 * log10(P / P_base)
                'zscore'     : (P - mean_base) / std_base
                'difference' : P - P_base
                
    Returns:
        ndarray: Rescaled power data of the same shape.
    """
    if baseline is None:
        return power_data.copy()
        
    bmin, bmax = baseline
    # Find indices corresponding to baseline range
    if bmin is None:
        bmin_idx = 0
    else:
        bmin_idx = np.where(times >= bmin)[0]
        bmin_idx = bmin_idx[0] if len(bmin_idx) > 0 else 0
        
    if bmax is None:
        bmax_idx = len(times)
    else:
        bmax_idx = np.where(times <= bmax)[0]
        bmax_idx = bmax_idx[-1] + 1 if len(bmax_idx) > 0 else len(times)
        
    if bmin_idx >= bmax_idx:
        raise ValueError(f"Invalid baseline range: {baseline}. Check times values.")
        
    # Extract baseline period
    baseline_data = power_data[..., bmin_idx:bmax_idx]
    
    # Compute baseline mean/std
    mean_base = np.mean(baseline_data, axis=-1, keepdims=True)
    std_base = np.std(baseline_data, axis=-1, keepdims=True)
    
    # Avoid division by zero
    mean_base = np.where(mean_base == 0, 1e-10, mean_base)
    std_base = np.where(std_base == 0, 1e-10, std_base)
    
    # Apply rescaling method
    if method == 'ratio':
        rescaled = power_data / mean_base
    elif method == 'percent':
        rescaled = (power_data - mean_base) / mean_base
    elif method == 'logratio':
        rescaled = 10.0 * np.log10(np.maximum(power_data, 1e-10) / mean_base)
    elif method == 'zscore':
        rescaled = (power_data - mean_base) / std_base
    elif method == 'difference':
        rescaled = power_data - mean_base
    else:
        raise ValueError(f"Unknown rescaling method '{method}'")
        
    return rescaled


def compute_psd(
    epochs: mne.BaseEpochs,
    fmin: float = 1.0,
    fmax: float = 100.0,
    method: str = 'welch',
    n_fft: int = 2048,
    n_overlap: int = 150,
    n_jobs: int = 1
) -> tuple:
    """
    Computes the Power Spectral Density (PSD) for MNE Epochs.
    
    Args:
        epochs: MNE Epochs object.
        fmin: Minimum frequency of interest.
        fmax: Maximum frequency of interest.
        method: 'welch' or 'multitaper'.
        n_fft: The length of FFT used for the Welch method.
        n_overlap: The number of points of overlap between segments for Welch.
        n_jobs: Number of parallel jobs.
        
    Returns:
        tuple: (psds, freqs)
            psds: array of shape (n_epochs, n_channels, n_freqs)
            freqs: array of frequencies
    """
    if method == 'welch':
        from mne.time_frequency import psd_welch
        psds, freqs = psd_welch(
            epochs, fmin=fmin, fmax=fmax, n_fft=n_fft, 
            n_overlap=n_overlap, n_jobs=n_jobs, average='median'
        )
    elif method == 'multitaper':
        from mne.time_frequency import psd_multitaper
        psds, freqs = psd_multitaper(
            epochs, fmin=fmin, fmax=fmax, n_jobs=n_jobs
        )
    else:
        raise ValueError(f"Unknown PSD method: {method}")
        
    return psds, freqs


def fit_fooof(
    freqs: np.ndarray,
    spectra: np.ndarray,
    peak_width_limits: list = [1, 10],
    min_peak_height: float = 0.1,
    max_n_peaks: int = 6,
    peak_threshold: float = 2.0,
    freq_range: list = None
):
    """
    Fits the FOOOF (Fitting Oscillations & One Over F) model to extract 
    aperiodic and periodic components of the power spectrum.
    
    Args:
        freqs: 1D array of frequencies.
        spectra: 1D, 2D (n_spectra, n_freqs), or 3D array of power spectra.
                 If 3D, it will be collapsed or processed iteratively depending on shape.
        peak_width_limits: Limits on possible peak widths.
        min_peak_height: Minimum absolute height of a peak.
        max_n_peaks: Maximum number of peaks to fit.
        peak_threshold: Relative threshold for detecting peaks.
        freq_range: Range of frequencies to fit, e.g. [1, 100].
        
    Returns:
        FOOOFGroup or FOOOF object with the fitted model.
    """
    try:
        from fooof import FOOOF, FOOOFGroup
        from fooof.objs import fit_fooof_3d
    except ImportError:
        raise ImportError("FOOOF is not installed. Please install it via 'pip install fooof'.")
        
    if freq_range is None:
        freq_range = [freqs[0], freqs[-1]]
        
    if spectra.ndim == 1:
        # Single spectrum
        fm = FOOOF(
            peak_width_limits=peak_width_limits, 
            min_peak_height=min_peak_height,
            max_n_peaks=max_n_peaks, 
            peak_threshold=peak_threshold
        )
        fm.fit(freqs, spectra, freq_range)
        return fm
        
    elif spectra.ndim == 2:
        # Group of spectra
        fg = FOOOFGroup(
            peak_width_limits=peak_width_limits, 
            min_peak_height=min_peak_height,
            max_n_peaks=max_n_peaks, 
            peak_threshold=peak_threshold
        )
        fg.fit(freqs, spectra, freq_range)
        return fg
        
    elif spectra.ndim == 3:
        # 3D array: (n_groups, n_spectra, n_freqs)
        fg = FOOOFGroup(
            peak_width_limits=peak_width_limits, 
            min_peak_height=min_peak_height,
            max_n_peaks=max_n_peaks, 
            peak_threshold=peak_threshold
        )
        fgs = fit_fooof_3d(fg, freqs, spectra, freq_range=freq_range)
        return fgs
    else:
        raise ValueError("Spectra must be 1D, 2D, or 3D.")
