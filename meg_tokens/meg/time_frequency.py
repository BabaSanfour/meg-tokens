"""Time-frequency, Hilbert, and band-power analysis helpers."""

import numpy as np
import mne
from scipy.signal import hilbert

DEFAULT_BANDS = {
    'delta': (2, 4),
    'theta': (4, 8),
    'alpha': (8, 15),
    'beta': (15, 30),
    'gamma_low': (30, 60),
    'gamma_high': (60, 90)
}

SOURCE_ESTIMATE_TYPES = (
    mne.SourceEstimate,
    mne.VectorSourceEstimate,
    mne.VolSourceEstimate,
    mne.VolVectorSourceEstimate,
    mne.MixedSourceEstimate,
    mne.MixedVectorSourceEstimate,
)


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


def compute_window_times(tmin: float, sfreq: float, n_times: int, width: int, step: int) -> np.ndarray:
    """Return center times for the sliding windows used by `_slide_window`."""
    if width <= 0:
        raise ValueError("width must be a positive integer")
    if step <= 0:
        raise ValueError("step must be a positive integer")
    if n_times <= width:
        center = (n_times - 1) / 2.0
        return np.array([tmin + center / sfreq])
    starts = np.arange(0, n_times - width + 1, step)
    centers = starts + ((width - 1) / 2.0)
    return tmin + centers / sfreq


def _hilbert_power_array(
    raw_data: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
    n_jobs: int = 1,
) -> np.ndarray:
    """Filter data along the last axis and return squared Hilbert envelope."""
    orig_shape = raw_data.shape
    flat = raw_data.astype(np.float64, copy=False).reshape(-1, orig_shape[-1])
    filtered = mne.filter.filter_data(
        flat,
        sfreq=sfreq,
        l_freq=fmin,
        h_freq=fmax,
        n_jobs=n_jobs,
        verbose=False,
    )
    envelope = np.abs(hilbert(filtered, axis=-1)) ** 2
    return envelope.reshape(orig_shape)


def compute_hilbert_band_features(
    data: np.ndarray,
    sfreq: float,
    freq_bands: dict[str, tuple[float, float]] | None = None,
    *,
    features: tuple[str, ...] | list[str] = ("amplitude", "power", "phase", "sigfilt"),
    n_jobs: int = 1,
) -> dict[str, dict[str, np.ndarray]]:
    """Return Hilbert features for each frequency band.

    Parameters
    ----------
    data
        Numeric array with time on the last axis. The input shape is preserved
        in every output array.
    sfreq
        Sampling rate in Hz.
    freq_bands
        Mapping from band name to ``(fmin, fmax)`` in Hz.
    features
        Any of ``amplitude``, ``power``, ``phase``, and ``sigfilt``. ``sigfilt``
        is the band-filtered signal and mirrors the old Brainpipe output name.

    Returns
    -------
    dict
        ``{band_name: {feature_name: array}}``.
    """
    if freq_bands is None:
        freq_bands = DEFAULT_BANDS

    requested = tuple("sigfilt" if item == "filtered" else item for item in features)
    valid = {"amplitude", "power", "phase", "sigfilt"}
    unknown = sorted(set(requested) - valid)
    if unknown:
        raise ValueError(f"Unknown Hilbert feature(s): {unknown}. Valid features: {sorted(valid)}")

    values = np.asarray(data, dtype=np.float64)
    if values.ndim < 2:
        raise ValueError(f"Hilbert feature input must have at least 2 dimensions, got {values.shape}")
    if values.shape[-1] < 2:
        raise ValueError("Hilbert feature input must contain at least two time samples")
    if sfreq <= 0:
        raise ValueError("sfreq must be positive")

    orig_shape = values.shape
    flat = values.reshape(-1, orig_shape[-1])
    out: dict[str, dict[str, np.ndarray]] = {}

    for band_name, (fmin, fmax) in freq_bands.items():
        if fmin <= 0 or fmax <= fmin:
            raise ValueError(f"Band {band_name!r} must satisfy 0 < fmin < fmax")
        filtered = mne.filter.filter_data(
            flat,
            sfreq=float(sfreq),
            l_freq=float(fmin),
            h_freq=float(fmax),
            n_jobs=n_jobs,
            verbose=False,
        )
        analytic = hilbert(filtered, axis=-1)
        amplitude = np.abs(analytic)
        band_out: dict[str, np.ndarray] = {}
        if "sigfilt" in requested:
            band_out["sigfilt"] = filtered.reshape(orig_shape)
        if "amplitude" in requested:
            band_out["amplitude"] = amplitude.reshape(orig_shape)
        if "power" in requested:
            band_out["power"] = (amplitude ** 2).reshape(orig_shape)
        if "phase" in requested:
            band_out["phase"] = np.angle(analytic).reshape(orig_shape)
        out[band_name] = band_out

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
              MNE source-estimate subclass.
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
    if width <= 0:
        raise ValueError("width must be a positive integer")
    if step <= 0:
        raise ValueError("step must be a positive integer")

    is_mne_stc = isinstance(data, SOURCE_ESTIMATE_TYPES)
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
        tmin = 0.0
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
            power = _hilbert_power_array(raw_data, sfreq, fmin, fmax, n_jobs=n_jobs)
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
                actual_width = min(width, n_times)
                new_tmin = tmin + ((actual_width - 1) / 2.0) * dt
                new_tstep = step * dt
                
                results[band_name] = data.__class__(
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
                actual_width = min(width, n_times)
                new_tmin = data.tmin + ((actual_width - 1) / 2.0) * dt
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
    """Compute PSD for an MNE Epochs object using modern MNE APIs."""
    if not isinstance(epochs, mne.BaseEpochs):
        raise TypeError("epochs must be an MNE Epochs object")
    if fmax <= fmin:
        raise ValueError("fmax must be greater than fmin")

    if method == 'welch':
        if n_fft <= 0:
            raise ValueError("n_fft must be positive")
        if n_overlap < 0:
            raise ValueError("n_overlap must be non-negative")
        n_times = len(epochs.times)
        fft = min(int(n_fft), n_times)
        overlap = min(int(n_overlap), max(0, fft - 1))
        spectrum = epochs.compute_psd(
            method="welch",
            fmin=fmin,
            fmax=fmax,
            n_fft=fft,
            n_overlap=overlap,
            n_jobs=n_jobs,
            verbose=False,
        )
    elif method == 'multitaper':
        spectrum = epochs.compute_psd(
            method="multitaper",
            fmin=fmin,
            fmax=fmax,
            n_jobs=n_jobs,
            verbose=False,
        )
    else:
        raise ValueError(f"Unknown PSD method: {method}")

    return np.asarray(spectrum.get_data()), np.asarray(spectrum.freqs)


def fit_specparam(
    freqs: np.ndarray,
    spectra: np.ndarray,
    peak_width_limits: tuple[float, float] = (1.0, 10.0),
    min_peak_height: float = 0.1,
    max_n_peaks: int = 6,
    peak_threshold: float = 2.0,
    freq_range: tuple[float, float] | None = None,
    aperiodic_mode: str = "fixed",
    n_jobs: int = 1,
    verbose: bool = False,
):
    """Fit specparam spectral parameterization to one or more spectra."""
    try:
        from specparam import SpectralGroupModel, SpectralModel
    except ImportError:
        raise ImportError("specparam is required for spectral parameterization.")

    freq_values = np.asarray(freqs, dtype=float)
    spectrum_values = np.asarray(spectra, dtype=float)
    if freq_values.ndim != 1:
        raise ValueError("freqs must be one-dimensional")
    if spectrum_values.shape[-1] != freq_values.size:
        raise ValueError("spectra last dimension must match freqs")
    if not np.all(np.isfinite(spectrum_values)):
        raise ValueError("specparam requires finite spectra")
    if np.any(spectrum_values <= 0):
        raise ValueError("specparam requires strictly positive spectra")

    if freq_range is None:
        freq_range = (float(freq_values[0]), float(freq_values[-1]))

    settings = dict(
        aperiodic_mode=aperiodic_mode,
        peak_width_limits=peak_width_limits,
        min_peak_height=min_peak_height,
        max_n_peaks=max_n_peaks,
        peak_threshold=peak_threshold,
        verbose=verbose,
    )

    if spectrum_values.ndim == 1:
        model = SpectralModel(**settings)
        model.fit(freq_values, spectrum_values, list(freq_range))
        return model
    if spectrum_values.ndim == 2:
        model = SpectralGroupModel(**settings)
        model.fit(freq_values, spectrum_values, list(freq_range), n_jobs=n_jobs)
        return model
    if spectrum_values.ndim == 3:
        leading_shape = spectrum_values.shape[:-1]
        flat = spectrum_values.reshape(-1, spectrum_values.shape[-1])
        model = SpectralGroupModel(**settings)
        model.fit(freq_values, flat, list(freq_range), n_jobs=n_jobs)
        model.input_shape = leading_shape
        return model
    raise ValueError("spectra must be 1D, 2D, or 3D")


def fit_fooof(
    freqs: np.ndarray,
    spectra: np.ndarray,
    peak_width_limits: tuple[float, float] = (1.0, 10.0),
    min_peak_height: float = 0.1,
    max_n_peaks: int = 6,
    peak_threshold: float = 2.0,
    freq_range: tuple[float, float] | None = None,
):
    """Compatibility wrapper around :func:`fit_specparam`."""
    return fit_specparam(
        freqs,
        spectra,
        peak_width_limits=peak_width_limits,
        min_peak_height=min_peak_height,
        max_n_peaks=max_n_peaks,
        peak_threshold=peak_threshold,
        freq_range=freq_range,
    )
