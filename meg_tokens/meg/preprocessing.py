import os
import numpy as np
import mne
from mne.io import read_raw_ctf
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs

def load_and_filter_raw(
    raw_path: str,
    notch_freqs: np.ndarray = None,
    low_pass: float = 150.0,
    high_pass: float = 0.5,
    preload: bool = True
) -> mne.io.Raw:
    """
    Loads raw CTF dataset and applies notch and bandpass filters.
    
    Args:
        raw_path: Path to the raw CTF dataset (.ds directory).
        notch_freqs: Array of frequencies to notch filter. If None, defaults to 60Hz and harmonics.
        low_pass: Low-pass filter cutoff frequency in Hz.
        high_pass: High-pass filter cutoff frequency in Hz.
        preload: If True, preload data into memory.
        
    Returns:
        Filtered mne.io.Raw object.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset path does not exist: {raw_path}")
        
    raw = read_raw_ctf(raw_path, preload=preload)
    picks = mne.pick_types(raw.info, meg=True, eeg=True, exclude='bads')
    
    # Apply notch filter
    if notch_freqs is None:
        # Default to 60Hz line frequency harmonics up to Nyquist frequency
        sfreq = raw.info['sfreq']
        notch_freqs = np.arange(60, sfreq / 2, 60)
        
    raw.notch_filter(notch_freqs, picks=picks, filter_length='auto', phase='zero')
    
    # Apply bandpass filters
    raw.filter(high_pass, low_pass, picks=picks, fir_design='firwin')
    
    return raw

def run_ica_rejection(
    raw: mne.io.Raw,
    n_components: int = 20,
    eog_ch: str = 'EEG057',
    ecg_ch: str = 'EEG059',
    reject_threshold: dict = None,
    random_state: int = 0
) -> ICA:
    """
    Fits ICA on raw data and automatically finds and excludes EOG/ECG components.
    
    Args:
        raw: Raw MNE data object.
        n_components: Number of ICA components to fit.
        eog_ch: EOG channel name.
        ecg_ch: ECG channel name.
        reject_threshold: Threshold dictionary for ICA fitting rejection.
        random_state: Random state for reproducible ICA decomposition.
        
    Returns:
        Fit ICA object with excluded component indices populated.
    """
    # Fit ICA
    ica = ICA(n_components=n_components, random_state=random_state)
    ica.fit(raw, decim=3, reject=reject_threshold)
    
    # Find EOG artifacts
    try:
        eog_epochs = create_eog_epochs(raw, ch_name=eog_ch, reject=reject_threshold)
        eog_inds, _ = ica.find_bads_eog(eog_epochs, ch_name=eog_ch)
    except Exception as e:
        print(f"Warning: Failed to find EOG components using channel {eog_ch}: {e}")
        eog_inds = []
        
    # Find ECG artifacts
    try:
        ecg_epochs = create_ecg_epochs(raw, ch_name=ecg_ch, reject=reject_threshold)
        ecg_inds, _ = ica.find_bads_ecg(ecg_epochs, ch_name=ecg_ch)
    except Exception as e:
        print(f"Warning: Failed to find ECG components using channel {ecg_ch}: {e}")
        ecg_inds = []
        
    # Exclude identified components
    ica.exclude = list(set(eog_inds + ecg_inds))
    
    return ica

def convert_ctf_headshape_to_pos(eeg_path: str, pos_path: str):
    """
    Converts a raw CTF Polhemus head shape file (.eeg) into the standard MNE-Python 
    headshape points format (.pos) used for MRI/MEG coregistration.
    
    This function excludes standard fiducials (NZ, OG, OD) from the dense 
    headshape points, formats each coordinate line, and adds the total point count 
    header required by .pos files.
    
    Args:
        eeg_path: Path to the raw CTF headshape .eeg file.
        pos_path: Path to the target output .pos file.
    """
    if not os.path.exists(eeg_path):
        raise FileNotFoundError(f"Source headshape file does not exist: {eeg_path}")
        
    with open(eeg_path, 'r') as f:
        lines = f.readlines()
        
    formatted_points = []
    for index, line in enumerate(lines):
        clean_line = line.strip()
        if not clean_line:
            continue
        if clean_line.startswith('OD') or clean_line.startswith('OG') or clean_line.startswith('NZ'):
            continue
        else:
            # Format: {index}\t{coordinate_line}
            # Keep original newline since it's already in the raw line
            formatted_points.append(f"{index - 5}\t{line}")
            
    # Calculate head points count header
    total_lines = len(lines)
    header_count = total_lines - 11  # Replicating legacy alignment math
    
    if os.path.dirname(pos_path):
        os.makedirs(os.path.dirname(pos_path), exist_ok=True)
    with open(pos_path, 'w') as f:
        f.write(f"{header_count}\n")
        for point in formatted_points:
            f.write(point)

def realign_epochs(
    epochs: mne.Epochs,
    align_to: str = 'go',
    tmin_new: float = -1.0,
    tmax_new: float = 4.0,
    mean_rt: float = 0.0
) -> mne.Epochs:
    """
    Re-aligns and slices epochs trial-by-trial relative to a behavioral trigger
    stored in the epochs metadata ('tGO', 'tEnterTarget', or 'tTrialEnd').
    
    Args:
        epochs: mne.Epochs object with metadata populated.
        align_to: Event to align to ('go', 'enter', 'feedback', 'DT').
        tmin_new: New start time in seconds relative to the alignment event.
        tmax_new: New end time in seconds relative to the alignment event.
        mean_rt: Mean reaction time in milliseconds (subtracted if aligning to 'DT').
        
    Returns:
        A new mne.EpochsArray containing the re-aligned trials.
    """
    if epochs.metadata is None:
        raise ValueError("Epochs object must have metadata populated for re-alignment.")
        
    sfreq = epochs.info['sfreq']
    tmin = epochs.tmin # original tmin (negative)
    
    # Get original data shape: (n_epochs, n_channels, n_times)
    data = epochs.get_data()
    n_epochs, n_channels, _ = data.shape
    
    # Calculate target length in samples
    n_samples_new = int(round((tmax_new - tmin_new) * sfreq))
    realigned_data = np.zeros((n_epochs, n_channels, n_samples_new))
    
    metadata = epochs.metadata.copy()
    
    for i in range(n_epochs):
        row = metadata.iloc[i]
        
        # Calculate alignment offset relative to original Go trigger (time 0 in epoch)
        if align_to == 'go':
            offset_ms = 0.0
        elif align_to == 'enter':
            # tEnterTarget and tGO are in ms in metadata
            offset_ms = float(row['tEnterTarget'] - row['tGO'])
        elif align_to == 'feedback':
            offset_ms = float(row['tTrialEnd'] - row['tGO'])
        elif align_to == 'DT':
            # Align to Decision Time: shift target entry back by subject's mean RT
            offset_ms = float(row['tEnterTarget'] - row['tGO']) - mean_rt
        else:
            raise ValueError(f"Unknown alignment event: {align_to}")
            
        # Convert offset to samples from the start of the epoch
        # Start of epoch is at tmin seconds (which is negative)
        offset_samples = int(round((offset_ms / 1000.0 - tmin) * sfreq))
        
        # Slice window around the offset
        start_idx = offset_samples + int(round(tmin_new * sfreq))
        end_idx = start_idx + n_samples_new
        
        # Guard bounds
        if start_idx < 0 or end_idx > data.shape[2]:
            # Pad with zeros if slice is out of bounds
            trial_data = np.zeros((n_channels, n_samples_new))
            
            # Find overlap range
            src_start = max(0, start_idx)
            src_end = min(data.shape[2], end_idx)
            
            dest_start = src_start - start_idx
            dest_end = dest_start + (src_end - src_start)
            
            trial_data[:, dest_start:dest_end] = data[i, :, src_start:src_end]
        else:
            trial_data = data[i, :, start_idx:end_idx]
            
        realigned_data[i] = trial_data
        
    # Build new Info and EpochsArray
    info = epochs.info.copy()
    
    # Create EpochsArray
    realigned_epochs = mne.EpochsArray(
        realigned_data,
        info=info,
        tmin=tmin_new,
        metadata=metadata,
        verbose=False
    )
    
    return realigned_epochs


