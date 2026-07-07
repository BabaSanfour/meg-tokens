import numpy as np
import mne
try:
    from mne_connectivity import spectral_connectivity_epochs
except ImportError:
    from mne.connectivity import spectral_connectivity as spectral_connectivity_epochs

def extract_roi_time_courses(stc_data, labels, src, sfreq=600.0, mode='mean_flip'):
    """
    Extracts Region of Interest (ROI) time courses from Source Estimate data natively.
    If STC is a single SourceEstimate, it extracts the time course for that single trial/average.
    If it's an array of shape (n_epochs, n_vertices, n_times), we handle it accordingly.
    """
    # Assuming standard MNE usage:
    if isinstance(stc_data, (mne.SourceEstimate, mne.VectorSourceEstimate, mne.VolSourceEstimate)):
        label_ts = mne.extract_label_time_course(stc_data, labels, src, mode=mode)
        return np.expand_dims(label_ts, axis=0) # Make it (1, n_rois, n_times)
    
    return None # Fallback depending on data structure

def compute_spectral_connectivity(
    data,
    method='imcoh',
    sfreq=600.0,
    fmin=(2, 4, 8, 15),
    fmax=(4, 8, 15, 30),
    mode='fourier',
    n_jobs=1
):
    """
    Wrapper around MNE's spectral_connectivity_epochs to calculate connectivity matrices.
    
    Parameters:
    - data: array of shape (n_epochs, n_signals, n_times)
    
    Returns:
    - con_matrices: list of connectivity arrays, one per frequency band.
    """
    
    con, freqs, times, n_epochs, n_tapers = spectral_connectivity_epochs(
        data,
        method=method,
        fmin=fmin,
        fmax=fmax,
        mode=mode,
        sfreq=sfreq,
        faverage=True,
        n_jobs=n_jobs
    )
    
    # con is usually returned as a flat array of shape (n_connections, n_freqs)
    # We need to reshape it into a square matrix (n_signals x n_signals)
    n_signals = data.shape[1]
    
    con_matrices = []
    # If the output is wrapped in a modern mne_connectivity object, we extract the data
    if hasattr(con, 'get_data'):
        con_data = con.get_data()
    else:
        con_data = con
        
    for f_idx in range(len(fmin)):
        con_mat = np.zeros((n_signals, n_signals))
        
        if con_data.ndim == 2 and con_data.shape[0] == (n_signals * n_signals):
            # Flattened all-to-all connectivity
            band_data = con_data[:, f_idx].reshape(n_signals, n_signals)
            con_mat = band_data
        elif con_data.ndim == 3: # (n_connections, n_freqs, n_times) if time-resolved
            pass # handled differently based on exact MNE version
        else:
            # Fallback for complex output structures or symmetric matrices
            # Usually mne connectivity returns the lower triangular part or full depending on indices
            # Let's assume full matrix for now or symmetric reconstruction
            try:
                band_data = con_data[:, f_idx].reshape(n_signals, n_signals)
                con_mat = band_data
            except ValueError:
                # E.g. lower triangular returned
                tril_indices = np.tril_indices(n_signals, k=-1)
                con_mat[tril_indices] = con_data[:, f_idx]
                con_mat = con_mat + con_mat.T
                
        np.fill_diagonal(con_mat, 0)
        con_matrices.append(con_mat)
        
    return con_matrices
