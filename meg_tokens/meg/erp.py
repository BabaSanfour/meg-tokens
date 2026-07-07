import numpy as np
import mne
from typing import Optional
from meg_tokens.io import ensure_dir, save_array

def align_and_pad_epochs(
    stcs: list,
    df,
    align_to: str = 'go',
    tmin: float = -1.0,
    sfreq: float = 100.0,
    max_duration_samples: int = 400,
    cutoff_before_enter_ms: float = 300.0,
    min_rt_ms: float = 0.0
) -> list:
    """
    Slices and aligns source estimate epochs relative to Go and Enter Target triggers.
    For 'go' alignment, it truncates trial waveforms at (tEnterTarget - cutoff_before_enter_ms)
    and pads with NaNs up to max_duration_samples to prevent motor artifact contamination.
    
    Args:
        stcs: List of mne.SourceEstimate or mne.VectorSourceEstimate objects (one per epoch).
        df: Pandas DataFrame containing trial events ('tGO', 'tEnterTarget', etc.).
        align_to: Alignment trigger ('go' or 'enter').
        tmin: Epoch start time in seconds relative to Go (default -1.0s).
        sfreq: Sampling frequency in Hz of the downsampled STC data (default 100.0 Hz).
        max_duration_samples: Desired epoch duration in samples (default 400 samples = 4.0s).
        cutoff_before_enter_ms: Ms to cut off before the Enter Target trigger (default 300.0 ms).
        min_rt_ms: Minimum reaction time in ms to include a trial (default 0.0 ms).
        
    Returns:
        List of aligned and padded numpy arrays, or None for skipped trials.
    """
    if len(stcs) != len(df):
        raise ValueError(f"Length mismatch: {len(stcs)} STCs but behavioral dataframe has {len(df)} rows.")
        
    aligned_list = []
    
    for idx, stc in enumerate(stcs):
        # Extract trial parameters
        t_go = df.iloc[idx]['tGO']
        t_enter_target = df.iloc[idx]['tEnterTarget']
        
        t_enter = t_enter_target - t_go
        
        # Check RT constraint
        if t_enter <= min_rt_ms:
            aligned_list.append(None)
            continue
            
        data = stc.data  # shape: (n_vertices, n_times) or (3, n_vertices, n_times)
        is_vector = isinstance(stc, mne.VectorSourceEstimate)
        
        if align_to == 'go':
            # Cutoff time relative to epoch start
            cutoff_time_ms = -tmin * 1000.0 + t_enter - cutoff_before_enter_ms
            cutoff_sample = int(np.round(cutoff_time_ms * 0.001 * sfreq))
            
            # If the trial decision time is too long, skip it
            if cutoff_sample > max_duration_samples:
                aligned_list.append(None)
                continue
                
            # Perform alignment slicing and NaN padding
            if is_vector:
                # MNE VectorSourceEstimate.data has shape (n_vertices, 3, n_times)
                sliced = data[:, :, :cutoff_sample]
                padding = np.full((data.shape[0], 3, max_duration_samples - cutoff_sample), np.nan)
                padded = np.concatenate((sliced, padding), axis=2)
            else:
                # MNE SourceEstimate.data has shape (n_vertices, n_times)
                sliced = data[:, :cutoff_sample]
                padding = np.full((data.shape[0], max_duration_samples - cutoff_sample), np.nan)
                padded = np.concatenate((sliced, padding), axis=1)
                
            aligned_list.append(padded)
            
        elif align_to == 'enter':
            # Enter alignment: simply keep the full epoch duration without padding
            aligned_list.append(data)
            
        else:
            raise ValueError(f"Unknown alignment: {align_to}. Must be 'go' or 'enter'.")
            
    return aligned_list

def parcellate_source_estimates(
    stc: mne.SourceEstimate,
    subjects_dir: str,
    subject: str = 'fsaverage',
    parc: str = 'HCPMMP1',
    hemi: str = 'both',
    source_space: Optional[mne.SourceSpaces] = None,
    mode: str = 'mean'
) -> tuple:
    """
    Parcellates a source estimate into region-of-interest (ROI) labels using an annotation.
    Computes the spatial mean time series for each ROI label.
    
    Note: Legacy scripts prototyped parcellation using both the `Schaefer2018_400Parcels_17Networks_order` 
    atlas and the Destrieux `aparc.a2009s` atlas. You can specify parc='Schaefer2018...' or parc='aparc.a2009s' 
    to replicate those exact ROI groupings.
    
    Args:
        stc: An mne.SourceEstimate or mne.VectorSourceEstimate object.
        subjects_dir: Path to FreeSurfer subjects directory containing the subject labels.
        subject: Subject template name (default 'fsaverage').
        parc: Brain annotation parcellation atlas (e.g., 'HCPMMP1', 'aparc.a2009s').
        hemi: Hemisphere to parcellate ('left', 'right', or 'both').
        source_space: Optional MNE SourceSpaces object. If supplied, MNE's
            extract_label_time_course is used directly.
        mode: Extraction mode passed to MNE when source_space is supplied.
        
    Returns:
        tuple (label_names, parcellated_data)
            label_names: List of label name strings.
            parcellated_data: Numpy array of shape (n_labels, n_times) or (3, n_labels, n_times) for Vector STC.
    """
    # Load labels from annotation
    labels = []
    if hemi in ('left', 'both'):
        lh_labels = mne.read_labels_from_annot(subject, parc=parc, hemi='lh', subjects_dir=subjects_dir)
        # Exclude unknown/medial wall if they are first
        if lh_labels and (lh_labels[0].name.startswith('unknown') or 'Medial_wall' in lh_labels[0].name):
            lh_labels = lh_labels[1:]
        labels.extend(lh_labels)
        
    if hemi in ('right', 'both'):
        rh_labels = mne.read_labels_from_annot(subject, parc=parc, hemi='rh', subjects_dir=subjects_dir)
        if rh_labels and (rh_labels[0].name.startswith('unknown') or 'Medial_wall' in rh_labels[0].name):
            rh_labels = rh_labels[1:]
        labels.extend(rh_labels)

    if source_space is not None:
        label_tcs = mne.extract_label_time_course(
            [stc],
            labels,
            source_space,
            mode=mode,
            allow_empty=True,
            return_generator=False,
            verbose=False,
        )
        label_tcs = np.asarray(label_tcs)
        if label_tcs.shape[0] != 1:
            raise ValueError(f"Expected one source estimate after parcellation, got shape {label_tcs.shape}")
        return [label.name for label in labels], label_tcs[0]
        
    is_vector = stc.data.ndim == 3
    n_times = stc.data.shape[-1]
    
    parcellated = []
    label_names = []
    
    for label in labels:
        try:
            stc_in_label = stc.in_label(label)
        except Exception:
            # If label contains no vertices in this source estimate, pad with zeros or NaNs
            if is_vector:
                parcellated.append(np.zeros((3, n_times)))
            else:
                parcellated.append(np.zeros(n_times))
            label_names.append(label.name)
            continue
            
        label_data = stc_in_label.data
        if label_data.shape[0] == 0:  # No vertices
            if is_vector:
                mean_val = np.zeros((3, n_times))
            else:
                mean_val = np.zeros(n_times)
        else:
            # For both SourceEstimate and VectorSourceEstimate, axis 0 represents vertices
            mean_val = np.mean(label_data, axis=0)
                
        parcellated.append(mean_val)
        label_names.append(label.name)
        
    # If VectorSourceEstimate, transpose array from (n_labels, 3, n_times) to (3, n_labels, n_times)
    # to maintain standard Vector STC compatibility.
    parc_arr = np.array(parcellated)
    if is_vector and parc_arr.ndim == 3:
        parc_arr = np.transpose(parc_arr, (1, 0, 2))
        
    return label_names, parc_arr


def _source_group_slices(vertices: list[np.ndarray]) -> list[slice]:
    slices = []
    start = 0
    for group in vertices:
        stop = start + len(group)
        slices.append(slice(start, stop))
        start = stop
    return slices


def _source_group_prefix(vertices: list[np.ndarray], group_index: int) -> str:
    if len(vertices) == 1:
        return "vol"
    if group_index == 0:
        return "lh"
    if group_index == 1:
        return "rh"
    return f"vol{group_index - 1:02d}"


def _volume_group_indices(stc) -> list[int]:
    class_name = stc.__class__.__name__.lower()
    vertices = list(stc.vertices)
    if class_name.startswith("vol"):
        return list(range(len(vertices)))
    if "mixed" in class_name and len(vertices) > 2:
        return list(range(2, len(vertices)))
    raise ValueError(
        f"Source estimate type {stc.__class__.__name__} does not contain volume source groups"
    )


def source_feature_group_indices(stc, feature_space: str) -> list[int]:
    """Return vertex-group indices for an all-source or volume feature export."""
    vertices = list(stc.vertices)
    if feature_space == "all_source":
        return list(range(len(vertices)))
    if feature_space == "volume":
        return _volume_group_indices(stc)
    raise ValueError("feature_space must be 'all_source' or 'volume'")


def source_feature_labels(stc, feature_space: str) -> list[str]:
    """Build stable source-coordinate labels for non-parcellated exports."""
    vertices = list(stc.vertices)
    labels = []
    for group_index in source_feature_group_indices(stc, feature_space):
        prefix = _source_group_prefix(vertices, group_index)
        for vertex in np.asarray(vertices[group_index]).tolist():
            labels.append(f"{prefix}-{int(vertex)}")
    return labels


def select_source_feature_data(stc, feature_space: str) -> tuple[list[str], np.ndarray]:
    """Return source-level data for all-source or volume ERP exports.

    The returned array keeps MNE's source-major orientation:
    ``source x time`` for scalar estimates and ``source x orientation x time``
    for vector estimates.
    """
    data = np.asarray(stc.data)
    vertices = list(stc.vertices)
    group_slices = _source_group_slices(vertices)
    group_indices = source_feature_group_indices(stc, feature_space)
    row_indices = np.concatenate([
        np.arange(group_slices[group].start, group_slices[group].stop)
        for group in group_indices
    ])
    labels = source_feature_labels(stc, feature_space)
    return labels, np.take(data, row_indices, axis=0)


def export_neural_space(
    data: np.ndarray,
    label_names: list,
    output_dir: str,
    file_prefix: str,
    format: str = 'npy',
    metadata: Optional[dict] = None,
):
    """
    Export parcellated neural-space matrices as `.npy` plus JSON sidecar.
    
    Args:
        data: Parcellated data array of shape (n_labels, n_times) or (3, n_labels, n_times).
        label_names: List of label name strings.
        output_dir: Output directory to write files.
        file_prefix: Prefix filename.
        format: Export format. Only 'npy' is supported for new pipeline outputs.
        metadata: Optional metadata stored in the JSON sidecar.
    """
    if format != 'npy':
        raise ValueError("Only format='npy' is supported for new neural-space exports")

    out_dir = ensure_dir(output_dir)
    npy_path = out_dir / f"{file_prefix}.npy"
    dims = ("label", "time") if data.ndim == 2 else ("component", "label", "time")
    return save_array(
        npy_path,
        data,
        dims=dims,
        coords={"label": label_names},
        metadata=metadata or {},
    )
