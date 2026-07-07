"""
Source reconstruction, fetching, and estimates for MEG data.

Note on legacy implementations:
- Brain-level visualization (visbrain/mne.viz) and advanced source estimate manipulation
  (e.g., morphing and regrouping sources) were heavily prototyped in legacy notebooks:
  `archive/replicated/DDM_scripts/scripts_new/Untitled1.ipynb` and `Untitled1-Copy1.ipynb`.
"""

import os
import numpy as np
import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs

def compute_noise_covariance(
    noise_raw_path: str,
    ch_names: list = None,
    method: str = 'shrunk',
    cv: int = 5,
    tmin: float = 0.0,
    tmax: float = 10.0,
    preload: bool = True
) -> mne.Covariance:
    """
    Computes noise covariance matrix from an empty room noise raw file.
    
    Args:
        noise_raw_path: Path to the raw empty room noise recording.
        ch_names: Optional channel names list to restrict covariance computation.
        method: Covariance estimator method (default 'shrunk').
        cv: Cross-validation folds (default 5).
        tmin: Start time for covariance calculation in seconds.
        tmax: End time for covariance calculation in seconds.
        preload: If True, preload raw file into memory.
        
    Returns:
        mne.Covariance matrix object.
    """
    if not os.path.exists(noise_raw_path):
        raise FileNotFoundError(f"Noise raw path does not exist: {noise_raw_path}")
        
    if noise_raw_path.endswith('.ds'):
        noise_raw = mne.io.read_raw_ctf(noise_raw_path, preload=preload)
    else:
        noise_raw = mne.io.read_raw_fif(noise_raw_path, preload=preload)
        
    if ch_names:
        # Filter channels to match selected channels
        common_chs = [ch for ch in ch_names if ch in noise_raw.info['ch_names']]
        noise_raw.pick_channels(common_chs)
        
    cov = mne.compute_raw_covariance(
        noise_raw,
        method=method,
        cv=cv,
        tmin=tmin,
        tmax=tmax
    )
    return cov

def setup_bem_solution(
    subject: str,
    subjects_dir: str,
    conductivity: tuple = (0.3,)
) -> mne.bem.ConductorModel:
    """
    Sets up a single-layer Boundary Element Model (BEM) solution.
    
    Args:
        subject: Subject identifier name.
        subjects_dir: Path to the FreeSurfer IRM / subjects directory.
        conductivity: Single-layer conductivity (default (0.3,)).
        
    Returns:
        mne.bem.ConductorModel solution.
    """
    model = mne.make_bem_model(
        subject=subject,
        ico=4,
        conductivity=conductivity,
        subjects_dir=subjects_dir
    )
    bem_sol = mne.make_bem_solution(model)
    return bem_sol

def setup_mixed_source_space(
    subject: str,
    subjects_dir: str,
    spacing: str = 'oct6',
    volume_labels: list = None,
    volume_pos: float = 5.0,
    bem: mne.bem.ConductorModel = None
) -> mne.SourceSpaces:
    """
    Sets up a mixed source space containing cortical surfaces and subcortical volumes.
    
    Args:
        subject: Subject identifier name.
        subjects_dir: Path to FreeSurfer subjects directory.
        spacing: Surface source space grid spacing (default 'oct6').
        volume_labels: List of subcortical anatomical segmentation labels from aseg.mgz.
        volume_pos: Grid spacing in mm for volume source space (default 5.0).
        bem: Conductor model (BEM) required if setup_volume_source_space needs BEM boundaries.
        
    Returns:
        mne.SourceSpaces object containing surface and volume components.
    """
    # 1. Surface source space
    src = mne.setup_source_space(
        subject,
        spacing=spacing,
        add_dist=False,
        subjects_dir=subjects_dir
    )
    
    # 2. Add subcortical volumes if requested
    if volume_labels:
        # Default subcortical labels if none provided:
        # Left-Accumbens-area, Left-Pallidum, Left-Caudate, Left-Putamen, etc.
        vol_src = mne.setup_volume_source_space(
            subject,
            mri='aseg.mgz',
            pos=volume_pos,
            bem=bem,
            volume_label=volume_labels,
            subjects_dir=subjects_dir,
            add_interpolator=True,
            verbose=False
        )
        src += vol_src
        
    return src

def compute_forward_solution(
    info: mne.Info,
    trans_path: str,
    src: mne.SourceSpaces,
    bem: mne.bem.ConductorModel,
    mindist: float = 5.0,
    meg: bool = True,
    eeg: bool = False
) -> mne.Forward:
    """
    Computes a forward solution leadfield matrix.
    
    Args:
        info: MNE recording info object.
        trans_path: Path to coordinate trans file (-trans.fif).
        src: mne.SourceSpaces object.
        bem: BEM conductor solution model.
        mindist: Exclude sources closer than this distance (mm) to inner skull boundary.
        meg: If True, include MEG channels.
        eeg: If True, include EEG channels.
        
    Returns:
        mne.Forward solution object.
    """
    fwd = mne.make_forward_solution(
        info,
        trans=trans_path,
        src=src,
        bem=bem,
        mindist=mindist,
        meg=meg,
        eeg=eeg
    )
    return fwd

def build_inverse_operator(
    info: mne.Info,
    fwd: mne.Forward,
    cov: mne.Covariance,
    loose: dict = None,
    depth: float = None
) -> mne.minimum_norm.InverseOperator:
    """
    Compiles the minimum-norm inverse operator.
    
    Note: DICS Beamforming is a planned alternative frequency-domain source localization
    method that may be added alongside this minimum-norm approach in the future.
    
    Args:
        info: MNE recording info object.
        fwd: Computed forward solution model.
        cov: Measured noise covariance matrix.
        loose: Dict defining loose orientation constraints (default Dict(surface=0.2, volume=1.0)).
        depth: Depth weighting parameter (default None).
        
    Returns:
        mne.minimum_norm.InverseOperator object.
    """
    if loose is None:
        loose = dict(surface=0.2, volume=1.0)
        
    inverse_operator = make_inverse_operator(
        info,
        forward=fwd,
        noise_cov=cov,
        loose=loose,
        depth=depth,
        verbose=True
    )
    return inverse_operator

def apply_inverse_operator(
    epochs: mne.Epochs,
    inverse_operator: mne.minimum_norm.InverseOperator,
    method: str = 'dSPM',
    snr: float = 1.0
) -> list:
    """
    Applies the inverse operator to epochs to obtain trial-by-trial SourceEstimates.
    
    Args:
        epochs: mne.Epochs object.
        inverse_operator: Computed inverse operator object.
        method: Source estimation method ('dSPM', 'sLORETA', 'MNE', etc. Default 'dSPM').
        snr: Signal-to-noise ratio parameter (default 1.0).
        
    Returns:
        List of mne.SourceEstimate objects (one per epoch).
    """
    lambda2 = 1.0 / (snr ** 2)
    stc = apply_inverse_epochs(
        epochs,
        inverse_operator=inverse_operator,
        lambda2=lambda2,
        method=method,
        pick_ori=None
    )
    return stc

def morph_source_estimates(
    stc: mne.SourceEstimate,
    subject_from: str,
    subjects_dir: str,
    fname_src_to: str
) -> mne.SourceEstimate:
    """
    Morphs a subject-specific SourceEstimate into a standard target template space (e.g. fsaverage).
    
    Args:
        stc: Single-trial or averaged SourceEstimate object.
        subject_from: Subject identifier name.
        subjects_dir: Path to FreeSurfer subjects directory.
        fname_src_to: Path to standard target source space template file (-src.fif).
        
    Returns:
        Morphed source estimate in target template space.
    """
    src_to = mne.read_source_spaces(fname_src_to)
    
    # Compute source morph
    morph = mne.compute_source_morph(
        stc,
        subject_from=subject_from,
        subjects_dir=subjects_dir,
        src_to=src_to,
        verbose=False
    )
    
    stc_morphed = morph.apply(stc)
    return stc_morphed
