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
import pandas as pd
from pathlib import Path
from typing import List, Optional
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs, write_inverse_operator

from meg_tokens.io import derivative_path, ensure_dir, save_sidecar, save_table
from meg_tokens.utils.batch_processor import normalize_subject_id
from meg_tokens.utils.epochs_builder import parse_run_label


def source_derivative_path(
    output_root: str,
    subject_id: str,
    *,
    suffix: str,
    extension: str,
    run_id: Optional[str] = None,
    condition: Optional[str] = None,
    description: Optional[str] = None,
    processing: Optional[str] = None,
    space: Optional[str] = None,
) -> Path:
    """Build a Stage 3 source derivative path."""
    subject = normalize_subject_id(subject_id)
    run = None
    inferred_condition = None
    if run_id is not None:
        run, inferred_condition = parse_run_label(run_id)
    condition = condition or inferred_condition
    desc_parts = []
    if condition:
        desc_parts.append(condition.lower())
    if description:
        desc_parts.append(description)
    desc = "-".join(desc_parts) if desc_parts else None
    return derivative_path(
        output_root,
        subject=subject,
        datatype="meg",
        task="tokens",
        run=run,
        processing=processing,
        space=space,
        description=desc,
        suffix=suffix,
        extension=extension,
    )


def save_noise_covariance(cov: mne.Covariance, output_root: str, subject_id: str) -> str:
    path = source_derivative_path(output_root, subject_id, suffix="cov", extension=".fif", description="noise")
    ensure_dir(path.parent)
    mne.write_cov(str(path), cov, overwrite=True)
    save_sidecar(path, {"format": "mne-covariance-fif", "stage": "source_reconstruction", "kind": "noise_covariance", "subject": normalize_subject_id(subject_id)})
    return str(path)


def save_bem_solution(bem: mne.bem.ConductorModel, output_root: str, subject_id: str) -> str:
    path = source_derivative_path(output_root, subject_id, suffix="bem", extension=".fif", description="singlelayer")
    ensure_dir(path.parent)
    mne.write_bem_solution(str(path), bem, overwrite=True)
    save_sidecar(path, {"format": "mne-bem-fif", "stage": "source_reconstruction", "kind": "bem_solution", "subject": normalize_subject_id(subject_id)})
    return str(path)


def save_source_space(
    src: mne.SourceSpaces,
    output_root: str,
    subject_id: str,
    spacing: str,
    *,
    volume_labels: Optional[list[str]] = None,
    volume_pos: Optional[float] = None,
) -> str:
    path = source_derivative_path(output_root, subject_id, suffix="src", extension=".fif", description=spacing, space="subject")
    ensure_dir(path.parent)
    mne.write_source_spaces(str(path), src, overwrite=True)
    save_sidecar(path, {
        "format": "mne-source-space-fif",
        "stage": "source_reconstruction",
        "kind": "source_space",
        "subject": normalize_subject_id(subject_id),
        "spacing": spacing,
        "volume_labels": volume_labels or [],
        "volume_pos_mm": volume_pos,
    })
    return str(path)


def save_forward_solution(fwd: mne.Forward, output_root: str, subject_id: str, run_id: str, condition: Optional[str], alignment: str) -> str:
    path = source_derivative_path(output_root, subject_id, suffix="fwd", extension=".fif", run_id=run_id, condition=condition, description=alignment)
    ensure_dir(path.parent)
    mne.write_forward_solution(str(path), fwd, overwrite=True)
    save_sidecar(path, {"format": "mne-forward-fif", "stage": "source_reconstruction", "kind": "forward_solution", "subject": normalize_subject_id(subject_id), "run": parse_run_label(run_id)[0], "condition": condition, "alignment": alignment})
    return str(path)


def save_inverse_operator(inverse_operator, output_root: str, subject_id: str, run_id: str, condition: Optional[str], alignment: str, method: str) -> str:
    path = source_derivative_path(output_root, subject_id, suffix="inv", extension=".fif", run_id=run_id, condition=condition, description=f"{alignment}-{method}")
    ensure_dir(path.parent)
    write_inverse_operator(str(path), inverse_operator, overwrite=True)
    save_sidecar(path, {"format": "mne-inverse-operator-fif", "stage": "source_reconstruction", "kind": "inverse_operator", "subject": normalize_subject_id(subject_id), "run": parse_run_label(run_id)[0], "condition": condition, "alignment": alignment, "method": method})
    return str(path)


def save_source_estimates(
    stcs: List[mne.SourceEstimate],
    output_root: str,
    subject_id: str,
    run_id: str,
    condition: Optional[str],
    alignment: str,
    method: str,
) -> str:
    """Save trial source estimates and a manifest consumed by later stages."""
    subject = normalize_subject_id(subject_id)
    run, inferred_condition = parse_run_label(run_id)
    condition = condition or inferred_condition
    manifest_path = source_derivative_path(
        output_root,
        subject,
        suffix="stcmanifest",
        extension=".tsv",
        run_id=run,
        condition=condition,
        description=f"{alignment}-{method}",
    )
    ensure_dir(manifest_path.parent)

    rows = []
    for trial_idx, stc in enumerate(stcs, start=1):
        base = source_derivative_path(
            output_root,
            subject,
            suffix=f"trial{trial_idx:04d}stc",
            extension="",
            run_id=run,
            condition=condition,
            description=f"{alignment}-{method}",
        )
        ensure_dir(base.parent)
        stc.save(str(base), ftype="stc", overwrite=True)
        save_sidecar(
            base,
            {
                "format": "mne-source-estimate",
                "stage": "source_reconstruction",
                "kind": "source_estimate",
                "subject": subject,
                "run": run,
                "condition": condition,
                "alignment": alignment,
                "method": method,
                "trial_index": trial_idx,
                "n_times": int(stc.data.shape[-1]),
            },
        )
        rows.append({
            "trial": trial_idx,
            "stc_base": str(base),
            "subject": subject,
            "run": run,
            "condition": condition,
            "alignment": alignment,
            "method": method,
        })

    save_table(
        manifest_path,
        pd.DataFrame(rows),
        metadata={
            "stage": "source_reconstruction",
            "kind": "source_estimate_manifest",
            "subject": subject,
            "run": run,
            "condition": condition,
            "alignment": alignment,
            "method": method,
            "n_trials": len(rows),
        },
    )
    return str(manifest_path)

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
