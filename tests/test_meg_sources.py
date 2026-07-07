import pytest
import numpy as np
import mne
from unittest.mock import patch, MagicMock
from meg_tokens.meg.sources import (
    compute_noise_covariance,
    setup_bem_solution,
    setup_mixed_source_space,
    compute_forward_solution,
    build_inverse_operator,
    apply_inverse_operator
)

@patch('meg_tokens.meg.sources.os.path.exists')
@patch('meg_tokens.meg.sources.mne.io.read_raw_fif')
@patch('meg_tokens.meg.sources.mne.compute_raw_covariance')
def test_compute_noise_covariance(mock_compute_cov, mock_read_raw, mock_exists):
    mock_exists.return_value = True
    
    # Set up mock raw info
    mock_raw = MagicMock()
    mock_raw.info = {'ch_names': ['MEG001', 'MEG002']}
    mock_read_raw.return_value = mock_raw
    
    mock_cov = MagicMock()
    mock_compute_cov.return_value = mock_cov
    
    res = compute_noise_covariance('mock_path.fif', ch_names=['MEG001'])
    
    mock_read_raw.assert_called_once_with('mock_path.fif', preload=True)
    mock_raw.pick_channels.assert_called_once_with(['MEG001'])
    mock_compute_cov.assert_called_once()
    assert res == mock_cov

@patch('meg_tokens.meg.sources.mne.make_bem_model')
@patch('meg_tokens.meg.sources.mne.make_bem_solution')
def test_setup_bem_solution(mock_make_bem_sol, mock_make_bem_model):
    mock_model = MagicMock()
    mock_make_bem_model.return_value = mock_model
    
    mock_sol = MagicMock()
    mock_make_bem_sol.return_value = mock_sol
    
    res = setup_bem_solution('H1', 'mock_subjects_dir')
    
    mock_make_bem_model.assert_called_once_with(
        subject='H1', ico=4, conductivity=(0.3,), subjects_dir='mock_subjects_dir'
    )
    mock_make_bem_sol.assert_called_once_with(mock_model)
    assert res == mock_sol

@patch('meg_tokens.meg.sources.mne.setup_source_space')
@patch('meg_tokens.meg.sources.mne.setup_volume_source_space')
def test_setup_mixed_source_space(mock_setup_vol, mock_setup_surf):
    mock_surf_src = MagicMock()
    mock_setup_surf.return_value = mock_surf_src
    
    mock_vol_src = MagicMock()
    mock_setup_vol.return_value = mock_vol_src
    
    # 1. Test surface-only setup
    res = setup_mixed_source_space('H1', 'mock_subjects_dir', volume_labels=None)
    mock_setup_surf.assert_called_once_with(
        'H1', spacing='oct6', add_dist=False, subjects_dir='mock_subjects_dir'
    )
    assert res == mock_surf_src
    
    # 2. Test mixed surface + volume setup
    mock_setup_surf.reset_mock()
    res_mixed = setup_mixed_source_space(
        'H1', 'mock_subjects_dir', spacing='oct6', volume_labels=['Left-Putamen']
    )
    mock_setup_vol.assert_called_once_with(
        'H1', mri='aseg.mgz', pos=5.0, bem=None,
        volume_label=['Left-Putamen'], subjects_dir='mock_subjects_dir',
        add_interpolator=True, verbose=False
    )
    mock_surf_src.__iadd__.assert_called_once_with(mock_vol_src)

@patch('meg_tokens.meg.sources.mne.make_forward_solution')
def test_compute_forward_solution(mock_make_fwd):
    mock_info = MagicMock()
    mock_src = MagicMock()
    mock_bem = MagicMock()
    mock_fwd = MagicMock()
    mock_make_fwd.return_value = mock_fwd
    
    res = compute_forward_solution(mock_info, 'mock_trans.fif', mock_src, mock_bem)
    
    mock_make_fwd.assert_called_once_with(
        mock_info, trans='mock_trans.fif', src=mock_src, bem=mock_bem,
        mindist=5.0, meg=True, eeg=False
    )
    assert res == mock_fwd

@patch('meg_tokens.meg.sources.make_inverse_operator')
def test_build_inverse_operator(mock_make_inv):
    mock_info = MagicMock()
    mock_fwd = MagicMock()
    mock_cov = MagicMock()
    mock_inv = MagicMock()
    mock_make_inv.return_value = mock_inv
    
    res = build_inverse_operator(mock_info, mock_fwd, mock_cov)
    
    mock_make_inv.assert_called_once_with(
        mock_info, forward=mock_fwd, noise_cov=mock_cov,
        loose=dict(surface=0.2, volume=1.0), depth=None, verbose=True
    )
    assert res == mock_inv

@patch('meg_tokens.meg.sources.apply_inverse_epochs')
def test_apply_inverse_operator(mock_apply_inv):
    mock_epochs = MagicMock()
    mock_inv = MagicMock()
    mock_stc = MagicMock()
    mock_apply_inv.return_value = mock_stc
    
    res = apply_inverse_operator(mock_epochs, mock_inv, method='dSPM', snr=1.0)
    
    mock_apply_inv.assert_called_once_with(
        mock_epochs, inverse_operator=mock_inv, lambda2=1.0, method='dSPM', pick_ori=None
    )
    assert res == mock_stc
