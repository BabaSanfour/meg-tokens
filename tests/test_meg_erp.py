import pytest
import numpy as np
import pandas as pd
import mne
import tempfile
import os
import scipy.io as sio
from unittest.mock import patch, MagicMock

from meg_tokens.meg.erp import (
    align_and_pad_epochs,
    parcellate_source_estimates,
    export_neural_space
)

def test_align_and_pad_epochs():
    # Set up mock behavioral DataFrame
    df = pd.DataFrame({
        'tGO': [1000.0, 1000.0, 1000.0],
        'tEnterTarget': [2500.0, 1500.0, 5000.0]  # RT = 1500ms, 500ms, 4000ms
    })
    
    # 1. Standard SourceEstimates (2D data: n_vertices=10, n_times=500)
    # The epochs will go from -1.0s to 4.0s (500 samples at 100 Hz)
    sfreq = 100.0
    tmin = -1.0
    n_vertices = 10
    n_times = 500
    
    stcs = []
    for _ in range(3):
        data = np.ones((n_vertices, n_times))
        vertices = [np.arange(5), np.arange(5, 10)]
        stc = mne.SourceEstimate(data, vertices=vertices, tmin=tmin, tstep=1.0/sfreq)
        stcs.append(stc)
        
    # Align to Go
    # Trial 1 (RT=1500ms): cutoff_time_ms = 1000 + 1500 - 300 = 2200ms -> 220 samples.
    # Trial 2 (RT=500ms): cutoff_time_ms = 1000 + 500 - 300 = 1200ms -> 120 samples.
    # Trial 3 (RT=4000ms): cutoff_sample = 1000 + 4000 - 300 = 4700ms -> 470 samples.
    # Trial 3 should be skipped because 470 > max_duration_samples (400).
    aligned = align_and_pad_epochs(
        stcs, df, align_to='go', tmin=tmin, sfreq=sfreq,
        max_duration_samples=400, cutoff_before_enter_ms=300.0
    )
    
    assert len(aligned) == 3
    assert aligned[0].shape == (10, 400)
    assert aligned[1].shape == (10, 400)
    assert aligned[2] is None
    
    # Check that padding contains NaNs
    assert not np.any(np.isnan(aligned[0][:, :220]))
    assert np.all(np.isnan(aligned[0][:, 220:]))
    assert not np.any(np.isnan(aligned[1][:, :120]))
    assert np.all(np.isnan(aligned[1][:, 120:]))

def test_align_and_pad_epochs_vector():
    df = pd.DataFrame({
        'tGO': [1000.0],
        'tEnterTarget': [2500.0]
    })
    sfreq = 100.0
    tmin = -1.0
    n_vertices = 5
    n_times = 500
    
    # shape: (n_vertices, 3, n_times)
    data_vector = np.ones((n_vertices, 3, n_times))
    
    vertices = [np.arange(3), np.arange(3, 5)]
    stc = mne.VectorSourceEstimate(data_vector, vertices=vertices, tmin=tmin, tstep=1.0/sfreq)
    
    aligned = align_and_pad_epochs(
        [stc], df, align_to='go', tmin=tmin, sfreq=sfreq,
        max_duration_samples=400, cutoff_before_enter_ms=300.0
    )
    
    # aligned shape should be (n_vertices, 3, max_duration_samples)
    assert aligned[0].shape == (5, 3, 400)
    assert np.all(np.isnan(aligned[0][:, :, 220:]))
    assert not np.any(np.isnan(aligned[0][:, :, :220]))


@patch('meg_tokens.meg.erp.mne.read_labels_from_annot')
def test_parcellate_source_estimates(mock_read_labels):
    # Set up mock labels
    label_lh = MagicMock()
    label_lh.name = 'Label_1-lh'
    
    label_rh = MagicMock()
    label_rh.name = 'Label_2-rh'
    
    # Mock return values for read_labels_from_annot
    # Medal wall or unknown labels are ignored by parcellation
    mock_read_labels.side_effect = [
        [label_lh],  # left hemi
        [label_rh]   # right hemi
    ]
    
    sfreq = 100.0
    n_vertices = 10
    n_times = 100
    data = np.ones((n_vertices, n_times)) * 5.0
    vertices = [np.arange(5), np.arange(5, 10)]
    stc = mne.SourceEstimate(data, vertices=vertices, tmin=0.0, tstep=1.0/sfreq)
    
    # Mock stc.in_label
    mock_stc_in_label = MagicMock()
    mock_stc_in_label.data = np.ones((3, n_times)) * 10.0  # 3 vertices in label
    stc.in_label = MagicMock(return_value=mock_stc_in_label)
    
    label_names, parcellated = parcellate_source_estimates(
        stc, subjects_dir='mock_dir', subject='fsaverage', parc='HCPMMP1'
    )
    
    assert label_names == ['Label_1-lh', 'Label_2-rh']
    assert parcellated.shape == (2, n_times)
    # The average across vertices should be 10.0
    assert np.allclose(parcellated, 10.0)

def test_export_neural_space():
    data = np.random.randn(5, 100)
    label_names = ['L1', 'L2', 'L3', 'L4', 'L5']
    
    with tempfile.TemporaryDirectory() as tmpdir:
        export_neural_space(data, label_names, tmpdir, 'test_export', format='both')
        
        # Check files exist
        assert os.path.exists(os.path.join(tmpdir, 'test_export.npy'))
        assert os.path.exists(os.path.join(tmpdir, 'test_export.mat'))
        
        # Load and verify
        loaded_npy = np.load(os.path.join(tmpdir, 'test_export.npy'))
        assert np.allclose(loaded_npy, data)
        
        loaded_mat = sio.loadmat(os.path.join(tmpdir, 'test_export.mat'))
        assert 'data' in loaded_mat
        assert np.allclose(loaded_mat['data'], data)
