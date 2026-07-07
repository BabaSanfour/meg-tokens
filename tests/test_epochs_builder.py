import os
import tempfile
import numpy as np
import pandas as pd
import pytest
import mne
from mne.io import RawArray
from meg_tokens.utils.epochs_builder import build_epochs_with_metadata, save_epochs_and_events

@pytest.fixture
def mock_meg_data():
    # Create 3-channel mock raw dataset (2 mag channels, 1 stim channel)
    info = mne.create_info(
        ch_names=['MEG 001', 'MEG 002', 'STI 014'],
        sfreq=1000.0,
        ch_types=['mag', 'mag', 'stim']
    )
    # 10 seconds of data
    data = np.zeros((3, 10000))
    # Place 'Go' trigger events (code 524288) at samples 1000, 3000, 5000, 7000 (4 events)
    data[2, 1000] = 524288
    data[2, 3000] = 524288
    data[2, 5000] = 524288
    data[2, 7000] = 524288
    
    raw = RawArray(data, info)
    events = mne.find_events(raw, stim_channel='STI 014')
    
    # 4 corresponding behavioral trials
    behavior_df = pd.DataFrame({
        'nTrialIndex': [1, 2, 3, 4],
        'sTrialClass': [1, 2, 1, 3],
        'nChoiceMade': [1, 2, 1, 2],
        'tGO': [100, 105, 110, 115]
    })
    
    return raw, events, behavior_df

def test_build_epochs_matched(mock_meg_data):
    raw, events, behavior_df = mock_meg_data
    event_ids = {"Go": 524288}
    
    epochs = build_epochs_with_metadata(
        raw, events, event_ids,
        tmin=-0.1, tmax=0.5,
        behavior_df=behavior_df,
        baseline=None
    )
    
    assert len(epochs) == 4
    assert epochs.metadata is not None
    assert len(epochs.metadata) == 4
    assert list(epochs.metadata['nTrialIndex']) == [1, 2, 3, 4]
    assert list(epochs.metadata['meg_sample']) == [1000, 3000, 5000, 7000]

def test_build_epochs_mismatch_too_many_trials(mock_meg_data):
    raw, events, behavior_df = mock_meg_data
    event_ids = {"Go": 524288}
    
    # Add an extra trial to behavioral dataframe (5 trials instead of 4)
    extra_row = pd.DataFrame({
        'nTrialIndex': [5], 'sTrialClass': [2], 'nChoiceMade': [1], 'tGO': [120]
    })
    behavior_df_mismatched = pd.concat([behavior_df, extra_row], ignore_index=True)
    
    epochs = build_epochs_with_metadata(
        raw, events, event_ids,
        tmin=-0.1, tmax=0.5,
        behavior_df=behavior_df_mismatched,
        baseline=None
    )
    
    # Should truncate the behavioral dataframe to match the 4 events
    assert len(epochs) == 4
    assert len(epochs.metadata) == 4
    assert list(epochs.metadata['nTrialIndex']) == [1, 2, 3, 4]

def test_build_epochs_mismatch_too_many_events(mock_meg_data):
    raw, events, behavior_df = mock_meg_data
    event_ids = {"Go": 524288}
    
    # Truncate behavioral dataframe to 3 trials
    behavior_df_short = behavior_df.iloc[:3].copy()
    
    epochs = build_epochs_with_metadata(
        raw, events, event_ids,
        tmin=-0.1, tmax=0.5,
        behavior_df=behavior_df_short,
        baseline=None
    )
    
    # Should truncate events to match the 3 trials
    assert len(epochs) == 3
    assert len(epochs.metadata) == 3
    assert list(epochs.metadata['nTrialIndex']) == [1, 2, 3]
    assert list(epochs.metadata['meg_sample']) == [1000, 3000, 5000]

def test_save_epochs_and_events_bids(mock_meg_data):
    raw, events, behavior_df = mock_meg_data
    event_ids = {"Go": 524288}
    
    epochs = build_epochs_with_metadata(
        raw, events, event_ids,
        tmin=-0.1, tmax=0.5,
        behavior_df=behavior_df,
        baseline=None
    )
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = save_epochs_and_events(
            epochs, tmp_dir,
            subject_id='H1', run_id='Slow1',
            bids_format=True
        )
        
        # Verify folder structure and paths
        expected_meg_dir = os.path.join(tmp_dir, 'sub-H1', 'meg')
        assert os.path.exists(expected_meg_dir)
        
        assert os.path.exists(paths['epochs'])
        assert os.path.exists(paths['events_tsv'])
        assert os.path.exists(paths['eve'])
        
        # Check BIDS tsv content
        bids_df = pd.read_csv(paths['events_tsv'], sep='\t')
        assert 'onset' in bids_df.columns
        assert 'duration' in bids_df.columns
        assert 'trial_type' in bids_df.columns
        assert 'value' in bids_df.columns
        assert 'nTrialIndex' in bids_df.columns
        assert 'sTrialClass' in bids_df.columns
        
        # Onsets should match samples / sfreq
        np.testing.assert_allclose(bids_df['onset'].values, [1.0, 3.0, 5.0, 7.0])
