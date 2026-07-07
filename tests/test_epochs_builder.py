import os
import tempfile
import numpy as np
import pandas as pd
import pytest
import mne
from mne.io import RawArray
from meg_tokens.utils.epochs_builder import (
    build_epochs_with_metadata,
    find_behavior_table,
    get_event_id,
    load_behavior_table,
    parse_run_label,
    save_epochs_and_events,
)
from meg_tokens.io import save_table

@pytest.fixture
def sample_meg_data():
    # Create 3-channel raw dataset (2 mag channels, 1 stim channel)
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
        'subject': ['H01'] * 4,
        'condition': ['Slow'] * 4,
        'run': [1] * 4,
        'source_file': ['H1Slow1_180131.tdms'] * 4,
        'nTrialIndex': [1, 2, 3, 4],
        'sTrialClass': [1, 2, 1, 3],
        'nInitialTime': [0, 0, 0, 0],
        'nChoiceMade': [1, 2, 1, 2],
        'nCorrectChoice': [1, 2, 1, 1],
        'tGO': [100, 105, 110, 115],
        'tEnterTarget': [400, 455, 510, 565],
        'tTrialEnd': [900, 955, 1010, 1065],
        'sTokenDirs': ['121', '212', '121', '212'],
        'tTime': ['[200, 300]'] * 4,
        'nProb': ['[0.6, 0.8]'] * 4,
        'rawRT': [300, 350, 400, 450],
        'isCorrect': [True, True, True, False],
    })
    
    return raw, events, behavior_df

def test_build_epochs_matched(sample_meg_data):
    raw, events, behavior_df = sample_meg_data
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

def test_build_epochs_mismatch_too_many_trials_raises(sample_meg_data):
    raw, events, behavior_df = sample_meg_data
    event_ids = {"Go": 524288}
    
    # Add an extra trial to behavioral dataframe (5 trials instead of 4)
    extra_row = pd.DataFrame({
        **{col: [behavior_df.iloc[-1][col]] for col in behavior_df.columns}
    })
    extra_row['nTrialIndex'] = [5]
    behavior_df_mismatched = pd.concat([behavior_df, extra_row], ignore_index=True)

    with pytest.raises(ValueError, match="Trial count mismatch"):
        build_epochs_with_metadata(
            raw, events, event_ids,
            tmin=-0.1, tmax=0.5,
            behavior_df=behavior_df_mismatched,
            baseline=None
        )

def test_build_epochs_mismatch_can_truncate_when_explicit(sample_meg_data):
    raw, events, behavior_df = sample_meg_data
    event_ids = {"Go": 524288}
    
    # Truncate behavioral dataframe to 3 trials
    behavior_df_short = behavior_df.iloc[:3].copy()
    
    epochs = build_epochs_with_metadata(
        raw, events, event_ids,
        tmin=-0.1, tmax=0.5,
        behavior_df=behavior_df_short,
        baseline=None,
        on_mismatch="truncate",
    )
    
    # Should truncate events to match the 3 trials
    assert len(epochs) == 3
    assert len(epochs.metadata) == 3
    assert list(epochs.metadata['nTrialIndex']) == [1, 2, 3]
    assert list(epochs.metadata['meg_sample']) == [1000, 3000, 5000]

def test_save_epochs_and_events_bids(sample_meg_data):
    raw, events, behavior_df = sample_meg_data
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
            bids_format=True,
            align_to='go',
        )
        
        # Verify folder structure and paths
        expected_meg_dir = os.path.join(tmp_dir, 'derivatives', 'meg-tokens', 'sub-H01', 'meg')
        assert os.path.exists(expected_meg_dir)
        
        assert os.path.exists(paths['epochs'])
        assert os.path.exists(paths['events_tsv'])
        assert os.path.exists(paths['eve'])
        
        # Check BIDS tsv content
        bids_df = pd.read_csv(paths['events_tsv'], sep='\t')
        assert 'onset' in bids_df.columns
        assert 'duration' in bids_df.columns
        assert 'sample' in bids_df.columns
        assert 'trial_type' in bids_df.columns
        assert 'value' in bids_df.columns
        assert 'nTrialIndex' in bids_df.columns
        assert 'sTrialClass' in bids_df.columns
        
        # Onsets should match samples / sfreq
        np.testing.assert_allclose(bids_df['onset'].values, [1.0, 3.0, 5.0, 7.0])
        assert os.path.exists(paths['epochs'].replace('.fif', '.json'))


def test_get_event_id_subject_override():
    assert get_event_id('go', 'H06') == {'Go': 262144}
    assert get_event_id('go', 'H01') == {'Go': 524288}


def test_parse_run_label():
    assert parse_run_label('Slow1') == ('1', 'Slow')
    assert parse_run_label('run-2') == ('2', None)


def test_find_and_load_behavior_table(tmp_path, sample_meg_data):
    _, _, behavior_df = sample_meg_data
    beh_path = (
        tmp_path
        / "derivatives"
        / "meg-tokens"
        / "sub-H01"
        / "beh"
        / "sub-H01_task-tokens_run-1_desc-slow_beh.tsv"
    )
    save_table(beh_path, behavior_df, metadata={"stage": "behavior_parsing"})

    found = find_behavior_table(tmp_path, "H1", "Slow1")
    loaded = load_behavior_table(str(found))

    assert found == beh_path
    assert list(loaded["nTrialIndex"]) == [1, 2, 3, 4]
