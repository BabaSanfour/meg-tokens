import os
import tempfile
import numpy as np
import pandas as pd
import pytest
import mne
from mne.io import RawArray
from meg_tokens.behavior.tdms import OUTCOME_NEVER_STARTED
from meg_tokens.meg.epoching import (
    build_epochs_with_metadata,
    exclude_unrecoverable_trials,
    find_behavior_table,
    get_event_id,
    load_behavior_table,
    mismatch_policy,
    needs_go_reconstruction,
    parse_run_label,
    reconstruct_missing_go_events,
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
        'sTrialClassRaw': ['e', 'a', 'e', 'm'],
        'trial_class_source': ['design'] * 4,
        'trial_class_rule': ['recorded_label'] * 4,
        'sp_design_correct': ['[0.6, 0.8]'] * 4,
        'nInitialTime': [0, 0, 0, 0],
        'nChoiceMade': [1, 2, 1, 2],
        'nCorrectChoice': [1, 2, 1, 1],
        'tGO': [100, 105, 110, 115],
        'tEnterCenter': [0, 0, 0, 0],
        'tExitCenter': [400, 455, 510, 565],
        'tEnterTarget': [400, 455, 510, 565],
        'tTrialEnd': [900, 955, 1010, 1065],
        'sTokenDirs': ['121', '212', '121', '212'],
        'nTokenNum': ['[1, 2]'] * 4,
        'nTokenDir': ['[1, 2]'] * 4,
        'tTime': ['[200, 300]'] * 4,
        'nProb': ['[0.6, 0.8]'] * 4,
        'token_log_rows': [2] * 4,
        'token_log_short': [False] * 4,
        'nOutcome': [0, 0, 0, 0],
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


def test_mismatch_policy_truncates_only_known_trailing_trial_runs():
    """Confirmed via MEG timing cross-check (docs/behavior_qc_report.md):
    a real, trailing boundary discrepancy (extra or missing pulse), not
    scattered/interior data loss. H08RT1 is included here: once
    synchronize_events_and_behavior excludes its OUTCOME_NEVER_STARTED
    trials, its remaining discrepancy is a single trailing missing trial,
    the same pattern as everything else in this list.
    """
    for subject, condition, run in [
        ('H04', 'RT', '1'),
        ('H05', 'RT', '1'),
        ('H07', 'RT', '1'),
        ('H08', 'RT', '1'),
        ('H09', 'RT', '2'),
        ('H11', 'RT', '2'),
        ('H12', 'RT', '1'),
        ('H16', 'RT', '1'),
        ('H23', 'Slow', '3'),
    ]:
        assert mismatch_policy(subject, condition, run) == 'truncate'
    assert mismatch_policy('h4', 'RT', '1') == 'truncate'  # normalizes subject


def test_mismatch_policy_defaults_to_error_for_unlisted_runs():
    """H12Slow2 has irregularities at both ends (not a single trailing
    discrepancy) and must never be silently truncated.
    """
    assert mismatch_policy('H01', 'Fast', '1') == 'error'  # not a verified run
    assert mismatch_policy('H12', 'Slow', '2') == 'error'
    assert mismatch_policy('H04', 'RT', '2') == 'error'  # wrong run number
    assert mismatch_policy('H04', 'Slow', '1') == 'error'  # wrong condition


def test_needs_go_reconstruction():
    assert needs_go_reconstruction('H02', 'RT', '1') is True
    assert needs_go_reconstruction('h4', 'Fast', '2') is True  # normalizes subject
    assert needs_go_reconstruction('H01', 'Fast', '1') is False  # not a verified run


def _go_recon_inputs(missing_trial=1, include_never_started=False, n_trials=3):
    sfreq = 1000.0
    start_samples = [i * 1000 for i in range(n_trials)]
    tgo_ms = [100] * n_trials
    lag_s = 0.005
    starts = [[s, 0, 262144] for s in start_samples]
    gos = []
    for i, (s, tgo) in enumerate(zip(start_samples, tgo_ms)):
        if i == missing_trial:
            continue
        go_sample = round(s + tgo + lag_s * 1000)
        gos.append([go_sample, 0, 524288])
    events = np.array(starts + gos)
    nOutcome = [0] * n_trials
    if include_never_started:
        nOutcome[missing_trial] = OUTCOME_NEVER_STARTED
    behavior_df = pd.DataFrame({
        'nTrialIndex': list(range(1, n_trials + 1)),
        'tGO': tgo_ms,
        'nOutcome': nOutcome,
    })
    return events, behavior_df, sfreq, start_samples, tgo_ms, lag_s


def test_reconstruct_missing_go_events_fills_single_gap():
    events, behavior_df, sfreq, start_samples, tgo_ms, lag_s = _go_recon_inputs(missing_trial=1)

    augmented = reconstruct_missing_go_events(
        events, behavior_df, sfreq, start_code=262144, go_code=524288,
    )

    go_events = augmented[augmented[:, 2] == 524288]
    assert len(go_events) == 3
    expected_sample = round(start_samples[1] + tgo_ms[1] + lag_s * 1000)
    assert expected_sample in go_events[:, 0].tolist()
    # calibrated from the two real go pulses, not the fallback default
    assert abs(expected_sample - (start_samples[1] + tgo_ms[1])) < 10


def test_reconstruct_missing_go_events_skips_never_started_trials():
    events, behavior_df, sfreq, *_ = _go_recon_inputs(missing_trial=1, include_never_started=True)

    augmented = reconstruct_missing_go_events(
        events, behavior_df, sfreq, start_code=262144, go_code=524288,
    )

    # Trial 1 (0-indexed) was never started, so no go cue is synthesized.
    assert len(augmented[augmented[:, 2] == 524288]) == len(events[events[:, 2] == 524288])


def test_reconstruct_missing_go_events_falls_back_to_default_lag_when_uncalibrated():
    n_trials = 2
    sfreq = 1000.0
    start_samples = [0, 1000]
    tgo_ms = [100, 100]
    starts = [[s, 0, 262144] for s in start_samples]
    events = np.array(starts)  # no go pulses at all, like H02RT1
    behavior_df = pd.DataFrame({
        'nTrialIndex': [1, 2],
        'tGO': tgo_ms,
        'nOutcome': [0, 0],
    })

    augmented = reconstruct_missing_go_events(
        events, behavior_df, sfreq, start_code=262144, go_code=524288,
    )

    go_events = augmented[augmented[:, 2] == 524288]
    assert len(go_events) == 2
    from meg_tokens.meg.epoching import DEFAULT_GO_LAG_S
    for i, sample in enumerate(start_samples):
        expected = round(sample + tgo_ms[i] + DEFAULT_GO_LAG_S * 1000)
        assert expected in go_events[:, 0].tolist()


def test_reconstruct_missing_go_events_raises_when_fewer_starts_than_trials():
    events = np.array([[0, 0, 262144]])
    behavior_df = pd.DataFrame({'nTrialIndex': [1, 2], 'tGO': [100, 100], 'nOutcome': [0, 0]})
    with pytest.raises(ValueError, match="trial-start pulses"):
        reconstruct_missing_go_events(events, behavior_df, 1000.0, start_code=262144, go_code=524288)


def test_exclude_unrecoverable_trials_drops_known_trial():
    behavior_df = pd.DataFrame({'nTrialIndex': [1, 2, 3, 4, 5]})
    out = exclude_unrecoverable_trials('H12', 'Slow', '2', behavior_df)
    assert out['nTrialIndex'].tolist() == [1, 3, 4, 5]
    # normalizes subject
    out2 = exclude_unrecoverable_trials('h12', 'Slow', '2', behavior_df)
    assert out2['nTrialIndex'].tolist() == [1, 3, 4, 5]


def test_exclude_unrecoverable_trials_is_noop_for_other_runs():
    behavior_df = pd.DataFrame({'nTrialIndex': [1, 2, 3]})
    out = exclude_unrecoverable_trials('H12', 'Slow', '1', behavior_df)
    assert out['nTrialIndex'].tolist() == [1, 2, 3]
    out = exclude_unrecoverable_trials('H01', 'Fast', '1', behavior_df)
    assert out['nTrialIndex'].tolist() == [1, 2, 3]


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
