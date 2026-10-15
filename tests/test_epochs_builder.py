import json

import mne
import numpy as np
import pandas as pd
import pytest

from meg_tokens.behavior.schema import OUTCOME_NEVER_STARTED
from meg_tokens.core import EPOCH_LOCKS
from meg_tokens.meg.epoching import (
    OUTLIER_REASON,
    OUTLIER_SD,
    PADDING,
    WINDOWS,
    align_go,
    amplitudes,
    drop_outliers,
    epoch,
    outlier_thresholds,
    read_triggers,
    reconstruct_missing_go,
    save_epochs,
    trial_metadata,
)


def _behavior(n, *, never_started=(), t_go=100):
    outcome = [OUTCOME_NEVER_STARTED if i in never_started else 0 for i in range(n)]
    return pd.DataFrame({
        "nTrialIndex": np.arange(1, n + 1),
        "nOutcome": outcome,
        "tGO": [0 if i in never_started else t_go for i in range(n)],
        "tEnterTarget": [t_go + 400 + 10 * i for i in range(n)],
        "tTrialEnd": [t_go + 900 + 10 * i for i in range(n)],
    })


def test_every_lock_has_a_window():
    assert tuple(WINDOWS) == EPOCH_LOCKS


def test_a_count_mismatch_raises_unless_the_run_is_a_known_trailing_one():
    behavior = _behavior(5)
    triggers = {"Go": np.arange(4) * 3.0}
    with pytest.raises(ValueError, match="4 go cues for 5 started trials"):
        align_go(triggers, behavior, "H01", "Fast", "1")

    # H04 RT1 is a verified trailing mismatch (normalized from "h4"): the tail goes.
    go, trials, record = align_go(triggers, behavior, "h4", "RT", "1")
    assert len(go) == len(trials) == 4
    assert trials["nTrialIndex"].tolist() == [1, 2, 3, 4]
    assert record["n_truncated"] == 1


def test_never_started_and_unrecoverable_trials_are_not_paired():
    behavior = _behavior(6, never_started={0})
    # H12 Slow2: trial 2 has no MEG event at all; trial 1 never started.
    go, trials, record = align_go({"Go": np.arange(4) * 3.0}, behavior, "H12", "Slow", "2")
    assert trials["nTrialIndex"].tolist() == [3, 4, 5, 6]
    assert record["n_trials_unrecoverable"] == 1
    assert record["n_truncated"] == 0


def test_a_missing_go_cue_is_placed_at_start_plus_tgo_plus_the_run_lag():
    starts = np.array([0.0, 1.0, 2.0])
    lag = 0.005
    gos = np.array([0.1 + lag, 2.1 + lag])
    full = reconstruct_missing_go(starts, gos, _behavior(3))
    np.testing.assert_allclose(full, [0.1 + lag, 1.1 + lag, 2.1 + lag])

    # A never-started trial gets none.
    assert len(reconstruct_missing_go(starts, gos, _behavior(3, never_started={1}))) == 2


def test_triggers_are_read_by_name_from_the_raw_bids_events(tmp_path):
    meg = tmp_path / "sub-H06" / "meg"
    meg.mkdir(parents=True)
    pd.DataFrame({
        "onset": [1.5, 0.5, 1.0, 2.0],
        "duration": 0.0,
        "trial_type": ["Go", "Start", "Go", "Start"],
        "value": [262144, 524288, 262144, 524288],
        "sample": [1800, 600, 1200, 2400],
    }).to_csv(meg / "sub-H06_task-tokens_acq-slow_run-2_events.tsv", sep="\t", index=False)
    triggers = read_triggers(tmp_path, "H06", "Slow", 2)
    np.testing.assert_allclose(triggers["Go"], [1.0, 1.5])
    np.testing.assert_allclose(triggers["Start"], [0.5, 2.0])


def _recording(seconds=40.0, sfreq=600.0):
    info = mne.create_info(["MEG1", "MEG2", "REF1", "EEG1"], sfreq, ["mag", "mag", "ref_meg", "eeg"])
    data = np.random.default_rng(0).standard_normal((4, int(seconds * sfreq))) * 1e-13
    return mne.io.RawArray(data, info, verbose=False)


def _metadata(n):
    behavior = _behavior(n)
    features = pd.DataFrame({
        "nTrialIndex": behavior["nTrialIndex"],
        "rawRT": [400.0, np.nan, 420.0, 430.0][:n],
        "dt_ms": [150.0, np.nan, 170.0, 180.0][:n],
        "trial_class": [1, 2, 1, 3][:n],
    })
    return trial_metadata(behavior, features)


def test_epochs_are_padded_and_dropped_only_for_a_mark_inside_the_analysis_window():
    raw = _recording()
    go = np.array([5.0, 12.0, 20.0, 28.0])
    tmin, tmax = WINDOWS["go"]
    raw.set_annotations(mne.Annotations(
        [5.0 + tmax + PADDING / 2, 20.3], [0.2, 0.2], ["BAD_muscle", "BAD_peak"]
    ))
    epochs = epoch(raw, go, _metadata(4), "go")

    assert epochs.tmin == pytest.approx(tmin - PADDING)
    assert epochs.tmax == pytest.approx(tmax + PADDING)
    assert epochs.ch_names == ["MEG1", "MEG2", "REF1"]
    # The first epoch's mark sits in its margin only, the third's in its window.
    assert epochs.selection.tolist() == [0, 1, 3]
    assert epochs.drop_log[2] == ("BAD_peak",)
    assert epochs.metadata["nTrialIndex"].tolist() == [1, 2, 4]


def test_each_lock_sits_at_its_behavioral_offset_from_the_go_cue():
    raw = _recording()
    go = np.array([5.0, 12.0, 20.0, 28.0])
    metadata = _metadata(4)
    sfreq = raw.info["sfreq"]

    enter = epoch(raw, go, metadata, "enter")
    # Trial 2 has no response, so it has no press or decision epoch.
    assert enter.metadata["nTrialIndex"].tolist() == [1, 3, 4]
    np.testing.assert_allclose(enter.events[:, 0] / sfreq, [5.4, 20.42, 28.43], atol=1 / sfreq)
    np.testing.assert_allclose(enter.metadata["lock_onset_s"], [5.4, 20.42, 28.43])

    decision = epoch(raw, go, metadata, "decision")
    np.testing.assert_allclose(decision.events[:, 0] / sfreq, [5.15, 20.17, 28.18], atol=1 / sfreq)

    feedback = epoch(raw, go, metadata, "feedback")
    assert len(feedback) == 4
    np.testing.assert_allclose(
        feedback.metadata["lock_onset_s"], go + (metadata["tTrialEnd"] - metadata["tGO"]) / 1000
    )


def test_the_sidecar_counts_dropped_epochs_by_reason(tmp_path):
    raw = _recording()
    raw.set_annotations(mne.Annotations([12.2], [0.1], ["BAD_muscle"]))
    epochs = epoch(raw, np.array([5.0, 12.0, 39.0]), _metadata(3), "go")
    path = save_epochs(epochs, tmp_path / "x_epo.fif", {"stream": "dss"})

    sidecar = json.loads(path.with_suffix(".json").read_text())
    assert sidecar["n_events"] == 3
    assert sidecar["n_epochs"] == 1
    # The last epoch's margin runs past the end of the recording.
    assert sidecar["dropped"] == {"BAD_muscle": 1, "TOO_SHORT": 1}
    assert sidecar["analysis_window_s"] == [-0.5, 1.0]
    assert len(mne.read_epochs(path, verbose=False)) == 1


def test_an_epoch_near_the_recording_edge_keeps_the_margin_it_has_and_zero_beyond():
    raw = _recording()
    sfreq = raw.info["sfreq"]
    # The first window starts 0.3 s into the recording, the last ends 0.5 s before its end.
    go = np.array([0.8, 12.0, 38.5])
    epochs = epoch(raw, go, _metadata(3), "go")

    assert epochs.selection.tolist() == [0, 1, 2]
    np.testing.assert_allclose(epochs.metadata["pad_before_s"], [0.3, PADDING, PADDING])
    np.testing.assert_allclose(epochs.metadata["pad_after_s"], [PADDING, PADDING, 0.5 - 1 / sfreq])
    first, last = epochs.get_data()[[0, 2]]
    missing = int(round((PADDING - 0.3) * sfreq))
    assert not first[:, :missing].any()
    np.testing.assert_array_equal(first[:, missing:], raw.get_data(picks=epochs.ch_names, stop=first.shape[1] - missing))
    assert not last[:, -int(round(0.5 * sfreq)):].any()


def test_an_epoch_whose_analysis_window_leaves_the_recording_is_dropped():
    raw = _recording()
    epochs = epoch(raw, np.array([0.3, 12.0, 39.5]), _metadata(3), "go")
    assert epochs.selection.tolist() == [1]
    assert epochs.drop_log[0] and epochs.drop_log[2]


def test_the_sidecar_counts_epochs_with_a_short_margin(tmp_path):
    epochs = epoch(_recording(), np.array([0.8, 12.0, 20.0]), _metadata(3), "go")
    path = save_epochs(epochs, tmp_path / "x_epo.fif", {"stream": "dss"})
    assert json.loads(path.with_suffix(".json").read_text())["n_short_padding"] == 1
    np.testing.assert_allclose(mne.read_epochs(path, verbose=False).metadata["pad_before_s"], [0.3, PADDING, PADDING])


def test_amplitudes_are_the_log_peak_to_peak_of_each_magnetometer_over_the_analysis_window():
    raw = _recording()
    sfreq = raw.info["sfreq"]
    # a 1 pT step inside the first epoch's margin, and one inside the second's window
    data = raw.get_data()
    data[0, int(4.0 * sfreq)] += 1e-12
    data[1, int(12.5 * sfreq)] += 1e-12
    raw = mne.io.RawArray(data, raw.info, verbose=False)
    epochs = epoch(raw, np.array([5.0, 12.0]), _metadata(2), "go")
    table = amplitudes(epochs)

    assert list(table.columns) == ["MEG1", "MEG2"]
    tmin, tmax = WINDOWS["go"]
    window = epochs.copy().pick("mag").crop(tmin, tmax).get_data()
    np.testing.assert_allclose(table.to_numpy(), np.log10(np.ptp(window, axis=2)))
    assert table.loc[0, "MEG1"] < -12.0        # the margin's step is not in the window
    assert table.loc[1, "MEG2"] > -12.0        # the window's is


def test_thresholds_are_the_median_plus_robust_sds_and_outliers_are_dropped_by_reason():
    table = pd.DataFrame({"A": [0.0, 1.0, 2.0, 3.0, 4.0], "B": [10.0, 10.0, 11.0, 12.0, 12.0]})
    thresholds = outlier_thresholds(table)
    assert thresholds["A"] == pytest.approx(2.0 + OUTLIER_SD * 1.4826 * 1.0)
    assert thresholds["B"] == pytest.approx(11.0 + OUTLIER_SD * 1.4826 * 1.0)

    raw = _recording()
    data = raw.get_data()
    data[1, int(20.2 * raw.info["sfreq"])] += 1e-11   # a large excursion in the third epoch's window
    raw = mne.io.RawArray(data, raw.info, verbose=False)
    epochs = epoch(raw, np.array([5.0, 12.0, 20.0, 28.0]), _metadata(4), "go")
    cohort = amplitudes(epochs).drop(index=2)
    kept = drop_outliers(epochs, outlier_thresholds(pd.concat([cohort] * 5, ignore_index=True)) + 0.1)
    assert kept.selection.tolist() == [0, 1, 3]
    assert kept.drop_log[2] == (OUTLIER_REASON,)
