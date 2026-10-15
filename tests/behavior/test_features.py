"""Tests for meg_tokens.behavior.features."""

import pandas as pd
import pytest

from meg_tokens.behavior.features import (
    build_trial_features,
    calculate_decision_times,
    logged_spd_at_decision,
)
from meg_tokens.behavior.schema import OUTCOME_NEVER_STARTED

from tests.behavior.factories import GO_CUE_MS, stage1_run


BASELINE = 350.0


def _built(**run_kwargs):
    run = stage1_run(**run_kwargs)
    subject = str(run["subject"].iloc[0])
    return build_trial_features({subject: [run]}, {subject: BASELINE})


# --- decision times -----------------------------------------------------------


def test_decision_time_removes_the_baseline_keeps_the_index_and_keeps_anticipations():
    assert calculate_decision_times(
        stage1_run(raw_rts=[1000.0, 1200.0, 1100.0]), 360.0
    ).tolist() == [640.0, 840.0, 740.0]

    with_gap = calculate_decision_times(stage1_run(raw_rts=[1000.0, None, 1100.0]), 360.0)
    assert with_gap.index.tolist() == [0, 2]
    assert with_gap.tolist() == [640.0, 740.0]

    # A response faster than the motor baseline is flagged later, not dropped.
    assert calculate_decision_times(stage1_run(raw_rts=[200.0]), 360.0).tolist() == [-160.0]


# --- logged success probability at commitment ---------------------------------


def _trial(raw_rt, probabilities, times):
    return pd.Series({
        "nProb": list(probabilities),
        "tTime": list(times),
        "tEnterTarget": GO_CUE_MS + raw_rt,
    })


def test_the_logged_spd_is_read_in_the_token_log_clock_after_the_motor_latency():
    # Commitment lands at 1500 ms once the 400 ms baseline is removed.
    trial = _trial(900.0, [0.6, 0.7, 0.8], [1200, 1400, 1600])
    assert logged_spd_at_decision(trial, motor_baseline=400.0) == (0.7, 1)
    # The same response reads a different token without the correction, so the
    # baseline is not cosmetic.
    assert logged_spd_at_decision(trial, motor_baseline=0.0)[1] == 2

    # Commitment before the first token is the prior, with no index.
    assert logged_spd_at_decision(_trial(500.0, [0.6], [1200]), motor_baseline=400.0) == (
        0.5, None
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        logged_spd_at_decision(_trial(900.0, [0.6, 0.7], [1200, 1200]), motor_baseline=0.0)


# --- the canonical feature table ----------------------------------------------


def test_the_join_key_is_positional_within_the_run_and_counts_started_trials_apart():
    features = _built(raw_rts=[900.0, 1100.0, 1000.0], first_trial_index=7)
    assert features["run_trial_index"].tolist() == [1, 2, 3]
    # The acquisition counter is preserved separately, not used as the key.
    assert features["nTrialIndex"].tolist() == [7, 8, 9]

    started = _built(raw_rts=[900.0, None, 1100.0], outcomes=[0, OUTCOME_NEVER_STARTED, 0])
    assert started["started_trial_index"].tolist()[0] == 1
    assert pd.isna(started["started_trial_index"].iloc[1])
    assert started["started_trial_index"].tolist()[2] == 2


def test_per_hand_baselines_are_applied_to_the_hand_that_answered_or_fall_back():
    run = stage1_run(condition="Fast", raw_rts=[900.0, 900.0], choices=[1, 2])
    subject = str(run["subject"].iloc[0])

    # Pooled: both hands lose the same constant, so identical rawRTs give
    # identical decision times and any hand difference survives into dt_ms.
    pooled = build_trial_features({subject: [run]}, {subject: 350.0})
    assert pooled["dt_ms"].tolist() == pytest.approx([550.0, 550.0])

    per_hand = build_trial_features(
        {subject: [run]}, {subject: 350.0}, {subject: {1: 300.0, 2: 400.0}}
    )
    assert per_hand["dt_ms"].tolist() == pytest.approx([600.0, 500.0])
    assert per_hand["motor_baseline_ms"].tolist() == pytest.approx([300.0, 400.0])

    # Only the right hand (side 1) has an RT-run estimate; the left must not be dropped
    # or invented, it falls back to the pooled baseline.
    partial = build_trial_features({subject: [run]}, {subject: 350.0}, {subject: {1: 300.0}})
    assert partial["motor_baseline_ms"].tolist() == pytest.approx([300.0, 350.0])


def test_design_evidence_is_measured_against_the_correct_target_or_withheld():
    features = _built(
        raw_rts=[1000.0, 1000.0], correct_choices=[1, 2], token_directions="112" + "1" * 12
    )
    # After three jumps target 1 leads 2-1 and target 2 trails by the same
    # amount, so their early evidence is symmetric about chance.
    leading, trailing = features["sp_design_early"].tolist()
    assert leading > 0.5 > trailing
    assert leading + trailing == pytest.approx(1.0)
    leading_odds, trailing_odds = features["sum_log_lr_design_early"].tolist()
    assert leading_odds == pytest.approx(-trailing_odds)

    # The lead counts only the jumps the subject had actually seen.
    seen = _built(raw_rts=[1000.0], token_directions="111" + "2" * 12)
    assert seen["decision_token_index"].tolist() == [3]
    assert seen["token_lead_at_decision"].tolist() == [3]

    withheld = _built(raw_rts=[900.0], correct_choices=[0]).iloc[0]
    assert pd.isna(withheld["sp_design_early"])
    assert pd.isna(withheld["sum_log_lr_design_early"])
    assert pd.isna(withheld["token_lead_at_decision"])
    assert pd.isna(withheld["correct_side"])
