"""Tests for meg_tokens.behavior.features."""

import numpy as np
import pandas as pd
import pytest

from meg_tokens.behavior.features import (
    build_trial_features,
    calculate_decision_times,
    calculate_motor_baseline,
    calculate_motor_baseline_by_side,
    logged_spd_at_decision,
)
from meg_tokens.behavior.schema import (
    OUTCOME_NEVER_STARTED,
    TRIAL_FEATURE_COLUMNS,
    validate_trial_features,
)

from tests.behavior.factories import GO_CUE_MS, stage1_run


BASELINE = 350.0


def _built(**run_kwargs):
    run = stage1_run(**run_kwargs)
    subject = str(run["subject"].iloc[0])
    return build_trial_features({subject: [run]}, {subject: BASELINE})


# --- motor baseline -----------------------------------------------------------


def test_the_motor_baseline_weights_runs_equally_and_ignores_unusable_ones():
    first = stage1_run(condition="RT", raw_rts=[300.0, 400.0, 350.0])
    second = stage1_run(condition="RT", run=2, raw_rts=[320.0, 430.0])
    # Runs, not trials, are weighted equally: mean(350, 375), not the mean of
    # all five values (360.0), which is what unequal run sizes would give.
    assert calculate_motor_baseline([first, second]) == pytest.approx(362.5)

    usable = stage1_run(condition="RT", raw_rts=[300.0, 400.0])
    # A run with no responses, a trial with no response, and an empty table are
    # each skipped rather than counted as zero.
    assert calculate_motor_baseline(
        [usable, stage1_run(condition="RT", run=2, raw_rts=[None, None])]
    ) == pytest.approx(350.0)
    assert calculate_motor_baseline(
        [stage1_run(condition="RT", raw_rts=[300.0, None, 400.0])]
    ) == pytest.approx(350.0)
    assert calculate_motor_baseline([usable.iloc[:0], usable]) == pytest.approx(350.0)


def test_the_per_hand_baseline_separates_the_hands_and_omits_one_that_never_answered():
    run = stage1_run(condition="RT", raw_rts=[300.0, 500.0, 320.0, 520.0], choices=[1, 2, 1, 2])
    assert calculate_motor_baseline_by_side([run]) == {
        1: pytest.approx(310.0), 2: pytest.approx(510.0)
    }
    # The pooled estimator averages over both hands and so hides the split.
    assert calculate_motor_baseline([run]) == pytest.approx(410.0)

    # Runs are weighted equally here too: mean(350, 380), not mean(300, 400, 380).
    first = stage1_run(condition="RT", raw_rts=[300.0, 400.0], choices=[1, 1])
    second = stage1_run(condition="RT", run=2, raw_rts=[380.0], choices=[1])
    assert calculate_motor_baseline_by_side([first, second])[1] == pytest.approx(365.0)
    # Callers fall back to the pooled baseline for a hand with no response
    # rather than receiving an invented value for it.
    assert set(calculate_motor_baseline_by_side([first])) == {1}


@pytest.mark.parametrize(
    "estimator", [calculate_motor_baseline, calculate_motor_baseline_by_side]
)
@pytest.mark.parametrize(
    "tables",
    [[], [stage1_run(condition="RT", raw_rts=[None, None])]],
    ids=["no_runs", "no_responses"],
)
def test_a_motor_baseline_is_never_invented(estimator, tables):
    with pytest.raises(ValueError, match="No valid RT trials"):
        estimator(tables)


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


def test_build_trial_features_emits_the_declared_schema_and_keeps_every_staged_trial():
    features = _built(raw_rts=[900.0, 1100.0])
    assert list(features.columns) == TRIAL_FEATURE_COLUMNS
    validate_trial_features(features)
    assert features["dt_ms"].tolist() == [550.0, 750.0]
    assert features["motor_baseline_ms"].tolist() == [BASELINE, BASELINE]

    # Selection is expressed with flags; no observation is deleted here.
    flagged = _built(raw_rts=[900.0, None, None], outcomes=[0, 7006, OUTCOME_NEVER_STARTED])
    assert len(flagged) == 3
    assert flagged["is_started"].tolist() == [True, True, False]
    assert flagged["has_choice"].tolist() == [True, False, False]
    assert flagged["is_no_response"].tolist() == [False, True, False]
    assert flagged["outcome_label"].tolist()[:2] == ["completed", "reaction_time_too_long"]

    classes = _built(raw_rts=[900.0] * 4, trial_classes=[1, 2, 3, 0])
    assert classes["trial_class_name"].tolist() == [
        "easy", "ambiguous", "misleading", "unclassified",
    ]


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

    # Only the left hand has an RT-run estimate; the right must not be dropped
    # or invented, it falls back to the pooled baseline.
    partial = build_trial_features({subject: [run]}, {subject: 350.0}, {subject: {1: 300.0}})
    assert partial["motor_baseline_ms"].tolist() == pytest.approx([300.0, 350.0])


def test_the_rt_baseline_run_carries_no_decision_and_anticipations_stay_eligible():
    """RT runs measure motor latency; they carry no decision to time."""
    rt_run = _built(condition="RT", raw_rts=[340.0, 360.0])
    assert rt_run["dt_ms"].isna().all()
    assert rt_run["logged_spd"].isna().all()
    assert not rt_run["primary_analysis_eligible"].any()

    anticipations = _built(raw_rts=[200.0, 900.0])
    assert anticipations["dt_ms"].tolist() == [-150.0, 550.0]
    assert anticipations["dt_anticipation"].tolist() == [True, False]
    assert anticipations["primary_analysis_eligible"].tolist() == [True, True]


def test_the_logged_spd_scales_agree_and_the_strict_view_needs_a_complete_log():
    row = _built(raw_rts=[1000.0]).iloc[0]
    # Commitment at 1000 + 1000 - 350 = 1650 ms: the token logged at 1600 ms.
    assert row["decision_token_index"] == 3
    assert row["logged_spd"] == pytest.approx(0.62)
    assert row["evidence_at_decision"] == pytest.approx(0.62 - 0.5)
    assert row["logged_spd_log_odds"] == pytest.approx(np.log(0.62 / 0.38))

    early = _built(raw_rts=[100.0])
    assert early["decision_token_index"].tolist() == [0]
    assert early["logged_spd"].tolist() == [0.5]

    # A 14-row log cannot be aligned to the 15-jump design, so the strict view
    # is empty rather than silently reusing the all-logged value.
    short = _built(raw_rts=[1000.0], probabilities=[0.6] * 14)
    complete = _built(raw_rts=[1000.0])
    assert short["token_log_rows"].tolist() == [14]
    assert short["logged_spd"].notna().all()
    assert short["logged_spd_validated_15row"].isna().all()
    assert not short["design_time_alignment_valid"].any()
    assert complete["logged_spd_validated_15row"].notna().all()
    assert complete["design_time_alignment_valid"].all()


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


def test_several_subjects_are_ordered_keyed_apart_and_must_each_have_a_baseline():
    features = build_trial_features(
        {
            "H02": [stage1_run(subject="H02", raw_rts=[900.0])],
            "H01": [
                stage1_run(subject="H01", condition="Fast", raw_rts=[900.0]),
                stage1_run(subject="H01", condition="Slow", raw_rts=[1300.0]),
            ],
        },
        {"H01": 300.0, "H02": 400.0},
    )
    assert features["subject"].tolist() == ["H01", "H01", "H02"]
    assert features["motor_baseline_ms"].tolist() == [300.0, 300.0, 400.0]
    assert not features["trial_id"].duplicated().any()
    validate_trial_features(features)

    with pytest.raises(KeyError):
        build_trial_features({"H01": [stage1_run(raw_rts=[900.0])]}, {"H02": 350.0})
