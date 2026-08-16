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


# --- motor baseline -----------------------------------------------------------


def test_motor_baseline_is_the_mean_of_each_run_mean():
    first = stage1_run(condition="RT", raw_rts=[300.0, 400.0, 350.0])
    second = stage1_run(condition="RT", run=2, raw_rts=[320.0, 430.0])

    # Runs, not trials, are weighted equally: mean(350, 375), not mean of all
    # five values (360.0), which is what unequal run sizes would give.
    assert calculate_motor_baseline([first, second]) == pytest.approx(362.5)


def test_motor_baseline_ignores_runs_without_any_response():
    run = stage1_run(condition="RT", raw_rts=[300.0, 400.0])
    unusable = stage1_run(condition="RT", run=2, raw_rts=[None, None])

    assert calculate_motor_baseline([run, unusable]) == pytest.approx(350.0)


def test_motor_baseline_ignores_trials_without_a_response():
    run = stage1_run(condition="RT", raw_rts=[300.0, None, 400.0])

    assert calculate_motor_baseline([run]) == pytest.approx(350.0)


def test_motor_baseline_skips_empty_run_tables():
    run = stage1_run(condition="RT", raw_rts=[300.0, 400.0])
    empty = run.iloc[:0]

    assert calculate_motor_baseline([empty, run]) == pytest.approx(350.0)


@pytest.mark.parametrize(
    "tables",
    [[], [stage1_run(condition="RT", raw_rts=[None, None])]],
    ids=["no_runs", "no_responses"],
)
def test_motor_baseline_refuses_to_invent_a_value(tables):
    with pytest.raises(ValueError, match="No valid RT trials"):
        calculate_motor_baseline(tables)


def test_motor_baseline_by_side_separates_the_two_hands():
    run = stage1_run(
        condition="RT",
        raw_rts=[300.0, 500.0, 320.0, 520.0],
        choices=[1, 2, 1, 2],
    )

    baselines = calculate_motor_baseline_by_side([run])

    assert baselines == {1: pytest.approx(310.0), 2: pytest.approx(510.0)}
    # The pooled estimator averages over both hands and so hides the split.
    assert calculate_motor_baseline([run]) == pytest.approx(410.0)


def test_motor_baseline_by_side_weights_runs_equally_like_the_pooled_form():
    first = stage1_run(condition="RT", raw_rts=[300.0, 400.0], choices=[1, 1])
    second = stage1_run(condition="RT", run=2, raw_rts=[380.0], choices=[1])

    # mean(350, 380), not the trial-pooled mean(300, 400, 380) = 360.
    assert calculate_motor_baseline_by_side([first, second])[1] == pytest.approx(365.0)


def test_motor_baseline_by_side_omits_a_hand_with_no_response():
    run = stage1_run(condition="RT", raw_rts=[300.0, 400.0], choices=[1, 1])

    # Callers fall back to the pooled baseline for the missing hand rather
    # than receiving an invented value for it.
    assert set(calculate_motor_baseline_by_side([run])) == {1}


@pytest.mark.parametrize(
    "tables",
    [[], [stage1_run(condition="RT", raw_rts=[None, None])]],
    ids=["no_runs", "no_responses"],
)
def test_motor_baseline_by_side_refuses_to_invent_a_value(tables):
    with pytest.raises(ValueError, match="No valid RT trials"):
        calculate_motor_baseline_by_side(tables)


# --- decision times -----------------------------------------------------------


def test_decision_time_removes_the_motor_baseline_from_each_response():
    run = stage1_run(raw_rts=[1000.0, 1200.0, 1100.0])

    assert calculate_decision_times(run, 360.0).tolist() == [640.0, 840.0, 740.0]


def test_decision_time_keeps_the_original_index_and_skips_no_responses():
    run = stage1_run(raw_rts=[1000.0, None, 1100.0])

    decision_times = calculate_decision_times(run, 360.0)

    assert decision_times.index.tolist() == [0, 2]
    assert decision_times.tolist() == [640.0, 740.0]


def test_anticipations_are_kept_as_negative_decision_times():
    """A response faster than the motor baseline is flagged later, not dropped."""
    run = stage1_run(raw_rts=[200.0])

    assert calculate_decision_times(run, 360.0).tolist() == [-160.0]


# --- logged success probability at commitment ---------------------------------


def _trial(raw_rt, probabilities, times):
    return pd.Series({
        "nProb": list(probabilities),
        "tTime": list(times),
        "tEnterTarget": GO_CUE_MS + raw_rt,
    })


def test_logged_spd_uses_the_last_token_visible_at_commitment():
    # Commitment lands at 1500 ms once the 400 ms baseline is removed.
    probability, index = logged_spd_at_decision(
        _trial(900.0, [0.6, 0.7, 0.8], [1200, 1400, 1600]),
        motor_baseline=400.0,
    )

    assert probability == 0.7
    assert index == 1


def test_logged_spd_is_the_prior_when_commitment_precedes_the_first_token():
    probability, index = logged_spd_at_decision(
        _trial(500.0, [0.6], [1200]),
        motor_baseline=400.0,
    )

    assert probability == 0.5
    assert index is None


def test_the_motor_baseline_shifts_commitment_into_the_token_log_clock():
    """The same response time reads a different token once the motor latency
    is removed, so the baseline is not a cosmetic correction."""
    trial = _trial(900.0, [0.6, 0.7, 0.8], [1200, 1400, 1600])

    assert logged_spd_at_decision(trial, motor_baseline=0.0)[1] == 2
    assert logged_spd_at_decision(trial, motor_baseline=400.0)[1] == 1


def test_logged_spd_rejects_a_token_log_that_does_not_advance():
    with pytest.raises(ValueError, match="strictly increasing"):
        logged_spd_at_decision(
            _trial(900.0, [0.6, 0.7], [1200, 1200]),
            motor_baseline=0.0,
        )


# --- the canonical feature table ----------------------------------------------


BASELINE = 350.0


def _built(**run_kwargs):
    run = stage1_run(**run_kwargs)
    subject = str(run["subject"].iloc[0])
    return build_trial_features({subject: [run]}, {subject: BASELINE})


def test_per_hand_baselines_are_applied_to_the_hand_that_answered():
    run = stage1_run(
        condition="Fast",
        raw_rts=[900.0, 900.0],
        choices=[1, 2],
    )
    subject = str(run["subject"].iloc[0])

    pooled = build_trial_features({subject: [run]}, {subject: 350.0})
    per_hand = build_trial_features(
        {subject: [run]}, {subject: 350.0}, {subject: {1: 300.0, 2: 400.0}}
    )

    # Pooled: both hands lose the same constant, so the identical rawRTs give
    # identical decision times and any hand difference survives into dt_ms.
    assert pooled["dt_ms"].tolist() == pytest.approx([550.0, 550.0])
    # Per hand: each trial loses its own hand's latency.
    assert per_hand["dt_ms"].tolist() == pytest.approx([600.0, 500.0])
    assert per_hand["motor_baseline_ms"].tolist() == pytest.approx([300.0, 400.0])


def test_per_hand_baselines_fall_back_to_pooled_for_a_missing_hand():
    run = stage1_run(condition="Fast", raw_rts=[900.0, 900.0], choices=[1, 2])
    subject = str(run["subject"].iloc[0])

    # Only the left hand has an RT-run estimate; the right must not be dropped
    # or invented, it falls back to the pooled baseline.
    features = build_trial_features(
        {subject: [run]}, {subject: 350.0}, {subject: {1: 300.0}}
    )

    assert features["motor_baseline_ms"].tolist() == pytest.approx([300.0, 350.0])


def test_build_trial_features_emits_the_declared_schema():
    features = _built(raw_rts=[900.0, 1100.0])

    assert list(features.columns) == TRIAL_FEATURE_COLUMNS
    validate_trial_features(features)


def test_every_staged_trial_keeps_a_feature_row():
    """Selection is expressed with flags; no observation is deleted here."""
    features = _built(
        raw_rts=[900.0, None, None],
        outcomes=[0, 7006, OUTCOME_NEVER_STARTED],
    )

    assert len(features) == 3
    assert features["is_started"].tolist() == [True, True, False]
    assert features["has_choice"].tolist() == [True, False, False]
    assert features["is_no_response"].tolist() == [False, True, False]


def test_the_join_key_is_one_based_and_positional_within_the_run():
    features = _built(raw_rts=[900.0, 1100.0, 1000.0], first_trial_index=7)

    assert features["run_trial_index"].tolist() == [1, 2, 3]
    # The acquisition counter is preserved separately, not used as the key.
    assert features["nTrialIndex"].tolist() == [7, 8, 9]


def test_the_started_trial_index_counts_only_trials_that_received_a_go_cue():
    features = _built(
        raw_rts=[900.0, None, 1100.0],
        outcomes=[0, OUTCOME_NEVER_STARTED, 0],
    )

    assert features["started_trial_index"].tolist()[0] == 1
    assert pd.isna(features["started_trial_index"].iloc[1])
    assert features["started_trial_index"].tolist()[2] == 2


def test_decision_time_is_the_response_minus_the_subject_baseline():
    features = _built(raw_rts=[900.0, 1100.0])

    assert features["dt_ms"].tolist() == [550.0, 750.0]
    assert features["motor_baseline_ms"].tolist() == [BASELINE, BASELINE]


def test_the_rt_baseline_run_gets_no_decision_time_or_logged_spd():
    """RT runs measure motor latency; they carry no decision to time."""
    features = _built(condition="RT", raw_rts=[340.0, 360.0])

    assert features["dt_ms"].isna().all()
    assert features["logged_spd"].isna().all()
    assert not features["primary_analysis_eligible"].any()


def test_anticipations_are_flagged_rather_than_removed():
    features = _built(raw_rts=[200.0, 900.0])

    assert features["dt_ms"].tolist() == [-150.0, 550.0]
    assert features["dt_anticipation"].tolist() == [True, False]
    assert features["primary_analysis_eligible"].tolist() == [True, True]


def test_logged_spd_and_its_derived_scales_agree():
    features = _built(raw_rts=[1000.0])
    row = features.iloc[0]

    # Commitment at 1000 + 1000 - 350 = 1650 ms: the token logged at 1600 ms.
    assert row["decision_token_index"] == 3
    assert row["logged_spd"] == pytest.approx(0.62)
    assert row["evidence_at_decision"] == pytest.approx(0.62 - 0.5)
    assert row["logged_spd_log_odds"] == pytest.approx(np.log(0.62 / 0.38))


def test_a_decision_before_the_first_token_reports_index_zero():
    features = _built(raw_rts=[100.0])

    assert features["decision_token_index"].tolist() == [0]
    assert features["logged_spd"].tolist() == [0.5]


def test_the_validated_spd_view_is_withheld_on_a_short_token_log():
    """A 14-row log cannot be aligned to the 15-jump design, so the strict view
    is empty rather than silently reusing the all-logged value."""
    short = _built(raw_rts=[1000.0], probabilities=[0.6] * 14)
    complete = _built(raw_rts=[1000.0])

    assert short["token_log_rows"].tolist() == [14]
    assert short["logged_spd"].notna().all()
    assert short["logged_spd_validated_15row"].isna().all()
    assert not short["design_time_alignment_valid"].any()
    assert complete["logged_spd_validated_15row"].notna().all()
    assert complete["design_time_alignment_valid"].all()


def test_design_evidence_is_measured_against_the_correct_target():
    features = _built(
        raw_rts=[1000.0, 1000.0],
        correct_choices=[1, 2],
        token_directions="112" + "1" * 12,
    )

    # After three jumps target 1 leads 2-1 and target 2 trails by the same
    # amount, so their early evidence is symmetric about chance.
    leading, trailing = features["sp_design_early"].tolist()
    assert leading > 0.5 > trailing
    assert leading + trailing == pytest.approx(1.0)

    leading_odds, trailing_odds = features["sum_log_lr_design_early"].tolist()
    assert leading_odds == pytest.approx(-trailing_odds)


def test_the_token_lead_counts_only_the_jumps_the_subject_had_seen():
    features = _built(raw_rts=[1000.0], token_directions="111" + "2" * 12)

    # Commitment after three jumps, all of which favoured target 1.
    assert features["decision_token_index"].tolist() == [3]
    assert features["token_lead_at_decision"].tolist() == [3]


def test_design_evidence_is_withheld_without_a_usable_correct_target():
    features = _built(raw_rts=[900.0], correct_choices=[0])

    row = features.iloc[0]
    assert pd.isna(row["sp_design_early"])
    assert pd.isna(row["sum_log_lr_design_early"])
    assert pd.isna(row["token_lead_at_decision"])
    assert pd.isna(row["correct_side"])


def test_outcome_codes_are_translated_to_their_labview_labels():
    features = _built(raw_rts=[900.0, None], outcomes=[0, 7006])

    assert features["outcome_label"].tolist() == [
        "completed",
        "reaction_time_too_long",
    ]


def test_trial_classes_carry_their_names_and_unclassified_stays_unclassified():
    features = _built(
        raw_rts=[900.0] * 4,
        trial_classes=[1, 2, 3, 0],
    )

    assert features["trial_class_name"].tolist() == [
        "easy", "ambiguous", "misleading", "unclassified",
    ]


def test_features_from_several_subjects_are_ordered_and_keyed_apart():
    runs = {
        "H02": [stage1_run(subject="H02", raw_rts=[900.0])],
        "H01": [
            stage1_run(subject="H01", condition="Fast", raw_rts=[900.0]),
            stage1_run(subject="H01", condition="Slow", raw_rts=[1300.0]),
        ],
    }

    features = build_trial_features(runs, {"H01": 300.0, "H02": 400.0})

    assert features["subject"].tolist() == ["H01", "H01", "H02"]
    assert features["motor_baseline_ms"].tolist() == [300.0, 300.0, 400.0]
    assert not features["trial_id"].duplicated().any()
    validate_trial_features(features)


def test_a_subject_without_a_motor_baseline_is_an_error():
    runs = {"H01": [stage1_run(raw_rts=[900.0])]}

    with pytest.raises(KeyError):
        build_trial_features(runs, {"H02": 350.0})
