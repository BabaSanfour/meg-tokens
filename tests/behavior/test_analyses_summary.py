"""Tests for meg_tokens.behavior.analyses.summary."""

import numpy as np
import pandas as pd
import pytest

from meg_tokens.behavior.analyses.summary import summarize_behavior
from meg_tokens.behavior.schema import OUTCOME_NEVER_STARTED

from tests.behavior.factories import trial_features


def _session(**overrides) -> pd.DataFrame:
    """Two RT trials plus four Fast and four Slow task trials for one subject."""
    columns = {
        "condition": ["RT"] * 2 + ["Fast"] * 4 + ["Slow"] * 4,
        "run": [1] * 2 + [2] * 4 + [3] * 4,
        "run_trial_index": [1, 2] + [1, 2, 3, 4] * 2,
        "rawRT": [340.0, 360.0] + [1250.0, 1450.0, 1350.0, 1300.0]
        + [1650.0, 1850.0, 1750.0, 1700.0],
        "dt_ms": [float("nan")] * 2
        + [900.0, 1100.0, 1000.0, 950.0]
        + [1300.0, 1500.0, 1400.0, 1350.0],
        "trial_class": [0, 0] + [1, 2, 3, 1] * 2,
        "trial_class_name": ["unclassified"] * 2
        + ["easy", "ambiguous", "misleading", "easy"] * 2,
        "isCorrect": [True, True] + [True, True, False, True] * 2,
        "logged_spd": [float("nan")] * 2 + [0.9, 0.7, 0.6, 0.8] * 2,
        "primary_analysis_eligible": [False] * 2 + [True] * 8,
    }
    columns.update(overrides)
    return trial_features(**columns)


def test_the_summary_has_one_row_per_subject_sorted_by_identifier():
    features = pd.concat(
        [
            _session(subject=["H02"] * 10),
            _session(subject=["H01"] * 10),
        ],
        ignore_index=True,
    )

    summary = summarize_behavior(features)

    assert summary["subject"].tolist() == ["H01", "H02"]


def test_the_summary_reports_the_subjects_motor_baseline():
    summary = summarize_behavior(_session()).iloc[0]

    assert summary["motor_baseline_ms"] == pytest.approx(350.0)


def test_a_subject_must_have_exactly_one_finite_motor_baseline():
    features = _session()
    features.loc[0, "motor_baseline_ms"] = 400.0

    with pytest.raises(ValueError, match="exactly one finite motor baseline"):
        summarize_behavior(features)


def test_trial_counts_separate_started_trials_from_usable_decision_times():
    features = _session()
    features.loc[2, ["dt_ms", "logged_spd"]] = float("nan")
    features.loc[2, "primary_analysis_eligible"] = False
    summary = summarize_behavior(features).iloc[0]

    # The Fast run still has four started trials, but only three timed ones.
    assert summary["n_fast_trials"] == 4
    assert summary["n_fast_dt_trials"] == 3
    assert summary["n_slow_trials"] == summary["n_slow_dt_trials"] == 4


def test_never_started_trials_are_counted_and_excluded_from_the_condition():
    features = _session()
    features.loc[2, "is_started"] = False
    features.loc[2, "primary_analysis_eligible"] = False
    features.loc[2, "dt_ms"] = float("nan")

    summary = summarize_behavior(features).iloc[0]

    assert summary["n_never_started_trials"] == 1
    assert summary["n_fast_trials"] == 3


def test_the_rt_run_contributes_baseline_trials_and_no_decision_times():
    summary = summarize_behavior(_session()).iloc[0]

    assert summary["n_rt_trials"] == 2
    assert summary["n_rt_baseline_trials"] == 2


def test_condition_means_come_from_the_declared_speed_summary():
    summary = summarize_behavior(_session()).iloc[0]

    assert summary["mean_fast_dt_ms"] == pytest.approx(987.5)
    assert summary["mean_slow_dt_ms"] == pytest.approx(1387.5)


def test_accuracy_is_reported_overall_and_within_each_condition():
    summary = summarize_behavior(_session()).iloc[0]

    assert summary["percent_correct"] == pytest.approx(75.0)
    assert summary["n_correct_trials"] == 6
    assert summary["n_error_trials"] == 2
    assert summary["fast_percent_correct"] == pytest.approx(75.0)
    assert summary["slow_percent_correct"] == pytest.approx(75.0)
    assert summary["fast_percent_error"] == pytest.approx(25.0)
    assert summary["n_fast_error_trials"] == summary["n_slow_error_trials"] == 1


def test_class_decision_times_and_counts_are_reported_for_every_class():
    summary = summarize_behavior(_session()).iloc[0]

    assert summary["n_easy_trials"] == 4
    assert summary["n_ambiguous_trials"] == summary["n_misleading_trials"] == 2
    assert summary["mean_easy_dt_ms"] == pytest.approx(
        (900.0 + 950.0 + 1300.0 + 1350.0) / 4
    )
    assert summary["mean_ambiguous_dt_ms"] == pytest.approx(1300.0)


def test_logged_spd_is_summarized_overall_and_by_class():
    summary = summarize_behavior(_session()).iloc[0]

    assert summary["n_spd_all_logged"] == 8
    assert summary["mean_spd_all_logged"] == pytest.approx(0.75)
    assert summary["mean_easy_spd_all_logged"] == pytest.approx(0.85)
    assert summary["mean_ambiguous_spd_all_logged"] == pytest.approx(0.7)


def test_the_validated_view_is_empty_when_no_trial_has_a_full_token_log():
    features = _session(token_log_rows=[14] * 10)
    features["logged_spd_validated_15row"] = float("nan")

    summary = summarize_behavior(features).iloc[0]

    # The strict view withholds rather than falling back to the logged value.
    assert summary["n_spd_all_logged"] == 8
    assert summary["n_spd_validated_15row"] == 0
    assert np.isnan(summary["mean_spd_validated_15row"])


def test_the_two_spd_views_agree_when_every_log_is_complete():
    summary = summarize_behavior(_session()).iloc[0]

    assert summary["n_spd_validated_15row"] == summary["n_spd_all_logged"]
    assert summary["mean_spd_validated_15row"] == pytest.approx(
        summary["mean_spd_all_logged"]
    )


def test_post_error_decision_times_are_carried_into_the_summary():
    summary = summarize_behavior(_session()).iloc[0]

    # Trial 4 of each run follows that run's error on trial 3.
    assert summary["n_post_error_trials"] == 2
    assert summary["mean_post_error_dt_ms"] == pytest.approx(1150.0)
    assert summary["n_post_correct_trials"] == 4


def test_anticipations_are_reported_without_being_removed():
    features = _session()
    features.loc[2, "dt_ms"] = -50.0

    summary = summarize_behavior(features).iloc[0]

    assert summary["n_fast_dt_anticipations"] == 1
    assert summary["n_fast_dt_trials"] == 4


# --- refusals -----------------------------------------------------------------


def test_an_empty_feature_table_is_refused_rather_than_summarized():
    with pytest.raises(ValueError, match="at least one trial"):
        summarize_behavior(trial_features(dt_ms=[]))


def test_an_undeclared_condition_is_refused():
    features = _session()
    features.loc[2, "condition"] = "Medium"

    with pytest.raises(ValueError, match=r"unknown behavioral conditions: \['medium'\]"):
        summarize_behavior(features)


def test_a_table_that_fails_the_feature_contract_is_refused():
    features = _session()
    features["is_started"] = "true"

    with pytest.raises(ValueError, match="is_started must be boolean"):
        summarize_behavior(features)
