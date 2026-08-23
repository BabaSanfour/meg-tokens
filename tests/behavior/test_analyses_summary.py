"""Tests for meg_tokens.behavior.analyses.summary."""

import numpy as np
import pandas as pd
import pytest

from meg_tokens.behavior.analyses.summary import summarize_behavior

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


def test_the_summary_reports_the_motor_baseline_that_was_removed():
    assert summarize_behavior(_session()).iloc[0]["motor_baseline_ms"] == (
        pytest.approx(350.0)
    )

    # Per-hand baselines are the intended shape: two values, but each one
    # constant within its own response side.
    features = _session()
    left = features["choice_side"] == 1
    features.loc[left, "motor_baseline_ms"] = 300.0
    features.loc[~left, "motor_baseline_ms"] = 400.0

    summary = summarize_behavior(features).iloc[0]

    # Reported as the mean over trials -- the latency actually removed.
    assert summary["motor_baseline_ms"] == pytest.approx(
        features["motor_baseline_ms"].mean()
    )


def test_a_motor_baseline_must_be_one_finite_value_per_response_side():
    two_on_one_side = _session()
    same_side = two_on_one_side.index[
        two_on_one_side["choice_side"] == two_on_one_side.loc[0, "choice_side"]
    ]
    two_on_one_side.loc[same_side[0], "motor_baseline_ms"] = 400.0
    two_on_one_side.loc[same_side[1], "motor_baseline_ms"] = 500.0

    with pytest.raises(ValueError, match="more than one motor baseline"):
        summarize_behavior(two_on_one_side)

    missing = _session()
    missing.loc[0, "motor_baseline_ms"] = float("nan")

    with pytest.raises(ValueError, match="finite motor baseline"):
        summarize_behavior(missing)


def test_trial_counts_separate_started_trials_from_usable_decision_times():
    untimed = _session()
    untimed.loc[2, ["dt_ms", "logged_spd"]] = float("nan")
    untimed.loc[2, "primary_analysis_eligible"] = False

    summary = summarize_behavior(untimed).iloc[0]

    # The Fast run still has four started trials, but only three timed ones.
    assert summary["n_fast_trials"] == 4
    assert summary["n_fast_dt_trials"] == 3
    assert summary["n_slow_trials"] == summary["n_slow_dt_trials"] == 4

    never_started = _session()
    never_started.loc[2, "is_started"] = False
    never_started.loc[2, "primary_analysis_eligible"] = False
    never_started.loc[2, "dt_ms"] = float("nan")

    summary = summarize_behavior(never_started).iloc[0]

    assert summary["n_never_started_trials"] == 1
    assert summary["n_fast_trials"] == 3

    # The RT run contributes baseline trials and no decision times.
    summary = summarize_behavior(_session()).iloc[0]
    assert summary["n_rt_trials"] == 2
    assert summary["n_rt_baseline_trials"] == 2


def test_decision_times_and_accuracy_are_reported_by_condition_and_by_class():
    summary = summarize_behavior(_session()).iloc[0]

    assert summary["mean_fast_dt_ms"] == pytest.approx(987.5)
    assert summary["mean_slow_dt_ms"] == pytest.approx(1387.5)

    assert summary["percent_correct"] == pytest.approx(75.0)
    assert summary["n_correct_trials"] == 6
    assert summary["n_error_trials"] == 2
    assert summary["fast_percent_correct"] == pytest.approx(75.0)
    assert summary["slow_percent_correct"] == pytest.approx(75.0)
    assert summary["fast_percent_error"] == pytest.approx(25.0)
    assert summary["n_fast_error_trials"] == summary["n_slow_error_trials"] == 1

    assert summary["n_easy_trials"] == 4
    assert summary["n_ambiguous_trials"] == summary["n_misleading_trials"] == 2
    assert summary["mean_easy_dt_ms"] == pytest.approx(
        (900.0 + 950.0 + 1300.0 + 1350.0) / 4
    )
    assert summary["mean_ambiguous_dt_ms"] == pytest.approx(1300.0)


def test_logged_spd_is_summarized_overall_by_class_and_in_both_views():
    summary = summarize_behavior(_session()).iloc[0]

    assert summary["n_spd_all_logged"] == 8
    assert summary["mean_spd_all_logged"] == pytest.approx(0.75)
    assert summary["mean_easy_spd_all_logged"] == pytest.approx(0.85)
    assert summary["mean_ambiguous_spd_all_logged"] == pytest.approx(0.7)

    # The two views agree when every log is complete.
    assert summary["n_spd_validated_15row"] == summary["n_spd_all_logged"]
    assert summary["mean_spd_validated_15row"] == pytest.approx(
        summary["mean_spd_all_logged"]
    )

    short_logs = _session(token_log_rows=[14] * 10)
    short_logs["logged_spd_validated_15row"] = float("nan")

    summary = summarize_behavior(short_logs).iloc[0]

    # The strict view withholds rather than falling back to the logged value.
    assert summary["n_spd_all_logged"] == 8
    assert summary["n_spd_validated_15row"] == 0
    assert np.isnan(summary["mean_spd_validated_15row"])


def test_post_error_and_anticipated_trials_are_reported_without_being_removed():
    summary = summarize_behavior(_session()).iloc[0]

    # Trial 4 of each run follows that run's error on trial 3.
    assert summary["n_post_error_trials"] == 2
    assert summary["mean_post_error_dt_ms"] == pytest.approx(1150.0)
    assert summary["n_post_correct_trials"] == 4

    anticipated = _session()
    anticipated.loc[2, "dt_ms"] = -50.0

    summary = summarize_behavior(anticipated).iloc[0]

    assert summary["n_fast_dt_anticipations"] == 1
    assert summary["n_fast_dt_trials"] == 4


def test_a_table_the_summary_cannot_describe_is_refused():
    with pytest.raises(ValueError, match="at least one trial"):
        summarize_behavior(trial_features(dt_ms=[]))

    undeclared = _session()
    undeclared.loc[2, "condition"] = "Medium"

    with pytest.raises(ValueError, match=r"unknown behavioral conditions: \['medium'\]"):
        summarize_behavior(undeclared)

    off_contract = _session()
    off_contract["is_started"] = "true"

    with pytest.raises(ValueError, match="is_started must be boolean"):
        summarize_behavior(off_contract)
