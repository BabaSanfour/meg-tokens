"""Tests for meg_tokens.behavior.analyses.performance."""

import numpy as np
import pandas as pd
import pytest

from meg_tokens.behavior.analyses.performance import (
    analyze_post_error_slowing,
    analyze_trial_classes,
    behavior_group_statistics,
    compare_correct_error,
    compare_fast_slow,
)

from tests.behavior.factories import trial_features


# --- Fast versus Slow ---------------------------------------------------------


def test_fast_and_slow_decision_times_are_summarized_separately():
    features = trial_features(
        condition=["Fast", "Fast", "Slow", "Slow"],
        dt_ms=[640.0, 840.0, 1140.0, 1440.0],
    )

    speed = compare_fast_slow(features)

    assert (speed["n_fast"], speed["n_slow"]) == (2, 2)
    assert speed["mean_fast"] == pytest.approx(740.0)
    assert speed["mean_slow"] == pytest.approx(1290.0)


def test_the_rt_baseline_run_never_contributes_a_decision_time():
    features = trial_features(
        condition=["RT", "Fast"],
        dt_ms=[float("nan"), 640.0],
        primary_analysis_eligible=[False, True],
    )

    speed = compare_fast_slow(features)

    assert (speed["n_fast"], speed["n_slow"]) == (1, 0)
    assert np.isnan(speed["mean_slow"])


def test_anticipations_are_counted_but_still_enter_the_mean():
    """Declared policy for the preprint comparison: nothing is trimmed."""
    features = trial_features(
        condition=["Fast", "Fast", "Slow"],
        dt_ms=[-100.0, 900.0, -50.0],
    )

    speed = compare_fast_slow(features)

    assert speed["n_fast_anticipations"] == 1
    assert speed["n_slow_anticipations"] == 1
    assert speed["mean_fast"] == pytest.approx(400.0)


def test_an_empty_condition_reports_a_missing_mean_not_a_zero():
    speed = compare_fast_slow(trial_features(dt_ms=[], condition=[]))

    assert (speed["n_fast"], speed["n_slow"]) == (0, 0)
    assert np.isnan(speed["mean_fast"]) and np.isnan(speed["mean_slow"])


# --- trial classes ------------------------------------------------------------


def test_decision_time_is_summarized_for_each_design_class():
    features = trial_features(
        trial_class=[1, 2, 3, 1],
        dt_ms=[640.0, 840.0, 740.0, 690.0],
    )

    classes = analyze_trial_classes(features)

    assert classes["counts"] == {"easy": 2, "ambiguous": 1, "misleading": 1}
    assert classes["means"]["easy"] == pytest.approx(665.0)
    assert classes["means"]["ambiguous"] == pytest.approx(840.0)
    assert classes["means"]["misleading"] == pytest.approx(740.0)


def test_unclassified_trials_are_excluded_rather_than_assigned_a_class():
    features = trial_features(
        trial_class=[1, 0],
        dt_ms=[640.0, 5000.0],
    )

    classes = analyze_trial_classes(features)

    assert classes["counts"] == {"easy": 1, "ambiguous": 0, "misleading": 0}
    assert classes["means"]["easy"] == pytest.approx(640.0)


def test_an_empty_class_reports_a_missing_mean():
    classes = analyze_trial_classes(trial_features(trial_class=[1], dt_ms=[640.0]))

    assert np.isnan(classes["means"]["ambiguous"])


# --- correct versus error -----------------------------------------------------


def test_correct_and_error_decision_times_and_accuracy_are_reported():
    features = trial_features(
        dt_ms=[640.0, 840.0, 740.0, 690.0],
        isCorrect=[True, True, False, True],
    )

    accuracy = compare_correct_error(features)

    assert (accuracy["n_correct"], accuracy["n_error"]) == (3, 1)
    assert accuracy["mean_correct"] == pytest.approx((640 + 840 + 690) / 3)
    assert accuracy["mean_error"] == pytest.approx(740.0)
    assert accuracy["percent_correct"] == pytest.approx(75.0)


def test_accuracy_can_be_restricted_to_one_declared_condition():
    features = trial_features(
        condition=["Fast", "Fast", "Slow", "Slow"],
        dt_ms=[640.0, 840.0, 1140.0, 1440.0],
        isCorrect=[True, False, True, True],
    )

    assert compare_correct_error(features, condition="Fast")["percent_correct"] == (
        pytest.approx(50.0)
    )
    assert compare_correct_error(features, condition="slow")["percent_correct"] == (
        pytest.approx(100.0)
    )


def test_an_undeclared_condition_is_refused():
    with pytest.raises(ValueError, match=r"condition must be one of \['fast', 'slow'\]"):
        compare_correct_error(trial_features(dt_ms=[640.0]), condition="RT")


def test_a_trial_without_a_defined_outcome_leaves_the_accuracy_denominator():
    features = trial_features(
        dt_ms=[640.0, 840.0, 740.0],
        isCorrect=pd.array([True, False, None], dtype="boolean"),
    )

    accuracy = compare_correct_error(features)

    assert (accuracy["n_correct"], accuracy["n_error"]) == (1, 1)
    assert accuracy["percent_correct"] == pytest.approx(50.0)


def test_correctness_is_read_as_a_boolean_not_through_truthiness():
    """The string ``"false"`` is truthy in Python; accepting it would silently
    turn every error into a correct trial."""
    features = trial_features(dt_ms=[640.0, 840.0], isCorrect=["true", "false"])

    with pytest.raises(ValueError, match="isCorrect must be boolean"):
        compare_correct_error(features)


def test_accuracy_is_missing_rather_than_zero_without_any_response():
    accuracy = compare_correct_error(trial_features(dt_ms=[], isCorrect=[]))

    assert (accuracy["n_correct"], accuracy["n_error"]) == (0, 0)
    assert np.isnan(accuracy["percent_correct"])


# --- classical post-error slowing ---------------------------------------------


def test_post_error_slowing_compares_the_trials_that_follow_each_outcome():
    features = trial_features(
        run_trial_index=[1, 2, 3, 4],
        dt_ms=[640.0, 840.0, 740.0, 690.0],
        isCorrect=[True, True, False, True],
    )

    slowing = analyze_post_error_slowing(features)

    # Trials 2 and 3 follow a correct trial; trial 4 follows the error.
    assert (slowing["n_post_correct"], slowing["n_post_error"]) == (2, 1)
    assert slowing["mean_post_correct"] == pytest.approx(790.0)
    assert slowing["mean_post_error"] == pytest.approx(690.0)


def test_adjacency_does_not_cross_a_run_or_condition_boundary():
    features = trial_features(
        condition=["Fast", "Fast", "Slow", "Slow"],
        run=[1, 1, 2, 2],
        run_trial_index=[1, 2, 1, 2],
        dt_ms=[640.0, 840.0, 1140.0, 1440.0],
        isCorrect=[True, False, True, False],
    )

    slowing = analyze_post_error_slowing(features)

    # Only trial 2 of each run follows anything; the last Fast error does not
    # pair with the first Slow trial.
    assert (slowing["n_post_correct"], slowing["n_post_error"]) == (2, 0)


def test_pairs_are_formed_in_trial_order_regardless_of_row_order():
    ordered = trial_features(
        run_trial_index=[1, 2, 3],
        dt_ms=[640.0, 840.0, 740.0],
        isCorrect=[True, False, True],
    )

    assert analyze_post_error_slowing(ordered.iloc[::-1]) == (
        analyze_post_error_slowing(ordered)
    )


def test_a_pair_is_dropped_when_either_side_is_undefined():
    features = trial_features(
        run_trial_index=[1, 2, 3, 4],
        dt_ms=[640.0, float("nan"), 740.0, 690.0],
        isCorrect=pd.array([True, None, True, True], dtype="boolean"),
    )

    slowing = analyze_post_error_slowing(features)

    # Trial 2 has no decision time and trial 3 follows an undefined outcome.
    assert (slowing["n_post_correct"], slowing["n_post_error"]) == (1, 0)
    assert slowing["mean_post_correct"] == pytest.approx(690.0)


def test_never_started_trials_do_not_become_a_neighbour():
    features = trial_features(
        run_trial_index=[1, 2, 3],
        dt_ms=[640.0, float("nan"), 740.0],
        isCorrect=pd.array([False, None, True], dtype="boolean"),
        is_started=[True, False, True],
        has_choice=[True, False, True],
        primary_analysis_eligible=[True, False, True],
    )

    slowing = analyze_post_error_slowing(features)

    # Trial 3 directly follows the error on trial 1 once the never-started
    # trial between them is removed.
    assert (slowing["n_post_correct"], slowing["n_post_error"]) == (0, 1)
    assert slowing["mean_post_error"] == pytest.approx(740.0)


def test_post_error_slowing_is_missing_rather_than_zero_without_pairs():
    slowing = analyze_post_error_slowing(trial_features(dt_ms=[640.0]))

    assert (slowing["n_post_correct"], slowing["n_post_error"]) == (0, 0)
    assert np.isnan(slowing["mean_post_correct"])
    assert np.isnan(slowing["mean_post_error"])


# --- the fixed group contrasts ------------------------------------------------


def _summary() -> pd.DataFrame:
    subjects = ["H01", "H02", "H03", "H04"]
    columns = {
        "subject": subjects,
        "mean_fast_dt_ms": [900.0, 1000.0, 1100.0, 1200.0],
        "mean_slow_dt_ms": [1100.0, 1300.0, 1400.0, 1700.0],
        "n_fast_error_trials": [10, 12, 9, 14],
        "n_slow_error_trials": [6, 7, 5, 9],
    }
    # Class separation varies by subject, so paired contrasts have non-zero
    # variance in their differences and a defined effect size.
    for index, name in enumerate(("easy", "ambiguous", "misleading")):
        columns[f"mean_{name}_dt_ms"] = [
            900.0 + 100 * index * (1 + 0.2 * subject) + 50 * subject
            for subject in range(4)
        ]
        for view in ("all_logged", "validated_15row"):
            columns[f"mean_{name}_spd_{view}"] = [
                0.9 - 0.05 * index * (1 + 0.3 * subject) - 0.01 * subject
                for subject in range(4)
            ]
    return pd.DataFrame(columns)


def test_the_declared_contrast_set_is_computed_for_every_view():
    statistics = behavior_group_statistics(_summary())

    keys = set(zip(statistics["analysis"], statistics["contrast"], statistics["view"]))
    assert ("decision_time", "fast_vs_slow", "primary") in keys
    assert ("error_count", "fast_vs_slow", "primary") in keys
    for pair in ("easy_vs_ambiguous", "easy_vs_misleading", "ambiguous_vs_misleading"):
        assert ("decision_time", pair, "primary") in keys
        assert ("spd", pair, "all_logged") in keys
        assert ("spd", pair, "validated_15row") in keys
    assert len(statistics) == len(keys) == 11


def test_each_contrast_carries_its_unit_labels_and_source_columns():
    statistics = behavior_group_statistics(_summary()).set_index(
        ["analysis", "contrast", "view"]
    )
    row = statistics.loc[("decision_time", "fast_vs_slow", "primary")]

    assert row["unit"] == "ms"
    assert (row["label_a"], row["label_b"]) == ("Fast", "Slow")
    assert (row["column_a"], row["column_b"]) == ("mean_fast_dt_ms", "mean_slow_dt_ms")
    assert statistics.loc[("error_count", "fast_vs_slow", "primary"), "unit"] == "trials"


def test_the_contrast_statistics_pair_subjects_by_row():
    summary = _summary()
    statistics = behavior_group_statistics(summary).set_index(
        ["analysis", "contrast", "view"]
    )
    row = statistics.loc[("decision_time", "fast_vs_slow", "primary")]

    assert row["n_subjects"] == 4
    assert row["mean_a"] == pytest.approx(summary["mean_fast_dt_ms"].mean())
    assert row["mean_b"] == pytest.approx(summary["mean_slow_dt_ms"].mean())
    assert row["mean_difference"] == pytest.approx(row["mean_a"] - row["mean_b"])


def test_error_counts_are_contrasted_as_counts_not_rates():
    summary = _summary()
    statistics = behavior_group_statistics(summary).set_index(
        ["analysis", "contrast", "view"]
    )
    row = statistics.loc[("error_count", "fast_vs_slow", "primary")]

    assert row["mean_a"] == pytest.approx(summary["n_fast_error_trials"].mean())


def test_a_subject_missing_one_side_of_a_contrast_is_dropped_from_it_only():
    summary = _summary()
    summary.loc[0, "mean_slow_dt_ms"] = float("nan")

    statistics = behavior_group_statistics(summary).set_index(
        ["analysis", "contrast", "view"]
    )

    assert statistics.loc[("decision_time", "fast_vs_slow", "primary"), "n_subjects"] == 3
    assert statistics.loc[("error_count", "fast_vs_slow", "primary"), "n_subjects"] == 4


def test_a_summary_missing_a_declared_column_is_refused():
    summary = _summary().drop(columns=["mean_easy_spd_validated_15row"])

    with pytest.raises(ValueError, match="mean_easy_spd_validated_15row"):
        behavior_group_statistics(summary)
