"""Tests for meg_tokens.behavior.analyses.sequential."""

import numpy as np
import pandas as pd
import pytest

from meg_tokens.behavior.analyses.sequential import (
    choice_history,
    choice_history_statistics,
    post_error_statistics,
    robust_post_error_slowing,
)

from tests.behavior.factories import trial_features


def _pooled(table):
    return table.loc[table["condition"] == "all"].iloc[0]


# --- robust post-error slowing ------------------------------------------------


def test_the_robust_measure_straddles_the_error_within_the_session():
    """DT(error + 1) - DT(error - 1). Both sides come from the same moment, so
    a slow stretch of the session cannot masquerade as post-error slowing."""
    features = trial_features(
        run_trial_index=[1, 2, 3, 4, 5],
        dt_ms=[900.0, 1000.0, 950.0, 1300.0, 1000.0],
        isCorrect=[True, True, False, True, True],
    )

    overall = _pooled(robust_post_error_slowing(features))

    assert overall["n_error_pairs"] == 1
    assert overall["mean_pre_error_dt_ms"] == pytest.approx(1000.0)
    assert overall["mean_post_error_dt_ms"] == pytest.approx(1300.0)
    assert overall["robust_pes_ms"] == pytest.approx(300.0)


def test_the_classical_contrast_is_reported_beside_the_robust_one():
    features = trial_features(
        run_trial_index=[1, 2, 3, 4, 5],
        dt_ms=[900.0, 1000.0, 950.0, 1300.0, 1000.0],
        isCorrect=[True, True, False, True, True],
    )

    overall = _pooled(robust_post_error_slowing(features))

    assert overall["n_post_error_trials"] == 1
    assert overall["n_post_correct_trials"] == 3
    # Mean post-error DT minus mean post-correct DT over the same trials.
    assert overall["classical_pes_ms"] == pytest.approx(
        1300.0 - np.mean([1000.0, 950.0, 1000.0])
    )


def test_the_error_trials_own_decision_time_is_not_part_of_the_pair():
    slow_error = trial_features(
        run_trial_index=[1, 2, 3],
        dt_ms=[1000.0, 5000.0, 1300.0],
        isCorrect=[True, False, True],
    )
    fast_error = trial_features(
        run_trial_index=[1, 2, 3],
        dt_ms=[1000.0, 100.0, 1300.0],
        isCorrect=[True, False, True],
    )

    assert _pooled(robust_post_error_slowing(slow_error))["robust_pes_ms"] == (
        _pooled(robust_post_error_slowing(fast_error))["robust_pes_ms"]
    )


def test_a_pair_does_not_span_a_gap_in_the_run():
    features = trial_features(
        run_trial_index=[1, 2, 5, 6],
        dt_ms=[900.0, 1000.0, 950.0, 1300.0],
        isCorrect=[True, False, False, True],
    )

    overall = _pooled(robust_post_error_slowing(features))

    # Trial 2 is an error, but trial 5 is not its neighbour.
    assert overall["n_error_pairs"] == 0
    assert np.isnan(overall["robust_pes_ms"])


def test_a_never_started_trial_breaks_the_chain_rather_than_being_skipped():
    features = trial_features(
        run_trial_index=[1, 2, 3],
        dt_ms=[900.0, float("nan"), 1300.0],
        isCorrect=pd.array([False, None, True], dtype="boolean"),
        is_started=[True, False, True],
        has_choice=[True, False, True],
        primary_analysis_eligible=[True, False, True],
    )

    overall = _pooled(robust_post_error_slowing(features))

    assert overall["n_error_pairs"] == 0
    assert overall["n_post_error_trials"] == 0


def test_an_error_at_a_run_boundary_has_no_pair():
    features = trial_features(
        run_trial_index=[1, 2, 3],
        dt_ms=[900.0, 1000.0, 1300.0],
        isCorrect=[False, True, False],
    )

    overall = _pooled(robust_post_error_slowing(features))

    assert overall["n_error_pairs"] == 0


def test_post_error_slowing_is_reported_pooled_and_per_condition():
    features = trial_features(
        condition=["Fast"] * 3 + ["Slow"] * 3,
        run=[1, 1, 1, 2, 2, 2],
        run_trial_index=[1, 2, 3] * 2,
        dt_ms=[900.0, 950.0, 1200.0, 1300.0, 1350.0, 1800.0],
        isCorrect=[True, False, True] * 2,
    )

    slowing = robust_post_error_slowing(features).set_index("condition")

    assert set(slowing.index) == {"all", "fast", "slow"}
    assert slowing.loc["fast", "robust_pes_ms"] == pytest.approx(300.0)
    assert slowing.loc["slow", "robust_pes_ms"] == pytest.approx(500.0)


def _post_error_table(n_subjects=4):
    return pd.concat(
        [
            robust_post_error_slowing(
                trial_features(
                    subject=[f"H{index:02d}"] * 6,
                    condition=["Fast"] * 3 + ["Slow"] * 3,
                    run=[1, 1, 1, 2, 2, 2],
                    run_trial_index=[1, 2, 3] * 2,
                    dt_ms=[
                        900.0, 950.0, 900.0 + 100.0 * index,
                        1300.0, 1350.0, 1300.0 + 150.0 * index,
                    ],
                    isCorrect=[True, False, True] * 2,
                )
            )
            for index in range(1, n_subjects + 1)
        ],
        ignore_index=True,
    )


def test_post_error_statistics_test_both_definitions_and_contrast_conditions():
    statistics = post_error_statistics(_post_error_table())

    assert set(statistics["measure"]) == {"robust_pes_ms", "classical_pes_ms"}
    assert set(statistics["condition"]) == {"all", "fast", "slow", "fast_vs_slow"}
    robust = statistics.loc[
        (statistics["measure"] == "robust_pes_ms")
        & (statistics["condition"] == "fast")
    ].iloc[0]
    assert robust["n_subjects"] == 4
    assert robust["mean"] > 0


def test_post_error_statistics_require_the_subject_table_they_document():
    incomplete = _post_error_table().drop(columns=["classical_pes_ms"])

    with pytest.raises(ValueError, match="classical_pes_ms"):
        post_error_statistics(incomplete)


# --- choice history -----------------------------------------------------------


def test_win_stay_and_lose_shift_are_measured_against_the_previous_outcome():
    features = trial_features(
        run_trial_index=[1, 2, 3, 4, 5],
        choice_side=[1, 1, 2, 2, 1],
        isCorrect=[True, False, True, False, True],
        dt_ms=[900.0, 1000.0, 1100.0, 1200.0, 1300.0],
    )

    overall = _pooled(choice_history(features))

    assert overall["n_linked_trials"] == 4
    # Trials 2 and 4 follow a correct trial and both repeat the side.
    assert overall["win_stay"] == pytest.approx(1.0)
    # Trials 3 and 5 follow an error and both switch.
    assert overall["lose_stay"] == pytest.approx(0.0)
    assert overall["lose_shift"] == pytest.approx(1.0)


def test_the_side_autocorrelation_detects_alternation_and_repetition():
    alternating = trial_features(
        run_trial_index=[1, 2, 3, 4, 5, 6],
        choice_side=[1, 2, 1, 2, 1, 2],
        dt_ms=[900.0] * 6,
    )
    repeating = trial_features(
        run_trial_index=[1, 2, 3, 4, 5, 6],
        choice_side=[1, 1, 1, 2, 2, 2],
        dt_ms=[900.0] * 6,
    )

    assert _pooled(choice_history(alternating))["side_autocorrelation_lag1"] == (
        pytest.approx(-1.0)
    )
    assert _pooled(choice_history(repeating))["side_autocorrelation_lag1"] > 0


def test_the_autocorrelation_is_withheld_when_a_side_never_varies():
    features = trial_features(
        run_trial_index=[1, 2, 3, 4],
        choice_side=[1, 1, 1, 1],
        dt_ms=[900.0] * 4,
    )

    assert np.isnan(_pooled(choice_history(features))["side_autocorrelation_lag1"])


def test_decision_time_is_summarized_by_the_previous_outcome():
    features = trial_features(
        run_trial_index=[1, 2, 3, 4, 5],
        choice_side=[1, 1, 2, 2, 1],
        isCorrect=[True, False, True, False, True],
        dt_ms=[900.0, 1000.0, 1100.0, 1200.0, 1300.0],
    )

    overall = _pooled(choice_history(features))

    assert overall["mean_dt_after_correct_ms"] == pytest.approx(1100.0)
    assert overall["mean_dt_after_error_ms"] == pytest.approx(1200.0)


def test_decision_time_is_summarized_by_the_previous_trial_class():
    features = trial_features(
        run_trial_index=[1, 2, 3, 4],
        trial_class=[1, 2, 3, 1],
        trial_class_name=["easy", "ambiguous", "misleading", "easy"],
        choice_side=[1, 1, 2, 2],
        dt_ms=[900.0, 1000.0, 1100.0, 1200.0],
    )

    overall = _pooled(choice_history(features))

    assert overall["mean_dt_after_easy_ms"] == pytest.approx(1000.0)
    assert overall["mean_dt_after_ambiguous_ms"] == pytest.approx(1100.0)
    assert overall["mean_dt_after_misleading_ms"] == pytest.approx(1200.0)


def test_history_never_carries_across_a_run_or_condition_boundary():
    features = trial_features(
        condition=["Fast"] * 2 + ["Slow"] * 2,
        run=[1, 1, 2, 2],
        run_trial_index=[1, 2, 1, 2],
        choice_side=[1, 1, 2, 2],
        dt_ms=[900.0, 1000.0, 1300.0, 1400.0],
    )

    overall = _pooled(choice_history(features))

    # Only the second trial of each run is linked to a predecessor.
    assert overall["n_linked_trials"] == 2


def test_choice_history_rejects_a_noncanonical_correctness_value():
    features = trial_features(
        run_trial_index=[1, 2],
        isCorrect=["true", "false"],
    )

    with pytest.raises(ValueError, match="isCorrect must be boolean"):
        choice_history(features)


def test_choice_history_rejects_an_impossible_choice_side():
    features = trial_features(
        run_trial_index=[1, 2, 3],
        choice_side=[1, 3, 2],
        dt_ms=[900.0, 1000.0, 1100.0],
    )

    with pytest.raises(ValueError, match="choice_side must contain only 1, 2"):
        choice_history(features)


def test_choice_history_rejects_a_duplicated_position_in_a_run():
    features = trial_features(
        run_trial_index=[1, 1],
        subject=["H01", "H01"],
        condition=["Fast", "Fast"],
        run=[1, 1],
    )

    with pytest.raises(ValueError, match="unique within each task run"):
        choice_history(features)


def test_choice_history_statistics_run_the_fixed_group_contrasts():
    history = pd.concat(
        [
            choice_history(
                trial_features(
                    subject=[f"H{index:02d}"] * 6,
                    run_trial_index=[1, 2, 3, 4, 5, 6],
                    trial_class=[1, 2, 3, 1, 2, 3],
                    trial_class_name=[
                        "easy", "ambiguous", "misleading",
                        "easy", "ambiguous", "misleading",
                    ],
                    choice_side=[1, 1, 2, 2, 1, 1],
                    isCorrect=[True, False, True, False, True, True],
                    dt_ms=[
                        900.0 + 10 * index, 1000.0, 1100.0 + 20 * index,
                        1200.0, 1300.0 + 30 * index, 1400.0,
                    ],
                )
            )
            for index in range(1, 5)
        ],
        ignore_index=True,
    )

    statistics = choice_history_statistics(history)

    assert set(statistics["measure"]) == {
        "win_stay_vs_lose_stay",
        "side_autocorrelation_lag1",
        "dt_after_error_vs_correct",
        "dt_after_easy_vs_ambiguous",
        "dt_after_easy_vs_misleading",
        "dt_after_ambiguous_vs_misleading",
    }
    assert set(statistics["condition"]) == {"all", "fast"}
    assert (statistics["n_subjects"] <= 4).all()


def test_choice_history_statistics_require_the_subject_table_they_document():
    history = pd.DataFrame({"condition": ["all"], "win_stay": [1.0]})

    with pytest.raises(ValueError, match="lose_stay"):
        choice_history_statistics(history)
