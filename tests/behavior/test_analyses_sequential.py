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

    # The classical contrast is reported beside the robust one.
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


def test_a_pair_needs_both_neighbours_of_the_error_in_the_same_run():
    gapped = trial_features(
        run_trial_index=[1, 2, 5, 6],
        dt_ms=[900.0, 1000.0, 950.0, 1300.0],
        isCorrect=[True, False, False, True],
    )

    overall = _pooled(robust_post_error_slowing(gapped))

    # Trial 2 is an error, but trial 5 is not its neighbour.
    assert overall["n_error_pairs"] == 0
    assert np.isnan(overall["robust_pes_ms"])

    never_started = trial_features(
        run_trial_index=[1, 2, 3],
        dt_ms=[900.0, float("nan"), 1300.0],
        isCorrect=pd.array([False, None, True], dtype="boolean"),
        is_started=[True, False, True],
        has_choice=[True, False, True],
        primary_analysis_eligible=[True, False, True],
    )

    overall = _pooled(robust_post_error_slowing(never_started))

    # A never-started trial breaks the chain rather than being skipped.
    assert overall["n_error_pairs"] == 0
    assert overall["n_post_error_trials"] == 0

    at_boundary = trial_features(
        run_trial_index=[1, 2, 3],
        dt_ms=[900.0, 1000.0, 1300.0],
        isCorrect=[False, True, False],
    )

    assert _pooled(robust_post_error_slowing(at_boundary))["n_error_pairs"] == 0


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

    assert set(statistics["measure"]) == {
        "robust_pes_ms",
        "classical_pes_ms",
        # The local pair is DT(error+1) - DT(error-1), which is post-error
        # slowing only if pre-error trials are typical. Both baseline-relative
        # components are tested so the composite is never reported as if it
        # were slowing alone.
        "post_error_minus_baseline_ms",
        "pre_error_minus_baseline_ms",
        # Descriptive, not a test: how far the two conventional measures track
        # each other, which the write-up quotes and so must not recompute.
        "robust_vs_classical_agreement",
    }
    assert set(statistics["condition"]) == {"all", "fast", "slow", "fast_vs_slow"}
    robust = statistics.loc[
        (statistics["measure"] == "robust_pes_ms")
        & (statistics["condition"] == "fast")
    ].iloc[0]
    assert robust["n_subjects"] == 4
    assert robust["mean"] > 0
    assert robust["ci95_low"] < robust["mean"] < robust["ci95_high"]

    # They require the subject table they document.
    with pytest.raises(ValueError, match="classical_pes_ms"):
        post_error_statistics(_post_error_table().drop(columns=["classical_pes_ms"]))


def test_the_two_post_error_measures_carry_a_registered_agreement_row():
    """The write-up argues from the correlation and the sign counts, so they
    are a derivative row rather than something each consumer recomputes."""
    table = _post_error_table()
    statistics = post_error_statistics(table)
    row = statistics.loc[
        (statistics["measure"] == "robust_vs_classical_agreement")
        & (statistics["condition"] == "all")
    ].iloc[0]

    pooled = table.loc[table["condition"] == "all"]
    assert row["n_subjects"] == len(pooled)
    assert row["pearson_r"] == pytest.approx(
        pooled["robust_pes_ms"].corr(pooled["classical_pes_ms"])
    )
    assert row["robust_positive"] == int((pooled["robust_pes_ms"] > 0).sum())
    assert row["classical_positive"] == int((pooled["classical_pes_ms"] > 0).sum())
    # It is descriptive: a correlation between two measures of the same thing
    # is not a hypothesis, and a p on it would only ever be misread.
    assert row["test"] == "descriptive"
    assert pd.isna(row.get("p", float("nan")))


def test_local_pair_slowing_decomposes_into_its_two_baseline_effects():
    """`robust_pes_ms` is a composite, and the components must reconstruct it.

    DT(error+1) - DT(error-1) is post-error slowing only when pre-error trials
    are typical. Against the real cohort they are not: pre-error trials run
    fast, so a quarter of the composite is pre-error speeding. The two
    baseline-relative columns exist so the figure can state that split instead
    of attributing all of it to slowing.
    """
    table = _post_error_table()
    pooled = table.loc[table["condition"] == "all"]

    reconstructed = (
        pooled["post_error_minus_baseline_ms"] - pooled["pre_error_minus_baseline_ms"]
    )
    assert np.allclose(reconstructed, pooled["robust_pes_ms"], equal_nan=True)
    assert pooled["mean_baseline_dt_ms"].notna().all()


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

    across_runs = trial_features(
        condition=["Fast"] * 2 + ["Slow"] * 2,
        run=[1, 1, 2, 2],
        run_trial_index=[1, 2, 1, 2],
        choice_side=[1, 1, 2, 2],
        dt_ms=[900.0, 1000.0, 1300.0, 1400.0],
    )

    # History never carries across a run or condition boundary: only the second
    # trial of each run is linked to a predecessor.
    assert _pooled(choice_history(across_runs))["n_linked_trials"] == 2


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

    invariant = trial_features(
        run_trial_index=[1, 2, 3, 4],
        choice_side=[1, 1, 1, 1],
        dt_ms=[900.0] * 4,
    )

    # It is withheld rather than reported when a side never varies.
    assert np.isnan(_pooled(choice_history(invariant))["side_autocorrelation_lag1"])


def test_choice_history_rejects_a_table_it_cannot_read_as_a_sequence():
    noncanonical_correctness = trial_features(
        run_trial_index=[1, 2],
        isCorrect=["true", "false"],
    )

    with pytest.raises(ValueError, match="isCorrect must be boolean"):
        choice_history(noncanonical_correctness)

    impossible_side = trial_features(
        run_trial_index=[1, 2, 3],
        choice_side=[1, 3, 2],
        dt_ms=[900.0, 1000.0, 1100.0],
    )

    with pytest.raises(ValueError, match="choice_side must contain only 1, 2"):
        choice_history(impossible_side)

    duplicated_position = trial_features(
        run_trial_index=[1, 1],
        subject=["H01", "H01"],
        condition=["Fast", "Fast"],
        run=[1, 1],
    )

    with pytest.raises(ValueError, match="unique within each task run"):
        choice_history(duplicated_position)


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
    }
    assert set(statistics["condition"]) == {"all", "fast"}
    assert (statistics["n_subjects"] <= 4).all()

    # They require the subject table they document.
    with pytest.raises(ValueError, match="lose_stay"):
        choice_history_statistics(pd.DataFrame({"condition": ["all"], "win_stay": [1.0]}))
