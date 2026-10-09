"""Tests for meg_tokens.behavior.analyses.sequential."""

import numpy as np
import pandas as pd
import pytest

from meg_tokens.behavior.analyses.sequential import (
    choice_history,
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


# --- choice history -----------------------------------------------------------


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
