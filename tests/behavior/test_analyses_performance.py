"""Tests for meg_tokens.behavior.analyses.performance."""

import numpy as np
import pandas as pd
import pytest

from meg_tokens.behavior.analyses.performance import (
    analyze_post_error_slowing,
    compare_correct_error,
)

from tests.behavior.factories import trial_features


# --- correct versus error -----------------------------------------------------


def test_accuracy_is_missing_rather_than_zero_without_a_defined_outcome():
    features = trial_features(
        dt_ms=[640.0, 840.0, 740.0],
        isCorrect=pd.array([True, False, None], dtype="boolean"),
    )

    accuracy = compare_correct_error(features)

    # A trial without a defined outcome leaves the denominator.
    assert (accuracy["n_correct"], accuracy["n_error"]) == (1, 1)
    assert accuracy["percent_correct"] == pytest.approx(50.0)

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

    # Pairs are formed in trial order regardless of row order.
    ordered = trial_features(
        run_trial_index=[1, 2, 3],
        dt_ms=[640.0, 840.0, 740.0],
        isCorrect=[True, False, True],
    )

    assert analyze_post_error_slowing(ordered.iloc[::-1]) == (
        analyze_post_error_slowing(ordered)
    )
