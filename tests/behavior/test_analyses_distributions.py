"""Tests for meg_tokens.behavior.analyses.distributions."""

import numpy as np
import pandas as pd
import pytest

from meg_tokens.behavior.analyses.distributions import (
    decision_time_geometry,
    decision_time_geometry_statistics,
    decision_time_distributions,
)

from tests.behavior.factories import trial_features


# --- decision-time distributions ----------------------------------------------


def _distribution_session(subject="H01", shift=0.0):
    conditions = ["Fast"] * 6 + ["Slow"] * 6
    classes = [1, 1, 2, 2, 3, 3] * 2
    return trial_features(
        subject=[subject] * 12,
        condition=conditions,
        run=[1] * 6 + [2] * 6,
        run_trial_index=list(range(1, 7)) * 2,
        trial_class=classes,
        trial_class_name=[
            {1: "easy", 2: "ambiguous", 3: "misleading"}[code] for code in classes
        ],
        dt_ms=[
            800.0 + shift, 900.0 + shift, 1200.0 + shift, 1300.0 + shift,
            1100.0 + shift, 1200.0 + shift,
            1000.0 + shift, 1100.0 + shift, 1400.0 + shift, 1500.0 + shift,
            1300.0 + shift, 1400.0 + shift,
        ],
    )


def test_the_distributions_are_built_from_dt_ms_and_ignore_rawrt():
    """Regression guard for the defect rule 2 exists to fix (docs/behavior.md,
    Figure conventions: decision time is `dt_ms`, never `rawRT`).

    The preprint's analysis used ``rawRT``, which carries the motor component.
    This asserts the property rather than the spelling: the quantiles track
    ``dt_ms`` exactly, and perturbing ``rawRT`` alone changes nothing. The
    figure-level check can only inspect a declared column name, which is why
    the real guarantee lives here.
    """
    features = _distribution_session()
    baseline = decision_time_distributions(features)

    fast = features.loc[features["condition"] == "Fast", "dt_ms"]
    row = baseline.loc[
        (baseline["stratum_type"] == "condition") & (baseline["stratum"] == "fast")
    ].iloc[0]
    assert row["q50"] == pytest.approx(float(fast.median()))
    assert row["mean"] == pytest.approx(float(fast.mean()))

    shifted = features.copy()
    shifted["rawRT"] = shifted["rawRT"] + 500.0
    pd.testing.assert_frame_equal(baseline, decision_time_distributions(shifted))


# --- decision-time geometry ---------------------------------------------------


def _two_condition_session(subject="H01", *, delay=0.0, factor=1.0, seed=0):
    """One subject whose Slow trials are Fast trials delayed and/or stretched.

    Built so the two models are separable: a delay moves the mean and leaves
    the spread, a stretch scales both. Slow is an *independent* draw from the
    transformed distribution, not the same trials transformed — Fast and Slow
    are different trials in the real design, and a noiseless one-to-one
    transform gives the quantile contrasts zero variance, which makes their t
    undefined rather than non-significant.
    """
    rng = np.random.default_rng(seed)
    fast = 600.0 + rng.gamma(shape=3.0, scale=150.0, size=120)
    slow = (600.0 + rng.gamma(shape=3.0, scale=150.0, size=120)) * factor + delay
    return trial_features(
        subject=[subject] * 240,
        condition=["Fast"] * 120 + ["Slow"] * 120,
        run=[1] * 120 + [2] * 120,
        run_trial_index=list(range(1, 121)) * 2,
        dt_ms=list(fast) + list(slow),
    )


def _geometry_rows(n_subjects=8, **kwargs):
    features = pd.concat(
        [
            _two_condition_session(f"H{index:02d}", seed=index, **kwargs)
            for index in range(1, n_subjects + 1)
        ],
        ignore_index=True,
    )
    statistics = decision_time_geometry_statistics(decision_time_geometry(features))
    return statistics.loc[statistics["contrast"] == "fast_vs_slow"]


def test_a_pure_delay_and_a_pure_stretch_are_each_identified_as_themselves():
    delay = _geometry_rows(delay=200.0).set_index("term")
    # A delay predicts constant quantile differences and changing ratios, and
    # leaves the SD alone, so the delay model's SD prediction is the right one.
    assert delay.loc["fixed_delay_q90_minus_q10_ms", "p"] > 0.05
    assert delay.loc["proportional_stretch_log_ratio_q90_minus_q10", "p"] < 0.05
    assert delay.loc["sd_observed_minus_delay_predicted", "p"] > 0.05
    assert delay.loc["sd_observed_minus_stretch_predicted", "p"] < 0.05

    stretch = _geometry_rows(factor=1.30).set_index("term")
    # Mirror image: ratios stable, differences not, and the factor recovered.
    assert stretch.loc["proportional_stretch_log_ratio_q90_minus_q10", "p"] > 0.05
    assert stretch.loc["fixed_delay_q90_minus_q10_ms", "p"] < 0.05
    assert stretch.loc["cv_observed_minus_stretch_predicted", "p"] > 0.05
    assert stretch.loc["cv_observed_minus_delay_predicted", "p"] < 0.05
    assert stretch.loc["stretch_factor", "estimate"] == pytest.approx(1.30, abs=0.03)


def test_the_ks_ladder_removes_the_transform_that_generated_the_difference():
    ladder = _geometry_rows(factor=1.40)
    ladder = ladder.loc[ladder["term"] == "ks_subjects_differing"].set_index("normalization")
    # Dividing out each side's median should remove a pure stretch; subtracting
    # a median should not, since a stretch is not an offset.
    assert ladder.loc["median_divided", "estimate"] < ladder.loc["raw", "estimate"]
    assert ladder.loc["median_divided", "estimate"] <= ladder.loc["median_subtracted", "estimate"]
    # A count is only readable against what chance produces.
    assert ladder.loc["raw", "n_expected_by_chance"] == pytest.approx(0.05 * 8)
