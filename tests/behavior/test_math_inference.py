"""Tests for meg_tokens.behavior.math.inference.

These estimators are the shared statistical layer every analysis reports
through, so each test pins one contract end to end rather than one field.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from meg_tokens.behavior.math.inference import (
    one_sample_statistics,
    paired_subject_statistics,
    repeated_measures_anova,
    subject_bootstrap,
)


# --- paired subject statistics ------------------------------------------------


def test_paired_statistics_match_scipy_describe_both_conditions_and_run_a_minus_b():
    first = [1.0, 2.0, 4.0]
    second = [2.0, 4.0, 8.0]
    expected = stats.ttest_rel(first, second)

    result = paired_subject_statistics(pd.Series(first), pd.Series(second))

    assert result["n_subjects"] == 3
    assert result["df"] == 2.0
    assert result["mean_a"] == pytest.approx(7 / 3)
    assert result["mean_b"] == pytest.approx(14 / 3)
    assert result["sem_a"] == pytest.approx(np.std(first, ddof=1) / np.sqrt(3))
    assert result["t"] == pytest.approx(float(expected.statistic))
    assert result["p"] == pytest.approx(float(expected.pvalue))

    # dz is the paired difference over its own deviation, so it differs from t
    # only by the square root of the sample size.
    differences = np.array(first) - np.array(second)
    assert result["cohens_dz"] == pytest.approx(differences.mean() / differences.std(ddof=1))
    assert result["t"] == pytest.approx(result["cohens_dz"] * np.sqrt(3))

    # The contrast direction is always a minus b.
    forward = paired_subject_statistics([5.0, 7.0, 9.0], [1.0, 2.0, 6.0])
    assert forward["mean_difference"] == pytest.approx(4.0)
    assert forward["t"] > 0 and forward["cohens_dz"] > 0


# --- one-sample statistics ----------------------------------------------------


def test_one_sample_statistics_match_scipy_and_degrade_explicitly():
    values = [1.0, 2.0, 4.0, 3.0, 2.5]
    expected = stats.ttest_1samp(values, 0.0)

    result = one_sample_statistics(values)

    assert result["n_subjects"] == 5
    assert result["df"] == 4.0
    assert result["mean"] == pytest.approx(2.5)
    assert result["sem"] == pytest.approx(np.std(values, ddof=1) / np.sqrt(5))
    assert result["t"] == pytest.approx(float(expected.statistic))
    assert result["p"] == pytest.approx(float(expected.pvalue))
    assert result["cohens_dz"] == pytest.approx(2.5 / np.std(values, ddof=1))

    dropped = one_sample_statistics([1.0, float("nan"), 3.0, float("inf")])
    assert dropped["n_subjects"] == 2
    assert dropped["mean"] == pytest.approx(2.0)

    empty = one_sample_statistics([])
    assert empty["n_subjects"] == 0
    assert all(
        np.isnan(empty[field]) for field in ("mean", "sem", "t", "p", "df", "cohens_dz")
    )

    single = one_sample_statistics([2.0])
    assert single["n_subjects"] == 1
    assert single["mean"] == pytest.approx(2.0)
    assert all(np.isnan(single[field]) for field in ("sem", "t", "p", "cohens_dz"))

    # A cell where no subject differs has no test. scipy reports t = inf,
    # p = 0 for it -- and t ~ 1e16, p ~ 1e-32 once rounding leaves a residue,
    # as [0.1] * 3 does -- which reads as overwhelming evidence. Two real
    # conditional-accuracy cells (every subject at 100%) were reported so.
    for flat_values in ([2.0, 2.0, 2.0], [0.1, 0.1, 0.1]):
        flat = one_sample_statistics(flat_values)
        assert flat["mean"] == pytest.approx(flat_values[0])
        assert flat["sem"] == 0.0
        assert all(np.isnan(flat[field]) for field in ("t", "p", "cohens_dz"))
    # Real spread, however small, is still tested.
    assert np.isfinite(one_sample_statistics([1.0, 1.0, 1.000001])["p"])


# --- subject bootstrap --------------------------------------------------------


def _mean_of(values):
    return lambda index: values[index].mean(axis=-1)


def test_the_bootstrap_interval_on_a_mean_matches_the_t_interval_and_is_reproducible():
    """The one case with a closed form: on a plain mean the percentile interval
    should land where the t interval does, or the resampling is wrong. They
    converge in the subject count — at n = 10 the percentile interval is
    visibly the narrower of the two, which is the known small-sample behaviour
    and the reason this is checked at a realistic n instead."""
    values = np.linspace(-2.0, 8.0, 60)
    result = subject_bootstrap(_mean_of(values), len(values), n_resamples=8000)
    statistics = one_sample_statistics(values)
    assert result["estimate"] == pytest.approx(statistics["mean"])
    assert result["ci_low"] == pytest.approx(statistics["ci95_low"], abs=0.1)
    assert result["ci_high"] == pytest.approx(statistics["ci95_high"], abs=0.1)

    # A derivative that changes on re-run is not a derivative.
    seeded = np.array([0.3, -0.1, 0.8, 0.2, -0.4, 0.5, 0.1, 0.9])
    run = lambda: subject_bootstrap(_mean_of(seeded), len(seeded), n_resamples=500, seed=7)
    assert run() == run()


def test_two_statistics_share_one_resample_so_their_difference_can_cancel():
    """The point of driving both from one index array: a difference between two
    quantities measured on the same subjects is far better determined than two
    independent draws would make it."""
    first = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    second = first + 10.0  # identical up to a constant

    difference = subject_bootstrap(
        lambda index: second[index].mean(axis=-1) - first[index].mean(axis=-1),
        len(first), n_resamples=2000,
    )

    assert difference["estimate"] == pytest.approx(10.0)
    # Each mean alone varies by more than a unit across resamples; the
    # difference does not vary at all.
    assert difference["ci_low"] == pytest.approx(10.0)
    assert difference["ci_high"] == pytest.approx(10.0)


# --- repeated-measures ANOVA --------------------------------------------------


def _cells() -> np.ndarray:
    """Five subjects by a 2 x 3 within-subject design, row-major cells."""
    return np.array([
        [10.0, 14.0, 19.0, 12.0, 17.0, 20.0],
        [11.0, 16.0, 18.0, 15.0, 18.0, 24.0],
        [9.0, 13.0, 21.0, 11.0, 16.0, 23.0],
        [12.0, 15.0, 20.0, 14.0, 19.0, 22.0],
        [8.0, 12.0, 17.0, 13.0, 15.0, 21.0],
    ])


def _effects(matrix, levels=(2, 3)):
    return {row["effect"]: row for row in repeated_measures_anova(matrix, levels)}


def test_each_effect_is_tested_against_its_own_error_term_and_carries_none_of_the_others():
    matrix = _cells()
    effects = _effects(matrix)

    assert set(effects) == {"factor1", "factor2", "factor1xfactor2"}
    assert effects["factor1"]["df_effect"] == 1.0
    assert effects["factor2"]["df_effect"] == 2.0
    assert effects["factor1xfactor2"]["df_effect"] == 2.0
    assert effects["factor1xfactor2"]["df_error"] == 8.0

    # A two-level main effect is the squared paired t on the level means.
    paired = stats.ttest_rel(matrix[:, :3].mean(axis=1), matrix[:, 3:].mean(axis=1))
    assert effects["factor1"]["F"] == pytest.approx(float(paired.statistic) ** 2)
    assert effects["factor1"]["p"] == pytest.approx(float(paired.pvalue))

    # A raw cell-mean deviation still contains both main effects. The 2x3
    # interaction must equal a one-way test on the by-subject differences
    # between the two levels of the first factor.
    differences = matrix[:, :3] - matrix[:, 3:]
    n_subjects, n_levels = differences.shape
    grand = differences.mean()
    level_means = differences.mean(axis=0)
    subject_means = differences.mean(axis=1)
    residual = differences - level_means[None, :] - subject_means[:, None] + grand
    expected = (n_subjects * np.sum((level_means - grand) ** 2) / (n_levels - 1)) / (
        np.sum(residual**2) / ((n_levels - 1) * (n_subjects - 1))
    )
    assert effects["factor1xfactor2"]["F"] == pytest.approx(float(expected))

    # Shifting one condition moves only its own main effect.
    shifted = matrix.copy()
    shifted[:, 3:] += 100.0
    moved = _effects(shifted)
    assert moved["factor1"]["F"] > effects["factor1"]["F"]
    assert moved["factor2"]["F"] == pytest.approx(effects["factor2"]["F"])
    assert moved["factor1xfactor2"]["F"] == pytest.approx(effects["factor1xfactor2"]["F"])

    effect = effects["factor2"]
    assert effect["partial_eta_squared"] == pytest.approx(
        effect["sum_squares_effect"]
        / (effect["sum_squares_effect"] + effect["sum_squares_error"])
    )
    assert 0.0 <= effect["partial_eta_squared"] <= 1.0
