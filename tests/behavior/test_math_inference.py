"""Tests for meg_tokens.behavior.math.inference."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from meg_tokens.behavior.math.inference import (
    confidence_interval,
    confidence_interval_from_t,
    equivalence_bound,
    one_sample_statistics,
    paired_subject_statistics,
    repeated_measures_anova,
    subject_bootstrap,
)


def test_an_interval_recovered_from_t_matches_the_direct_computation():
    values = np.array([2.0, 5.0, -1.0, 4.0, 3.0, 7.0, 0.5, 2.5])
    statistics = one_sample_statistics(values)

    low, high = confidence_interval_from_t(
        statistics["mean"], statistics["t"], statistics["df"], coverage=0.90
    )

    half_width = stats.t.ppf(0.95, statistics["df"]) * statistics["sem"]
    assert low == pytest.approx(statistics["mean"] - half_width)
    assert high == pytest.approx(statistics["mean"] + half_width)


def test_a_wider_coverage_gives_a_wider_interval():
    narrow = confidence_interval_from_t(2.0, 4.0, 31.0, coverage=0.90)
    wide = confidence_interval_from_t(2.0, 4.0, 31.0, coverage=0.99)

    assert wide[0] < narrow[0] and narrow[1] < wide[1]


def test_the_interval_is_symmetric_about_its_estimate_whatever_the_sign():
    low, high = confidence_interval_from_t(-10.4, -1.15, 31.0, coverage=0.90)

    assert (low + high) / 2 == pytest.approx(-10.4)
    assert low < -10.4 < high


@pytest.mark.parametrize(
    "estimate, t_statistic, df",
    [
        (1.0, 0.0, 31.0),        # zero t hides the standard error entirely
        (1.0, 2.0, 0.0),         # no degrees of freedom
        (np.nan, 2.0, 31.0),     # missing estimate
        (1.0, np.nan, 31.0),     # missing test statistic
    ],
)
def test_an_unrecoverable_interval_is_missing_rather_than_guessed(
    estimate, t_statistic, df
):
    low, high = confidence_interval_from_t(estimate, t_statistic, df)

    assert np.isnan(low) and np.isnan(high)


# --- paired subject statistics ------------------------------------------------


def test_paired_statistics_match_scipy_and_describe_both_conditions():
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


def test_the_contrast_direction_is_always_a_minus_b():
    result = paired_subject_statistics([5.0, 7.0, 9.0], [1.0, 2.0, 6.0])

    assert result["mean_difference"] == pytest.approx(4.0)
    assert result["t"] > 0
    assert result["cohens_dz"] > 0


def test_cohens_dz_is_the_paired_difference_divided_by_its_own_deviation():
    first = [1.0, 2.0, 4.0]
    second = [2.0, 4.0, 8.0]

    result = paired_subject_statistics(first, second)

    differences = np.array(first) - np.array(second)
    assert result["cohens_dz"] == pytest.approx(
        differences.mean() / differences.std(ddof=1)
    )
    # dz and t differ only by the square root of the sample size.
    assert result["t"] == pytest.approx(result["cohens_dz"] * np.sqrt(3))


def test_a_subject_missing_either_value_is_dropped_from_the_pair():
    result = paired_subject_statistics(
        [1.0, 2.0, float("nan"), 4.0],
        [2.0, 4.0, 6.0, float("inf")],
    )

    assert result["n_subjects"] == 2
    assert result["mean_a"] == pytest.approx(1.5)


def test_a_constant_difference_has_no_effect_size_but_still_has_a_test():
    result = paired_subject_statistics([2.0, 3.0, 4.0], [1.0, 2.0, 3.0])

    assert result["mean_difference"] == pytest.approx(1.0)
    assert np.isnan(result["cohens_dz"])


@pytest.mark.parametrize(
    ("first", "second"), [([], []), ([1.0], [2.0])], ids=["empty", "one_subject"]
)
def test_too_few_pairs_report_missing_inference_rather_than_a_number(first, second):
    result = paired_subject_statistics(first, second)

    assert result["n_subjects"] == len(first)
    assert all(np.isnan(result[field]) for field in ("t", "p", "df", "cohens_dz"))


def test_mismatched_inputs_are_an_error_rather_than_a_silent_truncation():
    with pytest.raises(ValueError, match="identical shapes"):
        paired_subject_statistics([1.0, 2.0], [1.0])


# --- one-sample statistics ----------------------------------------------------


def test_one_sample_statistics_match_scipy_against_zero():
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


def test_one_sample_statistics_drop_subjects_without_a_finite_estimate():
    result = one_sample_statistics([1.0, float("nan"), 3.0, float("inf")])

    assert result["n_subjects"] == 2
    assert result["mean"] == pytest.approx(2.0)


def test_an_empty_sample_is_reported_explicitly():
    result = one_sample_statistics([])

    assert result["n_subjects"] == 0
    assert all(
        np.isnan(result[field])
        for field in ("mean", "sem", "t", "p", "df", "cohens_dz")
    )


def test_a_single_subject_yields_a_mean_but_no_test():
    result = one_sample_statistics([2.0])

    assert result["n_subjects"] == 1
    assert result["mean"] == pytest.approx(2.0)
    assert all(np.isnan(result[field]) for field in ("sem", "t", "p", "cohens_dz"))


def test_a_sample_without_variance_has_no_effect_size():
    result = one_sample_statistics([2.0, 2.0, 2.0])

    assert result["mean"] == pytest.approx(2.0)
    assert np.isnan(result["cohens_dz"])


# --- equivalence bound --------------------------------------------------------


def test_the_bound_is_the_furthest_endpoint_of_its_own_interval():
    values = [0.4, -0.2, 0.9, 0.1, -0.5, 0.3]
    statistics = one_sample_statistics(values)
    low, high = confidence_interval(
        statistics["mean"], statistics["sem"], statistics["df"], coverage=0.90
    )

    result = equivalence_bound(values)

    assert result["equivalence_bound"] == pytest.approx(max(abs(low), abs(high)))
    assert result["equivalence_bound_coverage"] == 0.90
    assert result["equivalence_bound_dz"] == pytest.approx(
        result["equivalence_bound"] / np.std(values, ddof=1)
    )


def test_the_bound_does_not_return_an_interval_of_its_own():
    """A row usually carries an interval already, often at another coverage."""
    result = equivalence_bound([0.4, -0.2, 0.9, 0.1])

    assert "ci_low" not in result and "ci_high" not in result
    assert "coverage" not in result


def test_a_wider_coverage_excludes_less():
    values = [0.4, -0.2, 0.9, 0.1, -0.5, 0.3]

    assert (
        equivalence_bound(values, coverage=0.99)["equivalence_bound"]
        > equivalence_bound(values, coverage=0.90)["equivalence_bound"]
    )


@pytest.mark.parametrize("values", [[], [1.0], [np.nan, 2.0]])
def test_too_few_finite_values_emit_no_columns_rather_than_a_number(values):
    assert equivalence_bound(values) == {}


# --- subject bootstrap --------------------------------------------------------


def test_the_bootstrap_interval_on_a_mean_matches_the_t_interval():
    """The one case with a closed form: on a plain mean the percentile interval
    should land where the t interval does, or the resampling is wrong. They
    converge in the subject count -- at n = 10 the percentile interval is
    visibly the narrower of the two, which is the known small-sample behaviour
    and the reason this is checked at a realistic n instead."""
    values = np.linspace(-2.0, 8.0, 60)
    result = subject_bootstrap(
        lambda index: values[index].mean(axis=-1), len(values), n_resamples=8000
    )
    statistics = one_sample_statistics(values)

    assert result["estimate"] == pytest.approx(statistics["mean"])
    assert result["ci_low"] == pytest.approx(statistics["ci95_low"], abs=0.1)
    assert result["ci_high"] == pytest.approx(statistics["ci95_high"], abs=0.1)


def test_the_same_seed_gives_the_same_interval_twice():
    """A derivative that changes on re-run is not a derivative."""
    values = np.array([0.3, -0.1, 0.8, 0.2, -0.4, 0.5, 0.1, 0.9])

    def run():
        return subject_bootstrap(
            lambda index: values[index].mean(axis=-1), len(values),
            n_resamples=500, seed=7,
        )

    assert run() == run()


def test_two_statistics_share_one_resample_so_their_difference_can_cancel():
    """The point of driving both from one index array: a difference between
    two quantities measured on the same subjects is far better determined than
    two independent draws would make it."""
    first = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    # Identical up to a constant, so every resample shifts them together.
    second = first + 10.0

    difference = subject_bootstrap(
        lambda index: second[index].mean(axis=-1) - first[index].mean(axis=-1),
        len(first), n_resamples=2000,
    )

    assert difference["estimate"] == pytest.approx(10.0)
    # Each mean alone varies by more than a unit across resamples; the
    # difference does not vary at all.
    assert difference["ci_low"] == pytest.approx(10.0)
    assert difference["ci_high"] == pytest.approx(10.0)


def test_the_achieved_significance_level_cannot_report_zero():
    """It is bounded below by the resolution of the resampling, and saying so
    is the difference between a small p and a p the bootstrap cannot see."""
    values = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    result = subject_bootstrap(
        lambda index: values[index].mean(axis=-1), len(values), n_resamples=1000
    )

    assert result["p"] == pytest.approx(2.0 / 1001)


def test_the_bound_is_read_off_a_narrower_interval_than_the_one_reported():
    values = np.array([0.4, -0.2, 0.9, 0.1, -0.5, 0.3, 0.7, -0.1])
    result = subject_bootstrap(
        lambda index: values[index].mean(axis=-1), len(values), n_resamples=4000
    )

    assert result["coverage"] == 0.95
    assert result["equivalence_bound_coverage"] == 0.90
    assert result["equivalence_bound"] < max(
        abs(result["ci_low"]), abs(result["ci_high"])
    )


@pytest.mark.parametrize("n_subjects", [0, 1])
def test_too_few_subjects_yield_nan_rather_than_a_number(n_subjects):
    result = subject_bootstrap(lambda index: np.ones(np.shape(index)[:-1]), n_subjects)

    assert result["n_subjects"] == n_subjects
    assert np.isnan(result["estimate"])
    assert np.isnan(result["ci_low"]) and np.isnan(result["p"])


def test_a_statistic_that_never_resolves_yields_nan_rather_than_raising():
    """A vanishing denominator is the realistic case: the caller returns NaN
    and the interval has to come back undefined, not empty or exploded."""
    result = subject_bootstrap(lambda index: np.full(np.shape(index)[:-1], np.nan), 8, n_resamples=50)

    assert np.isnan(result["ci_low"]) and np.isnan(result["ci_high"])
    assert result["n_subjects"] == 8


# --- repeated-measures ANOVA --------------------------------------------------


def _cells() -> np.ndarray:
    """Five subjects by a 2 x 3 within-subject design, row-major cells."""
    return np.array(
        [
            [10.0, 14.0, 19.0, 12.0, 17.0, 20.0],
            [11.0, 16.0, 18.0, 15.0, 18.0, 24.0],
            [9.0, 13.0, 21.0, 11.0, 16.0, 23.0],
            [12.0, 15.0, 20.0, 14.0, 19.0, 22.0],
            [8.0, 12.0, 17.0, 13.0, 15.0, 21.0],
        ]
    )


def _effects(matrix, levels=(2, 3)):
    return {row["effect"]: row for row in repeated_measures_anova(matrix, levels)}


def test_the_anova_reports_every_main_effect_and_interaction():
    effects = _effects(_cells())

    assert set(effects) == {"factor1", "factor2", "factor1xfactor2"}
    assert effects["factor1"]["df_effect"] == 1.0
    assert effects["factor2"]["df_effect"] == 2.0
    assert effects["factor1xfactor2"]["df_effect"] == 2.0
    assert effects["factor1xfactor2"]["df_error"] == 8.0


def test_a_two_level_main_effect_equals_the_squared_paired_t():
    matrix = _cells()
    expected = stats.ttest_rel(matrix[:, :3].mean(axis=1), matrix[:, 3:].mean(axis=1))

    effects = _effects(matrix)

    assert effects["factor1"]["F"] == pytest.approx(float(expected.statistic) ** 2)
    assert effects["factor1"]["p"] == pytest.approx(float(expected.pvalue))


def test_the_interaction_carries_none_of_the_main_effects():
    """A raw cell-mean deviation still contains both main effects. The 2x3
    interaction must equal a one-way test on the by-subject differences between
    the two levels of the first factor.
    """
    matrix = _cells()
    differences = matrix[:, :3] - matrix[:, 3:]
    n_subjects, n_levels = differences.shape
    grand = differences.mean()
    level_means = differences.mean(axis=0)
    subject_means = differences.mean(axis=1)
    residual = differences - level_means[None, :] - subject_means[:, None] + grand
    expected = (
        n_subjects * np.sum((level_means - grand) ** 2) / (n_levels - 1)
    ) / (np.sum(residual**2) / ((n_levels - 1) * (n_subjects - 1)))

    assert _effects(matrix)["factor1xfactor2"]["F"] == pytest.approx(float(expected))


def test_shifting_one_condition_changes_only_its_own_main_effect():
    matrix = _cells()
    shifted = matrix.copy()
    shifted[:, 3:] += 100.0

    baseline = _effects(matrix)
    moved = _effects(shifted)

    assert moved["factor1"]["F"] > baseline["factor1"]["F"]
    assert moved["factor2"]["F"] == pytest.approx(baseline["factor2"]["F"])
    assert moved["factor1xfactor2"]["F"] == pytest.approx(
        baseline["factor1xfactor2"]["F"]
    )


def test_partial_eta_squared_is_the_effect_share_of_its_own_error_term():
    effect = _effects(_cells())["factor2"]

    assert effect["partial_eta_squared"] == pytest.approx(
        effect["sum_squares_effect"]
        / (effect["sum_squares_effect"] + effect["sum_squares_error"])
    )
    assert 0.0 <= effect["partial_eta_squared"] <= 1.0


def test_a_subject_missing_any_cell_is_removed_to_keep_the_design_balanced():
    matrix = _cells()
    incomplete = np.vstack([matrix, [1.0, 2.0, 3.0, 4.0, 5.0, np.nan]])

    assert _effects(incomplete)["factor1"]["n_subjects"] == 5


def test_the_anova_refuses_a_design_it_cannot_balance():
    with pytest.raises(ValueError, match="at least two complete subjects"):
        repeated_measures_anova(_cells()[:1], (2, 3))


@pytest.mark.parametrize(
    ("matrix", "levels", "message"),
    [
        (np.zeros((5,)), (2, 3), "must be a .* matrix"),
        (np.zeros((5, 6)), (2, 4), "must describe every column"),
    ],
)
def test_the_anova_rejects_a_mismatched_cell_matrix(matrix, levels, message):
    with pytest.raises(ValueError, match=message):
        repeated_measures_anova(matrix, levels)
