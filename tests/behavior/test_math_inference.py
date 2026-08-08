"""Tests for meg_tokens.behavior.math.inference."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from meg_tokens.behavior.math.inference import (
    one_sample_statistics,
    paired_subject_statistics,
    repeated_measures_anova,
)


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
