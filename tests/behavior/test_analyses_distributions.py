"""Tests for meg_tokens.behavior.analyses.distributions."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from meg_tokens.behavior.analyses.distributions import (
    MIN_TRIALS_FOR_SHAPE,
    decision_time_distribution_statistics,
    decision_time_distributions,
    distribution_summary,
    ex_gaussian_parameters,
    spd_cumulative_distributions,
)

from tests.behavior.factories import trial_features


# --- descriptive summary ------------------------------------------------------


def test_the_summary_reports_moments_and_the_requested_quantiles():
    values = list(range(1, 25))

    summary = distribution_summary(values, quantiles=(0.25, 0.5, 0.75))

    assert summary["n_trials"] == 24
    assert summary["mean"] == pytest.approx(12.5)
    assert summary["sd"] == pytest.approx(np.std(values, ddof=1))
    assert summary["min"] == 1.0 and summary["max"] == 24.0
    assert summary["q25"] == pytest.approx(np.quantile(values, 0.25))
    assert summary["q50"] == pytest.approx(12.5)
    assert summary["q75"] == pytest.approx(np.quantile(values, 0.75))


def test_quantile_keys_are_named_by_integer_percentage():
    summary = distribution_summary(list(range(100)), quantiles=(0.1, 0.9))

    assert {"q10", "q90"} <= set(summary)


def test_shape_statistics_are_bias_corrected():
    values = list(range(1, 25))

    summary = distribution_summary(values)

    assert summary["skewness"] == pytest.approx(float(stats.skew(values, bias=False)))
    assert summary["kurtosis"] == pytest.approx(
        float(stats.kurtosis(values, bias=False))
    )


def test_shape_statistics_are_withheld_on_a_sample_too_small_to_trust_them():
    short = distribution_summary(list(range(MIN_TRIALS_FOR_SHAPE - 1)))
    enough = distribution_summary(list(range(MIN_TRIALS_FOR_SHAPE)))

    assert np.isnan(short["skewness"]) and np.isnan(short["kurtosis"])
    assert np.isfinite(enough["skewness"])


def test_non_finite_observations_are_discarded_before_summarizing():
    summary = distribution_summary([1.0, float("nan"), 3.0, float("inf")])

    assert summary["n_trials"] == 2
    assert summary["mean"] == pytest.approx(2.0)


def test_an_empty_sample_reports_missing_statistics_not_zeros():
    summary = distribution_summary([])

    assert summary["n_trials"] == 0
    assert all(
        np.isnan(summary[field]) for field in ("mean", "sd", "min", "max", "q50")
    )


# --- ex-Gaussian decomposition ------------------------------------------------


def test_the_ex_gaussian_fit_separates_the_body_from_the_tail():
    """``tau`` is the exponential tail a mean cannot distinguish from a shift
    in the body of the distribution."""
    generator = np.random.default_rng(0)
    values = generator.normal(800.0, 50.0, 400) + generator.exponential(200.0, 400)

    parameters = ex_gaussian_parameters(values)

    assert parameters["exgaussian_fitted"]
    assert parameters["exgaussian_mu"] == pytest.approx(800.0, abs=40.0)
    assert parameters["exgaussian_sigma"] == pytest.approx(50.0, abs=25.0)
    assert parameters["exgaussian_tau"] == pytest.approx(200.0, abs=50.0)


def test_the_fit_is_skipped_rather_than_forced_on_too_few_trials():
    parameters = ex_gaussian_parameters([900.0, 1000.0, 1100.0])

    assert not parameters["exgaussian_fitted"]
    assert np.isnan(parameters["exgaussian_tau"])


def test_the_fit_is_skipped_on_a_sample_without_variance():
    parameters = ex_gaussian_parameters([900.0] * (MIN_TRIALS_FOR_SHAPE + 5))

    assert not parameters["exgaussian_fitted"]
    assert np.isnan(parameters["exgaussian_mu"])


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


def test_distributions_are_reported_for_every_declared_stratum():
    distributions = decision_time_distributions(_distribution_session())

    assert set(distributions["stratum_type"]) == {
        "overall", "condition", "class", "condition_class",
    }
    assert set(distributions.loc[distributions["stratum_type"] == "class", "stratum"]) == (
        {"easy", "ambiguous", "misleading"}
    )
    # 1 overall + 2 conditions + 3 classes + 6 cells.
    assert len(distributions) == 12


def test_each_stratum_summarizes_only_its_own_trials():
    distributions = decision_time_distributions(
        _distribution_session(), fit_ex_gaussian=False
    ).set_index("stratum")

    assert distributions.loc["all_task_trials", "n_trials"] == 12
    assert distributions.loc["fast", "mean"] == pytest.approx(1083.3333333333333)
    assert distributions.loc["easy", "n_trials"] == 4
    assert distributions.loc["fast_easy", "mean"] == pytest.approx(850.0)


def test_an_empty_declared_cell_is_kept_with_no_statistics():
    features = _distribution_session()
    features = features.loc[
        ~((features["condition"] == "Slow") & (features["trial_class"] == 3))
    ]

    distributions = decision_time_distributions(features).set_index("stratum")

    assert distributions.loc["slow_misleading", "n_trials"] == 0
    assert np.isnan(distributions.loc["slow_misleading", "mean"])


def test_the_ex_gaussian_columns_are_absent_when_the_fit_is_not_requested():
    with_fit = decision_time_distributions(_distribution_session())
    without_fit = decision_time_distributions(
        _distribution_session(), fit_ex_gaussian=False
    )

    assert "exgaussian_tau" in with_fit.columns
    assert "exgaussian_tau" not in without_fit.columns


def test_distributions_are_computed_within_each_subject():
    features = pd.concat(
        [_distribution_session("H01"), _distribution_session("H02", shift=500.0)],
        ignore_index=True,
    )

    distributions = decision_time_distributions(
        features, fit_ex_gaussian=False
    ).set_index(["subject", "stratum"])

    assert distributions.loc[("H02", "fast"), "mean"] == pytest.approx(
        distributions.loc[("H01", "fast"), "mean"] + 500.0
    )


def test_distribution_statistics_run_the_declared_paired_contrasts():
    distributions = pd.concat(
        [
            decision_time_distributions(
                _distribution_session(f"H{index:02d}", shift=50.0 * index),
                fit_ex_gaussian=False,
            )
            for index in range(1, 5)
        ],
        ignore_index=True,
    )

    statistics = decision_time_distribution_statistics(
        distributions, metrics=("q50", "skewness")
    )

    assert set(statistics["contrast"]) == {
        "easy_vs_ambiguous",
        "easy_vs_misleading",
        "ambiguous_vs_misleading",
        "fast_vs_slow",
    }
    assert set(statistics["metric"]) == {"q50", "skewness"}
    median = statistics.loc[
        (statistics["metric"] == "q50") & (statistics["contrast"] == "fast_vs_slow")
    ].iloc[0]
    assert median["n_subjects"] == 4
    assert median["mean_difference"] < 0


def _subject_distributions(n_subjects=3, **kwargs):
    return pd.concat(
        [
            decision_time_distributions(
                _distribution_session(f"H{index:02d}", shift=50.0 * index),
                **kwargs,
            )
            for index in range(1, n_subjects + 1)
        ],
        ignore_index=True,
    )


def test_a_contrast_whose_stratum_is_missing_is_omitted_not_invented():
    distributions = _subject_distributions(fit_ex_gaussian=False)
    distributions = distributions.loc[distributions["stratum"] != "misleading"]

    statistics = decision_time_distribution_statistics(
        distributions, metrics=("q50",)
    )

    assert set(statistics["contrast"]) == {"easy_vs_ambiguous", "fast_vs_slow"}


def test_a_metric_the_distributions_do_not_carry_is_skipped():
    """``exgaussian_tau`` is absent whenever the caller declined the fit, so
    asking for it must not take the other metrics down with it."""
    distributions = _subject_distributions(fit_ex_gaussian=False)

    statistics = decision_time_distribution_statistics(
        distributions, metrics=("q50", "exgaussian_tau")
    )

    assert set(statistics["metric"]) == {"q50"}


def test_no_available_metric_yields_an_empty_table():
    distributions = _subject_distributions(fit_ex_gaussian=False)

    assert decision_time_distribution_statistics(
        distributions, metrics=("exgaussian_tau",)
    ).empty


def test_duplicate_subject_strata_are_refused_rather_than_averaged():
    distributions = pd.concat(
        [
            decision_time_distributions(_distribution_session(), fit_ex_gaussian=False)
        ] * 2,
        ignore_index=True,
    )

    with pytest.raises(ValueError):
        decision_time_distribution_statistics(distributions, metrics=("q50",))


# --- cumulative logged SPD ----------------------------------------------------


def _spd_session(subject="H01"):
    return trial_features(
        subject=[subject] * 6,
        run_trial_index=list(range(1, 7)),
        trial_class=[1, 1, 2, 2, 3, 3],
        trial_class_name=[
            "easy", "easy", "ambiguous", "ambiguous", "misleading", "misleading",
        ],
        logged_spd=[0.55, 0.95, 0.60, 0.75, 0.45, 0.85],
    )


def test_the_cumulative_curve_is_monotone_and_spans_zero_to_one():
    curves = spd_cumulative_distributions(_spd_session())
    easy = curves.loc[
        (curves["view"] == "all_logged") & (curves["trial_class_name"] == "easy")
    ].sort_values("threshold")

    assert easy["pooled_proportion"].is_monotonic_increasing
    assert easy["pooled_proportion"].iloc[0] == pytest.approx(0.0)
    assert easy["pooled_proportion"].iloc[-1] == pytest.approx(1.0)


def test_both_logged_spd_views_are_reported_for_every_class():
    curves = spd_cumulative_distributions(_spd_session())

    assert set(curves["view"]) == {"all_logged", "validated_15row"}
    assert set(curves["trial_class_name"]) == {"easy", "ambiguous", "misleading"}


def test_the_curve_is_evaluated_at_the_thresholds_supplied():
    curves = spd_cumulative_distributions(_spd_session(), thresholds=(0.6, 0.9))
    easy = curves.loc[
        (curves["view"] == "all_logged") & (curves["trial_class_name"] == "easy")
    ].set_index("threshold")

    assert easy.loc[0.6, "pooled_proportion"] == pytest.approx(0.5)
    assert easy.loc[0.9, "pooled_proportion"] == pytest.approx(0.5)


def test_the_pooled_and_per_subject_readings_are_reported_side_by_side():
    """Pooling weights trials equally; the subject mean weights subjects
    equally, and they differ when subjects contribute unevenly."""
    heavy = _spd_session("H01")
    light = trial_features(
        subject=["H02"],
        run_trial_index=[1],
        trial_class=[1],
        logged_spd=[0.95],
    )
    curves = spd_cumulative_distributions(
        pd.concat([heavy, light], ignore_index=True), thresholds=(0.6,)
    )
    easy = curves.loc[
        (curves["view"] == "all_logged") & (curves["trial_class_name"] == "easy")
    ].iloc[0]

    assert easy["n_trials"] == 3
    assert easy["n_subjects"] == 2
    assert easy["pooled_proportion"] == pytest.approx(1 / 3)
    assert easy["mean_subject_proportion"] == pytest.approx(0.25)
    assert np.isfinite(easy["sem_subject_proportion"])


def test_the_validated_view_reports_no_trials_when_no_log_is_complete():
    features = _spd_session()
    features["logged_spd_validated_15row"] = float("nan")

    curves = spd_cumulative_distributions(features)
    validated = curves.loc[curves["view"] == "validated_15row"]

    assert (validated["n_trials"] == 0).all()
    assert validated["pooled_proportion"].isna().all()
