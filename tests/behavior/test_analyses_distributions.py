"""Tests for meg_tokens.behavior.analyses.distributions."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from meg_tokens.behavior.analyses.distributions import (
    decision_time_geometry,
    decision_time_geometry_statistics,
    MIN_TRIALS_FOR_SHAPE,
    decision_time_distribution_statistics,
    decision_time_distributions,
    distribution_summary,
    spd_cumulative_distributions,
)

from tests.behavior.factories import trial_features


# --- descriptive summary ------------------------------------------------------


def test_the_summary_reports_bias_corrected_moments_and_the_requested_quantiles():
    values = list(range(1, 25))
    summary = distribution_summary(values, quantiles=(0.25, 0.5, 0.75))

    assert summary["n_trials"] == 24
    assert summary["mean"] == pytest.approx(12.5)
    assert summary["sd"] == pytest.approx(np.std(values, ddof=1))
    assert summary["min"] == 1.0 and summary["max"] == 24.0
    assert summary["q25"] == pytest.approx(np.quantile(values, 0.25))
    assert summary["q50"] == pytest.approx(12.5)
    assert summary["q75"] == pytest.approx(np.quantile(values, 0.75))
    assert summary["skewness"] == pytest.approx(float(stats.skew(values, bias=False)))
    assert summary["kurtosis"] == pytest.approx(float(stats.kurtosis(values, bias=False)))
    # Quantile keys are named by integer percentage.
    assert {"q10", "q90"} <= set(distribution_summary(list(range(100)), quantiles=(0.1, 0.9)))


def test_the_summary_withholds_what_the_sample_cannot_support():
    short = distribution_summary(list(range(MIN_TRIALS_FOR_SHAPE - 1)))
    enough = distribution_summary(list(range(MIN_TRIALS_FOR_SHAPE)))
    assert np.isnan(short["skewness"]) and np.isnan(short["kurtosis"])
    assert np.isfinite(enough["skewness"])

    finite = distribution_summary([1.0, float("nan"), 3.0, float("inf")])
    assert finite["n_trials"] == 2
    assert finite["mean"] == pytest.approx(2.0)

    empty = distribution_summary([])
    assert empty["n_trials"] == 0
    assert all(np.isnan(empty[field]) for field in ("mean", "sd", "min", "max", "q50"))


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


def _subject_distributions(n_subjects=3, **kwargs):
    return pd.concat(
        [
            decision_time_distributions(
                _distribution_session(f"H{index:02d}", shift=50.0 * index), **kwargs
            )
            for index in range(1, n_subjects + 1)
        ],
        ignore_index=True,
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


def test_every_declared_stratum_is_reported_from_its_own_trials_within_each_subject():
    distributions = decision_time_distributions(_distribution_session())
    assert set(distributions["stratum_type"]) == {
        "overall", "condition", "class", "condition_class",
    }
    assert set(distributions.loc[distributions["stratum_type"] == "class", "stratum"]) == {
        "easy", "ambiguous", "misleading",
    }
    # 1 overall + 2 conditions + 3 classes + 6 cells.
    assert len(distributions) == 12

    indexed = distributions.set_index("stratum")
    assert indexed.loc["all_task_trials", "n_trials"] == 12
    assert indexed.loc["fast", "mean"] == pytest.approx(1083.3333333333333)
    assert indexed.loc["easy", "n_trials"] == 4
    assert indexed.loc["fast_easy", "mean"] == pytest.approx(850.0)

    # A declared cell nobody ran is kept with no statistics.
    features = _distribution_session()
    sparse = decision_time_distributions(
        features.loc[~((features["condition"] == "Slow") & (features["trial_class"] == 3))]
    ).set_index("stratum")
    assert sparse.loc["slow_misleading", "n_trials"] == 0
    assert np.isnan(sparse.loc["slow_misleading", "mean"])

    # Each subject is summarized on their own trials.
    per_subject = decision_time_distributions(
        pd.concat(
            [_distribution_session("H01"), _distribution_session("H02", shift=500.0)],
            ignore_index=True,
        )
    ).set_index(["subject", "stratum"])
    assert per_subject.loc[("H02", "fast"), "mean"] == pytest.approx(
        per_subject.loc[("H01", "fast"), "mean"] + 500.0
    )


def test_the_distribution_statistics_run_the_declared_contrasts_and_skip_what_is_absent():
    distributions = _subject_distributions(n_subjects=4)
    statistics = decision_time_distribution_statistics(
        distributions, metrics=("q50", "skewness")
    )
    assert set(statistics["contrast"]) == {
        "easy_vs_ambiguous", "easy_vs_misleading", "ambiguous_vs_misleading", "fast_vs_slow",
    }
    assert set(statistics["metric"]) == {"q50", "skewness"}
    median = statistics.loc[
        (statistics["metric"] == "q50") & (statistics["contrast"] == "fast_vs_slow")
    ].iloc[0]
    assert median["n_subjects"] == 4
    assert median["mean_difference"] < 0

    # A missing stratum drops its contrasts rather than inventing them.
    partial = _subject_distributions()
    partial = partial.loc[partial["stratum"] != "misleading"]
    assert set(
        decision_time_distribution_statistics(partial, metrics=("q50",))["contrast"]
    ) == {"easy_vs_ambiguous", "fast_vs_slow"}

    # A caller may name a column the summary does not produce; that must not
    # take the other metrics down with it.
    mixed = decision_time_distribution_statistics(
        _subject_distributions(), metrics=("q50", "not_a_column")
    )
    assert set(mixed["metric"]) == {"q50"}
    assert decision_time_distribution_statistics(
        _subject_distributions(), metrics=("not_a_column",)
    ).empty

    duplicated = pd.concat([decision_time_distributions(_distribution_session())] * 2)
    with pytest.raises(ValueError):
        decision_time_distribution_statistics(duplicated, metrics=("q50",))


# --- cumulative logged SPD ----------------------------------------------------


def _spd_session(subject="H01"):
    return trial_features(
        subject=[subject] * 6,
        run_trial_index=list(range(1, 7)),
        trial_class=[1, 1, 2, 2, 3, 3],
        trial_class_name=["easy", "easy", "ambiguous", "ambiguous", "misleading", "misleading"],
        logged_spd=[0.55, 0.95, 0.60, 0.75, 0.45, 0.85],
    )


def test_the_cumulative_curve_is_monotone_spans_zero_to_one_and_covers_both_views():
    curves = spd_cumulative_distributions(_spd_session())
    assert set(curves["view"]) == {"all_logged", "validated_15row"}
    assert set(curves["trial_class_name"]) == {"easy", "ambiguous", "misleading"}

    easy = curves.loc[
        (curves["view"] == "all_logged") & (curves["trial_class_name"] == "easy")
    ].sort_values("threshold")
    assert easy["pooled_proportion"].is_monotonic_increasing
    assert easy["pooled_proportion"].iloc[0] == pytest.approx(0.0)
    assert easy["pooled_proportion"].iloc[-1] == pytest.approx(1.0)

    # Evaluated at the thresholds supplied, not at ones of its own choosing.
    chosen = spd_cumulative_distributions(_spd_session(), thresholds=(0.6, 0.9))
    chosen = chosen.loc[
        (chosen["view"] == "all_logged") & (chosen["trial_class_name"] == "easy")
    ].set_index("threshold")
    assert chosen.loc[0.6, "pooled_proportion"] == pytest.approx(0.5)
    assert chosen.loc[0.9, "pooled_proportion"] == pytest.approx(0.5)

    # With no complete log the validated view reports zero trials, not zeros.
    withheld = _spd_session()
    withheld["logged_spd_validated_15row"] = float("nan")
    validated = spd_cumulative_distributions(withheld)
    validated = validated.loc[validated["view"] == "validated_15row"]
    assert (validated["n_trials"] == 0).all()
    assert validated["pooled_proportion"].isna().all()


def test_the_pooled_and_per_subject_readings_are_reported_side_by_side():
    """Pooling weights trials equally; the subject mean weights subjects
    equally, and they differ when subjects contribute unevenly."""
    light = trial_features(
        subject=["H02"], run_trial_index=[1], trial_class=[1], logged_spd=[0.95]
    )
    curves = spd_cumulative_distributions(
        pd.concat([_spd_session("H01"), light], ignore_index=True), thresholds=(0.6,)
    )
    easy = curves.loc[
        (curves["view"] == "all_logged") & (curves["trial_class_name"] == "easy")
    ].iloc[0]

    assert easy["n_trials"] == 3
    assert easy["n_subjects"] == 2
    assert easy["pooled_proportion"] == pytest.approx(1 / 3)
    assert easy["mean_subject_proportion"] == pytest.approx(0.25)
    assert np.isfinite(easy["sem_subject_proportion"])


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


def test_a_contrast_without_enough_trials_is_reported_rather_than_measured():
    geometry = decision_time_geometry(_two_condition_session(), min_trials=500)
    row = geometry.loc[geometry["contrast"] == "fast_vs_slow"].iloc[0]
    assert row["n_a"] == 120 and row["n_b"] == 120
    assert "stretch_factor" not in row or pd.isna(row.get("stretch_factor"))
