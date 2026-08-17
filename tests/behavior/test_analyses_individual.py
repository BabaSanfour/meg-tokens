"""Tests for meg_tokens.behavior.analyses.individual."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from meg_tokens.behavior.analyses.individual import (
    comparison_statistics,
    individual_correlations,
    individual_profile,
)


SUBJECTS = ["H01", "H02", "H03", "H04"]


def _subject_summary(**overrides) -> pd.DataFrame:
    columns = {
        "subject": SUBJECTS,
        "mean_fast_dt_ms": [900.0, 1000.0, 1100.0, 1200.0],
        "mean_slow_dt_ms": [1100.0, 1300.0, 1400.0, 1700.0],
        "percent_correct": [80.0, 78.0, 75.0, 70.0],
        "motor_baseline_ms": [340.0, 355.0, 360.0, 370.0],
        "mean_easy_dt_ms": [800.0, 900.0, 1000.0, 1100.0],
        "mean_ambiguous_dt_ms": [1100.0, 1250.0, 1350.0, 1600.0],
        "mean_misleading_dt_ms": [1050.0, 1200.0, 1300.0, 1500.0],
        "mean_easy_spd_all_logged": [0.90, 0.88, 0.86, 0.84],
        "mean_ambiguous_spd_all_logged": [0.72, 0.70, 0.69, 0.66],
        "mean_misleading_spd_all_logged": [0.65, 0.63, 0.60, 0.58],
    }
    columns.update(overrides)
    return pd.DataFrame(columns)


def _condition_table(column, values, condition="all", **extra):
    table = {
        "subject": SUBJECTS,
        "condition": [condition] * len(SUBJECTS),
        column: values,
    }
    table.update({key: [value] * len(SUBJECTS) for key, value in extra.items()})
    return pd.DataFrame(table)


# --- assembling the profile ---------------------------------------------------


def test_the_profile_derives_the_speed_accuracy_adjustment_per_subject():
    profile = individual_profile(_subject_summary())

    assert profile["sat_adjustment_ms"].tolist() == pytest.approx(
        [200.0, 300.0, 300.0, 500.0]
    )


def test_the_overall_decision_time_weights_the_two_conditions_equally():
    """Not a trial-weighted mean: a subject with more Fast trials must not pull
    their overall decision time toward the Fast block."""
    profile = individual_profile(_subject_summary())

    assert profile["mean_dt_ms"].tolist() == pytest.approx(
        [1000.0, 1150.0, 1250.0, 1450.0]
    )


def test_a_subject_missing_one_condition_keeps_the_other_but_loses_the_adjustment():
    summary = _subject_summary()
    summary.loc[0, "mean_slow_dt_ms"] = float("nan")

    profile = individual_profile(summary)

    assert profile.loc[0, "mean_dt_ms"] == pytest.approx(900.0)
    assert np.isnan(profile.loc[0, "sat_adjustment_ms"])


def test_only_supported_summary_columns_are_carried_into_the_profile():
    summary = _subject_summary()
    summary["some_other_measure"] = 1.0

    profile = individual_profile(summary)

    assert "percent_correct" in profile.columns
    assert "motor_baseline_ms" in profile.columns
    assert "some_other_measure" not in profile.columns


def test_urgency_and_criterion_slopes_are_joined_under_distinct_names():
    """Both are criterion fits; only the predictor differs, so the profile has
    to keep them apart rather than overwrite one with the other."""
    profile = individual_profile(
        _subject_summary(),
        urgency=_condition_table(
            "slope", [-0.05, -0.04, -0.03, -0.01], response="logged_spd"
        ),
        criterion=_condition_table(
            "slope",
            [-0.02, -0.03, -0.01, -0.04],
            response="chosen_sum_log_lr_first_order_at_decision",
        ),
    )

    assert profile["urgency_slope_per_second"].tolist() == pytest.approx(
        [-0.05, -0.04, -0.03, -0.01]
    )
    assert profile["criterion_slope_sum_log_lr_per_second"].tolist() == pytest.approx(
        [-0.02, -0.03, -0.01, -0.04]
    )


def test_only_the_pooled_condition_row_is_joined():
    urgency = pd.concat(
        [
            _condition_table(
                "slope", [-0.05, -0.04, -0.03, -0.01],
                condition="all", response="logged_spd",
            ),
            _condition_table(
                "slope", [9.0, 9.0, 9.0, 9.0],
                condition="fast", response="logged_spd",
            ),
        ],
        ignore_index=True,
    )

    profile = individual_profile(_subject_summary(), urgency=urgency)

    assert profile["urgency_slope_per_second"].tolist() == pytest.approx(
        [-0.05, -0.04, -0.03, -0.01]
    )


def test_an_unknown_evidence_scale_yields_no_values_rather_than_a_substitute():
    """The measure is defined on ``logged_spd``; a fit on another scale is not
    quietly promoted into its place."""
    urgency = _condition_table(
        "slope", [-0.05, -0.04, -0.03, -0.01], response="some_other_scale"
    )

    profile = individual_profile(_subject_summary(), urgency=urgency)

    assert profile["urgency_slope_per_second"].isna().all()


def test_the_evidence_predictor_is_selected_by_name_not_by_position():
    evidence = pd.concat(
        [
            _condition_table(
                "dt_slope_ms_per_unit", [-100.0, -200.0, -300.0, -400.0],
                predictor="sp_design_early",
                accuracy_log_odds_per_unit=1.0,
            ),
            _condition_table(
                "dt_slope_ms_per_unit", [-9.0, -9.0, -9.0, -9.0],
                predictor="sum_log_lr_design_early",
                accuracy_log_odds_per_unit=2.0,
            ),
        ],
        ignore_index=True,
    )

    profile = individual_profile(
        _subject_summary(), evidence=evidence, evidence_predictor="sp_design_early"
    )

    assert profile["dt_slope_ms_per_unit"].tolist() == pytest.approx(
        [-100.0, -200.0, -300.0, -400.0]
    )
    assert profile["accuracy_log_odds_per_unit"].tolist() == pytest.approx([1.0] * 4)


def test_neural_metrics_are_joined_unchanged_for_later_analysis():
    neural = pd.DataFrame({"subject": SUBJECTS, "alpha_power": [1.0, 2.0, 3.0, 4.0]})

    profile = individual_profile(_subject_summary(), neural_metrics=neural)

    assert profile["alpha_power"].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_an_optional_input_missing_a_subject_never_removes_them():
    lapses = pd.DataFrame({
        "subject": SUBJECTS[:2],
        "condition": ["all"] * 2,
        "lapse_rate": [0.1, 0.2],
    })

    profile = individual_profile(_subject_summary(), lapses=lapses)

    assert profile["subject"].tolist() == SUBJECTS
    assert profile["lapse_rate"].isna().tolist() == [False, False, True, True]


def test_a_duplicated_subject_is_refused_rather_than_aggregated():
    summary = pd.concat([_subject_summary()] * 2, ignore_index=True)

    with pytest.raises(ValueError, match="one row per subject"):
        individual_profile(summary)


def test_a_measure_with_several_pooled_rows_per_subject_is_refused():
    urgency = pd.concat(
        [_condition_table("slope", [-0.05] * 4, response="logged_spd")] * 2,
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="multiple 'all' rows per subject"):
        individual_profile(_subject_summary(), urgency=urgency)


def test_a_summary_without_the_condition_means_is_refused():
    with pytest.raises(ValueError, match="mean_slow_dt_ms"):
        individual_profile(_subject_summary().drop(columns=["mean_slow_dt_ms"]))


# --- correlating the profile --------------------------------------------------


def _profile():
    return individual_profile(
        _subject_summary(),
        urgency=_condition_table(
            "slope", [-0.05, -0.04, -0.03, -0.01], response="logged_spd"
        ),
    )


def test_correlations_are_reported_for_every_estimable_measure_pair():
    correlations = individual_correlations(_profile())

    pairs = set(zip(correlations["measure_a"], correlations["measure_b"]))
    assert ("mean_dt_ms", "urgency_slope_per_second") in pairs
    assert ("mean_dt_ms", "percent_correct") in pairs
    # Unordered pairs only, and never a measure against itself.
    assert all(a != b for a, b in pairs)
    assert len(pairs) == len(correlations)


def test_both_correlation_coefficients_are_reported_with_their_sample_size():
    profile = _profile()
    correlations = individual_correlations(profile).set_index(
        ["measure_a", "measure_b"]
    )
    row = correlations.loc[("mean_dt_ms", "urgency_slope_per_second")]

    assert row["n_subjects"] == 4
    assert row["df"] == 2.0
    assert row["pearson_r"] == pytest.approx(
        float(
            stats.pearsonr(
                profile["mean_dt_ms"], profile["urgency_slope_per_second"]
            ).statistic
        )
    )
    assert row["spearman_rho"] == pytest.approx(
        float(
            stats.spearmanr(
                profile["mean_dt_ms"], profile["urgency_slope_per_second"]
            ).statistic
        )
    )


def test_a_subject_missing_one_member_is_excluded_from_that_pair_only():
    profile = _profile()
    profile.loc[0, "urgency_slope_per_second"] = float("nan")

    correlations = individual_correlations(profile).set_index(
        ["measure_a", "measure_b"]
    )

    assert correlations.loc[("mean_dt_ms", "urgency_slope_per_second"), "n_subjects"] == 3
    assert correlations.loc[("mean_dt_ms", "percent_correct"), "n_subjects"] == 4


def test_a_pair_without_enough_subjects_is_omitted():
    profile = _profile().iloc[:2]

    assert individual_correlations(profile).empty


def test_a_pair_without_variance_is_omitted_rather_than_reported_as_zero():
    profile = _profile()
    profile["urgency_slope_per_second"] = 1.0

    correlations = individual_correlations(profile)
    pairs = set(zip(correlations["measure_a"], correlations["measure_b"]))

    assert ("mean_dt_ms", "urgency_slope_per_second") not in pairs


def test_measures_absent_from_the_profile_are_skipped():
    correlations = individual_correlations(individual_profile(_subject_summary()))
    measures = set(correlations["measure_a"]) | set(correlations["measure_b"])

    # No urgency, criterion, evidence or lapse inputs were supplied.
    assert "urgency_slope_per_second" not in measures
    assert "mean_dt_ms" in measures


# --- cross-species comparison -------------------------------------------------


def test_every_class_measure_is_summarized():
    statistics = comparison_statistics(_subject_summary())

    assert set(statistics["measure"]) == {
        "decision_time_easy_ms",
        "decision_time_ambiguous_ms",
        "decision_time_misleading_ms",
        "success_probability_at_decision_easy",
        "success_probability_at_decision_ambiguous",
        "success_probability_at_decision_misleading",
    }
    assert (statistics["analysis"] == "cross_species_comparison").all()


def test_each_measure_carries_its_own_one_sample_summary():
    summary = _subject_summary()
    statistics = comparison_statistics(summary).set_index("measure")
    row = statistics.loc["decision_time_easy_ms"]

    assert row["n_subjects"] == 4
    assert row["mean"] == pytest.approx(summary["mean_easy_dt_ms"].mean())


def test_the_criterion_slope_uses_first_order_sum_log_lr():
    """The comparable monkey measure is first-order chosen-target SumLogLR,
    not exact posterior log odds."""
    criterion = pd.concat(
        [
            _condition_table(
                "slope", [-0.05, -0.04, -0.03, -0.01], response="logged_spd"
            ),
            _condition_table(
                "slope",
                [-0.5, -0.4, -0.3, -0.1],
                response="chosen_sum_log_lr_first_order_at_decision",
            ),
        ],
        ignore_index=True,
    )

    statistics = comparison_statistics(
        _subject_summary(), criterion=criterion
    ).set_index("measure")
    row = statistics.loc["criterion_slope_sum_log_lr_per_second"]

    assert row["n_subjects"] == 4
    assert row["mean"] == pytest.approx(np.mean([-0.5, -0.4, -0.3, -0.1]))


def test_the_criterion_row_is_absent_when_no_fit_is_supplied():
    statistics = comparison_statistics(_subject_summary())

    assert "criterion_slope_sum_log_lr_per_second" not in set(
        statistics["measure"]
    )


def _sequential_sampling_fits() -> pd.DataFrame:
    """Accumulator fits for four subjects in the pooled and both block conditions."""
    rows = []
    for index, subject in enumerate(["H01", "H02", "H03", "H04"]):
        for condition, rate in (("all", 0.8), ("fast", 1.0), ("slow", 0.6)):
            rows.append(
                {
                    "subject": subject,
                    "condition": condition,
                    "model": "urgency",
                    "delta_bic": -10.0 - index,
                    "urgency_scale": rate + 0.1 * index,
                }
            )
            rows.append(
                {
                    "subject": subject,
                    "condition": condition,
                    "model": "ddm",
                    "delta_bic": 0.0,
                    "urgency_scale": np.nan,
                }
            )
    return pd.DataFrame(rows)


def test_the_urgency_gating_rows_are_computed_from_the_fits():
    statistics = comparison_statistics(
        _subject_summary(), sequential_sampling=_sequential_sampling_fits()
    ).set_index("measure")

    assert statistics.loc["urgency_minus_integrator_bic", "mean"] == pytest.approx(
        np.mean([-10.0, -11.0, -12.0, -13.0])
    )
    assert statistics.loc["urgency_scale_criterion_seconds", "n_subjects"] == 4
    # Fast minus Slow, per subject, so the row stays a one-sample test.
    assert statistics.loc[
        "urgency_scale_fast_minus_slow", "mean"
    ] == pytest.approx(0.4)


def test_the_urgency_gating_rows_are_absent_when_no_fit_is_supplied():
    statistics = comparison_statistics(_subject_summary())

    assert "urgency_minus_integrator_bic" not in set(statistics["measure"])


def test_a_summary_without_a_class_measure_is_refused():
    summary = _subject_summary().drop(columns=["mean_misleading_spd_all_logged"])

    with pytest.raises(ValueError, match="mean_misleading_spd_all_logged"):
        comparison_statistics(summary)
