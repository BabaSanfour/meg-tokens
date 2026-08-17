"""Tests for meg_tokens.behavior.analyses.evidence."""

import numpy as np
import pandas as pd
import pytest

from meg_tokens.behavior.analyses.evidence import (
    conditional_accuracy_functions,
    conditional_accuracy_statistics,
    continuous_evidence_effects,
    continuous_evidence_statistics,
    criterion_decline,
    criterion_decline_statistics,
    evidence_at_decision_responses,
    first_order_chosen_sum_log_lr,
    first_order_criterion_decline,
    reverse_correlation,
    reverse_correlation_statistics,
)
from meg_tokens.behavior.math.evidence import FIRST_ORDER_TOKEN_LOG_LR

from tests.behavior.factories import TOKEN_DIRECTIONS, trial_features


def _pooled(table):
    return table.loc[table["condition"] == "all"]


# --- criterion decline --------------------------------------------------------


def test_criterion_decline_recovers_a_planted_slope():
    tokens = [2, 4, 6, 8, 10, 12]
    features = trial_features(
        decision_token_index=tokens,
        logged_spd=[0.5 + 0.02 * value for value in tokens],
    )

    overall = _pooled(criterion_decline(features)).iloc[0]

    assert overall["slope"] == pytest.approx(0.02)
    assert overall["intercept"] == pytest.approx(0.5)
    assert overall["n_trials"] == 6
    assert overall["converged"]


def test_urgency_is_fitted_in_evidence_units_per_second():
    """``dt_ms`` is converted to seconds, so a 1000 ms change is one unit."""
    features = trial_features(
        dt_ms=[1000.0, 2000.0, 3000.0, 4000.0],
        logged_spd=[0.9, 0.8, 0.7, 0.6],
    )

    overall = _pooled(criterion_decline(features, predictor="dt_ms")).iloc[0]

    assert overall["slope"] == pytest.approx(-0.1)
    assert overall["intercept"] == pytest.approx(1.0)


def test_criterion_decline_is_fitted_pooled_and_within_each_condition():
    features = trial_features(
        condition=["Fast"] * 4 + ["Slow"] * 4,
        run=[1] * 4 + [2] * 4,
        run_trial_index=[1, 2, 3, 4] * 2,
        decision_token_index=[2, 4, 6, 8] * 2,
        logged_spd=[0.9, 0.8, 0.7, 0.6, 0.9, 0.85, 0.8, 0.75],
    )

    fits = criterion_decline(features).set_index("condition")

    assert set(fits.index) == {"all", "fast", "slow"}
    assert fits.loc["fast", "slope"] == pytest.approx(-0.05)
    assert fits.loc["slow", "slope"] == pytest.approx(-0.025)


def test_an_unestimable_fit_is_reported_as_such_rather_than_guessed():
    features = trial_features(
        decision_token_index=[5, 5, 5],
        logged_spd=[0.8, 0.7, 0.9],
    )

    overall = _pooled(criterion_decline(features)).iloc[0]

    assert not overall["converged"]
    assert np.isnan(overall["slope"])


def test_an_empty_condition_cell_is_still_reported():
    features = trial_features(
        condition=["Fast"] * 4,
        decision_token_index=[2, 4, 6, 8],
        logged_spd=[0.9, 0.8, 0.7, 0.6],
    )

    fits = criterion_decline(features).set_index("condition")

    assert fits.loc["slow", "n_trials"] == 0
    assert not fits.loc["slow", "converged"]


def test_both_evidence_scales_are_fitted_against_the_same_predictor():
    features = trial_features(
        decision_token_index=[2, 4, 6, 8],
        logged_spd=[0.9, 0.8, 0.7, 0.6],
    )

    fits = evidence_at_decision_responses(features, predictor="decision_token_index")

    assert set(fits["response"]) == {"logged_spd", "logged_spd_log_odds"}
    assert set(fits["predictor"]) == {"decision_token_index"}
    # Both scales decline; the log-odds slope is not the probability slope.
    slopes = fits.loc[fits["condition"] == "all"].set_index("response")["slope"]
    assert slopes["logged_spd"] < 0 and slopes["logged_spd_log_odds"] < 0
    assert slopes["logged_spd"] != pytest.approx(slopes["logged_spd_log_odds"])


def test_first_order_criterion_uses_continuous_time_and_valid_alignments():
    features = trial_features(
        dt_ms=[-50, 400, 800, 1200, 1600],
        decision_token_index=[0, 2, 4, 6, 8],
        token_lead_at_decision=[0, 4, 3, 99, 1],
        design_time_alignment_valid=[True, True, True, False, True],
    )

    fits = first_order_criterion_decline(features)
    overall = _pooled(fits).iloc[0]

    assert overall["n_trials"] == 3
    assert overall["predictor"] == "dt_ms"
    assert overall["slope"] == pytest.approx(-2.5 * FIRST_ORDER_TOKEN_LOG_LR)


def test_first_order_sum_log_lr_reverses_errors_into_the_chosen_frame():
    features = trial_features(
        token_lead_at_decision=[2, 2],
        isCorrect=[True, False],
    )

    values = first_order_chosen_sum_log_lr(features)

    assert values.tolist() == pytest.approx([
        2 * FIRST_ORDER_TOKEN_LOG_LR,
        -2 * FIRST_ORDER_TOKEN_LOG_LR,
    ])


def _criterion_fits(n_subjects=4):
    return pd.concat(
        [
            criterion_decline(
                trial_features(
                    subject=[f"H{index:02d}"] * 8,
                    condition=["Fast"] * 4 + ["Slow"] * 4,
                    run=[1] * 4 + [2] * 4,
                    run_trial_index=[1, 2, 3, 4] * 2,
                    decision_token_index=[2, 4, 6, 8] * 2,
                    logged_spd=[
                        0.9 - 0.05 * step * (1 + 0.1 * index) for step in range(4)
                    ] + [
                        0.9 - 0.02 * step * (1 + 0.1 * index) for step in range(4)
                    ],
                )
            )
            for index in range(1, n_subjects + 1)
        ],
        ignore_index=True,
    )


def test_criterion_statistics_test_both_terms_and_contrast_the_conditions():
    statistics = criterion_decline_statistics(_criterion_fits())

    assert set(statistics["term"]) == {"intercept", "slope"}
    assert set(statistics["condition"]) == {"all", "fast", "slow", "fast_vs_slow"}
    slope = statistics.loc[
        (statistics["term"] == "slope") & (statistics["condition"] == "fast")
    ].iloc[0]
    assert slope["n_subjects"] == 4
    assert slope["mean"] < 0


def test_criterion_statistics_are_reported_per_predictor_and_response():
    features = trial_features(
        decision_token_index=[2, 4, 6, 8],
        logged_spd=[0.9, 0.8, 0.7, 0.6],
    )
    fits = pd.concat(
        [
            evidence_at_decision_responses(
                features.assign(subject=f"H{index:02d}"),
                predictor="decision_token_index",
            )
            for index in range(1, 4)
        ],
        ignore_index=True,
    )

    statistics = criterion_decline_statistics(fits)

    assert set(statistics["response"]) == {"logged_spd", "logged_spd_log_odds"}
    assert set(statistics["predictor"]) == {"decision_token_index"}


# --- reverse correlation ------------------------------------------------------


def test_tokens_that_fell_after_the_choice_carry_no_kernel_weight():
    """Every trial commits after two tokens, so jumps 3 and later were never
    visible and must not contribute."""
    features = trial_features(
        decision_token_index=[2] * 6,
        choice_side=[1, 1, 1, 2, 2, 2],
        token_directions=["112" + "2" * 12] * 6,
    )

    kernels = _pooled(reverse_correlation(features, n_jumps=4)).set_index("jump")

    assert kernels.loc[1, "n_trials_token_seen"] == 6
    assert kernels.loc[3, "n_trials_token_seen"] == 0
    assert np.isnan(kernels.loc[3, "mean_signed_direction"])


def test_the_model_free_kernel_measures_direction_relative_to_the_choice():
    features = trial_features(
        decision_token_index=[2] * 6,
        choice_side=[1, 1, 1, 2, 2, 2],
        token_directions=["112" + "2" * 12] * 6,
    )

    kernels = _pooled(reverse_correlation(features, n_jumps=4)).set_index("jump")

    # Both early tokens went to target 1: +1 for the subjects who chose it and
    # -1 for those who chose target 2, which averages to zero over an even split.
    assert kernels.loc[1, "mean_signed_direction"] == pytest.approx(0.0)


def test_a_token_that_always_predicts_the_choice_gets_a_kernel_of_one():
    features = trial_features(
        decision_token_index=[3] * 6,
        choice_side=[1, 1, 1, 2, 2, 2],
        token_directions=["1" * 15] * 3 + ["2" * 15] * 3,
    )

    kernels = _pooled(reverse_correlation(features, n_jumps=3)).set_index("jump")

    assert kernels.loc[1, "mean_signed_direction"] == pytest.approx(1.0)


def test_the_kernel_length_is_the_requested_resolution():
    features = trial_features(
        decision_token_index=[5] * 4,
        choice_side=[1, 1, 2, 2],
        token_directions=[TOKEN_DIRECTIONS] * 4,
    )

    kernels = _pooled(reverse_correlation(features, n_jumps=6))

    assert kernels["jump"].tolist() == [1, 2, 3, 4, 5, 6]


def test_a_logistic_kernel_that_cannot_be_fitted_stays_missing():
    """Every subject chose the same target, so choice has no variance."""
    features = trial_features(
        decision_token_index=[5] * 4,
        choice_side=[1, 1, 1, 1],
        token_directions=[TOKEN_DIRECTIONS] * 4,
    )

    kernels = _pooled(reverse_correlation(features, n_jumps=3))

    assert not kernels["converged"].any()
    assert kernels["logistic_weight"].isna().all()
    # The model-free kernel is still reported.
    assert kernels["mean_signed_direction"].notna().all()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"choice_side": [3, 1, 2, 1]}, "choice_side must be 1 or 2"),
        (
            {"decision_token_index": [5, None, 5, 5]},
            "decision_token_index must not be missing",
        ),
    ],
)
def test_reverse_correlation_refuses_to_guess_a_missing_trial_field(
    overrides, message
):
    columns = {"decision_token_index": [5] * 4, "choice_side": [1, 1, 2, 2]}
    columns.update(overrides)

    with pytest.raises(ValueError, match=message):
        reverse_correlation(trial_features(**columns), n_jumps=3)


def test_the_kernel_resolution_must_be_positive():
    with pytest.raises(ValueError, match="n_jumps must be at least 1"):
        reverse_correlation(trial_features(dt_ms=[900.0]), n_jumps=0)


def test_kernel_statistics_test_both_metrics_at_every_jump():
    kernels = pd.concat(
        [
            reverse_correlation(
                trial_features(
                    subject=[f"H{index:02d}"] * 6,
                    decision_token_index=[3] * 6,
                    choice_side=[1, 1, 1, 2, 2, 2],
                    token_directions=["1" * 15] * 3 + ["2" * 15] * 3,
                ),
                n_jumps=2,
            )
            for index in range(1, 5)
        ],
        ignore_index=True,
    )

    statistics = reverse_correlation_statistics(kernels)

    assert set(statistics["metric"]) == {"logistic_weight", "mean_signed_direction"}
    assert set(statistics["jump"]) == {1, 2}
    signed = statistics.loc[
        (statistics["metric"] == "mean_signed_direction")
        & (statistics["condition"] == "all")
        & (statistics["jump"] == 1)
    ].iloc[0]
    assert signed["n_subjects"] == 4
    assert signed["mean"] == pytest.approx(1.0)


# --- conditional accuracy -----------------------------------------------------


def test_decision_time_bins_are_formed_inside_each_subject():
    """A globally slow subject must not be pushed entirely into the slow bin."""
    features = trial_features(
        subject=["H01"] * 4 + ["H02"] * 4,
        dt_ms=[500.0, 700.0, 900.0, 1100.0, 5000.0, 5200.0, 5400.0, 5600.0],
        isCorrect=[True, True, False, False] * 2,
        run_trial_index=[1, 2, 3, 4] * 2,
    )

    functions = _pooled(conditional_accuracy_functions(features, n_bins=2))

    assert set(functions["dt_bin"]) == {1, 2}
    assert functions.loc[functions["dt_bin"] == 1, "accuracy"].tolist() == [1.0, 1.0]
    assert functions.loc[functions["dt_bin"] == 2, "accuracy"].tolist() == [0.0, 0.0]


def test_each_bin_reports_its_own_decision_time_and_trial_count():
    features = trial_features(
        dt_ms=[500.0, 700.0, 900.0, 1100.0],
        isCorrect=[True, True, True, False],
    )

    functions = _pooled(conditional_accuracy_functions(features, n_bins=2))

    assert functions["n_trials"].tolist() == [2, 2]
    assert functions.loc[functions["dt_bin"] == 1, "mean_dt_ms"].iloc[0] == (
        pytest.approx(600.0)
    )


def test_a_subset_that_cannot_be_binned_is_omitted_rather_than_collapsed():
    features = trial_features(dt_ms=[900.0] * 3, isCorrect=[True, True, False])

    assert conditional_accuracy_functions(features, n_bins=3).empty


def test_conditional_accuracy_rejects_a_noncanonical_correctness_value():
    features = trial_features(dt_ms=[500.0, 900.0], isCorrect=["true", "false"])

    with pytest.raises(ValueError, match="isCorrect must be boolean"):
        conditional_accuracy_functions(features, n_bins=2)


def test_the_bin_count_must_be_positive():
    with pytest.raises(ValueError, match="n_bins must be at least 1"):
        conditional_accuracy_functions(trial_features(dt_ms=[900.0]), n_bins=0)


def test_conditional_accuracy_statistics_report_bin_means_and_the_trend():
    functions = pd.concat(
        [
            conditional_accuracy_functions(
                trial_features(
                    subject=[f"H{index:02d}"] * 6,
                    dt_ms=[500.0, 700.0, 900.0, 1100.0, 1300.0, 1500.0],
                    isCorrect=[True, True, True, True, False, False],
                    run_trial_index=[1, 2, 3, 4, 5, 6],
                ),
                n_bins=3,
            )
            for index in range(1, 5)
        ],
        ignore_index=True,
    )

    statistics = conditional_accuracy_statistics(functions)

    assert set(statistics["test"]) == {"mean_accuracy", "accuracy_slope_across_bins"}
    slope = statistics.loc[
        (statistics["test"] == "accuracy_slope_across_bins")
        & (statistics["condition"] == "all")
    ].iloc[0]
    # Accuracy falls across bins, so the fitted trend is negative.
    assert slope["n_subjects"] == 4
    assert slope["mean"] < 0


# --- continuous evidence ------------------------------------------------------


def test_continuous_evidence_keeps_trials_the_class_rule_discards():
    features = trial_features(
        trial_class=[1, 2, 0, 0],
        trial_class_name=["easy", "ambiguous", "unclassified", "unclassified"],
        sp_design_early=[0.9, 0.6, 0.5, 0.8],
        dt_ms=[800.0, 1200.0, 1400.0, 900.0],
        isCorrect=[True, True, False, True],
    )

    overall = _pooled(
        continuous_evidence_effects(features, predictors={"sp_design_early": 0.5})
    ).iloc[0]

    assert overall["n_dt_trials"] == 4


def test_stronger_early_evidence_predicts_a_faster_decision():
    features = trial_features(
        sp_design_early=[0.5, 0.6, 0.7, 0.8, 0.9],
        dt_ms=[1400.0, 1300.0, 1200.0, 1100.0, 1000.0],
        isCorrect=[False, True, True, True, True],
    )

    overall = _pooled(
        continuous_evidence_effects(features, predictors={"sp_design_early": 0.5})
    ).iloc[0]

    # Distance from chance, so a 0.1 rise in SP removes 100 ms.
    assert overall["dt_slope_ms_per_unit"] == pytest.approx(-1000.0)


def test_accuracy_is_fitted_on_signed_evidence_toward_the_correct_target():
    # Accuracy rises with signed evidence without separating it perfectly,
    # which a logistic fit cannot estimate.
    features = trial_features(
        sp_design_early=[0.2, 0.3, 0.4, 0.6, 0.7, 0.8],
        dt_ms=[1400.0, 1300.0, 1250.0, 1150.0, 1100.0, 1000.0],
        isCorrect=[False, True, False, False, True, True],
    )

    overall = _pooled(
        continuous_evidence_effects(features, predictors={"sp_design_early": 0.5})
    ).iloc[0]

    assert overall["n_accuracy_trials"] == 6
    assert overall["accuracy_log_odds_per_unit"] > 0


def test_the_neutral_point_is_declared_not_inferred_from_the_data():
    """Evidence strength is distance from the declared neutral point. These
    trials speed up as evidence moves away from 0.5 in either direction, which
    only a centred predictor can see."""
    features = trial_features(
        sp_design_early=[0.1, 0.3, 0.5, 0.7, 0.9],
        dt_ms=[1000.0, 1200.0, 1400.0, 1200.0, 1000.0],
        isCorrect=[True, False, False, True, True],
    )

    centred = _pooled(
        continuous_evidence_effects(features, predictors={"sp_design_early": 0.5})
    ).iloc[0]
    uncentred = _pooled(
        continuous_evidence_effects(features, predictors={"sp_design_early": 0.0})
    ).iloc[0]

    assert centred["dt_slope_ms_per_unit"] == pytest.approx(-1000.0)
    assert uncentred["dt_slope_ms_per_unit"] == pytest.approx(0.0, abs=1e-9)


def test_each_declared_predictor_is_fitted_separately():
    features = trial_features(
        sp_design_early=[0.5, 0.6, 0.7, 0.8, 0.9],
        sum_log_lr_design_early=[0.0, 0.4, 0.8, 1.2, 1.6],
        dt_ms=[1400.0, 1300.0, 1200.0, 1100.0, 1000.0],
        isCorrect=[False, True, True, True, True],
    )

    effects = _pooled(continuous_evidence_effects(features))

    assert set(effects["predictor"]) == {
        "sp_design_early", "sum_log_lr_design_early",
    }


def test_a_model_that_cannot_be_fitted_stays_missing_and_is_marked():
    features = trial_features(
        sp_design_early=[0.8, 0.8, 0.8],
        dt_ms=[900.0, 1000.0, 1100.0],
        isCorrect=[True, True, True],
    )

    overall = _pooled(
        continuous_evidence_effects(features, predictors={"sp_design_early": 0.5})
    ).iloc[0]

    assert not overall["converged"]
    assert np.isnan(overall["dt_slope_ms_per_unit"])
    assert np.isnan(overall["accuracy_log_odds_per_unit"])


def test_continuous_evidence_statistics_test_both_coefficients():
    fits = pd.concat(
        [
            continuous_evidence_effects(
                trial_features(
                    subject=[f"H{index:02d}"] * 10,
                    condition=["Fast"] * 5 + ["Slow"] * 5,
                    run=[1] * 5 + [2] * 5,
                    run_trial_index=[1, 2, 3, 4, 5] * 2,
                    sp_design_early=[0.5, 0.6, 0.7, 0.8, 0.9] * 2,
                    dt_ms=[
                        1400.0 - 100 * step * index for step in range(5)
                    ] + [
                        1600.0 - 80 * step * index for step in range(5)
                    ],
                    isCorrect=[False, True, True, True, True] * 2,
                ),
                predictors={"sp_design_early": 0.5},
            )
            for index in range(1, 5)
        ],
        ignore_index=True,
    )

    statistics = continuous_evidence_statistics(fits)

    assert set(statistics["term"]) == {
        "dt_slope_ms_per_unit", "accuracy_log_odds_per_unit",
    }
    assert "fast_vs_slow" in set(statistics["condition"])
    dt_trend = statistics.loc[
        (statistics["term"] == "dt_slope_ms_per_unit")
        & (statistics["condition"] == "all")
    ].iloc[0]
    assert dt_trend["n_subjects"] == 4
    assert dt_trend["mean"] < 0
