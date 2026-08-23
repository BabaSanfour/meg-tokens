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
    discretization_overshoot_fits,
    discretization_overshoot_statistics,
    early_evidence_order_profile,
    early_evidence_order_statistics,
    early_evidence_profile,
    evidence_at_decision_responses,
    first_order_chosen_sum_log_lr,
    first_order_criterion_decline,
    posterior_evidence_criterion_agreement,
    posterior_evidence_sensitivity,
    posterior_evidence_sensitivity_statistics,
    posterior_grid_by_token_index,
    posterior_grid_step_summary,
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


# --- posterior discretization audit -------------------------------------------


def test_posterior_grid_matches_the_known_first_order_token_likelihood():
    """The single-token leading state reproduces Cisek et al. (2009)'s
    ``P[Binomial(14, .5) >= 7]`` constant, which is where
    ``FIRST_ORDER_TOKEN_LOG_LR`` itself comes from."""
    grid = posterior_grid_by_token_index()

    first_token = grid.loc[
        (grid["token_index"] == 1) & (grid["target_tokens"] == 1)
    ].iloc[0]

    assert first_token["success_probability"] == pytest.approx(0.604736328125)


def test_posterior_grid_step_summary_is_coarsest_at_the_final_token():
    grid = posterior_grid_by_token_index()

    summary = posterior_grid_step_summary(grid).set_index("token_index")

    assert summary.loc[15, "n_reachable_states"] == 2
    assert summary.loc[15, "max_step"] == pytest.approx(1.0)
    assert summary.loc[1, "max_step"] < summary.loc[15, "max_step"]


def test_discretization_overshoot_fits_do_not_converge_from_one_shared_design():
    """A fixed threshold applied to identical designs yields one constant
    synthetic (token index, SPD) pair per subject/condition -- no variance
    for the slope to be estimated from, so the fit must be reported as
    unconverged rather than a spurious zero or missing slope."""
    features = trial_features(
        subject=["H01"] * 3 + ["H02"] * 3,
        condition=["Fast"] * 6,
        run=[1, 1, 1, 2, 2, 2],
        run_trial_index=[1, 2, 3, 1, 2, 3],
        token_directions=[TOKEN_DIRECTIONS] * 6,
    )

    fits = discretization_overshoot_fits(features, thresholds=(0.75,))

    assert set(fits["threshold"]) == {0.75}
    assert not fits["converged"].any()


def test_discretization_overshoot_fits_use_each_trials_own_design():
    """Varied designs produce varied synthetic decision indices, confirming
    the replay reads each trial's own token sequence rather than one shared
    value."""
    varied_directions = [
        TOKEN_DIRECTIONS,
        "111111111111111",
        "222222222222222",
        "121212121212121",
    ]
    features = trial_features(
        subject=["H01"] * 4,
        condition=["Fast"] * 4,
        run_trial_index=[1, 2, 3, 4],
        token_directions=varied_directions,
    )

    fits = discretization_overshoot_fits(features, thresholds=(0.7, 0.85))

    assert set(fits["threshold"]) == {0.7, 0.85}
    assert set(fits["predictor"]) == {"decision_token_index"}
    assert set(fits["response"]) == {"logged_spd"}
    assert (fits["n_trials"] > 0).any()


def test_discretization_overshoot_statistics_keeps_thresholds_separate():
    features = trial_features(
        subject=["H01", "H02", "H03", "H04"] * 4,
        condition=["Fast"] * 16,
        run_trial_index=list(range(1, 17)),
        token_directions=[
            TOKEN_DIRECTIONS, "111111111111111", "222222222222222", "121212121212121",
        ] * 4,
    )

    fits = discretization_overshoot_fits(features, thresholds=(0.7, 0.85))
    statistics = discretization_overshoot_statistics(fits)

    assert set(statistics["threshold"]) == {0.7, 0.85}
    assert set(statistics["term"]) == {"intercept", "slope"}


def test_posterior_evidence_sensitivity_filters_each_variant_correctly():
    features = trial_features(
        decision_token_index=[0, 2, 4, 6],
        dt_ms=[100.0, 2000.0, 3500.0, 4000.0],
        design_time_alignment_valid=[True, True, False, True],
        logged_spd=[0.5, 0.7, 0.8, 0.9],
    )

    fits = posterior_evidence_sensitivity(features)
    pooled = fits.loc[fits["condition"] == "all"].set_index("sensitivity")

    assert set(pooled.index) == {
        "baseline", "token0_excluded", "complete_log_only", "three_second_horizon",
    }
    assert pooled.loc["baseline", "n_trials"] == 4
    assert pooled.loc["token0_excluded", "n_trials"] == 3
    assert pooled.loc["complete_log_only", "n_trials"] == 3
    assert pooled.loc["three_second_horizon", "n_trials"] == 2


def test_posterior_evidence_sensitivity_statistics_keeps_variants_separate():
    fits = posterior_evidence_sensitivity(
        trial_features(
            subject=["H01", "H02", "H03"] * 4,
            decision_token_index=[2, 4, 6] * 4,
            dt_ms=[500.0, 1500.0, 2500.0] * 4,
            logged_spd=[0.9, 0.8, 0.7] * 4,
        )
    )

    statistics = posterior_evidence_sensitivity_statistics(fits)

    assert set(statistics["sensitivity"]) == {
        "baseline", "token0_excluded", "complete_log_only", "three_second_horizon",
    }


def _fits_table(subjects, slopes, intercepts, *, response="logged_spd"):
    return pd.DataFrame(
        {
            "subject": subjects,
            "condition": ["all"] * len(subjects),
            "response": [response] * len(subjects),
            "converged": [True] * len(subjects),
            "intercept": intercepts,
            "slope": slopes,
        }
    )


def test_posterior_evidence_criterion_agreement_recovers_a_perfect_correlation():
    subjects = [f"H{index:02d}" for index in range(1, 6)]
    criterion_fits = _fits_table(
        subjects, slopes=[1, 2, 3, 4, 5], intercepts=[5, 4, 3, 2, 1]
    ).drop(columns="response")
    urgency_fits = _fits_table(
        subjects, slopes=[2, 4, 6, 8, 10], intercepts=[10, 8, 6, 4, 2]
    )

    agreement = posterior_evidence_criterion_agreement(urgency_fits, criterion_fits)

    slope_row = agreement.loc[agreement["term"] == "slope"].iloc[0]
    assert slope_row["n_subjects"] == 5
    assert slope_row["pearson_r"] == pytest.approx(1.0)
    assert slope_row["spearman_rho"] == pytest.approx(1.0)


def test_posterior_evidence_criterion_agreement_omits_pairs_with_too_few_subjects():
    subjects = ["H01", "H02"]
    criterion_fits = _fits_table(subjects, slopes=[1, 2], intercepts=[1, 2]).drop(
        columns="response"
    )
    urgency_fits = _fits_table(subjects, slopes=[1, 2], intercepts=[1, 2])

    agreement = posterior_evidence_criterion_agreement(urgency_fits, criterion_fits)

    assert agreement.empty


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
    # Per-jump rows carry a jump; the across-jump summary deliberately does not.
    per_jump = statistics.loc[statistics["test"] != "all_jumps_vs_zero"]
    assert set(per_jump["jump"]) == {1, 2}
    assert statistics.loc[statistics["test"] == "all_jumps_vs_zero", "jump"].isna().all()
    signed = statistics.loc[
        (statistics["metric"] == "mean_signed_direction")
        & (statistics["condition"] == "all")
        & (statistics["jump"] == 1)
    ].iloc[0]
    assert signed["n_subjects"] == 4
    assert signed["mean"] == pytest.approx(1.0)


def test_the_kernel_registers_the_difference_sem_not_just_each_condition_s():
    """An error bar on a paired difference needs the SEM *of the difference*.
    Recovering it as `mean / t` in figure code fails at exactly zero, which is
    the case a kernel figure most needs to draw."""
    kernels = pd.concat(
        [
            reverse_correlation(
                trial_features(
                    subject=[f"H{index:02d}"] * 12,
                    condition=["Fast"] * 6 + ["Slow"] * 6,
                    run=[1] * 6 + [2] * 6,
                    run_trial_index=[1, 2, 3, 4, 5, 6] * 2,
                    decision_token_index=[3] * 12,
                    choice_side=[1, 1, 1, 2, 2, 2] * 2,
                    token_directions=(["1" * 15] * 3 + ["2" * 15] * 3) * 2,
                ),
                n_jumps=2,
            )
            for index in range(1, 5)
        ],
        ignore_index=True,
    )

    statistics = reverse_correlation_statistics(kernels)
    contrast = statistics.loc[statistics["condition"] == "fast_vs_slow"]

    assert "sem_difference" in contrast
    assert "equivalence_bound" in contrast
    # The interval is registered too, so the figure never reaches for a
    # t quantile of its own.
    assert {"ci_low", "ci_high"}.issubset(contrast.columns)


def test_every_jump_row_carries_the_corrected_threshold():
    """Eight independent per-jump tests need a stated threshold; the figure
    marks significance against it rather than against 0.05."""
    kernels = pd.concat(
        [
            reverse_correlation(
                trial_features(
                    subject=[f"H{index:02d}"] * 6,
                    decision_token_index=[3] * 6,
                    choice_side=[1, 1, 1, 2, 2, 2],
                    token_directions=["1" * 15] * 3 + ["2" * 15] * 3,
                ),
                n_jumps=4,
            )
            for index in range(1, 5)
        ],
        ignore_index=True,
    )

    statistics = reverse_correlation_statistics(kernels)

    assert (statistics["n_jumps_tested"] == 4).all()
    assert statistics["p_threshold"].eq(0.05 / 4).all()
    summary = statistics.loc[
        (statistics["test"] == "all_jumps_vs_zero")
        & (statistics["metric"] == "logistic_weight")
        & (statistics["condition"] == "all")
    ].iloc[0]
    # The count the report prints, so it never reduces a p-value column itself.
    assert summary["n_jumps_tested"] == 4
    assert 0 <= summary["n_jumps_significant"] <= 4


# --- conditional accuracy -----------------------------------------------------


def _stratum(table, stratum="all"):
    return _pooled(table).loc[table["trial_class_name"] == stratum]


def test_decision_time_bins_are_formed_inside_each_subject():
    """A globally slow subject must not be pushed entirely into the slow bin."""
    features = trial_features(
        subject=["H01"] * 4 + ["H02"] * 4,
        dt_ms=[500.0, 700.0, 900.0, 1100.0, 5000.0, 5200.0, 5400.0, 5600.0],
        isCorrect=[True, True, False, False] * 2,
        run_trial_index=[1, 2, 3, 4] * 2,
    )

    functions = _stratum(conditional_accuracy_functions(features, n_bins=2))

    assert set(functions["dt_bin"]) == {1, 2}
    assert functions.loc[functions["dt_bin"] == 1, "accuracy"].tolist() == [1.0, 1.0]
    assert functions.loc[functions["dt_bin"] == 2, "accuracy"].tolist() == [0.0, 0.0]


def test_each_bin_reports_its_own_decision_time_and_trial_count():
    features = trial_features(
        dt_ms=[500.0, 700.0, 900.0, 1100.0],
        isCorrect=[True, True, True, False],
    )

    functions = _stratum(conditional_accuracy_functions(features, n_bins=2))

    assert functions["n_trials"].tolist() == [2, 2]
    assert functions.loc[functions["dt_bin"] == 1, "mean_dt_ms"].iloc[0] == (
        pytest.approx(600.0)
    )


def test_bins_are_cut_pooled_so_the_class_rows_partition_the_bin():
    """Class rows must be read at the same times, not rebinned inside a class."""
    features = trial_features(
        dt_ms=[500.0, 700.0, 900.0, 1100.0],
        isCorrect=[True, True, False, False],
        trial_class=[1, 2, 1, 2],
        trial_class_name=["easy", "ambiguous", "easy", "ambiguous"],
    )

    functions = _pooled(conditional_accuracy_functions(features, n_bins=2))

    first_bin = functions.loc[functions["dt_bin"] == 1].set_index("trial_class_name")
    assert first_bin.loc["all", "n_trials"] == 2
    assert first_bin.loc["easy", "n_trials"] == 1
    assert first_bin.loc["ambiguous", "n_trials"] == 1
    assert first_bin.loc[["easy", "ambiguous"], "proportion_of_bin"].sum() == 1.0
    # Every class is read at the pooled bin's times, so a class holding only
    # the slower trial of the bin reports that trial's own decision time.
    assert first_bin.loc["ambiguous", "mean_dt_ms"] == pytest.approx(700.0)


def test_unclassified_trials_are_a_stratum_rather_than_a_discard():
    features = trial_features(
        dt_ms=[500.0, 700.0, 900.0, 1100.0],
        isCorrect=[True, True, True, False],
        trial_class=[1, 0, 1, 0],
        trial_class_name=["easy", "unclassified", "easy", "unclassified"],
    )

    functions = _pooled(conditional_accuracy_functions(features, n_bins=2))

    assert "unclassified" in set(functions["trial_class_name"])
    assert functions.loc[
        functions["trial_class_name"] == "unclassified", "n_trials"
    ].sum() == 2


def test_class_adjusted_accuracy_removes_the_composition_of_the_bins():
    """Both classes are flat in time; only their bin shares differ."""
    features = trial_features(
        dt_ms=[500.0, 600.0, 700.0, 1000.0, 1100.0, 1200.0],
        # easy always correct, hard always wrong, at every decision time.
        isCorrect=[True, True, False, True, False, False],
        trial_class=[1, 1, 2, 1, 2, 2],
        trial_class_name=["easy", "easy", "ambiguous", "easy", "ambiguous",
                          "ambiguous"],
        run_trial_index=[1, 2, 3, 4, 5, 6],
    )

    functions = _stratum(conditional_accuracy_functions(features, n_bins=2))

    observed = functions.sort_values("dt_bin")["accuracy"].tolist()
    adjusted = functions.sort_values("dt_bin")["accuracy_class_adjusted"].tolist()
    # Observed falls purely because the slow bin holds more ambiguous trials.
    assert observed == [pytest.approx(2 / 3), pytest.approx(1 / 3)]
    # Removing each class's own mean leaves the subject's grand mean in both.
    assert adjusted == [pytest.approx(0.5), pytest.approx(0.5)]


def test_class_adjusted_accuracy_is_only_defined_on_the_pooled_row():
    features = trial_features(
        dt_ms=[500.0, 700.0, 900.0, 1100.0],
        isCorrect=[True, True, False, False],
        trial_class=[1, 2, 1, 2],
        trial_class_name=["easy", "ambiguous", "easy", "ambiguous"],
    )

    functions = conditional_accuracy_functions(features, n_bins=2)

    classes = functions.loc[functions["trial_class_name"] != "all"]
    assert classes["accuracy_class_adjusted"].isna().all()


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


def _falling_accuracy_functions():
    return pd.concat(
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


def test_conditional_accuracy_statistics_report_bin_means_and_the_trend():
    statistics = conditional_accuracy_statistics(_falling_accuracy_functions())

    assert set(statistics["term"]) >= {
        "mean_accuracy", "accuracy_slope_per_bin", "accuracy_slope_per_second",
    }
    slope = statistics.loc[
        (statistics["term"] == "accuracy_slope_per_bin")
        & (statistics["measure"] == "accuracy")
        & (statistics["condition"] == "all")
        & (statistics["trial_class_name"] == "all")
        & (statistics["min_cell_trials"] == 1)
    ].iloc[0]
    # Accuracy falls across bins, so the fitted trend is negative.
    assert slope["n_subjects"] == 4
    assert slope["estimate"] < 0


def test_accuracy_is_tested_against_chance_rather_than_zero():
    """A 2AFC accuracy of 0.5 carries no information; against zero it is huge."""
    # Subject bin accuracies of 1.0, 0.0, 0.5 and 0.5: mean exactly chance,
    # with real between-subject variance so the t is defined.
    outcomes = [
        [True, True, True, True],
        [False, False, False, False],
        [True, False, True, False],
        [False, True, False, True],
    ]
    functions = pd.concat(
        [
            conditional_accuracy_functions(
                trial_features(
                    subject=[f"H{index:02d}"] * 4,
                    dt_ms=[500.0, 700.0, 900.0, 1100.0],
                    isCorrect=outcome,
                    run_trial_index=[1, 2, 3, 4],
                ),
                n_bins=2,
            )
            for index, outcome in enumerate(outcomes, start=1)
        ],
        ignore_index=True,
    )

    rows = conditional_accuracy_statistics(functions)
    rows = rows.loc[rows["term"] == "mean_accuracy"]

    assert set(rows["test"]) == {"one_sample_vs_chance"}
    # The estimate stays on the accuracy scale; the test is of the difference
    # from chance, which is exactly zero here.
    assert rows["estimate"].tolist() == [pytest.approx(0.5)] * len(rows)
    assert rows["t"].abs().max() == pytest.approx(0.0)


def test_the_slope_is_registered_in_both_bin_and_second_units():
    statistics = conditional_accuracy_statistics(_falling_accuracy_functions())
    selection = (
        (statistics["measure"] == "accuracy")
        & (statistics["condition"] == "all")
        & (statistics["trial_class_name"] == "all")
        & (statistics["min_cell_trials"] == 1)
    )
    per_bin = statistics.loc[
        selection & (statistics["term"] == "accuracy_slope_per_bin")
    ].iloc[0]
    per_second = statistics.loc[
        selection & (statistics["term"] == "accuracy_slope_per_second")
    ].iloc[0]

    # Bins are 400 ms apart here, so a per-second slope is 2.5x the per-bin one.
    assert per_second["estimate"] == pytest.approx(per_bin["estimate"] * 2.5)


def test_statistics_survive_a_table_with_nothing_to_decompose():
    """An all-empty intermediate has no columns, so grouping it raises rather
    than yielding nothing -- the edge case a single realized bin reaches."""
    functions = conditional_accuracy_functions(
        trial_features(
            subject=["H01"] * 4 + ["H02"] * 4,
            run_trial_index=[1, 2, 3, 4] * 2,
            dt_ms=[900.0, 1000.0, 1100.0, 1200.0] * 2,
            isCorrect=[True, True, False, False] * 2,
        ),
        n_bins=1,
    )

    statistics = conditional_accuracy_statistics(functions)

    assert "accuracy_change_decomposition" not in set(statistics["term"])
    assert "mean_accuracy" in set(statistics["term"])


def test_the_change_decomposition_reproduces_the_observed_change_exactly():
    """Composition plus within-class is an identity, not an approximation --
    if it were approximate the residual could absorb a mistake."""
    statistics = conditional_accuracy_statistics(_falling_accuracy_functions())
    rows = statistics.loc[
        (statistics["term"] == "accuracy_change_decomposition")
        & (statistics["condition"] == "all")
        & (statistics["trial_class_name"] == "all")
    ].set_index("measure")

    total = rows.loc["composition", "estimate"] + rows.loc["within_class", "estimate"]
    assert total == pytest.approx(rows.loc["composition", "observed_change"])
    assert rows.loc["composition", "share_of_observed_change"] + rows.loc[
        "within_class", "share_of_observed_change"
    ] == pytest.approx(1.0)


def test_the_decomposition_reports_what_dropping_unclassified_would_leave():
    """The `designed_classes` row is why the figure draws four strata; it has
    to be a registered number rather than an argument made in prose."""
    features = pd.concat(
        [
            trial_features(
                subject=[f"H{index:02d}"] * 8,
                run_trial_index=list(range(1, 9)),
                # Easy trials crowd the fast bin and unclassified the slow one,
                # and only unclassified falls with decision time.
                dt_ms=[500.0, 600.0, 700.0, 800.0, 1200.0, 1300.0, 1400.0, 1500.0],
                trial_class=[1, 1, 1, 0, 1, 0, 0, 0],
                trial_class_name=["easy"] * 3 + ["unclassified"]
                + ["easy"] + ["unclassified"] * 3,
                isCorrect=[True] * 5 + [False, True, False],
            )
            for index in range(1, 5)
        ],
        ignore_index=True,
    )
    statistics = conditional_accuracy_statistics(
        conditional_accuracy_functions(features, n_bins=2)
    )
    within = statistics.loc[
        (statistics["term"] == "accuracy_change_decomposition")
        & (statistics["measure"] == "within_class")
        & (statistics["condition"] == "all")
    ].set_index("trial_class_name")

    assert "designed_classes" in within.index
    # Dropping unclassified drops the stratum that carries the decline, so what
    # remains runs the other way.
    assert within.loc["unclassified", "estimate"] < 0
    assert within.loc["designed_classes", "estimate"] >= 0
    assert within.loc["designed_classes", "estimate"] != pytest.approx(
        within.loc["all", "estimate"]
    )


def test_a_subject_missing_a_class_from_an_end_bin_is_not_decomposed():
    """An absent class has a proportion of zero but no accuracy, so neither
    term is defined -- the subject drops out rather than contributing a zero."""
    complete = [
        trial_features(
            subject=[f"H{index:02d}"] * 8,
            run_trial_index=list(range(1, 9)),
            dt_ms=[500.0, 600.0, 700.0, 800.0, 1200.0, 1300.0, 1400.0, 1500.0],
            trial_class=[1, 2, 1, 2, 1, 2, 1, 2],
            trial_class_name=["easy", "ambiguous"] * 4,
            isCorrect=[True, True, True, False, True, False, False, False],
        )
        for index in range(1, 5)
    ]
    # One extra subject who never contributes an ambiguous trial.
    partial = trial_features(
        subject=["H99"] * 8,
        run_trial_index=list(range(1, 9)),
        dt_ms=[500.0, 600.0, 700.0, 800.0, 1200.0, 1300.0, 1400.0, 1500.0],
        trial_class=[1] * 8,
        trial_class_name=["easy"] * 8,
        isCorrect=[True] * 5 + [False] * 3,
    )
    statistics = conditional_accuracy_statistics(
        conditional_accuracy_functions(
            pd.concat([*complete, partial], ignore_index=True), n_bins=2
        )
    )
    row = statistics.loc[
        (statistics["term"] == "accuracy_change_decomposition")
        & (statistics["measure"] == "composition")
        & (statistics["condition"] == "all")
        & (statistics["trial_class_name"] == "all")
    ].iloc[0]

    assert row["n_subjects"] == 4


def _composition_driven_functions():
    """Eight subjects whose pooled accuracy falls mostly because the fast bins
    hold easy trials and the slow bins ambiguous ones.

    The single-class fixture above cannot exercise the agreement rows at all --
    with nothing to mix, the composition share is zero and every resample of it
    is identical. This one has two classes, a real compositional gradient, and
    enough between-subject variation for the bootstrap interval to be non-
    degenerate.
    """
    # Three bins of four trials. Easy trials thin out across them and ambiguous
    # ones take their place, so the pooled accuracy falls even though neither
    # class need change; both classes appear in every bin, without which a
    # subject cannot be decomposed at all.
    names = (
        ["easy"] * 3 + ["ambiguous"]
        + ["easy"] * 2 + ["ambiguous"] * 2
        + ["easy"] + ["ambiguous"] * 3
    )
    subjects = []
    for index in range(8):
        # Ambiguous trials are the ones that go wrong, and which of them do
        # varies by subject -- that variation is what the bootstrap resamples.
        correct = [name == "easy" for name in names]
        correct[3 + index % 4] = index % 2 == 0
        correct[7 + index % 4] = False
        subjects.append(
            trial_features(
                subject=[f"H{index + 1:02d}"] * len(names),
                run_trial_index=list(range(1, len(names) + 1)),
                dt_ms=[
                    400.0, 450.0, 500.0, 550.0,
                    900.0, 950.0, 1000.0, 1050.0,
                    1400.0, 1450.0, 1500.0, 1550.0,
                ],
                trial_class=[1 if name == "easy" else 2 for name in names],
                trial_class_name=names,
                isCorrect=correct,
            )
        )
    return conditional_accuracy_functions(
        pd.concat(subjects, ignore_index=True), n_bins=3
    )


def _agreement(statistics, measure, slope_unit="accuracy_slope_per_bin"):
    return statistics.loc[
        (statistics["term"] == "composition_share_agreement")
        & (statistics["measure"] == measure)
        & (statistics["condition"] == "all")
        & (statistics["slope_unit"] == slope_unit)
    ].iloc[0]


def test_each_composition_route_reproduces_the_number_it_is_built_from():
    """The agreement rows have to be the *same* two quantities the write-up
    already quotes elsewhere in this table, or the test compares something
    nobody claimed."""
    statistics = conditional_accuracy_statistics(_composition_driven_functions())

    decomposition = statistics.loc[
        (statistics["term"] == "accuracy_change_decomposition")
        & (statistics["measure"] == "composition")
        & (statistics["condition"] == "all")
        & (statistics["trial_class_name"] == "all")
    ].iloc[0]
    assert _agreement(statistics, "endpoint_share")["estimate"] == pytest.approx(
        decomposition["share_of_observed_change"]
    )

    slopes = statistics.loc[
        (statistics["term"] == "accuracy_slope_per_bin")
        & (statistics["condition"] == "all")
        & (statistics["trial_class_name"] == "all")
        & (statistics["min_cell_trials"] == 1)
    ].set_index("measure")
    removed = abs(
        slopes.loc["accuracy_class_adjusted", "estimate"]
        / slopes.loc["accuracy", "estimate"]
    )
    assert _agreement(statistics, "slope_share")["estimate"] == pytest.approx(
        1.0 - removed
    )


def test_the_route_difference_is_the_difference_of_the_two_routes():
    statistics = conditional_accuracy_statistics(_composition_driven_functions())

    difference = (
        _agreement(statistics, "endpoint_share")["estimate"]
        - _agreement(statistics, "slope_share")["estimate"]
    )
    assert _agreement(statistics, "endpoint_minus_slope_share")[
        "estimate"
    ] == pytest.approx(difference)


def test_only_the_route_difference_carries_a_test_against_zero():
    """A share has no null at zero -- "this route attributes none of the
    decline to composition" is not a hypothesis -- so a p on those rows would
    only ever be quoted by mistake."""
    statistics = conditional_accuracy_statistics(_composition_driven_functions())

    for measure in ("endpoint_share", "slope_share"):
        row = _agreement(statistics, measure)
        assert row["test"] == "subject_bootstrap_descriptive"
        assert pd.isna(row["p"]) and pd.isna(row["equivalence_bound"])
        # The interval is the whole content of these rows.
        assert row["ci_low"] < row["estimate"] < row["ci_high"]

    difference = _agreement(statistics, "endpoint_minus_slope_share")
    assert difference["test"] == "subject_bootstrap_vs_zero"
    assert difference["equivalence_bound"] > 0


def test_the_route_difference_states_a_bound_rather_than_only_a_null():
    """The routes agreeing to a fraction of a point is worth nothing on its own
    if the comparison could not have resolved a large difference either; the
    bound is what makes that readable."""
    statistics = conditional_accuracy_statistics(_composition_driven_functions())
    row = _agreement(statistics, "endpoint_minus_slope_share")

    assert row["equivalence_bound"] == pytest.approx(
        max(abs(row["ci_low"]), abs(row["ci_high"])), rel=0.5
    )
    assert row["equivalence_bound_coverage"] == 0.90
    assert row["coverage"] == 0.95


def test_the_agreement_is_reported_in_both_slope_units():
    """The endpoint route does not depend on the unit; the slope route does,
    and if the two units disagreed the write-up could not quote either."""
    statistics = conditional_accuracy_statistics(_composition_driven_functions())

    per_bin = _agreement(statistics, "endpoint_share")
    per_second = _agreement(
        statistics, "endpoint_share", slope_unit="accuracy_slope_per_second"
    )
    assert per_bin["estimate"] == pytest.approx(per_second["estimate"])
    assert {"accuracy_slope_per_bin", "accuracy_slope_per_second"} <= set(
        statistics.loc[
            statistics["term"] == "composition_share_agreement", "slope_unit"
        ]
    )


def test_nothing_to_decompose_emits_no_agreement_rows():
    functions = conditional_accuracy_functions(
        trial_features(
            subject=["H01"] * 4 + ["H02"] * 4,
            run_trial_index=[1, 2, 3, 4] * 2,
            dt_ms=[900.0, 1000.0, 1100.0, 1200.0] * 2,
            isCorrect=[True, True, False, False] * 2,
        ),
        n_bins=1,
    )

    statistics = conditional_accuracy_statistics(functions)

    assert "composition_share_agreement" not in set(statistics["term"])


def test_every_slope_carries_the_bound_it_can_rule_out():
    statistics = conditional_accuracy_statistics(_falling_accuracy_functions())
    slopes = statistics.loc[statistics["term"] == "accuracy_slope_per_bin"]

    assert (slopes["equivalence_bound"] > 0).all()
    # The interval and the bound are read at different coverages, and each says
    # which -- a bound silently relabelled 95% is the defect this guards.
    assert (slopes["coverage"] == 0.95).all()
    assert (slopes["equivalence_bound_coverage"] == 0.90).all()


def _order_features(subject="H01"):
    """One class reaching a 2-1 lead by two different token orders."""
    cell = 6
    return trial_features(
        subject=[subject] * (2 * cell),
        run_trial_index=list(range(1, 2 * cell + 1)),
        sp_design_early=[0.6] * (2 * cell),
        trial_class=[1] * (2 * cell),
        trial_class_name=["easy"] * (2 * cell),
        correct_side=[1] * (2 * cell),
        # "112..." leads with the correct target; "211..." does not. Same lead
        # after three tokens, opposite first jump.
        token_directions=["112121212121212"] * cell + ["211121212121212"] * cell,
        # A correct first jump is slower here, as it is in the real data.
        dt_ms=[1200.0] * cell + [1000.0] * cell,
        isCorrect=[True] * (2 * cell - 1) + [False],
    )


def test_the_order_profile_reads_the_same_tokens_the_level_summarises():
    profile = _pooled(early_evidence_order_profile(_order_features()))

    assert set(profile["token_order"]) == {"CCW", "WCC"}
    assert set(profile["first_token"]) == {"correct", "wrong"}
    # Three characters, one per jump, over the window `sp_design_early` covers.
    assert profile["token_order"].str.len().eq(3).all()


def test_order_is_coded_against_the_correct_target_not_a_fixed_side():
    """Otherwise the code would mean opposite things on left- and right-correct
    trials and could not be pooled."""
    left = early_evidence_order_profile(
        _order_features().assign(correct_side=1, nCorrectChoice=1)
    )
    mirrored = _order_features().assign(correct_side=2, nCorrectChoice=2)
    mirrored["token_directions"] = mirrored["token_directions"].str.translate(
        str.maketrans("12", "21")
    )
    right = early_evidence_order_profile(mirrored)

    assert set(left["token_order"]) == set(right["token_order"])


def test_the_first_token_contrast_holds_class_and_level_fixed():
    features = pd.concat(
        [_order_features(f"H{index:02d}") for index in range(1, 5)],
        ignore_index=True,
    )
    profile = early_evidence_order_profile(features)

    statistics = early_evidence_order_statistics(profile)
    row = statistics.loc[
        (statistics["term"] == "first_token_contrast")
        & (statistics["measure"] == "mean_dt_ms")
    ].iloc[0]

    assert row["trial_class_name"] == "easy"
    assert row["sp_design_early"] == 0.6
    assert row["n_subjects"] == 4
    # Correct-first minus wrong-first, the direction the real data shows.
    assert row["mean_difference"] == pytest.approx(200.0)


def test_order_composition_shows_a_class_confined_to_one_order():
    """The fact the write-up leans on: classes at one level can hold disjoint
    orders, so 'matched on early evidence' is true only of the summary."""
    features = pd.concat(
        [_order_features(f"H{index:02d}") for index in range(1, 5)],
        ignore_index=True,
    )
    only_correct_first = features.loc[
        features["token_directions"].str.startswith("1")
    ]

    statistics = early_evidence_order_statistics(
        early_evidence_order_profile(only_correct_first)
    )
    shares = statistics.loc[statistics["term"] == "order_composition"]

    assert shares["token_order"].tolist() == ["CCW"]
    assert shares["estimate"].iloc[0] == pytest.approx(1.0)


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


def test_every_coefficient_records_how_many_levels_its_predictor_had():
    """A per-unit slope over two levels is a difference of means; the count
    is what stops it being read as a tested linear trend."""
    features = trial_features(
        sp_design_early=[0.2, 0.4, 0.6, 0.8],
        dt_ms=[1400.0, 1300.0, 1200.0, 1000.0],
        isCorrect=[False, False, True, True],
    )

    fits = _pooled(
        continuous_evidence_effects(features, predictors={"sp_design_early": 0.5})
    ).iloc[0]

    # Signed evidence keeps four levels; its distance from chance has two.
    assert fits["n_signed_levels"] == 4
    assert fits["n_strength_levels"] == 2

    statistics = continuous_evidence_statistics(
        continuous_evidence_effects(features, predictors={"sp_design_early": 0.5})
    )
    by_term = statistics.set_index(["term", "condition"])["n_predictor_levels"]
    assert by_term[("dt_slope_ms_per_unit", "all")] == 2
    assert by_term[("accuracy_log_odds_per_unit", "all")] == 4


# --- early-evidence profile ---------------------------------------------------


def _early_evidence_features(subject="H01"):
    """Two classes that share an early-evidence level and diverge after it.

    Six trials per class-by-level cell, which is what
    ``SPARSE_CELL_MIN_TRIALS`` requires before a contrast will read a cell.
    """
    cell = 6
    return trial_features(
        subject=[subject] * (4 * cell),
        run_trial_index=list(range(1, 4 * cell + 1)),
        sp_design_early=([0.6] * cell + [0.8] * cell) * 2,
        trial_class=[1] * (2 * cell) + [2] * (2 * cell),
        trial_class_name=["easy"] * (2 * cell) + ["ambiguous"] * (2 * cell),
        # At the shared level (0.6) easy is fast and ambiguous slow; both
        # speed up by 200 ms at 0.8. Early evidence moves behaviour, and so
        # does class, and the two are separable here by construction.
        dt_ms=(
            [1000.0, 1100.0] * (cell // 2) + [800.0, 900.0] * (cell // 2)
            + [1400.0, 1500.0] * (cell // 2) + [1200.0, 1300.0] * (cell // 2)
        ),
        isCorrect=[True] * (4 * cell - 1) + [False],
    )


def test_the_profile_resolves_each_evidence_level_by_trial_class():
    profile = _pooled(early_evidence_profile(_early_evidence_features()))

    assert sorted(profile["sp_design_early"].unique()) == [0.6, 0.8]
    shared = profile.loc[profile["sp_design_early"] == 0.6].set_index(
        "trial_class_name"
    )
    # The class rows partition the pooled row at every level.
    assert shared.loc["all", "n_trials"] == 12
    assert shared.loc["easy", "n_trials"] == 6
    assert shared.loc["ambiguous", "n_trials"] == 6
    assert shared.loc["easy", "mean_dt_ms"] == pytest.approx(1050.0)
    assert shared.loc["ambiguous", "mean_dt_ms"] == pytest.approx(1450.0)


def test_the_profile_reports_the_log_odds_of_each_level_beside_it():
    """`sum_log_lr_design_early` is logit(sp), the exact posterior -- not the
    first-order SumLogLR, which is a different scale."""
    profile = _pooled(early_evidence_profile(_early_evidence_features()))

    level = profile.loc[profile["sp_design_early"] == 0.8].iloc[0]
    assert level["sum_log_lr_design_early"] == pytest.approx(np.log(0.8 / 0.2))


def test_a_class_absent_from_a_level_is_absent_rather_than_zero_filled():
    features = trial_features(
        sp_design_early=[0.6, 0.6, 0.8, 0.8],
        trial_class=[1, 1, 2, 2],
        trial_class_name=["easy", "easy", "ambiguous", "ambiguous"],
        dt_ms=[900.0, 1000.0, 1300.0, 1400.0],
        isCorrect=[True, True, False, True],
    )

    profile = _pooled(early_evidence_profile(features))

    at_low = profile.loc[profile["sp_design_early"] == 0.6, "trial_class_name"]
    assert set(at_low) == {"all", "easy"}


def test_the_class_contrast_at_a_level_is_registered_and_within_subject():
    features = pd.concat(
        [_early_evidence_features(f"H{index:02d}") for index in range(1, 5)],
        ignore_index=True,
    )
    profile = early_evidence_profile(features)

    statistics = continuous_evidence_statistics(
        continuous_evidence_effects(features), profile
    )
    contrast = statistics.loc[
        (statistics["term"] == "class_contrast_at_level")
        & (statistics["measure"] == "mean_dt_ms")
        & (statistics["sp_design_early"] == 0.6)
    ].iloc[0]

    assert contrast["trial_class_name"] == "ambiguous_vs_easy"
    assert contrast["n_subjects"] == 4
    assert contrast["mean_difference"] == pytest.approx(400.0)


def test_the_within_class_evidence_contrast_is_registered_in_its_own_units():
    """The honest form of `dt_slope_ms_per_unit`: a level-to-level contrast
    in milliseconds, holding the class fixed."""
    features = pd.concat(
        [_early_evidence_features(f"H{index:02d}") for index in range(1, 5)],
        ignore_index=True,
    )

    statistics = continuous_evidence_statistics(
        continuous_evidence_effects(features), early_evidence_profile(features)
    )
    within = statistics.loc[
        (statistics["term"] == "evidence_contrast_within_class")
        & (statistics["measure"] == "mean_dt_ms")
        & (statistics["trial_class_name"] == "easy")
    ].iloc[0]

    assert within["sp_design_early"] == 0.8
    assert within["sp_design_early_reference"] == 0.6
    assert within["mean_difference"] == pytest.approx(-200.0)


def test_a_contrast_never_reaches_a_cell_too_thin_to_support_it():
    """A one-trial cell would silently restrict the contrast to whoever owns
    one, so cells below the threshold are dropped before any contrast."""
    features = pd.concat(
        [
            trial_features(
                subject=[f"H{index:02d}"] * 13,
                run_trial_index=list(range(1, 14)),
                # Six trials at each of two levels, plus a single trial at a
                # third that no contrast should be able to see.
                sp_design_early=[0.6] * 6 + [0.8] * 6 + [0.2],
                trial_class=[1] * 13,
                trial_class_name=["easy"] * 13,
                dt_ms=[1200.0] * 6 + [1000.0] * 6 + [700.0],
                isCorrect=[True] * 13,
            )
            for index in range(1, 5)
        ],
        ignore_index=True,
    )
    profile = early_evidence_profile(features)
    assert 0.2 in set(profile["sp_design_early"])  # kept in the derivative

    statistics = continuous_evidence_statistics(
        continuous_evidence_effects(features), profile
    )
    within = statistics.loc[
        (statistics["term"] == "evidence_contrast_within_class")
        & (statistics["measure"] == "mean_dt_ms")
    ].iloc[0]

    assert within["sp_design_early_reference"] == 0.6
    assert within["mean_difference"] == pytest.approx(-200.0)
