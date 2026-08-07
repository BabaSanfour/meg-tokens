"""Tests for the docs/behavior_analysis_roadmap.md analyses."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from meg_tokens.behavior.analyses.design_effects import (
    choice_side_summary,
    condition_class_cells,
    condition_class_statistics,
    condition_order_effects,
    extreme_decision_times,
    lapse_summary,
    time_on_task,
)
from meg_tokens.behavior.analyses.distributions import (
    distribution_summary,
    spd_cumulative_distributions,
)
from meg_tokens.behavior.analyses.evidence import (
    conditional_accuracy_functions,
    continuous_evidence_effects,
    criterion_decline,
    reverse_correlation,
)
from meg_tokens.behavior.analyses.individual import (
    individual_correlations,
    individual_profile,
)
from meg_tokens.behavior.math.inference import (
    one_sample_statistics,
    repeated_measures_anova,
)
from meg_tokens.behavior.math.evidence import (
    MAX_LOG_ODDS,
    log_posterior_odds,
    sum_log_lr_profile,
    token_lead_profile,
)
from meg_tokens.behavior.schema import parse_token_directions
from meg_tokens.behavior.analyses.sequential import (
    choice_history,
    robust_post_error_slowing,
)
from meg_tokens.core import ProjectConfig
from meg_tokens.io import DerivativeLayout, save_table
from meg_tokens.workflows.behavior_analysis import analyze_behavior
from meg_tokens.workflows.behavior_extended import (
    ROADMAP_ITEMS,
    analyze_behavior_extended,
)


TOKENS = "121212121212121"


def make_features(**columns) -> pd.DataFrame:
    """Build a trial-feature table with roadmap-relevant defaults."""
    length = len(next(iter(columns.values())))
    defaults = {
        "subject": ["H01"] * length,
        "condition": ["Fast"] * length,
        "run": [1] * length,
        "run_trial_index": list(range(1, length + 1)),
        "initial_time_ms": [1000 * index for index in range(length)],
        "trial_class": [1] * length,
        "trial_class_name": ["easy"] * length,
        "dt_ms": [1000.0] * length,
        "logged_spd": [0.8] * length,
        "logged_spd_log_odds": [float(np.log(0.8 / 0.2))] * length,
        "logged_spd_validated_15row": [0.8] * length,
        "decision_token_index": [5] * length,
        "token_directions": [TOKENS] * length,
        "sp_design_early": [0.6] * length,
        "sum_log_lr_design_early": [float(np.log(0.6 / 0.4))] * length,
        "choice_side": [1] * length,
        "correct_side": [1] * length,
        "isCorrect": [True] * length,
        "nOutcome": [0] * length,
        "is_started": [True] * length,
        "has_choice": [True] * length,
        "primary_analysis_eligible": [True] * length,
    }
    defaults.update(columns)
    table = pd.DataFrame(defaults)
    table["trial_id"] = [
        f"{row.subject}_{row.condition}_{row.run}_{row.run_trial_index:03d}"
        for row in table.itertuples()
    ]
    return table


# --- inference ---------------------------------------------------------------


def test_one_sample_statistics_matches_scipy():
    values = [1.0, 2.0, 4.0, 3.0, 2.5]
    result = one_sample_statistics(values)
    expected = stats.ttest_1samp(values, 0.0)
    assert result["n_subjects"] == 5
    assert result["t"] == pytest.approx(float(expected.statistic))
    assert result["p"] == pytest.approx(float(expected.pvalue))
    assert result["df"] == 4.0


def test_one_sample_statistics_is_explicit_when_empty():
    result = one_sample_statistics([])
    assert result["n_subjects"] == 0
    assert np.isnan(result["mean"])


def _anova_matrix() -> np.ndarray:
    return np.array(
        [
            [10.0, 14.0, 19.0, 12.0, 17.0, 20.0],
            [11.0, 16.0, 18.0, 15.0, 18.0, 24.0],
            [9.0, 13.0, 21.0, 11.0, 16.0, 23.0],
            [12.0, 15.0, 20.0, 14.0, 19.0, 22.0],
            [8.0, 12.0, 17.0, 13.0, 15.0, 21.0],
        ]
    )


def test_repeated_measures_anova_two_level_main_effect_equals_paired_t_squared():
    matrix = _anova_matrix()
    effects = {row["effect"]: row for row in repeated_measures_anova(matrix, (2, 3))}
    first = matrix[:, :3].mean(axis=1)
    second = matrix[:, 3:].mean(axis=1)
    expected = stats.ttest_rel(first, second)
    assert effects["factor1"]["F"] == pytest.approx(float(expected.statistic) ** 2)
    assert effects["factor1"]["p"] == pytest.approx(float(expected.pvalue))


def test_repeated_measures_anova_interaction_excludes_the_main_effects():
    """A cell-mean deviation still contains both main effects; the interaction
    term must not. The 2x3 interaction equals a one-way repeated-measures test
    on the by-subject differences between the two levels of the first factor.
    """
    matrix = _anova_matrix()
    effects = {row["effect"]: row for row in repeated_measures_anova(matrix, (2, 3))}

    differences = matrix[:, :3] - matrix[:, 3:]
    n_subjects, n_levels = differences.shape
    grand = differences.mean()
    level_means = differences.mean(axis=0)
    subject_means = differences.mean(axis=1)
    residual = differences - level_means[None, :] - subject_means[:, None] + grand
    expected = (
        n_subjects * np.sum((level_means - grand) ** 2) / (n_levels - 1)
    ) / (np.sum(residual**2) / ((n_levels - 1) * (n_subjects - 1)))

    assert effects["factor1xfactor2"]["F"] == pytest.approx(float(expected))
    assert effects["factor1xfactor2"]["df_effect"] == 2.0
    assert effects["factor1xfactor2"]["df_error"] == 8.0


# --- evidence -----------------------------------------------------------------


def test_parse_token_directions_rejects_impossible_targets():
    assert parse_token_directions("1221") == [1, 2, 2, 1]
    with pytest.raises(ValueError, match="only 1 and 2"):
        parse_token_directions("1231")
    with pytest.raises(ValueError, match="non-empty string"):
        parse_token_directions(None)
    with pytest.raises(ValueError, match="only 1 and 2"):
        parse_token_directions("1,2")


def test_token_lead_profile_tracks_the_running_difference():
    assert token_lead_profile([1, 1, 2, 1], target=1) == [1, 2, 1, 2]
    assert token_lead_profile([1, 1, 2, 1], target=2) == [-1, -2, -1, -2]
    with pytest.raises(TypeError, match="sequence of integer"):
        token_lead_profile("1121", target=1)


def test_sum_log_lr_profile_follows_success_probability():
    profile = sum_log_lr_profile([1] * 8 + [2] * 7, target=1)
    # The eighth token secures the majority: evidence saturates and stays there.
    assert profile[7] == pytest.approx(MAX_LOG_ODDS)
    assert profile[-1] == pytest.approx(MAX_LOG_ODDS)
    assert profile[0] < profile[3] < profile[7]


def test_evidence_profile_lookup_rejects_out_of_range_token_counts():
    from meg_tokens.behavior.math.evidence import evidence_after_tokens

    assert evidence_after_tokens([0.6, 0.7], 0, prior=0.5) == 0.5
    assert evidence_after_tokens([0.6, 0.7], 2, prior=0.5) == 0.7
    with pytest.raises(ValueError, match="non-negative"):
        evidence_after_tokens([0.6, 0.7], -1, prior=0.5)
    with pytest.raises(ValueError, match="cannot exceed"):
        evidence_after_tokens([0.6, 0.7], 3, prior=0.5)


def test_log_posterior_odds_flags_certainty_instead_of_returning_infinity():
    odds, saturated = log_posterior_odds(1.0)
    assert saturated and odds == pytest.approx(MAX_LOG_ODDS)
    odds, saturated = log_posterior_odds(0.0)
    assert saturated and odds == pytest.approx(-MAX_LOG_ODDS)
    odds, saturated = log_posterior_odds(0.5)
    assert not saturated and odds == pytest.approx(0.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        log_posterior_odds(1.5)


def test_criterion_decline_recovers_a_planted_slope():
    tokens = [2, 4, 6, 8, 10, 12]
    features = make_features(
        decision_token_index=tokens,
        logged_spd=[0.5 + 0.02 * value for value in tokens],
    )
    fits = criterion_decline(features)
    overall = fits.loc[fits["condition"] == "all"].iloc[0]
    assert overall["slope"] == pytest.approx(0.02)
    assert overall["intercept"] == pytest.approx(0.5)
    assert overall["n_trials"] == 6


def test_reverse_correlation_ignores_tokens_that_fell_after_the_choice():
    # Every trial commits after two tokens, so jumps 3 and later were never
    # visible and must not contribute a kernel weight.
    features = make_features(
        decision_token_index=[2] * 6,
        choice_side=[1, 1, 1, 2, 2, 2],
        token_directions=["112" + "2" * 12] * 6,
    )
    kernels = reverse_correlation(features, n_jumps=4)
    overall = kernels.loc[kernels["condition"] == "all"].set_index("jump")
    assert overall.loc[1, "n_trials_token_seen"] == 6
    assert overall.loc[3, "n_trials_token_seen"] == 0
    assert np.isnan(overall.loc[3, "mean_signed_direction"])
    # Both tokens went to target 1, so subjects choosing target 1 have a
    # positive kernel and those choosing target 2 a negative one; averaged
    # over an even split the model-free kernel is zero.
    assert overall.loc[1, "mean_signed_direction"] == pytest.approx(0.0)


def test_conditional_accuracy_functions_bin_within_subject():
    features = make_features(
        subject=["H01"] * 4 + ["H02"] * 4,
        dt_ms=[500.0, 700.0, 900.0, 1100.0, 5000.0, 5200.0, 5400.0, 5600.0],
        isCorrect=[True, True, False, False, True, True, False, False],
        run_trial_index=[1, 2, 3, 4, 1, 2, 3, 4],
    )
    functions = conditional_accuracy_functions(features, n_bins=2)
    overall = functions.loc[functions["condition"] == "all"]
    assert set(overall["dt_bin"]) == {1, 2}
    # Each subject's own quantiles split its own trials, so the slow subject
    # is not pushed entirely into the slow bin.
    assert overall.loc[overall["dt_bin"] == 1, "accuracy"].tolist() == [1.0, 1.0]
    assert overall.loc[overall["dt_bin"] == 2, "accuracy"].tolist() == [0.0, 0.0]


def test_continuous_evidence_effects_keep_unclassified_trials():
    features = make_features(
        trial_class=[1, 2, 0, 0],
        trial_class_name=["easy", "ambiguous", "unclassified", "unclassified"],
        sp_design_early=[0.9, 0.6, 0.5, 0.8],
        dt_ms=[800.0, 1200.0, 1400.0, 900.0],
        isCorrect=[True, True, False, True],
    )
    effects = continuous_evidence_effects(
        features, predictors={"sp_design_early": 0.5}
    )
    overall = effects.loc[effects["condition"] == "all"].iloc[0]
    assert overall["n_dt_trials"] == 4
    assert overall["dt_slope_ms_per_unit"] < 0


# --- design, session, and lapse effects ---------------------------------------


def test_condition_class_cells_cover_every_cell_and_feed_the_anova():
    features = make_features(
        subject=["H01"] * 6 + ["H02"] * 6,
        condition=(["Fast"] * 3 + ["Slow"] * 3) * 2,
        trial_class=[1, 2, 3] * 4,
        dt_ms=[800.0, 1200.0, 1100.0, 900.0, 1400.0, 1300.0] * 2,
        isCorrect=[True, True, False, True, False, True] * 2,
        run=[1, 1, 1, 2, 2, 2] * 2,
        run_trial_index=[1, 2, 3, 1, 2, 3] * 2,
    )
    cells = condition_class_cells(features)
    assert len(cells) == 12
    assert set(cells["trial_class_name"]) == {"easy", "ambiguous", "misleading"}
    assert cells["n_trials"].tolist() == [1] * 12

    statistics = condition_class_statistics(cells)
    assert set(statistics["effect"]) == {
        "condition",
        "trial_class",
        "condition_x_trial_class",
    }
    assert set(statistics["measure"]) == {"mean_dt_ms", "accuracy"}


def test_analyses_reject_noncanonical_correctness_values():
    features = make_features(
        isCorrect=["true", "false"],
        run_trial_index=[1, 2],
    )
    with pytest.raises(ValueError, match="isCorrect must be boolean"):
        condition_class_cells(features)
    with pytest.raises(ValueError, match="isCorrect must be boolean"):
        choice_history(features)


def test_choice_side_summary_reports_balance_and_asymmetry():
    features = make_features(
        choice_side=[1, 1, 1, 2],
        correct_side=[1, 1, 2, 2],
        dt_ms=[900.0, 1100.0, 1000.0, 1400.0],
        isCorrect=[True, True, False, True],
    )
    summary = choice_side_summary(features)
    overall = summary.loc[summary["condition"] == "all"].iloc[0]
    assert overall["proportion_left_choices"] == pytest.approx(0.75)
    assert overall["proportion_right_choices"] == pytest.approx(0.25)
    assert overall["mean_left_dt_ms"] == pytest.approx(1000.0)
    assert overall["mean_right_dt_ms"] == pytest.approx(1400.0)
    assert overall["accuracy_left"] == pytest.approx(2 / 3)


def test_time_on_task_orders_blocks_by_the_session_clock_not_by_run_number():
    """Fast and Slow blocks interleave, so block order has to come from the
    session clock. Slow run 1 here ran before Fast run 2.
    """
    features = make_features(
        condition=["Fast"] * 2 + ["Slow"] * 2 + ["Fast"] * 2,
        run=[1, 1, 1, 1, 2, 2],
        run_trial_index=[1, 2, 1, 2, 1, 2],
        initial_time_ms=[100, 200, 300, 400, 500, 600],
        dt_ms=[1000.0, 1010.0, 1200.0, 1210.0, 900.0, 910.0],
    )
    drift = time_on_task(features)
    fast = drift.loc[drift["condition"] == "fast"].iloc[0]
    assert fast["n_blocks"] == 2
    # Fast blocks are session positions 1 and 3, and got 100 ms faster.
    assert fast["dt_per_block_ms"] < 0

    order = condition_order_effects(features)
    assert order["first_condition"].tolist() == ["fast"]


def test_lapse_summary_counts_started_trials_without_a_choice():
    features = make_features(
        has_choice=[True, True, False, False],
        primary_analysis_eligible=[True, True, False, False],
        nOutcome=[0, 0, 7006, 7011],
        dt_ms=[900.0, 1000.0, float("nan"), float("nan")],
    )
    summary = lapse_summary(features)
    overall = summary.loc[summary["condition"] == "all"].iloc[0]
    assert overall["n_started_trials"] == 4
    assert overall["n_lapse_trials"] == 2
    assert overall["lapse_rate"] == pytest.approx(0.5)
    assert overall["n_outcome_7006_reaction_time_too_long"] == 1
    assert overall["n_outcome_7011_delay_1_error"] == 1


def test_extreme_decision_times_flags_without_removing():
    features = make_features(
        dt_ms=[900.0, 950.0, 1000.0, 1050.0, 1100.0, 20000.0],
    )
    counts, flagged = extreme_decision_times(features, mad_threshold=5.0)
    assert counts.iloc[0]["n_dt_trials"] == 6
    assert counts.iloc[0]["n_extreme_dt"] == 1
    assert counts.iloc[0]["max_dt_ms"] == pytest.approx(20000.0)
    assert flagged["dt_ms"].tolist() == pytest.approx([20000.0])
    assert flagged.iloc[0]["robust_z"] > 5.0


# --- history and individual differences ---------------------------------------


def test_robust_post_error_slowing_compares_across_the_error():
    # Trial 3 is the error: the robust measure is DT(4) - DT(2) = 300 ms,
    # while the classical post-error/post-correct contrast is larger because
    # it also compares against the fast first two trials.
    features = make_features(
        run_trial_index=[1, 2, 3, 4, 5],
        dt_ms=[900.0, 1000.0, 950.0, 1300.0, 1000.0],
        isCorrect=[True, True, False, True, True],
    )
    slowing = robust_post_error_slowing(features)
    overall = slowing.loc[slowing["condition"] == "all"].iloc[0]
    assert overall["n_error_pairs"] == 1
    assert overall["robust_pes_ms"] == pytest.approx(300.0)
    assert overall["mean_pre_error_dt_ms"] == pytest.approx(1000.0)
    assert overall["mean_post_error_dt_ms"] == pytest.approx(1300.0)
    assert overall["n_post_error_trials"] == 1
    assert overall["n_post_correct_trials"] == 3


def test_post_error_pairs_do_not_cross_a_gap_in_the_run():
    features = make_features(
        run_trial_index=[1, 2, 5, 6],
        dt_ms=[900.0, 1000.0, 950.0, 1300.0],
        isCorrect=[True, False, False, True],
    )
    slowing = robust_post_error_slowing(features)
    overall = slowing.loc[slowing["condition"] == "all"].iloc[0]
    # Trial 2 is an error but trial 5 is not its neighbour, so no usable pair.
    assert overall["n_error_pairs"] == 0
    assert np.isnan(overall["robust_pes_ms"])


def test_choice_history_reports_win_stay_and_lose_shift():
    features = make_features(
        run_trial_index=[1, 2, 3, 4, 5],
        choice_side=[1, 1, 2, 2, 1],
        isCorrect=[True, False, True, False, True],
        dt_ms=[900.0, 1000.0, 1100.0, 1200.0, 1300.0],
    )
    history = choice_history(features)
    overall = history.loc[history["condition"] == "all"].iloc[0]
    assert overall["n_linked_trials"] == 4
    # Trials 2 and 4 follow a correct trial; trial 2 repeats, trial 4 repeats.
    assert overall["win_stay"] == pytest.approx(1.0)
    # Trials 3 and 5 follow an error; both switch side.
    assert overall["lose_stay"] == pytest.approx(0.0)
    assert overall["lose_shift"] == pytest.approx(1.0)


def test_individual_profile_joins_subject_measures_and_correlates_them():
    summary = pd.DataFrame(
        {
            "subject": ["H01", "H02", "H03", "H04"],
            "mean_fast_dt_ms": [900.0, 1000.0, 1100.0, 1200.0],
            "mean_slow_dt_ms": [1100.0, 1300.0, 1400.0, 1700.0],
            "percent_correct": [80.0, 78.0, 75.0, 70.0],
        }
    )
    urgency = pd.DataFrame(
        {
            "subject": ["H01", "H02", "H03", "H04"],
            "condition": ["all"] * 4,
            "response": ["logged_spd"] * 4,
            "slope": [-0.05, -0.04, -0.03, -0.01],
        }
    )
    profile = individual_profile(summary, urgency=urgency)
    assert profile["sat_adjustment_ms"].tolist() == pytest.approx(
        [200.0, 300.0, 300.0, 500.0]
    )
    assert profile["urgency_slope_per_second"].tolist() == pytest.approx(
        [-0.05, -0.04, -0.03, -0.01]
    )

    correlations = individual_correlations(profile)
    pair = correlations.loc[
        (correlations["measure_a"] == "mean_dt_ms")
        & (correlations["measure_b"] == "urgency_slope_per_second")
    ].iloc[0]
    assert pair["n_subjects"] == 4
    assert pair["pearson_r"] == pytest.approx(
        float(
            stats.pearsonr(
                profile["mean_dt_ms"], profile["urgency_slope_per_second"]
            ).statistic
        )
    )


# --- distributions ------------------------------------------------------------


def test_distribution_summary_reports_quantiles_and_withholds_unstable_shape():
    values = list(range(1, 25))
    summary = distribution_summary(values, quantiles=(0.25, 0.5, 0.75))
    assert summary["n_trials"] == 24
    assert summary["q50"] == pytest.approx(12.5)
    assert summary["q25"] == pytest.approx(float(np.quantile(values, 0.25)))
    assert np.isfinite(summary["skewness"])

    short = distribution_summary([1.0, 2.0, 3.0])
    assert np.isnan(short["skewness"])


def test_spd_cumulative_distribution_is_monotone_and_reaches_one():
    features = make_features(
        trial_class=[1, 1, 2, 2, 3, 3],
        trial_class_name=["easy", "easy", "ambiguous", "ambiguous", "misleading", "misleading"],
        logged_spd=[0.55, 0.95, 0.60, 0.75, 0.45, 0.85],
        logged_spd_validated_15row=[0.55, 0.95, 0.60, 0.75, 0.45, 0.85],
    )
    curves = spd_cumulative_distributions(features)
    easy = curves.loc[
        (curves["view"] == "all_logged") & (curves["trial_class_name"] == "easy")
    ].sort_values("threshold")
    assert easy["pooled_proportion"].is_monotonic_increasing
    assert easy["pooled_proportion"].iloc[-1] == pytest.approx(1.0)
    assert easy["pooled_proportion"].iloc[0] == pytest.approx(0.0)
    assert set(curves["view"]) == {"all_logged", "validated_15row"}


# --- workflow -----------------------------------------------------------------


def _run_table(subject, condition, run, enter_times, choices, correct, initial):
    n_trials = len(enter_times)
    return pd.DataFrame(
        {
            "subject": [subject] * n_trials,
            "condition": [condition] * n_trials,
            "run": [run] * n_trials,
            "source_file": [f"{subject}{condition}{run}_recording.tdms"] * n_trials,
            "nTrialIndex": list(range(1, n_trials + 1)),
            "sTrialClass": [1, 2, 3, 1][:n_trials],
            "sTrialClassRaw": ["e", "a", "m", "e"][:n_trials],
            "trial_class_source": ["design"] * n_trials,
            "trial_class_rule": ["recorded_label"] * n_trials,
            "sp_design_correct": ["[0.6, 0.8]"] * n_trials,
            "nInitialTime": [initial + 1000 * index for index in range(n_trials)],
            "nChoiceMade": choices,
            "nCorrectChoice": correct,
            "tGO": [1000] * n_trials,
            "tEnterCenter": [0] * n_trials,
            "tExitCenter": enter_times,
            "tEnterTarget": enter_times,
            "tTrialEnd": [value + 200 for value in enter_times],
            "sTokenDirs": [TOKENS] * n_trials,
            "nTokenNum": ["[1, 2]"] * n_trials,
            "nTokenDir": ["[1, 2]"] * n_trials,
            "tTime": ["[1100, 1300]"] * n_trials,
            "nProb": ["[0.6, 0.8]"] * n_trials,
            "token_log_rows": [2] * n_trials,
            "token_log_short": [False] * n_trials,
            "nOutcome": [0] * n_trials,
            "rawRT": [value - 1000 for value in enter_times],
            "isCorrect": [
                choice == expected for choice, expected in zip(choices, correct)
            ],
        }
    )


def _stage_behavior(bids_root):
    layout = DerivativeLayout(bids_root)
    for index, subject in enumerate(("H01", "H02", "H03")):
        offset = 100 * index
        tables = [
            _run_table(subject, "RT", 1, [1300, 1400], [1, 2], [1, 2], 0),
            _run_table(
                subject,
                "Fast",
                1,
                [1900 + offset, 2100 + offset, 2000 + offset, 2200 + offset],
                [1, 2, 1, 2],
                [1, 1, 2, 2],
                10_000,
            ),
            _run_table(
                subject,
                "Slow",
                1,
                [2400 + offset, 2600 + offset, 2500 + offset, 2700 + offset],
                [1, 2, 2, 1],
                [1, 2, 2, 2],
                20_000,
            ),
        ]
        for table in tables:
            save_table(
                layout.behavior(
                    subject=subject,
                    run=str(table["run"].iloc[0]),
                    condition=table["condition"].iloc[0],
                ),
                table,
                metadata={"stage": "behavior_parsing"},
            )


def test_analyze_behavior_extended_writes_every_roadmap_table(tmp_path):
    project = ProjectConfig(data_root=tmp_path)
    _stage_behavior(project.bids_root)
    analyze_behavior(project)

    result = analyze_behavior_extended(project)

    assert result.stage == "behavior_extended_analysis"
    assert result.settings["n_subjects"] == 3
    layout = DerivativeLayout(project.bids_root)
    for name in ROADMAP_ITEMS:
        assert layout.behavior_analysis(name) in result.outputs
    for path in result.outputs:
        assert path.is_file()

    cells = pd.read_csv(layout.behavior_analysis("conditionclass"), sep="\t")
    assert set(cells["subject"]) == {"H01", "H02", "H03"}
    criterion = pd.read_csv(layout.behavior_analysis("criteriondecline"), sep="\t")
    assert set(criterion["response"]) == {"logged_spd", "logged_spd_log_odds"}


def test_analyze_behavior_extended_requires_staged_trial_features(tmp_path):
    project = ProjectConfig(data_root=tmp_path)
    with pytest.raises(FileNotFoundError, match="behavior analyze"):
        analyze_behavior_extended(project)


def test_behavior_analysis_paths_are_named_from_the_analysis(tmp_path):
    layout = DerivativeLayout(tmp_path)
    assert layout.behavior_analysis("urgency").as_posix().endswith(
        "sub-group/beh/sub-group_task-tokens_desc-urgency_beh.tsv"
    )
    with pytest.raises(ValueError, match="alphanumeric"):
        layout.behavior_analysis("post_error")
