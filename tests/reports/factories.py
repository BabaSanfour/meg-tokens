"""Schema-exact minimal group derivatives for the reports test suite.

Every factory reproduces the exact column names the real analysis modules
write (`meg_tokens/behavior/analyses/*.py`), with a handful of synthetic
subjects -- a factory that invented a column would let a figure pass its
test and then fail on real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from meg_tokens.io import DerivativeLayout, save_table

from tests.behavior.factories import trial_features as _trial_features_factory

SUBJECTS = ("H01", "H02", "H03", "H04")


def write_group_derivative(
    layout: DerivativeLayout, name: str, frame: pd.DataFrame
) -> None:
    save_table(layout.behavior_analysis(name), frame, metadata={"stage": "test_fixture"})


def ssmcomparison(subjects: tuple[str, ...] = SUBJECTS) -> pd.DataFrame:
    """Two models (ddm, urgency) x three conditions (all, fast, slow) per subject."""
    rows = []
    for index, subject in enumerate(subjects):
        for condition, bic_ddm in (("all", 900.0), ("fast", 420.0), ("slow", 480.0)):
            bic_urgency = bic_ddm - 220.0 - 10.0 * index
            rows.append({
                "subject": subject, "condition": condition, "model": "ddm",
                "n_trials": 300, "n_token_sequences": 76, "n_parameters": 3,
                "log_likelihood": -bic_ddm / 2, "t_dur_s": 3.0,
                "aic": bic_ddm - 5, "bic": bic_ddm,
                "delta_aic": 0.0, "delta_bic": 0.0, "converged": True,
                "drift_scale": 1.0 + 0.05 * index, "bound": 0.8 + 0.03 * index, "nondecision_s": 0.5,
                "drift_scale_se": 0.1, "bound_se": 0.08, "nondecision_s_se": 0.05,
                "urgency_scale": np.nan, "urgency_onset_s": np.nan,
                "urgency_scale_se": np.nan, "urgency_onset_s_se": np.nan,
            })
            urgency_scale = 1.5 + (0.1 if condition == "fast" else -0.1) - 0.02 * index
            rows.append({
                "subject": subject, "condition": condition, "model": "urgency",
                "n_trials": 300, "n_token_sequences": 76, "n_parameters": 4,
                "log_likelihood": -bic_urgency / 2, "t_dur_s": 3.0,
                "aic": bic_urgency - 5, "bic": bic_urgency,
                "delta_aic": bic_urgency - 5 - (bic_ddm - 5),
                "delta_bic": bic_urgency - bic_ddm, "converged": True,
                "drift_scale": 0.2 + 0.01 * index, "bound": np.nan, "nondecision_s": 0.01,
                "urgency_scale": urgency_scale, "urgency_onset_s": 0.05 + 0.01 * index,
                "drift_scale_se": 0.03, "bound_se": np.nan, "nondecision_s_se": 0.01,
                "urgency_scale_se": 0.1, "urgency_onset_s_se": 0.02,
            })
    return pd.DataFrame(rows)


def ssmcomparisonstats(fits: pd.DataFrame) -> pd.DataFrame:
    """Mirrors sequential_sampling.model_comparison_statistics +
    urgency_condition_contrast, computed directly from the fixture fits so
    the stats rows are internally consistent with the fits table."""
    from meg_tokens.behavior.analyses.sequential_sampling import (
        model_comparison_statistics,
        urgency_condition_contrast,
    )

    return pd.concat(
        [model_comparison_statistics(fits), urgency_condition_contrast(fits)],
        ignore_index=True,
    )


def dtdistribution(subjects: tuple[str, ...] = SUBJECTS) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    strata = [
        ("condition", "fast", 950.0), ("condition", "slow", 1250.0),
        ("class", "easy", 900.0), ("class", "ambiguous", 1300.0),
        ("class", "misleading", 1250.0),
    ]
    for index, subject in enumerate(subjects):
        for stratum_type, stratum, base_mean in strata:
            mean = base_mean + 15.0 * index
            rows.append({
                "subject": subject, "stratum_type": stratum_type, "stratum": stratum,
                "n_trials": 60, "mean": mean, "sd": 220.0, "min": mean - 400, "max": mean + 600,
                "q10": mean - 300, "q25": mean - 150, "q50": mean, "q75": mean + 150, "q90": mean + 350,
                "skewness": 0.8, "kurtosis": 1.2,
            })
    return pd.DataFrame(rows)


def dtdistributionstats(distributions: pd.DataFrame) -> pd.DataFrame:
    from meg_tokens.behavior.analyses.distributions import (
        decision_time_distribution_statistics,
    )

    return decision_time_distribution_statistics(distributions)


def groupstats(subjects: tuple[str, ...] = SUBJECTS) -> pd.DataFrame:
    from meg_tokens.behavior.math.inference import paired_subject_statistics

    rng = np.random.default_rng(2)
    fast = pd.Series(950.0 + rng.normal(0, 40, len(subjects)), index=subjects)
    slow = pd.Series(1250.0 + rng.normal(0, 40, len(subjects)), index=subjects)
    row = {
        "analysis": "decision_time", "contrast": "fast_vs_slow", "view": "primary",
        "unit": "ms", "label_a": "Fast", "label_b": "Slow",
        **paired_subject_statistics(fast, slow),
    }
    return pd.DataFrame([row])


def trial_features_group(subjects: tuple[str, ...] = SUBJECTS) -> pd.DataFrame:
    frames = []
    for index, subject in enumerate(subjects):
        for condition, mean_dt in (("Fast", 950.0 + 15 * index), ("Slow", 1250.0 + 15 * index)):
            n = 20
            frames.append(
                _trial_features_factory(
                    subject=[subject] * n,
                    condition=[condition] * n,
                    dt_ms=[mean_dt] * n,
                    run_trial_index=list(range(1, n + 1)),
                    nTrialIndex=list(range(1, n + 1)),
                )
            )
    return pd.concat(frames, ignore_index=True)


_TOKEN_LIBRARY = ("1121212121", "2212121212", "1112122121", "2221211212")


def rich_trial_features(
    subjects: tuple[str, ...] = SUBJECTS, n_trials_per_block: int = 30
) -> pd.DataFrame:
    """A larger, varied trial-feature table sufficient to exercise every
    analysis module's real production code (design effects, evidence,
    sequential effects) without degenerating to all-NaN/empty results.

    Two runs per subject x condition, contiguous run_trial_index within each
    run (required for post-error/choice-history adjacency), varied decision
    times, token directions, and evidence so quantile/logistic/OLS fits have
    something to find.
    """
    rng = np.random.default_rng(7)
    rows = []
    trial_id = 0
    for subject_index, subject in enumerate(subjects):
        base_time = 0
        condition_order = (
            (("Fast", 950.0), ("Slow", 1300.0))
            if subject_index % 2 == 0
            else (("Slow", 1300.0), ("Fast", 950.0))
        )
        for condition, base_dt in condition_order:
            for run in (1, 2):
                for trial_index in range(1, n_trials_per_block + 1):
                    trial_id += 1
                    trial_class_code = int(rng.integers(1, 4))
                    class_name = {1: "easy", 2: "ambiguous", 3: "misleading"}[trial_class_code]
                    raw_label = {1: "e", 2: "a", 3: "m"}[trial_class_code]
                    correct_side = int(rng.integers(1, 3))
                    # Evidence and choice correlated with correct_side, not identical
                    # to it, so accuracy is realistic (not 0% or 100%).
                    noisy_correct = rng.random() < 0.75
                    choice_side = correct_side if noisy_correct else (3 - correct_side)
                    dt = float(base_dt + rng.normal(0, 220) + 40 * (trial_index % 5))
                    dt = max(dt, 250.0)
                    if condition == condition_order[0][0] and run == 1 and trial_index == 1:
                        # One deliberate outlier per subject so extreme_decision_times'
                        # flagged table is never empty (a genuinely all-empty table
                        # has zero columns and cannot round-trip through save_table/
                        # read_csv -- a real, pre-existing edge case, not something
                        # to paper over by avoiding it entirely in this fixture).
                        dt += 4000.0
                    decision_token_index = int(rng.integers(2, 9))
                    signed_evidence = (1.0 if correct_side == 1 else -1.0) * float(rng.uniform(0.2, 1.4))
                    logged_spd = float(np.clip(0.5 + 0.05 * signed_evidence + rng.normal(0, 0.05), 0.05, 0.95))
                    initial_time_ms = base_time
                    base_time += int(dt) + 500
                    rows.append(
                        _trial_features_factory(
                            subject=[subject],
                            condition=[condition],
                            run=[run],
                            block_index=[run],
                            run_trial_index=[trial_index],
                            started_trial_index=[trial_index],
                            nTrialIndex=[trial_index],
                            initial_time_ms=[initial_time_ms],
                            trial_class=[trial_class_code],
                            trial_class_name=[class_name],
                            sTrialClassRaw=[raw_label],
                            nChoiceMade=[choice_side],
                            nCorrectChoice=[correct_side],
                            choice_side=[choice_side],
                            correct_side=[correct_side],
                            isCorrect=[bool(choice_side == correct_side)],
                            dt_ms=[dt],
                            rawRT=[dt + 350.0],
                            logged_spd=[logged_spd],
                            decision_token_index=[decision_token_index],
                            token_directions=[_TOKEN_LIBRARY[trial_id % len(_TOKEN_LIBRARY)]],
                            sp_design_early=[float(np.clip(0.5 + 0.1 * signed_evidence, 0.05, 0.95))],
                            sum_log_lr_design_early=[signed_evidence],
                        ).iloc[0].to_dict()
                    )
    return pd.DataFrame(rows)


def group_derivatives_from_rich_features(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run the real analysis functions over `rich_trial_features()` and return
    every group derivative the design/evidence/sequential figure builders need,
    keyed by derivative name."""
    from meg_tokens.behavior.analyses.design_effects import (
        condition_class_cells, condition_class_statistics,
        choice_side_summary, choice_side_statistics,
        time_on_task, time_on_task_statistics,
        condition_order_effects, condition_order_statistics,
        lapse_summary, lapse_statistics, extreme_decision_times,
    )
    from meg_tokens.behavior.analyses.distributions import spd_cumulative_distributions
    from meg_tokens.behavior.analyses.evidence import (
        evidence_at_decision_responses, criterion_decline_statistics,
        reverse_correlation, reverse_correlation_statistics,
        conditional_accuracy_functions, conditional_accuracy_statistics,
        continuous_evidence_effects, continuous_evidence_statistics,
    )
    from meg_tokens.behavior.analyses.sequential import (
        robust_post_error_slowing, post_error_statistics,
        choice_history, choice_history_statistics,
    )
    from meg_tokens.behavior.analyses.summary import summarize_behavior

    cells = condition_class_cells(features)
    choiceside = choice_side_summary(features)
    timeontask = time_on_task(features)
    conditionorder = condition_order_effects(features)
    lapses = lapse_summary(features)
    extremedt, extremedttrials = extreme_decision_times(features)
    criteriondecline = evidence_at_decision_responses(features, predictor="decision_token_index")
    urgency = evidence_at_decision_responses(features, predictor="dt_ms")
    kernels = reverse_correlation(features)
    caf = conditional_accuracy_functions(features)
    continuous = continuous_evidence_effects(features)
    posterror = robust_post_error_slowing(features)
    history = choice_history(features)
    summary = summarize_behavior(features)

    return {
        "conditionclass": cells,
        "conditionclassstats": condition_class_statistics(cells),
        "choiceside": choiceside,
        "choicesidestats": choice_side_statistics(choiceside),
        "timeontask": timeontask,
        "timeontaskstats": time_on_task_statistics(timeontask),
        "conditionorder": conditionorder,
        "conditionorderstats": condition_order_statistics(conditionorder),
        "lapses": lapses,
        "lapsestats": lapse_statistics(lapses),
        "extremedt": extremedt,
        "extremedttrials": extremedttrials,
        "spdcumulative": spd_cumulative_distributions(features),
        "criteriondecline": criteriondecline,
        "criteriondeclinestats": criterion_decline_statistics(criteriondecline),
        "urgency": urgency,
        "urgencystats": criterion_decline_statistics(urgency),
        "reversecorrelation": kernels,
        "reversecorrelationstats": reverse_correlation_statistics(kernels),
        "conditionalaccuracy": caf,
        "conditionalaccuracystats": conditional_accuracy_statistics(caf),
        "continuousevidence": continuous,
        "continuousevidencestats": continuous_evidence_statistics(continuous),
        "posterror": posterror,
        "posterrorstats": post_error_statistics(posterror),
        "choicehistory": history,
        "choicehistorystats": choice_history_statistics(history),
        "summary": summary,
    }


def individual_and_species_derivatives(
    summary: pd.DataFrame, criteriondecline: pd.DataFrame, urgency: pd.DataFrame,
    continuousevidence: pd.DataFrame, lapses: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    from meg_tokens.behavior.analyses.individual import (
        individual_profile, individual_correlations,
    )

    profile = individual_profile(
        summary, urgency=urgency, criterion=criteriondecline,
        evidence=continuousevidence, lapses=lapses,
    )
    return {
        "individualprofile": profile,
        "individualcorrelations": individual_correlations(profile),
    }


def speciescomparison(subjects: tuple[str, ...] = SUBJECTS) -> pd.DataFrame:
    """Minimal speciescomparison rows -- the 10 measures F25 facets over."""
    from meg_tokens.behavior.math.inference import one_sample_statistics

    rng = np.random.default_rng(11)
    measures = {
        "decision_time_easy_ms": 900.0, "decision_time_ambiguous_ms": 1350.0,
        "decision_time_misleading_ms": 1300.0,
        "success_probability_at_decision_easy": 0.78, "success_probability_at_decision_ambiguous": 0.65,
        "success_probability_at_decision_misleading": 0.62,
        "criterion_slope_log_odds_per_token": 0.1,
        "urgency_minus_integrator_bic": -230.0,
        "urgency_scale_criterion_seconds": 1.5, "urgency_scale_fast_minus_slow": -0.1,
    }
    rows = []
    for measure, center in measures.items():
        values = center + rng.normal(0, abs(center) * 0.1 + 0.01, len(subjects))
        rows.append({"analysis": "cross_species_comparison", "measure": measure, **one_sample_statistics(values)})
    return pd.DataFrame(rows)


def ssmtimecourse(subjects: tuple[str, ...] = SUBJECTS) -> pd.DataFrame:
    """Duplicated-across-rows exactly like the real derivative (criterion
    repeats over trial_class; observed_density_* repeats over model) so the
    de-duplication trap is exercised, not just tolerated."""
    rows = []
    times = np.linspace(0, 3, 10)
    for subject in subjects:
        for condition in ("all", "fast", "slow"):
            for model in ("ddm", "urgency"):
                for trial_class in ("easy", "ambiguous", "misleading"):
                    for time_s in times:
                        criterion = 0.5 if model == "ddm" else 0.5 / (time_s + 0.1)
                        rows.append({
                            "subject": subject, "condition": condition, "model": model,
                            "trial_class": trial_class, "n_trials": 20, "time_s": float(time_s),
                            "criterion": float(criterion),
                            "mean_decision_variable": float(0.3 * np.sin(time_s) if model == "urgency" else 0.2 * time_s),
                            "predicted_density_correct": float(np.exp(-time_s)), "predicted_density_error": float(0.3 * np.exp(-time_s)),
                            "observed_density_correct": float(np.exp(-time_s) * 0.9), "observed_density_error": float(0.25 * np.exp(-time_s)),
                        })
    return pd.DataFrame(rows)


def ssmpopulation(subjects: tuple[str, ...] = SUBJECTS) -> tuple[pd.DataFrame, pd.DataFrame]:
    fits = ssmcomparison(subjects)
    from meg_tokens.behavior.analyses.sequential_sampling import population_parameters

    return population_parameters(fits)
