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
    """All four candidate models x three conditions (all, fast, slow) per subject.

    The production derivative has carried `collapsing` and `additive_urgency`
    since the mechanistic fit landed; a two-model fixture let figures silently
    assume the comparison was urgency-versus-baseline. The advantages are
    ordered as they are in the real fit -- urgency largest, collapsing real but
    smaller, additive urgency straddling zero -- so a figure that mis-selects a
    model produces a visibly wrong picture rather than a plausible one.
    """
    rows = []
    for index, subject in enumerate(subjects):
        for condition, bic_ddm in (("all", 900.0), ("fast", 420.0), ("slow", 480.0)):
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
            advantages = {
                "urgency": -220.0 - 10.0 * index,
                "collapsing": -150.0 - 6.0 * index,
                # Straddles zero across subjects, as the real one does.
                "additive_urgency": -20.0 + 4.0 * (index % 9),
            }
            for model, advantage in advantages.items():
                bic = bic_ddm + advantage
                # Only the parameters MODEL_PARAMETERS names for this model are
                # finite; the population model read every one of them and
                # divides by its standard error, so a missing column is a
                # KeyError rather than a silently empty cell.
                rows.append({
                    "subject": subject, "condition": condition, "model": model,
                    "n_trials": 300, "n_token_sequences": 76, "n_parameters": 4,
                    "log_likelihood": -bic / 2, "t_dur_s": 3.0,
                    "aic": bic - 5, "bic": bic,
                    "delta_aic": advantage, "delta_bic": advantage, "converged": True,
                    "drift_scale": 0.2 + 0.01 * index,
                    "bound": np.nan if model == "urgency" else 0.9 + 0.02 * index,
                    "nondecision_s": 0.01,
                    "urgency_scale": urgency_scale if model == "urgency" else np.nan,
                    "urgency_onset_s": 0.05 + 0.01 * index if model == "urgency" else np.nan,
                    "collapse_rate": 0.4 + 0.01 * index if model == "collapsing" else np.nan,
                    "additive_scale": (
                        0.6 + 0.01 * index if model == "additive_urgency" else np.nan
                    ),
                    "drift_scale_se": 0.03,
                    "bound_se": np.nan if model == "urgency" else 0.07,
                    "nondecision_s_se": 0.01,
                    "urgency_scale_se": 0.1, "urgency_onset_s_se": 0.02,
                    "collapse_rate_se": 0.05, "additive_scale_se": 0.06,
                })
    return pd.DataFrame(rows)


def ssmboundaryaudit(fits: pd.DataFrame) -> pd.DataFrame:
    """Mirrors what the evaluation aggregation recomputes from the fits."""
    from meg_tokens.behavior.analyses.sequential_sampling import (
        boundary_convergence_audit,
    )

    return boundary_convergence_audit(fits)


def ssmheldout(fits: pd.DataFrame, folds: int = 3) -> pd.DataFrame:
    """Three-fold held-out scores, ordered to match the fits' model ranking."""
    per_trial = {
        "ddm": -0.90, "urgency": -0.62, "collapsing": -0.72, "additive_urgency": -0.89,
    }
    rows = []
    for index, (_, fit) in enumerate(fits.iterrows()):
        for fold in range(folds):
            rows.append({
                "subject": fit["subject"], "condition": fit["condition"],
                "model": fit["model"], "fold": fold,
                "n_train": 200, "n_test": 100,
                "heldout_log_likelihood_per_trial": (
                    per_trial[fit["model"]] + 0.004 * (index % 7) - 0.002 * fold
                ),
                "predicted_accuracy": 0.7, "converged": True,
            })
    return pd.DataFrame(rows)


def ssmcomparisonstats(fits: pd.DataFrame) -> pd.DataFrame:
    """Mirrors what aggregate_mechanistic_fits writes -- mechanistic_model_statistics
    + model_comparison_statistics + urgency_condition_contrast, in that order --
    computed directly from the fixture fits so the stats rows are internally
    consistent with the fits table.

    The first family is the one F13 reads: it is the only one carrying `model`
    and `n_subjects_favoring_candidate`, and omitting it here let a figure that
    recounted the subjects from the fits pass."""
    from meg_tokens.behavior.analyses.sequential_sampling import (
        mechanistic_model_statistics,
        model_comparison_statistics,
        urgency_condition_contrast,
    )

    return pd.concat(
        [
            mechanistic_model_statistics(fits),
            model_comparison_statistics(fits),
            urgency_condition_contrast(fits),
        ],
        ignore_index=True,
    )


def dtdistribution(subjects: tuple[str, ...] = SUBJECTS) -> pd.DataFrame:
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

# The four success probabilities a 3-of-15-token split can produce, and which
# of them each trial class can reach. These are the real design's values
# (0-3, 1-2, 2-1, 3-0 leads), so a figure that resolves early evidence by level
# sees the same structure here as in the data.
_EARLY_SP_LEVELS = (0.193848, 0.387207, 0.612793, 0.806152)
_EARLY_SP_BY_CLASS = {
    "easy": _EARLY_SP_LEVELS[2:],
    "ambiguous": _EARLY_SP_LEVELS[1:3],
    "misleading": _EARLY_SP_LEVELS[:2],
    "unclassified": _EARLY_SP_LEVELS[2:],
}


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
                    # Unclassified is 43% of real task trials and the largest
                    # occupant of every decision-time bin in F10, so the
                    # fixture carries it rather than a fully classified cohort
                    # no figure ever sees.
                    trial_class_code = int(rng.integers(0, 4))
                    class_name = {
                        0: "unclassified", 1: "easy", 2: "ambiguous", 3: "misleading",
                    }[trial_class_code]
                    raw_label = {0: "x", 1: "e", 2: "a", 3: "m"}[trial_class_code]
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
                    # Early evidence is a four-level factor in the real design,
                    # not a continuum: three tokens admit exactly four splits,
                    # and which of them a trial can take is constrained by its
                    # class (no random sequence leaves the correct target
                    # trailing; every negative level is a misleading trial).
                    # A continuous draw here would let F11 pass against a
                    # predictor the task cannot produce.
                    early_sp = float(rng.choice(_EARLY_SP_BY_CLASS[class_name]))
                    signed_evidence = float(np.log(early_sp / (1.0 - early_sp)))
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
                            sp_design_early=[early_sp],
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
        condition_class_contrasts,
        choice_side_summary, choice_side_statistics,
        time_on_task, time_on_task_statistics,
        within_block_decile_profile, within_block_decile_statistics,
        condition_order_effects, condition_order_statistics,
        lapse_summary, extreme_decision_times,
    )
    from meg_tokens.behavior.analyses.distributions import (
        decision_time_geometry, decision_time_geometry_statistics,
        spd_cumulative_distributions,
    )
    from meg_tokens.behavior.analyses.evidence import (
        evidence_at_decision_responses, first_order_criterion_decline,
        criterion_scale_comparison, criterion_scale_comparison_statistics,
        criterion_time_profile,
        reverse_correlation, reverse_correlation_statistics,
        conditional_accuracy_functions, conditional_accuracy_statistics,
        continuous_evidence_effects, continuous_evidence_statistics,
        early_evidence_order_profile, early_evidence_order_statistics,
        early_evidence_profile,
    )
    from meg_tokens.behavior.analyses.sequential import (
        robust_post_error_slowing, post_error_statistics,
        choice_history, choice_history_statistics,
    )
    from meg_tokens.behavior.analyses.summary import summarize_behavior

    cells = condition_class_cells(features)
    choiceside = choice_side_summary(features)
    timeontask = time_on_task(features)
    decile_profile = within_block_decile_profile(features)
    conditionorder = condition_order_effects(features)
    lapses = lapse_summary(features)
    extremedt, extremedttrials = extreme_decision_times(features)
    criteriondecline = first_order_criterion_decline(features)
    urgency = evidence_at_decision_responses(features, predictor="dt_ms")
    kernels = reverse_correlation(features)
    caf = conditional_accuracy_functions(features)
    continuous = continuous_evidence_effects(features)
    evidence_profile = early_evidence_profile(features)
    evidence_order = early_evidence_order_profile(features)
    posterror = robust_post_error_slowing(features)
    history = choice_history(features)
    summary = summarize_behavior(features)

    return {
        "conditionclass": cells,
        "conditionclassstats": condition_class_statistics(cells),
        "conditionclasscontrasts": condition_class_contrasts(cells),
        "choiceside": choiceside,
        "choicesidestats": choice_side_statistics(choiceside),
        "timeontask": timeontask,
        "timeontaskstats": time_on_task_statistics(timeontask),
        "withinblockdecile": decile_profile,
        "withinblockdecilestats": within_block_decile_statistics(decile_profile),
        "conditionorder": conditionorder,
        "conditionorderstats": condition_order_statistics(conditionorder),
        "lapses": lapses,
        "extremedt": extremedt,
        "extremedttrials": extremedttrials,
        "spdcumulative": spd_cumulative_distributions(features),
        "dtgeometry": decision_time_geometry(features),
        "dtgeometrystats": decision_time_geometry_statistics(decision_time_geometry(features)),
        "criteriondecline": criteriondecline,
        "criterionscale": criterion_scale_comparison(features),
        "criterionscaleprofile": criterion_time_profile(features),
        "criterionscalestats": criterion_scale_comparison_statistics(
            criterion_scale_comparison(features), criterion_time_profile(features)
        ),
        "urgency": urgency,
        "reversecorrelation": kernels,
        "reversecorrelationstats": reverse_correlation_statistics(kernels),
        "conditionalaccuracy": caf,
        "conditionalaccuracystats": conditional_accuracy_statistics(caf),
        "continuousevidence": continuous,
        "continuousevidenceprofile": evidence_profile,
        "continuousevidenceorder": evidence_order,
        "continuousevidencestats": pd.concat(
            [
                continuous_evidence_statistics(continuous, evidence_profile),
                early_evidence_order_statistics(evidence_order),
            ],
            ignore_index=True,
        ),
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
    """Minimal speciescomparison rows -- the 10 comparison-ready measures.

    No figure draws these any more (the forest was retired); the derivative
    is still written by characterization and mapped to the published
    literature in docs/behavior.md, "Cross-species translation".
    """
    from meg_tokens.behavior.math.inference import one_sample_statistics

    rng = np.random.default_rng(11)
    measures = {
        "decision_time_easy_ms": 900.0, "decision_time_ambiguous_ms": 1350.0,
        "decision_time_misleading_ms": 1300.0,
        "success_probability_at_decision_easy": 0.78, "success_probability_at_decision_ambiguous": 0.65,
        "success_probability_at_decision_misleading": 0.62,
        "criterion_slope_sum_log_lr_per_second": 0.1,
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
    de-duplication trap is exercised, not just tolerated.

    ``trial_class`` carries its own pooled ``all`` level, as the real
    derivative does. Without it the density panel -- which selects exactly
    those rows -- draws nothing, and a test asserting on that panel passes
    against an empty axes.
    """
    rows = []
    times = np.linspace(0, 3, 10)
    for subject in subjects:
        for condition in ("all", "fast", "slow"):
            for model in ("ddm", "urgency"):
                for trial_class in ("all", "easy", "ambiguous", "misleading"):
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


def ssmdistributioncheckstats(subjects: tuple[str, ...] = SUBJECTS) -> pd.DataFrame:
    """Observed-minus-predicted RT quantiles, through the real statistics function.

    Deviations are ordered as they are in the real fit so a figure that maps a
    model to the wrong series produces a visibly wrong picture: the fixed bound
    misses badly at the 90th correct percentile and on fast errors, urgency
    sits near zero everywhere, and the other two fall between. Building this
    through `fitted_distribution_check_statistics` rather than hand-writing its
    output means the fixture cannot drift from the function the figure reads.
    """
    from meg_tokens.behavior.analyses.sequential_sampling import (
        fitted_distribution_check_statistics,
    )

    # (correct q10, q50, q90, error q10, q50, q90) residuals per model.
    deviations = {
        "ddm": (0.11, 0.04, -0.38, 0.37, 0.34, -0.16),
        "urgency": (-0.02, 0.00, 0.05, 0.09, 0.05, 0.01),
        "collapsing": (0.01, 0.04, -0.06, 0.17, 0.18, -0.09),
        "additive_urgency": (0.15, 0.01, -0.11, 0.50, 0.29, 0.01),
    }
    quantiles = (0.1, 0.5, 0.9)
    rows = []
    for index, subject in enumerate(subjects):
        jitter = 0.01 * ((index % 5) - 2)
        for model, residuals in deviations.items():
            for position, (outcome, quantile) in enumerate(
                [("correct", q) for q in quantiles] + [("error", q) for q in quantiles]
            ):
                observed = 0.8 + 0.4 * quantile
                rows.append({
                    "subject": subject, "condition": "all", "model": model,
                    "trial_class": "all", "outcome": outcome,
                    "quantile": quantile, "n_observed": 40,
                    "observed_rt_s": observed,
                    "predicted_rt_s": observed - residuals[position] - jitter,
                })
    return fitted_distribution_check_statistics(pd.DataFrame(rows))


def ssmrobustnessstats(subjects: tuple[str, ...] = SUBJECTS) -> pd.DataFrame:
    """The six-configuration robustness grid, through the real statistics function.

    The Fast-minus-Slow `urgency_scale` contrast is negative in every
    configuration and **largest** in `expanded_bounds`, which is also the only
    one with no cell pinned at the upper bound -- the ordering that carries the
    censoring defence. A figure that mapped configurations to the wrong rows
    would show the defence backwards rather than merely wrong.
    """
    from meg_tokens.behavior.analyses.sequential_sampling import robustness_statistics

    configurations = {
        "baseline": (-0.118, 2.0),
        "expanded_bounds": (-0.181, 4.0),
        "post_horizon_evidence_zero": (-0.115, 2.0),
        "solver_20ms": (-0.112, 2.0),
        "tau_100ms": (-0.101, 2.0),
        "tau_300ms": (-0.114, 2.0),
    }
    rows = []
    for index, subject in enumerate(subjects):
        spread = 0.05 * ((index % 7) - 3)
        for configuration, (contrast, upper) in configurations.items():
            for condition in ("all", "fast", "slow"):
                for model in ("ddm", "urgency"):
                    # Fast sits below Slow by `contrast`; only expanded_bounds
                    # leaves room above 2.0, so only it escapes the ceiling.
                    scale = 1.5 + spread + (contrast if condition == "fast" else 0.0)
                    if condition == "slow" and configuration != "expanded_bounds":
                        scale = min(scale + 0.5, upper)
                    rows.append({
                        "subject": subject, "configuration": configuration,
                        "condition": condition, "model": model,
                        "delta_bic": -200.0 + spread if model == "urgency" else 0.0,
                        "converged": True, "boundary_hit": False,
                        "boundary_parameters": "",
                        "urgency_scale": scale if model == "urgency" else np.nan,
                        "urgency_scale_lower_bound": 0.01,
                        "urgency_scale_upper_bound": upper,
                        "urgency_onset_s": 1.0 if model == "urgency" else np.nan,
                        "urgency_onset_s_lower_bound": 0.0,
                        "urgency_onset_s_upper_bound": upper,
                        "nondecision_s": 0.0,
                        "nondecision_s_lower_bound": 0.0,
                        "nondecision_s_upper_bound": 1.0,
                        "drift_scale": 0.2,
                        "drift_scale_lower_bound": 0.0,
                        "drift_scale_upper_bound": 4.0,
                    })
    return robustness_statistics(pd.DataFrame(rows))


def ssmrecovery():
    """Synthetic recovery tables: per-case estimates, their summary, confusion.

    Shaped like the real derivatives rather than minimal: the figure reads a
    spread-versus-solver-step ratio, so at least one parameter must span many
    solver steps and one must span less than a single step, or the panel that
    explains the non-decision failures has nothing to explain.
    """
    from meg_tokens.behavior.analyses.sequential_sampling import (
        MECHANISTIC_MODELS,
        MODEL_PARAMETERS,
        SOLVER_STEP_S,
    )

    cases = []
    rng = np.random.default_rng(11)
    for model in MECHANISTIC_MODELS:
        for parameter in MODEL_PARAMETERS[model]:
            # Non-decision spans a fraction of one solver step; everything else
            # spans many, which is the contrast panel D is built on.
            wide = parameter != "nondecision_s"
            spread = 1.2 if wide else 0.6 * SOLVER_STEP_S
            truths = np.linspace(0.1, 0.1 + spread, 12)
            noise = rng.normal(0.0, 0.02 * spread if wide else spread, size=12)
            for repetition, (truth, error) in enumerate(zip(truths, noise)):
                cases.append({
                    "analysis": "ssm_parameter_recovery",
                    "repetition": repetition,
                    "true_model": model,
                    "parameter": parameter,
                    "true_value": float(truth),
                    "estimated_value": float(truth + error),
                    "n_starts": 3,
                    "n_per_sequence": 30,
                    "truth_source": '{"truth_region": "accepted_fit"}',
                })
    parameter = pd.DataFrame(cases)

    summary = []
    for (model, name), group in parameter.groupby(["true_model", "parameter"]):
        r = float(np.corrcoef(group["true_value"], group["estimated_value"])[0, 1])
        summary.append({
            "analysis": "ssm_parameter_recovery_summary",
            "true_model": model,
            "parameter": name,
            "n_recovery_cases": len(group),
            "n_finite_estimates": len(group),
            "bias": 0.0,
            "rmse": 0.1,
            "correlation": r,
            "correlation_ci95_low": max(-1.0, r - 0.1),
            "correlation_ci95_high": min(1.0, r + 0.1),
            "convergence_rate": 1.0,
            "boundary_rate": 0.5,
        })

    confusion = [
        {
            "analysis": "ssm_model_recovery_confusion",
            "true_model": true_model,
            "selected_model": selected,
            "n_repetitions": 12,
            "n_selected": 12 if selected == true_model else 0,
            "selection_rate": 1.0 if selected == true_model else 0.0,
        }
        for true_model in MECHANISTIC_MODELS
        for selected in MECHANISTIC_MODELS
    ]
    return parameter, pd.DataFrame(summary), pd.DataFrame(confusion)
