"""Run the Stage 2b behavioral characterization analysis pipeline.

``analyze_behavior`` produces the preprint-comparison summaries. This workflow
consumes its trial-feature table and adds distributional descriptions, design
and session effects, continuous evidence, criterion decline, urgency, temporal
weighting, conditional accuracy, trial-history effects, sequential-sampling
model fits, individual differences, and cross-species comparison.

Each analysis writes its own subject-level table and, where it has one, a
matching group-statistics table, so that a single failed model never truncates
the rest of the output.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
from importlib.metadata import PackageNotFoundError, version
import subprocess
import sys
from typing import Optional, Sequence

import pandas as pd

from meg_tokens.behavior.analyses.design_effects import (
    choice_side_statistics,
    choice_side_summary,
    condition_class_cells,
    condition_class_statistics,
    condition_order_effects,
    condition_order_statistics,
    extreme_decision_times,
    lapse_statistics,
    lapse_summary,
    time_on_task,
    time_on_task_statistics,
)
from meg_tokens.behavior.analyses.distributions import (
    decision_time_distribution_statistics,
    decision_time_distributions,
    spd_cumulative_distributions,
)
from meg_tokens.behavior.analyses.evidence import (
    conditional_accuracy_functions,
    conditional_accuracy_statistics,
    continuous_evidence_effects,
    continuous_evidence_statistics,
    criterion_decline_statistics,
    evidence_at_decision_responses,
    first_order_criterion_decline,
    reverse_correlation,
    reverse_correlation_statistics,
)
from meg_tokens.behavior.analyses.individual import (
    comparison_statistics,
    individual_correlations,
    individual_profile,
)
from meg_tokens.behavior.analyses.sequential import (
    choice_history,
    choice_history_statistics,
    post_error_statistics,
    robust_post_error_slowing,
)
from meg_tokens.behavior.analyses.sequential_sampling import (
    EVIDENCE_AFTER_LAST,
    FILTER_TAU_S,
    FIT_SEED,
    HESSIAN_STEP,
    MINIMUM_FIT_TRIALS,
    MIXTURE_COEF,
    MODEL_PARAMETERS,
    PARAMETER_RANGES,
    SOLVER_STEP_S,
    TIME_COURSE_STEP_S,
    TOKEN_INTERVAL_S,
    fit_sequential_sampling_models,
    fitted_predictions,
    model_comparison_statistics,
    population_parameters,
    urgency_condition_contrast,
)
from meg_tokens.behavior.tables import read_trial_features
from meg_tokens.core import ProjectConfig, WorkflowResult, normalize_subject_id
from meg_tokens.io import DerivativeLayout, load_table, require_file, save_table


def _ssm_source_tree_sha256() -> str:
    """Hash source/scripts/docs because cluster analysis is uncommitted."""
    root = Path(__file__).resolve().parents[2]
    paths = [
        path
        for base in (root / "meg_tokens", root / "scripts")
        if base.exists()
        for path in base.rglob("*")
        if path.is_file() and path.suffix in {".py", ".sh"}
    ]
    paths.extend(
        path for path in (
            root / "docs" / "behavior.md",
            root / "docs" / "behavior_reporting_session_handoff.md",
        ) if path.is_file()
    )
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unavailable"


def _ssm_fit_metadata(
    *,
    models: tuple[str, ...],
    n_jobs: int,
    n_starts: int,
    features_path: Path,
    excluded: set[str],
) -> dict[str, object]:
    """Return complete reproducibility metadata for subject-level SSM fits."""
    return {
        "stage": "behavior_characterization_analysis",
        "analysis": "ssm mechanistic model fit",
        "models": list(models),
        "model_parameters": {model: list(names) for model, names in MODEL_PARAMETERS.items()},
        "parameter_ranges": {name: list(bounds) for name, bounds in PARAMETER_RANGES.items()},
        "condition_parameterization": "separate subject × condition × model cells",
        "input_files": [str(features_path)],
        "excluded_subjects": sorted(excluded),
        "timing": {
            "token_interval_s": TOKEN_INTERVAL_S,
            "decision_time_origin": "first token jump; dt_ms is motor-baseline-corrected",
            "evidence_after_last_token": EVIDENCE_AFTER_LAST,
        },
        "filter_tau_s": FILTER_TAU_S,
        "solver": {
            "dt_s": SOLVER_STEP_S,
            "state_grid_step": SOLVER_STEP_S,
            "state_grid_units": "decision-variable/noise units (not seconds)",
            "timecourse_dt_s": TIME_COURSE_STEP_S,
            "boundary": "absorbing finite-grid first passage",
            "overshoot_correction": "none; numerical grid only",
        },
        "likelihood": {
            "loss": "PyDDM LossLikelihood",
            "mixture_coef": MIXTURE_COEF,
            "mixture_distribution": "uniform contaminant",
        },
        "minimum_fit_trials": MINIMUM_FIT_TRIALS,
        "fit_seed": FIT_SEED,
        "n_jobs": n_jobs,
        "n_starts": n_starts,
        "uncertainty": {
            "final_full_data": "finite-difference observed-information Hessian",
            "hessian_step": HESSIAN_STEP,
            "outside_final_full_data": "disabled for held-out/recovery/robustness",
        },
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip(),
        "git_dirty": bool(subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
        ).stdout.strip()),
        "source_tree_sha256": _ssm_source_tree_sha256(),
        "python": sys.version,
        "software": {
            name: _package_version(name)
            for name in ("pyddm", "numpy", "pandas", "scipy")
        },
    }


def _selected_features(
    project: ProjectConfig,
    layout: DerivativeLayout,
    subjects: Optional[Sequence[str]],
) -> tuple[pd.DataFrame, Path, set[str]]:
    """Load the trial-feature table and apply exclusions and subject selection.

    Parameters
    ----------
    project
        Project configuration supplying the exclusion list.
    layout
        Derivative layout for the project's BIDS root.
    subjects
        Optional subject selection; every retained subject when omitted.

    Returns
    -------
    features
        The selected rows of the canonical trial-feature table.
    features_path
        Path the table was read from, for provenance.
    excluded
        Subjects removed by the project's exclusion list.

    Raises
    ------
    FileNotFoundError
        If Stage 2 has not been run.
    ValueError
        If no rows remain after selection.
    """
    features_path = layout.behavior_trial_features()
    if not features_path.is_file():
        raise FileNotFoundError(
            "The trial-feature table is required by the behavioral characterization "
            f"analyses but does not exist: {features_path}. Run "
            "'meg-tokens behavior analyze' first."
        )
    features = read_trial_features(features_path)
    excluded = set(project.subject_exclusions)
    if excluded:
        features = features.loc[~features["subject"].isin(excluded)]
    if subjects:
        selected = {normalize_subject_id(subject) for subject in subjects}
        features = features.loc[features["subject"].isin(selected)]
    if features.empty:
        raise ValueError(
            "No trial-feature rows remain after applying subject selection "
            "and exclusions"
        )
    return features, features_path, excluded


def characterization_subjects(project: ProjectConfig) -> list[str]:
    """List the subjects Stage 2b will analyze, in a stable order.

    Parameters
    ----------
    project
        Project configuration supplying the exclusion list.

    Returns
    -------
    list of str
        Sorted subject identifiers present in the trial-feature table and not
        excluded.

    Notes
    -----
    An array job needs this list before it can map a task index to a batch of
    subjects, and it must be the same list the aggregation step will expect.
    """
    layout = DerivativeLayout(project.bids_root, task=project.task)
    features, _, _ = _selected_features(project, layout, None)
    return sorted(features["subject"].astype(str).unique())


def fit_subject_sequential_sampling(
    project: ProjectConfig,
    *,
    subjects: Optional[Sequence[str]] = None,
    n_jobs: int = 1,
    models: tuple[str, ...] = ("ddm", "urgency"),
    n_starts: int = 1,
) -> WorkflowResult:
    """Fit the sequential-sampling models and write one table per subject.

    The fits are the whole runtime of Stage 2b and are independent across
    subjects, so this splits them out of
    :func:`analyze_behavior_characterization`: run it once per subject, or once
    per batch of subjects, then run the characterization, which pools the
    written tables instead of refitting.

    Parameters
    ----------
    project
        Project configuration.
    subjects
        Subjects to fit. Every retained subject when omitted.
    n_jobs
        Worker processes. Each subject contributes six independent fits (three
        conditions by two models), so a batch of subjects is what fills a node
        with more cores than that.

    Returns
    -------
    WorkflowResult
        Outputs are the per-subject fit tables, one per fitted subject.

    Notes
    -----
    Every subject's table holds that subject's rows only, in the schema
    :func:`~meg_tokens.behavior.analyses.sequential_sampling.fit_sequential_sampling_models`
    returns, so pooling them is a concatenation. Writing per subject rather
    than per batch keeps an array job's tasks from racing for one path.
    """
    layout = DerivativeLayout(project.bids_root, task=project.task)
    features, features_path, excluded = _selected_features(project, layout, subjects)
    fits = fit_sequential_sampling_models(
        features, n_jobs=n_jobs, models=models, n_starts=n_starts
    )
    time_courses, trial_predictions = fitted_predictions(features, fits)

    written = {
        "ssmcomparison": fits,
        "ssmtimecourse": time_courses,
        "ssmtrialpredictions": trial_predictions,
    }
    provenance = _ssm_fit_metadata(
        models=models,
        n_jobs=n_jobs,
        n_starts=n_starts,
        features_path=features_path,
        excluded=excluded,
    )
    outputs = []
    for subject in sorted(fits["subject"].astype(str).unique()):
        for name, table in written.items():
            # A cell too small to fit produces no prediction rows at all, and an
            # empty table is not worth a file.
            if table.empty:
                continue
            rows = table.loc[table["subject"].astype(str) == subject]
            output_path = layout.behavior_subject_analysis(subject, name)
            save_table(
                output_path,
                rows.reset_index(drop=True),
                metadata={
                    **provenance,
                    "analysis": name,
                    "subjects": [subject],
                    "inference": (
                        "maximum-likelihood fit per subject, condition and "
                        "model; pooled by dedicated Thura et al. aggregation"
                    ),
                },
            )
            outputs.append(output_path)
    return WorkflowResult(
        stage="behavior_sequential_sampling_fit",
        inputs=(features_path,),
        outputs=tuple(outputs),
        settings={
            "subjects": sorted(fits["subject"].astype(str).unique()),
            "n_jobs": n_jobs,
            "models": list(models),
            "n_starts": n_starts,
            "n_converged_fits": int(fits["converged"].sum()),
        },
    )


SUBJECT_LEVEL_ANALYSES: tuple[str, ...] = (
    "ssmcomparison",
    "ssmtimecourse",
    "ssmtrialpredictions",
)


def _pooled_sequential_sampling_fits(
    layout: DerivativeLayout,
    features: pd.DataFrame,
    name: str = "ssmcomparison",
) -> tuple[pd.DataFrame, list[str]]:
    """Pool whichever per-subject sequential-sampling fits have been written.

    Parameters
    ----------
    layout
        Derivative layout for the project's BIDS root.
    features
        Selected trial-feature rows, naming the subjects to look for.

    Returns
    -------
    fits
        Concatenated fit table, empty when no subject has been fitted.
    missing
        Selected subjects with no fit table yet.

    Notes
    -----
    Missing subjects are reported rather than fitted. The fits are the runtime
    of Stage 2b and are run separately by
    :func:`fit_subject_sequential_sampling`; the characterization only pools
    them, so a partly finished array job still yields every other derivative.
    """
    subjects = sorted(features["subject"].astype(str).unique())
    tables = []
    missing = []
    for subject in subjects:
        path = layout.behavior_subject_analysis(subject, name)
        if path.is_file():
            tables.append(load_table(path))
        else:
            missing.append(subject)
    fits = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    return fits, missing


def analyze_behavior_characterization(
    project: ProjectConfig,
    *,
    subjects: Optional[Sequence[str]] = None,
    neural_metrics_path: Optional[str | Path] = None,
) -> WorkflowResult:
    """Run the Stage 2b behavioral characterization pipeline over staged trial features.

    This is also the aggregation step for the sequential-sampling models:
    it fits nothing itself, and pools whatever per-subject tables
    :func:`fit_subject_sequential_sampling` has written. Subjects without one
    are named in the result rather than fitted, and the ``ssm*`` derivatives
    are omitted entirely when none has been.
    """
    layout = DerivativeLayout(
        project.bids_root,
        task=project.task,
    )
    features_path = layout.behavior_trial_features()
    summary_path = layout.behavior_summary()
    if not features_path.is_file():
        raise FileNotFoundError(
            "The trial-feature table is required by the behavioral characterization "
            f"analyses but does not exist: {features_path}. Run "
            "'meg-tokens behavior analyze' first."
        )
    features = read_trial_features(features_path)
    summary = load_table(
        require_file(summary_path, purpose="subject behavior summary")
    )

    excluded = set(project.subject_exclusions)
    if excluded:
        features = features.loc[~features["subject"].isin(excluded)]
        summary = summary.loc[~summary["subject"].isin(excluded)]
    if subjects:
        selected = {normalize_subject_id(subject) for subject in subjects}
        features = features.loc[features["subject"].isin(selected)]
        summary = summary.loc[summary["subject"].isin(selected)]
    if features.empty:
        raise ValueError(
            "No trial-feature rows remain after applying subject selection "
            "and exclusions"
        )

    neural_metrics = None
    if neural_metrics_path is not None:
        neural_path = Path(neural_metrics_path)
        separator = "\t" if neural_path.suffix == ".tsv" else ","
        neural_metrics = pd.read_csv(neural_path, sep=separator)

    dt_distributions = decision_time_distributions(features)
    cells = condition_class_cells(features)
    sides = choice_side_summary(features)
    drift = time_on_task(features)
    order = condition_order_effects(features)
    lapses = lapse_summary(features)
    extreme_counts, extreme_flagged = extreme_decision_times(features)
    criterion = first_order_criterion_decline(features)
    urgency = evidence_at_decision_responses(features, predictor="dt_ms")
    kernels = reverse_correlation(features)
    accuracy_functions = conditional_accuracy_functions(features)
    evidence = continuous_evidence_effects(features)
    ssm_pooled = {
        name: _pooled_sequential_sampling_fits(layout, features, name)[0]
        for name in SUBJECT_LEVEL_ANALYSES
    }
    ssm_fits, ssm_missing = _pooled_sequential_sampling_fits(layout, features)
    post_error = robust_post_error_slowing(features)
    history = choice_history(features)
    profile = individual_profile(
        summary,
        urgency=urgency,
        criterion=criterion,
        evidence=evidence,
        lapses=lapses,
        neural_metrics=neural_metrics,
    )

    tables: dict[str, pd.DataFrame] = {
        "spdcumulative": spd_cumulative_distributions(features),
        "dtdistribution": dt_distributions,
        "dtdistributionstats": decision_time_distribution_statistics(dt_distributions),
        "conditionclass": cells,
        "conditionclassstats": condition_class_statistics(cells),
        "choiceside": sides,
        "choicesidestats": choice_side_statistics(sides),
        "timeontask": drift,
        "timeontaskstats": time_on_task_statistics(drift),
        "conditionorder": order,
        "conditionorderstats": condition_order_statistics(order),
        "lapses": lapses,
        "lapsestats": lapse_statistics(lapses),
        "extremedt": extreme_counts,
        "extremedttrials": extreme_flagged,
        "criteriondecline": criterion,
        "criteriondeclinestats": criterion_decline_statistics(criterion),
        "urgency": urgency,
        "urgencystats": criterion_decline_statistics(urgency),
        "reversecorrelation": kernels,
        "reversecorrelationstats": reverse_correlation_statistics(kernels),
        "conditionalaccuracy": accuracy_functions,
        "conditionalaccuracystats": conditional_accuracy_statistics(accuracy_functions),
        "continuousevidence": evidence,
        "continuousevidencestats": continuous_evidence_statistics(evidence),
        "posterror": post_error,
        "posterrorstats": post_error_statistics(post_error),
        "choicehistory": history,
        "choicehistorystats": choice_history_statistics(history),
        "individualprofile": profile,
        "individualcorrelations": individual_correlations(profile),
        "speciescomparison": comparison_statistics(
            summary, criterion=criterion, sequential_sampling=ssm_fits
        ),
    }
    if len(ssm_fits):
        ssm_estimates, ssm_population = population_parameters(ssm_fits)
        tables.update(
            {
                "ssmcomparison": ssm_fits,
                "ssmcomparisonstats": pd.concat(
                    [
                        model_comparison_statistics(ssm_fits),
                        urgency_condition_contrast(ssm_fits),
                    ],
                    ignore_index=True,
                ),
                "ssmpopulation": ssm_estimates,
                "ssmpopulationstats": ssm_population,
            }
        )
        tables.update(
            {
                name: table
                for name, table in ssm_pooled.items()
                if name != "ssmcomparison" and not table.empty
            }
        )

    analysis_subjects = sorted(features["subject"].unique())
    outputs = []
    for name, table in tables.items():
        output_path = layout.behavior_analysis(name)
        save_table(
            output_path,
            table,
            metadata={
                "stage": "behavior_characterization_analysis",
                "analysis": name,
                "subjects": analysis_subjects,
                "excluded_subjects": sorted(excluded),
                "inference": (
                    "per-subject fit followed by a group test on the fitted "
                    "values (two-stage summary statistics)"
                ),
                "input_files": [str(features_path), str(summary_path)],
            },
        )
        outputs.append(output_path)

    return WorkflowResult(
        stage="behavior_characterization_analysis",
        inputs=(features_path, summary_path),
        outputs=tuple(outputs),
        settings={
            "subjects": analysis_subjects,
            "excluded_subjects": sorted(excluded),
            "n_subjects": len(analysis_subjects),
            "n_trials": int(len(features)),
            "analyses": sorted(tables),
            "sequential_sampling_subjects_missing": ssm_missing,
        },
    )
