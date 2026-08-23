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
from typing import Optional, Sequence

import pandas as pd

from meg_tokens.behavior.analyses.design_effects import (
    choice_side_statistics,
    choice_side_summary,
    condition_class_cells,
    condition_class_contrasts,
    condition_class_statistics,
    condition_order_effects,
    condition_order_statistics,
    extreme_decision_times,
    lapse_statistics,
    lapse_summary,
    time_on_task,
    time_on_task_statistics,
    within_block_decile_profile,
    within_block_decile_statistics,
)
from meg_tokens.behavior.analyses.distributions import (
    decision_time_distribution_statistics,
    decision_time_distributions,
    decision_time_geometry,
    decision_time_geometry_statistics,
    spd_cumulative_distributions,
)
from meg_tokens.behavior.analyses.evidence import (
    conditional_accuracy_functions,
    conditional_accuracy_statistics,
    continuous_evidence_effects,
    continuous_evidence_statistics,
    early_evidence_order_profile,
    early_evidence_order_statistics,
    early_evidence_profile,
    criterion_decline_statistics,
    criterion_scale_comparison,
    criterion_scale_comparison_statistics,
    criterion_time_profile,
    discretization_overshoot_fits,
    discretization_overshoot_statistics,
    evidence_at_decision_responses,
    first_order_criterion_decline,
    posterior_evidence_criterion_agreement,
    posterior_evidence_sensitivity,
    posterior_evidence_sensitivity_statistics,
    posterior_grid_by_token_index,
    posterior_grid_step_summary,
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
from meg_tokens.behavior.tables import read_trial_features
from meg_tokens.core import ProjectConfig, WorkflowResult, normalize_subject_id
from meg_tokens.io import DerivativeLayout, load_table, require_file, save_table


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
    :func:`~meg_tokens.workflows.sequential_sampling.fit_subject_sequential_sampling`;
    the characterization only pools them, so a partly finished array job
    still yields every other derivative.
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

    This *consumes* the sequential-sampling fits but does not aggregate them.
    It pools whatever per-subject tables
    :func:`~meg_tokens.workflows.sequential_sampling.fit_subject_sequential_sampling`
    has written, uses them for ``speciescomparison``, and names any unfitted
    subject in the result. It writes no ``ssm*`` group derivative.

    Notes
    -----
    ``ssm-aggregate`` in :mod:`meg_tokens.workflows.sequential_sampling` is the
    sole writer of every ``ssm*`` group table. It used to share four of those
    filenames with this workflow -- ``ssmcomparison``, ``ssmcomparisonstats``,
    ``ssmtimecourse`` and ``ssmtrialpredictions`` -- so whichever pipeline ran
    last silently won. The two differed in substance, not just in timing: this
    one pooled whatever
    per-subject files existed, while ``ssm-aggregate`` admits only fits that
    pass its three-start acceptance gate, and its ``ssmcomparisonstats``
    additionally carries the ``ssm_mechanistic_model_comparison`` rows that the
    Act 3 adjudication panel selects on. Running characterization second
    therefore replaced gated results with ungated ones and deleted an entire
    analysis block. Single ownership removes the failure mode rather than
    ordering around it, so neither run order can now corrupt the other.
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
    dt_geometry = decision_time_geometry(features)
    cells = condition_class_cells(features)
    sides = choice_side_summary(features)
    drift = time_on_task(features)
    decile_profile = within_block_decile_profile(features)
    order = condition_order_effects(features)
    lapses = lapse_summary(features)
    extreme_counts, extreme_flagged = extreme_decision_times(features)
    criterion = first_order_criterion_decline(features)
    criterion_scale = criterion_scale_comparison(features)
    criterion_profile = criterion_time_profile(features)
    urgency = evidence_at_decision_responses(features, predictor="dt_ms")
    posterior_evidence_index = evidence_at_decision_responses(
        features, predictor="decision_token_index"
    )
    posterior_grid = posterior_grid_by_token_index()
    discretization_fits = discretization_overshoot_fits(features)
    posterior_sensitivity = posterior_evidence_sensitivity(features)
    kernels = reverse_correlation(features)
    accuracy_functions = conditional_accuracy_functions(features)
    evidence = continuous_evidence_effects(features)
    evidence_profile = early_evidence_profile(features)
    evidence_order = early_evidence_order_profile(features)
    # Pooled here to feed `speciescomparison` and to report who is unfitted.
    # These fits are *not* written back out as group tables: `ssm-aggregate`
    # owns every `ssm*` group derivative. See the module docstring.
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
        "dtgeometry": dt_geometry,
        "dtgeometrystats": decision_time_geometry_statistics(dt_geometry),
        "conditionclass": cells,
        "conditionclassstats": condition_class_statistics(cells),
        "conditionclasscontrasts": condition_class_contrasts(cells),
        "choiceside": sides,
        "choicesidestats": choice_side_statistics(sides),
        "timeontask": drift,
        "timeontaskstats": time_on_task_statistics(drift),
        "withinblockdecile": decile_profile,
        "withinblockdecilestats": within_block_decile_statistics(decile_profile),
        "conditionorder": order,
        "conditionorderstats": condition_order_statistics(order),
        "lapses": lapses,
        "lapsestats": lapse_statistics(lapses),
        "extremedt": extreme_counts,
        "extremedttrials": extreme_flagged,
        "criteriondecline": criterion,
        "criteriondeclinestats": criterion_decline_statistics(criterion),
        "criterionscale": criterion_scale,
        "criterionscaleprofile": criterion_profile,
        "criterionscalestats": criterion_scale_comparison_statistics(
            criterion_scale, criterion_profile
        ),
        "urgency": urgency,
        "urgencystats": criterion_decline_statistics(urgency),
        "posteriorevidenceindex": posterior_evidence_index,
        "posteriorevidenceindexstats": criterion_decline_statistics(
            posterior_evidence_index
        ),
        "posteriorgrid": posterior_grid,
        "posteriorgridstats": posterior_grid_step_summary(posterior_grid),
        "posteriordiscretization": discretization_fits,
        "posteriordiscretizationstats": discretization_overshoot_statistics(
            discretization_fits
        ),
        "posteriorevidencesensitivity": posterior_sensitivity,
        "posteriorevidencesensitivitystats": posterior_evidence_sensitivity_statistics(
            posterior_sensitivity
        ),
        "posteriorevidenceagreement": posterior_evidence_criterion_agreement(
            urgency_fits=urgency, criterion_fits=criterion
        ),
        "reversecorrelation": kernels,
        "reversecorrelationstats": reverse_correlation_statistics(kernels),
        "conditionalaccuracy": accuracy_functions,
        "conditionalaccuracystats": conditional_accuracy_statistics(accuracy_functions),
        "continuousevidence": evidence,
        "continuousevidenceprofile": evidence_profile,
        "continuousevidenceorder": evidence_order,
        # The order rows share a table with the level rows so the write-up
        # cites one statistics derivative for this figure, not two.
        "continuousevidencestats": pd.concat(
            [
                continuous_evidence_statistics(evidence, evidence_profile),
                early_evidence_order_statistics(evidence_order),
            ],
            ignore_index=True,
        ),
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
