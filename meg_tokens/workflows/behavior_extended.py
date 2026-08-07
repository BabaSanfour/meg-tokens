"""Run the complete extended behavioral analysis pipeline.

``analyze_behavior`` produces the preprint-comparison summaries. This workflow
consumes its trial-feature table and adds distributional descriptions, design
and session effects, continuous evidence, criterion decline, urgency, temporal
weighting, conditional accuracy, trial-history effects, individual
differences, and cross-species comparison.

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


# One entry per roadmap item, so that the derivative names state which analysis
# they come from and the sidecars can carry the item back to the roadmap.
ROADMAP_ITEMS: dict[str, str] = {
    "spdcumulative": "A1 SPD distributions by class",
    "dtdistribution": "A2 DT distributions",
    "conditionclass": "A3 condition x class breakdown",
    "choiceside": "A4 choice-side bias",
    "timeontask": "A5 time-on-task effects",
    "conditionorder": "A5 condition-order effects",
    "lapses": "A6 lapses",
    "extremedt": "A6 extreme DT review",
    "criteriondecline": "B2 accuracy-criterion decline",
    "reversecorrelation": "B3 psychophysical reverse correlation",
    "conditionalaccuracy": "B4 conditional accuracy functions",
    "continuousevidence": "B1/B5 continuous evidence effects",
    "posterror": "B6 robust post-error slowing",
    "choicehistory": "B7 choice-history effects",
    "individualprofile": "B9 individual differences",
    "individualcorrelations": "B9 individual differences",
    "urgency": "C3 explicit urgency extraction",
    "speciescomparison": "C6 monkey-human comparison statistics",
}


def analyze_behavior_extended(
    project: ProjectConfig,
    *,
    subjects: Optional[Sequence[str]] = None,
    neural_metrics_path: Optional[str | Path] = None,
) -> WorkflowResult:
    """Run the extended behavioral pipeline over staged trial features."""
    layout = DerivativeLayout(
        project.bids_root,
        pipeline=project.pipeline,
        task=project.task,
    )
    features_path = layout.behavior_trial_features()
    summary_path = layout.behavior_summary()
    if not features_path.is_file():
        raise FileNotFoundError(
            "The trial-feature table is required by the extended behavioral "
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
    criterion = evidence_at_decision_responses(
        features, predictor="decision_token_index"
    )
    urgency = evidence_at_decision_responses(features, predictor="dt_ms")
    kernels = reverse_correlation(features)
    accuracy_functions = conditional_accuracy_functions(features)
    evidence = continuous_evidence_effects(features)
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
        "speciescomparison": comparison_statistics(summary, criterion=criterion),
    }

    analysis_subjects = sorted(features["subject"].unique())
    outputs = []
    for name, table in tables.items():
        output_path = layout.behavior_analysis(name)
        save_table(
            output_path,
            table,
            metadata={
                "stage": "behavior_extended_analysis",
                "analysis": name,
                "roadmap_item": ROADMAP_ITEMS.get(
                    name.removesuffix("stats").removesuffix("trials"), name
                ),
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
        stage="behavior_extended_analysis",
        inputs=(features_path, summary_path),
        outputs=tuple(outputs),
        settings={
            "subjects": analysis_subjects,
            "excluded_subjects": sorted(excluded),
            "n_subjects": len(analysis_subjects),
            "n_trials": int(len(features)),
            "analyses": sorted(tables),
        },
    )
