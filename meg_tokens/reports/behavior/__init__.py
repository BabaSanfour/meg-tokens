"""Behavioral figure registry.

Each :class:`FigureSpec` names the derivative(s) it reads (`requires`), the
group(s) it belongs to for `--figures` selection, and the builder function
that draws it. Figure-module names mirror `meg_tokens/behavior/analyses/`
one-for-one, so the figure for an analysis is found by module name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from matplotlib.figure import Figure

from meg_tokens.reports.behavior._tables import BehaviorTableSet
from meg_tokens.reports.behavior import design, distributions, evidence, individual, modeling, sequential

Builder = Callable[[BehaviorTableSet], tuple[Figure, dict[str, object]]]


@dataclass(frozen=True)
class FigureSpec:
    key: str  # "ssmcomparison-model"
    analysis: str  # desc segment 1 / statistics derivative
    view: str  # desc segment 2
    title: str
    kind: str
    groups: tuple[str, ...]  # ("headline", "modeling")
    requires: tuple[str, ...]  # derivative names that must exist
    builder: Builder
    caveat: str | None = None


REGISTRY: tuple[FigureSpec, ...] = (
    # --- Act 1: behavioral foundation (F01-F08) ---
    # Cohort/QC, the Fast/Slow manipulation, stimulus-frame difficulty,
    # decision-time distributions, and confidence at commitment.
    FigureSpec(
        key="summary-cohort", analysis="summary", view="cohort",
        title="Cohort composition and data quality",
        kind="cohort_overview", groups=("design", "qc"),
        requires=("summary", "lapses", "extremedt"),
        builder=design.build_summary_cohort,
    ),
    FigureSpec(
        key="conditionorder-balance", analysis="conditionorder", view="balance",
        title="First-block counterbalancing",
        kind="between_subject_balance_check", groups=("design",),
        requires=("conditionorder", "conditionorderstats"),
        builder=design.build_conditionorder_balance,
    ),
    FigureSpec(
        key="choiceside-asymmetry", analysis="choiceside", view="asymmetry",
        title="Left/right choice asymmetry",
        kind="asymmetry_grid", groups=("design",),
        requires=("choiceside", "choicesidestats"),
        builder=design.build_choiceside_asymmetry,
    ),
    FigureSpec(
        key="timeontask-drift", analysis="timeontask", view="drift",
        title="Session and within-block drift",
        kind="trend_and_coefficient", groups=("design",),
        requires=("timeontaskstats", "withinblockdecile"),
        builder=design.build_timeontask_drift,
    ),
    FigureSpec(
        key="conditionclass-anova", analysis="conditionclass", view="anova",
        title="Condition x class",
        kind="factorial_interaction", groups=("design",),
        requires=("conditionclass", "conditionclassstats", "conditionclasscontrasts"),
        builder=design.build_conditionclass_anova,
    ),
    FigureSpec(
        key="dtdistribution-condition", analysis="dtdistribution", view="condition",
        title="Decision time, Fast vs. Slow",
        kind="condition_distribution", groups=("core", "distributions"),
        requires=("dtdistribution", "dtdistributionstats", "dtgeometry", "groupstats"),
        builder=distributions.build_dtdistribution_condition,
    ),
    FigureSpec(
        key="dtdistribution-class", analysis="dtdistribution", view="class",
        title="Decision time by trial class",
        kind="class_distribution", groups=("core", "distributions"),
        requires=("dtdistribution", "dtdistributionstats"),
        builder=distributions.build_dtdistribution_class,
        caveat="Unclassified trials excluded (~43% of the sample; see "
        "docs/behavior.md, \"Trial-classification reference frame\").",
    ),
    FigureSpec(
        key="spdcumulative-class", analysis="spdcumulative", view="class",
        title="SPD at decision, cumulative by class",
        kind="cumulative_distribution", groups=("distributions",),
        requires=("spdcumulative", "groupstats"),
        builder=distributions.build_spdcumulative_class,
        caveat="validated_15row only (the higher-integrity subset; excludes "
        "unvalidated \"14-row\" logs). all_logged was checked against it and "
        "agrees within a few points despite excluding up to 40% fewer trials "
        "per class, so showing both was redundant. all_logged is still used "
        "elsewhere in the pipeline (summary, individual.py, performance.py) "
        "-- whether those should also move to validated_15row is TBD.",
    ),
    # --- Act 2: dynamic commitment policy (F09-F12) ---
    # How changing evidence becomes a choice: criterion across time,
    # conditional accuracy, continuous-evidence effects, and the temporal
    # choice kernel. F10 is the exact-posterior sensitivity companion to F09.
    FigureSpec(
        key="criterionscale-comparison",
        analysis="criterionscale",
        view="comparison",
        title="Evidence criterion vs. decision time, on two evidence scales",
        kind="criterion_scale_comparison", groups=("evidence",),
        requires=("criterionscale", "criterionscaleprofile", "criterionscalestats"),
        builder=evidence.build_criterionscale_comparison,
    ),
    FigureSpec(
        key="conditionalaccuracy-caf", analysis="conditionalaccuracy", view="caf",
        title="Conditional accuracy function, by trial class",
        kind="conditional_accuracy_function", groups=("evidence",),
        requires=("conditionalaccuracy", "conditionalaccuracystats"),
        builder=evidence.build_conditionalaccuracy_caf,
    ),
    FigureSpec(
        key="continuousevidence-effects", analysis="continuousevidence", view="effects",
        title="Early evidence by trial class",
        kind="early_evidence_by_class", groups=("evidence",),
        # The key is kept even though the figure no longer claims the predictor
        # is continuous: keys are the durable reference across revisions of
        # docs/behavior.md, and the derivative it reads has the same name.
        requires=("continuousevidenceprofile", "continuousevidencestats"),
        builder=evidence.build_continuousevidence_effects,
    ),
    FigureSpec(
        key="reversecorrelation-kernel", analysis="reversecorrelation", view="kernel",
        title="Psychophysical kernel",
        kind="kernel_by_jump", groups=("evidence",),
        requires=("reversecorrelation", "reversecorrelationstats"),
        builder=evidence.build_reversecorrelation_kernel,
    ),
    # --- Act 3: mechanistic adjudication (F13-F15) ---
    # Formal comparison of urgency gating, bounded integration,
    # collapsing/adaptive bounds, and plausible urgency variants.
    # (behavior/analyses/sequential_sampling.py)
    FigureSpec(
        key="ssmcomparison-model", analysis="ssmcomparison", view="model",
        title="Model comparison: mechanism and adjudication",
        kind="mechanistic_model_comparison", groups=("headline", "modeling"),
        requires=(
            "ssmcomparison", "ssmcomparisonstats", "ssmtimecourse",
            "ssmheldout", "ssmheldoutstats", "ssmheldoutpairwise",
            "ssmdistributioncheckstats",
        ),
        builder=modeling.build_ssmcomparison_model,
    ),
    FigureSpec(
        key="ssmcomparison-urgency", analysis="ssmcomparison", view="urgency",
        title="Urgency parameters, Fast vs. Slow",
        kind="paired_condition_contrast_grid", groups=("headline", "modeling"),
        requires=(
            "ssmcomparison", "ssmcomparisonstats", "ssmboundaryaudit",
            "ssmrobustnessstats",
        ),
        builder=modeling.build_ssmcomparison_urgency,
    ),
    FigureSpec(
        key="ssmparameterrecovery-identifiability", analysis="ssmparameterrecovery",
        view="identifiability",
        title="Parameter and model recovery, in the fitted region",
        kind="recovery_identifiability", groups=("modeling",),
        requires=(
            "ssmparameterrecovery", "ssmparameterrecoverystats",
            "ssmmodelrecoverystats",
        ),
        builder=modeling.build_ssmrecovery_identifiability,
    ),
    # --- Act 4: sequential adaptation (F16-F17) ---
    # Post-error slowing and choice/history effects.
    # (behavior/analyses/sequential.py)
    FigureSpec(
        key="posterror-slowing", analysis="posterror", view="slowing",
        title="Robust post-error slowing",
        kind="post_error_pairing", groups=("sequential",),
        requires=("posterror", "posterrorstats"),
        builder=sequential.build_posterror_slowing,
    ),
    FigureSpec(
        key="choicehistory-effects", analysis="choicehistory", view="effects",
        title="Choice-history effects",
        kind="choice_history", groups=("sequential",),
        requires=("choicehistory", "choicehistorystats"),
        builder=sequential.build_choicehistory_effects,
    ),
    # --- Act 5: individual differences (F18) ---
    # Cross-subject structure, cross-species comparison, and the later
    # behavioral-MEG join. (behavior/analyses/individual.py)
    # The matrix and the scatter grid were two figures; they are one argument.
    # At n = 32 a correlation matrix cannot show whether an r is outlier-driven,
    # so the scatters are its required companion, and each of them is one cell
    # of the matrix magnified. Split across figures, the check a reader is meant
    # to perform meant flipping pages and matching cells by name.
    FigureSpec(
        key="individualcorrelations-structure", analysis="individualcorrelations",
        view="structure",
        title="Individual-differences structure",
        kind="correlation_structure", groups=("individual",),
        requires=("individualprofile", "individualcorrelations"),
        builder=individual.build_individualcorrelations_structure,
    ),
    # No cross-species figure. `speciescomparison` is still computed and still
    # the canonical comparison-ready table, but every one of its ten rows is
    # already drawn elsewhere -- DT by class in F05/F07, SP at decision in F08,
    # the criterion slope in F09, the BIC margin and urgency scale in F13/F14 --
    # and published monkey values may not be plotted beside them by design. What
    # was left was a table of numbers reported elsewhere, drawn as a chart,
    # which is what rule 2.3 rejects for `groupstats`. The measure-to-citation
    # mapping it carried now lives in docs/behavior.md, which is where a
    # textual comparison belongs.

)

GROUP_NAMES: tuple[str, ...] = (
    "headline", "core", "distributions", "design", "evidence",
    "sequential", "modeling", "individual", "qc",
)


def list_behavior_figures() -> list[FigureSpec]:
    return list(REGISTRY)


def resolve_figure_keys(selectors: tuple[str, ...]) -> list[str]:
    """Expand a `--figures` selector (keys and/or group names) into an
    ordered, de-duplicated list of registry keys.

    Raises ValueError naming the unknown selector and the valid keys/groups
    when a selector matches neither a figure key nor a group name.
    """
    if selectors == ("all",):
        return [spec.key for spec in REGISTRY]

    valid_keys = {spec.key for spec in REGISTRY}
    resolved: list[str] = []
    seen: set[str] = set()
    for selector in selectors:
        if selector == "all":
            for spec in REGISTRY:
                if spec.key not in seen:
                    resolved.append(spec.key)
                    seen.add(spec.key)
            continue
        if selector in valid_keys:
            if selector not in seen:
                resolved.append(selector)
                seen.add(selector)
            continue
        matched_group = [spec.key for spec in REGISTRY if selector in spec.groups]
        if matched_group:
            for key in matched_group:
                if key not in seen:
                    resolved.append(key)
                    seen.add(key)
            continue
        raise ValueError(
            f"Unknown figure selector {selector!r}. Valid figure keys: "
            f"{sorted(valid_keys)}. Valid groups: {list(GROUP_NAMES)}."
        )
    return resolved
