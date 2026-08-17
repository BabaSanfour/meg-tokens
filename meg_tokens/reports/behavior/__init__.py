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
    key: str  # "ssmcomparison-deltabic"
    analysis: str  # desc segment 1 / statistics derivative
    view: str  # desc segment 2
    title: str
    kind: str
    groups: tuple[str, ...]  # ("headline", "modeling")
    requires: tuple[str, ...]  # derivative names that must exist
    builder: Builder
    caveat: str | None = None


REGISTRY: tuple[FigureSpec, ...] = (
    # --- Headline / modeling (behavior/analyses/sequential_sampling.py) ---
    FigureSpec(
        key="ssmcomparison-deltabic", analysis="ssmcomparison", view="deltabic",
        title="Urgency gating versus bounded integration (H1)",
        kind="per_subject_criterion_difference", groups=("headline", "modeling"),
        requires=("ssmcomparison", "ssmcomparisonstats"),
        builder=modeling.build_ssmcomparison_deltabic,
    ),
    FigureSpec(
        key="ssmcomparison-urgencyscale", analysis="ssmcomparison", view="urgencyscale",
        title="Urgency scale, Fast vs. Slow (H2)",
        kind="paired_condition_contrast", groups=("headline", "modeling"),
        requires=("ssmcomparison", "ssmcomparisonstats"),
        builder=modeling.build_ssmcomparison_urgencyscale,
    ),
    FigureSpec(
        key="ssmcomparison-urgencyparams", analysis="ssmcomparison", view="urgencyparams",
        title="Urgency-model parameters, Fast vs. Slow",
        kind="paired_condition_contrast_grid", groups=("modeling",),
        requires=("ssmcomparison", "ssmcomparisonstats"),
        builder=modeling.build_ssmcomparison_urgencyparams,
    ),
    FigureSpec(
        key="ssmtimecourse-fit", analysis="ssmtimecourse", view="fit",
        title="Model internals: criterion, fit quality, decision variable",
        kind="model_time_course", groups=("modeling",),
        requires=("ssmtimecourse",),
        builder=modeling.build_ssmtimecourse_fit,
    ),
    FigureSpec(
        key="ssmpopulation-shrinkage", analysis="ssmpopulation", view="shrinkage",
        title="Population sequential-sampling parameters",
        kind="shrinkage_forest", groups=("modeling",),
        requires=("ssmpopulation", "ssmpopulationstats"),
        builder=modeling.build_ssmpopulation_shrinkage,
    ),
    # --- Core / distributions (behavior/analyses/distributions.py) ---
    FigureSpec(
        key="dtdistribution-condition", analysis="dtdistribution", view="condition",
        title="Decision time, Fast vs. Slow",
        kind="condition_distribution", groups=("core", "distributions"),
        requires=("dtdistribution", "dtdistributionstats"),
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
        requires=("spdcumulative",),
        builder=distributions.build_spdcumulative_class,
        caveat="validated_15row only (the higher-integrity subset; excludes "
        "unvalidated \"14-row\" logs). all_logged was checked against it and "
        "agrees within a few points despite excluding up to 40% fewer trials "
        "per class, so showing both was redundant. all_logged is still used "
        "elsewhere in the pipeline (summary, individual.py, performance.py) "
        "-- whether those should also move to validated_15row is TBD.",
    ),
    # --- Design effects (behavior/analyses/design_effects.py) ---
    FigureSpec(
        key="conditionclass-anova", analysis="conditionclass", view="anova",
        title="Condition x class",
        kind="factorial_interaction", groups=("design",),
        requires=("conditionclass", "conditionclassstats"),
        builder=design.build_conditionclass_anova,
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
        requires=("timeontaskstats",),
        builder=design.build_timeontask_drift,
    ),
    FigureSpec(
        key="conditionorder-balance", analysis="conditionorder", view="balance",
        title="First-block counterbalancing",
        kind="between_subject_balance_check", groups=("design",),
        requires=("conditionorder", "conditionorderstats"),
        builder=design.build_conditionorder_balance,
    ),
    FigureSpec(
        key="summary-cohort", analysis="summary", view="cohort",
        title="Cohort composition and data quality",
        kind="cohort_overview", groups=("design", "qc"),
        requires=("summary", "lapses", "extremedt"),
        builder=design.build_summary_cohort,
    ),
    # --- Evidence and criterion (behavior/analyses/evidence.py) ---
    FigureSpec(
        key="criteriondecline-sumloglr",
        analysis="criteriondecline",
        view="sumloglr",
        title="Chosen-target evidence criterion vs. decision time",
        kind="fitted_criterion", groups=("evidence",),
        requires=("criteriondecline", "criteriondeclinestats"),
        builder=evidence.build_criteriondecline_sumloglr,
    ),
    FigureSpec(
        key="urgency-decisiontime", analysis="urgency", view="decisiontime",
        title="Evidence at decision vs. decision time",
        kind="fitted_criterion", groups=("evidence",),
        requires=("urgency", "urgencystats"),
        builder=evidence.build_urgency_decisiontime,
    ),
    FigureSpec(
        key="reversecorrelation-kernel", analysis="reversecorrelation", view="kernel",
        title="Psychophysical kernel",
        kind="kernel_by_jump", groups=("evidence",),
        requires=("reversecorrelation", "reversecorrelationstats"),
        builder=evidence.build_reversecorrelation_kernel,
    ),
    FigureSpec(
        key="conditionalaccuracy-caf", analysis="conditionalaccuracy", view="caf",
        title="Conditional accuracy function",
        kind="conditional_accuracy_function", groups=("evidence",),
        requires=("conditionalaccuracy", "conditionalaccuracystats"),
        builder=evidence.build_conditionalaccuracy_caf,
    ),
    FigureSpec(
        key="continuousevidence-effects", analysis="continuousevidence", view="effects",
        title="Continuous early-evidence effects",
        kind="continuous_predictor", groups=("evidence",),
        requires=("continuousevidence", "continuousevidencestats"),
        builder=evidence.build_continuousevidence_effects,
    ),
    # --- Sequential effects (behavior/analyses/sequential.py) ---
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
    # --- Individual differences and cross-species (behavior/analyses/individual.py) ---
    FigureSpec(
        key="individualcorrelations-matrix", analysis="individualcorrelations", view="matrix",
        title="Individual-differences correlation matrix",
        kind="correlation_matrix", groups=("individual",),
        requires=("individualcorrelations",),
        builder=individual.build_individualcorrelations_matrix,
    ),
    FigureSpec(
        key="individualprofile-scatter", analysis="individualprofile", view="scatter",
        title="Individual-differences scatter grid",
        kind="scatter_grid", groups=("individual",),
        requires=("individualprofile", "individualcorrelations"),
        builder=individual.build_individualprofile_scatter,
    ),
    FigureSpec(
        key="speciescomparison-forest", analysis="speciescomparison", view="forest",
        title="Cross-species comparison statistics",
        kind="unit_faceted_forest", groups=("individual",),
        requires=("speciescomparison",),
        builder=individual.build_speciescomparison_forest,
    ),
    FigureSpec(
        key="individualprofile-neural", analysis="individualprofile", view="neural",
        title="Behaviour vs. neural metric",
        kind="neural_correlation", groups=("individual", "qc"),
        requires=("individualprofile", "--neural-metrics"),
        builder=individual.build_individualprofile_neural,
    ),
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
