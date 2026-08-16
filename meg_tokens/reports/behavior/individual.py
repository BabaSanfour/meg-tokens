"""Individual-differences and cross-species figures: F23-F26."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import stats as scipy_stats

from meg_tokens.reports import style
from meg_tokens.reports.annotations import format_p
from meg_tokens.reports.behavior._tables import BehaviorTableSet
from meg_tokens.reports.panels import correlation_heatmap, forest, scatter_fit

_PROFILE_MEASURES = (
    "mean_dt_ms", "sat_adjustment_ms", "urgency_slope_per_second",
    "criterion_slope_per_token", "accuracy_log_odds_per_unit",
    "percent_correct", "lapse_rate",
)


def build_individualcorrelations_matrix(tables: BehaviorTableSet) -> tuple[Figure, dict[str, Any]]:
    """F23: pairwise correlation matrix over the individual-differences measures."""
    correlations = tables.analysis("individualcorrelations")

    r_matrix = pd.DataFrame(index=_PROFILE_MEASURES, columns=_PROFILE_MEASURES, dtype=float)
    p_matrix = pd.DataFrame(index=_PROFILE_MEASURES, columns=_PROFILE_MEASURES, dtype=float)
    for measure in _PROFILE_MEASURES:
        r_matrix.loc[measure, measure] = 1.0
        p_matrix.loc[measure, measure] = 0.0
    for _, row in correlations.iterrows():
        a, b = row["measure_a"], row["measure_b"]
        if a in _PROFILE_MEASURES and b in _PROFILE_MEASURES:
            r_matrix.loc[b, a] = row["pearson_r"]
            p_matrix.loc[b, a] = row["pearson_p"]

    with style.apply_publication_style():
        # Seven long measure names, 45-degree rotated, plus a colorbar do not
        # fit in a single-column figure: constrained_layout silently
        # collapses the main axes to zero size rather than raising. Double
        # width gives the labels and colorbar room to coexist.
        fig, axes = style.figure_grid(1, 1, width="double", panel_height_in=5.2)
        ax = axes[0, 0]
        image = correlation_heatmap(ax, r_matrix=r_matrix, p_matrix=p_matrix, alpha=0.05, lower_triangle_only=True)
        fig.colorbar(image, ax=ax, shrink=0.7, label="Pearson r")
        ax.text(
            0.02, -0.12, "Pearson r, n = 32, uncorrected; outlined = p < .05",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color=style.INK_SECONDARY,
        )
        fig.suptitle("Individual differences: pairwise correlations")

    metadata = {
        "kind": "correlation_matrix",
        "title": "Individual-differences correlation matrix",
        "columns_read": {
            "individualcorrelations": ["measure_a", "measure_b", "n_subjects", "pearson_r", "pearson_p"],
        },
        "statistics_source": "individualcorrelations",
        "multiplicity_correction": "none",
    }
    return fig, metadata


_SCATTER_PAIRS = (
    ("mean_dt_ms", "accuracy_log_odds_per_unit"),
    ("mean_dt_ms", "criterion_slope_per_token"),
    ("mean_dt_ms", "urgency_slope_per_second"),
    ("mean_dt_ms", "percent_correct"),
    ("urgency_slope_per_second", "criterion_slope_per_token"),
    ("sat_adjustment_ms", "mean_dt_ms"),
)


def build_individualprofile_scatter(tables: BehaviorTableSet) -> tuple[Figure, dict[str, Any]]:
    """F24: the correlations that matter, as a scatter grid. A matrix alone
    can't show whether an r at n=32 is outlier-driven."""
    profile = tables.analysis("individualprofile")
    correlations = tables.analysis("individualcorrelations")

    with style.apply_publication_style():
        fig, axes = style.figure_grid(2, 3, width="double", panel_height_in=2.2)
        for index, (measure_a, measure_b) in enumerate(_SCATTER_PAIRS):
            ax = axes[index // 3, index % 3]
            x = profile[measure_a].to_numpy(dtype=float)
            y = profile[measure_b].to_numpy(dtype=float)
            scatter_fit(ax, x=x, y=y, xlabel=measure_a, ylabel=measure_b)
            row = correlations.loc[
                ((correlations["measure_a"] == measure_a) & (correlations["measure_b"] == measure_b))
                | ((correlations["measure_a"] == measure_b) & (correlations["measure_b"] == measure_a))
            ]
            if len(row):
                r, p, n = row.iloc[0]["pearson_r"], row.iloc[0]["pearson_p"], int(row.iloc[0]["n_subjects"])
                ax.text(
                    0.02, 0.98, f"r = {r:.2f}, {format_p(p)}, n = {n}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color=style.INK,
                )
        fig.suptitle("Individual differences: correlations that matter")

    metadata = {
        "kind": "scatter_grid",
        "title": "Individual-differences scatter grid",
        "columns_read": {
            "individualprofile": list(_PROFILE_MEASURES) + ["subject"],
            "individualcorrelations": ["measure_a", "measure_b", "n_subjects", "pearson_r", "pearson_p"],
        },
        "statistics_source": "individualcorrelations",
    }
    return fig, metadata


_SPECIESCOMPARISON_FACETS = (
    ("Milliseconds", ["decision_time_easy_ms", "decision_time_ambiguous_ms", "decision_time_misleading_ms"]),
    ("Probability", ["success_probability_at_decision_easy", "success_probability_at_decision_ambiguous", "success_probability_at_decision_misleading"]),
    ("Log odds per token", ["criterion_slope_log_odds_per_token"]),
    ("BIC", ["urgency_minus_integrator_bic"]),
    ("Criterion-seconds", ["urgency_scale_criterion_seconds", "urgency_scale_fast_minus_slow"]),
)

# The published measure each row is comparable to. Not in the derivative
# (which carries no citation column by design, docs/behavior.md); this is
# the one place that mapping is a constant, taken verbatim from the C6
# table in docs/behavior_roadmap_results.md.
_COMPARABLE_TO = {
    "decision_time_easy_ms": "DT by class (Cisek et al. 2009)",
    "decision_time_ambiguous_ms": "DT by class (Cisek et al. 2009)",
    "decision_time_misleading_ms": "DT by class (Cisek et al. 2009)",
    "success_probability_at_decision_easy": "SP at decision by class (Thura et al. 2012)",
    "success_probability_at_decision_ambiguous": "SP at decision by class (Thura et al. 2012)",
    "success_probability_at_decision_misleading": "SP at decision by class (Thura et al. 2012)",
    "criterion_slope_log_odds_per_token": "evidence at commitment (Thura et al. 2012)",
    "urgency_minus_integrator_bic": "urgency gating preferred over integration (Cisek 2009; Thura 2012)",
    "urgency_scale_criterion_seconds": "urgency signal fit (Thura & Cisek 2014; Carland et al. 2019)",
    "urgency_scale_fast_minus_slow": "urgency differs Fast/Slow (Cisek et al. 2009)",
}


def build_speciescomparison_forest(tables: BehaviorTableSet) -> tuple[Figure, dict[str, Any]]:
    """F25: comparison-ready statistics, faceted by unit group (never a
    shared axis across incommensurate units). Published monkey values are
    NEVER plotted -- comparison is textual only, via the right-hand labels."""
    comparison = tables.analysis("speciescomparison")

    with style.apply_publication_style():
        fig, axes = style.figure_grid(len(_SPECIESCOMPARISON_FACETS), 1, width="double", panel_height_in=1.0)
        for index, (unit_label, measures) in enumerate(_SPECIESCOMPARISON_FACETS):
            ax = axes[index, 0]
            rows = comparison.set_index("measure").loc[[m for m in measures if m in comparison["measure"].values]]
            if not len(rows):
                ax.axis("off")
                continue
            df = rows["df"].to_numpy(dtype=float)
            ci_half = rows["sem"].to_numpy(dtype=float) * scipy_stats.t.ppf(0.975, np.where(df > 0, df, 1))
            centres = rows["mean"].to_numpy(dtype=float)
            forest(
                ax, labels=list(rows.index), centres=centres,
                lows=centres - ci_half, highs=centres + ci_half, reference=0.0,
            )
            ax.set_xlabel(unit_label, fontsize=6.5)
            for y, measure in enumerate(rows.index):
                ax.text(
                    1.02, y, _COMPARABLE_TO.get(measure, ""), transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=5.5, color=style.INK_SECONDARY,
                )
            if unit_label == "BIC":
                ax.text(
                    0.02, 0.95,
                    "one-sample test against zero on a criterion difference is\n"
                    "a model-preference test, not a parameter estimate",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=5, color=style.INK_SECONDARY,
                )

        fig.suptitle("Cross-species comparison statistics (our side only; published values are not plotted)")

    metadata = {
        "kind": "unit_faceted_forest",
        "title": "Cross-species comparison statistics",
        "columns_read": {
            "speciescomparison": ["measure", "n_subjects", "mean", "sem", "t", "p", "df", "cohens_dz"],
        },
        "statistics_source": "speciescomparison",
        "comparable_to_source": "docs/behavior_roadmap_results.md, C6 table (constant in this module, not in the derivative)",
    }
    return fig, metadata


def build_individualprofile_neural(tables: BehaviorTableSet) -> tuple[Figure, dict[str, Any]]:
    """F26: behaviour vs. a subject-level neural metric. Gated on
    --neural-metrics. Correlation r/p is computed HERE, in the plotting
    layer -- the one documented exception to "never compute an inferential
    statistic", because no derivative owns it yet. Revisit once the Tier C5
    join lands."""
    profile = tables.analysis("individualprofile")
    neural = tables.neural_metrics()

    merged = profile.merge(neural[["subject", "neural_peak_ms"]], on="subject", how="inner")
    merged = merged.loc[
        merged["mean_dt_ms"].notna() & merged["neural_peak_ms"].notna()
    ]
    if len(merged) < 3:
        raise ValueError(
            "At least three subjects with behavior and neural metrics are required "
            "for the individualprofile-neural correlation."
        )

    x = merged["mean_dt_ms"].to_numpy(dtype=float)
    y = merged["neural_peak_ms"].to_numpy(dtype=float)
    r, p = scipy_stats.pearsonr(x, y)

    with style.apply_publication_style():
        fig, axes = style.figure_grid(1, 1, width="single", panel_height_in=3.42)
        ax = axes[0, 0]
        scatter_fit(ax, x=x, y=y, xlabel="Mean decision time, dt_ms (ms)", ylabel="Neural peak commitment (ms)")
        ax.text(
            0.02, 0.98, f"r = {r:.2f}, {format_p(float(p))}, n = {len(merged)}",
            transform=ax.transAxes, ha="left", va="top", fontsize=7, color=style.INK,
        )
        fig.suptitle("Behaviour vs. neural metric")

    metadata = {
        "kind": "neural_correlation",
        "title": "Behaviour vs. neural metric",
        "columns_read": {
            "individualprofile": ["subject", "mean_dt_ms"],
            "neural_metrics": ["subject", "neural_peak_ms"],
        },
        "statistics_source": "computed_in_report",
        "n_subjects": len(merged),
    }
    return fig, metadata
