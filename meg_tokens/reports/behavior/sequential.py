"""Sequential-effects figures: F19, F20."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.figure import Figure

from meg_tokens.reports import style
from meg_tokens.reports.annotations import annotate_stat_block, format_stat, stat_from_row
from meg_tokens.reports.behavior._tables import BehaviorTableSet
from meg_tokens.reports.panels import estimation_axis, paired_slope, subject_strip


def build_posterror_slowing(tables: BehaviorTableSet) -> tuple[Figure, dict[str, Any]]:
    """F19: robust post-error slowing. Panel B's connectors are the point --
    the claim is that the effect survives the stricter definition."""
    posterror = tables.analysis("posterror")
    posterror_stats = tables.analysis("posterrorstats")

    pooled = posterror.loc[posterror["condition"] == "all"]

    with style.apply_publication_style():
        fig, axes = style.figure_grid(1, 2, width="double")
        ax_a, ax_b = axes[0, 0], axes[0, 1]
        style.panel_label(ax_a, "A")
        style.panel_label(ax_b, "B")

        pre = pooled["mean_pre_error_dt_ms"].to_numpy(dtype=float)
        post = pooled["mean_post_error_dt_ms"].to_numpy(dtype=float)
        paired_slope(ax_a, values_a=pre, values_b=post, label_a="pre-error", label_b="post-error", color_a=style.INK_SECONDARY, color_b=style.INK, ylabel="Decision time (ms)")

        robust = pooled["robust_pes_ms"].to_numpy(dtype=float)
        classical = pooled["classical_pes_ms"].to_numpy(dtype=float)
        paired_slope(ax_b, values_a=robust, values_b=classical, label_a="robust", label_b="classical", color_a=style.MODEL_COLORS["urgency"], color_b=style.MODEL_COLORS["ddm"], ylabel="Post-error slowing (ms)", reference=0.0)

        lines = []
        for measure, label in (("robust_pes_ms", "robust"), ("classical_pes_ms", "classical")):
            row = posterror_stats.loc[(posterror_stats["measure"] == measure) & (posterror_stats["condition"] == "all")]
            if len(row):
                lines.append(f"{label}: " + format_stat(stat_from_row(row.iloc[0], label=label)))
        annotate_stat_block(ax_b, lines=lines, loc="lower left")

        fig.suptitle("Robust post-error slowing")

    metadata = {
        "kind": "post_error_pairing",
        "title": "Robust post-error slowing",
        "columns_read": {
            "posterror": ["subject", "condition", "mean_pre_error_dt_ms", "mean_post_error_dt_ms", "robust_pes_ms", "classical_pes_ms"],
            "posterrorstats": ["measure", "condition", "mean", "sem", "t", "p", "df", "cohens_dz"],
        },
        "statistics_source": "posterrorstats",
    }
    return fig, metadata


_CLASS_HISTORY_MEASURES = ("mean_dt_after_easy_ms", "mean_dt_after_ambiguous_ms", "mean_dt_after_misleading_ms")


def build_choicehistory_effects(tables: BehaviorTableSet) -> tuple[Figure, dict[str, Any]]:
    """F20: choice-history effects (win-stay/lose-shift, side autocorrelation,
    post-error DT, DT by previous class)."""
    history = tables.analysis("choicehistory")
    history_stats = tables.analysis("choicehistorystats")

    pooled = history.loc[history["condition"] == "all"]

    with style.apply_publication_style():
        fig, axes = style.figure_grid(2, 2, width="double")
        style.panel_label(axes[0, 0], "A")
        style.panel_label(axes[0, 1], "B")
        style.panel_label(axes[1, 0], "C")
        style.panel_label(axes[1, 1], "D")

        paired_slope(axes[0, 0], values_a=pooled["win_stay"].to_numpy(dtype=float), values_b=pooled["lose_stay"].to_numpy(dtype=float), label_a="win-stay", label_b="lose-stay", color_a=style.INK, color_b=style.INK_SECONDARY, ylabel="Proportion", reference=0.5)
        row = history_stats.loc[(history_stats["measure"] == "win_stay_vs_lose_stay") & (history_stats["condition"] == "all")]
        if len(row):
            axes[0, 0].text(0.5, 1.08, format_stat(stat_from_row(row.iloc[0], label="a")), transform=axes[0, 0].transAxes, ha="center", fontsize=6)

        result_row = history_stats.loc[(history_stats["measure"] == "side_autocorrelation_lag1") & (history_stats["condition"] == "all")]
        result = stat_from_row(result_row.iloc[0], label="autocorr") if len(result_row) else None
        if result is not None:
            estimation_axis(axes[0, 1], differences=pooled["side_autocorrelation_lag1"].to_numpy(dtype=float), result=result)
        axes[0, 1].set_title("Side autocorrelation (lag 1)", fontsize=7.5)

        paired_slope(axes[1, 0], values_a=pooled["mean_dt_after_correct_ms"].to_numpy(dtype=float), values_b=pooled["mean_dt_after_error_ms"].to_numpy(dtype=float), label_a="after correct", label_b="after error", color_a=style.INK_SECONDARY, color_b=style.INK, ylabel="Decision time (ms)")
        row = history_stats.loc[(history_stats["measure"] == "dt_after_error_vs_correct") & (history_stats["condition"] == "all")]
        if len(row):
            axes[1, 0].text(0.5, 1.08, format_stat(stat_from_row(row.iloc[0], label="a")), transform=axes[1, 0].transAxes, ha="center", fontsize=6)

        groups = {
            cls: pooled[f"mean_dt_after_{cls}_ms"].to_numpy(dtype=float)
            for cls in style.CLASS_ORDER
        }
        subject_strip(axes[1, 1], groups=groups, colors=style.CLASS_COLORS, reference=None, ylabel="Decision time (ms)")
        axes[1, 1].set_title("DT by previous trial class", fontsize=7.5)
        lines = []
        for measure, label in (
            ("dt_after_easy_vs_ambiguous", "easy vs ambiguous"),
            ("dt_after_ambiguous_vs_misleading", "ambiguous vs misleading"),
        ):
            row = history_stats.loc[(history_stats["measure"] == measure) & (history_stats["condition"] == "all")]
            if len(row):
                lines.append(f"{label}: " + format_stat(stat_from_row(row.iloc[0], label=label), include_effect_size=False))
        annotate_stat_block(axes[1, 1], lines=lines, loc="upper right")

        fig.suptitle("Choice-history effects")

    metadata = {
        "kind": "choice_history",
        "title": "Choice-history effects",
        "columns_read": {
            "choicehistory": [
                "subject", "condition", "win_stay", "lose_stay", "side_autocorrelation_lag1",
                "mean_dt_after_correct_ms", "mean_dt_after_error_ms", *_CLASS_HISTORY_MEASURES,
            ],
            "choicehistorystats": ["measure", "condition", "mean", "sem", "t", "p", "df", "cohens_dz"],
        },
        "statistics_source": "choicehistorystats",
        "palette": dict(style.CLASS_COLORS),
    }
    return fig, metadata
