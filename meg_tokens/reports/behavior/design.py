"""Design-effects figures: F08-F13."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from meg_tokens.behavior.analyses.design_effects import _session_block_order
from meg_tokens.reports import style
from meg_tokens.reports.annotations import annotate_anova, annotate_stat_block, format_stat, stat_from_row
from meg_tokens.reports.behavior._tables import BehaviorTableSet
from meg_tokens.reports.panels import group_line, paired_slope, subject_strip


def build_conditionclass_anova(tables: BehaviorTableSet) -> tuple[Figure, dict[str, Any]]:
    """F08: condition x class -- the claim is the ABSENCE of an interaction."""
    cells = tables.analysis("conditionclass")
    cell_stats = tables.analysis("conditionclassstats")

    with style.apply_publication_style():
        fig, axes = style.figure_grid(1, 2, width="double")
        for index, (measure, ylabel) in enumerate((("mean_dt_ms", "Decision time (ms)"), ("accuracy", "Accuracy"))):
            ax = axes[0, index]
            style.panel_label(ax, "AB"[index])
            for condition in style.CONDITION_ORDER:
                subject_matrix = cells.loc[cells["condition"] == condition].pivot(
                    index="subject", columns="trial_class_name", values=measure
                )[list(style.CLASS_ORDER)].to_numpy(dtype=float)
                group_line(
                    ax, x=np.arange(3), subject_matrix=subject_matrix,
                    color=style.CONDITION_COLORS[condition], label=condition,
                    draw_subjects=True,
                )
            if measure == "accuracy":
                ax.axhline(0.5, color=style.INK_MUTED, linewidth=0.7, linestyle="--")
                ax.text(0.02, 0.05, "chance", transform=ax.transAxes, fontsize=6, color=style.INK_SECONDARY)
            ax.set_xticks(np.arange(3))
            ax.set_xticklabels(style.CLASS_ORDER)
            ax.set_ylabel(ylabel)
            ax.legend(loc="best", fontsize=6)
            rows = cell_stats.loc[cell_stats["measure"] == measure]
            annotate_anova(ax, rows, loc="upper left" if measure == "mean_dt_ms" else "lower right")

        fig.suptitle("Condition x class: no interaction on either measure")

    metadata = {
        "kind": "factorial_interaction",
        "title": "Condition x class",
        "columns_read": {
            "conditionclass": ["subject", "condition", "trial_class_name", "mean_dt_ms", "accuracy"],
            "conditionclassstats": ["measure", "effect", "F", "df_effect", "df_error", "p", "partial_eta_squared", "n_subjects"],
        },
        "statistics_source": "conditionclassstats",
        "palette": dict(style.CONDITION_COLORS),
    }
    return fig, metadata


def build_choiceside_asymmetry(tables: BehaviorTableSet) -> tuple[Figure, dict[str, Any]]:
    """F09: left/right choice, DT, and accuracy asymmetry."""
    summary = tables.analysis("choiceside")
    stats_table = tables.analysis("choicesidestats")

    measures = (
        ("choice_proportion", "proportion_left_choices", "proportion_right_choices", "Choice proportion"),
        ("decision_time", "mean_left_dt_ms", "mean_right_dt_ms", "Decision time (ms)"),
        ("accuracy", "accuracy_left", "accuracy_right", "Accuracy"),
    )
    conditions = ("all", "fast", "slow")

    with style.apply_publication_style():
        fig, axes = style.figure_grid(3, 3, width="double", panel_height_in=2.0)
        for row_index, condition in enumerate(conditions):
            group = summary.loc[summary["condition"] == condition]
            for col_index, (measure, column_a, column_b, ylabel) in enumerate(measures):
                ax = axes[row_index, col_index]
                if row_index == 0:
                    style.panel_label(ax, chr(ord("A") + col_index))
                values_a = group[column_a].to_numpy(dtype=float)
                values_b = group[column_b].to_numpy(dtype=float)
                paired_slope(
                    ax, values_a=values_a, values_b=values_b,
                    label_a="left", label_b="right",
                    color_a=style.INK, color_b=style.INK_SECONDARY,
                    ylabel=ylabel if col_index == 0 else "",
                    reference=0.5 if measure == "choice_proportion" else None,
                )
                row = stats_table.loc[(stats_table["measure"] == measure) & (stats_table["condition"] == condition)]
                if len(row):
                    result = stat_from_row(row.iloc[0], label=measure)
                    ax.text(
                        0.5, 1.05, format_stat(result, include_effect_size=False),
                        transform=ax.transAxes, ha="center", va="bottom", fontsize=5.5, color=style.INK,
                    )
                if col_index == 0:
                    ax.text(-0.55, 0.5, condition, transform=ax.transAxes, ha="right", va="center", fontsize=7, rotation=90)

        fig.suptitle("Left/right choice asymmetry")

    metadata = {
        "kind": "asymmetry_grid",
        "title": "Left/right choice, DT, and accuracy asymmetry",
        "columns_read": {
            "choiceside": ["subject", "condition", "proportion_left_choices", "proportion_right_choices", "mean_left_dt_ms", "mean_right_dt_ms", "accuracy_left", "accuracy_right"],
            "choicesidestats": ["measure", "condition", "mean_a", "mean_b", "mean_difference", "t", "p", "df", "cohens_dz"],
        },
        "statistics_source": "choicesidestats",
    }
    return fig, metadata


def build_timeontask_drift(tables: BehaviorTableSet) -> tuple[Figure, dict[str, Any]]:
    """F10: session (block-order) and within-block drift. Data before coefficient."""
    trial_features = tables.trial_features()
    timeontask = tables.analysis("timeontask")
    timeontaskstats = tables.analysis("timeontaskstats")

    trials = trial_features.copy()
    trials["block_position"] = _session_block_order(trials)
    trials["decile"] = trials.groupby(["subject"])["run_trial_index"].transform(
        lambda values: pd.qcut(values, 10, labels=False, duplicates="drop")
    )

    with style.apply_publication_style():
        fig, axes = style.figure_grid(2, 2, width="double")
        style.panel_label(axes[0, 0], "A")
        style.panel_label(axes[0, 1], "B")
        style.panel_label(axes[1, 0], "C")
        style.panel_label(axes[1, 1], "D")

        # Panel A: observed DT by session block.
        block_means = trials.pivot_table(index="subject", columns="block_position", values="dt_ms", aggfunc="mean")
        group_line(axes[0, 0], x=block_means.columns.to_numpy(dtype=float), subject_matrix=block_means.to_numpy(dtype=float), color=style.INK, label="observed", draw_subjects=True)
        axes[0, 0].set_xlabel("Session block order")
        axes[0, 0].set_ylabel("Mean dt_ms (ms)")

        # Panel B: observed DT by within-block decile.
        decile_means = trials.pivot_table(index="subject", columns="decile", values="dt_ms", aggfunc="mean")
        group_line(axes[0, 1], x=decile_means.columns.to_numpy(dtype=float), subject_matrix=decile_means.to_numpy(dtype=float), color=style.INK, label="observed", draw_subjects=True)
        axes[0, 1].set_xlabel("Within-block trial decile")
        axes[0, 1].set_ylabel("Mean dt_ms (ms)")

        # Panels C, D: fitted coefficients.
        for ax, term, letter in ((axes[1, 0], "dt_per_block_ms", "C"), (axes[1, 1], "dt_per_within_block_trial_ms", "D")):
            groups = {
                condition: timeontask.loc[timeontask["condition"] == condition, term].to_numpy(dtype=float)
                for condition in ("all", "fast", "slow")
            }
            subject_strip(ax, groups=groups, colors={"all": style.INK, "fast": style.CONDITION_COLORS["fast"], "slow": style.CONDITION_COLORS["slow"]}, reference=0.0, ylabel=term)
            all_row = timeontaskstats.loc[(timeontaskstats["term"] == term) & (timeontaskstats["condition"] == "all")]
            fvs_row = timeontaskstats.loc[(timeontaskstats["term"] == term) & (timeontaskstats["condition"] == "fast_vs_slow")]
            lines = []
            if len(all_row):
                lines.append("all: " + format_stat(stat_from_row(all_row.iloc[0], label="all")))
            if len(fvs_row):
                lines.append("fast vs slow: " + format_stat(stat_from_row(fvs_row.iloc[0], label="fvs"), include_effect_size=False))
            annotate_stat_block(ax, lines=lines, loc="upper right")

        fig.suptitle("Session drift and within-block drift")

    metadata = {
        "kind": "trend_and_coefficient",
        "title": "Session and within-block drift",
        "columns_read": {
            "trialfeatures": ["subject", "condition", "run_trial_index", "initial_time_ms", "dt_ms"],
            "timeontask": ["subject", "condition", "dt_per_block_ms", "dt_per_within_block_trial_ms", "converged"],
            "timeontaskstats": ["term", "condition", "mean", "sem", "t", "p", "df", "cohens_dz"],
        },
        "statistics_source": "timeontaskstats",
    }
    return fig, metadata


def build_conditionorder_balance(tables: BehaviorTableSet) -> tuple[Figure, dict[str, Any]]:
    """F11: first-block counterbalancing check (supplementary, BETWEEN-subject)."""
    order = tables.analysis("conditionorder")
    order_stats = tables.analysis("conditionorderstats")

    with style.apply_publication_style():
        fig, axes = style.figure_grid(1, 1, width="single")
        ax = axes[0, 0]
        groups = {
            label: order.loc[order["first_condition"] == label, "slow_minus_fast_dt_ms"].to_numpy(dtype=float)
            for label in ("fast", "slow")
        }
        subject_strip(
            ax, groups=groups,
            colors={"fast": style.CONDITION_COLORS["fast"], "slow": style.CONDITION_COLORS["slow"]},
            reference=0.0, ylabel="Slow − Fast dt_ms (ms)",
        )
        ax.set_xticklabels(["Fast-first", "Slow-first"])
        row = order_stats.loc[order_stats["measure"] == "slow_minus_fast_dt_ms"]
        lines = []
        if len(row):
            r = row.iloc[0]
            lines.append(
                f"{r['mean_a']:.0f} vs {r['mean_b']:.0f} ms, Welch t({r['df']:.1f}) = {r['t']:.2f}, "
                f"p = {r['p']:.2f} (n = {int(r['n_a'])} vs {int(r['n_b'])})"
            )
        lines.append("design-balance check, not a result")
        annotate_stat_block(ax, lines=lines, loc="upper left")
        fig.suptitle("Condition-order balance (between-subject)")

    metadata = {
        "kind": "between_subject_balance_check",
        "title": "First-block counterbalancing",
        "columns_read": {
            "conditionorder": ["subject", "first_condition", "slow_minus_fast_dt_ms"],
            "conditionorderstats": ["measure", "label_a", "label_b", "n_a", "n_b", "mean_a", "mean_b", "t", "p", "df"],
        },
        "statistics_source": "conditionorderstats",
    }
    return fig, metadata


def build_lapses_quality(tables: BehaviorTableSet) -> tuple[Figure, dict[str, Any]]:
    """F12: lapses and extreme decision times (supplementary/QC census)."""
    lapses = tables.analysis("lapses")
    extremedt = tables.analysis("extremedt")
    extremedttrials = tables.analysis("extremedttrials")

    pooled_lapses = lapses.loc[lapses["condition"] == "all"].sort_values("subject")
    outcome_columns = [c for c in pooled_lapses.columns if c.startswith("n_outcome_")]

    with style.apply_publication_style():
        fig, axes = style.figure_grid(1, 3, width="double")
        style.panel_label(axes[0, 0], "A")
        style.panel_label(axes[0, 1], "B")
        style.panel_label(axes[0, 2], "C")

        ax_a = axes[0, 0]
        y_positions = np.arange(len(pooled_lapses))
        markers = ("o", "^")
        for column, marker in zip(outcome_columns, markers):
            ax_a.scatter(pooled_lapses[column], y_positions, marker=marker, color=style.INK, s=14, label=column.replace("n_outcome_", ""))
        ax_a.set_yticks(y_positions)
        ax_a.set_yticklabels(pooled_lapses["subject"], fontsize=5)
        ax_a.set_xlabel("Lapse trials")
        ax_a.legend(fontsize=5.5, loc="lower right")
        total = int(pooled_lapses["n_lapse_trials"].sum())
        total_started = int(pooled_lapses["n_started_trials"].sum())
        ax_a.text(0.98, 0.02, f"{total}/{total_started} ({100*total/total_started:.2f}%)", transform=ax_a.transAxes, ha="right", va="bottom", fontsize=6)

        ax_b = axes[0, 1]
        extremedt_sorted = extremedt.sort_values("subject")
        y_positions_b = np.arange(len(extremedt_sorted))
        ax_b.scatter(extremedt_sorted["n_extreme_dt"], y_positions_b, color=style.INK, s=14, label="n_extreme_dt")
        ax_b.scatter(extremedt_sorted["n_extreme_slow"], y_positions_b, color=style.CONDITION_COLORS["slow"], s=10, marker="s", label="n_extreme_slow")
        ax_b.scatter(extremedt_sorted["n_extreme_fast"], y_positions_b, color=style.CONDITION_COLORS["fast"], s=10, marker="s", label="n_extreme_fast")
        ax_b.scatter(extremedt_sorted["n_negative_dt"], y_positions_b, color="none", edgecolor=style.INK, s=30, marker="D", label="n_negative_dt")
        ax_b.set_yticks(y_positions_b)
        ax_b.set_yticklabels(extremedt_sorted["subject"], fontsize=5)
        ax_b.set_xlabel("Trial count")
        ax_b.legend(fontsize=5, loc="lower right")

        ax_c = axes[0, 2]
        subjects_c = sorted(extremedttrials["subject"].unique())
        subject_index = {subject: index for index, subject in enumerate(subjects_c)}
        y_c = extremedttrials["subject"].map(subject_index)
        is_negative = extremedttrials["dt_ms"] < 0
        ax_c.scatter(extremedttrials.loc[~is_negative, "robust_z"], y_c.loc[~is_negative], color=style.INK, s=10, alpha=0.7)
        ax_c.scatter(extremedttrials.loc[is_negative, "robust_z"], y_c.loc[is_negative], color=style.CONDITION_COLORS["fast"], s=14, marker="x", label="anticipation (dt < 0)")
        ax_c.axvline(5, color=style.INK_MUTED, linewidth=0.6, linestyle="--")
        ax_c.axvline(-5, color=style.INK_MUTED, linewidth=0.6, linestyle="--")
        ax_c.set_yticks(range(len(subjects_c)))
        ax_c.set_yticklabels(subjects_c, fontsize=5)
        ax_c.set_xlabel("Robust z")
        ax_c.legend(fontsize=5.5, loc="lower right")
        ax_c.text(0.02, 0.02, f"{len(extremedttrials)}/{int(extremedt['n_dt_trials'].sum())} at 5 MAD; nothing removed", transform=ax_c.transAxes, fontsize=5.5, color=style.INK_SECONDARY)

        fig.suptitle("Lapses and extreme decision times (QC census)")

    metadata = {
        "kind": "qc_census",
        "title": "Lapses and extreme decision times",
        "columns_read": {
            "lapses": ["subject", "condition", "n_started_trials", "n_lapse_trials", "lapse_rate", *outcome_columns],
            "extremedt": ["subject", "n_dt_trials", "n_extreme_dt", "n_extreme_slow", "n_extreme_fast", "n_negative_dt"],
            "extremedttrials": ["subject", "trial_id", "condition", "run", "dt_ms", "robust_z", "nOutcome"],
        },
        "statistics_source": "none (descriptive census)",
    }
    return fig, metadata


def build_summary_cohort(tables: BehaviorTableSet) -> tuple[Figure, dict[str, Any]]:
    """F13: dataset overview (supplementary). Counts have a meaningful zero,
    so bars are correct here -- one of the few figures in the set where they are."""
    summary = tables.summary()
    class_names = style.CLASS_ORDER

    with style.apply_publication_style():
        fig, axes = style.figure_grid(1, 3, width="double")
        style.panel_label(axes[0, 0], "A")
        style.panel_label(axes[0, 1], "B")
        style.panel_label(axes[0, 2], "C")

        ax_a = axes[0, 0]
        subjects = summary.sort_values("subject")["subject"].tolist()
        y_positions = np.arange(len(subjects))
        ordered = summary.set_index("subject").loc[subjects]
        ax_a.barh(y_positions, ordered["n_fast_trials"], color=style.CONDITION_COLORS["fast"], height=0.35, label="n_fast_trials", alpha=0.5)
        ax_a.barh(y_positions, ordered["n_fast_dt_trials"], color=style.CONDITION_COLORS["fast"], height=0.35, label="n_fast_dt_trials")
        ax_a.barh(y_positions + 0.4, ordered["n_slow_trials"], color=style.CONDITION_COLORS["slow"], height=0.35, label="n_slow_trials", alpha=0.5)
        ax_a.barh(y_positions + 0.4, ordered["n_slow_dt_trials"], color=style.CONDITION_COLORS["slow"], height=0.35, label="n_slow_dt_trials")
        ax_a.set_yticks(y_positions + 0.2)
        ax_a.set_yticklabels(subjects, fontsize=5)
        ax_a.set_xlabel("Trials")
        ax_a.legend(fontsize=5, loc="lower right")

        ax_b = axes[0, 1]
        bottoms = np.zeros(len(subjects))
        for trial_class in class_names:
            values = ordered[f"n_{trial_class}_trials"].to_numpy(dtype=float)
            ax_b.barh(y_positions, values, left=bottoms, color=style.CLASS_COLORS[trial_class], height=0.6, label=trial_class)
            bottoms += values
        ax_b.set_yticks(y_positions)
        ax_b.set_yticklabels(subjects, fontsize=5)
        ax_b.set_xlabel("Trials")
        ax_b.legend(fontsize=5.5, loc="lower right")

        ax_c = axes[0, 2]
        ax_c.scatter(ordered["motor_baseline_ms"], y_positions, color=style.INK, s=14, label="motor_baseline_ms")
        twin = ax_c.twiny()
        twin.scatter(ordered["percent_correct"], y_positions, color=style.MODEL_COLORS["urgency"], s=14, marker="s", label="percent_correct")
        ax_c.set_yticks(y_positions)
        ax_c.set_yticklabels(subjects, fontsize=5)
        ax_c.set_xlabel("motor_baseline_ms", color=style.INK)
        twin.set_xlabel("percent_correct", color=style.MODEL_COLORS["urgency"])

        fig.suptitle(f"Cohort overview ({len(subjects)} subjects, {int(ordered['n_fast_dt_trials'].sum() + ordered['n_slow_dt_trials'].sum())} started-and-chosen task trials)")

    metadata = {
        "kind": "cohort_overview",
        "title": "Dataset overview",
        "columns_read": {
            "summary": [
                "subject", "motor_baseline_ms", "n_fast_trials", "n_slow_trials",
                "n_fast_dt_trials", "n_slow_dt_trials", "n_never_started_trials",
                "n_easy_trials", "n_ambiguous_trials", "n_misleading_trials", "percent_correct",
            ],
        },
        "statistics_source": "none (descriptive)",
        "palette": dict(style.CLASS_COLORS),
    }
    return fig, metadata
