"""Design-effects figures: F08-F13."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from meg_tokens.behavior.analyses.design_effects import _session_block_order
from meg_tokens.reports import style
from meg_tokens.reports.annotations import (
    annotate_anova, annotate_stat_block, format_stat, significance_marker, stat_from_row,
)
from meg_tokens.reports.behavior._tables import BehaviorTableSet
from meg_tokens.reports.panels import group_line, paired_slope, subject_strip, within_subject_error


def build_conditionclass_anova(tables: BehaviorTableSet) -> tuple[Figure, dict[str, Any]]:
    """F08: decision time and accuracy across condition x difficulty.

    The title states what was measured, not what was found: an earlier version
    read 'no interaction on either measure', which asserted absence from
    p > .05. The interaction is reported on the log scale, where it asks the
    question the condition effect actually poses -- whether Fast/Slow scales
    every difficulty by the same factor -- and the per-class factors are drawn
    on panel A so the positive form of that claim is visible.
    """
    cells = tables.analysis("conditionclass")
    cell_stats = tables.analysis("conditionclassstats")

    # Per-class Fast->Slow stretch factors, geometric mean over subjects (the
    # estimator matching the log-scale test annotated alongside them).
    wide = cells.pivot_table(
        index="subject", columns=["condition", "trial_class_name"], values="mean_dt_ms"
    )
    factors = []
    for class_name in style.CLASS_ORDER:
        if ("slow", class_name) in wide.columns and ("fast", class_name) in wide.columns:
            ratio = wide[("slow", class_name)] / wide[("fast", class_name)]
            ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
            factors.append(float(np.exp(np.log(ratio[ratio > 0]).mean())) if len(ratio) else float("nan"))

    with style.apply_publication_style():
        fig, axes = style.figure_grid(1, 2, width="double", panel_height_in=3.6)
        # Decision time is annotated from the log-scale ANOVA even though the
        # axis stays in milliseconds: the panel is read in ms, but the effect
        # being tested is multiplicative.
        panels = (
            ("mean_dt_ms", "log_mean_dt_ms", "Decision time (ms)"),
            ("accuracy", "accuracy", "Accuracy"),
        )
        for index, (measure, stat_measure, ylabel) in enumerate(panels):
            ax = axes[0, index]
            style.panel_label(ax, "AB"[index])
            # Group means with within-subject 95% CI bands, no subject traces.
            # The paired design removes between-subject spread, so the traces
            # showed variability the test does not use -- and they forced a
            # y-range four times wider than the effect, which hid both the
            # condition gap and the band itself.
            plotted_low, plotted_high = [], []
            for condition in style.CONDITION_ORDER:
                subject_matrix = cells.loc[cells["condition"] == condition].pivot(
                    index="subject", columns="trial_class_name", values=measure
                )[list(style.CLASS_ORDER)].to_numpy(dtype=float)
                group_line(
                    ax, x=np.arange(3), subject_matrix=subject_matrix,
                    color=style.CONDITION_COLORS[condition], label=condition,
                    error="within_ci95",
                )
                mean = np.nanmean(subject_matrix, axis=0)
                band = within_subject_error(subject_matrix, kind="ci95")
                plotted_low.append(np.nanmin(mean - band))
                plotted_high.append(np.nanmax(mean + band))
            if measure == "accuracy":
                ax.axhline(0.5, color=style.INK_MUTED, linewidth=0.7, linestyle="--")
                # x in axes fraction, y in data units, so the label tracks the
                # 0.5 line rather than a fixed height in the panel.
                ax.text(
                    0.01, 0.515, "chance", transform=ax.get_yaxis_transform(),
                    ha="left", va="bottom", fontsize=11, color=style.INK_SECONDARY,
                )
            ax.set_xticks(np.arange(3))
            ax.set_xticklabels(style.CLASS_ORDER)
            # Category axes carry no data between the ticks, so the padding is
            # dead space. Keep only enough that the end markers clear the
            # spines -- narrowing the range also spreads the tick labels
            # further apart in pixels, which is what stops them colliding.
            ax.set_xlim(-0.15, 2.15)
            ax.set_ylabel(ylabel)
            # Limits track what is actually drawn -- the mean +- CI band, not
            # every subject cell -- plus a small buffer, so nothing touches a
            # spine and no empty canvas is left above or below.
            low, high = float(min(plotted_low)), float(max(plotted_high))
            if measure == "accuracy":
                ax.set_ylim(low - 0.03, high + 0.03)
                ax.set_yticks([0.4, 0.6, 0.8, 1.0])
            else:
                # Rounded outward to a 25 ms grid (currently 936-1547 ms
                # plotted -> 925-1550), then 75 ms of extra floor to hold the
                # stretch-factor row: without it the label lands on the Fast
                # easy point, which sits near the bottom of this range.
                ax.set_ylim(
                    np.floor(low / 25.0) * 25.0 - 75.0,
                    np.ceil(high / 25.0) * 25.0,
                )
                ax.set_yticks([1000, 1250, 1500])
            rows = cell_stats.loc[cell_stats["measure"] == stat_measure]
            if index == 0:
                # Lower right, lifted clear of the stretch-factor row below it.
                annotate_anova(ax, rows, loc="lower right", y=0.16)
            else:
                annotate_anova(ax, rows, loc="lower left")

        if len(factors) == len(style.CLASS_ORDER) and np.all(np.isfinite(factors)):
            # One factor per category, sitting under its own x position -- a
            # single run of numbers left the reader to guess the mapping.
            ax_a = axes[0, 0]
            for position, value in enumerate(factors):
                ax_a.text(
                    position, 0.015, f"{value:.2f}",
                    transform=ax_a.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=11, color=style.INK_SECONDARY,
                )
            # Label sits on its own line above the row: at this x-range there
            # is no room beside the first category without overprinting it.
            ax_a.text(
                0.02, 0.075, "Slow ÷ Fast", transform=ax_a.transAxes,
                ha="left", va="bottom", fontsize=11, color=style.INK_SECONDARY,
            )

        # fontsize matches F04-F06. Placement does not: those three use
        # "lower right", which here is exactly where misleading accuracy
        # lands (~0.37). Upper right is the corner that is actually empty for
        # this data shape, which is the rule those figures were following.
        axes[0, 1].legend(loc="upper right", fontsize=12)
        fig.suptitle("Decision time and accuracy by condition and difficulty")
        # Trim the default outer margins; the panels carry their own padding.
        engine = fig.get_layout_engine()
        if engine is not None:
            engine.set(w_pad=0.01, h_pad=0.01, wspace=0.02, hspace=0.02)

    metadata = {
        "kind": "factorial_interaction",
        "title": "Condition x difficulty",
        "columns_read": {
            "conditionclass": ["subject", "condition", "trial_class_name", "mean_dt_ms", "accuracy"],
            "conditionclassstats": ["measure", "effect", "F", "df_effect", "df_error", "p", "partial_eta_squared", "n_subjects"],
        },
        "statistics_source": "conditionclassstats",
        "stretch_factors": {
            class_name: factor for class_name, factor in zip(style.CLASS_ORDER, factors)
        },
        "palette": dict(style.CONDITION_COLORS),
    }
    return fig, metadata


def build_choiceside_asymmetry(tables: BehaviorTableSet) -> tuple[Figure, dict[str, Any]]:
    """F09: left-minus-right asymmetry in choice, decision time, and accuracy.

    Three difference panels, not a 3x3 grid of left-vs-right slopes. Two
    reasons. `proportion_left_choices` and `proportion_right_choices` sum to
    one by construction, so drawing both produced a guaranteed mirror image
    crossing at 0.5 -- one number plotted as two, and the apparent "effect"
    was an artefact of the complement. And the paired difference is the
    quantity every test here operates on, so plotting it directly puts the
    data and the statistic on the same axis.
    """
    summary = tables.analysis("choiceside")
    stats_table = tables.analysis("choicesidestats")

    # Fixed limits per measure, with the ticks the panel should carry. A
    # handful of subjects fall outside; they are drawn as open circles on the
    # limit line rather than silently dropped (4 of 96 points).
    measures = (
        ("choice_proportion", "proportion_left_choices", "proportion_right_choices",
         "Δ choice proportion", (-0.15, 0.25), [-0.1, 0.0, 0.1, 0.2], 2, ""),
        ("decision_time", "mean_left_dt_ms", "mean_right_dt_ms",
         "Δ decision time (ms)", (-212.0, 122.0), [-200, -100, 0, 100], 1, " ms"),
        ("accuracy", "accuracy_left", "accuracy_right",
         "Δ accuracy", (-0.15, 0.165), [-0.15, -0.05, 0.05, 0.15], 2, ""),
    )
    conditions = ("all", "fast", "slow")

    with style.apply_publication_style():
        fig, axes = style.figure_grid(1, 3, width="double", panel_height_in=3.4)
        for index, (measure, column_a, column_b, ylabel, limits, ticks, precision, unit) in enumerate(measures):
            ax = axes[0, index]
            style.panel_label(ax, "ABC"[index])
            groups = {}
            for condition in conditions:
                rows = summary.loc[summary["condition"] == condition]
                groups[condition] = (rows[column_a] - rows[column_b]).to_numpy(dtype=float)
            subject_strip(
                ax, groups=groups,
                colors={"all": style.INK, "fast": style.CONDITION_COLORS["fast"],
                        "slow": style.CONDITION_COLORS["slow"]},
                reference=0.0, ylabel=ylabel,
            )
            ax.set_xlim(-0.45, 2.45)

            bottom, top = limits
            ax.set_ylim(bottom, top)
            ax.set_yticks(ticks)

            colours = {"all": style.INK, "fast": style.CONDITION_COLORS["fast"],
                       "slow": style.CONDITION_COLORS["slow"]}
            for position, condition in enumerate(conditions):
                values = groups[condition]
                values = values[np.isfinite(values)]
                # Out-of-range subjects: open circles sitting on the limit, so
                # the panel never implies a tighter spread than the data has.
                for edge, outside in ((bottom, values[values < bottom]),
                                      (top, values[values > top])):
                    if not outside.size:
                        continue
                    offsets = np.linspace(-0.05, 0.05, outside.size)
                    ax.plot(
                        position + offsets, np.full(outside.size, edge),
                        linestyle="none", marker="o", markersize=5,
                        markerfacecolor=style.SURFACE,
                        markeredgecolor=colours[condition], markeredgewidth=1.0,
                        clip_on=False, zorder=5,
                    )
                    # An out-of-range marker is only useful with its value.
                    for offset, value in zip(offsets, np.sort(outside)):
                        ax.text(
                            position + offset + 0.10, edge, f"{value:+.{precision}f}{unit}",
                            ha="left", va="center", fontsize=9,
                            color=colours[condition], clip_on=False, zorder=5,
                        )
                # Significance marker, inset from the top so it clears any
                # open circle sitting on the ceiling.
                row = stats_table.loc[
                    (stats_table["measure"] == measure) & (stats_table["condition"] == condition)
                ]
                if not len(row):
                    continue
                ax.text(
                    position, 1.02, significance_marker(row.iloc[0]["p"]) or "n.s.",
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=12, color=style.INK,
                )

        fig.suptitle("Left − right asymmetry")

    metadata = {
        "kind": "asymmetry_difference",
        "title": "Left/right choice, DT, and accuracy asymmetry",
        "columns_read": {
            "choiceside": [
                "subject", "condition", "proportion_left_choices", "proportion_right_choices",
                "mean_left_dt_ms", "mean_right_dt_ms", "accuracy_left", "accuracy_right",
            ],
            "choicesidestats": ["measure", "condition", "mean_difference", "t", "p", "df", "cohens_dz"],
        },
        "statistics_source": "choicesidestats",
        "caveat": (
            "Choice proportion left and right are complements, so only the "
            "difference is informative; the left-minus-right form is plotted "
            "for all three measures so the panels read the same way."
        ),
    }
    return fig, metadata


def _drift_lines(stats_table: pd.DataFrame, term: str, unit: str) -> list[str]:
    """Compact 'value marker' lines for one time-on-task term.

    Spelled-out stat text for two conditions does not fit a half-width panel;
    t, df and p stay in `timeontaskstats` and the JSON sidecar.
    """
    lines = []
    for condition, prefix in (("all", ""), ("fast_vs_slow", "Fast − Slow  ")):
        row = stats_table.loc[
            (stats_table["term"] == term) & (stats_table["condition"] == condition)
        ]
        if not len(row):
            continue
        r = row.iloc[0]
        value = r["mean"] if np.isfinite(r["mean"]) else r["mean_difference"]
        marker = significance_marker(r["p"]) or "n.s."
        # Unit spelled out on the slope line only; the contrast shares it.
        suffix = f" ms per {unit}" if condition == "all" else ""
        lines.append(f"{prefix}{value:+.2f}{suffix}  {marker}")
    return lines


def build_timeontask_drift(tables: BehaviorTableSet) -> tuple[Figure, dict[str, Any]]:
    """F10: session (block-order) and within-block drift. Data before coefficient.

    A and B show group means with within-subject 95% CI rather than subject
    traces: the traces spanned ~500-2300 ms and buried both the trend and the
    band, and between-subject spread is not what the paired tests use.
    B's deciles are now taken *within each block*; they were previously
    binned on `run_trial_index` pooled across a subject's blocks, which mixes
    positions because blocks run 50-85 trials.
    """
    trial_features = tables.trial_features()
    timeontaskstats = tables.analysis("timeontaskstats")

    trials = trial_features.copy()
    trials["block_position"] = _session_block_order(trials)
    trials["decile"] = trials.groupby(["subject", "block_position"])["run_trial_index"].transform(
        lambda values: pd.qcut(values.rank(method="first"), 10, labels=False, duplicates="drop")
    )

    with style.apply_publication_style():
        fig, axes = style.figure_grid(1, 2, width="double", panel_height_in=3.4)
        for index, letter in enumerate("AB"):
            style.panel_label(axes[0, index], letter)

        # Trial class is balanced across session blocks (25.1/21.4/10.5% in
        # every one) but NOT across within-block positions: the first decile
        # is 29% easy with almost no ambiguous trials, the second the reverse.
        # Easy trials are ~450 ms faster, so part of B's profile is the
        # schedule rather than the subject. The class-adjusted line removes
        # each subject's per-class mean and adds back their grand mean, so
        # the gap between the two lines *is* the compositional contribution.
        trials["dt_class_adjusted"] = (
            trials["dt_ms"]
            - trials.groupby(["subject", "trial_class_name"])["dt_ms"].transform("mean")
            + trials.groupby("subject")["dt_ms"].transform("mean")
        )

        # A: session drift split by condition. Every subject runs exactly four
        # Fast and four Slow blocks, so indexing by the n-th block *of that
        # condition* keeps all 32 subjects at every point -- a within-subject
        # comparison, unlike splitting on absolute session position where each
        # point would average a different half of the cohort. It also shows
        # the Fast/Slow gap holding at every session stage, which is the
        # confound this panel exists to rule out.
        trials["nth_block"] = trials.groupby(["subject", "condition"])["block_position"].transform(
            lambda values: values.rank(method="dense")
        )
        ax_a = axes[0, 0]
        for condition in style.CONDITION_ORDER:
            subset = trials.loc[trials["condition"] == condition]
            means = subset.pivot_table(index="subject", columns="nth_block", values="dt_ms", aggfunc="mean")
            matrix = means.to_numpy(dtype=float)
            x = means.columns.to_numpy(dtype=float)
            group_line(ax_a, x=x, subject_matrix=matrix,
                       color=style.CONDITION_COLORS[condition], label=condition,
                       error="within_ci95")
        # Fixed limits. The ceiling clears the top tick rather than sitting on
        # it -- at 1500 the label collided with the panel letter -- and also
        # uncovers the top of the Slow band at the first block (~1510).
        ax_a.set_ylim(1050, 1525)
        ax_a.set_yticks([1100, 1300, 1500])
        ax_a.set_xlabel("n-th block")
        ax_a.set_ylabel("Mean decision time (ms)")
        ax_a.set_xticks(x)
        ax_a.legend(loc="upper right", fontsize=12)
        # Session coefficient annotated here rather than given its own strip
        # panel. Lower left is empty: both lines start high and fall away.
        annotate_stat_block(
            ax_a,
            lines=_drift_lines(timeontaskstats, "dt_per_block_ms", "block"),
            loc="lower left",
        )

        # B: observed against class-adjusted, not split by condition. The
        # per-condition decile profiles do not replicate each other (they peak
        # at different deciles), so splitting would show two noisy sawtooths
        # and lose the comparison that matters here -- how much of the profile
        # is trial scheduling.
        ax_b = axes[0, 1]
        for value_column, label, colour, linestyle in (
            ("dt_ms", "observed", style.INK, "-"),
            ("dt_class_adjusted", "class-adjusted", style.INK_SECONDARY, "--"),
        ):
            means = trials.pivot_table(index="subject", columns="decile", values=value_column, aggfunc="mean")
            matrix = means.to_numpy(dtype=float)
            x = means.columns.to_numpy(dtype=float) + 1
            group_line(ax_b, x=x, subject_matrix=matrix, color=colour,
                       label=label, error="within_ci95")
            if linestyle != "-":
                ax_b.get_lines()[-1].set_linestyle(linestyle)
        # Fixed limits: headroom for the upper-left annotation, and the
        # ceiling clears the 1400 tick so it does not sit under the panel
        # letter.
        ax_b.set_ylim(1050, 1435)
        ax_b.set_yticks([1100, 1200, 1300, 1400])
        ax_b.set_xticks([1, 5, 10])
        ax_b.set_xlabel("Block decile")
        ax_b.set_ylabel("Mean decision time (ms)")
        ax_b.legend(loc="lower right", fontsize=12)
        # The within-block coefficient and its Fast/Slow contrast, annotated
        # here rather than given their own strip panel: the only thing the
        # strip added beyond these two numbers was per-subject spread, which
        # panel C already demonstrates for the session term.
        annotate_stat_block(
            ax_b,
            lines=_drift_lines(timeontaskstats, "dt_per_within_block_trial_ms", "trial"),
            loc="upper left",
        )

        fig.suptitle("Session drift and within-block drift")

    metadata = {
        "kind": "trend_and_coefficient",
        "title": "Session and within-block drift",
        "columns_read": {
            "trialfeatures": ["subject", "condition", "run_trial_index", "initial_time_ms", "dt_ms"],
            "timeontaskstats": ["term", "condition", "mean", "sem", "t", "p", "df", "cohens_dz"],
        },
        "statistics_source": "timeontaskstats",
    }
    return fig, metadata


def build_conditionorder_balance(tables: BehaviorTableSet) -> tuple[Figure, dict[str, Any]]:
    """F11: the Slow-Fast effect split by which condition a subject started on.

    Reframed from a between-group balance test to a within-group one. The
    between-group Welch test (p = .33) was annotated as though it showed
    balance; at n = 15 vs 17 it can only exclude an order effect larger than
    ~116 ms, which is about the size of the whole Slow-Fast effect, so it
    demonstrates nothing either way. What the design actually needs is that
    the effect holds in *both* order groups, which is directly testable and
    does hold. The between-group contrast is kept, bounded rather than
    asserted.
    """
    order = tables.analysis("conditionorder")
    order_stats = tables.analysis("conditionorderstats")

    with style.apply_publication_style():
        # Two categories only, so width is set by the title and the y-axis
        # furniture rather than by the data. Short title plus label and
        # ticks stepped down from the rc defaults keep the panel at 2.9in
        # with no dead gap left or right.
        fig, axes = style.figure_grid(1, 1, width=2.9, panel_height_in=3.8)
        ax = axes[0, 0]
        groups = {
            label: order.loc[order["first_condition"] == label, "slow_minus_fast_dt_ms"].to_numpy(dtype=float)
            for label in ("fast", "slow")
        }
        subject_strip(
            ax, groups=groups,
            colors={"fast": style.CONDITION_COLORS["fast"], "slow": style.CONDITION_COLORS["slow"]},
            reference=0.0, ylabel="Slow − Fast (ms)",
        )
        ax.set_xticklabels(
            [f"Fast-first\n(n = {np.isfinite(groups['fast']).sum()})",
             f"Slow-first\n(n = {np.isfinite(groups['slow']).sum()})"]
        )
        # Trim the category margins hard; the jitter only spans about
        # +-0.15, so anything beyond that is dead space at both edges.
        ax.set_xlim(-0.3, 1.3)
        ax.yaxis.label.set_fontsize(13)
        ax.tick_params(axis="y", labelsize=11)

        finite = {k: v[np.isfinite(v)] for k, v in groups.items()}
        low = min(float(v.min()) for v in finite.values())
        high = max(float(v.max()) for v in finite.values())
        span = high - low

        # Per-group "vs 0" markers, both at one shared height so they read as
        # the same test rather than as two heights meaning something.
        marker_y = high + 0.05 * span
        within = {}
        for position, label in enumerate(("fast", "slow")):
            values = finite[label]
            result = ttest_1samp(values, 0.0)
            within[label] = {"mean": float(values.mean()), "p": float(result.pvalue)}
            ax.text(
                position, marker_y,
                significance_marker(result.pvalue) or "n.s.",
                ha="center", va="bottom", fontsize=13, color=style.INK,
            )

        # Between-group contrast as a bracket spanning the two categories --
        # the same 'Δ = value marker' form as F04/F05, and it occupies the gap
        # the two-category layout leaves down the middle. Drawn inline rather
        # than via significance_bracket(), which scales y by 0.98/1.02 and so
        # misplaces itself on an axis that crosses zero.
        row = order_stats.loc[order_stats["measure"] == "slow_minus_fast_dt_ms"]
        if len(row):
            r = row.iloc[0]
            bracket_y = high + 0.24 * span
            ax.plot(
                [0, 0, 1, 1],
                [bracket_y - 0.02 * span, bracket_y, bracket_y, bracket_y - 0.02 * span],
                color=style.INK, linewidth=0.8,
            )
            ax.text(
                0.5, bracket_y + 0.01 * span,
                f"Δ = {abs(r['mean_difference']):.0f} ms {significance_marker(r['p']) or 'n.s.'}",
                ha="center", va="bottom", fontsize=12, color=style.INK,
            )
            ax.set_ylim(low - 0.08 * span, bracket_y + 0.13 * span)
            ax.set_yticks([-100, 200, 500])
        fig.suptitle("Order check")

    metadata = {
        "kind": "between_subject_balance_check",
        "title": "First-block counterbalancing",
        "columns_read": {
            "conditionorder": ["subject", "first_condition", "slow_minus_fast_dt_ms"],
            "conditionorderstats": ["measure", "label_a", "label_b", "n_a", "n_b", "mean_a", "mean_b", "t", "p", "df"],
        },
        "statistics_source": "conditionorderstats",
        "within_group_vs_zero": within,
        "caveat": (
            "Between-group comparison is underpowered (n = 15 vs 17): it excludes "
            "only order effects larger than ~116 ms, comparable to the Slow-Fast "
            "effect itself. Read the within-group tests, not the between-group null."
        ),
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
    so bars are correct here -- one of the few figures in the set where they are.

    Four panels sharing one subject axis, labelled once on the left. The
    earlier version drew the subject list three times at fontsize 5, plotted
    each condition's total and its DT-usable subset as overlapping bars (they
    differ by 13 trials in 16,337, so the overlap conveyed nothing), and put
    motor baseline and accuracy on a shared twinned x-axis in two different
    scopes' colours.
    """
    summary = tables.summary()

    with style.apply_publication_style():
        fig, axes = style.figure_grid(
            1, 4, width="double", panel_height_in=5.4, width_ratios=[1.2, 1.2, 1.0, 1.0]
        )
        for index in range(4):
            style.panel_label(axes[0, index], "ABCD"[index])

        subjects = summary.sort_values("subject")["subject"].tolist()
        y_positions = np.arange(len(subjects))
        ordered = summary.set_index("subject").loc[subjects]
        dt_trials = (ordered["n_fast_dt_trials"] + ordered["n_slow_dt_trials"]).to_numpy(dtype=float)

        # A -- trials by condition, stacked so the bar length is the subject's
        # total and the split is still readable.
        ax_a = axes[0, 0]
        left = np.zeros(len(subjects))
        for condition in style.CONDITION_ORDER:
            values = ordered[f"n_{condition}_dt_trials"].to_numpy(dtype=float)
            ax_a.barh(
                y_positions, values, left=left, height=0.72,
                color=style.CONDITION_COLORS[condition], label=condition,
            )
            left += values
        ax_a.set_xlabel("Trials")

        # B -- the same totals split by difficulty. Unclassified is drawn, not
        # dropped: it is 43% of trials, and omitting it made B's bars look
        # comparable to A's while covering little more than half the data.
        ax_b = axes[0, 1]
        left = np.zeros(len(subjects))
        for trial_class in style.CLASS_ORDER:
            values = ordered[f"n_{trial_class}_trials"].to_numpy(dtype=float)
            ax_b.barh(
                y_positions, values, left=left, height=0.72,
                color=style.CLASS_COLORS[trial_class], label=trial_class,
            )
            left += values
        class_totals = left.copy()
        ax_b.barh(
            y_positions, dt_trials - class_totals, left=class_totals, height=0.72,
            color=style.SUBJECT_LINE, label="unclassified",
        )
        ax_b.set_xlabel("Trials")

        # C, D -- one quantity per axis, in ink. The previous twinned x-axis
        # implied a relationship between two unrelated per-subject measures,
        # and borrowed the model palette for an accuracy series.
        for index, (column, label, precision) in enumerate(
            (
                ("motor_baseline_ms", "Motor baseline (ms)", 0),
                ("percent_correct", "Accuracy (%)", 1),
            ),
            start=2,
        ):
            ax = axes[0, index]
            values = ordered[column].to_numpy(dtype=float)
            centre = float(np.nanmean(values))
            # Bars anchored at the cohort mean, not at zero. Neither quantity
            # has a meaningful zero -- from zero every accuracy bar would be
            # ~90% of full length and the spread would vanish, and a bar on a
            # truncated axis misreads as a proportion. Anchored at the mean
            # the zero point is real, the length is the deviation, and the
            # bar end still lands on the subject's value on an absolute axis.
            ax.barh(
                y_positions, values - centre, left=centre, height=0.72,
                color=style.INK_SECONDARY,
            )
            ax.axvline(centre, color=style.INK, linewidth=0.8)
            ax.text(
                centre, 1.0, f" mean {centre:.{precision}f}", transform=ax.get_xaxis_transform(),
                ha="left", va="bottom", fontsize=9, color=style.INK_SECONDARY,
            )
            # Smaller than the rc default: at this panel width the two labels
            # run into each other at 16pt.
            ax.set_xlabel(label, fontsize=12)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))

        for index in range(4):
            ax = axes[0, index]
            ax.set_ylim(-0.8, len(subjects) - 0.2)
            ax.set_yticks(y_positions)
            # Subject labels once, on the leftmost panel only.
            ax.set_yticklabels(subjects if index == 0 else [], fontsize=7)
        # One figure-level legend in a single row, below everything. Per-axes
        # legends sat beside the "Trials" labels and crowded them; "outside
        # lower center" makes constrained_layout reserve its own strip under
        # the x-axis labels instead.
        handles, labels = [], []
        for ax in (axes[0, 0], axes[0, 1]):
            ax_handles, ax_labels = ax.get_legend_handles_labels()
            handles.extend(ax_handles)
            labels.extend(ax_labels)
        fig.legend(
            handles, labels, loc="outside lower center", ncol=len(labels),
            fontsize=10, frameon=False, handlelength=1.2,
            columnspacing=1.2, handletextpad=0.5,
        )

        # Retention belongs in the caveat, not the title: spelled out here the
        # suptitle is wider than the figure and clips at both ends.
        retained = 100.0 * dt_trials.sum() / float(
            (ordered["n_fast_trials"] + ordered["n_slow_trials"]).sum()
        )
        fig.suptitle(
            f"Cohort overview ({len(subjects)} subjects, "
            f"{int(dt_trials.sum()):,} task trials)"
        )

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
        "dt_retention_percent": round(retained, 2),
        "unclassified_percent": round(
            100.0 * float(dt_trials.sum() - class_totals.sum()) / float(dt_trials.sum()), 1
        ),
        "palette": {**style.CLASS_COLORS, "unclassified": style.SUBJECT_LINE},
    }
    return fig, metadata
