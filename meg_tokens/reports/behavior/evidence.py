"""Evidence and criterion figures: F14-F18."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from meg_tokens.reports import style
from meg_tokens.reports.annotations import annotate_stat_block, format_stat, stat_from_row
from meg_tokens.reports.behavior._tables import BehaviorTableSet
from meg_tokens.reports.panels import group_line, subject_strip

_DISCRETIZATION_CAVEAT = (
    "Success probability moves in larger steps at later jumps, so a subject "
    "holding a fixed criterion overshoots it more the later they commit -- "
    "part of the positive slope is this discretization, not a rising "
    "criterion (see docs/behavior_roadmap_results.md, B1-B2)."
)


def _fitted_criterion_figure(
    tables: BehaviorTableSet,
    *,
    analysis_name: str,
    stats_name: str,
    predictor_value: str,
    predictor_column: str,
    predictor_label: str,
    predictor_scale: float,
    title: str,
) -> tuple[Figure, dict[str, Any]]:
    """Shared construction for F14 (tokens) and F15 (seconds): per-subject
    fitted criterion lines + binned observed overlay, plus a fitted-slope
    strip vs. zero. Two parameterisations of the same criterion; kept as
    separate figures because they read separate derivatives, and their
    near-redundancy (r=0.98) is F23's finding, not an assumption here."""
    fits = tables.analysis(analysis_name)
    stats_table = tables.analysis(stats_name)
    trial_features = tables.trial_features()

    responses = ("logged_spd", "logged_spd_log_odds")
    response_labels = {"logged_spd": "Success probability", "logged_spd_log_odds": "Log odds"}
    condition_colors = {"all": style.INK, "fast": style.CONDITION_COLORS["fast"], "slow": style.CONDITION_COLORS["slow"]}

    with style.apply_publication_style():
        fig, axes = style.figure_grid(2, 2, width="double")
        for row_index, response in enumerate(responses):
            response_fits = fits.loc[(fits["predictor"] == predictor_value) & (fits["response"] == response)]
            pooled_fits = response_fits.loc[response_fits["condition"] == "all"]

            ax_left = axes[row_index, 0]
            if row_index == 0:
                style.panel_label(ax_left, "A")
            x_max = float(pd.to_numeric(trial_features[predictor_column], errors="coerce").max() or 1.0)
            x_max = x_max / predictor_scale if predictor_column == "dt_ms" else x_max
            x_grid = np.linspace(0, x_max, 50)
            for _, row in pooled_fits.iterrows():
                if not row["converged"]:
                    continue
                y = row["intercept"] + row["slope"] * x_grid
                ax_left.plot(x_grid, y, color=style.SUBJECT_LINE, alpha=style.SUBJECT_ALPHA, linewidth=0.7)
            mean_intercept = pooled_fits.loc[pooled_fits["converged"], "intercept"].mean()
            mean_slope = pooled_fits.loc[pooled_fits["converged"], "slope"].mean()
            if np.isfinite(mean_intercept) and np.isfinite(mean_slope):
                ax_left.plot(x_grid, mean_intercept + mean_slope * x_grid, color=style.INK, linewidth=1.8)
            # Binned observed overlay.
            x_values = pd.to_numeric(trial_features[predictor_column], errors="coerce") / predictor_scale
            y_values = pd.to_numeric(trial_features[response], errors="coerce")
            bins = pd.cut(x_values, bins=15)
            binned = pd.DataFrame({"x": x_values, "y": y_values, "bin": bins}).groupby("bin", observed=True).agg(
                x_mean=("x", "mean"), y_mean=("y", "mean"), y_sem=("y", lambda v: v.std(ddof=1) / np.sqrt(v.count()) if v.count() > 1 else np.nan)
            )
            ax_left.errorbar(binned["x_mean"], binned["y_mean"], yerr=binned["y_sem"], fmt="o", color=style.OBSERVED, markersize=3, capsize=1.5, alpha=0.8, zorder=3)
            ax_left.set_xlabel(predictor_label)
            ax_left.set_ylabel(response_labels[response])

            ax_right = axes[row_index, 1]
            if row_index == 0:
                style.panel_label(ax_right, "B")
            groups = {
                condition: response_fits.loc[response_fits["condition"] == condition, "slope"].to_numpy(dtype=float)
                for condition in ("all", "fast", "slow")
            }
            subject_strip(ax_right, groups=groups, colors=condition_colors, reference=0.0, ylabel=f"slope ({response_labels[response]} per {predictor_label.split(' ')[0].lower()})")
            all_row = stats_table.loc[
                (stats_table["predictor"] == predictor_value) & (stats_table["response"] == response)
                & (stats_table["term"] == "slope") & (stats_table["condition"] == "all")
            ]
            fvs_row = stats_table.loc[
                (stats_table["predictor"] == predictor_value) & (stats_table["response"] == response)
                & (stats_table["term"] == "slope") & (stats_table["condition"] == "fast_vs_slow")
            ]
            lines = []
            if len(all_row):
                lines.append(format_stat(stat_from_row(all_row.iloc[0], label="all")))
            if len(fvs_row):
                lines.append("fast vs slow: " + format_stat(stat_from_row(fvs_row.iloc[0], label="fvs"), include_effect_size=False))
            annotate_stat_block(ax_right, lines=lines, loc="upper left")

        fig.suptitle(title)

    metadata = {
        "kind": "fitted_criterion",
        "title": title,
        "columns_read": {
            analysis_name: ["subject", "condition", "predictor", "response", "intercept", "slope", "slope_se", "converged"],
            stats_name: ["predictor", "response", "term", "condition", "mean", "sem", "t", "p", "df", "cohens_dz"],
            "trialfeatures": [predictor_column, "logged_spd", "logged_spd_log_odds"],
        },
        "statistics_source": f"{stats_name}[predictor={predictor_value}]",
        "palette": dict(condition_colors),
        "caveat": _DISCRETIZATION_CAVEAT,
    }
    return fig, metadata


def build_criteriondecline_tokens(tables: BehaviorTableSet) -> tuple[Figure, dict[str, Any]]:
    """F14: evidence at decision vs. tokens observed."""
    return _fitted_criterion_figure(
        tables,
        analysis_name="criteriondecline", stats_name="criteriondeclinestats",
        predictor_value="decision_token_index", predictor_column="decision_token_index",
        predictor_label="Tokens observed", predictor_scale=1.0,
        title="Evidence at decision vs. tokens observed",
    )


def build_urgency_decisiontime(tables: BehaviorTableSet) -> tuple[Figure, dict[str, Any]]:
    """F15: evidence at decision vs. decision time (urgencystats carries
    analysis == 'criterion_decline' -- filtered here on predictor, not
    analysis)."""
    return _fitted_criterion_figure(
        tables,
        analysis_name="urgency", stats_name="urgencystats",
        predictor_value="dt_ms", predictor_column="dt_ms",
        predictor_label="Decision time (s)", predictor_scale=1000.0,
        title="Evidence at decision vs. decision time",
    )


def build_reversecorrelation_kernel(tables: BehaviorTableSet) -> tuple[Figure, dict[str, Any]]:
    """F16: psychophysical kernel by token jump."""
    kernels = tables.analysis("reversecorrelation")
    kernel_stats = tables.analysis("reversecorrelationstats")

    weight_stats = kernel_stats.loc[kernel_stats["metric"] == "logistic_weight"]
    jumps = sorted(kernels["jump"].unique())

    with style.apply_publication_style():
        fig, axes = style.figure_grid(1, 2, width="double")
        ax_a, ax_b = axes[0, 0], axes[0, 1]
        style.panel_label(ax_a, "A")
        style.panel_label(ax_b, "B")

        pooled = kernels.loc[kernels["condition"] == "all"]
        pooled_matrix = pooled.pivot(index="subject", columns="jump", values="logistic_weight")[jumps].to_numpy(dtype=float)
        group_line(ax_a, x=np.array(jumps, dtype=float), subject_matrix=pooled_matrix, color=style.INK, label="all", draw_subjects=True)
        for condition in style.CONDITION_ORDER:
            group = kernels.loc[kernels["condition"] == condition]
            matrix = group.pivot(index="subject", columns="jump", values="logistic_weight")[jumps].to_numpy(dtype=float)
            group_line(ax_a, x=np.array(jumps, dtype=float), subject_matrix=matrix, color=style.CONDITION_COLORS[condition], label=condition)
        ax_a.set_xlabel("Token jump")
        ax_a.set_ylabel("Logistic weight")
        ax_a.legend(loc="upper right", fontsize=6)
        pooled_stats = weight_stats.loc[weight_stats["condition"] == "all"]
        all_significant = bool((pooled_stats["p"] < 0.001).all()) if len(pooled_stats) else False
        ax_a.text(
            0.02, 0.02, "every weight differs from zero (all p < .001)" if all_significant else "see stats table",
            transform=ax_a.transAxes, fontsize=6, color=style.INK_SECONDARY,
        )

        # fast_vs_slow rows come from paired_subject_statistics, which writes
        # mean_difference/sem_a/sem_b, not mean/sem -- go through
        # stat_from_row (mean falls back to mean_difference) rather than
        # reading those columns directly, which would silently read all-NaN.
        diff_rows = weight_stats.loc[weight_stats["condition"] == "fast_vs_slow"].sort_values("jump")
        diff_results = [stat_from_row(row, label=str(row["jump"])) for _, row in diff_rows.iterrows()]
        diff_jumps = diff_rows["jump"].to_numpy(dtype=float)
        diff_means = np.array([r.mean if r.mean is not None else np.nan for r in diff_results])
        diff_sems = np.array([
            (r.mean / r.t) if (r.t not in (None, 0) and r.mean is not None) else np.nan
            for r in diff_results
        ])
        ax_b.axhline(0, color=style.INK_MUTED, linewidth=0.6)
        ax_b.errorbar(diff_jumps, diff_means, yerr=np.abs(diff_sems), color=style.INK, marker="o", markersize=4, linewidth=1.4, capsize=2)
        for jump, mean, result in zip(diff_jumps, diff_means, diff_results):
            if result.p is not None and result.p < 0.05:
                ax_b.scatter([jump], [mean], s=60, facecolors="none", edgecolors=style.INK, zorder=4)
        ax_b.set_xlabel("Token jump")
        ax_b.set_ylabel("Fast − Slow logistic weight")

        fig.suptitle("Psychophysical kernel: primacy and Fast/Slow reweighting")

    metadata = {
        "kind": "kernel_by_jump",
        "title": "Psychophysical kernel",
        "columns_read": {
            "reversecorrelation": ["subject", "condition", "jump", "logistic_weight", "converged"],
            "reversecorrelationstats": ["metric", "condition", "jump", "mean", "sem", "t", "p", "df"],
        },
        "statistics_source": "reversecorrelationstats[metric=logistic_weight]",
        "palette": dict(style.CONDITION_COLORS),
    }
    return fig, metadata


def build_conditionalaccuracy_caf(tables: BehaviorTableSet) -> tuple[Figure, dict[str, Any]]:
    """F17: conditional accuracy function, plotted against bin RT (not bin
    index), which also shows the unequal bin spacing."""
    functions = tables.analysis("conditionalaccuracy")
    caf_stats = tables.analysis("conditionalaccuracystats")

    with style.apply_publication_style():
        fig, axes = style.figure_grid(1, 1, width="single")
        ax = axes[0, 0]
        for condition in ("all", "fast", "slow"):
            group = functions.loc[functions["condition"] == condition]
            bins = sorted(group["dt_bin"].unique())
            x_matrix = group.pivot(index="subject", columns="dt_bin", values="mean_dt_ms")[bins]
            y_matrix = group.pivot(index="subject", columns="dt_bin", values="accuracy")[bins]
            x_mean = x_matrix.mean(axis=0).to_numpy(dtype=float)
            color = style.INK if condition == "all" else style.CONDITION_COLORS[condition]
            group_line(ax, x=x_mean, subject_matrix=y_matrix.to_numpy(dtype=float), color=color, label=condition, draw_subjects=(condition == "all"))
        ax.set_xlabel("Bin mean decision time (ms)")
        ax.set_ylabel("Accuracy")
        ax.legend(loc="lower left", fontsize=6.5)

        slope_rows = caf_stats.loc[caf_stats["test"] == "accuracy_slope_across_bins"]
        lines = []
        for condition in ("all", "fast", "slow"):
            row = slope_rows.loc[slope_rows["condition"] == condition]
            if len(row):
                lines.append(f"{condition}: " + format_stat(stat_from_row(row.iloc[0], label=condition), include_effect_size=False))
        annotate_stat_block(ax, lines=lines, loc="upper right")
        ax.text(
            0.02, 0.02, "slow decisions are not more accurate", transform=ax.transAxes,
            fontsize=6.5, color=style.INK_SECONDARY,
        )

        fig.suptitle("Conditional accuracy function")

    metadata = {
        "kind": "conditional_accuracy_function",
        "title": "Conditional accuracy function",
        "columns_read": {
            "conditionalaccuracy": ["subject", "condition", "dt_bin", "mean_dt_ms", "accuracy"],
            "conditionalaccuracystats": ["condition", "dt_bin", "test", "mean_dt_ms", "mean", "sem", "t", "p", "df"],
        },
        "statistics_source": "conditionalaccuracystats[test=accuracy_slope_across_bins]",
        "palette": dict(style.CONDITION_COLORS),
    }
    return fig, metadata


def build_continuousevidence_effects(tables: BehaviorTableSet) -> tuple[Figure, dict[str, Any]]:
    """F18: continuous early evidence, over EVERY task trial including the
    43% unclassified (the one figure in the set where that's the point)."""
    trial_features = tables.trial_features()
    fits = tables.analysis("continuousevidence")
    stats_table = tables.analysis("continuousevidencestats")

    predictor = "sum_log_lr_design_early"

    with style.apply_publication_style():
        fig, axes = style.figure_grid(2, 2, width="double")
        style.panel_label(axes[0, 0], "A")
        style.panel_label(axes[0, 1], "B")
        style.panel_label(axes[1, 0], "C")
        style.panel_label(axes[1, 1], "D")

        evidence = pd.to_numeric(trial_features[predictor], errors="coerce")
        dt_values = pd.to_numeric(trial_features["dt_ms"], errors="coerce")
        correct = trial_features["isCorrect"].astype("boolean")

        strength_bins = pd.qcut(evidence.abs(), 10, labels=False, duplicates="drop")
        binned_dt = pd.DataFrame({"strength": evidence.abs(), "dt": dt_values, "bin": strength_bins}).groupby("bin", observed=True).agg(x=("strength", "mean"), y=("dt", "mean"))
        axes[0, 0].plot(binned_dt["x"], binned_dt["y"], color=style.INK, marker="o", markersize=3, linewidth=1.4)
        axes[0, 0].set_xlabel("|evidence strength| (SumLogLR)")
        axes[0, 0].set_ylabel("Mean dt_ms (ms)")

        signed_bins = pd.qcut(evidence, 10, labels=False, duplicates="drop")
        binned_acc = pd.DataFrame({"signed": evidence, "correct": correct.astype(float), "bin": signed_bins}).groupby("bin", observed=True).agg(x=("signed", "mean"), y=("correct", "mean"))
        axes[0, 1].plot(binned_acc["x"], binned_acc["y"], color=style.INK, marker="o", markersize=3, linewidth=1.4)
        axes[0, 1].set_xlabel("Signed evidence (SumLogLR)")
        axes[0, 1].set_ylabel("P(correct)")
        for ax in (axes[0, 0], axes[0, 1]):
            ax.set_title("every task trial, incl. unclassified", fontsize=6.5)

        for ax, term, ylabel in (
            (axes[1, 0], "dt_slope_ms_per_unit", "dt_slope_ms_per_unit"),
            (axes[1, 1], "accuracy_log_odds_per_unit", "accuracy_log_odds_per_unit"),
        ):
            rows = fits.loc[fits["predictor"] == predictor]
            groups = {
                condition: rows.loc[rows["condition"] == condition, term].to_numpy(dtype=float)
                for condition in ("all", "fast", "slow")
            }
            subject_strip(ax, groups=groups, colors={"all": style.INK, "fast": style.CONDITION_COLORS["fast"], "slow": style.CONDITION_COLORS["slow"]}, reference=0.0, ylabel=ylabel)
            all_row = stats_table.loc[(stats_table["predictor"] == predictor) & (stats_table["term"] == term) & (stats_table["condition"] == "all")]
            fvs_row = stats_table.loc[(stats_table["predictor"] == predictor) & (stats_table["term"] == term) & (stats_table["condition"] == "fast_vs_slow")]
            lines = []
            if len(all_row):
                lines.append(format_stat(stat_from_row(all_row.iloc[0], label="all")))
            if len(fvs_row):
                lines.append("fast vs slow: " + format_stat(stat_from_row(fvs_row.iloc[0], label="fvs"), include_effect_size=False))
            annotate_stat_block(ax, lines=lines, loc="upper left")

        fig.suptitle("Continuous early-evidence effects (SumLogLR)")

    metadata = {
        "kind": "continuous_predictor",
        "title": "Continuous early-evidence effects",
        "columns_read": {
            "trialfeatures": ["sum_log_lr_design_early", "dt_ms", "isCorrect"],
            "continuousevidence": ["subject", "condition", "predictor", "dt_slope_ms_per_unit", "accuracy_log_odds_per_unit", "converged"],
            "continuousevidencestats": ["predictor", "term", "condition", "mean", "sem", "t", "p", "df", "cohens_dz"],
        },
        "statistics_source": "continuousevidencestats[predictor=sum_log_lr_design_early]",
        "palette": dict(style.CONDITION_COLORS),
        "includes_unclassified": True,
    }
    return fig, metadata
