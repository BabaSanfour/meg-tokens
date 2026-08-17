"""Evidence and criterion figures."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from meg_tokens.behavior.analyses.evidence import first_order_chosen_sum_log_lr
from meg_tokens.reports import style
from meg_tokens.reports.annotations import (
    annotate_stat_block,
    format_stat,
    significance_marker,
    stat_from_row,
)
from meg_tokens.reports.behavior._tables import BehaviorTableSet
from meg_tokens.reports.panels import group_line, subject_strip

_EXACT_POSTERIOR_CAVEAT = (
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
    """Exact-posterior diagnostic used by F15: per-subject fitted criterion
    lines plus a binned observed overlay and fitted-slope strip vs. zero."""
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
        "caveat": _EXACT_POSTERIOR_CAVEAT,
    }
    return fig, metadata


def build_criteriondecline_sumloglr(
    tables: BehaviorTableSet,
) -> tuple[Figure, dict[str, Any]]:
    """Cisek et al. (2009) SumLogLR against decision time."""
    fits = tables.analysis("criteriondecline")
    statistics = tables.analysis("criteriondeclinestats")
    trial_features = tables.trial_features()
    response = "chosen_sum_log_lr_first_order_at_decision"
    predictor = "dt_ms"

    response_fits = fits.loc[
        (fits["predictor"] == predictor) & (fits["response"] == response)
    ].copy()
    response_statistics = statistics.loc[
        (statistics["predictor"] == predictor)
        & (statistics["response"] == response)
        & (statistics["term"] == "slope")
    ].copy()

    primary_eligible = (
        trial_features["primary_analysis_eligible"]
        .astype("boolean")
        .fillna(False)
    )
    alignment_valid = (
        trial_features["design_time_alignment_valid"]
        .astype("boolean")
        .fillna(False)
    )
    decision_token = pd.to_numeric(
        trial_features["decision_token_index"], errors="coerce"
    )
    post_first_token = decision_token.gt(0).fillna(False)
    eligible = primary_eligible & alignment_valid & post_first_token
    excluded_alignment_trials = int((primary_eligible & ~alignment_valid).sum())
    excluded_pre_token_trials = int(
        (primary_eligible & alignment_valid & ~post_first_token).sum()
    )
    observed = trial_features.loc[
        eligible,
        [
            "subject",
            "condition",
            predictor,
            "decision_token_index",
            "token_lead_at_decision",
            "isCorrect",
        ],
    ].copy()
    observed[response] = first_order_chosen_sum_log_lr(observed)
    observed[predictor] = pd.to_numeric(observed[predictor], errors="coerce")
    observed[response] = pd.to_numeric(observed[response], errors="coerce")
    observed = observed.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[predictor, response]
    )
    post_horizon_trials = int((observed[predictor] > 3000.0).sum())

    observed["decision_time_s"] = observed[predictor] / 1000.0
    observed["display_bin"] = np.floor(observed[predictor] / 200.0).astype(int)
    subject_bin = (
        observed.groupby(["subject", "display_bin"], as_index=False)
        .agg(
            decision_time_s=("decision_time_s", "mean"),
            evidence=(response, "mean"),
        )
    )
    bin_summary = subject_bin.groupby("display_bin", as_index=False).agg(
        decision_time_s=("decision_time_s", "mean"),
        mean=("evidence", "mean"),
        sem=(
            "evidence",
            lambda values: (
                values.std(ddof=1) / np.sqrt(values.count())
                if values.count() > 1
                else np.nan
            ),
        ),
        n_subjects=("subject", "nunique"),
    )
    n_observed_subjects = int(observed["subject"].nunique())
    display_subject_threshold = min(5, max(1, n_observed_subjects))
    supported_bins = bin_summary.loc[
        bin_summary["n_subjects"] >= display_subject_threshold,
        "display_bin",
    ]
    last_display_bin = int(
        supported_bins.max()
        if len(supported_bins)
        else bin_summary["display_bin"].max()
    )
    displayed_bins = bin_summary.loc[
        bin_summary["display_bin"] <= last_display_bin
    ].copy()
    display_end_s = float((last_display_bin + 1) * 0.2)
    beyond_display_trials = int(
        (observed["decision_time_s"] > display_end_s).sum()
    )

    with style.apply_publication_style():
        fig, axes = style.figure_grid(
            1,
            2,
            width="double",
            panel_height_in=3.7,
            width_ratios=(1.4, 0.6),
        )
        ax_data, ax_slopes = axes[0]
        style.panel_label(ax_data, "A")
        style.panel_label(ax_slopes, "B")

        pooled = response_fits.loc[
            (response_fits["condition"] == "all")
            & response_fits["converged"].astype(bool)
        ]
        x_values = np.linspace(0.0, display_end_s, 100)
        for _, row in pooled.iterrows():
            ax_data.plot(
                x_values,
                row["intercept"] + row["slope"] * x_values,
                color=style.SUBJECT_LINE,
                alpha=style.SUBJECT_ALPHA,
                linewidth=0.7,
                zorder=1,
            )

        fitted_values: list[np.ndarray] = []
        for condition, label, color, linewidth in (
            ("all", "All", style.INK, 2.0),
            ("fast", "Fast", style.CONDITION_COLORS["fast"], 1.8),
            ("slow", "Slow", style.CONDITION_COLORS["slow"], 1.8),
        ):
            condition_fits = response_fits.loc[
                (response_fits["condition"] == condition)
                & response_fits["converged"].astype(bool)
            ]
            mean_intercept = condition_fits["intercept"].mean()
            mean_slope = condition_fits["slope"].mean()
            if not (np.isfinite(mean_intercept) and np.isfinite(mean_slope)):
                continue
            fitted = mean_intercept + mean_slope * x_values
            fitted_values.append(fitted)
            ax_data.plot(
                x_values,
                fitted,
                color=color,
                linewidth=linewidth,
                label=label,
                zorder=2,
            )
        ax_data.errorbar(
            displayed_bins["decision_time_s"],
            displayed_bins["mean"],
            yerr=displayed_bins["sem"],
            fmt="o",
            color=style.OBSERVED,
            markersize=4,
            capsize=2,
            linewidth=1.0,
            label="Observed mean ± SEM",
            zorder=3,
        )
        ax_data.set_xlim(0.0, display_end_s)
        ax_data.set_xticks(
            np.arange(0.0, np.floor(display_end_s) + 1.0, 1.0)
        )
        observed_low = (
            displayed_bins["mean"] - displayed_bins["sem"].fillna(0.0)
        ).to_numpy(dtype=float)
        observed_high = (
            displayed_bins["mean"] + displayed_bins["sem"].fillna(0.0)
        ).to_numpy(dtype=float)
        y_extent = np.concatenate(
            [observed_low, observed_high, *fitted_values]
        )
        y_extent = y_extent[np.isfinite(y_extent)]
        if y_extent.size:
            y_span = max(float(np.ptp(y_extent)), 0.2)
            y_min = np.floor((float(y_extent.min()) - 0.06 * y_span) * 10) / 10
            y_max = np.ceil((float(y_extent.max()) + 0.06 * y_span) * 10) / 10
            ax_data.set_ylim(y_min, y_max)
        ax_data.set_yticks([0.0, 0.5, 1.0, 1.5])
        ax_data.set_xlabel("Decision time (s)")
        ax_data.set_ylabel("Chosen-target SumLogLR")
        ax_data.legend(
            loc="upper left",
            fontsize=10,
            handlelength=1.3,
            handletextpad=0.4,
            borderpad=0.4,
            labelspacing=0.3,
        )

        all_row = response_statistics.loc[
            response_statistics["condition"] == "all"
        ]
        if len(all_row):
            all_result = stat_from_row(all_row.iloc[0], label="all")
            all_marker = significance_marker(all_result.p) or "n.s."
            annotate_stat_block(
                ax_data,
                lines=[
                    f"overall slope = {all_result.mean:+.3f} "
                    f"SumLogLR/s {all_marker}"
                ],
                loc="lower left",
            )

        slope_wide = response_fits.pivot(
            index="subject", columns="condition", values="slope"
        ).sort_index()
        groups = {
            condition: (
                slope_wide[condition].to_numpy(dtype=float)
                if condition in slope_wide
                else np.array([], dtype=float)
            )
            for condition in ("fast", "slow")
        }
        subject_strip(
            ax_slopes,
            groups=groups,
            colors={
                "fast": style.CONDITION_COLORS["fast"],
                "slow": style.CONDITION_COLORS["slow"],
            },
            reference=0.0,
            ylabel="Slope (SumLogLR/s)",
            connect=(("fast", "slow"),),
        )
        ax_slopes.set_xticklabels(["Fast", "Slow"])
        ax_slopes.set_xlim(-0.18, 1.18)
        ax_slopes.set_ylim(-0.5, 0.32)

        clipped_slope_outliers = []
        slow_values = slope_wide.get("slow", pd.Series(dtype=float))
        finite_slow = slow_values.dropna()
        slow_x = 1.0 + np.linspace(-0.08, 0.08, len(finite_slow))
        for x_value, (subject, slope) in zip(slow_x, finite_slow.items()):
            if float(slope) >= -0.5:
                continue
            clipped_slope_outliers.append(
                {
                    "subject": str(subject),
                    "condition": "slow",
                    "slope": float(slope),
                }
            )
            ax_slopes.scatter(
                [x_value],
                [-0.485],
                marker="o",
                s=20,
                facecolor=style.SURFACE,
                edgecolor=style.CONDITION_COLORS["slow"],
                linewidth=1.2,
                zorder=4,
            )
            ax_slopes.text(
                x_value - 0.055,
                -0.485,
                f"{float(slope):.2f}",
                ha="right",
                va="center",
                fontsize=8,
                color=style.CONDITION_COLORS["slow"],
                zorder=4,
            )

        fast_slow_row = response_statistics.loc[
            response_statistics["condition"] == "fast_vs_slow"
        ]
        if len(fast_slow_row):
            result = stat_from_row(fast_slow_row.iloc[0], label="fast_vs_slow")
            marker = significance_marker(result.p) or "n.s."
            displayed_slope_values = np.concatenate(
                [
                    values[np.isfinite(values) & (values >= -0.5)]
                    for values in groups.values()
                ]
            )
            highest_point = (
                float(displayed_slope_values.max())
                if displayed_slope_values.size
                else 0.0
            )
            bracket_y = highest_point + 0.07
            ax_slopes.plot(
                [0, 0, 1, 1],
                [bracket_y - 0.02, bracket_y, bracket_y, bracket_y - 0.02],
                color=style.INK,
                linewidth=0.8,
            )
            ax_slopes.text(
                0.5,
                bracket_y + 0.015,
                f"Δ = {result.mean:+.3f} {marker}",
                ha="center",
                va="bottom",
                fontsize=12,
                color=style.INK,
                zorder=5,
            )
            ax_slopes.set_ylim(-0.5, bracket_y + 0.13)

        fig.suptitle("Evidence criterion across decision time")

    metadata = {
        "kind": "first_order_criterion",
        "title": "Evidence criterion across decision time",
        "method": (
            "Cisek et al. (2009, Eq. 22 and Fig. 8) first-order chosen-target "
            "SumLogLR at commitment fitted by trial-level OLS against continuous "
            "decision time in seconds; 200-ms bins are display-only"
        ),
        "eligibility": (
            "primary_analysis_eligible and design_time_alignment_valid "
            "(complete 15-row token log), with decision_token_index > 0"
        ),
        "columns_read": {
            "criteriondecline": [
                "subject", "condition", "predictor", "response", "intercept",
                "slope", "converged",
            ],
            "criteriondeclinestats": [
                "predictor", "response", "term", "condition", "mean", "sem",
                "t", "p", "df", "cohens_dz",
            ],
            "trialfeatures": [
                "subject", "condition", predictor, "decision_token_index",
                "token_lead_at_decision", "isCorrect",
                "primary_analysis_eligible", "design_time_alignment_valid",
            ],
        },
        "statistics_source": (
            "criteriondeclinestats[response="
            "chosen_sum_log_lr_first_order_at_decision, term=slope]"
        ),
        "display_window_s": [0.0, display_end_s],
        "display_min_subjects_per_200ms_bin": display_subject_threshold,
        "n_eligible_trials_beyond_display": beyond_display_trials,
        "n_eligible_trials_after_3s": post_horizon_trials,
        "clipped_slope_outliers": clipped_slope_outliers,
        "observed_n_subjects_by_200ms_bin": {
            f"{int(row.display_bin * 200)}-"
            f"{int((row.display_bin + 1) * 200)}": int(row.n_subjects)
            for row in bin_summary.itertuples(index=False)
        },
        "palette": {
            "all": style.INK,
            "fast": style.CONDITION_COLORS["fast"],
            "slow": style.CONDITION_COLORS["slow"],
        },
        "caveat": (
            f"The {excluded_alignment_trials:,} primary trials without a "
            "complete, alignable 15-row token log are excluded because "
            "their designed token sequence cannot be aligned unambiguously "
            "to the commitment index; "
            f"{excluded_pre_token_trials:,} aligned pre-first-token "
            "commitments are excluded because they contain no sensory evidence. "
            "Panel A extends through the last 200-ms bin represented by at "
            f"least {display_subject_threshold} subjects ({display_end_s:g} s); "
            f"the primary fits retain all eligible trials, including "
            f"{beyond_display_trials:,} later trials. Panel A shows the "
            "subject-balanced observed-bin SEM and subject-level overall fits; "
            "Panel B shows subject estimates with group 95% CIs and marks "
            "estimates below -0.5 with a hollow boundary point labeled by "
            "its exact slope. The fitted "
            "lines are drawn from 0 s, but their intercepts are extrapolated "
            "because pre-first-token trials are excluded."
        ),
    }
    return fig, metadata


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
