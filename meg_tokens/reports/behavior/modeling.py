"""Sequential-sampling model figures: F01, F02, F03, F21, F22."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.figure import Figure

from meg_tokens.reports import style
from meg_tokens.reports.annotations import (
    StatResult,
    annotate_stat_block,
    format_stat,
    stat_from_row,
)
from meg_tokens.reports.behavior._tables import BehaviorTableSet
from meg_tokens.reports.panels import estimation_axis, forest, paired_slope


def build_ssmcomparison_deltabic(tables: BehaviorTableSet) -> tuple[Figure, dict[str, Any]]:
    """F01: H1 -- urgency gating beats bounded integration."""
    fits = tables.analysis("ssmcomparison")
    stats_table = tables.analysis("ssmcomparisonstats")

    urgency = fits.loc[fits["model"] == "urgency"]
    pooled = urgency.loc[urgency["condition"] == "all"].sort_values("delta_bic")

    model_stats = stats_table.loc[stats_table["analysis"] == "ssm_model_comparison"]
    bic_stats = model_stats.loc[model_stats["criterion"] == "bic"]
    pooled_row = bic_stats.loc[bic_stats["condition"] == "all"]
    fast_row = bic_stats.loc[bic_stats["condition"] == "fast"]
    slow_row = bic_stats.loc[bic_stats["condition"] == "slow"]

    with style.apply_publication_style():
        fig, axes = style.figure_grid(1, 2, width="double", width_ratios=[1.3, 1.0])
        ax_a, ax_b = axes[0, 0], axes[0, 1]
        style.panel_label(ax_a, "A")
        style.panel_label(ax_b, "B")

        # Panel A: ordered per-subject delta BIC lollipop.
        n = len(pooled)
        y_positions = np.arange(n)
        colors = [
            style.MODEL_COLORS["urgency"] if value < 0 else style.MODEL_COLORS["ddm"]
            for value in pooled["delta_bic"]
        ]
        ax_a.axvline(0, color=style.INK_MUTED, linewidth=0.7, zorder=0)
        ax_a.hlines(
            y_positions, 0, pooled["delta_bic"], color=colors, linewidth=1.2, zorder=1
        )
        ax_a.scatter(pooled["delta_bic"], y_positions, color=colors, s=14, zorder=2)
        ax_a.set_yticks([])
        ax_a.set_ylabel(f"Subjects (n = {n}), sorted")
        ax_a.set_xlabel("ΔBIC (urgency − integrator)")
        ax_a.text(
            0.02, -0.10, "urgency preferred", transform=ax_a.transAxes,
            ha="left", va="top", fontsize=6.5, color=style.MODEL_COLORS["urgency"],
        )
        ax_a.text(
            0.98, -0.10, "integrator preferred", transform=ax_a.transAxes,
            ha="right", va="top", fontsize=6.5, color=style.MODEL_COLORS["ddm"],
        )

        n_urgency = int((pooled["delta_bic"] < 0).sum())
        n_ddm = int((pooled["delta_bic"] > 0).sum())
        lines = []
        if len(pooled_row):
            result = stat_from_row(pooled_row.iloc[0], label="all")
            lines.append(format_stat(result))
            lines.append(f"urgency preferred in {n_urgency}/{n} subjects")
        for label, row in (("fast", fast_row), ("slow", slow_row)):
            if len(row):
                result = stat_from_row(row.iloc[0], label=label)
                lines.append(f"{label}: {format_stat(result, include_effect_size=False)}")
        ax_a.text(
            0.02, 0.98, "\n".join(lines), transform=ax_a.transAxes,
            ha="left", va="top", fontsize=6.5, color=style.INK, linespacing=1.4,
        )

        # Panel B: BIC scatter, urgency vs integrator, with identity line.
        wide = fits.loc[fits["condition"] == "all"].pivot(
            index="subject", columns="model", values="bic"
        )
        finite = wide.dropna(subset=["urgency", "ddm"])
        ax_b.scatter(
            finite["ddm"], finite["urgency"], color=style.INK, s=14, alpha=0.75, zorder=2
        )
        lo = float(min(finite["ddm"].min(), finite["urgency"].min()))
        hi = float(max(finite["ddm"].max(), finite["urgency"].max()))
        pad = (hi - lo) * 0.05
        ax_b.plot(
            [lo - pad, hi + pad], [lo - pad, hi + pad],
            color=style.INK_MUTED, linewidth=0.8, zorder=1, linestyle="--",
        )
        ax_b.set_xlim(lo - pad, hi + pad)
        ax_b.set_ylim(lo - pad, hi + pad)
        ax_b.set_aspect("equal")
        ax_b.set_xlabel("BIC, bounded integrator")
        ax_b.set_ylabel("BIC, urgency gating")
        ax_b.text(
            0.02, 0.98, "below identity line = urgency wins",
            transform=ax_b.transAxes, ha="left", va="top",
            fontsize=6.5, color=style.INK_SECONDARY,
        )

        fig.suptitle("Urgency gating versus bounded integration (H1)")

    metadata = {
        "kind": "per_subject_criterion_difference",
        "title": "Urgency gating versus bounded integration",
        "columns_read": {
            "ssmcomparison": [
                "subject", "condition", "model", "bic", "delta_bic",
                "n_trials", "n_token_sequences", "converged",
            ],
            "ssmcomparisonstats": [
                "analysis", "condition", "criterion", "mean", "sem", "t", "p",
                "df", "cohens_dz", "n_subjects",
            ],
        },
        "statistics_source": "ssmcomparisonstats[analysis=ssm_model_comparison,criterion=bic]",
        "n_subjects": n,
        "palette": dict(style.MODEL_COLORS),
    }
    return fig, metadata


def build_ssmcomparison_urgencyscale(tables: BehaviorTableSet) -> tuple[Figure, dict[str, Any]]:
    """F02: H2 -- urgency scale differs Fast vs. Slow."""
    fits = tables.analysis("ssmcomparison")
    stats_table = tables.analysis("ssmcomparisonstats")

    urgency = fits.loc[fits["model"] == "urgency"].set_index("subject")
    fast = urgency.loc[urgency["condition"] == "fast", "urgency_scale"]
    slow = urgency.loc[urgency["condition"] == "slow", "urgency_scale"]
    paired_index = fast.index.intersection(slow.index)
    fast_values = fast.loc[paired_index].to_numpy(dtype=float)
    slow_values = slow.loc[paired_index].to_numpy(dtype=float)

    contrast_rows = stats_table.loc[
        (stats_table["analysis"] == "ssm_urgency_condition_contrast")
        & (stats_table["parameter"] == "urgency_scale")
    ]
    result = (
        stat_from_row(contrast_rows.iloc[0], label="fast_minus_slow")
        if len(contrast_rows)
        else StatResult(
            label="fast_minus_slow", n_subjects=0, mean=None, sem=None,
            t=None, p=None, df=None, cohens_dz=None,
        )
    )

    with style.apply_publication_style():
        fig, axes = style.figure_grid(1, 2, width="double", width_ratios=[2.0, 1.0])
        ax_left, ax_right = axes[0, 0], axes[0, 1]
        style.panel_label(ax_left, "A")
        style.panel_label(ax_right, "B")

        paired_slope(
            ax_left,
            values_a=fast_values, values_b=slow_values,
            label_a="Fast", label_b="Slow",
            color_a=style.CONDITION_COLORS["fast"],
            color_b=style.CONDITION_COLORS["slow"],
            ylabel="urgency_scale (criterion-seconds;\nsmaller = urgency rises faster)",
        )

        differences = fast_values - slow_values
        estimation_axis(ax_right, differences=differences, result=result, unit="s")
        ax_right.set_title("Fast − Slow", fontsize=8)

        fig.suptitle("Urgency rises faster under time pressure (H2)")

    caveat = (
        "urgency_scale is threshold / urgency slope: smaller means faster-rising "
        "urgency, so a negative Fast-minus-Slow difference means urgency grows "
        "FASTER in Fast blocks. The sign is easy to misread."
    )
    metadata = {
        "kind": "paired_condition_contrast",
        "title": "Urgency scale, Fast vs. Slow",
        "columns_read": {
            "ssmcomparison": ["subject", "condition", "model", "urgency_scale", "urgency_scale_se", "converged"],
            "ssmcomparisonstats": [
                "analysis", "parameter", "mean_a", "sem_a", "mean_b", "sem_b",
                "mean_difference", "t", "p", "df", "cohens_dz", "n_subjects",
            ],
        },
        "statistics_source": "ssmcomparisonstats[analysis=ssm_urgency_condition_contrast,parameter=urgency_scale]",
        "n_subjects": int(paired_index.size),
        "palette": dict(style.CONDITION_COLORS),
        "caveat": caveat,
    }
    return fig, metadata


_URGENCY_PARAMETERS = ("drift_scale", "urgency_scale", "urgency_onset_s", "nondecision_s")


def build_ssmcomparison_urgencyparams(tables: BehaviorTableSet) -> tuple[Figure, dict[str, Any]]:
    """F03: all four urgency-model parameters, Fast vs. Slow.

    Includes the two null parameters (urgency_onset_s, nondecision_s):
    the claim that Fast/Slow acts on urgency growth rather than these is
    only supported if they are shown not to move.
    """
    fits = tables.analysis("ssmcomparison")
    stats_table = tables.analysis("ssmcomparisonstats")

    urgency = fits.loc[fits["model"] == "urgency"].set_index("subject")
    contrast_rows = stats_table.loc[stats_table["analysis"] == "ssm_urgency_condition_contrast"]

    with style.apply_publication_style():
        fig, axes = style.figure_grid(2, 2, width="double", panel_height_in=2.6)
        for index, parameter in enumerate(_URGENCY_PARAMETERS):
            ax = axes[index // 2, index % 2]
            style.panel_label(ax, "ABCD"[index])
            fast = urgency.loc[urgency["condition"] == "fast", parameter]
            slow = urgency.loc[urgency["condition"] == "slow", parameter]
            paired_index = fast.index.intersection(slow.index)
            paired_slope(
                ax,
                values_a=fast.loc[paired_index].to_numpy(dtype=float),
                values_b=slow.loc[paired_index].to_numpy(dtype=float),
                label_a="Fast", label_b="Slow",
                color_a=style.CONDITION_COLORS["fast"], color_b=style.CONDITION_COLORS["slow"],
                ylabel=parameter,
            )
            row = contrast_rows.loc[contrast_rows["parameter"] == parameter]
            if len(row):
                result = stat_from_row(row.iloc[0], label=parameter)
                annotate_stat_block(ax, lines=[format_stat(result)], loc="lower right")

        fig.suptitle("Urgency-model parameters, Fast vs. Slow")

    metadata = {
        "kind": "paired_condition_contrast_grid",
        "title": "Urgency-model parameters, Fast vs. Slow",
        "columns_read": {
            "ssmcomparison": ["subject", "condition", "model", *_URGENCY_PARAMETERS],
            "ssmcomparisonstats": [
                "analysis", "parameter", "mean_a", "mean_b", "mean_difference",
                "t", "p", "df", "cohens_dz", "n_subjects",
            ],
        },
        "statistics_source": "ssmcomparisonstats[analysis=ssm_urgency_condition_contrast]",
        "palette": dict(style.CONDITION_COLORS),
    }
    return fig, metadata


def build_ssmtimecourse_fit(tables: BehaviorTableSet) -> tuple[Figure, dict[str, Any]]:
    """F21: what the two models actually do -- criterion, fitted densities,
    and the noise-free decision-variable trajectory.

    De-duplication trap: `criterion` does not depend on trial_class and
    `observed_density_*` does not depend on model; both repeat across rows
    in the raw table. Averaging without dropping duplicates first would
    weight them by how many trial classes/models happen to be present.
    """
    timecourse = tables.analysis("ssmtimecourse")
    pooled = timecourse.loc[timecourse["condition"] == "all"]

    with style.apply_publication_style():
        fig, axes = style.figure_grid(1, 3, width="double")
        ax_a, ax_b, ax_c = axes[0, 0], axes[0, 1], axes[0, 2]
        style.panel_label(ax_a, "A")
        style.panel_label(ax_b, "B")
        style.panel_label(ax_c, "C")

        # Panel A: criterion time course, one line per model.
        criterion_rows = pooled.drop_duplicates(["subject", "condition", "model", "time_s"])
        for model in style.MODEL_ORDER:
            model_rows = criterion_rows.loc[criterion_rows["model"] == model]
            grouped = model_rows.groupby("time_s")["criterion"].mean().sort_index()
            ax_a.plot(
                grouped.index, grouped.to_numpy(),
                color=style.MODEL_COLORS[model], label=style.MODEL_LABELS[model],
                linewidth=1.6,
            )
        ax_a.set_xlabel("Time (s)")
        ax_a.set_ylabel("Criterion")
        ax_a.legend(loc="upper right", fontsize=6)

        # Panel B: predicted (line, per model) vs observed (grey fill, model-
        # independent) densities, correct up / error mirrored below.
        observed_rows = pooled.drop_duplicates(["subject", "condition", "trial_class", "time_s"])
        observed_correct = observed_rows.groupby("time_s")["observed_density_correct"].mean().sort_index()
        observed_error = observed_rows.groupby("time_s")["observed_density_error"].mean().sort_index()
        ax_b.fill_between(observed_correct.index, observed_correct.to_numpy(), color=style.OBSERVED, alpha=0.25, label="observed")
        ax_b.fill_between(observed_error.index, -observed_error.to_numpy(), color=style.OBSERVED, alpha=0.25)
        for model in style.MODEL_ORDER:
            model_rows = pooled.loc[pooled["model"] == model]
            predicted_correct = model_rows.groupby("time_s")["predicted_density_correct"].mean().sort_index()
            predicted_error = model_rows.groupby("time_s")["predicted_density_error"].mean().sort_index()
            ax_b.plot(predicted_correct.index, predicted_correct.to_numpy(), color=style.MODEL_COLORS[model], linewidth=1.4)
            ax_b.plot(predicted_error.index, -predicted_error.to_numpy(), color=style.MODEL_COLORS[model], linewidth=1.4)
        ax_b.axhline(0, color=style.INK_MUTED, linewidth=0.6)
        ax_b.set_xlabel("Time (s)")
        ax_b.set_ylabel("Density (correct up / error down)")
        ax_b.legend(loc="upper right", fontsize=6)
        # n_trials repeats across every time_s row AND across model (both
        # models are fit to the same trial cell) for a given subject/class;
        # de-duplicate on those two keys before summing, exactly like the
        # criterion/observed_density traps above -- a naive sum over `pooled`
        # would count each trial ~2x(models) x (time points) times over.
        n_trials = int(pooled.drop_duplicates(["subject", "trial_class"])["n_trials"].sum()) if len(pooled) else 0
        ax_b.text(0.02, 0.02, f"n_trials (all classes, all subjects) = {n_trials}", transform=ax_b.transAxes, fontsize=5.5, color=style.INK_SECONDARY)

        # Panel C: mean decision variable under the urgency model, per class,
        # criterion overlaid dashed. This is the noise-free UNABSORBED
        # trajectory, so it legitimately crosses the criterion.
        urgency_rows = pooled.loc[pooled["model"] == "urgency"]
        for trial_class in style.CLASS_ORDER:
            class_rows = urgency_rows.loc[urgency_rows["trial_class"] == trial_class]
            grouped = class_rows.groupby("time_s")["mean_decision_variable"].mean().sort_index()
            ax_c.plot(grouped.index, grouped.to_numpy(), color=style.CLASS_COLORS[trial_class], label=trial_class, linewidth=1.4)
        urgency_criterion = criterion_rows.loc[criterion_rows["model"] == "urgency"]
        criterion_grouped = urgency_criterion.groupby("time_s")["criterion"].mean().sort_index()
        ax_c.plot(criterion_grouped.index, criterion_grouped.to_numpy(), color=style.INK, linestyle="--", linewidth=1.0, label="criterion")
        ax_c.plot(criterion_grouped.index, -criterion_grouped.to_numpy(), color=style.INK, linestyle="--", linewidth=1.0)
        ax_c.set_xlabel("Time (s)")
        ax_c.set_ylabel("Mean decision variable (urgency model)")
        ax_c.legend(loc="center right", fontsize=6)
        ax_c.text(
            0.02, 0.02,
            "noise-free, unabsorbed trajectory --\ncrossing the criterion is expected",
            transform=ax_c.transAxes, ha="left", va="bottom", fontsize=5.5, color=style.INK_SECONDARY,
        )

        fig.suptitle("Urgency gating vs. bounded integration: model internals")

    metadata = {
        "kind": "model_time_course",
        "title": "Model internals: criterion, fit quality, decision variable",
        "columns_read": {
            "ssmtimecourse": [
                "subject", "condition", "model", "trial_class", "n_trials", "time_s",
                "criterion", "mean_decision_variable", "predicted_density_correct",
                "predicted_density_error", "observed_density_correct", "observed_density_error",
            ],
        },
        "statistics_source": "none (see ssmcomparison-deltabic for the formal H1 test)",
        "palette": {**style.MODEL_COLORS, **style.CLASS_COLORS},
        "deduplication": (
            "criterion de-duplicated on [subject,condition,model,time_s]; "
            "observed_density_* de-duplicated on [subject,condition,trial_class,time_s]"
        ),
    }
    return fig, metadata


_INTEGRATOR_PARAMETERS = ("drift_scale", "bound", "nondecision_s")


def build_ssmpopulation_shrinkage(tables: BehaviorTableSet) -> tuple[Figure, dict[str, Any]]:
    """F22: empirical-Bayes population model, per parameter, condition == 'all'.

    n_subjects is not 32 for every parameter (e.g. urgency_scale n=15 in
    condition='all', from non-finite observed-information SEs) -- printed
    on every panel since a forest plot silently missing half the cohort is
    a serious misread.
    """
    subject_estimates = tables.analysis("ssmpopulation")
    population = tables.analysis("ssmpopulationstats")

    panels = [("urgency", parameter) for parameter in _URGENCY_PARAMETERS]
    panels += [("ddm", parameter) for parameter in _INTEGRATOR_PARAMETERS]

    with style.apply_publication_style():
        fig, axes = style.figure_grid(1, len(panels), width="double", panel_height_in=3.4)
        for index, (model, parameter) in enumerate(panels):
            ax = axes[0, index]
            subject_rows = subject_estimates.loc[
                (subject_estimates["model"] == model)
                & (subject_estimates["parameter"] == parameter)
                & (subject_estimates["condition"] == "all")
            ].sort_values("estimate")
            pop_row = population.loc[
                (population["model"] == model)
                & (population["parameter"] == parameter)
                & (population["condition"] == "all")
            ]

            if len(subject_rows):
                centres = subject_rows["estimate"].to_numpy(dtype=float)
                errors = subject_rows["standard_error"].to_numpy(dtype=float)
                forest(
                    ax,
                    labels=subject_rows["subject"].tolist(),
                    centres=centres, lows=centres - errors, highs=centres + errors,
                    secondary=subject_rows["population_informed_estimate"].to_numpy(dtype=float),
                )
                ax.set_yticks([])
            if len(pop_row):
                mean = float(pop_row.iloc[0]["population_mean"])
                sd = float(pop_row.iloc[0]["between_subject_sd"])
                ax.axvspan(mean - sd, mean + sd, color=style.INK_MUTED, alpha=0.15, zorder=0)
                ax.axvline(mean, color=style.INK, linewidth=1.0, zorder=0)
                n = int(pop_row.iloc[0]["n_subjects"])
            else:
                n = 0
            ax.set_title(f"{model}\n{parameter}\n(n = {n})", fontsize=5.8, linespacing=1.3)

        fig.suptitle("Population model: subject estimates and empirical-Bayes shrinkage")

    metadata = {
        "kind": "shrinkage_forest",
        "title": "Population sequential-sampling parameters",
        "columns_read": {
            "ssmpopulation": [
                "subject", "condition", "model", "parameter", "estimate",
                "standard_error", "population_informed_estimate", "own_data_weight",
            ],
            "ssmpopulationstats": [
                "condition", "model", "parameter", "n_subjects", "population_mean",
                "population_mean_se", "between_subject_sd", "z", "p",
            ],
        },
        "statistics_source": "ssmpopulationstats",
    }
    return fig, metadata
