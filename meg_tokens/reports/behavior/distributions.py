"""Decision-time and SPD distribution figures: F04, F05, F06."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from meg_tokens.reports import style
from meg_tokens.reports.annotations import (
    annotate_stat_block,
    format_stat,
    significance_marker,
    stat_from_row,
)
from meg_tokens.reports.behavior._tables import BehaviorTableSet
from meg_tokens.reports.panels import quantile_function, raincloud

_QUANTILE_PROBABILITIES = (0.10, 0.25, 0.50, 0.75, 0.90)
_QUANTILE_COLUMNS = ("q10", "q25", "q50", "q75", "q90")


def build_dtdistribution_condition(tables: BehaviorTableSet) -> tuple[Figure, dict[str, Any]]:
    """F04: Fast vs. Slow decision time. Replaces the rawRT-based, pooled figure."""
    trial_features = tables.trial_features()
    dtdistribution = tables.analysis("dtdistribution")
    dtdistributionstats = tables.analysis("dtdistributionstats")
    groupstats = tables.group_statistics()

    condition_rows = dtdistribution.loc[
        (dtdistribution["stratum_type"] == "condition")
        & (dtdistribution["stratum"].isin(style.CONDITION_ORDER))
    ]
    subject_quantiles = {
        condition: (
            condition_rows.loc[condition_rows["stratum"] == condition]
            .set_index("subject")[list(_QUANTILE_COLUMNS)]
            .to_numpy(dtype=float)
        )
        for condition in style.CONDITION_ORDER
    }
    # Subject-aligned mean/SD, used for panel A's CV note. Indexed by subject
    # and reindexed onto a common order so the per-subject CV pairs up.
    _subject_order = (
        condition_rows.loc[condition_rows["stratum"] == style.CONDITION_ORDER[0], "subject"]
        .tolist()
    )
    per_subject_mean, per_subject_sd = {}, {}
    for condition in style.CONDITION_ORDER:
        rows = condition_rows.loc[condition_rows["stratum"] == condition].set_index("subject")
        per_subject_mean[condition] = rows["mean"].reindex(_subject_order).to_numpy(dtype=float)
        per_subject_sd[condition] = rows["sd"].reindex(_subject_order).to_numpy(dtype=float)

    contrast_stats = dtdistributionstats.loc[
        (dtdistributionstats["stratum_type"] == "condition")
        & (dtdistributionstats["contrast"] == "fast_vs_slow")
    ]

    with style.apply_publication_style():
        # Taller than the shared default, and A:B width 3:1 rather than an
        # even split -- A is the distributional claim and needs the room;
        # B is a two-category paired plot and needs height (for the
        # mean-difference slope to read against B's ~500-2000 ms range) more
        # than it needs width.
        fig, axes = style.figure_grid(1, 2, width="double", width_ratios=[2.7, 1.3], panel_height_in=4.6)
        ax_a, ax_b = axes[0, 0], axes[0, 1]
        style.panel_label(ax_a, "A")
        style.panel_label(ax_b, "B")

        # A carries the shape claim and nothing else. The pooled-trial KDE
        # that used to sit behind these curves was removed: the finding is
        # that the *gap between the curves widens* with the quantile, and a
        # density filling the panel behind them is exactly the wrong
        # background to judge a widening gap against. The distribution now
        # lives on B (half-violins), matching F05's grammar.
        display_range = (500, 2000)
        quantile_function(
            ax_a,
            probabilities=_QUANTILE_PROBABILITIES,
            subject_quantiles=subject_quantiles,
            colors=style.CONDITION_COLORS,
        )
        ax_a.set_xlim(*display_range)
        ax_a.set_xticks(list(display_range))
        ax_a.set_yticks([0.1, 0.5, 0.9])
        ax_a.set_xlabel("Decision time (ms)")
        ax_a.legend(loc="lower right", fontsize=12)

        # Per-quantile significance markers (q10/q50/q90), placed on panel A
        # next to the quantile the contrast is about -- not as text on B.
        for probability, metric in (0.10, "q10"), (0.50, "q50"), (0.90, "q90"):
            rows = contrast_stats.loc[contrast_stats["metric"] == metric]
            if not len(rows):
                continue
            marker = significance_marker(stat_from_row(rows.iloc[0], label=metric).p)
            column = _QUANTILE_COLUMNS.index(metric)
            x_right = max(
                np.nanmean(subject_quantiles["fast"][:, column]),
                np.nanmean(subject_quantiles["slow"][:, column]),
            )
            # Cap derived from the display window, not hardcoded -- the
            # window narrowed to 500-2000 ms and a literal would silently
            # stop clamping.
            ax_a.text(
                min(x_right + 60, display_range[1] - 50), probability, marker or "n.s.",
                ha="left", va="center", fontsize=13, color=style.INK,
            )

        # The shape note reports CV, not skewness. Skewness is invariant
        # under *any* positive linear transform, so a fixed delay and a
        # proportional stretch both predict "skew n.s." -- it cannot
        # discriminate the two models this panel exists to distinguish, and
        # printing it invited the reader to treat a null as evidence. CV is
        # scale-invariant but shift-variant, so it does discriminate: a fixed
        # delay would have driven it down to ~0.36, a stretch leaves it put.
        # Observed value and the counterfactual side by side, since the
        # contrast between them is the whole point.
        cv = {
            condition: float(np.nanmean(
                per_subject_sd[condition] / per_subject_mean[condition]
            ))
            for condition in style.CONDITION_ORDER
        }
        shifted_cv = float(np.nanmean(
            per_subject_sd["fast"]
            / (per_subject_mean["fast"] + (per_subject_mean["slow"] - per_subject_mean["fast"]))
        ))
        # Top-left: bottom-right is the legend's corner, and every quantile
        # curve/marker in this figure sits at x >= ~650 ms, so the
        # low-x/high-probability corner is the one that's actually empty.
        ax_a.text(
            0.02, 0.98,
            f"CV {cv['fast']:.2f} → {cv['slow']:.2f}\n"
            f"(fixed delay predicts {shifted_cv:.2f})",
            transform=ax_a.transAxes, ha="left", va="top",
            fontsize=11, color=style.INK_SECONDARY, linespacing=1.4,
        )

        pooled_fast = (
            condition_rows.loc[condition_rows["stratum"] == "fast"].set_index("subject")["mean"]
        )
        pooled_slow = (
            condition_rows.loc[condition_rows["stratum"] == "slow"].set_index("subject")["mean"]
        )
        paired_index = pooled_fast.index.intersection(pooled_slow.index)
        fast_values = pooled_fast.loc[paired_index].to_numpy(dtype=float)
        slow_values = pooled_slow.loc[paired_index].to_numpy(dtype=float)
        # Raincloud rather than a bare paired slope: half-violins put the
        # distribution back in the figure (it left panel A), and this is the
        # same chart type as F05 panel A, so the two decision-time figures
        # now share one grammar. Subject connectors and the direction
        # coloring survive the swap.
        raincloud(
            ax_b,
            groups={"Fast": fast_values, "Slow": slow_values},
            colors={
                "Fast": style.CONDITION_COLORS["fast"],
                "Slow": style.CONDITION_COLORS["slow"],
            },
            ylabel="Subject mean DT (ms)",
            # Slow > Fast is the group direction (dz around -1); reversed
            # subjects are the interesting exceptions and get the color that
            # pops. Reusing "fast" red keeps this inside CONDITION_COLORS'
            # already-validated scope instead of adding a third palette to
            # one panel (style.py's scoping rule).
            increase_color=style.SUBJECT_LINE, decrease_color=style.CONDITION_COLORS["fast"],
        )

        # Bracket + compact stat text directly on the comparison, same idiom
        # as F05 panel A's class brackets -- not a text block off to the side.
        mean_contrast = groupstats.loc[
            (groupstats["analysis"] == "decision_time") & (groupstats["contrast"] == "fast_vs_slow")
        ]
        if len(mean_contrast) and fast_values.size:
            result = stat_from_row(mean_contrast.iloc[0], label="mean")
            y_top = max(fast_values.max(), slow_values.max())
            y_bottom = min(fast_values.min(), slow_values.min())
            bracket_y = y_top + 0.08 * (y_top - y_bottom)
            # One line, tightly boxed: "Delta + marker" is the report-wide
            # convention for a bracket comparison now (matches F05 panel A)
            # -- t and dz stay in dtdistributionstats/the sidecar rather than
            # being spelled out on every figure.
            ax_b.set_ylim(top=bracket_y + 0.14 * (y_top - y_bottom))
            ax_b.plot([0, 0, 1, 1], [bracket_y * 0.99, bracket_y, bracket_y, bracket_y * 0.99], color=style.INK, linewidth=0.8)
            marker = significance_marker(result.p) or "n.s."
            ax_b.text(
                0.5, bracket_y * 1.02, f"Δ = {result.mean:.0f} ms {marker}", ha="center", va="bottom",
                fontsize=12, color=style.INK,
            )
        ax_b.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))

        fig.suptitle("Decision time: Fast vs. Slow")

    metadata = {
        "kind": "condition_distribution",
        "title": "Decision time, Fast vs. Slow",
        "columns_read": {
            "trialfeatures": ["subject", "condition", "dt_ms", "primary_analysis_eligible"],
            "dtdistribution": ["subject", "stratum_type", "stratum", "mean", "q10", "q25", "q50", "q75", "q90", "n_trials"],
            "dtdistributionstats": ["stratum_type", "contrast", "metric", "mean_a", "mean_b", "t", "p", "df", "cohens_dz"],
            "groupstats": ["analysis", "contrast", "mean_a", "mean_b", "t", "p", "df", "cohens_dz"],
        },
        "statistics_source": "dtdistributionstats[stratum_type=condition,contrast=fast_vs_slow] + groupstats[analysis=decision_time]",
        "palette": dict(style.CONDITION_COLORS),
    }
    return fig, metadata


def build_dtdistribution_class(tables: BehaviorTableSet) -> tuple[Figure, dict[str, Any]]:
    """F05: decision time by trial class. Replaces the dead
    plot_trial_class_distributions (pooled tri-KDE, never wired to the CLI)."""
    dtdistribution = tables.analysis("dtdistribution")
    dtdistributionstats = tables.analysis("dtdistributionstats")

    class_rows = dtdistribution.loc[
        (dtdistribution["stratum_type"] == "class")
        & (dtdistribution["stratum"].isin(style.CLASS_ORDER))
    ]
    subject_means = {
        trial_class: (
            class_rows.loc[class_rows["stratum"] == trial_class]
            .set_index("subject")["mean"]
        )
        for trial_class in style.CLASS_ORDER
    }
    subject_quantiles = {
        trial_class: (
            class_rows.loc[class_rows["stratum"] == trial_class]
            .set_index("subject")[list(_QUANTILE_COLUMNS)]
            .to_numpy(dtype=float)
        )
        for trial_class in style.CLASS_ORDER
    }

    class_contrasts = dtdistributionstats.loc[dtdistributionstats["stratum_type"] == "class"]

    with style.apply_publication_style():
        # Taller than the shared default, same reason as F04: at the bigger
        # publication font sizes, three stacked brackets on A and a legend +
        # note on B both need more room than the old default gave them.
        fig, axes = style.figure_grid(1, 2, width="double", panel_height_in=3.8)
        ax_a, ax_b = axes[0, 0], axes[0, 1]
        style.panel_label(ax_a, "A")
        style.panel_label(ax_b, "B")

        raincloud(
            ax_a,
            groups={name: values.to_numpy(dtype=float) for name, values in subject_means.items()},
            colors=style.CLASS_COLORS,
            ylabel="Subject mean DT (ms)",
            # Report-wide convention (see F04 panel B): gray with the
            # step's own direction, red against it. Per-leg, not per-subject
            # -- a subject can go up easy->ambiguous and down
            # ambiguous->misleading, and both legs are worth seeing.
            increase_color=style.SUBJECT_LINE, decrease_color=style.CONDITION_COLORS["fast"],
        )
        ax_a.set_xlabel("Trial class")
        # Rotated: "ambiguous" and "misleading" collide at horizontal at this
        # font size in a 3-category axis this narrow.
        for label in ax_a.get_xticklabels():
            label.set_rotation(20)
            label.set_ha("right")
        ax_a.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))

        # "Delta + marker" on the bracket, matching F04 panel B -- t and dz
        # stay in dtdistributionstats/the sidecar. Two of these three
        # brackets span just one category (~1/3 of the panel), so even this
        # short a string is close to the limit.
        mean_rows = class_contrasts.loc[class_contrasts["metric"] == "q50"]
        y_max = max((series.max() for series in subject_means.values() if len(series)), default=1500)
        bracket_y = y_max * 1.08
        contrasts = (("easy", "ambiguous"), ("ambiguous", "misleading"), ("easy", "misleading"))
        for offset, (first, second) in enumerate(contrasts):
            contrast_name = f"{first}_vs_{second}"
            row = mean_rows.loc[mean_rows["contrast"] == contrast_name]
            if not len(row):
                continue
            result = stat_from_row(row.iloc[0], label=contrast_name)
            marker = significance_marker(result.p) or "n.s."
            x_a, x_b = style.CLASS_ORDER.index(first), style.CLASS_ORDER.index(second)
            y = bracket_y * (1 + 0.11 * offset)
            ax_a.plot([x_a, x_a, x_b, x_b], [y * 0.99, y, y, y * 0.99], color=style.INK, linewidth=0.9)
            ax_a.text((x_a + x_b) / 2, y, f"Δ = {result.mean:.0f} {marker}", ha="center", va="bottom", fontsize=11, color=style.INK)
        ax_a.set_ylim(top=bracket_y * (1 + 0.11 * (len(contrasts) - 1)) * 1.02)

        # No pooled-trial KDE here: panel A's violins already show
        # distribution shape per class, so B's job is just the quantile
        # functions -- zoomed to 500-2000 ms (class quantiles run roughly
        # 600-2000 ms; a little margin past 2000 so the q90 marker and its
        # SEM whisker for ambiguous/misleading, ~1971 and ~1958, don't clip
        # the right edge) instead of F04's 0-3000 ms, since there's no
        # trial-level long tail here that needs the wider window.
        quantile_function(
            ax_b,
            probabilities=_QUANTILE_PROBABILITIES,
            subject_quantiles=subject_quantiles,
            colors=style.CLASS_COLORS,
        )
        ax_b.set_xlim(500, 2100)
        ax_b.set_xticks([500, 2000])
        ax_b.set_yticks([0.1, 0.5, 0.9])
        ax_b.set_xlabel("Decision time (ms)")
        # ambiguous's q25 SEM whisker grazes a default-sized legend's top-left
        # corner here; tighter handles/padding pulls the box in just enough
        # to clear it without moving off lower-right.
        ax_b.legend(loc="lower right", fontsize=12, handlelength=1.3, handletextpad=0.4, borderpad=0.4, labelspacing=0.3)

        fig.suptitle("Decision time by trial class")

    metadata = {
        "kind": "class_distribution",
        "title": "Decision time by trial class",
        "columns_read": {
            "dtdistribution": ["subject", "stratum_type", "stratum", "mean", "q10", "q25", "q50", "q75", "q90"],
            "dtdistributionstats": ["stratum_type", "contrast", "metric", "mean_a", "mean_b", "t", "p", "df", "cohens_dz"],
        },
        "statistics_source": "dtdistributionstats[stratum_type=class]",
        "palette": dict(style.CLASS_COLORS),
        "excludes": "unclassified",
    }
    return fig, metadata


_SPD_VIEW = "validated_15row"


def build_spdcumulative_class(tables: BehaviorTableSet) -> tuple[Figure, dict[str, Any]]:
    """F06: SPD at decision, cumulative by class.

    validated_15row only, not both logging views: the two views were checked
    against each other (they agree -- validated_15row's proportions land
    within a few points of all_logged's despite excluding up to 40% of
    trials per class, the ones from unvalidated "14-row" logs), so showing
    both permanently was paying two panels to say the same thing twice.
    validated_15row is the higher-integrity subset and is kept as the sole
    view here. all_logged is still computed and used elsewhere in the
    pipeline (summary, individual.py's cross-species comparison,
    performance.py) -- whether those should also drop to validated_15row is
    open, not decided by this figure; TBD.
    """
    spd = tables.analysis("spdcumulative")
    groupstats = tables.group_statistics()
    spd_stats = groupstats.loc[groupstats["analysis"] == "spd"]

    view_rows = spd.loc[spd["view"] == _SPD_VIEW]
    view_stats = spd_stats.loc[spd_stats["view"] == _SPD_VIEW]

    with style.apply_publication_style():
        # Wider than single (3.42in) so the curves spread apart horizontally,
        # but narrower than the full double-width preset (7.09in).
        fig, axes = style.figure_grid(1, 1, width=5.7, panel_height_in=3.8)
        ax = axes[0, 0]
        for trial_class in style.CLASS_ORDER:
            class_rows = view_rows.loc[view_rows["trial_class_name"] == trial_class].sort_values("threshold")
            color = style.CLASS_COLORS[trial_class]
            ax.plot(
                class_rows["threshold"], class_rows["mean_subject_proportion"],
                color=color, linewidth=1.6, label=trial_class,
            )
            ax.fill_between(
                class_rows["threshold"],
                class_rows["mean_subject_proportion"] - class_rows["sem_subject_proportion"],
                class_rows["mean_subject_proportion"] + class_rows["sem_subject_proportion"],
                color=color, alpha=0.20, linewidth=0,
            )
            ax.plot(
                class_rows["threshold"], class_rows["pooled_proportion"],
                color=color, linewidth=0.9, linestyle="--", alpha=0.7,
            )
        # Curves are flat/uninformative below 0.2 for every class; starting
        # there instead of 0 devotes the panel's width to the range that
        # actually separates the classes.
        ax.set_xlim(0.2, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0.2, 0.6, 1.0])
        ax.set_yticks([0.3, 0.6, 0.9])
        ax.set_xlabel("Logged SPD threshold")
        ax.set_ylabel("P(SPD ≤ threshold)")
        # Linestyle meaning as its own two legend rows (neutral gray, not
        # tied to a class color) instead of a text note explaining it --
        # matches how the class rows already key color.
        class_handles, class_labels = ax.get_legend_handles_labels()
        style_handles = [
            Line2D([0], [0], color=style.INK_MUTED, linewidth=1.6, linestyle="-", label="mean of subjects"),
            Line2D([0], [0], color=style.INK_MUTED, linewidth=0.9, linestyle="--", label="pooled across trials"),
        ]
        ax.legend(handles=class_handles + style_handles, loc="lower right", fontsize=10)

        # Δ + marker, not full t/p text: this is a single narrow (single-
        # column) panel, and "ambiguous vs misleading: t(31) = ..., p = ..."
        # for all three contrasts collapsed the axes to zero width -- same
        # bug class fixed on F04/F05, spelled-out stats don't fit a panel
        # this size regardless of figure.
        lines = []
        for first, second in (("easy", "ambiguous"), ("easy", "misleading"), ("ambiguous", "misleading")):
            row = view_stats.loc[view_stats["contrast"] == f"{first}_vs_{second}"]
            if len(row):
                result = stat_from_row(row.iloc[0], label=f"{first}_vs_{second}")
                marker = significance_marker(result.p) or "n.s."
                lines.append(f"{first}-{second}: Δ = {result.mean:.2f} {marker}")
        annotate_stat_block(ax, lines=lines, loc="upper left")

        fig.suptitle("SPD at decision, cumulative by class")

    metadata = {
        "kind": "cumulative_distribution",
        "title": "SPD at decision, cumulative by class",
        "columns_read": {
            "spdcumulative": [
                "view", "trial_class_name", "threshold", "n_trials", "n_subjects",
                "pooled_proportion", "mean_subject_proportion", "sem_subject_proportion",
            ],
            "groupstats": ["analysis", "view", "contrast", "mean_a", "mean_b", "t", "p", "df", "cohens_dz"],
        },
        "statistics_source": f"groupstats[analysis=spd,view={_SPD_VIEW}]",
        "palette": dict(style.CLASS_COLORS),
        "multiplicity_correction": "none",
        "view": _SPD_VIEW,
    }
    return fig, metadata
