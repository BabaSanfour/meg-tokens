"""Evidence and criterion figures: F09-F12."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from meg_tokens.behavior.analyses.evidence import DISPLAY_BIN_SUBJECT_FRACTION
from meg_tokens.reports import style
from meg_tokens.reports.annotations import (
    annotate_stat_block,
    is_significant,
    marker_or_missing,
)
from meg_tokens.reports.behavior._tables import BehaviorTableSet
from meg_tokens.reports.panels import (
    group_line,
    paired_slope,
    within_subject_error,
)

_CAF_STRATA: tuple[tuple[str, str], ...] = (
    ("easy", style.CLASS_COLORS["easy"]),
    ("ambiguous", style.CLASS_COLORS["ambiguous"]),
    ("misleading", style.CLASS_COLORS["misleading"]),
    ("unclassified", style.INK_MUTED),
)


def _subject_matrix(
    table: pd.DataFrame, *, x: str, values: str, order: Sequence, **filters: object
) -> np.ndarray:
    """Subject-by-`x` matrix of `values`, over the rows `filters` selects.

    F10's bins, F11's evidence levels and F12's token jumps are the same shape
    of question -- one row per subject and x, reindexed onto a fixed x order so
    a cell no subject reached is NaN rather than absent. `pivot` rather than
    `pivot_table`: a duplicated subject-by-x row is a broken derivative and
    should raise here, not be silently averaged.
    """
    subset = table
    for name, value in filters.items():
        subset = subset.loc[subset[name] == value]
    wide = subset.pivot(index="subject", columns=x, values=values)
    return wide.reindex(columns=list(order)).to_numpy(dtype=float)


def build_conditionalaccuracy_caf(tables: BehaviorTableSet) -> tuple[Figure, dict[str, Any]]:
    """F10: the conditional accuracy function, resolved by trial class.

    The pooled function falls with decision time, and read alone that says
    slow decisions are less accurate. It does not survive resolution: the
    decline is almost entirely *which trials* occupy each bin. Easy trials are
    fast and fill the first bin; ambiguous ones are slow and fill the last.
    Within the one class where early evidence actively misleads, accuracy
    climbs steeply with time -- the opposite sign to the pooled curve.

    A draws the four strata against the same bins. B draws the pooled function
    against its class-adjusted counterpart, the same construction as F04's
    panel B, so the gap between the two lines is the compositional part and
    what remains below it is not.

    x is the bin's mean decision time, the convention in this literature and
    the only axis that shows how unequal the quintile spacing is (678 to
    1949 ms). The annotated slopes are therefore per *second* of that axis;
    the per-quintile slopes are registered beside them in
    `conditionalaccuracystats` and quoted in `docs/behavior.md`.
    """
    functions = tables.analysis("conditionalaccuracy")
    caf_stats = tables.analysis("conditionalaccuracystats")

    pooled = functions.loc[
        (functions["condition"] == "all")
        & (functions["trial_class_name"] == "all")
    ]
    bins = sorted(int(value) for value in pooled["dt_bin"].unique())
    # One x for every series: the pooled bin's mean decision time. Each class
    # has its own mean within a bin (they differ by ~30 ms), but drawing four
    # slightly offset x-grids would imply a precision the binning does not
    # have and would stop the classes being readable at a common time.
    x_ms = (
        pooled.pivot_table(index="subject", columns="dt_bin", values="mean_dt_ms")
        .reindex(columns=bins)
        .mean(axis=0)
        .to_numpy(dtype=float)
    )

    def slope_line(label: str, stratum: str, measure: str = "accuracy") -> str | None:
        row = caf_stats.loc[
            (caf_stats["term"] == "accuracy_slope_per_second")
            & (caf_stats["measure"] == measure)
            & (caf_stats["condition"] == "all")
            & (caf_stats["trial_class_name"] == stratum)
            & (caf_stats["min_cell_trials"] == 1)
        ]
        if not len(row):
            return None
        row = row.iloc[0]
        # A null this figure argues from states what it excludes; a bare
        # 'n.s.' would let "ambiguous is flat" rest on a failure to detect.
        # An untested or unbounded stratum falls back to marker_or_missing's
        # 'n/a' rather than printing a formatted NaN as though it were one.
        if is_significant(row["p"]):
            marker = marker_or_missing(row["p"])
        elif np.isfinite(row["equivalence_bound"]):
            marker = f"|slope| < {row['equivalence_bound']:.3f}"
        else:
            marker = marker_or_missing(None)
        return f"{label}  {row['estimate']:+.3f} /s  {marker}"

    with style.apply_publication_style():
        fig, axes = style.figure_grid(1, 2, width="double", panel_height_in=3.4)
        ax_a, ax_b = axes[0, 0], axes[0, 1]
        for index, letter in enumerate("AB"):
            style.panel_label(axes[0, index], letter)

        # A -- by trial class. No per-subject traces: they span 0 to 1 in the
        # misleading stratum and flatten every class into one grey band.
        for stratum, colour in _CAF_STRATA:
            matrix = _subject_matrix(functions, x="dt_bin", values="accuracy", order=bins,
                                    condition="all", trial_class_name=stratum)
            if not np.isfinite(matrix).any():
                continue  # a stratum this table has no trials for
            group_line(
                ax_a, x=x_ms, subject_matrix=matrix,
                color=colour, label=stratum, error="within_ci95",
            )
        # Chance, not zero, is the reference a two-alternative accuracy is read
        # against -- and misleading trials start well below it. Labelled under
        # the line at the slow end, the one stretch of the panel no series and
        # no legend entry occupies.
        ax_a.axhline(0.5, color=style.INK_MUTED, linewidth=0.6, linestyle=":")
        ax_a.text(x_ms[-1], 0.485, "chance", va="top", ha="right",
                  color=style.INK_SECONDARY, fontsize=style.ANNOTATION_PT)
        ax_a.set_ylim(-0.02, 1.06)
        ax_a.set_yticks([0.0, 0.5, 1.0])
        ax_a.set_xlabel("Bin mean decision time (ms)")
        ax_a.set_ylabel("Accuracy")
        # In-axes and framed, the same style as every other figure here. The
        # gap between ambiguous and misleading is the only place it fits.
        ax_a.legend(loc="center left")
        # Two lines, not four. The legend already names every class, so
        # annotating all four repeated the names and printed two slopes whose
        # sign the panel shows anyway. What is left is the pair a reader
        # cannot get by looking: the class that reverses, and the null the
        # figure argues from. easy and unclassified keep their slopes in
        # `conditionalaccuracystats`.
        #
        # Lower right is the only free corner: upper right sits on the easy
        # and unclassified lines, and both left corners are taken by the
        # legend and by misleading's fast end.
        annotate_stat_block(
            ax_a,
            lines=[
                line for line in (
                    slope_line("misl.", "misleading"),
                    slope_line("ambig.", "ambiguous"),
                ) if line is not None
            ],
            loc="lower right",
        )

        # B -- the pooled function against its class-adjusted counterpart,
        # built exactly as F04's panel B: each trial has its subject's mean
        # accuracy for its own class removed and that subject's grand mean
        # added back, so the gap between the lines *is* composition.
        for column, label, colour, linestyle in (
            ("accuracy", "observed", style.INK, "-"),
            ("accuracy_class_adjusted", "class-adjusted", style.INK_SECONDARY, "--"),
        ):
            group_line(
                ax_b, x=x_ms,
                subject_matrix=_subject_matrix(
                    functions, x="dt_bin", values=column, order=bins,
                    condition="all", trial_class_name="all",
                ),
                color=colour, label=label, error="within_ci95",
            )
            if linestyle != "-":
                ax_b.get_lines()[-1].set_linestyle(linestyle)
        # The drawn bands span 0.759 to 0.912, so these limits clear them by
        # under a hundredth. The old ceiling of 0.94 was headroom for a third
        # annotation line that is no longer there; the annotation now needs
        # clearance only on the right, where the highest band reaches 0.863.
        ax_b.set_ylim(0.75, 0.92)
        ax_b.set_yticks([0.80, 0.85, 0.90])
        ax_b.set_xlabel("Bin mean decision time (ms)")
        ax_b.set_ylabel("Accuracy")
        # Four ticks inside the drawn range (678-1949 ms). The default six ran
        # their labels together at this panel width.
        for ax in (ax_a, ax_b):
            ax.set_xticks([700, 1100, 1500, 1900])
        # Tighter handles and padding, the same fix F07's panel B uses: with
        # the panel pulled in to the bands, a default-width box at lower left
        # reaches far enough right to sit on the observed line's tail. The
        # free region is under the class-adjusted line on the fast side.
        ax_b.legend(loc="lower left", handlelength=1.3,
                    handletextpad=0.4, borderpad=0.4, labelspacing=0.3)
        # The two slopes are the panel's whole content. The composition shift
        # that produces the gap between them used to be a third line here; it
        # is the explanation rather than the result, and it belongs in the
        # sidecar and `docs/behavior.md` where it has room to be stated
        # properly.
        annotate_stat_block(
            ax_b,
            lines=[
                line for line in (
                    slope_line("observed", "all"),
                    slope_line("adjusted", "all", measure="accuracy_class_adjusted"),
                ) if line is not None
            ],
            loc="upper right",
        )

        fig.suptitle("Conditional accuracy")

    metadata = {
        "kind": "conditional_accuracy_function",
        "title": "Conditional accuracy function, by trial class",
        "columns_read": {
            "conditionalaccuracy": [
                "subject", "condition", "trial_class_name", "dt_bin",
                "mean_dt_ms", "accuracy", "accuracy_class_adjusted",
            ],
            "conditionalaccuracystats": [
                "term", "measure", "condition", "trial_class_name",
                "min_cell_trials", "estimate", "p", "equivalence_bound",
                "first_bin_proportion", "last_bin_proportion",
            ],
        },
        "statistics_source": (
            "conditionalaccuracystats[term=accuracy_slope_per_second, "
            "min_cell_trials=1]"
        ),
        "palette": {**style.CLASS_COLORS, "unclassified": style.INK_MUTED},
        "caveat": (
            "Bins are within-subject decision-time quintiles cut on all of a "
            "subject's trials, so the four class series are read at the same "
            "times and partition the pooled series exactly. The class "
            "gap between B's two lines is that composition: easy falls from "
            "43% to 10% of a bin between the fastest and slowest quintile "
            "while ambiguous rises from 8% to 34% "
            "(term=class_proportion_first_to_last). The class "
            "adjustment removes between-class composition only: evidence "
            "strength still varies inside a class, most of all inside "
            "'unclassified', so B's residual adjusted slope is an upper bound "
            "on any genuine accuracy-versus-time relation, not an estimate of "
            "one. 'unclassified' is drawn because it is load-bearing, not for "
            "completeness: it carries the whole within-class part of the "
            "first-to-last change, and without it the three designed classes "
            "sum to +0.038 (p = .0015), i.e. accuracy rising with decision "
            "time. See term=accuracy_change_decomposition. Slopes are "
            "annotated per second of the drawn axis; the per-quintile values "
            "are registered under term=accuracy_slope_per_bin."
        ),
    }
    return fig, metadata


# F11 draws the three classes the design names. `unclassified` is a residual
# category, and once the figure stopped needing it for the 0.81 comparison it
# had no result of its own to carry here. The analysis still computes it --
# `evidence_contrast_within_class` for `unclassified` is what F10 cites when
# it calls its class-adjusted residual an upper bound -- so nothing leaves the
# derivative, only the panels. F10 still draws all four, because there the
# pooled curve is a mixture that cannot be explained without it.
_EARLY_EVIDENCE_STRATA: tuple[tuple[str, str], ...] = tuple(
    entry for entry in _CAF_STRATA if entry[0] != "unclassified"
)
# Each class keeps the same slot inside every cluster, so a class missing from
# a level leaves a visible gap rather than shuffling its neighbours along --
# and the connectors between levels stay straight.
_CLUSTER_OFFSETS: dict[str, float] = {
    "easy": -0.16, "ambiguous": 0.0, "misleading": 0.16,
}
# The correct-target token lead each success probability corresponds to, in
# level order. Three tokens admit exactly these four splits.
_LEADS: tuple[int, ...] = (-3, -1, 1, 3)

# The class pairs the drawn strata can form. `class_contrast_at_level` also
# holds pairs involving `unclassified`, which this figure does not draw, so
# the cluster marker has to select rather than take whatever row it finds.
_DRAWN_PAIRS: frozenset[str] = frozenset({
    "ambiguous_vs_easy", "ambiguous_vs_misleading", "easy_vs_misleading",
})


def _span(ax) -> float:
    """Height of the axis in data units, for offsetting a marker off a point."""
    low, high = ax.get_ylim()
    return high - low


def _level_cluster(profile: pd.DataFrame, level: float, measure: str):
    """Subject x class matrix at one evidence level, classes present only."""
    subset = profile.loc[
        (profile["condition"] == "all")
        & np.isclose(profile["sp_design_early"], level)
        & (profile["trial_class_name"] != "all")
    ]
    wide = subset.pivot_table(
        index="subject", columns="trial_class_name", values=measure
    )
    order = [name for name, _ in _EARLY_EVIDENCE_STRATA if name in wide.columns]
    if not order:
        return [], np.empty((0, 0))
    return order, wide[order].to_numpy(dtype=float)


def build_continuousevidence_effects(tables: BehaviorTableSet) -> tuple[Figure, dict[str, Any]]:
    """F11: what the early read of the evidence does and does not determine.

    Early evidence is read after three tokens, and three tokens admit exactly
    four splits -- so `sp_design_early` takes exactly four values, 0.19/0.39/
    0.61/0.81, and nothing about this predictor is continuous. A coefficient
    "per unit" fitted on it is a contrast between two of those levels; the
    registered `n_predictor_levels` says so, and this figure draws the levels
    instead.

    Resolving them by trial class answers the question the coefficient hides.
    The classification rule reads the success-probability profile at jumps 5, 8
    and 11 as well as 3, so trials that are indistinguishable after three
    tokens go on to differ. A: decision time at each level, by class. B:
    accuracy the same way. The error bars are within-subject across the
    classes in a cluster, so they already encode that the class gaps are
    paired comparisons rather than between-subject differences; a per-subject
    raincloud of the SP = 0.61 level said the same thing again and cost a
    panel, so the effect sizes carry it instead (`cohens_dz` on every
    `class_contrast_at_level` row; ambiguous minus easy is dz = 2.80).

    `unclassified` is drawn for one result: at the 0.81 level it and easy are
    indistinguishable in accuracy (the data exclude any difference above
    0.008) while still differing by 161 ms in decision time. Everywhere else
    the classes differ on both measures at once, so that is the only place
    accuracy and timing come apart -- the sharpest form of this figure's
    claim, and no other pair of classes is matched on accuracy to supply it.
    It is drawn in neutral grey rather than a class colour, because it is a
    residual category and not a fourth designed peer.
    """
    profile = tables.analysis("continuousevidenceprofile")
    stats_table = tables.analysis("continuousevidencestats")

    pooled = profile.loc[
        (profile["condition"] == "all") & (profile["trial_class_name"] == "all")
    ]
    # Every level is drawn. An earlier version hid the 0-3 level behind the
    # cohort-fraction rule F09 uses for its time bins, which was a threshold
    # borrowed from a different figure and never re-justified here -- and it
    # silently removed one of misleading's two levels. The trial count under
    # each cluster does that job honestly instead: 12 against 11,196 needs no
    # rule to be discounted.
    levels = sorted(float(level) for level in pooled["sp_design_early"].unique())
    # Counted over the classes actually drawn, not over the level: the pooled
    # row includes `unclassified`, which this figure does not show, and a tick
    # advertising trials the panel never plots is worse than no tick at all.
    drawn = profile.loc[
        (profile["condition"] == "all")
        & profile["trial_class_name"].isin(
            [name for name, _ in _EARLY_EVIDENCE_STRATA]
        )
    ]
    level_trials = drawn.groupby("sp_design_early")["n_trials"].sum()

    def cluster_marker(level: float, measure: str) -> str | None:
        """Significance marker for the one class contrast at this level."""
        rows = stats_table.loc[
            (stats_table["term"] == "class_contrast_at_level")
            & (stats_table["measure"] == measure)
            & np.isclose(stats_table["sp_design_early"], level)
            & stats_table["trial_class_name"].isin(_DRAWN_PAIRS)
        ]
        if len(rows) != 1:
            return None
        return marker_or_missing(rows.iloc[0]["p"])

    with style.apply_publication_style():
        fig, axes = style.figure_grid(1, 2, width="double", panel_height_in=3.4)
        for index, letter in enumerate("AB"):
            style.panel_label(axes[0, index], letter)
        ax_a, ax_b = axes[0, 0], axes[0, 1]

        # A and B are clustered by evidence level, not drawn as lines over it.
        # The claim is "at the same early evidence, class still decides the
        # trial", and a line per class puts that comparison at the *ends* of
        # lines -- ambiguous stops at 0.61 exactly where easy and unclassified
        # start, so the reader has to compare endpoints across a gap. Clustered,
        # the comparison is the vertical spread inside one cluster, and the
        # faint connectors carry the second, weaker claim (what a class does
        # across levels) without competing for attention.
        x = np.arange(len(levels), dtype=float)
        # Limits are set before anything is drawn, because the cluster markers
        # are offset by a fraction of the axis height and would otherwise be
        # placed against matplotlib's autoscale rather than the final scale.
        # Each ceiling sits just above the tallest marker; the top tick is the
        # last round value below the data, as in F09.
        for ax, measure, ylabel, ylim, yticks in (
            (ax_a, "mean_dt_ms", "Mean decision time (ms)",
             (720, 1560), [800, 1000, 1200, 1400]),
            (ax_b, "accuracy", "Accuracy",
             (-0.05, 1.08), [0.0, 0.5, 1.0]),
        ):
            ax.set_ylim(*ylim)
            ax.set_yticks(yticks)
            for stratum, colour in _EARLY_EVIDENCE_STRATA:
                matrix = _subject_matrix(
                    profile, x="sp_design_early", values=measure, order=levels,
                    condition="all", trial_class_name=stratum,
                )
                present = np.isfinite(matrix).any(axis=0)
                if not present.any():
                    continue
                offset = _CLUSTER_OFFSETS[stratum]
                means = np.nanmean(matrix, axis=0)
                # Faint connector first, so markers sit on top of it.
                ax.plot(x[present] + offset, means[present], color=colour,
                        linewidth=1.0, alpha=0.45, zorder=1)
                ax.plot([], [], color=colour, marker="o", label=stratum)
            for index, level in enumerate(levels):
                names, cluster = _level_cluster(profile, level, measure)
                if not names:
                    continue
                means = np.nanmean(cluster, axis=0)
                # The error bar supports the comparison the cluster makes:
                # between classes, within subject. A level holding one class
                # has no such comparison, so it carries no bar.
                complete = cluster[np.all(np.isfinite(cluster), axis=1)]
                band = (
                    within_subject_error(complete, kind="ci95")
                    if cluster.shape[1] > 1 and complete.shape[0] > 1
                    else np.zeros(len(names))
                )
                for position, name in enumerate(names):
                    ax.errorbar(
                        index + _CLUSTER_OFFSETS[name], means[position],
                        yerr=band[position], color=dict(_EARLY_EVIDENCE_STRATA)[name],
                        marker="o", markersize=5, capsize=2, linewidth=1.2,
                        zorder=3,
                    )
                # The cluster's class contrast as a marker over the cluster,
                # rather than a block of text in a corner. With three designed
                # classes each level holds at most one pair, so one marker per
                # cluster is unambiguous. The effect sizes stay in
                # `continuousevidencestats` and in docs/behavior.md.
                marker = cluster_marker(level, measure)
                if marker is not None:
                    ax.text(index, np.nanmax(means + band) + 0.04 * _span(ax),
                            marker, ha="center", va="bottom", fontsize=style.SIGNIFICANCE_PT,
                            color=style.INK)
            # Just the quantity; the tick's second and third lines are the
            # token lead and the trial count, and spelling both out here made
            # the label wider than the panel.
            ax.set_xlabel("SP after 3 tokens")
            ax.set_ylabel(ylabel)
            ax.set_xlim(-0.5, len(levels) - 0.5)
            ax.set_xticks(x)
            # Probability, the token lead it corresponds to, and the trials
            # behind it -- all three in the tick, so the retained n is visible
            # without a hand-offset row that constrained_layout cannot see and
            # so drops on top of the axis label.
            ax.set_xticklabels([
                f"{level:.2f}\n({lead:+d})\n{int(level_trials[level])}"
                for level, lead in zip(levels, _LEADS)
            ])
            # Three short lines per tick need a smaller face than the default
            # 13pt, or the trial counts of adjacent clusters run together.
            ax.tick_params(axis="x", labelsize=11)

        # Chance is the reference an accuracy is read against, as in F10.
        ax_b.axhline(0.5, color=style.INK_MUTED, linewidth=0.6, linestyle=":")
        ax_b.text(0.02, 0.52, "chance", transform=ax_b.get_yaxis_transform(),
                  va="bottom", ha="left", color=style.INK_SECONDARY, fontsize=style.ANNOTATION_PT)
        # The class legend lives in B, not A. Both panels draw the same series,
        # and A has no free region tall enough. B's lower right is empty --
        # nothing reaches below 0.7 there.
        ax_b.legend(loc="lower right")

        fig.suptitle("Early evidence by trial class")

    metadata = {
        "kind": "early_evidence_by_class",
        "title": "Early evidence by trial class",
        "columns_read": {
            "continuousevidenceprofile": [
                "subject", "condition", "trial_class_name", "sp_design_early",
                "n_trials", "mean_dt_ms", "accuracy",
            ],
            "continuousevidencestats": [
                "term", "measure", "trial_class_name", "sp_design_early",
                "mean_difference", "p", "partial_eta_squared",
                "equivalence_bound",
            ],
        },
        "statistics_source": (
            "continuousevidencestats[term in {class_omnibus_at_level, "
            "class_contrast_at_level, evidence_contrast_within_class}]"
        ),
        "palette": {**style.CLASS_COLORS, "unclassified": style.INK_MUTED},
        "includes_unclassified": True,
        "caveat": (
            "Early evidence is read after 3 of 15 tokens, which is where the "
            "classification rule first looks and where 71% of trials are still "
            "running. Three tokens admit four splits, so this predictor has "
            "exactly four levels and is not continuous; the per-unit "
            "coefficients in `continuousevidence` are two- and four-point "
            "contrasts, and carry `n_predictor_levels` saying so. Every level "
            "is drawn, including the 0-3 split at SP = 0.19 that holds only 12 "
            "trials: each tick carries its probability, the correct-target "
            "token lead it corresponds to, and the trials behind it, so a thin "
            "level is discounted by the reader rather than removed by a "
            "threshold. A and B are clustered by level rather than drawn as "
            "lines over it, because the claim is a comparison *within* a "
            "level -- the error bars are within-subject across the classes in "
            "that cluster, and the faint connectors carry the weaker "
            "across-level claim. A cluster holding one class makes no such "
            "comparison and carries no bar. "
            "`sum_log_lr_design_early` is exactly "
            "logit(sp_design_early) -- the exact posterior, not the "
            "first-order SumLogLR of Cisek et al. (2009); the two disagree in "
            "sign on this dataset."
        ),
    }
    return fig, metadata


def build_reversecorrelation_kernel(tables: BehaviorTableSet) -> tuple[Figure, dict[str, Any]]:
    """F12: the psychophysical kernel by token jump.

    A shows the kernel itself, pooled and per condition: how much each of the
    first eight jumps weighs on the choice. B shows the Fast-minus-Slow
    difference at each jump, which is where the condition claim lives.

    Every number printed here is read from `reversecorrelationstats`. The
    previous version recovered the difference's standard error in figure code
    as `mean / t` -- which yields nothing when a difference is exactly zero,
    the case a kernel figure most needs to draw -- and reduced a column of
    p-values itself to assert that every weight differs from zero. Both are
    registered now.

    B's markers use the Bonferroni threshold for the eight jumps tested, not
    0.05. At 0.05 the panel marked jumps 1-5; corrected, jumps 1 and 5 do not
    survive, and marking them would have overstated where Fast and Slow
    actually differ.
    """
    kernels = tables.analysis("reversecorrelation")
    kernel_stats = tables.analysis("reversecorrelationstats")

    weight_stats = kernel_stats.loc[kernel_stats["metric"] == "logistic_weight"]
    jumps = sorted(kernels["jump"].unique())
    x = np.array(jumps, dtype=float)

    with style.apply_publication_style():
        fig, axes = style.figure_grid(1, 2, width="double", panel_height_in=3.4)
        ax_a, ax_b = axes[0, 0], axes[0, 1]
        for index, letter in enumerate("AB"):
            style.panel_label(axes[0, index], letter)

        # A -- no per-subject traces. One subject runs to -3 while the group
        # lines sit between 1.2 and 2.7, so the traces set the y-range and
        # flatten the three series they are meant to contextualise.
        group_line(ax_a, x=x, subject_matrix=_subject_matrix(kernels, x="jump", values="logistic_weight",
                                    order=jumps, condition="all"),
                   color=style.INK, label="all", error="within_ci95")
        for condition in style.CONDITION_ORDER:
            group_line(ax_a, x=x, subject_matrix=_subject_matrix(kernels, x="jump", values="logistic_weight",
                                        order=jumps, condition=condition),
                       color=style.CONDITION_COLORS[condition], label=condition,
                       error="within_ci95")
        ax_a.axhline(0, color=style.INK_MUTED, linewidth=0.6, zorder=0)
        ax_a.set_xlabel("Token jump")
        ax_a.set_ylabel("Logistic weight")
        ax_a.set_xticks(jumps)
        ax_a.legend(loc="upper right")

        # The "every weight differs from zero" line is a registered row, not a
        # column of p-values reduced here.
        summary = weight_stats.loc[
            (weight_stats["test"] == "all_jumps_vs_zero")
            & (weight_stats["condition"] == "all")
        ]
        if len(summary):
            row = summary.iloc[0]
            # Just above the zero line it is testing against, rather than in
            # the corner where it sat on the axis spine.
            ax_a.text(
                jumps[0] - 0.3, 0.08,
                f"{int(row['n_jumps_significant'])}/{int(row['n_jumps_tested'])} "
                f"jumps differ from zero",
                ha="left", va="bottom", fontsize=style.ANNOTATION_PT, color=style.INK_SECONDARY,
            )
        # Zero is kept in view because every weight is tested against it, but
        # the floor sits just below rather than at it, so the label above the
        # line is not clipped.
        ax_a.set_ylim(-0.2, 3.4)
        ax_a.set_yticks([0, 1, 2, 3])

        # B -- the Fast-minus-Slow difference, with the interval and the
        # standard error both taken from the derivative.
        differences = weight_stats.loc[
            weight_stats["condition"] == "fast_vs_slow"
        ].sort_values("jump")
        jump_positions = differences["jump"].to_numpy(dtype=float)
        means = differences["mean_difference"].to_numpy(dtype=float)
        low = differences["ci_low"].to_numpy(dtype=float)
        high = differences["ci_high"].to_numpy(dtype=float)
        ax_b.axhline(0, color=style.INK_MUTED, linewidth=0.6, zorder=0)
        ax_b.errorbar(
            jump_positions, means, yerr=[means - low, high - means],
            color=style.INK, marker="o", markersize=4, linewidth=1.4, capsize=2,
        )
        # Corrected threshold, and a bound wherever a jump does not clear it --
        # "identical from jump 6" is a null this panel argues from.
        threshold = float(differences["p_threshold"].iloc[0]) if len(differences) else 0.05
        for _, row in differences.iterrows():
            if np.isfinite(row["p"]) and row["p"] < threshold:
                ax_b.scatter([row["jump"]], [row["mean_difference"]], s=70,
                             facecolors="none", edgecolors=style.INK, zorder=4)
        ax_b.set_xlabel("Token jump")
        ax_b.set_ylabel("Fast − Slow weight")
        ax_b.set_xticks(jumps)
        # Intervals span -0.60 to +1.00; the ceiling leaves the two-line
        # threshold note clear of the jump-1 and jump-3 bars beneath it.
        ax_b.set_ylim(-0.72, 1.32)
        ax_b.set_yticks([-0.5, 0.0, 0.5, 1.0])
        # Upper left: the differences fall from left to right, so the top-left
        # corner is the one clear of every error bar.
        ax_b.text(
            0.02, 0.98,
            f"ringed: p < {threshold:.4f}\n(Bonferroni, {len(differences)} jumps)",
            transform=ax_b.transAxes, ha="left", va="top",
            fontsize=style.ANNOTATION_PT, color=style.INK_SECONDARY, linespacing=1.3,
        )

        fig.suptitle("Psychophysical kernel")

    metadata = {
        "kind": "kernel_by_jump",
        "title": "Psychophysical kernel",
        "columns_read": {
            "reversecorrelation": ["subject", "condition", "jump", "logistic_weight"],
            "reversecorrelationstats": [
                "metric", "condition", "jump", "test", "mean_difference",
                "ci_low", "ci_high", "p", "p_threshold", "n_jumps_tested",
                "n_jumps_significant",
            ],
        },
        "statistics_source": "reversecorrelationstats[metric=logistic_weight]",
        "palette": dict(style.CONDITION_COLORS),
        "caveat": (
            "B's rings mark the Bonferroni threshold for the eight jumps "
            "tested, not 0.05; at 0.05 jumps 1 and 5 would also be marked. "
            "Error bars are 95% intervals on the paired difference, taken from "
            "`reversecorrelationstats` -- the standard error of a difference is "
            "`sem_difference`, not the `sem_a`/`sem_b` of the two conditions. "
            "A subject whose logistic fit did not converge is reported as "
            "non-converged in `reversecorrelation` rather than drawn as a "
            "large arbitrary weight."
        ),
    }
    return fig, metadata


_SCALE_LABELS: dict[str, str] = {
    "first_order": "First-order SumLogLR",
    "exact_log_odds": "Exact posterior (log odds)",
}


def _profile_matrix(profile: pd.DataFrame, scale: str):
    """Subject × time-bin matrix for one scale, on the bins the analysis tests.

    Display rule: any bin held by at least half the cohort. The omnibus test
    uses a stricter rule (see `TEST_BIN_SUBJECT_FRACTION`) because a
    repeated-measures ANOVA is listwise; the drawn range is therefore wider
    than the tested one.
    """
    wide = profile.loc[profile["scale"] == scale].pivot_table(
        index="subject", columns="time_bin_center_s", values="criterion"
    )
    threshold = int(np.ceil(DISPLAY_BIN_SUBJECT_FRACTION * wide.shape[0]))
    wide = wide.loc[:, wide.notna().sum() >= threshold]
    return wide.columns.to_numpy(dtype=float), wide.to_numpy(dtype=float)


def build_criterionscale_comparison(
    tables: BehaviorTableSet,
) -> tuple[Figure, dict[str, Any]]:
    """F09: the evidence criterion against decision time, on both scales.

    Replaces the two figures that measured the same commitments with different
    scorings and disagreed. Drawing them apart invited each to be read as a
    result; drawing them together makes the disagreement the result, which is
    what the data actually support.

    A and B are the same 10,849 trials and the same subjects. Only the
    definition of "evidence at commitment" differs, so the reversal between
    them is attributable to the scale and to nothing else. C shows it holding
    within subjects rather than only in the group mean.
    """
    fits = tables.analysis("criterionscale")
    profile = tables.analysis("criterionscaleprofile")
    stats_table = tables.analysis("criterionscalestats")

    with style.apply_publication_style():
        fig, axes = style.figure_grid(1, 3, width="double", panel_height_in=3.4)
        for index, letter in enumerate("ABC"):
            style.panel_label(axes[0, index], letter)

        for index, scale in enumerate(("first_order", "exact_log_odds")):
            ax = axes[0, index]
            centres, matrix = _profile_matrix(profile, scale)
            group_line(ax, x=centres, subject_matrix=matrix,
                       color=style.INK, label="observed", error="within_ci95")
            # The fitted line is the group mean of the per-subject trial-level
            # fits -- not a fit to the binned means, which would weight a
            # sparse late bin like a dense early one.
            scale_fits = fits.loc[
                (fits["scale"] == scale) & (fits["condition"] == "all")
            ]
            intercept = float(np.nanmean(scale_fits["intercept"]))
            slope = float(np.nanmean(scale_fits["slope"]))
            # No colour scope in this figure: the condition and model palettes
            # both mean something else, and a new scope would need the same
            # CVD validation the others carry. Black is the observed series and
            # dark grey the derived one, in all three panels -- so C's two
            # columns cannot be read as A and B's fitted line.
            ax.plot(centres, intercept + slope * centres,
                    color=style.INK_SECONDARY, linewidth=1.6,
                    linestyle="--", label="mean subject fit")
            ax.set_xlabel("Decision time (s)")
            ax.set_ylabel(_SCALE_LABELS[scale])
            # Ticks at the readable round values; limits sit just outside the
            # drawn CI band rather than on the requested round numbers, so no
            # band is clipped -- A's last bin reaches 1.19 and B's 4.85.
            if scale == "first_order":
                ax.set_ylim(0.15, 1.25)
                ax.set_yticks([0.2, 0.6, 1.0])
            else:
                ax.set_ylim(0.15, 4.95)
                ax.set_yticks([1, 2, 3, 4])
            ax.set_xticks([1, 2, 3])

            row = stats_table.loc[
                (stats_table["scale"] == scale) & (stats_table["term"] == "slope")
            ]
            contrast = stats_table.loc[
                (stats_table["scale"] == scale)
                & (stats_table["term"] == "fast_minus_slow")
            ]
            lines = []
            if len(row):
                r = row.iloc[0]
                lines.append(
                    f"{r['estimate']:+.3f} /s  {marker_or_missing(r['p'])}"
                )
            if len(contrast):
                c = contrast.iloc[0]
                # Bounded null: the interval, not a bare marker.
                lines.append(f"Fast − Slow  |Δ| < {c['equivalence_bound']:.3f}")
            # Both blocks upper-left: A's data peaks below 0.9 and B's rises
            # to the right, so the top-left corner is the one free on both.
            # Lower-left put A's second line on top of the zero reference.
            annotate_stat_block(ax, lines=lines, loc="upper left")
            if index == 0:
                ax.legend(loc="lower right")

        # C -- the reversal, within subject.
        ax_c = axes[0, 2]
        wide = fits.loc[fits["condition"] == "all"].pivot_table(
            index="subject", columns="scale", values="slope"
        )
        values_a = wide["first_order"].to_numpy(dtype=float)
        values_b = wide["exact_log_odds"].to_numpy(dtype=float)
        paired_slope(
            ax_c, values_a=values_a, values_b=values_b,
            label_a="First-order", label_b="Exact",
            color_a=style.INK, color_b=style.INK_SECONDARY,
            ylabel="Criterion slope (/s)", reference=0.0,
        )
        ax_c.set_ylim(-0.75, 1.35)
        ax_c.set_yticks([-0.6, -0.2, 0.2, 0.6, 1.0])
        # Bracket spanning the two columns, the same idiom as the other paired
        # contrasts. Drawn inline rather than via significance_bracket, which
        # scales y by 0.98/1.02 and so misplaces itself on an axis crossing
        # zero -- the same reason F02 draws its own.
        difference = stats_table.loc[stats_table["term"] == "slope_difference"]
        correlation = stats_table.loc[stats_table["term"] == "slope_correlation"]
        if len(difference):
            d = difference.iloc[0]
            top = float(np.nanmax(np.concatenate([values_a, values_b])))
            bracket_y = top + 0.10
            ax_c.plot([0, 0, 1, 1],
                      [bracket_y - 0.04, bracket_y, bracket_y, bracket_y - 0.04],
                      color=style.INK, linewidth=0.8)
            label = f"Δ = {d['estimate']:+.3f} {marker_or_missing(d['p'])}"
            if len(correlation):
                label += f"\nr = {correlation.iloc[0]['estimate']:.2f}"
            ax_c.text(0.5, bracket_y + 0.05, label, ha="center", va="bottom",
                      fontsize=style.ANNOTATION_PT, color=style.INK, linespacing=1.25)

        fig.suptitle("Evidence criterion vs. decision time, on two evidence scales")

    metadata = {
        "kind": "criterion_scale_comparison",
        "title": "Evidence criterion vs. decision time, on two evidence scales",
        "columns_read": {
            "criterionscale": ["subject", "condition", "scale", "intercept", "slope"],
            "criterionscaleprofile": ["subject", "scale", "time_bin_center_s", "criterion"],
            "criterionscalestats": [
                "scale", "term", "estimate", "p", "equivalence_bound",
            ],
        },
        "statistics_source": "criterionscalestats",
        "caveat": (
            "Both panels are the same 10,849 trials: complete 15-token logs "
            "with a post-first-token commitment, the only trials on which "
            "either scale is defined. The exact posterior is read off a grid "
            "that coarsens late in a trial, so part of B's positive slope is "
            "discretization; neither scale identifies the sign on its own."
        ),
    }
    return fig, metadata
