"""Individual-differences figures: F18."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from scipy import stats as scipy_stats

from meg_tokens.reports import style
from meg_tokens.reports.annotations import (
    ALPHA, format_p, format_r, marker_or_missing,
)
from meg_tokens.reports.behavior._tables import BehaviorTableSet
from meg_tokens.reports.panels import correlation_heatmap, scatter_fit

_PROFILE_MEASURES = (
    "mean_dt_ms", "sat_adjustment_ms", "urgency_slope_per_second",
    "criterion_slope_sum_log_lr_per_second", "accuracy_log_odds_per_unit",
    "percent_correct", "lapse_rate",
)

# Reader-facing names. The derivative columns are identifiers, and a figure
# labelled `criterion_slope_sum_log_lr_per_second` is asking its reader to know
# the schema.
#
# One vocabulary for the whole figure: the matrix row that says "SAT adj." and
# the scatter axis that says "SAT adj. (ms)" have to be recognisably the same
# measure, because the scatters *are* cells of the matrix. Earlier these
# disagreed ("SAT adjustment" against "SAT adj."), which is precisely the seam
# a reader has to cross to use the figure.
#
# They are also short on purpose. The matrix's column labels are rotated 45
# degrees, so a label's length costs the heatmap height as well as width at
# 1/sqrt(2) each; trimming "Evidence sensitivity" to "Evidence sens." gives the
# drawing back most of what the labels were taking.
_MEASURE_LABELS = {
    "mean_dt_ms": "Mean DT",
    "sat_adjustment_ms": "SAT adj.",
    "urgency_slope_per_second": "Urgency slope",
    "criterion_slope_sum_log_lr_per_second": "Criterion slope",
    "accuracy_log_odds_per_unit": "Evidence sens.",
    "percent_correct": "Accuracy",
    "lapse_rate": "Lapse rate",
}

# The same names with their units, for the scatter axes. The matrix carries no
# units: a correlation has none, and repeating them in every cell label is
# noise. Full definitions are in docs/behavior.md.
_MEASURE_UNITS = {
    "mean_dt_ms": "ms",
    "sat_adjustment_ms": "ms",
    "urgency_slope_per_second": "/s",
    "criterion_slope_sum_log_lr_per_second": "SumLogLR/s",
    "accuracy_log_odds_per_unit": "log odds",
    "percent_correct": "%",
    "lapse_rate": "",
}


def _axis_label(measure: str) -> str:
    unit = _MEASURE_UNITS[measure]
    return f"{_MEASURE_LABELS[measure]} ({unit})" if unit else _MEASURE_LABELS[measure]


def _correlation_matrices(
    correlations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Square, **symmetric** r and p matrices over :data:`_PROFILE_MEASURES`.

    The derivative stores each pair once, in whichever ``measure_a``/
    ``measure_b`` order the analysis emitted. Writing that single orientation
    into the matrix leaves the transposed cell empty, and since the heatmap
    draws one triangle, every pair whose stored order falls in the *other*
    triangle vanishes -- silently, as a blank cell that reads "not measured"
    rather than "measured, near zero". Four of the twenty-one pairs went that
    way, all of them `percent_correct`.

    Filling both orientations is the fix rather than reversing the lookup at
    the point of use: a correlation matrix *is* symmetric, so the structure
    should be, and then it cannot matter which triangle a caller chooses to
    draw.
    """
    r_matrix = pd.DataFrame(
        index=_PROFILE_MEASURES, columns=_PROFILE_MEASURES, dtype=float
    )
    p_matrix = pd.DataFrame(
        index=_PROFILE_MEASURES, columns=_PROFILE_MEASURES, dtype=float
    )
    for measure in _PROFILE_MEASURES:
        r_matrix.loc[measure, measure] = 1.0
        p_matrix.loc[measure, measure] = 0.0
    for _, row in correlations.iterrows():
        a, b = row["measure_a"], row["measure_b"]
        if a in _PROFILE_MEASURES and b in _PROFILE_MEASURES:
            for first, second in ((a, b), (b, a)):
                r_matrix.loc[first, second] = row["pearson_r"]
                p_matrix.loc[first, second] = row["pearson_p"]
    return r_matrix, p_matrix


# How many cells of the matrix get magnified. Five fills the 3 x 2 grid the
# matrix leaves.
SCATTER_PANEL_COUNT: int = 5


def _scatter_pairs(
    correlations: pd.DataFrame, *, limit: int = SCATTER_PANEL_COUNT
) -> list[tuple[str, str]]:
    """The strongest significant pairs, largest |r| first.

    A rule read off the derivative, not a hand-picked list. Which cells deserve
    magnifying is a property of the data, so hard-coding them means the figure
    silently keeps drawing yesterday's answer -- under a subject-exclusion list,
    say, or any refit -- while the matrix beside it shows today's.

    Ordering by |r| also orders the panels: the reader meets the strongest
    relation first and the weakest last, which is the order the write-up argues
    them in.

    Selecting on significance is selection on the outcome, and it is worth
    being clear what that costs: B-F are guaranteed to be a gallery of
    significant results, so the panels alone say nothing about how *many* pairs
    reached significance. A answers that -- it draws all 21 -- which is the
    other reason the two halves belong in one figure.
    """
    numeric_p = pd.to_numeric(correlations["pearson_p"], errors="coerce")
    numeric_r = pd.to_numeric(correlations["pearson_r"], errors="coerce")
    usable = correlations.loc[
        correlations["measure_a"].isin(_PROFILE_MEASURES)
        & correlations["measure_b"].isin(_PROFILE_MEASURES)
        & numeric_p.notna() & (numeric_p < ALPHA)
        & numeric_r.notna()
    ].copy()
    # Recomputed from the selected rows rather than aligned in from a Series
    # built over the whole table: the two carry different indices whenever the
    # derivative's own index is not a clean range, and the aligned version then
    # yields rows whose measure names are NaN.
    usable["magnitude"] = pd.to_numeric(
        usable["pearson_r"], errors="coerce"
    ).abs()
    # Measure names break ties so the same table always yields the same panels.
    ordered = usable.sort_values(
        ["magnitude", "measure_a", "measure_b"], ascending=[False, True, True]
    ).head(limit)
    return [
        _orient(measure_a, measure_b)
        for measure_a, measure_b in zip(
            ordered["measure_a"], ordered["measure_b"]
        )
    ]


def _orient(measure_a: str, measure_b: str) -> tuple[str, str]:
    """Which of a pair goes on x. A correlation's axes are interchangeable.

    Mean DT whenever it is in the pair, so every panel that has it reads as
    another view of the same speed factor rather than an unrelated plane.
    Otherwise the longer label goes on x, where labels run horizontally and
    have room: on y the longest of them sits exactly where the panel letter
    goes.
    """
    if measure_a == "mean_dt_ms" or measure_b == "mean_dt_ms":
        other = measure_b if measure_a == "mean_dt_ms" else measure_a
        return "mean_dt_ms", other
    if len(_axis_label(measure_b)) > len(_axis_label(measure_a)):
        return measure_b, measure_a
    return measure_a, measure_b


def _correlation_row(correlations: pd.DataFrame, measure_a: str, measure_b: str):
    """The derivative row for a pair, in whichever order it was stored."""
    rows = correlations.loc[
        (
            (correlations["measure_a"] == measure_a)
            & (correlations["measure_b"] == measure_b)
        )
        | (
            (correlations["measure_a"] == measure_b)
            & (correlations["measure_b"] == measure_a)
        )
    ]
    return rows.iloc[0] if len(rows) else None


def build_individualcorrelations_structure(
    tables: BehaviorTableSet,
) -> tuple[Figure, dict[str, Any]]:
    """F18: how the individual-differences measures covary, matrix and detail.

    The matrix (A) and the scatters (B-F) are one argument, not two figures.
    At n = 32 a correlation matrix cannot show whether an r is outlier-driven,
    so the scatter grid is its required companion -- and a companion a reader
    has to find in a different figure, matching cells by name, is not doing
    that job. Every scatter here is one cell of A at higher magnification.

    B-F are the significant pairs with the largest |r|, strongest first, chosen
    from the derivative rather than listed here -- see :func:`_scatter_pairs`.
    On this cohort exactly five of the twenty-one pairs reach p < .05, so the
    grid holds all of them.

    Panel notes that are results, not layout:

    - **B is a convergent-validity check, not just the largest r.** The
      first-order chosen-target criterion slope and the exact-posterior urgency
      slope agree at r = +0.89 across subjects while remaining different scales
      -- which is what licenses Acts 2 and 3 quoting either.
    - **C-F are one speed factor.** Slower subjects are less sensitive to early
      evidence per unit (-0.81), show flatter urgency (-0.53) and criterion
      (-0.37) slopes, and are more accurate (+0.46).
    - **What is *not* in B-F is a result too.** The Fast/Slow SAT adjustment
      reaches significance with nothing, so the manipulation Acts 2-3 lean on is
      an independent trait in this cohort rather than a facet of the speed
      factor. That claim is read off A, which draws all 21 pairs; selecting the
      scatters on significance means they cannot make it.

    A and B-F mark significance differently, on purpose. B-F carry the
    battery's ``r = -.81 ***``. A prints the marker alone, without r: twenty-one
    coefficients printed at once is a table rendered as a picture, and the
    ramp already carries magnitude. A blank cell in A is tested and under the
    bar -- it is still filled with its colour -- while absent means outside the
    triangle.
    """
    profile = tables.analysis("individualprofile")
    correlations = tables.analysis("individualcorrelations")
    r_matrix, p_matrix = _correlation_matrices(correlations)
    scatter_pairs = _scatter_pairs(correlations)

    with style.apply_publication_style():
        # One uniform 3 x 2 grid: six panels, six cells, nothing spanning and
        # every panel the same size. A and the scatters are peers -- each is a
        # view of the same 21 correlations -- and a matrix spanning both rows,
        # or merely given a wider column, read instead as one large panel with
        # smaller ones hanging off it. The matrix pays for the equal cell with
        # a slim colorbar and hairline separators rather than by taking width
        # from B and C.
        fig, axes = style.figure_grid(2, 3, width="double", panel_height_in=2.9)
        ax_matrix = axes[0, 0]
        image = correlation_heatmap(
            ax_matrix, r_matrix=r_matrix, p_matrix=p_matrix, alpha=0.05,
            lower_triangle_only=True, include_diagonal=False,
            row_labels=[_MEASURE_LABELS[m] for m in _PROFILE_MEASURES],
            column_labels=[_MEASURE_LABELS[m] for m in _PROFILE_MEASURES],
            # It shares the figure now and has to fill the cell it is given;
            # square cells would letterbox that cell, shrinking the numbers
            # until they collide while leaving the panel half empty.
            aspect="auto",
            # Markers, not values. Twenty-one r's printed at once is a table
            # rendered as a picture; the ramp already carries magnitude and the
            # five that matter are drawn full size in B-F. What A is for is the
            # pattern -- which pairs move together at all, and how few of the
            # twenty-one do.
            cell_text="marker",
        )
        # Inset into the empty upper-right of A's own axes. A lower triangle
        # leaves that corner blank by construction, so the bar costs nothing:
        # beside the heatmap it took width the cells needed, and beneath it it
        # made A's cell taller than the scatter cells and opened a band of dead
        # space across the whole first row.
        colorbar_axes = ax_matrix.inset_axes([0.42, 0.86, 0.56, 0.055])
        colorbar = fig.colorbar(
            image, cax=colorbar_axes, orientation="horizontal",
        )
        colorbar.set_ticks([-1, 0, 1])
        colorbar.set_label("Pearson r", fontsize=style.DENSE_PT)
        colorbar_axes.tick_params(labelsize=style.DENSE_PT, pad=1)
        colorbar.outline.set_visible(False)
        style.panel_label(ax_matrix, "A")

        # Cells 1-5 of the grid, reading across then down; the matrix has cell 0.
        scatter_axes = [axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]
        for ax in scatter_axes[len(scatter_pairs):]:
            # Fewer significant pairs than cells: hide the spare rather than
            # leaving an empty framed axes, which reads as a panel whose data
            # failed to draw.
            ax.set_visible(False)
        for letter, ax, (measure_a, measure_b) in zip(
            "BCDEF", scatter_axes, scatter_pairs
        ):
            # Wide enough to clear the longest y-axis label in the set.
            # "Criterion slope (SumLogLR/s)" is the one that sets this: at the
            # default offset the letter lands on top of it.
            style.panel_label(ax, letter, offset_points=(-52.0, 8.0))
            scatter_fit(
                ax,
                x=profile[measure_a].to_numpy(dtype=float),
                y=profile[measure_b].to_numpy(dtype=float),
                xlabel=_axis_label(measure_a),
                ylabel=_axis_label(measure_b),
            )
            # Three ticks per axis. At this panel width four ran the numbers
            # into each other on every mean-DT axis.
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
            row = _correlation_row(correlations, measure_a, measure_b)
            if row is not None:
                # r and p read from the derivative; the OLS line and its band
                # are display-only, so the panel never recomputes what the text
                # quotes. `n` is constant at 32 and is stated once in the figure
                # note rather than five more times here.
                #
                # `r = -.81 ***`, the battery's convention: the marker, not
                # a numeric p. `marker_or_missing` spells a non-significant
                # result 'n.s.' and an *untested* one 'n/a', so a pair that
                # could not be tested is never reported as a tested null. The
                # exact p is in the derivative.
                #
                # Inside the panel, in the corner the cloud leaves empty: a
                # negative slope runs away from the top right, a positive one
                # from the top left. As an axes title it needed two lines, and
                # a row of titles across three narrow panels crowded the panel
                # letters and cost height the scatters could use.
                correlation = float(row["pearson_r"])
                ax.text(
                    0.96 if correlation < 0 else 0.04, 0.97,
                    f"r = {format_r(correlation)} "
                    f"{marker_or_missing(row['pearson_p'])}",
                    transform=ax.transAxes,
                    ha="right" if correlation < 0 else "left", va="top",
                    fontsize=style.ANNOTATION_PT, color=style.INK,
                )
        fig.suptitle("Individual differences: structure across subjects")
        # No figure-level note. The three things it carried -- that the 21
        # correlations are uncorrected, that A's marker scale is the usual
        # one, and that a blank cell in A is *tested and under the bar* rather
        # than missing -- now live only in the sidecar `caveat` below, and have
        # to be repeated in this figure's caption in docs/behavior.md. The
        # blank-cell reading in particular is not self-evident from the panel:
        # a blank that reads as "not measured" is the exact misreading a real
        # bug produced here once already.

    metadata = {
        "kind": "correlation_structure",
        "title": "Individual-differences structure",
        "columns_read": {
            "individualprofile": list(_PROFILE_MEASURES) + ["subject"],
            "individualcorrelations": [
                "measure_a", "measure_b", "n_subjects", "pearson_r", "pearson_p",
            ],
        },
        "statistics_source": "individualcorrelations",
        "uncertainty": (
            "A: none drawn; colour is the point estimate and the marker its "
            "significance. B-F: OLS line with a 95% band on the fitted mean, "
            "both display-only."
        ),
        "multiplicity_correction": "none",
        "scatter_pairs": [list(pair) for pair in scatter_pairs],
        "scatter_selection": (
            "significant (p < .05) pairs, largest |r| first, capped at "
            f"{SCATTER_PANEL_COUNT}"
        ),
        "caption_required": (
            "Pearson r, n = 32, uncorrected across 21 pairs. A: colour is r; "
            "* p<.05, ** p<.01, *** p<.001; a blank cell is tested and not "
            "significant. B-F: the significant pairs, largest |r| first. "
            "This figure draws no note of its own, so its caption has to "
            "carry these -- the multiplicity statement is required of any "
            "panel with more than three contrasts, and the blank-cell reading "
            "is not recoverable from the panel."
        ),
        "caveat": (
            "21 pairwise correlations at n = 32, uncorrected. A prints the "
            "significance marker without r: 21 coefficients at once is a table "
            "rendered as a picture, and the ramp already carries magnitude. A "
            "blank cell there is tested and not significant, not missing -- it "
            "is still filled with its colour. The self-correlations are not "
            "drawn: structurally 1.0, they would take the most saturated "
            "colour on the ramp for no information. "
            "The scatters exist because an r at this n can be outlier-driven "
            "and the matrix cannot show that; every one of them is a cell of "
            "A. They are SELECTED ON SIGNIFICANCE -- the significant pairs "
            "with the largest |r| -- so they are a gallery of significant "
            "results by construction and say nothing about how many pairs "
            "reached significance. A answers that, and is the only panel that "
            "can: on this cohort 5 of 21 did, which is why the grid holds all "
            "of them. The Fast/Slow SAT adjustment correlates with nothing and "
            "so has no scatter; that null is read off A."
        ),
    }
    return fig, metadata
