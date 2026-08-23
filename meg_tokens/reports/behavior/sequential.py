"""Sequential-effects figures: F16, F17."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from meg_tokens.reports import style
from meg_tokens.reports.annotations import marker_or_missing, stat_from_row
from meg_tokens.reports.behavior._tables import BehaviorTableSet
from meg_tokens.reports.panels import estimation_axis, forest, paired_slope, subject_strip

# Ink shades, deliberately not `style.MODEL_COLORS`. Green and magenta there
# mean "urgency gating" and "fixed-bound integrator" in F13-F14; reusing them
# for a measure contrast imports a meaning this figure does not have.
_MEASURE_COLORS = {
    "pre": style.INK_SECONDARY,
    "post": style.INK,
    "robust": style.INK_SECONDARY,
    "classical": style.INK,
}


# Short enough that the forest's row labels do not run under the panel letter,
# while "post - pre" still states the arithmetic the panel exists to show.
_PES_MEASURE_LABELS = {
    "post_error_minus_baseline_ms": "Post-error slowing",
    "pre_error_minus_baseline_ms": "Pre-error speeding",
    "robust_pes_ms": "Local pair (post − pre)",
    "classical_pes_ms": "Classical",
}


def build_posterror_slowing(tables: BehaviorTableSet) -> tuple[Figure, dict[str, Any]]:
    """F16: what the local-pair post-error measure actually contains.

    The conventional "robust" measure is ``DT(error + 1) - DT(error - 1)``,
    which is post-error slowing **only if pre-error trials are typical**. In
    this cohort they are not, so the figure is built around the decomposition
    rather than the composite.

    Panel notes that are results, not layout:

    - **A is the correction.** Against each subject's own baseline, post-error
      trials run +106 ms slow *and* pre-error trials run -38 ms fast. Both are
      significant. The composite everyone quotes is their difference, so 27% of
      it is pre-error speeding and naming the whole thing "post-error slowing"
      overstates the effect it is named for.
    - **B ranks the four quantities on one axis** so the arithmetic is visible:
      the local pair sits above the classical contrast by about the amount of
      the pre-error speeding, which is why the "stricter" definition returns
      the *larger* number. It is not stricter, it is broader.
    - **C is the agreement check.** The two conventional measures correlate at
      r = 0.82 but are not interchangeable: the local pair is positive in every
      subject, the classical contrast is negative in three. Those numbers are
      in the sidecar's `measure_agreement`, not over the panel -- the set
      reserves a panel title for a `Delta = value marker` bracket.
    - **D answers whether time pressure changes any of it.** Nothing survives:
      every Fast-minus-Slow contrast is null, each with the smallest effect it
      excludes rather than a bare "n.s.".

    `robust` in the derivative means local temporal *matching*, not a robust
    estimator -- no trimming or median is involved -- so the word is avoided in
    the panels.
    """
    posterror = tables.analysis("posterror")
    posterror_stats = tables.analysis("posterrorstats")

    pooled = posterror.loc[posterror["condition"] == "all"]

    def stat_row(measure: str, condition: str = "all"):
        rows = posterror_stats.loc[
            (posterror_stats["measure"] == measure)
            & (posterror_stats["condition"] == condition)
        ]
        return rows.iloc[0] if len(rows) else None

    with style.apply_publication_style():
        # One row, so the four quantities read left to right in the order the
        # argument makes them: the decomposition, the ranking it implies, the
        # agreement check, the condition null. Stacked 2x2, C sat under A with
        # no relation to it and the reading order had to be inferred from the
        # letters.
        #
        # B carries four text row labels on its y-axis and needs the width for
        # them; A, C and D each hold two x positions and need very little, so
        # the ratios buy B's labels from panels that were only padding.
        fig, axes = style.figure_grid(
            1, 4, width="double", width_ratios=[0.8, 1.75, 0.95, 0.9],
            panel_height_in=3.1,
        )
        ax_a, ax_b, ax_c, ax_d = (axes[0, index] for index in range(4))
        for ax, letter in zip((ax_a, ax_b, ax_c, ax_d), "ABCD"):
            style.panel_label(ax, letter)
        # Four panels at a quarter of the double width each; the default
        # locator gave seven y ticks per panel and they collided. B is excluded
        # because its y axis carries the four measure names, not numbers.
        for ax in (ax_a, ax_c, ax_d):
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))

        # --- A: the two effects the local pair contains ----------------------
        # "pre"/"post", not "pre-error"/"post-error": at a quarter of the double
        # width the two long forms overlapped each other. B sits immediately
        # right and spells both out in full, so nothing is lost by the short
        # form here -- and tick text is the one lever available, since the type
        # scale is set by the rc block and must not be overridden per call.
        components = {
            "pre": pooled["pre_error_minus_baseline_ms"].to_numpy(dtype=float),
            "post": pooled["post_error_minus_baseline_ms"].to_numpy(dtype=float),
        }
        estimates = {}
        for key, measure in (
            ("pre", "pre_error_minus_baseline_ms"),
            ("post", "post_error_minus_baseline_ms"),
        ):
            row = stat_row(measure)
            if row is not None:
                estimates[key] = (
                    float(row["mean"]), float(row["ci95_low"]), float(row["ci95_high"])
                )
        subject_strip(
            ax_a, groups=components, colors=_MEASURE_COLORS, reference=0.0,
            ylabel="Decision time − subject mean (ms)",
            connect=(("pre", "post"),),
            estimates=estimates or None,
        )
        # Autoscale leaves the two columns floating in the middle of a narrow
        # panel; pinning the limits puts them where the tick labels have room.
        ax_a.set_xlim(-0.32, 1.32)

        # --- B: the four quantities on one axis ------------------------------
        order = (
            "classical_pes_ms",
            "robust_pes_ms",
            "pre_error_minus_baseline_ms",
            "post_error_minus_baseline_ms",
        )
        rows = [(measure, stat_row(measure)) for measure in order]
        rows = [(measure, row) for measure, row in rows if row is not None]
        if rows:
            forest(
                ax_b,
                labels=[_PES_MEASURE_LABELS[measure] for measure, _ in rows],
                centres=np.array([float(row["mean"]) for _, row in rows]),
                lows=np.array([float(row["ci95_low"]) for _, row in rows]),
                highs=np.array([float(row["ci95_high"]) for _, row in rows]),
                reference=0.0,
            )
        ax_b.set_xlabel("Effect on decision time (ms)")
        # Two ticks, 0 and 100, against the zero rule. B spends most of its
        # width on the four row labels, and a 50 ms step -- which is what any
        # larger bin count resolves to over this range -- puts five numbers in
        # about an inch of axis, where they touch. Fewer tick labels is the
        # sanctioned trade (reduce their number, never hide them); dropping the
        # row names to buy ticks would be the wrong way round, since the names
        # are what the panel is for.
        ax_b.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))

        # --- C: do the two conventional measures agree? ----------------------
        robust = pooled["robust_pes_ms"].to_numpy(dtype=float)
        classical = pooled["classical_pes_ms"].to_numpy(dtype=float)
        paired_slope(
            ax_c, values_a=robust, values_b=classical,
            label_a="local pair", label_b="classical",
            color_a=_MEASURE_COLORS["robust"], color_b=_MEASURE_COLORS["classical"],
            ylabel="Post-error effect (ms)", reference=0.0,
        )
        # `paired_slope`'s default limits leave a fifth of the panel as margin
        # on each side, which at this width pushed the two tick labels into
        # each other. Pushing the columns out to the spines is what separates
        # them; the labels are already the shortest form that names them, and B
        # carries both in full immediately left.
        ax_c.set_xlim(-0.06, 1.06)
        # C's agreement summary goes to the sidecar, not into a panel title:
        # the set puts a `Delta = value marker` bracket over a panel and
        # everything else in the sidecar. It is *read* from the derivative
        # rather than recomputed here -- the write-up quotes these numbers, and
        # a figure that recomputes what the write-up cites is a second
        # implementation of the same statistic waiting to drift from the first.
        row = stat_row("robust_vs_classical_agreement")
        agreement = (
            {
                "pearson_r": float(row["pearson_r"]),
                "n_subjects": int(row["n_subjects"]),
                "local_pair_positive": int(row["robust_positive"]),
                "classical_positive": int(row["classical_positive"]),
            }
            if row is not None else {}
        )

        # --- D: does time pressure change it? --------------------------------
        contrast = stat_row("post_error_minus_baseline_ms", "fast_vs_slow")
        by_condition = posterror.pivot(
            index="subject", columns="condition", values="post_error_minus_baseline_ms"
        )
        if {"fast", "slow"} <= set(by_condition.columns) and contrast is not None:
            paired_slope(
                ax_d,
                values_a=by_condition["fast"].to_numpy(dtype=float),
                values_b=by_condition["slow"].to_numpy(dtype=float),
                label_a="Fast", label_b="Slow",
                color_a=style.CONDITION_COLORS["fast"],
                color_b=style.CONDITION_COLORS["slow"],
                ylabel="Post-error slowing (ms)", reference=0.0,
            )
            bound = float(contrast.get("equivalence_bound", float("nan")))
            # Two lines, not one: the single-line form is wider than a quarter
            # of the double width and ran off the figure's right edge, taking
            # the panel letter with it.
            ax_d.set_title(
                "Δ = {:+.0f} ms\n|Δ| < {:.0f}".format(
                    float(contrast["mean_difference"]), bound
                ) if np.isfinite(bound) else "Fast vs Slow",
                fontsize=style.ANNOTATION_PT, fontweight="normal",
                color=style.INK,
            )

        fig.suptitle("Post-error slowing: what the local pair contains")

    metadata = {
        "kind": "post_error_decomposition",
        "title": "Post-error slowing: what the local pair contains",
        "columns_read": {
            "posterror": [
                "subject", "condition", "mean_baseline_dt_ms",
                "pre_error_minus_baseline_ms", "post_error_minus_baseline_ms",
                "robust_pes_ms", "classical_pes_ms",
            ],
            "posterrorstats": [
                "measure", "condition", "mean", "ci95_low", "ci95_high",
                "t", "p", "df", "cohens_dz", "equivalence_bound",
                "pearson_r", "robust_positive", "classical_positive",
            ],
        },
        "statistics_source": (
            "posterrorstats throughout, including C's correlation and sign "
            "counts, which are registered as a descriptive row rather than "
            "recomputed here."
        ),
        "uncertainty": "registered 95% CIs in A and B; paired subjects in C and D",
        "measure_agreement": agreement,
        "palette": dict(_MEASURE_COLORS),
        "caveat": (
            "The derivative's `robust_pes_ms` is the local pair "
            "DT(error+1) - DT(error-1). `robust` there means local temporal "
            "matching, not a robust estimator: no trimming, median or "
            "M-estimator is involved, so the word is not used in the panels. "
            "That measure is post-error slowing only if pre-error trials are "
            "typical, and they are not -- pre-error trials run 38 ms fast "
            "against each subject's own mean (t(31) = -3.84, p = 5.7e-4), so "
            "27% of the 144 ms composite is pre-error speeding rather than "
            "post-error slowing. B shows why the classical contrast is the "
            "smaller number: it never touches pre-error trials. The classical "
            "contrast is in turn confounded by slow drift in time-on-task, "
            "which is what the local pair exists to remove -- neither measure "
            "is clean, which is the reason both are drawn. Panel D's null is "
            "reported with the smallest effect it excludes; the same Fast-Slow "
            "null holds for all four measures."
        ),
    }
    return fig, metadata




def build_choicehistory_effects(tables: BehaviorTableSet) -> tuple[Figure, dict[str, Any]]:
    """F17: what the previous trial's outcome carries into the next choice.

    Two panels: whether the previous outcome changes which side is chosen (A),
    and whether side repeats or alternates independently of outcome (B).

    Panel notes that are results, not layout:

    - **A is a stay rate against 0.5, not an accuracy.** Win-stay exceeds
      lose-stay, so the previous outcome moves choice; both sit near chance
      because the correct side is counterbalanced.
    - **B's negative mean is alternation, not perseveration.** It is small
      (-0.029) but reliable, and the per-subject spread crosses zero, so it is
      a group tendency rather than a strategy every subject shows.

    Decision time after an error is not here: F16 owns the post-error contrast
    and reports it decomposed against each subject's baseline.
    """
    history = tables.analysis("choicehistory")
    history_stats = tables.analysis("choicehistorystats")

    pooled = history.loc[history["condition"] == "all"]

    def stat_row(measure: str):
        rows = history_stats.loc[
            (history_stats["measure"] == measure)
            & (history_stats["condition"] == "all")
        ]
        return rows.iloc[0] if len(rows) else None

    with style.apply_publication_style():
        # Narrower than a full double width: two panels carrying two x
        # positions and one cloud were stretched across 2.9 in of axes each,
        # and the default locator gave both eight y ticks.
        # A holds two x positions and B one cloud, so A gets the narrower
        # column: at equal widths its outer margins had to carry the space that
        # pulling the two columns together freed up.
        fig, axes = style.figure_grid(
            1, 2, width=4.4, width_ratios=[1.0, 1.1], panel_height_in=2.9
        )
        ax_a, ax_b = axes[0, 0], axes[0, 1]
        for ax, letter in zip((ax_a, ax_b), "AB"):
            style.panel_label(ax, letter)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))

        # --- A: does the previous outcome change which side is chosen? -------
        paired_slope(
            ax_a,
            values_a=pooled["win_stay"].to_numpy(dtype=float),
            values_b=pooled["lose_stay"].to_numpy(dtype=float),
            label_a="win-stay", label_b="lose-stay",
            color_a=style.INK, color_b=style.INK_SECONDARY,
            ylabel="Stay proportion", reference=0.5,
        )
        # The two columns sat 69% of the panel apart, so the panel's most
        # prominent feature was the gap between them. Widening the limits pulls
        # them together without shrinking the axes.
        ax_a.set_xlim(-0.38, 1.38)
        row = stat_row("win_stay_vs_lose_stay")
        if row is not None:
            ax_a.set_title(
                f"Δ = {float(row['mean_difference']):+.3g} "
                f"{marker_or_missing(row['p'])}",
                fontsize=style.ANNOTATION_PT, fontweight="normal", color=style.INK,
            )

        # --- B: does side repeat or alternate, outcome aside? ----------------
        row = stat_row("side_autocorrelation_lag1")
        result = stat_from_row(row, label="autocorr") if row is not None else None
        if result is not None:
            estimation_axis(
                ax_b,
                differences=pooled["side_autocorrelation_lag1"].to_numpy(dtype=float),
                result=result,
                ylabel="Side autocorrelation, lag 1",
                spread=0.035,
                # Not a contrast: this is one correlation per subject tested
                # against zero, and A's panel already carries a real Δ. Two
                # adjacent titles both reading "Δ =" made B look like a
                # Fast-minus-Slow difference of something.
                symbol="r",
            )
            # The jittered cloud filled 43% of the panel, leaving as much empty
            # margin again on either side. Fitted to the cloud and its interval
            # marker rather than to the helper's default padding.
            spread = [
                offset[:, 0]
                for offset in (c.get_offsets() for c in ax_b.collections)
                if len(offset)
            ]
            if spread:
                extent = float(np.nanmax(np.abs(np.concatenate(spread))))
                ax_b.set_xlim(-1.25 * extent, 1.25 * extent)

        fig.suptitle("Choice-history effects")

    metadata = {
        "kind": "choice_history",
        "title": "Choice-history effects",
        "columns_read": {
            "choicehistory": [
                "subject", "condition", "win_stay", "lose_stay",
                "side_autocorrelation_lag1",
            ],
            "choicehistorystats": [
                "measure", "condition", "mean_difference", "mean", "sem",
                "t", "p", "df", "cohens_dz",
            ],
        },
        "statistics_source": "choicehistorystats",
        "uncertainty": "paired subjects in A; one-sample 95% interval in B",
        "caveat": (
            "A is a stay *rate* against a 0.5 reference, not an accuracy; both "
            "levels sit near chance because the correct side is "
            "counterbalanced. B is signed, and negative means the chosen side "
            "alternates rather than repeats -- the mean of -0.029 is small but "
            "reliable, while individual subjects fall on both sides of zero. "
            "Decision time after an error is not here: it is the same "
            "per-subject quantity as F16's classical contrast, and F16 reports "
            "it decomposed."
        ),
    }
    return fig, metadata


