"""Sequential-sampling model figures: F13-F15."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from meg_tokens.behavior.analyses.sequential_sampling import (
    SOLVER_STEP_S,
    heldout_pairwise_subject_differences,
    near_upper_bound,
)
from meg_tokens.reports import style
from meg_tokens.reports.annotations import (
    is_significant,
    marker_or_missing,
)
from meg_tokens.reports.behavior._tables import BehaviorTableSet
from meg_tokens.reports.panels import (
    forest,
    group_line,
    paired_slope,
    subject_strip,
)

# The fixed-bound integrator every candidate is scored against.
BASELINE_MODEL = "ddm"
HELDOUT_SCORE = "heldout_log_likelihood_per_trial"
URGENCY_SCALE = "urgency_scale"
# Registered configuration keys are identifiers; panels get prose.
_ROBUSTNESS_LABELS = {
    "baseline": "baseline",
    "expanded_bounds": "wider bounds",
    "post_horizon_evidence_zero": "post-horizon zero",
    "solver_20ms": "solver 20 ms",
    "tau_100ms": "tau 100 ms",
    "tau_300ms": "tau 300 ms",
}
_URGENCY_PARAMETERS = ("drift_scale", "urgency_scale", "urgency_onset_s", "nondecision_s")
# Column names are identifiers; axes get the quantity, its unit, and -- for
# nondecision_s -- the fact that what is fitted is the residual after motor
# correction, without which a few milliseconds looks like an error.
_PARAMETER_LABELS = {
    "drift_scale": "Drift scale",
    "urgency_scale": "Criterion scale (s)",
    "urgency_onset_s": "Urgency onset (s)",
    "nondecision_s": "Non-decision (ms)",
    "bound": "Bound",
    "collapse_rate": "Collapse rate",
    "additive_scale": "Additive scale",
}
# Display-only unit conversions, applied to the plotted values and to the
# annotated effect and bound together so a panel never mixes units.
#
# Non-decision time is fitted in seconds but lives at a few milliseconds, so in
# seconds every number spends three characters on leading zeros: the panel is
# one column of a seven-column 180 mm figure, about 1 in, and `|Δ| < 0.00226`
# at 9 pt is wider than the axes. In milliseconds the same statistic is
# `|Δ| < 2.3` and the y ticks read 0/5/10 instead of 0.000/0.005/0.010.
_PARAMETER_DISPLAY_SCALE = {"nondecision_s": 1000.0}


def _compact_effect(value: float) -> str:
    """Format an effect for a ~1 in panel title: short, but never bare.

    ``%.2g`` is short enough but strips a trailing zero, so -1.04 prints as
    "-1" and sits next to a bound printed as "2.3" -- an apparent difference in
    precision that is only a formatting artefact. Fixing the decimals by
    magnitude keeps both at the same visible resolution.
    """
    if not np.isfinite(value):
        return "n/a"
    magnitude = abs(value)
    if magnitude >= 10:
        return f"{value:.0f}"
    if magnitude >= 1:
        return f"{value:.1f}"
    return f"{value:.2g}"
# One word each, for axis ticks; style.MODEL_LABELS carries the full names.
_SHORT_MODEL_LABELS = {
    "urgency": "Urgency",
    "collapsing": "Collapsing",
    "additive_urgency": "Additive",
    "ddm": "Fixed bound",
}


def _pairwise_contrast(
    pairwise: pd.DataFrame, first: str, second: str
) -> dict[str, Any] | None:
    """A registered held-out pairwise contrast, oriented `first` minus `second`.

    The derivative stores each pair alphabetically, so half the rows are
    stored the other way round. Reading one raw and assuming its sign is how a
    figure ends up claiming the loser won; this flips `mean` and the interval
    together and swaps the two subject counts. Nothing is recomputed -- it is
    a relabelling of one registered row, done in one place so the flip is
    visible.
    """
    forward = (pairwise["model_a"] == first) & (pairwise["model_b"] == second)
    reverse = (pairwise["model_a"] == second) & (pairwise["model_b"] == first)
    rows = pairwise.loc[
        (forward | reverse)
        & (pairwise["condition"] == "all")
        & (pairwise["criterion"] == HELDOUT_SCORE)
    ]
    if rows.empty:
        return None
    row = rows.iloc[0]
    assert row["direction"] == "a_minus_b_positive_favors_model_a", row["direction"]
    sign = 1.0 if row["model_a"] == first else -1.0
    low, high = sign * float(row["ci95_low"]), sign * float(row["ci95_high"])
    return {
        "mean": sign * float(row["mean"]),
        # Flipping an interval reverses its ends, so re-sort rather than
        # trusting the stored order.
        "ci_low": min(low, high),
        "ci_high": max(low, high),
    }


def _slant_model_ticks(ax: Axes) -> None:
    """Slant model tick labels so neighbouring names cannot collide.

    Three model names side by side fit comfortably across a half-width panel
    and not at all across a third-width one. Slanting keeps every label
    present -- the palette must never be the only thing identifying a model,
    since two of the model hues fail the contrast floor on white and carry a
    text-label relief rule (`reports.style`).
    """
    for label in ax.get_xticklabels():
        label.set_rotation(18)
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")


def build_ssmcomparison_model(tables: BehaviorTableSet) -> tuple[Figure, dict[str, Any]]:
    """F13: the mechanistic model comparison -- what the models do, and which wins.

    Six panels in two rows. The top row is the *mechanism*: what each model
    claims the commitment policy is, and whether the winner reproduces the
    data in absolute terms. The bottom row is the *adjudication*: which model
    the data prefer, in sample and out.

    These were two figures. The split put the criterion time course -- the
    visual definition of the contrast being adjudicated -- in a different
    figure from the adjudication, and re-plotted the same held-out comparison
    in both, drawn two different ways. A reader met the same numbers twice
    and the mechanism never beside the result that rests on it.

    Panel notes that are results, not layout:

    - **D shows every candidate against the fixed bound, not urgency alone.**
      A collapsing bound also beats the fixed bound decisively (-169 BIC,
      31/32) and mimics a declining behavioural criterion just as urgency
      gating does; a two-model panel would award urgency a race it had not
      run. `ddm` is not missing from D -- it *is* the zero line, since every
      delta is measured against it, so the axis label names it rather than
      giving it a column of zeros.
    - **F races the candidates against each other**, out of sample, where an
      extra free parameter earns nothing for free. Only the three
      among-candidate pairs are drawn: the three candidate-versus-`ddm` pairs
      are already in D and E, and the six pairwise means carry only three
      independent numbers, being exactly additive.
    - **F is neutral in colour.** A paired difference is not either model's
      quantity; colouring by the first model made two of three rows green and
      implied green meant "winner". Colour identifies a model in A, C, D and
      E, never in F.
    - **B uses the class palette, not the model palette.** Class and model are
      disjoint colour scopes (`reports.style`), so B carries its own legend
      and the shared model legend never applies to it.
    - **C is a residual view, not an overlay of predicted densities.** The
      overlay could not answer the question it was there to answer: the four
      predictions are very nearly superimposed at this panel width, so the
      panel argued that all four fit the marginal distribution about equally
      well. Differencing against the observation shows that they do not --
      the fixed bound misses the 90th correct percentile by 375 ms and the
      fast errors by 373 ms, against 54 ms and 90 ms for urgency. The
      observation is the zero line, so it has no legend entry.

    Repeated `criterion`/`observed_density_*` fields in `ssmtimecourse` are
    de-duplicated before subject-balanced averaging; a naive groupby would
    weight them by the number of trial classes.
    """
    fits = tables.analysis("ssmcomparison")
    stats_table = tables.analysis("ssmcomparisonstats")
    courses = tables.analysis("ssmtimecourse")
    heldout = tables.analysis("ssmheldout")
    heldout_stats = tables.analysis("ssmheldoutstats")
    pairwise = tables.analysis("ssmheldoutpairwise")
    checks = tables.analysis("ssmdistributioncheckstats")

    converged = fits.loc[fits["converged"].astype(bool) & (fits["condition"] == "all")]
    candidates = [
        model for model in style.MODEL_ORDER
        if model != BASELINE_MODEL and model in set(converged["model"])
    ]
    delta_bic = converged.pivot(index="subject", columns="model", values="delta_bic")
    comparison = stats_table.loc[
        (stats_table["analysis"] == "ssm_mechanistic_model_comparison")
        & (stats_table["criterion"] == "bic")
        & (stats_table["condition"] == "all")
    ].set_index("model")

    pooled = courses.loc[courses["condition"] == "all"].copy()
    # Subjects' fits end at different horizons, so past some time the group
    # mean is an average over a shrinking, self-selected subset -- 32 subjects
    # contribute at 3 s, 14 at 5 s, 2 at 8 s. Drawn untrimmed, the panels' most
    # striking feature (curves fanning apart) is mostly subjects dropping out,
    # and the fixed bound acquires steps it cannot have.
    per_time = pooled.drop_duplicates(["subject", "model", "time_s"]).groupby(
        "time_s"
    )["subject"].nunique()
    common_horizon = (
        float(per_time[per_time == per_time.max()].index.max()) if len(per_time) else 0.0
    )
    pooled = pooled.loc[pooled["time_s"] <= common_horizon]
    model_colors = style.MODEL_COLORS
    model_labels = style.MODEL_LABELS
    models = [
        model for model in ("ddm", "urgency", "collapsing", "additive_urgency")
        if model in set(fits["model"])
    ]

    def subject_matrix(rows: pd.DataFrame, value: str) -> tuple[np.ndarray, np.ndarray]:
        """Return a subject x time matrix after explicit within-subject dedup."""
        if rows.empty:
            return np.array([]), np.empty((0, 0))
        grouped = rows.groupby(["subject", "time_s"], as_index=False)[value].mean()
        wide = grouped.pivot(
            index="subject", columns="time_s", values=value
        ).sort_index(axis=1)
        return wide.columns.to_numpy(dtype=float), wide.to_numpy(dtype=float)

    def all_trial_class(rows: pd.DataFrame) -> pd.DataFrame:
        """Select the derivative's explicit all-class rows."""
        if "trial_class" not in rows:
            return rows
        return rows.loc[rows["trial_class"].astype(str) == "all"]

    with style.apply_publication_style():
        fig, axes = style.figure_grid(2, 3, width="double", panel_height_in=2.55)
        ax_a, ax_b, ax_c = axes[0, 0], axes[0, 1], axes[0, 2]
        ax_d, ax_e, ax_f = axes[1, 0], axes[1, 1], axes[1, 2]
        for ax, label in zip((ax_a, ax_b, ax_c, ax_d, ax_e, ax_f), "ABCDEF"):
            style.panel_label(ax, label)

        criterion_rows = pooled.drop_duplicates(
            ["subject", "condition", "model", "time_s"]
        )

        # --- A: the criterion each model claims ------------------------------
        for model in models:
            rows = criterion_rows.loc[criterion_rows["model"] == model]
            x, matrix = subject_matrix(rows, "criterion")
            if x.size:
                group_line(
                    ax_a, x=x, subject_matrix=matrix, color=model_colors[model],
                    label=model_labels[model], error="within_ci95", marker=None,
                )
        ax_a.set(xlabel="Time (s)", ylabel="Criterion")

        # --- B: where commitment happens, by difficulty ----------------------
        urgency = pooled.loc[pooled["model"] == "urgency"]
        for trial_class, colour in style.CLASS_COLORS.items():
            rows = urgency.loc[urgency["trial_class"] == trial_class]
            x, matrix = subject_matrix(rows, "mean_decision_variable")
            if x.size:
                group_line(
                    ax_b, x=x, subject_matrix=matrix, color=colour,
                    label=trial_class, error="within_ci95", marker=None,
                )
        criterion = (
            criterion_rows.loc[criterion_rows["model"] == "urgency"]
            .groupby("time_s")["criterion"].mean().sort_index()
        )
        if len(criterion):
            ax_b.plot(
                criterion.index, criterion.to_numpy(), color=style.INK,
                linestyle="--", linewidth=0.9, label="criterion",
            )
        ax_b.set(xlabel="Time (s)", ylabel="Decision variable")
        # Upper right: the criterion falls from the top-left corner and the
        # class trajectories rise into the lower right, so the top right is
        # the only corner no series occupies.
        ax_b.legend(loc="upper right", framealpha=0.85)

        # --- C: absolute fit, as observed-minus-predicted residuals ----------
        #
        # This panel used to overlay the four predicted decision-time densities
        # on the observed one. That display cannot answer the question it is
        # here to answer: at this panel width the four predictions are very
        # nearly superimposed, so the panel argued "all four models fit the
        # marginal distribution about equally well" -- which is false. The
        # fixed bound misses the 90th correct percentile by 375 ms and the fast
        # errors by 373 ms, against 54 ms and 90 ms for urgency. Differencing
        # against the observation is what makes those failures visible.
        #
        # Every value is read from `ssmdistributioncheckstats`, including the
        # intervals; nothing here is refitted from the densities.
        residuals = checks.loc[
            (checks["condition"] == "all") & (checks["trial_class"] == "all")
        ]
        quantiles = sorted(residuals["quantile"].astype(float).unique())
        # Correct and error share the x positions. They were briefly split into
        # left and right halves of the axis *and* distinguished by line style,
        # which encodes the same distinction twice and leaves the dashes
        # looking like they mean something further. One encoding carries it:
        # line style. Sharing the x also puts a model's two outcomes at the
        # same percentile side by side, which is the comparison the panel is
        # for -- a model can track the correct distribution and still miss the
        # errors badly, and that is exactly the fixed bound's failure.
        ax_c.axhline(0, color=style.INK_MUTED, linewidth=0.6, zorder=0)
        for model in models:
            rows = residuals.loc[residuals["model"] == model]
            for outcome, dash in (("correct", None), ("error", (1.5, 1.2))):
                subset = rows.loc[rows["outcome"] == outcome].copy()
                if subset.empty:
                    continue
                subset = subset.sort_values("quantile")
                x = subset["quantile"].astype(float) * 100.0
                line, = ax_c.plot(
                    x, subset["mean_residual_s"],
                    color=model_colors[model], linewidth=1.4,
                    marker="o", markersize=3.5,
                    label=model_labels[model] if outcome == "correct" else None,
                )
                if dash is not None:
                    line.set_dashes(dash)
                ax_c.fill_between(
                    x, subset["ci95_low"], subset["ci95_high"],
                    color=model_colors[model], alpha=0.12, linewidth=0,
                )
        ax_c.set_xticks([q * 100 for q in quantiles])
        # Four round ticks over the drawn range. The default locator put six on
        # a panel a third of the figure wide, which is more numbers than the
        # panel is making an argument about.
        ax_c.set_yticks([-0.3, 0.0, 0.3, 0.6])
        ax_c.set(xlabel="Percentile", ylabel="Observed − predicted (s)")
        ax_c.set_xlim(
            min(quantiles) * 100 - 8, max(quantiles) * 100 + 8
        )
        # Solid/dashed is the only thing separating the two outcomes now, so it
        # needs a key. Neutral proxies: the distinction is outcome, not model,
        # and colouring them would claim one of the model hues for it.
        outcome_handles = [
            Line2D([], [], color=style.INK_SECONDARY, linewidth=1.4, label="correct"),
            Line2D([], [], color=style.INK_SECONDARY, linewidth=1.4,
                   dashes=(1.5, 1.2), label="error"),
        ]
        ax_c.legend(
            handles=outcome_handles, loc="upper right",
            framealpha=0.85, handlelength=1.6,
        )

        # --- D: in-sample BIC against the fixed bound ------------------------
        subject_strip(
            ax_d,
            groups={
                _SHORT_MODEL_LABELS[model]: delta_bic[model].to_numpy(dtype=float)
                for model in candidates
            },
            colors={
                _SHORT_MODEL_LABELS[model]: style.MODEL_COLORS[model]
                for model in candidates
            },
            estimates={
                _SHORT_MODEL_LABELS[model]: (
                    float(comparison.loc[model, "mean"]),
                    float(comparison.loc[model, "ci95_low"]),
                    float(comparison.loc[model, "ci95_high"]),
                )
                for model in candidates
                if model in comparison.index
            },
            reference=0.0,
            ylabel="ΔBIC vs fixed bound",
            spread=0.22,
        )
        ax_d.set_xlabel("")
        ax_d.set_xlim(-0.55, len(candidates) - 0.45)
        _slant_model_ticks(ax_d)
        ax_d.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 4, 5, 10]))
        bottom, top = ax_d.get_ylim()
        ax_d.set_ylim(bottom, top + 0.14 * (top - bottom))
        for position, model in enumerate(candidates):
            if model in comparison.index and is_significant(comparison.loc[model, "p"]):
                ax_d.text(
                    position, top, marker_or_missing(comparison.loc[model, "p"]),
                    ha="center", va="bottom",
                    fontsize=style.SIGNIFICANCE_PT, color=style.INK,
                )

        # --- E: held-out prediction against the fixed bound ------------------
        differences = heldout_pairwise_subject_differences(heldout)
        estimates = {}
        for model in candidates:
            rows = heldout_stats.loc[
                (heldout_stats["condition"] == "all")
                & (heldout_stats["model"] == model)
            ]
            if len(rows):
                row = rows.iloc[0]
                estimates[_SHORT_MODEL_LABELS[model]] = (
                    float(row["mean"]), float(row["ci95_low"]), float(row["ci95_high"])
                )
        subject_strip(
            ax_e,
            groups={
                _SHORT_MODEL_LABELS[model]: differences.loc[
                    (differences["model_a"] == model)
                    & (differences["model_b"] == BASELINE_MODEL),
                    "difference",
                ].to_numpy(dtype=float)
                for model in candidates
            },
            colors={
                _SHORT_MODEL_LABELS[model]: model_colors[model]
                for model in candidates
            },
            estimates=estimates,
            reference=0.0,
            ylabel="Δ held-out LL / trial",
            spread=0.22,
        )
        ax_e.set_xlabel("")
        ax_e.set_xlim(-0.55, len(candidates) - 0.45)
        _slant_model_ticks(ax_e)

        # --- F: the candidates against each other, out of sample -------------
        candidate_pairs = [
            (first, second)
            for index, first in enumerate(candidates)
            for second in candidates[index + 1:]
        ]
        per_subject = differences
        drawn = [
            (pair, contrast)
            for pair in candidate_pairs
            if (contrast := _pairwise_contrast(pairwise, *pair)) is not None
        ]
        ax_f.axvline(0, color=style.INK_MUTED, linewidth=0.6, zorder=0)
        for row_index, ((first, second), contrast) in enumerate(drawn):
            values = per_subject.loc[
                (per_subject["model_a"] == first) & (per_subject["model_b"] == second),
                "difference",
            ].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            ax_f.scatter(
                values,
                row_index + np.linspace(-0.16, 0.16, values.size),
                color=style.INK_SECONDARY, s=10, alpha=0.5, zorder=2,
            )
            # The interval is the registered one, never refitted from the dots.
            ax_f.errorbar(
                [contrast["mean"]], [row_index],
                xerr=[[contrast["mean"] - contrast["ci_low"]],
                      [contrast["ci_high"] - contrast["mean"]]],
                color=style.INK, marker="o", markersize=4, capsize=2.5, zorder=3,
            )
            ax_f.text(
                0.01, row_index - 0.42,
                f"{_SHORT_MODEL_LABELS[first]} − {_SHORT_MODEL_LABELS[second]}",
                transform=ax_f.get_yaxis_transform(),
                ha="left", va="bottom",
                fontsize=style.ANNOTATION_PT, color=style.INK,
            )
        ax_f.set_yticks([])
        ax_f.set_ylim(len(drawn) - 0.5, -0.75)
        ax_f.set_xlabel("Δ held-out LL / trial")
        ax_f.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))

        # One model legend for the whole figure. A and C carried identical
        # four-model legends when these were two figures; the duplicate spent
        # panel area restating the palette.
        # C labels each model once (the correct-outcome series); the error
        # series are the same colours dashed, so they must not double the
        # legend. There is no 'Observed' entry any more -- the observation
        # is the zero line.
        handles, labels = ax_c.get_legend_handles_labels()
        if handles:
            # Three columns, not five: one row of five ran off both edges
            # of the figure at this type size.
            fig.legend(
                handles, labels, loc="outside lower center",
                ncol=3, frameon=False,
            )
        fig.suptitle("Model comparison: mechanism and adjudication")

    return fig, {
        "kind": "mechanistic_model_comparison",
        "title": "Model comparison: mechanism and adjudication",
        "columns_read": {
            "ssmcomparison": ["subject", "condition", "model", "delta_bic", "converged"],
            "ssmcomparisonstats": [
                "analysis", "model", "condition", "criterion", "mean",
                "ci95_low", "ci95_high", "p",
            ],
            "ssmtimecourse": [
                "subject", "condition", "model", "trial_class", "time_s",
                "criterion", "mean_decision_variable",
            ],
            "ssmdistributioncheckstats": [
                "analysis", "condition", "model", "trial_class", "outcome",
                "quantile", "mean_residual_s", "ci95_low", "ci95_high",
                "n_subjects",
            ],
            "ssmheldout": [
                "subject", "condition", "model", "fold", HELDOUT_SCORE, "n_test",
            ],
            "ssmheldoutstats": [
                "analysis", "model", "condition", "criterion", "n_subjects",
                "mean", "ci95_low", "ci95_high", "t", "p", "cohens_dz",
            ],
            "ssmheldoutpairwise": [
                "analysis", "model_a", "model_b", "condition", "mean",
                "ci95_low", "ci95_high", "p",
            ],
        },
        "statistics_source": (
            "ssmdistributioncheckstats for C; "
            "ssmcomparisonstats[analysis=ssm_mechanistic_model_comparison] for D; "
            "ssmheldoutstats for E; ssmheldoutpairwise for F. Intervals are "
            "registered, never refitted from the dots or the curves."
        ),
        "time_horizon_s": common_horizon,
        "palette": {model: style.MODEL_COLORS[model] for model in models},
        "deduplication": "subject-balanced; explicit trial_class=all rows only",
        "uncertainty": (
            "within-subject 95% CI bands for A-B; registered 95% CIs for C-F"
        ),
        "models": models,
        "caveat": (
            "Read D-F as model *selection*, not measurement. Recovery shows the "
            "four models are mutually distinguishable in principle, but it was "
            "run on 160 synthetic trials over 4 stylised sequences, at 2 starts, "
            "in a parameter region the fits do not occupy (urgency drift_scale "
            "truths 1.125-3.875 against fitted 0.083-0.383), and 26 of its 48 "
            "cases were decided by under 10 BIC points -- so it does not "
            "establish that selection is reliable at this design's sample size. "
            "The fitted urgency parameters are not identified: urgency_scale "
            "recovers at r = 0.44, 95% CI [-0.17, 0.81], consistent with no "
            "recovery, so no single subject's dot in D or E is that subject's "
            "urgency. Additive urgency is a sensitivity analysis, not a test of "
            "additive mechanisms -- a common additive drive to two race channels "
            "cancels in a signed difference -- so its null is not evidence "
            "against additive urgency in general. Refitting on the strict "
            "complete-log subset leaves urgency's raw delta_bic smaller by 61.5 "
            "points, but that is the trial count, not the effect: per trial the "
            "advantage *rises* from 0.2451 to 0.2730 nats (p = 2.9e-5). Panels "
            "A and B are trimmed to "
            f"{common_horizon:.3g} s, the last time every subject's fit "
            "contributes; past it the group mean is over a shrinking, "
            "self-selected subset. C is an absolute-fit check and does not "
            "depend on the model set being complete, which is what makes it "
            "a stronger argument than D. D is in sample; E and F are not."
        ),
    }


def build_ssmcomparison_urgency(tables: BehaviorTableSet) -> tuple[Figure, dict[str, Any]]:
    """F14: what does the speed instruction change about urgency?

    Six panels. **A-D are the four urgency-model parameters**, Fast against
    Slow, one narrow paired panel each. **E and F interrogate the one that
    moves**: its effect size, and the six robustness refits that answer the
    censoring objection A-D raise.

    These were two figures, and the criterion-scale panel was in both -- the
    same parameter, the same contrast, drawn twice across the battery. Merged,
    the parameter that carries the effect sits beside the three that do not,
    which is the actual argument: the speed instruction acts on urgency growth
    *rather than* on when urgency starts or on motor delay.

    Panel notes that are results, not layout:

    - **The figure argues from two nulls**, so each null carries the smallest
      effect it excludes rather than a bare p. "n.s." asserts nothing; "within
      0.114 s of zero" does. The bound is read from
      `equivalence_bound` on the registered contrast -- the figure derives no
      inferential quantity of its own.
    - **Lower criterion scale means faster-rising urgency.** It is threshold
      over urgency slope, so B's Fast mean sitting *below* Slow is the effect
      and not its opposite.
    - **Ringed points are censored, not estimated** -- the optimizer left them
      within 1% of the parameter's upper bound -- and pairs touching one are
      dashed. A pair censored at both ends otherwise draws a flat line that
      reads as a measured "no change" the fit never claimed.
    - **F is the censoring defence.** The censoring is asymmetric in the
      direction of the effect (3 Fast cells against 8 Slow on criterion
      scale), so the refit with none at all is the one that matters: it is
      coloured, and its contrast is the largest of the six.
    """
    fits = tables.analysis("ssmcomparison")
    stats_table = tables.analysis("ssmcomparisonstats")
    boundary = tables.analysis("ssmboundaryaudit")
    robustness = tables.analysis("ssmrobustnessstats")

    urgency = fits.loc[
        (fits["model"] == "urgency") & fits["converged"].astype(bool)
    ]
    contrast_rows = stats_table.loc[
        stats_table["analysis"] == "ssm_urgency_condition_contrast"
    ].set_index("parameter")

    def paired(parameter: str):
        """Fast/Slow arrays and the censoring mask, aligned on paired subjects."""
        marked = urgency.assign(
            censored=lambda frame, name=parameter: near_upper_bound(frame, name)
        )
        by_condition = {
            condition: marked.loc[marked["condition"] == condition].set_index("subject")
            for condition in ("fast", "slow")
        }
        index = by_condition["fast"].index.intersection(by_condition["slow"].index)
        values = {
            condition: frame.loc[index, parameter].to_numpy(dtype=float)
            for condition, frame in by_condition.items()
        }
        censored = {
            condition: frame.loc[index, "censored"].to_numpy(dtype=bool)
            for condition, frame in by_condition.items()
        }
        return index, values, censored

    with style.apply_publication_style():
        fig = plt.figure(figsize=(style.WIDTH_DOUBLE_IN, 2.75))
        # Four narrow parameter panels over a wider pair. The paired panels
        # hold two x positions each and need no width; the refit forest needs
        # room for its row labels, so it takes three columns of the lower row.
        # One row. Four narrow paired panels and one wider forest: six panels
        # across 180 mm gave each ~1.1 in, at which the Fast/Slow tick labels
        # collide into one word, so the effect size and the refits share a
        # panel instead of taking one each.
        # Explicit ratios rather than equal columns: A-D each carry a two-line
        # effect annotation about as wide as the axes, while the forest's long
        # row labels are right-aligned against its axis and leave slack on its
        # left. Moving that slack to the paired panels costs the forest
        # nothing it was using.
        grid = fig.add_gridspec(1, 5, width_ratios=[1.15, 1.15, 1.15, 1.15, 2.55])
        parameter_axes = [fig.add_subplot(grid[0, column]) for column in range(4)]
        ax_robust = fig.add_subplot(grid[0, 4])

        for index, (ax, parameter) in enumerate(
            zip(parameter_axes, _URGENCY_PARAMETERS)
        ):
            style.panel_label(ax, "ABCD"[index])
            _, values, censored = paired(parameter)
            scale = _PARAMETER_DISPLAY_SCALE.get(parameter, 1.0)
            if scale != 1.0:
                values = {
                    condition: array * scale for condition, array in values.items()
                }
            paired_slope(
                ax,
                values_a=values["fast"], values_b=values["slow"],
                label_a="Fast", label_b="Slow",
                color_a=style.CONDITION_COLORS["fast"],
                color_b=style.CONDITION_COLORS["slow"],
                ylabel=_PARAMETER_LABELS[parameter],
                dashed_mask=censored["fast"] | censored["slow"],
            )
            finite = np.isfinite(values["fast"]) & np.isfinite(values["slow"])
            for x, condition in ((0.0, "fast"), (1.0, "slow")):
                pinned = values[condition][finite & censored[condition]]
                ax.scatter(
                    np.full(pinned.shape, x), pinned,
                    facecolors="none", edgecolors=style.INK, linewidths=1.0,
                    s=44, zorder=4,
                )
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
            if parameter not in contrast_rows.index:
                continue
            row = contrast_rows.loc[parameter]
            mean = float(row["mean_difference"]) * scale
            marker = marker_or_missing(row["p"])
            if not is_significant(row["p"]):
                # The bound is registered on the contrast; the figure reads it
                # rather than rebuilding it from the t statistic.
                bound = float(row.get("equivalence_bound", np.nan)) * scale
                if np.isfinite(bound):
                    marker = f"|Δ| < {_compact_effect(bound)}"
            # Effect over marker, on two lines: side by side they are wider
            # than a seventh of a 180 mm figure. Two significant digits, and
            # the unit stated once on the y label rather than repeated here --
            # at three digits the equivalence bounds still overran the
            # neighbouring panels even stacked.
            sign = "+" if mean >= 0 else "−"
            # DENSE_PT, not ANNOTATION_PT. These four axes are 0.57 in wide --
            # the y label and ticks take most of each 1.13 in column -- and at
            # 9 pt the longer annotations rendered 0.74 in, half again the
            # width of the panel they describe. Widening the axes to fit is not
            # available: the forest's row labels alone are 1.16 in, so the
            # 0.66 in needed would leave it under 0.6 in of plotting range.
            ax.set_title(
                f"Δ = {sign}{_compact_effect(abs(mean))}\n{marker}",
                fontsize=style.DENSE_PT, fontweight="normal",
                color=style.INK, linespacing=1.25,
            )

        # --- E: the effect size, and the refits that defend it ---------------
        #
        # These were two panels showing the same contrast: the production fit
        # with its interval, and the same contrast under six refits whose
        # baseline is a near-duplicate of it. As one forest the production fit
        # is simply the top row, which also puts it next to the robustness
        # baseline so the reader can see they are different runs agreeing
        # rather than one number quoted twice.
        style.panel_label(ax_robust, "E")
        scale_row = contrast_rows.loc[URGENCY_SCALE]
        contrast_grid = robustness.loc[
            (robustness["analysis"] == "ssm_robustness_parameter_contrast")
            & (robustness["parameter"] == URGENCY_SCALE)
        ].copy()
        order = ["baseline"] + [
            name for name in contrast_grid["configuration"] if name != "baseline"
        ]
        contrast_grid["order"] = contrast_grid["configuration"].map(
            {name: index for index, name in enumerate(order)}
        )
        contrast_grid = contrast_grid.sort_values("order", ascending=False)

        labels = [
            _ROBUSTNESS_LABELS.get(name, name)
            for name in contrast_grid["configuration"]
        ] + ["production fit"]
        centres = list(contrast_grid["mean_difference"]) + [
            float(scale_row["mean_difference"])
        ]
        lows = list(contrast_grid["ci95_low"]) + [float(scale_row["ci95_low"])]
        highs = list(contrast_grid["ci95_high"]) + [float(scale_row["ci95_high"])]
        # Colour marks the refit with no censored cell -- the one that answers
        # the censoring objection. The production fit is ink: it is the result,
        # not a sensitivity check.
        colors = [
            style.CONDITION_COLORS["fast"] if count == 0 else style.INK
            for count in contrast_grid["n_cells_near_upper_bound"]
        ] + [style.INK]
        forest(
            ax_robust,
            labels=labels,
            centres=np.asarray(centres, dtype=float),
            lows=np.asarray(lows, dtype=float),
            highs=np.asarray(highs, dtype=float),
            colors=colors,
            reference=0.0,
        )
        # A rule between the result and the checks on it.
        ax_robust.axhline(
            len(labels) - 1.5, color=style.INK_MUTED, linewidth=0.6,
            linestyle=":", zorder=0,
        )
        ax_robust.set_xlabel("Δ criterion scale (s)")
        ax_robust.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))

        fig.suptitle("Urgency parameters, Fast vs. Slow")

    counts = boundary.loc[boundary["model"] == "urgency"].set_index("condition")
    column = f"{URGENCY_SCALE}_near_upper"
    return fig, {
        "kind": "paired_condition_contrast_grid",
        "title": "Urgency parameters, Fast vs. Slow",
        "columns_read": {
            "ssmcomparison": [
                "subject", "condition", "model", "converged", *_URGENCY_PARAMETERS,
            ],
            "ssmcomparisonstats": [
                "analysis", "parameter", "mean_difference", "sem_difference",
                "ci95_low", "ci95_high", "equivalence_bound", "t", "p", "df",
                "cohens_dz",
            ],
            "ssmboundaryaudit": ["model", "condition", f"{URGENCY_SCALE}_near_upper"],
            "ssmrobustnessstats": [
                "analysis", "configuration", "parameter", "mean_difference",
                "ci95_low", "ci95_high", "n_cells_near_upper_bound",
            ],
        },
        "statistics_source": (
            "ssmcomparisonstats[analysis=ssm_urgency_condition_contrast] for A-E, "
            "intervals and equivalence bounds registered; "
            "ssmrobustnessstats[analysis=ssm_robustness_parameter_contrast] for F"
        ),
        "palette": {
            condition: style.CONDITION_COLORS[condition]
            for condition in ("fast", "slow")
        },
        "censored_cells": {
            condition: int(counts.loc[condition, column])
            for condition in ("fast", "slow")
            if column in counts and condition in counts.index
        },
        "caveat": (
            "criterion scale is threshold / urgency slope: smaller means "
            "faster-rising urgency, so a negative Fast-minus-Slow difference "
            "means urgency grows FASTER in Fast blocks. Ringed points are "
            "censored, not estimated, and pairs touching one are dashed; the "
            "censoring is asymmetric, 3 cells in Fast against 8 in Slow. F "
            "answers the objection that raises: the refit with no censored "
            "cell gives the largest contrast, so censoring attenuates this "
            "effect rather than creating it. F's baseline is the robustness "
            "grid's own refit (-0.118) rather than the production fit in A-E "
            "(-0.114) -- different runs, same conclusion. Parameter recovery "
            "is uneven: nondecision_s recovers at r = 0.99-1.00 and "
            "drift_scale at 0.94-0.99, but criterion scale at r = 0.44, 95% CI "
            "[-0.17, 0.81] and urgency onset at 0.33 -- these are group "
            "contrasts, never per-subject measurements. The non-decision panel "
            "is the residual after motor correction, which is why it sits at a "
            "few milliseconds."
        ),
    }


def build_ssmrecovery_identifiability(tables: BehaviorTableSet) -> tuple[Figure, dict[str, Any]]:
    """F15: can this design recover its own parameters and tell its models apart?

    Panel notes that are results, not layout:

    - **A is the contested pair.** `urgency_scale` and `urgency_onset_s` were
      the parameters previously reported as unidentified. Recovered against
      their true values they fall on the identity line.
    - **B is the whole grid**, so the two failures are visible rather than
      omitted: `nondecision_s` does not recover for urgency, `ddm` or
      collapsing.
    - **D explains B's failures instead of leaving them.** Recovery tracks how
      far a parameter's true spread exceeds the solver's own time
      discretisation. Additive urgency's non-decision spread is 98x the grid
      and recovers perfectly; the other three span less than one grid step and
      do not. A quantity cannot be estimated below the resolution of the
      solver that generated it, so this is a measured limit of the design, not
      a defect in the models.
    - **C is model selection**, which is the claim Act 3 actually rests on.

    Truths are drawn from the accepted fits, so `boundary_rate` is high: a
    subject whose `urgency_scale` sits at its 2.0 bound supplies a truth at
    that bound, and refitting it lands there again. The sidecar carries the
    rate per model.
    """
    parameter = tables.analysis("ssmparameterrecovery")
    parameter_stats = tables.analysis("ssmparameterrecoverystats")
    model_stats = tables.analysis("ssmmodelrecoverystats")

    with style.apply_publication_style():
        fig, axes = style.figure_grid(2, 2, width="double", panel_height_in=3.0)
        ax_a, ax_b = axes[0, 0], axes[0, 1]
        ax_c, ax_d = axes[1, 0], axes[1, 1]
        for ax, letter in zip((ax_a, ax_b, ax_c, ax_d), "ABCD"):
            style.panel_label(ax, letter)

        # --- A: the two parameters whose identifiability was contested ------
        contested = ("urgency_scale", "urgency_onset_s")
        markers = {"urgency_scale": "o", "urgency_onset_s": "s"}
        limits = []
        for name in contested:
            rows = parameter.loc[
                (parameter["true_model"] == "urgency")
                & (parameter["parameter"] == name)
            ]
            if rows.empty:
                continue
            true = rows["true_value"].to_numpy(dtype=float)
            estimated = rows["estimated_value"].to_numpy(dtype=float)
            limits.extend([true, estimated])
            ax_a.scatter(
                true, estimated, s=26, marker=markers[name],
                facecolors="none", edgecolors=style.INK, linewidths=1.0,
                label=_PARAMETER_LABELS.get(name, name),
            )
        if limits:
            flat = np.concatenate(limits)
            flat = flat[np.isfinite(flat)]
            span = (float(np.min(flat)), float(np.max(flat)))
            ax_a.plot(span, span, color=style.INK_MUTED, linewidth=0.9, zorder=0)
        ax_a.set(xlabel="True value", ylabel="Recovered value")
        ax_a.legend(loc="lower right", framealpha=0.85)

        # --- B: every model x parameter, with its interval ------------------
        grid = parameter_stats.dropna(subset=["correlation"]).copy()
        grid["label"] = (
            grid["true_model"].map(_SHORT_MODEL_LABELS).fillna(grid["true_model"])
            + " · " + grid["parameter"]
        )
        grid = grid.sort_values("correlation")
        if len(grid):
            forest(
                ax_b,
                labels=grid["label"].tolist(),
                centres=grid["correlation"].to_numpy(dtype=float),
                lows=grid["correlation_ci95_low"].to_numpy(dtype=float),
                highs=grid["correlation_ci95_high"].to_numpy(dtype=float),
                reference=0.0,
            )
            ax_b.tick_params(axis="y", labelsize=style.DENSE_PT)
        ax_b.set_xlabel("Recovery correlation")

        # --- C: model selection, true against selected ----------------------
        confusion = model_stats.pivot(
            index="true_model", columns="selected_model", values="selection_rate"
        )
        order = [name for name in style.MODEL_ORDER if name in confusion.index]
        confusion = confusion.reindex(index=order, columns=order).fillna(0.0)
        image = ax_c.imshow(
            confusion.to_numpy(dtype=float), cmap="Greys", vmin=0.0, vmax=1.0
        )
        ticks = range(len(order))
        labels = [_SHORT_MODEL_LABELS.get(name, name) for name in order]
        ax_c.set_xticks(list(ticks), labels, rotation=30, ha="right",
                        fontsize=style.DENSE_PT)
        ax_c.set_yticks(list(ticks), labels, fontsize=style.DENSE_PT)
        for row in ticks:
            for column in ticks:
                rate = float(confusion.to_numpy()[row, column])
                if rate <= 0:
                    continue
                ax_c.text(
                    column, row, f"{rate:.2f}", ha="center", va="center",
                    fontsize=style.DENSE_PT,
                    color=style.SURFACE if rate > 0.5 else style.INK,
                )
        ax_c.set(xlabel="Selected", ylabel="Simulated")
        image.set_rasterized(True)

        # --- D: why the non-decision rows in B fail -------------------------
        spans = (
            parameter.groupby(["true_model", "parameter"])["true_value"]
            .agg(lambda values: float(np.nanmax(values) - np.nanmin(values)))
            .rename("spread")
            .reset_index()
        )
        merged = parameter_stats.merge(spans, on=["true_model", "parameter"])
        merged = merged.dropna(subset=["correlation", "spread"])
        merged = merged.loc[merged["spread"] > 0]
        if len(merged):
            ratio = merged["spread"].to_numpy(dtype=float) / SOLVER_STEP_S
            ax_d.scatter(
                ratio, merged["correlation"].to_numpy(dtype=float),
                s=26, facecolors="none", edgecolors=style.INK, linewidths=1.0,
            )
            ax_d.set_xscale("log")
            ax_d.axvline(1.0, color=style.INK_MUTED, linewidth=0.9, zorder=0)
            ax_d.axhline(0.0, color=style.INK_MUTED, linewidth=0.9, zorder=0)
        ax_d.set(
            xlabel="True spread ÷ solver step", ylabel="Recovery correlation",
        )

        fig.suptitle("Parameter and model recovery, in the fitted region")

    boundary = (
        parameter_stats.groupby("true_model")["boundary_rate"].first().round(2).to_dict()
        if "boundary_rate" in parameter_stats else {}
    )
    metadata = {
        "kind": "recovery_identifiability",
        "title": "Parameter and model recovery, in the fitted region",
        "columns_read": {
            "ssmparameterrecovery": ["true_model", "parameter", "true_value", "estimated_value"],
            "ssmparameterrecoverystats": [
                "true_model", "parameter", "correlation",
                "correlation_ci95_low", "correlation_ci95_high", "boundary_rate",
            ],
            "ssmmodelrecoverystats": ["true_model", "selected_model", "selection_rate"],
        },
        "statistics_source": (
            "ssmparameterrecoverystats for A, B and D; ssmmodelrecoverystats "
            "for C. D's spread ratio is a descriptive summary of the persisted "
            "true values against SOLVER_STEP_S."
        ),
        "uncertainty": "registered 95% CIs on the recovery correlations in B",
        "solver_step_s": SOLVER_STEP_S,
        "boundary_rate": boundary,
        "caveat": (
            "Truths are the accepted per-subject fits, not a hypercube over the "
            "optimizer bounds, so this answers whether the models are separable "
            "*where the data live* rather than somewhere in parameter space. "
            "That is the correction: the previous study drew urgency drift_scale "
            "truths at 1.125-3.875 against fitted values of 0.083-0.383, and "
            "reported urgency_scale recovering at r = 0.44 with an interval "
            "spanning zero. Here it recovers at r = 0.99. Two consequences. "
            "First, boundary_rate is high (0.33-0.67) because real fits sitting "
            "at a bound supply truths at that bound. Second, nondecision_s does "
            "not recover for urgency, ddm or collapsing, and D shows why: their "
            "true spread is under one solver step, so the quantity is finer "
            "than the discretisation that generated it. Additive urgency's "
            "non-decision spread is 98x the step and recovers at r = 1.00. "
            "Model selection is 47 of 48 correct, and more decisive than the "
            "superseded study: the median BIC margin between the best and "
            "second model is 22.8 points against 8.9 before."
        ),
    }
    return fig, metadata


