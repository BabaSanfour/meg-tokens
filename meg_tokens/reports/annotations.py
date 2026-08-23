"""Turn a *stats derivative row into figure text and brackets.

The plotting layer never computes an inferential statistic (t, p, F, df, dz,
partial eta squared, r) -- every one printed on a figure is read from a
derivative via these helpers. The layer may compute only display quantities
(means, SEMs, within-subject confidence intervals, kernel densities, binned
means, quantiles of already-persisted subject-level values); see
`within_subject_error` in panels.py for the one such helper. This is what
keeps a figure from ever disagreeing with docs/behavior.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final, Mapping, Sequence

import pandas as pd
from matplotlib.axes import Axes

from meg_tokens.reports import style

SIGNIFICANCE_CONVENTION: Final[str] = "*** p<.001, ** p<.01, * p<.05, n.s. otherwise"


def _finite_or_none(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class StatResult:
    label: str
    n_subjects: int
    mean: float | None  # `mean` or `mean_difference`
    sem: float | None
    t: float | None
    p: float | None
    df: float | None
    cohens_dz: float | None
    mean_a: float | None = None
    mean_b: float | None = None
    # The registered interval on the contrast, when the analysis layer wrote
    # one. A figure must draw this rather than rebuilding an interval from
    # mean/t/df -- rebuilding is the plotting layer computing an inferential
    # statistic, which the reporting rules forbid.
    ci95_low: float | None = None
    ci95_high: float | None = None
    extra: Mapping[str, float] = field(default_factory=dict)  # F, eta_p2, r ...


def stat_from_row(row: pd.Series, *, label: str) -> StatResult:
    """Build a StatResult from any battery *stats row.

    Reads the shared column names emitted by
    `behavior/math/inference.one_sample_statistics` and
    `paired_subject_statistics`: n_subjects, mean, sem, mean_a, sem_a, mean_b,
    sem_b, mean_difference, sem_difference, ci95_low, ci95_high, t, p, df,
    cohens_dz. Prefers `mean`; falls back to `mean_difference`, and `sem`
    likewise falls back to `sem_difference` -- a paired contrast registers the
    uncertainty on the *difference* under that name, and `sem_a`/`sem_b` are
    the per-condition SEMs, which ignore the pairing and are not it. Absent or
    non-finite fields become None -- never 0, and never rendered as "nan".
    """
    mean = _finite_or_none(row.get("mean"))
    if mean is None:
        mean = _finite_or_none(row.get("mean_difference"))
    n_subjects_raw = row.get("n_subjects")
    n_subjects = int(n_subjects_raw) if n_subjects_raw is not None and not (
        isinstance(n_subjects_raw, float) and math.isnan(n_subjects_raw)
    ) else 0
    return StatResult(
        label=label,
        n_subjects=n_subjects,
        mean=mean,
        sem=(
            _finite_or_none(row.get("sem"))
            if _finite_or_none(row.get("sem")) is not None
            else _finite_or_none(row.get("sem_difference"))
        ),
        t=_finite_or_none(row.get("t")),
        p=_finite_or_none(row.get("p")),
        df=_finite_or_none(row.get("df")),
        cohens_dz=_finite_or_none(row.get("cohens_dz")),
        mean_a=_finite_or_none(row.get("mean_a")),
        mean_b=_finite_or_none(row.get("mean_b")),
        ci95_low=_finite_or_none(row.get("ci95_low")),
        ci95_high=_finite_or_none(row.get("ci95_high")),
    )


ALPHA = 0.05
"""The threshold :func:`significance_marker` stars at, and the one a figure
should test against when it decides whether to print a marker at all.

Figures that suppress a marker below significance, or that swap in an
equivalence bound instead, need the same number the marker function uses.
Spelling `0.05` again at each of those call sites is how the two drift apart.
"""


def is_significant(p: float | None) -> bool:
    """True when `p` is a real result below :data:`ALPHA`.

    Missing and non-finite p are False: they are untested, not null.
    """
    return p is not None and math.isfinite(p) and p < ALPHA


def significance_marker(p: float | None) -> str:
    """'***' p < .001 | '**' p < .01 | '*' p < .05 | 'n.s.' otherwise |
    '' when p is None or non-finite."""
    if p is None or not math.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < ALPHA:
        return "*"
    return "n.s."


def marker_or_missing(p: float | None) -> str:
    """:func:`significance_marker`, with 'n/a' when there is no test result.

    ``significance_marker(p) or "n.s."`` is the tempting spelling and it is
    wrong twice over. ``significance_marker`` already returns 'n.s.' for a
    non-significant p, so the fallback never fires on a real result -- and the
    only case it *does* fire on is a missing or non-finite p, where printing
    'n.s.' asserts a null that was never tested. A cell with too few subjects,
    or zero variance, would be reported as a tested non-effect.

    'n/a' matches :func:`format_p`, which already spells missing this way.
    """
    return significance_marker(p) or "n/a"


def format_p(p: float | None) -> str:
    """'p < .001' below 1e-3; otherwise 'p = .017' -- two significant digits,
    APA leading zero dropped."""
    if p is None or not math.isfinite(p):
        return "p = n/a"
    if p < 0.001:
        return "p < .001"
    text = f"{p:.3f}".lstrip("0")
    return f"p = {text}"


def format_r(value: float | None) -> str:
    """'-.81' -- a correlation with no leading zero, APA style.

    Same convention as :func:`format_p` applies to its p. Worth a quarter of
    the string's width, which in a heatmap cell is the difference between
    adjacent numbers reading as two correlations and reading as one run of
    digits.
    """
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.2f}".replace("0.", ".", 1)


def format_stat(
    result: StatResult, *, unit: str = "", include_effect_size: bool = True
) -> str:
    """'Δ = -108, t(31) = -2.52, p = .017, dz = -0.45'"""
    parts: list[str] = []
    if result.mean is not None:
        suffix = f" {unit}" if unit else ""
        parts.append(f"Δ = {result.mean:.3g}{suffix}")
    if result.t is not None and result.df is not None:
        parts.append(f"t({result.df:.0f}) = {result.t:.2f}")
    parts.append(format_p(result.p))
    if include_effect_size and result.cohens_dz is not None:
        parts.append(f"dz = {result.cohens_dz:.2f}")
    return ", ".join(parts)


def annotate_contrast(
    ax: Axes,
    *,
    x_a: float,
    x_b: float,
    y: float,
    result: StatResult,
    unit: str = "",
    show_text: bool = True,
) -> None:
    """Significance bracket between two x positions: marker plus optional stat text."""
    ax.plot([x_a, x_a, x_b, x_b], [y * 0.98, y, y, y * 0.98], color=style.INK, linewidth=0.8)
    marker = significance_marker(result.p)
    label = marker if marker not in ("", "n.s.") else "n.s."
    ax.text(
        (x_a + x_b) / 2,
        y,
        label,
        ha="center",
        va="bottom",
        color=style.INK,
        fontsize=style.SIGNIFICANCE_PT,
    )
    if show_text:
        ax.text(
            (x_a + x_b) / 2,
            y * 1.02,
            format_stat(result, unit=unit, include_effect_size=False),
            ha="center",
            va="bottom",
            color=style.INK_SECONDARY,
            fontsize=style.ANNOTATION_PT,
        )


def annotate_stat_block(
    ax: Axes, *, lines: Sequence[str], loc: str = "upper left", y: float | None = None
) -> None:
    """Multi-line stat text in ink (never a series colour), in axes coordinates.

    `y` overrides the corner's vertical position (axes fraction), keeping its
    horizontal alignment -- for panels whose corner is already occupied by
    another annotation.
    """
    positions = {
        "upper left": (0.02, 0.98, "left", "top"),
        "upper right": (0.98, 0.98, "right", "top"),
        "lower left": (0.02, 0.02, "left", "bottom"),
        "lower right": (0.98, 0.02, "right", "bottom"),
    }
    x, default_y, ha, va = positions.get(loc, positions["upper left"])
    y = default_y if y is None else y
    ax.text(
        x,
        y,
        "\n".join(lines),
        transform=ax.transAxes,
        ha=ha,
        va=va,
        color=style.INK,
        fontsize=style.ANNOTATION_PT,
        linespacing=1.4,
    )


# Raw factor names are column identifiers, not display text -- same reason
# 'dt_ms' is scrubbed from axis labels.
_EFFECT_LABELS = {
    "condition": "condition",
    "trial_class": "difficulty",
    "condition_x_trial_class": "interaction",
}


def annotate_anova(
    ax: Axes,
    rows: pd.DataFrame,
    *,
    loc: str = "upper left",
    y: float | None = None,
    bounds: dict[str, str | None] | None = None,
) -> None:
    """Render conditionclassstats rows as compact 'condition ηp² = .50 ***' lines.

    Effect size plus a significance marker, matching the figure-wide
    'Δ = value marker' convention: spelled-out 'F(2, 62) = 1.19, p = .310'
    text is wide enough to collapse constrained_layout's axes on a
    half-width panel. F, df and p stay in the derivative and the JSON
    sidecar, which is where anyone quoting them should read them.

    ``bounds`` maps an effect name to text shown in place of the significance
    marker -- for a null the figure argues *from* rather than merely reports,
    where a bound states the claim and 'n.s.' does not. It applies only to
    effects that did not reach significance, so a bound can never displace a
    marker on a real effect.
    """
    bounds = bounds or {}
    lines = []
    for _, row in rows.iterrows():
        effect = str(row.get("effect", ""))
        eta = _finite_or_none(row.get("partial_eta_squared"))
        p_value = _finite_or_none(row.get("p"))
        pieces = [_EFFECT_LABELS.get(effect, effect)]
        if eta is not None:
            eta_text = f"{eta:.2f}".lstrip("0") if eta < 1 else f"{eta:.2f}"
            pieces.append(f"ηp² = {eta_text}")
        marker = significance_marker(p_value) if p_value is not None else ""
        # `significance_marker` returns the literal 'n.s.' rather than an
        # empty string, so a bound has to be selected on that value, not on
        # falsiness -- an `or` chain here silently never fires.
        bound = bounds.get(effect)
        pieces.append(
            bound if bound and marker in ("", "n.s.") else (marker or "n.s.")
        )
        lines.append(" ".join(pieces))
    annotate_stat_block(ax, lines=lines, loc=loc, y=y)
