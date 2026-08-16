"""Turn a *stats derivative row into figure text and brackets.

The plotting layer never computes an inferential statistic (t, p, F, df, dz,
partial eta squared, r) -- every one printed on a figure is read from a
derivative via these helpers. The layer may compute only display quantities
(means, SEMs, within-subject confidence intervals, kernel densities, binned
means, quantiles of already-persisted subject-level values); see
`within_subject_error` in panels.py for the one such helper. This is what
keeps a figure from ever disagreeing with docs/behavior_roadmap_results.md.
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
    extra: Mapping[str, float] = field(default_factory=dict)  # F, eta_p2, r ...


def stat_from_row(row: pd.Series, *, label: str) -> StatResult:
    """Build a StatResult from any battery *stats row.

    Reads the shared column names emitted by
    `behavior/math/inference.one_sample_statistics` and
    `paired_subject_statistics`: n_subjects, mean, sem, mean_a, sem_a, mean_b,
    sem_b, mean_difference, t, p, df, cohens_dz. Prefers `mean`; falls back to
    `mean_difference`. Absent or non-finite fields become None -- never 0, and
    never rendered as "nan".
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
        sem=_finite_or_none(row.get("sem")),
        t=_finite_or_none(row.get("t")),
        p=_finite_or_none(row.get("p")),
        df=_finite_or_none(row.get("df")),
        cohens_dz=_finite_or_none(row.get("cohens_dz")),
        mean_a=_finite_or_none(row.get("mean_a")),
        mean_b=_finite_or_none(row.get("mean_b")),
    )


def significance_marker(p: float | None) -> str:
    """'***' p < .001 | '**' p < .01 | '*' p < .05 | 'n.s.' otherwise |
    '' when p is None or non-finite."""
    if p is None or not math.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def format_p(p: float | None) -> str:
    """'p < .001' below 1e-3; otherwise 'p = .017' -- two significant digits,
    APA leading zero dropped."""
    if p is None or not math.isfinite(p):
        return "p = n/a"
    if p < 0.001:
        return "p < .001"
    text = f"{p:.3f}".lstrip("0")
    return f"p = {text}"


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
        fontsize=7,
    )
    if show_text:
        ax.text(
            (x_a + x_b) / 2,
            y * 1.02,
            format_stat(result, unit=unit, include_effect_size=False),
            ha="center",
            va="bottom",
            color=style.INK_SECONDARY,
            fontsize=6,
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
        fontsize=9,
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
    ax: Axes, rows: pd.DataFrame, *, loc: str = "upper left", y: float | None = None
) -> None:
    """Render conditionclassstats rows as compact 'condition ηp² = .50 ***' lines.

    Effect size plus a significance marker, matching the figure-wide
    'Δ = value marker' convention: spelled-out 'F(2, 62) = 1.19, p = .310'
    text is wide enough to collapse constrained_layout's axes on a
    half-width panel. F, df and p stay in the derivative and the JSON
    sidecar, which is where anyone quoting them should read them.
    """
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
        pieces.append(marker or "n.s.")
        lines.append(" ".join(pieces))
    annotate_stat_block(ax, lines=lines, loc=loc, y=y)
