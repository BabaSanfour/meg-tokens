"""Shared palette, rcParams, and panel geometry for publication figures.

Project-wide, not behavior-specific, so MEG figures can adopt it later.

Palette validated with the ``dataviz`` skill's ``validate_palette.js`` in
light mode with ``--pairs all``: every group of hues that can co-occur in one
panel passes the lightness band, chroma floor, CVD separation, and
normal-vision floor checks.

**Scoping rule.** The three palettes below are *scoped*: class hues,
condition hues, and model hues are pairwise disjoint, and no panel mixes two
scopes as colour. Where class and condition both appear (e.g. condition x
class effects), class is on the x-axis and only condition is coloured. Where
model and condition both appear (e.g. model time courses), condition is a
facet and only model is coloured. This is why each palette can be validated
independently of the others.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Final, Iterator, Literal, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes

CLASS_ORDER: Final[tuple[str, ...]] = ("easy", "ambiguous", "misleading")
CLASS_COLORS: Final[Mapping[str, str]] = {
    "easy": "#1baf7a",  # aqua
    "ambiguous": "#2a78d6",  # blue
    "misleading": "#eb6834",  # orange
}
# validate_palette.js "#1baf7a,#2a78d6,#eb6834" --mode light --pairs all
#   CVD dE 9.2 (deutan) PASS - normal-vision dE 24.0 PASS
#   easy (#1baf7a) is 2.74:1 on white -> relief rule: it always carries a
#   direct label or legend text, never colour alone.

CONDITION_ORDER: Final[tuple[str, ...]] = ("fast", "slow")
CONDITION_COLORS: Final[Mapping[str, str]] = {
    "fast": "#e34948",  # red
    "slow": "#4a3aa7",  # violet
    "all": "#0b0b0b",  # ink, for the pooled reference line
}
# CVD dE 22.7 PASS - normal-vision dE 33.6 PASS - both >= 3:1 contrast

MODEL_ORDER: Final[tuple[str, ...]] = (
    "urgency", "collapsing", "additive_urgency", "ddm",
)
MODEL_COLORS: Final[Mapping[str, str]] = {
    "urgency": "#008300",  # green
    "collapsing": "#7a5c3e",  # brown
    "additive_urgency": "#008c95",  # teal
    "ddm": "#e87ba4",  # magenta
}
MODEL_LABELS: Final[Mapping[str, str]] = {
    "urgency": "Urgency gating",
    "collapsing": "Collapsing bound",
    "additive_urgency": "Additive urgency",
    "ddm": "Fixed-bound integrator",
}
# CVD dE 17.6 PASS - normal-vision dE 35.9 PASS
# ddm (#e87ba4) is 2.62:1 on white -> relief rule applies here too: always a
# direct label or legend text, never colour alone (same as CLASS_COLORS.easy).

QC_ORDER: Final[tuple[str, ...]] = ("extreme_dt", "lapse", "anticipation")
QC_COLORS: Final[Mapping[str, str]] = {
    "extreme_dt": "#0b0b0b",  # ink
    "lapse": "#b8860b",  # goldenrod
    "anticipation": "#b7dce8",  # powder blue
}
QC_LABELS: Final[Mapping[str, str]] = {
    "extreme_dt": "extreme DT",
    "lapse": "lapse",
    "anticipation": "anticipation",
}
# A fourth scope, for the data-quality census. The saturated hues are all
# spoken for -- aqua/blue/orange by CLASS, red/violet by CONDITION,
# green/magenta by MODEL -- and the one free neutral (#9a9992) is already
# 'unclassified', which shares F10's legend with these. So this scope
# separates primarily on lightness: solid ink, dark goldenrod, and a pale
# powder blue. Goldenrod sits at hue 43 deg against
# misleading's 15 deg and is darker than it, so the two do not trade even
# in the shared legend. Powder blue adds a cool/warm distinction from lapse,
# while its much higher lightness keeps it apart from the blue and aqua class
# fills. Counts here are frequently 1-3 trials, i.e. a bar a few pixels wide,
# which is why lightness carries the distinction rather than relying on hue
# alone.

OBSERVED: Final[str] = "#52514e"  # observed data behind model fits
SUBJECT_LINE: Final[str] = "#9a9992"  # per-subject traces
SUBJECT_ALPHA: Final[float] = 0.30
INK: Final[str] = "#0b0b0b"
INK_SECONDARY: Final[str] = "#52514e"
INK_MUTED: Final[str] = "#9a9992"
SURFACE: Final[str] = "#ffffff"
DIVERGING: Final[tuple[str, str, str]] = ("#2a78d6", "#f0efec", "#e34948")

WIDTH_SINGLE_IN: Final[float] = 3.42  # 87 mm, single column
WIDTH_DOUBLE_IN: Final[float] = 7.09  # 180 mm, double column
PANEL_HEIGHT_IN: Final[float] = 2.85

# Type scale for text drawn *inside* a panel. Axis labels, titles, ticks and
# legends come from the rcParams below and must never be overridden at the
# call site; these three cover the cases matplotlib has no rcParam for.
#
# They exist so that in-panel text stays tied to the rc scale. Passing a bare
# `fontsize=` at a call site silently pins that text to whatever the scale
# happened to be the day it was written, which is how this module ended up
# with annotations at 10-13 pt sitting beside 8 pt axis labels.
SIGNIFICANCE_PT: Final[float] = 10.0  # *, **, *** markers over a comparison
ANNOTATION_PT: Final[float] = 9.0     # in-panel notes, bounds, secondary stats
DENSE_PT: Final[float] = 7.5          # small multiples: per-subject panels, matrices
PANEL_LETTER_PT: Final[float] = 12.0  # A, B, C... panel letters

# Raster save resolution. Exported so callers saving a figure *outside* an
# `apply_publication_style()` block (behavior_summary._render_one always
# does -- the builder returns the figure, then the rc_context has already
# exited by the time savefig() runs) can still request it explicitly instead
# of silently falling back to matplotlib's 100 dpi default.
SAVEFIG_DPI: Final[int] = 400

_PUBLICATION_RC: Final[dict[str, object]] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    # Sized for a 180 mm double-column figure that is also read on screen.
    #
    # Axis names and tick labels are the text a reader must resolve to use the
    # figure at all, so they sit deliberately above the 8/7 pt journal floor.
    # The ceiling is set by layout, not taste: this module previously ran at
    # 16/18 pt, where constrained_layout gives up and the damage is real --
    # clipped suptitles, axis labels cut mid-word, panel letters landing on a
    # neighbour's y-label. These values clear that by a wide margin.
    #
    # Registered in docs/behavior.md (Figure conventions); change both
    # together.
    "font.size": 9,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "savefig.dpi": SAVEFIG_DPI,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.6,
    "lines.markersize": 4,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    # Numeric labels stay (the data needs them); the little dash marks are
    # pure clutter at this line weight, so their length is zeroed instead.
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "figure.constrained_layout.use": True,
    "savefig.bbox": "tight",
    "savefig.transparent": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}
# No axes.prop_cycle override: colour is always assigned explicitly by
# entity in this module (never left to cycle order), so a filter that
# changes a panel's series count can never silently repaint the survivors.


@contextmanager
def apply_publication_style() -> Iterator[None]:
    """Scoped publication rcParams. Never leaks into the caller's session.

    Colour is always assigned explicitly by entity in this codebase (never
    relies on the cycler to disambiguate series), so the cycler here is a
    single-colour fallback, not a rotation a filtered series count could
    silently repaint.
    """
    with plt.rc_context(_PUBLICATION_RC):
        yield


def figure_grid(
    n_rows: int = 1,
    n_cols: int = 1,
    *,
    width: Literal["single", "double"] | float = "double",
    panel_height_in: float = PANEL_HEIGHT_IN,
    width_ratios: Sequence[float] | None = None,
    height_ratios: Sequence[float] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Create a styled figure and a 2-D array of Axes at publication size.

    Must be called inside :func:`apply_publication_style`. Returns axes as a
    2-D ``(n_rows, n_cols)`` object array regardless of grid shape, so callers
    can always index ``axes[row, col]``.

    `width` is normally "single" or "double" (the two publication presets);
    pass a float (inches) directly for a one-off width that doesn't fit
    either preset.
    """
    if isinstance(width, str):
        fig_width = WIDTH_DOUBLE_IN if width == "double" else WIDTH_SINGLE_IN
    else:
        fig_width = width
    fig_height = panel_height_in * n_rows
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        gridspec_kw={
            "width_ratios": width_ratios,
            "height_ratios": height_ratios,
        },
    )
    return fig, axes


def panel_label(
    ax: Axes, letter: str, *, offset_points: tuple[float, float] = (-26.0, 10.0)
) -> None:
    """Stamp a bold panel letter (A, B, C...) outside the axes' top-left.

    The offset is in **points**, not axes fractions. An axes-fraction offset
    scales with panel width, so the same constant that clears the y-axis
    label in a one-column figure lands on top of the neighbouring panel's
    label in a four-column one -- which is what it used to do. Points are
    invariant to the grid shape, so one constant is correct everywhere.

    `offset_points` overrides it for the case the constant cannot cover: a
    panel whose y-axis label is long enough to sit where the letter goes. That
    is a property of the label, not of the grid, so it cannot be folded back
    into the default -- widening it for everyone would push the letters of
    short-labelled panels into their neighbours.
    """
    ax.annotate(
        letter,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=offset_points,
        textcoords="offset points",
        fontsize=PANEL_LETTER_PT,
        fontweight="bold",
        va="top",
        ha="left",
        color=INK,
        annotation_clip=False,
    )
