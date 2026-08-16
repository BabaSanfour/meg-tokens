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

MODEL_ORDER: Final[tuple[str, ...]] = ("urgency", "ddm")
MODEL_COLORS: Final[Mapping[str, str]] = {
    "urgency": "#008300",  # green
    "ddm": "#e87ba4",  # magenta
}
MODEL_LABELS: Final[Mapping[str, str]] = {
    "urgency": "Urgency gating",
    "ddm": "Bounded integrator",
}
# CVD dE 17.6 PASS - normal-vision dE 35.9 PASS
# ddm (#e87ba4) is 2.62:1 on white -> relief rule applies here too: always a
# direct label or legend text, never colour alone (same as CLASS_COLORS.easy).

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

# Raster save resolution. Exported so callers saving a figure *outside* an
# `apply_publication_style()` block (behavior_summary._render_one always
# does -- the builder returns the figure, then the rc_context has already
# exited by the time savefig() runs) can still request it explicitly instead
# of silently falling back to matplotlib's 100 dpi default.
SAVEFIG_DPI: Final[int] = 400

_PUBLICATION_RC: Final[dict[str, object]] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 13,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
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


def panel_label(ax: Axes, letter: str) -> None:
    """Stamp a bold panel letter (A, B, C...) at a fixed offset."""
    ax.text(
        -0.18,
        1.06,
        letter,
        transform=ax.transAxes,
        fontsize=17,
        fontweight="bold",
        va="top",
        ha="left",
        color=INK,
    )
