"""Reusable panel builders. Each draws onto a caller-supplied Axes and
returns None; the figure modules in reports/behavior/ own layout. This keeps
the panel library composable and testable without file I/O.
"""

from __future__ import annotations

from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy import stats as scipy_stats

from meg_tokens.reports import style
from meg_tokens.reports.annotations import (
    StatResult, format_r, marker_or_missing, significance_marker,
)


def within_subject_error(
    values: np.ndarray, *, axis: int = 0, kind: Literal["sem", "ci95"] = "sem"
) -> np.ndarray:
    """Cousineau (1994) normalisation with the Morey (2008) bias correction.

    DISPLAY ONLY. Never used to derive a reported statistic; every inferential
    number on a figure comes from a *stats derivative (annotations.py).

    Parameters
    ----------
    values
        (n_subjects, n_conditions) array. NaN-tolerant per condition column.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    n_conditions = array.shape[1]
    # Subjects with no finite value anywhere in this design (e.g. a
    # non-converged fit) contribute nothing either way; excluding their
    # all-NaN row up front avoids numpy's "Mean of empty slice" warning on
    # an expected, already-handled case rather than a real problem.
    has_data = np.any(np.isfinite(array), axis=1)
    array = array[has_data]
    if array.size == 0:
        return np.full(n_conditions, np.nan)
    subject_mean = np.nanmean(array, axis=1, keepdims=True)
    grand_mean = np.nanmean(array)
    normalized = array - subject_mean + grand_mean
    correction = np.sqrt(n_conditions / (n_conditions - 1)) if n_conditions > 1 else 1.0
    n_subjects = np.sum(~np.isnan(normalized), axis=0)
    sd = np.nanstd(normalized, axis=0, ddof=1) * correction
    sem = np.divide(
        sd, np.sqrt(n_subjects), out=np.full_like(sd, np.nan), where=n_subjects > 0
    )
    if kind == "sem":
        return sem
    ci95 = np.full_like(sem, np.nan)
    valid = n_subjects > 1
    ci95[valid] = sem[valid] * scipy_stats.t.ppf(0.975, n_subjects[valid] - 1)
    return ci95


def paired_slope(
    ax: Axes,
    *,
    values_a: np.ndarray,
    values_b: np.ndarray,
    label_a: str,
    label_b: str,
    color_a: str,
    color_b: str,
    ylabel: str = "",
    subject_line_color: str = style.SUBJECT_LINE,
    show_group_mean: bool = True,
    reference: float | None = None,
    increase_color: str | None = None,
    decrease_color: str | None = None,
    dashed_mask: np.ndarray | None = None,
) -> None:
    """Two x positions, one line per subject, group mean +/- within-subject CI.

    By default every subject line is `subject_line_color`. Pass both
    `increase_color` and `decrease_color` to color each line by the sign of
    b - a instead -- e.g. to show at a glance how many subjects go against
    the group's direction, not just that the group mean moved.

    `dashed_mask` dashes the lines it selects. Use it where an endpoint is not
    a measurement -- a censored estimate sitting on its bound, say. Two
    censored endpoints otherwise draw a flat, confident-looking segment that
    reads as "this subject did not change", which is a claim the fit never
    made.
    """
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    valid = np.isfinite(values_a) & np.isfinite(values_b)
    values_a, values_b = values_a[valid], values_b[valid]
    dashes = (
        np.asarray(dashed_mask, dtype=bool)[valid]
        if dashed_mask is not None
        else np.zeros(values_a.shape, dtype=bool)
    )

    if reference is not None:
        ax.axhline(reference, color=style.INK_MUTED, linewidth=0.6, zorder=0)

    for a, b, dashed in zip(values_a, values_b, dashes):
        line_color = subject_line_color
        if increase_color is not None and decrease_color is not None:
            line_color = increase_color if b >= a else decrease_color
        line, = ax.plot(
            [0, 1], [a, b],
            color=line_color, alpha=style.SUBJECT_ALPHA,
            linewidth=0.8, zorder=1,
        )
        if dashed:
            line.set_dashes((1.5, 1.2))
    ax.scatter(
        np.zeros_like(values_a), values_a, color=color_a, s=12, zorder=2, label=label_a
    )
    ax.scatter(
        np.ones_like(values_b), values_b, color=color_b, s=12, zorder=2, label=label_b
    )

    if show_group_mean and values_a.size:
        stacked = np.stack([values_a, values_b], axis=1)
        errors = within_subject_error(stacked, kind="ci95")
        means = np.nanmean(stacked, axis=0)
        ax.errorbar(
            [0, 1], means, yerr=errors,
            color=style.INK, marker="o", markersize=5,
            linewidth=1.6, capsize=3, zorder=3,
        )

    # The two columns sit at 0 and 1; 0.4 of margin on each side is more
    # empty panel than the data occupies between them.
    ax.set_xlim(-0.22, 1.22)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([label_a, label_b])
    ax.set_ylabel(ylabel)


def estimation_axis(
    ax_diff: Axes,
    *,
    differences: np.ndarray,
    result: StatResult,
    unit: str = "",
    color: str = style.INK,
    ylabel: str | None = None,
    spread: float = 0.06,
    symbol: str = "Δ",
) -> None:
    """Gardner-Altman difference axis: subject values, mean, 95% CI, zero rule.

    `ylabel` defaults to "Δ <unit>", which is only legible when the reader
    already knows what the difference is of. Name the quantity whenever the
    unit alone does not.

    `symbol` names the statistic in the title bracket. It defaults to Δ, the
    battery's convention for a *contrast*, and must be changed for anything
    else: the same axis draws any per-subject estimate tested against zero, and
    a correlation or a rate labelled Δ claims a comparison between two
    conditions that was never made.
    """
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    n = differences.size

    ax_diff.axhline(0, color=style.INK_MUTED, linewidth=0.6, zorder=0)
    jitter = np.linspace(-spread, spread, n) if n else np.array([])
    ax_diff.scatter(
        jitter, differences, color=style.SUBJECT_LINE, s=10, alpha=0.6, zorder=1
    )

    if result.mean is not None:
        # Prefer the interval the analysis layer registered. Only fall back to
        # rebuilding one from the registered sem, and never from mean/t: that
        # last route is the plotting layer computing an inferential statistic.
        #
        # The old guard asked for `sem` before running a formula that does not
        # use `sem`, so a contrast row carrying mean/t/df but no sem -- which
        # is every paired contrast this battery registers -- fell through both
        # branches and the panel drew a bare dot with no interval at all. An
        # estimation plot whose whole subject is the effect size showed no
        # uncertainty, silently.
        lower = upper = None
        if result.ci95_low is not None and result.ci95_high is not None:
            lower = result.mean - result.ci95_low
            upper = result.ci95_high - result.mean
        elif result.sem is not None and result.df is not None:
            half = result.sem * scipy_stats.t.ppf(0.975, result.df)
            lower = upper = half
        ax_diff.errorbar(
            [0], [result.mean],
            yerr=[[lower], [upper]] if lower is not None else None,
            color=color, marker="o", markersize=6, linewidth=1.8, capsize=4, zorder=2,
        )

    # The jitter spans +/-`spread`; -0.3..0.3 left most of the panel empty.
    ax_diff.set_xlim(-0.14, 0.14)
    ax_diff.set_xticks([])
    ax_diff.set_ylabel(
        ylabel if ylabel is not None else f"Δ{(' ' + unit) if unit else ''}"
    )
    # The effect and its p, on one short line, as the axes title -- which is
    # the one place constrained_layout reserves room for. The full t/df/dz is
    # in the sidecar; spelled out here it was two wrapped lines over the
    # narrowest panel in the figure, and free-floating text fared worse still
    # (below the axis it hung off the figure edge; inside it landed on the top
    # tick label).
    # `<symbol> = value unit marker`, the battery's compact convention. The
    # marker, not a numeric p: `marker_or_missing` spells a non-significant
    # result 'n.s.' and an untested one 'n/a', so a cell with too few subjects
    # is never reported as a tested null. Exact p, t, df and dz are in the
    # sidecar.
    compact = []
    if result.mean is not None:
        compact.append(
            f"{symbol} = {result.mean:.3g}{(' ' + unit) if unit else ''}"
        )
    compact.append(marker_or_missing(result.p))
    ax_diff.set_title(
        " ".join(compact),
        fontsize=style.ANNOTATION_PT, fontweight="normal", color=style.INK,
    )


def subject_strip(
    ax: Axes,
    *,
    groups: Mapping[str, np.ndarray],
    colors: Mapping[str, str],
    reference: float | None = 0.0,
    ylabel: str = "",
    connect: Sequence[tuple[str, str]] = (),
    spread: float = 0.08,
    estimates: Mapping[str, tuple[float, float, float]] | None = None,
) -> None:
    """Jittered per-subject estimates per group, group mean +/- 95% CI,
    optional subject connectors between named group pairs.

    `spread` is the half-width of the jitter in x-axis units, where the group
    pitch is 1.0. Widen it when a panel has few groups, so the columns read as
    distributions rather than as three pinstripes with gaps between them.

    `estimates` supplies the group summary as `{label: (mean, low, high)}`
    instead of computing it from the dots. Pass it whenever the derivative
    already registers the mean and its interval: a recomputed interval is a
    second, unpersisted inferential statistic that can disagree with the one
    the text quotes. Without it the summary is derived from the plotted
    values, which is a display quantity and fine for a panel whose groups are
    not themselves a registered contrast.
    """
    if reference is not None:
        ax.axhline(reference, color=style.INK_MUTED, linewidth=0.6, zorder=0)

    labels = list(groups.keys())
    positions = {label: index for index, label in enumerate(labels)}

    # Compute the jitter once on the original subject-indexed arrays. Reusing
    # these exact x coordinates for the paired connectors makes every line
    # terminate on its two subject dots, including when a group contains NaNs.
    values_by_label: dict[str, np.ndarray] = {}
    x_by_label: dict[str, np.ndarray] = {}
    for label in labels:
        values = np.asarray(groups[label], dtype=float)
        finite = np.isfinite(values)
        x_values = np.full(values.shape, np.nan, dtype=float)
        if finite.any():
            x_values[finite] = positions[label] + np.linspace(
                -spread, spread, int(finite.sum())
            )
        values_by_label[label] = values
        x_by_label[label] = x_values

    for label_a, label_b in connect:
        if label_a not in groups or label_b not in groups:
            continue
        values_a = values_by_label[label_a]
        values_b = values_by_label[label_b]
        x_a = x_by_label[label_a]
        x_b = x_by_label[label_b]
        n = min(values_a.size, values_b.size)
        for i in range(n):
            if np.isfinite(values_a[i]) and np.isfinite(values_b[i]):
                ax.plot(
                    [x_a[i], x_b[i]],
                    [values_a[i], values_b[i]],
                    color=style.SUBJECT_LINE, alpha=style.SUBJECT_ALPHA,
                    linewidth=0.8, zorder=1,
                )

    for label in labels:
        raw_values = values_by_label[label]
        finite = np.isfinite(raw_values)
        values = raw_values[finite]
        color = colors.get(label, style.INK)
        x = positions[label]
        ax.scatter(
            x_by_label[label][finite],
            values,
            color=color,
            s=12,
            alpha=0.7,
            zorder=2,
        )
        if estimates is not None:
            if label not in estimates:
                continue
            mean, low, high = (float(value) for value in estimates[label])
            yerr = (
                [[mean - low], [high - mean]]
                if np.isfinite(low) and np.isfinite(high)
                else None
            )
            ax.errorbar(
                [x], [mean], yerr=yerr,
                color=style.INK, marker="o", markersize=5, capsize=3, zorder=3,
            )
        elif values.size:
            mean = float(np.mean(values))
            sem = float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else np.nan
            ci = sem * scipy_stats.t.ppf(0.975, values.size - 1) if values.size > 1 else np.nan
            ax.errorbar(
                [x], [mean], yerr=[ci] if np.isfinite(ci) else None,
                color=style.INK, marker="o", markersize=5, capsize=3, zorder=3,
            )

    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)


def group_line(
    ax: Axes,
    *,
    x: np.ndarray,
    subject_matrix: np.ndarray,  # (n_subjects, n_x), NaN-tolerant
    color: str,
    label: str,
    error: Literal["within_sem", "within_ci95", "none"] = "within_sem",
    draw_subjects: bool = False,
    marker: str | None = "o",
) -> None:
    """Group mean line with a within-subject error band, optional subject traces.

    ``marker`` defaults to a dot per sample, which is right for a handful of
    binned x positions -- it says "these are the measured points, the line
    only joins them". Pass ``None`` for a densely sampled time course, where
    one marker per sample is not information: at a few hundred samples the
    markers merge into a rope several times the line width and the curve
    stops reading as a curve.
    """
    subject_matrix = np.asarray(subject_matrix, dtype=float)
    x = np.asarray(x, dtype=float)

    if draw_subjects:
        for row in subject_matrix:
            ax.plot(x, row, color=style.SUBJECT_LINE, alpha=style.SUBJECT_ALPHA, linewidth=0.7)

    mean = np.nanmean(subject_matrix, axis=0)
    ax.plot(
        x, mean, color=color, label=label, linewidth=1.8,
        marker=marker, markersize=3,
    )

    if error != "none" and subject_matrix.shape[0] > 1:
        kind = "sem" if error == "within_sem" else "ci95"
        band = within_subject_error(subject_matrix, kind=kind)
        ax.fill_between(x, mean - band, mean + band, color=color, alpha=0.20, linewidth=0)


def quantile_function(
    ax: Axes,
    *,
    probabilities: Sequence[float],
    subject_quantiles: Mapping[str, np.ndarray],  # label -> (n_subjects, n_q)
    colors: Mapping[str, str],
) -> None:
    """Vincentized quantile functions: mean quantile on x, probability on y,
    horizontal SEM bars."""
    probabilities = np.asarray(list(probabilities), dtype=float)
    for label, matrix in subject_quantiles.items():
        matrix = np.asarray(matrix, dtype=float)
        mean = np.nanmean(matrix, axis=0)
        n = np.sum(~np.isnan(matrix), axis=0)
        sd = np.nanstd(matrix, axis=0, ddof=1)
        sem = np.divide(sd, np.sqrt(n), out=np.full_like(sd, np.nan), where=n > 0)
        color = colors.get(label, style.INK)
        ax.errorbar(
            mean, probabilities, xerr=sem,
            color=color, marker="o", markersize=4, linewidth=1.6,
            capsize=2, label=label,
        )
    ax.set_ylabel("Quantile probability")


def raincloud(
    ax: Axes,
    *,
    groups: Mapping[str, np.ndarray],
    colors: Mapping[str, str],
    ylabel: str = "",
    connect_subjects: bool = True,
    increase_color: str | None = None,
    decrease_color: str | None = None,
    dashed_mask: np.ndarray | None = None,
) -> None:
    """Half-violin + box + individual points per group, subject connectors.

    By default every connector is `style.SUBJECT_LINE`. Pass both
    `increase_color` and `decrease_color` to color each *segment* (between
    consecutive categories, not the whole subject) by the sign of that
    step -- same idea as `paired_slope`'s direction coloring, just per-leg
    since there are more than two categories here.
    """
    labels = list(groups.keys())
    positions = {label: index for index, label in enumerate(labels)}

    for label in labels:
        values = np.asarray(groups[label], dtype=float)
        values = values[np.isfinite(values)]
        color = colors.get(label, style.INK)
        x = positions[label]
        if values.size >= 2 and np.std(values) > 0:
            violin_parts = ax.violinplot(
                [values], positions=[x + 0.22], widths=0.32, showextrema=False,
            )
            for body in violin_parts["bodies"]:
                # Clip to the right half only, for the half-violin (raincloud) shape.
                path = body.get_paths()[0]
                vertices = path.vertices
                vertices[:, 0] = np.clip(vertices[:, 0], x + 0.22, None)
                body.set_facecolor(color)
                body.set_alpha(0.35)
                body.set_edgecolor("none")
        box = ax.boxplot(
            [values] if values.size else [[np.nan]],
            positions=[x], widths=0.12, patch_artist=True,
            showfliers=False, zorder=2,
        )
        for patch in box["boxes"]:
            patch.set_facecolor(style.SURFACE)
            patch.set_edgecolor(color)
        for element in ("whiskers", "caps", "medians"):
            for line in box[element]:
                line.set_color(color)
        jitter = (
            np.linspace(-0.06, 0.06, values.size) if values.size else np.array([])
        )
        ax.scatter(
            x - 0.16 + jitter * 0, values, color=color, s=10, alpha=0.7, zorder=3
        )

    if connect_subjects and len(labels) > 1:
        arrays = [np.asarray(groups[label], dtype=float) for label in labels]
        n = min(arr.size for arr in arrays) if arrays else 0
        xs = [positions[label] - 0.16 for label in labels]
        direction_coloring = increase_color is not None and decrease_color is not None
        for i in range(n):
            ys = [arr[i] for arr in arrays]
            if not all(np.isfinite(y) for y in ys):
                continue
            if not direction_coloring:
                ax.plot(
                    xs, ys, color=style.SUBJECT_LINE, alpha=style.SUBJECT_ALPHA * 0.7,
                    linewidth=0.6, zorder=1,
                )
                continue
            for leg in range(len(labels) - 1):
                leg_color = increase_color if ys[leg + 1] >= ys[leg] else decrease_color
                ax.plot(
                    xs[leg:leg + 2], ys[leg:leg + 2],
                    color=leg_color, alpha=style.SUBJECT_ALPHA * 0.7,
                    linewidth=0.6, zorder=1,
                )

    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)


def forest(
    ax: Axes,
    *,
    labels: Sequence[str],
    centres: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    colors: Sequence[str] | str = style.INK,
    reference: float | None = None,
    secondary: np.ndarray | None = None,  # e.g. shrunk estimates
) -> None:
    """Horizontal forest/caterpillar plot: one row per label."""
    labels = list(labels)
    centres = np.asarray(centres, dtype=float)
    lows = np.asarray(lows, dtype=float)
    highs = np.asarray(highs, dtype=float)
    y_positions = np.arange(len(labels))
    color_list = colors if isinstance(colors, (list, tuple)) else [colors] * len(labels)

    if reference is not None:
        ax.axvline(reference, color=style.INK_MUTED, linewidth=0.6, zorder=0)

    ax.errorbar(
        centres, y_positions,
        xerr=[centres - lows, highs - centres],
        fmt="o", color=style.INK, markersize=4, linewidth=1.2, capsize=2, zorder=2,
    )
    for y, color in zip(y_positions, color_list):
        ax.plot(centres[y], y, "o", color=color, markersize=4, zorder=3)

    if secondary is not None:
        secondary = np.asarray(secondary, dtype=float)
        for y in y_positions:
            ax.plot(
                [centres[y], secondary[y]], [y, y],
                color=style.INK_MUTED, linewidth=0.7, zorder=1,
            )
            ax.plot(secondary[y], y, "s", color=style.INK_SECONDARY, markersize=3, zorder=3)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()


def correlation_heatmap(
    ax: Axes,
    *,
    r_matrix: pd.DataFrame,
    p_matrix: pd.DataFrame,
    alpha: float = 0.05,
    lower_triangle_only: bool = True,
    include_diagonal: bool = True,
    row_labels: Sequence[str] | None = None,
    column_labels: Sequence[str] | None = None,
    rotate_column_labels: bool = True,
    aspect: str = "equal",
    cell_text: Literal["value", "marker"] = "value",
) -> None:
    """Lower-triangular diverging heatmap of correlation coefficients.

    `include_diagonal` draws the self-correlations. They are structurally 1.0,
    so they carry no information while taking the most saturated colour on the
    ramp -- which also miscalibrates the reader, who anchors "full red" on a
    cell that could never have been anything else and then reads every real
    correlation as weaker than it is. With it off the triangle becomes strict,
    and the first row and last column are then empty by construction, so they
    are dropped: an *n* x *n* matrix draws as an (*n*-1) x (*n*-1) staircase.

    `row_labels`/`column_labels` name the measures for a reader. They default
    to the frame's own index and columns, which are dataframe identifiers --
    fine for a diagnostic, not for a figure.

    `aspect` is passed to ``imshow``. The default keeps cells square, which is
    right for a figure the matrix has to itself. Pass ``"auto"`` where it
    shares a figure and has to fill a box it did not choose: square cells then
    letterbox the panel, shrinking the numbers until they collide while
    leaving the panel half empty. Correlation cells carry no spatial meaning,
    so their proportions are free.

    `cell_text` chooses what each cell prints. ``"value"`` prints r and outlines
    the cells below `alpha`. ``"marker"`` prints the significance marker
    instead and draws no outline, which would then be saying the same thing
    twice; magnitude is left entirely to the ramp, so the colorbar stops being
    decoration and becomes the key. A tested-but-not-significant cell prints
    **nothing** rather than 'n.s.' -- sixteen of those is a wall of text over
    the one panel whose job is to be scanned. It is still visibly *measured*,
    because every drawn cell is filled with its colour; blank means tested and
    under the bar, absent means outside the triangle.
    """
    r_values = r_matrix.to_numpy(dtype=float)
    p_values = p_matrix.to_numpy(dtype=float) if p_matrix is not None else None
    rows = list(row_labels if row_labels is not None else r_matrix.index)
    columns = list(
        column_labels if column_labels is not None else r_matrix.columns
    )

    if lower_triangle_only and not include_diagonal:
        # Strict triangle: drop the row that would hold nothing (the first)
        # and the column that would hold nothing (the last).
        r_values, p_values = r_values[1:, :-1], (
            None if p_values is None else p_values[1:, :-1]
        )
        rows, columns = rows[1:], columns[:-1]

    n_rows, n_columns = r_values.shape
    values = r_values.copy()
    if lower_triangle_only:
        # Row i holds columns 0..i either way. With the diagonal that last
        # column *is* the self-correlation; after the strict trim, row i is
        # original row i+1 against original column i, so the same test holds
        # and no offset is wanted. An offset here empties the first row.
        column_index = np.arange(n_columns)[None, :]
        row_index = np.arange(n_rows)[:, None]
        values = np.where(column_index > row_index, np.nan, values)

    cmap = _diverging_colormap()
    image = ax.imshow(values, cmap=cmap, vmin=-1, vmax=1, aspect=aspect)

    # Hairline separators in the surface colour. With a printed value in every
    # cell and cells only a little wider than the numbers, adjacent entries
    # read as one run of digits without them -- "-.28-.29" rather than two
    # correlations. They cost nothing: the ramp still carries the magnitude.
    for edge in range(n_rows + 1):
        ax.axhline(edge - 0.5, color=style.SURFACE, linewidth=0.8, zorder=3)
    for edge in range(n_columns + 1):
        ax.axvline(edge - 0.5, color=style.SURFACE, linewidth=0.8, zorder=3)

    for i in range(n_rows):
        for j in range(n_columns):
            if not np.isfinite(values[i, j]):
                continue
            r_value = float(r_values[i, j])
            p_value = float(p_values[i, j]) if p_values is not None else np.nan
            text_color = style.SURFACE if abs(r_value) > 0.55 else style.INK
            if cell_text == "marker":
                marker = significance_marker(p_value)
                if marker and marker != "n.s.":
                    ax.text(
                        j, i, marker, ha="center", va="center",
                        color=text_color, fontsize=style.SIGNIFICANCE_PT,
                    )
                continue
            ax.text(
                j, i, format_r(r_value), ha="center", va="center",
                color=text_color, fontsize=style.DENSE_PT,
            )
            if np.isfinite(p_value) and p_value < alpha:
                ax.add_patch(plt_rectangle(j - 0.5, i - 0.5))

    ax.set_xticks(range(n_columns))
    ax.set_xticklabels(
        columns, rotation=45 if rotate_column_labels else 0,
        ha="right" if rotate_column_labels else "center",
    )
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(rows)
    return image


def plt_rectangle(x: float, y: float):
    from matplotlib.patches import Rectangle

    return Rectangle((x, y), 1, 1, fill=False, edgecolor=style.INK, linewidth=1.0)


def _diverging_colormap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("diverging", style.DIVERGING)


def scatter_fit(
    ax: Axes,
    *,
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    color: str = style.INK,
    point_color: str = style.OBSERVED,
) -> None:
    """Subject points with an OLS line and its 95% mean-response band.

    Drawn here rather than delegated to ``reports.meg.plot_correlation``. That
    helper is a seaborn ``regplot``: its points take seaborn's default palette
    and its line ``sns.color_palette("dark")[3]``, so a panel drawn through it
    ignores this module's palette entirely -- and lands on a blue and a red
    close enough to ``CONDITION_COLORS`` to import a Fast/Slow meaning into a
    figure that has no conditions. It also fixes marker size at 50 against the
    rc scale's 4, titles the axes with an ``r`` it computes itself, and made
    this function's own `color` argument dead, since nothing was passed on.

    The line and the band are **display quantities**, not registered
    inferential statistics: callers annotate r and p from a derivative. The
    band is the analytic interval on the fitted mean,
    ``t(.975, n-2) * s * sqrt(1/n + (x - x̄)^2 / Sxx)``, which is deterministic
    -- seaborn's is a bootstrap, so the same data redrew a slightly different
    band on every run.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]

    ax.scatter(
        x, y, s=18, color=point_color, alpha=0.55,
        linewidths=0.0, zorder=2,
    )

    if x.size > 2 and np.ptp(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(x.min(), x.max(), 100)
        fitted = intercept + slope * grid
        residuals = y - (intercept + slope * x)
        # Residual standard error, then the standard error of the fitted mean
        # at each grid point.
        sigma = float(np.sqrt(np.sum(residuals**2) / (x.size - 2)))
        centred = x - x.mean()
        sum_squares = float(np.sum(centred**2))
        standard_error = sigma * np.sqrt(
            1.0 / x.size + (grid - x.mean()) ** 2 / sum_squares
        )
        half_width = scipy_stats.t.ppf(0.975, x.size - 2) * standard_error
        ax.fill_between(
            grid, fitted - half_width, fitted + half_width,
            color=color, alpha=0.12, linewidth=0.0, zorder=1,
        )
        ax.plot(grid, fitted, color=color, linewidth=1.6, zorder=3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
