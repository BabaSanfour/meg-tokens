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
from meg_tokens.reports.annotations import StatResult, format_stat
from meg_tokens.reports.meg import plot_correlation


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
) -> None:
    """Two x positions, one line per subject, group mean +/- within-subject CI.

    By default every subject line is `subject_line_color`. Pass both
    `increase_color` and `decrease_color` to color each line by the sign of
    b - a instead -- e.g. to show at a glance how many subjects go against
    the group's direction, not just that the group mean moved.
    """
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    valid = np.isfinite(values_a) & np.isfinite(values_b)
    values_a, values_b = values_a[valid], values_b[valid]

    if reference is not None:
        ax.axhline(reference, color=style.INK_MUTED, linewidth=0.6, zorder=0)

    for a, b in zip(values_a, values_b):
        line_color = subject_line_color
        if increase_color is not None and decrease_color is not None:
            line_color = increase_color if b >= a else decrease_color
        ax.plot(
            [0, 1], [a, b],
            color=line_color, alpha=style.SUBJECT_ALPHA,
            linewidth=0.8, zorder=1,
        )
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

    ax.set_xlim(-0.4, 1.4)
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
) -> None:
    """Gardner-Altman difference axis: subject differences, mean, 95% CI, zero rule."""
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    n = differences.size

    ax_diff.axhline(0, color=style.INK_MUTED, linewidth=0.6, zorder=0)
    jitter = np.linspace(-0.06, 0.06, n) if n else np.array([])
    ax_diff.scatter(
        jitter, differences, color=style.SUBJECT_LINE, s=10, alpha=0.6, zorder=1
    )

    if result.mean is not None:
        ci_half = None
        if result.t is not None and result.sem is not None and result.df is not None:
            ci_half = abs(result.mean / result.t) * scipy_stats.t.ppf(
                0.975, result.df
            ) if result.t != 0 else None
        elif result.sem is not None and result.df is not None:
            ci_half = result.sem * scipy_stats.t.ppf(0.975, result.df)
        ax_diff.errorbar(
            [0], [result.mean],
            yerr=[[ci_half], [ci_half]] if ci_half is not None else None,
            color=color, marker="o", markersize=6, linewidth=1.8, capsize=4, zorder=2,
        )

    ax_diff.set_xlim(-0.3, 0.3)
    ax_diff.set_xticks([])
    ax_diff.set_ylabel(f"Δ{(' ' + unit) if unit else ''}")
    ax_diff.text(
        0.5, -0.12, format_stat(result, unit=unit),
        transform=ax_diff.transAxes, ha="center", va="top",
        fontsize=6.5, color=style.INK,
    )


def subject_strip(
    ax: Axes,
    *,
    groups: Mapping[str, np.ndarray],
    colors: Mapping[str, str],
    reference: float | None = 0.0,
    ylabel: str = "",
    connect: Sequence[tuple[str, str]] = (),
) -> None:
    """Jittered per-subject estimates per group, group mean +/- 95% CI,
    optional subject connectors between named group pairs."""
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
                -0.08, 0.08, int(finite.sum())
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
        if values.size:
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
) -> None:
    """Group mean line with a within-subject error band, optional subject traces."""
    subject_matrix = np.asarray(subject_matrix, dtype=float)
    x = np.asarray(x, dtype=float)

    if draw_subjects:
        for row in subject_matrix:
            ax.plot(x, row, color=style.SUBJECT_LINE, alpha=style.SUBJECT_ALPHA, linewidth=0.7)

    mean = np.nanmean(subject_matrix, axis=0)
    ax.plot(x, mean, color=color, label=label, linewidth=1.8, marker="o", markersize=3)

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
) -> None:
    """Lower-triangular diverging heatmap of correlation coefficients."""
    labels = list(r_matrix.columns)
    n = len(labels)
    values = r_matrix.to_numpy(dtype=float).copy()
    if lower_triangle_only:
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        values = np.where(mask, np.nan, values)

    cmap = _diverging_colormap()
    image = ax.imshow(values, cmap=cmap, vmin=-1, vmax=1, aspect="equal")

    for i in range(n):
        for j in range(n):
            if lower_triangle_only and j > i:
                continue
            r_value = r_matrix.iloc[i, j]
            if not np.isfinite(r_value):
                continue
            p_value = p_matrix.iloc[i, j] if p_matrix is not None else np.nan
            text_color = style.SURFACE if abs(r_value) > 0.55 else style.INK
            ax.text(
                j, i, f"{r_value:.2f}", ha="center", va="center",
                color=text_color, fontsize=6,
            )
            if np.isfinite(p_value) and p_value < alpha:
                rectangle = plt_rectangle(j - 0.5, i - 0.5)
                ax.add_patch(rectangle)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
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
) -> None:
    """Styled wrapper over reports.meg.plot_correlation; OLS line + 95% band.

    plot_correlation unconditionally titles the axes with an r it computes
    itself (plain np.corrcoef on whatever x/y it's given) -- an inferential
    statistic computed in the plotting layer, which the caller's own
    properly-sourced annotation (reading r/p/n from a *correlations
    derivative) would otherwise duplicate. Clear it here so there is exactly
    one r on the panel, and it is the sourced one.
    """
    plot_correlation(
        x_data=np.asarray(x, dtype=float),
        y_data=np.asarray(y, dtype=float),
        x_label=xlabel,
        y_label=ylabel,
        title="",
        ax=ax,
    )
    ax.set_title("")
