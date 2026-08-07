"""Pure group-level statistical inference for behavioral analyses.

The functions in this module operate on numeric subject-level estimates or
complete repeated-measures cell matrices. They do not know about behavioral
conditions, DataFrame columns, or planned scientific contrasts; analysis
modules are responsible for constructing and labeling those inputs.
"""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
from scipy import stats


def paired_subject_statistics(
    values_a,
    values_b,
) -> dict:
    """Summarize paired subject values with a t-test and Cohen's dz.

    Parameters
    ----------
    values_a
        Numeric values for the first member of each subject pair.
    values_b
        Numeric values for the second member of each subject pair. Its shape
        and ordering must match ``values_a`` exactly.

    Returns
    -------
    dict
        Number of complete subject pairs, condition means and standard errors,
        mean ``a - b`` difference, paired-samples t statistic, two-sided
        p-value, degrees of freedom, and Cohen's dz. Inferential fields are NaN
        when fewer than two complete pairs are available; Cohen's dz is also
        NaN when the paired differences have zero variance.

    Raises
    ------
    ValueError
        If the input arrays do not have identical shapes.

    Notes
    -----
    Non-finite observations are removed pairwise. The test and effect-size
    direction are always ``values_a - values_b``. Cohen's dz is the mean paired
    difference divided by the sample standard deviation of those differences.
    """
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    if values_a.shape != values_b.shape:
        raise ValueError("Paired values must have identical shapes")
    finite = np.isfinite(values_a) & np.isfinite(values_b)
    values_a = values_a[finite]
    values_b = values_b[finite]
    n_subjects = len(values_a)

    def mean_sem(values: np.ndarray) -> tuple[float, float]:
        if not len(values):
            return np.nan, np.nan
        mean = float(np.mean(values))
        sem = (
            float(np.std(values, ddof=1) / np.sqrt(len(values)))
            if len(values) > 1 else np.nan
        )
        return mean, sem

    mean_a, sem_a = mean_sem(values_a)
    mean_b, sem_b = mean_sem(values_b)
    t_stat = p_value = effect_size = np.nan
    if n_subjects > 1:
        result = stats.ttest_rel(values_a, values_b)
        t_stat = float(result.statistic)
        p_value = float(result.pvalue)
        differences = values_a - values_b
        difference_sd = float(np.std(differences, ddof=1))
        if difference_sd > 0:
            effect_size = float(np.mean(differences) / difference_sd)

    return {
        "n_subjects": n_subjects,
        "mean_a": mean_a,
        "sem_a": sem_a,
        "mean_b": mean_b,
        "sem_b": sem_b,
        "mean_difference": mean_a - mean_b,
        "t": t_stat,
        "p": p_value,
        "df": float(n_subjects - 1) if n_subjects > 1 else np.nan,
        "cohens_dz": effect_size,
    }


def one_sample_statistics(values: Sequence[float]) -> dict[str, float]:
    """Test finite subject-level values against zero.

    Parameters
    ----------
    values
        One scalar estimate per subject. Non-finite values are excluded.

    Returns
    -------
    dict
        Subject count, mean, standard error, one-sample t statistic, two-sided
        p-value, degrees of freedom, and Cohen's dz. Inferential fields are NaN
        when fewer than two finite subjects are available.

    Notes
    -----
    The test is two-sided with a null mean of zero. Cohen's dz is the sample
    mean divided by the sample standard deviation. It is reported as NaN when
    the values have zero variance. No multiple-comparison correction is
    applied here.
    """
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    n_subjects = int(array.size)
    if n_subjects == 0:
        return {
            "n_subjects": 0,
            "mean": float("nan"),
            "sem": float("nan"),
            "t": float("nan"),
            "p": float("nan"),
            "df": float("nan"),
            "cohens_dz": float("nan"),
        }
    mean = float(np.mean(array))
    if n_subjects == 1:
        return {
            "n_subjects": 1,
            "mean": mean,
            "sem": float("nan"),
            "t": float("nan"),
            "p": float("nan"),
            "df": float("nan"),
            "cohens_dz": float("nan"),
        }
    standard_deviation = float(np.std(array, ddof=1))
    result = stats.ttest_1samp(array, 0.0)
    return {
        "n_subjects": n_subjects,
        "mean": mean,
        "sem": standard_deviation / np.sqrt(n_subjects),
        "t": float(result.statistic),
        "p": float(result.pvalue),
        "df": float(n_subjects - 1),
        "cohens_dz": (
            mean / standard_deviation
            if standard_deviation > 0
            else float("nan")
        ),
    }


def _effect_component(data: np.ndarray, kept_axes: tuple[int, ...]) -> np.ndarray:
    """Extract one orthogonal component of a balanced factorial array.

    Parameters
    ----------
    data
        Balanced array with subjects on axis zero and within-subject factors
        on the remaining axes.
    kept_axes
        Axes defining the requested component. Including axis zero produces an
        effect-by-subject error component.

    Returns
    -------
    numpy.ndarray
        The requested component broadcast to the shape of ``data``.

    Notes
    -----
    The component is the inclusion-exclusion alternating sum of marginal means
    over all subsets of ``kept_axes``. This removes every lower-order term;
    the components therefore partition the balanced array's sums of squares.
    """
    component = np.zeros_like(data)
    for size in range(len(kept_axes) + 1):
        for subset in combinations(kept_axes, size):
            averaged = data.mean(
                axis=tuple(
                    axis for axis in range(data.ndim) if axis not in subset
                ),
                keepdims=True,
            )
            component = component + (
                (-1) ** (len(kept_axes) - size)
            ) * averaged
    return np.broadcast_to(component, data.shape)


def repeated_measures_anova(
    cells: np.ndarray,
    factor_levels: Sequence[int],
) -> list[dict]:
    """Run a fully within-subject factorial ANOVA on complete cell means.

    Parameters
    ----------
    cells
        Matrix shaped ``(n_subjects, n_cells)``. Columns must follow the
        row-major Cartesian product of the factor levels.
    factor_levels
        Number of levels in each within-subject factor. Their product must
        equal ``n_cells``. Factor names are positional: the first entry becomes
        ``factor1``, the second ``factor2``, and so forth.

    Returns
    -------
    list of dict
        One result for every main effect and interaction, including sums of
        squares, degrees of freedom, F statistic, p-value, and partial eta
        squared.

    Raises
    ------
    ValueError
        If the input is not a two-dimensional cell matrix, the factor levels
        do not describe its columns, or fewer than two subjects have complete
        data.

    Notes
    -----
    Subjects missing any cell are removed to retain a balanced design. Each
    effect is tested against its own effect-by-subject error component using an
    orthogonal inclusion-exclusion decomposition. The implementation assumes a
    fully within-subject design and applies neither a sphericity correction nor
    a multiple-comparison correction. Analysis code must label the positional
    factors and decide whether either correction is required.
    """
    matrix = np.asarray(cells, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("cells must be a (n_subjects, n_cells) matrix")
    levels = tuple(int(level) for level in factor_levels)
    if int(np.prod(levels)) != matrix.shape[1]:
        raise ValueError("factor_levels must describe every column of cells")
    complete = matrix[np.all(np.isfinite(matrix), axis=1)]
    n_subjects = complete.shape[0]
    if n_subjects < 2:
        raise ValueError(
            "A repeated-measures ANOVA needs at least two complete subjects"
        )

    n_factors = len(levels)
    data = complete.reshape((n_subjects, *levels))
    rows = []
    for size in range(1, n_factors + 1):
        for effect in combinations(range(n_factors), size):
            factor_axes = tuple(axis + 1 for axis in effect)
            ss_effect = float(
                np.sum(_effect_component(data, factor_axes) ** 2)
            )
            ss_error = float(
                np.sum(_effect_component(data, (0, *factor_axes)) ** 2)
            )
            df_effect = float(
                np.prod([levels[axis] - 1 for axis in effect])
            )
            df_error = df_effect * (n_subjects - 1)
            mean_square_error = (
                ss_error / df_error if df_error > 0 else float("nan")
            )
            f_statistic = (
                (ss_effect / df_effect) / mean_square_error
                if df_effect > 0 and mean_square_error > 0
                else float("nan")
            )
            p_value = (
                float(stats.f.sf(f_statistic, df_effect, df_error))
                if np.isfinite(f_statistic)
                else float("nan")
            )
            rows.append(
                {
                    "effect": "x".join(
                        f"factor{axis + 1}" for axis in effect
                    ),
                    "factors": tuple(effect),
                    "n_subjects": n_subjects,
                    "df_effect": df_effect,
                    "df_error": df_error,
                    "sum_squares_effect": ss_effect,
                    "sum_squares_error": ss_error,
                    "F": f_statistic,
                    "p": p_value,
                    "partial_eta_squared": (
                        ss_effect / (ss_effect + ss_error)
                        if (ss_effect + ss_error) > 0
                        else float("nan")
                    ),
                }
            )
    return rows
