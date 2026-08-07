"""Per-subject regression fits and two-stage group inference.

The roadmap's regression analyses (Tier A5, B2-B5, C3-C4) all follow the same
two-stage summary-statistics design: fit one model inside each subject, then
test the resulting coefficients across subjects with a one-sample t-test. For
the balanced within-subject designs used here this is the standard
random-effects equivalent of a mixed model with by-subject random slopes, and
it keeps the package free of a heavyweight modelling dependency. Where a true
hierarchical fit is required -- the sequential-sampling models of Tier C1-C2 --
the roadmap defers to HSSM rather than to this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class RegressionFit:
    """Coefficients and inference for one fitted within-subject model."""

    names: tuple[str, ...]
    coefficients: np.ndarray
    standard_errors: np.ndarray
    statistics: np.ndarray
    p_values: np.ndarray
    n_observations: int
    df_residual: float
    converged: bool = True

    def coefficient(self, name: str) -> float:
        """Return one fitted coefficient by name."""
        return float(self.coefficients[self.names.index(name)])

    def as_dict(self) -> dict[str, float]:
        """Return the fit as a flat ``name -> coefficient`` mapping."""
        return {name: float(value) for name, value in zip(self.names, self.coefficients)}


def _prepare(
    design: np.ndarray | Sequence[Sequence[float]],
    response: np.ndarray | Sequence[float],
    names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    matrix = np.asarray(design, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    values = np.asarray(response, dtype=float)
    if matrix.shape[0] != values.shape[0]:
        raise ValueError("design and response must have the same number of rows")
    if len(names) != matrix.shape[1]:
        raise ValueError("names must label every design column")
    finite = np.isfinite(values) & np.all(np.isfinite(matrix), axis=1)
    return matrix[finite], values[finite], tuple(names)


def fit_linear(
    design: np.ndarray | Sequence[Sequence[float]],
    response: np.ndarray | Sequence[float],
    names: Sequence[str],
) -> RegressionFit:
    """Fit one ordinary least-squares model with t-tests on its coefficients.

    Rows containing a non-finite predictor or response are dropped. A design
    with fewer observations than free parameters, or with a rank-deficient
    predictor matrix, yields NaN inference rather than a silently
    over-determined fit.
    """
    matrix, values, labels = _prepare(design, response, names)
    n_observations, n_parameters = matrix.shape
    rank = int(np.linalg.matrix_rank(matrix)) if n_observations else 0
    empty = np.full(n_parameters, np.nan)
    if n_observations <= n_parameters or rank < n_parameters:
        return RegressionFit(
            names=labels,
            coefficients=empty,
            standard_errors=empty.copy(),
            statistics=empty.copy(),
            p_values=empty.copy(),
            n_observations=n_observations,
            df_residual=float("nan"),
            converged=False,
        )

    coefficients, *_ = np.linalg.lstsq(matrix, values, rcond=None)
    residuals = values - matrix @ coefficients
    df_residual = float(n_observations - n_parameters)
    variance = float(residuals @ residuals) / df_residual
    covariance = variance * np.linalg.pinv(matrix.T @ matrix)
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        statistics = np.where(standard_errors > 0, coefficients / standard_errors, np.nan)
    p_values = 2.0 * stats.t.sf(np.abs(statistics), df_residual)
    return RegressionFit(
        names=labels,
        coefficients=coefficients,
        standard_errors=standard_errors,
        statistics=statistics,
        p_values=p_values,
        n_observations=n_observations,
        df_residual=df_residual,
    )


def fit_logistic(
    design: np.ndarray | Sequence[Sequence[float]],
    response: np.ndarray | Sequence[float],
    names: Sequence[str],
    *,
    max_iterations: int = 50,
    tolerance: float = 1e-8,
) -> RegressionFit:
    """Fit one binary logistic model by iteratively reweighted least squares.

    ``response`` must contain only 0/1 values. Wald z-tests use the inverse
    Fisher information. A design the subject's data separates perfectly has no
    finite maximum-likelihood solution; that case is reported as
    ``converged=False`` with NaN inference instead of arbitrarily large
    coefficients from a truncated iteration.
    """
    matrix, values, labels = _prepare(design, response, names)
    unique = set(np.unique(values).tolist())
    if not unique <= {0.0, 1.0}:
        raise ValueError("logistic response must contain only 0 and 1")
    n_observations, n_parameters = matrix.shape
    empty = np.full(n_parameters, np.nan)
    unconverged = RegressionFit(
        names=labels,
        coefficients=empty,
        standard_errors=empty.copy(),
        statistics=empty.copy(),
        p_values=empty.copy(),
        n_observations=n_observations,
        df_residual=float("nan"),
        converged=False,
    )
    if n_observations <= n_parameters or len(unique) < 2:
        return unconverged
    if int(np.linalg.matrix_rank(matrix)) < n_parameters:
        return unconverged

    coefficients = np.zeros(n_parameters)
    for _ in range(max_iterations):
        linear = matrix @ coefficients
        probability = 1.0 / (1.0 + np.exp(-np.clip(linear, -30.0, 30.0)))
        weights = probability * (1.0 - probability)
        if np.all(weights < 1e-10):
            return unconverged
        information = matrix.T @ (matrix * weights[:, None])
        try:
            step = np.linalg.solve(information, matrix.T @ (values - probability))
        except np.linalg.LinAlgError:
            return unconverged
        coefficients = coefficients + step
        if np.max(np.abs(step)) < tolerance:
            break
    else:
        return unconverged

    if np.max(np.abs(coefficients)) > 1e4:  # separated data
        return unconverged
    linear = matrix @ coefficients
    probability = 1.0 / (1.0 + np.exp(-np.clip(linear, -30.0, 30.0)))
    weights = probability * (1.0 - probability)
    covariance = np.linalg.pinv(matrix.T @ (matrix * weights[:, None]))
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        statistics = np.where(standard_errors > 0, coefficients / standard_errors, np.nan)
    p_values = 2.0 * stats.norm.sf(np.abs(statistics))
    return RegressionFit(
        names=labels,
        coefficients=coefficients,
        standard_errors=standard_errors,
        statistics=statistics,
        p_values=p_values,
        n_observations=n_observations,
        df_residual=float(n_observations - n_parameters),
    )


def one_sample_statistics(values: Sequence[float]) -> dict[str, float]:
    """Summarize subject-level values with a one-sample t-test against zero."""
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
        "cohens_dz": mean / standard_deviation if standard_deviation > 0 else float("nan"),
    }


def _effect_component(data: np.ndarray, kept_axes: tuple[int, ...]) -> np.ndarray:
    """Return the orthogonal ANOVA component belonging to ``kept_axes``.

    The component is the inclusion-exclusion alternating sum of the marginal
    means over every subset of ``kept_axes``, which removes all lower-order
    terms. Summing the components of every subset reconstructs the data, so
    their sums of squares partition the total exactly.
    """
    from itertools import combinations

    component = np.zeros_like(data)
    for size in range(len(kept_axes) + 1):
        for subset in combinations(kept_axes, size):
            averaged = data.mean(
                axis=tuple(axis for axis in range(data.ndim) if axis not in subset),
                keepdims=True,
            )
            component = component + ((-1) ** (len(kept_axes) - size)) * averaged
    return np.broadcast_to(component, data.shape)


def repeated_measures_anova(cells: np.ndarray, factor_levels: Sequence[int]) -> list[dict]:
    """Run a fully within-subject factorial ANOVA on complete cell means.

    ``cells`` is ``(n_subjects, n_cells)`` with the cells ordered as the
    row-major product of ``factor_levels``; subjects with any missing cell are
    dropped so that the design stays balanced. Returns one row per main effect
    and per interaction, each tested against its own ``effect x subject`` error
    term.

    Sums of squares come from the orthogonal component decomposition rather
    than from subtracting cell means: for an interaction the two are not the
    same, because a cell-mean deviation still contains both main effects.
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
        raise ValueError("A repeated-measures ANOVA needs at least two complete subjects")

    from itertools import combinations

    n_factors = len(levels)
    data = complete.reshape((n_subjects, *levels))
    rows = []
    for size in range(1, n_factors + 1):
        for effect in combinations(range(n_factors), size):
            factor_axes = tuple(axis + 1 for axis in effect)
            ss_effect = float(np.sum(_effect_component(data, factor_axes) ** 2))
            ss_error = float(np.sum(_effect_component(data, (0, *factor_axes)) ** 2))
            df_effect = float(np.prod([levels[axis] - 1 for axis in effect]))
            df_error = df_effect * (n_subjects - 1)
            mean_square_error = ss_error / df_error if df_error > 0 else float("nan")
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
                    "effect": "x".join(f"factor{axis + 1}" for axis in effect),
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
