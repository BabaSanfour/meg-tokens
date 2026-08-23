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

    # The paired difference needs its own standard error and interval. Only
    # the two condition SEMs were registered before, and neither of them is
    # the uncertainty on the contrast -- they ignore the pairing, which is the
    # whole design. A figure drawing this contrast had no registered interval
    # to read and either reconstructed one from mean/t (the plotting layer
    # computing an inferential statistic, which the reporting rules forbid) or,
    # where that reconstruction was mis-guarded, silently drew none at all.
    sem_difference = np.nan
    ci95_low = ci95_high = np.nan
    bound = bound_dz = np.nan
    if n_subjects > 1:
        differences = values_a - values_b
        sem_difference = float(
            np.std(differences, ddof=1) / np.sqrt(n_subjects)
        )
        if np.isfinite(sem_difference):
            half_width = float(
                stats.t.ppf(0.975, n_subjects - 1) * sem_difference
            )
            ci95_low = float(np.mean(differences) - half_width)
            ci95_high = float(np.mean(differences) + half_width)
        # The equivalence bound travels with the contrast. A figure that
        # argues from a null needs the smallest effect that null excludes, and
        # a bare "n.s." is how a null quietly becomes a claim -- but the bound
        # is an inferential quantity, so the figure layer must not derive it.
        # It is registered here, from the differences themselves, rather than
        # reconstructed downstream from t.
        equivalence = equivalence_bound(differences)
        bound = float(equivalence.get("equivalence_bound", np.nan))
        bound_dz = float(equivalence.get("equivalence_bound_dz", np.nan))

    return {
        "n_subjects": n_subjects,
        "mean_a": mean_a,
        "sem_a": sem_a,
        "mean_b": mean_b,
        "sem_b": sem_b,
        "mean_difference": mean_a - mean_b,
        "sem_difference": sem_difference,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "equivalence_bound": bound,
        "equivalence_bound_dz": bound_dz,
        "equivalence_bound_coverage": 0.90,
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
    standard_error = standard_deviation / np.sqrt(n_subjects)
    # Persisted so a figure drawing these means reads the interval rather than
    # rebuilding it from mean, sem and df -- the same reason the paired
    # contrasts already carry one.
    half_width = float(stats.t.ppf(0.975, n_subjects - 1)) * standard_error
    return {
        "n_subjects": n_subjects,
        "mean": mean,
        "sem": standard_error,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "t": float(result.statistic),
        "p": float(result.pvalue),
        "df": float(n_subjects - 1),
        "cohens_dz": (
            mean / standard_deviation
            if standard_deviation > 0
            else float("nan")
        ),
    }


def equivalence_bound(values, coverage: float = 0.90) -> dict[str, float]:
    """Smallest effect a set of subject-level differences can rule out.

    Parameters
    ----------
    values
        One difference per subject. Non-finite values are excluded.
    coverage
        Two-sided coverage of the interval the bound is read off. Defaults to
        0.90, which is the interval a two-one-sided-tests check at
        ``alpha = .05`` uses.

    Returns
    -------
    dict
        The smallest bound a TOST at ``alpha = 1 - coverage`` passes at, that
        bound in units of the differences' standard deviation, and the coverage
        it was read off. An empty dict when fewer than two finite values are
        supplied, so a caller can splat the result into a row without emitting
        columns it has no value for.

        The interval itself is deliberately *not* returned. A row nearly always
        already carries one, often at a different coverage, and silently
        overwriting it is how a 95% interval ends up labelled 90%. Call
        :func:`confidence_interval` for the interval and this for the bound.

    Notes
    -----
    A TOST passes exactly when the interval lies inside the bound, so the
    smallest bound that passes is the interval's furthest endpoint. Reporting
    that one number states what the data exclude instead of leaving a reader
    to try candidate bounds until one passes -- and it is the only honest
    summary of a null, which ``p`` alone never provides.
    """
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2:
        return {}
    statistics = one_sample_statistics(array)
    low, high = confidence_interval(
        statistics["mean"], statistics["sem"], statistics["df"], coverage=coverage
    )
    bound = max(abs(low), abs(high))
    deviation = float(np.std(array, ddof=1))
    return {
        "equivalence_bound": bound,
        "equivalence_bound_dz": bound / deviation if deviation > 0 else float("nan"),
        "equivalence_bound_coverage": coverage,
    }


def confidence_interval(
    estimate: float,
    standard_error: float,
    df: float,
    coverage: float = 0.95,
) -> tuple[float, float]:
    """Two-sided t interval from an estimate and its standard error.

    Parameters
    ----------
    estimate
        The point estimate.
    standard_error
        Its standard error.
    df
        Degrees of freedom for the t quantile.
    coverage
        Two-sided coverage probability. Defaults to 0.95.

    Returns
    -------
    tuple of float
        Lower and upper bounds, both NaN when any input is missing or ``df``
        is not positive.

    Notes
    -----
    Prefer this to :func:`confidence_interval_from_t` whenever the standard
    error is available: that function has to divide by the t statistic and so
    yields nothing when the estimate is exactly zero, which is precisely the
    case a bound is most often wanted for. A zero standard error gives a
    degenerate interval equal to the estimate, which is correct.
    """
    if not all(np.isfinite([estimate, standard_error, df])) or df <= 0:
        return float("nan"), float("nan")
    half_width = float(stats.t.ppf(0.5 + coverage / 2, df)) * abs(standard_error)
    return estimate - half_width, estimate + half_width


def confidence_interval_from_t(
    estimate: float,
    t_statistic: float,
    df: float,
    coverage: float = 0.95,
) -> tuple[float, float]:
    """Recover an estimate's confidence interval from its own t statistic.

    Parameters
    ----------
    estimate
        The point estimate the t statistic was computed for, such as a mean
        paired difference.
    t_statistic
        The t statistic testing ``estimate`` against zero.
    df
        Degrees of freedom of that test.
    coverage
        Two-sided coverage probability. Defaults to 0.95.

    Returns
    -------
    tuple of float
        Lower and upper bounds, both NaN when the interval is not recoverable.

    Notes
    -----
    ``estimate / t`` is the standard error the test itself used, so this
    reconstructs the interval exactly rather than re-deriving it from
    condition-level standard errors -- which is not possible for a paired
    difference, whose error term depends on the within-subject covariance. The
    interval is undefined when the t statistic is zero, since the standard
    error is then unrecoverable; that case returns NaN rather than an
    arbitrarily wide interval.
    """
    if not all(np.isfinite([estimate, t_statistic, df])) or df <= 0:
        return float("nan"), float("nan")
    if t_statistic == 0:
        return float("nan"), float("nan")
    standard_error = abs(estimate / t_statistic)
    half_width = float(stats.t.ppf(0.5 + coverage / 2, df)) * standard_error
    return estimate - half_width, estimate + half_width


def equivalence_bound_from_summary(
    estimate: float,
    df: float,
    *,
    standard_error: float | None = None,
    t_statistic: float | None = None,
    coverage: float = 0.90,
) -> float:
    """The equivalence bound for a contrast already summarized in a row.

    Parameters
    ----------
    estimate
        The mean difference.
    df
        Degrees of freedom of the test on it.
    standard_error
        Its standard error, when the row carries one.
    t_statistic
        The t statistic testing it against zero. Used when no standard error
        is given -- which is the common case for a *paired* contrast, whose
        rows carry ``sem_a`` and ``sem_b`` per condition but no paired
        ``sem``: that error term depends on the within-subject covariance and
        is not recoverable from the two condition standard errors. It is
        recoverable from the t statistic the test itself reported.
    coverage
        Two-sided coverage the bound is read off. Defaults to 0.90, the
        interval a TOST at ``alpha = .05`` uses.

    Returns
    -------
    float
        The smallest effect the contrast rules out, or NaN when the interval
        is not recoverable.

    Notes
    -----
    Same rule as :func:`equivalence_bound` -- the furthest endpoint of the
    interval -- for the case where the subject-level differences are no longer
    at hand but a persisted row summarizes them. A figure arguing from a null
    needs the bound and usually has only the row; without this it either
    re-derives the rule inline or prints "n.s.", and both are how a null
    quietly becomes a claim. The standardized companion is not returned,
    because the differences' own standard deviation is not recoverable from a
    summary row.
    """
    if standard_error is not None and np.isfinite(standard_error):
        low, high = confidence_interval(
            estimate, standard_error, df, coverage=coverage
        )
    elif t_statistic is not None:
        low, high = confidence_interval_from_t(
            estimate, t_statistic, df, coverage=coverage
        )
    else:
        return float("nan")
    if not np.isfinite(low) or not np.isfinite(high):
        return float("nan")
    return max(abs(low), abs(high))


def subject_bootstrap(
    statistic,
    n_subjects: int,
    *,
    n_resamples: int = 20000,
    seed: int = 0,
    coverage: float = 0.95,
    bound_coverage: float = 0.90,
) -> dict[str, float]:
    """Resample subjects to interval-estimate a statistic that is not a mean.

    Parameters
    ----------
    statistic
        Callable taking an integer index array whose **last** axis holds
        ``n_subjects`` resampled subject positions, and reducing over that axis
        to return one value per leading element. Written against
        ``numpy``'s ``axis=-1`` reductions it is the same expression for one
        resample or for all of them, and every replicate is computed in a
        single call rather than in a Python loop.
    n_subjects
        Number of subjects to resample from, with replacement.
    n_resamples
        Bootstrap replicates. The default resolves the interval endpoints to
        about a thousandth on the scales used here.
    seed
        Seed for the resampling generator. Fixed by default, because a
        derivative that changes on re-run is not a derivative.
    coverage
        Two-sided coverage of the reported percentile interval.
    bound_coverage
        Two-sided coverage the equivalence bound is read off, matching
        :func:`equivalence_bound`.

    Returns
    -------
    dict
        The point estimate, its percentile interval and that interval's
        coverage, the bootstrap achieved significance level against zero, the
        equivalence bound and its own coverage, the subject count, and the
        resampling settings. Inferential fields are NaN when fewer than two
        subjects are supplied or the statistic never returns a finite value.

    Notes
    -----
    Use this for a quantity the subject-level t machinery cannot reach: a ratio
    of subject means, or a difference between two such ratios. Those have no
    per-subject value to average -- a per-subject ratio explodes wherever a
    subject's denominator approaches zero, which is exactly where the group
    ratio is best determined -- so the sampling unit has to be resampled
    instead of the estimate.

    The same index array drives every statistic a caller composes, so two
    quantities computed from one set of subjects stay **paired** across
    replicates. That is what makes a difference between them testable at all:
    unpaired, the two sampling errors add instead of cancelling.

    ``p`` is the percentile achieved significance level,
    ``2 * min(P(theta* <= 0), P(theta* >= 0))``, counted with the standard
    ``+1`` correction so that it is bounded below by ``2 / (n_resamples + 1)``
    -- it can say "smaller than the resolution of this bootstrap", never
    "zero". No bias correction or acceleration is applied, so a heavily skewed
    statistic's interval is the percentile one and nothing better.
    """
    if n_subjects < 2:
        return {
            "n_subjects": int(max(n_subjects, 0)),
            "estimate": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "coverage": coverage,
            "p": float("nan"),
            "equivalence_bound": float("nan"),
            "equivalence_bound_coverage": bound_coverage,
            "n_resamples": int(n_resamples),
            "bootstrap_seed": int(seed),
        }
    estimate = float(np.asarray(statistic(np.arange(n_subjects))).reshape(()))
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, n_subjects, size=(n_resamples, n_subjects))
    replicates = np.asarray(statistic(indices), dtype=float).ravel()
    replicates = replicates[np.isfinite(replicates)]
    if replicates.size < 2:
        return {
            "n_subjects": int(n_subjects),
            "estimate": estimate,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "coverage": coverage,
            "p": float("nan"),
            "equivalence_bound": float("nan"),
            "equivalence_bound_coverage": bound_coverage,
            "n_resamples": int(n_resamples),
            "bootstrap_seed": int(seed),
        }
    low, high = _percentile_interval(replicates, coverage)
    bound_low, bound_high = _percentile_interval(replicates, bound_coverage)
    # The +1 on both counts is the standard correction: a replicate count of
    # zero would otherwise print p = 0, which asserts more than any finite
    # number of resamples can support.
    total = replicates.size + 1
    below = (float(np.sum(replicates <= 0.0)) + 1.0) / total
    above = (float(np.sum(replicates >= 0.0)) + 1.0) / total
    return {
        "n_subjects": int(n_subjects),
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "coverage": coverage,
        "p": min(1.0, 2.0 * min(below, above)),
        "equivalence_bound": max(abs(bound_low), abs(bound_high)),
        "equivalence_bound_coverage": bound_coverage,
        "n_resamples": int(replicates.size),
        "bootstrap_seed": int(seed),
    }


def _percentile_interval(
    replicates: np.ndarray, coverage: float
) -> tuple[float, float]:
    """Two-sided percentile interval of a bootstrap distribution."""
    tail = (1.0 - coverage) / 2.0
    low, high = np.percentile(replicates, [100.0 * tail, 100.0 * (1.0 - tail)])
    return float(low), float(high)


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
