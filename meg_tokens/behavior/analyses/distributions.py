"""Distributional analyses of decision time and success probability.

Means alone hide the right tail that distinguishes a threshold change from a
drift change, so these helpers report subject-level quantiles, skewness, and
optional ex-Gaussian parameters for decision time, and cumulative
distributions for logged success probability at decision.
"""

from __future__ import annotations

from typing import Final, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from meg_tokens.behavior.math.inference import paired_subject_statistics
from meg_tokens.behavior.trials import (
    CLASS_NAMES,
    TASK_CONDITIONS,
    finite_values,
    task_trials,
)


DEFAULT_QUANTILES: Final[tuple[float, ...]] = (0.1, 0.25, 0.5, 0.75, 0.9)
DEFAULT_SPD_THRESHOLDS: Final[tuple[float, ...]] = tuple(
    round(value, 2) for value in np.arange(0.0, 1.0001, 0.05)
)
SPD_VIEWS: Final[dict[str, str]] = {
    "all_logged": "logged_spd",
    "validated_15row": "logged_spd_validated_15row",
}

# Fewer trials than this cannot support a stable third or fourth moment, let
# alone a three-parameter ex-Gaussian fit; those statistics are reported as
# NaN rather than as noise with a number attached.
MIN_TRIALS_FOR_SHAPE: Final[int] = 20


def distribution_summary(
    values: Sequence[float],
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> dict[str, float]:
    """Summarize a finite sample with descriptive moments and quantiles.

    Non-finite observations are discarded before any statistic is computed.
    The standard deviation uses the sample convention (``ddof=1``), and
    SciPy computes bias-corrected skewness and excess kurtosis.  Shape
    statistics are withheld for samples smaller than
    :data:`MIN_TRIALS_FOR_SHAPE`; this cutoff is an analysis stability policy,
    not a value inferred from the observations.

    Parameters
    ----------
    values
        Numeric observations in the unit to be summarized.  Decision-time
        callers pass milliseconds.
    quantiles
        Quantile probabilities in the closed interval ``[0, 1]``.  Each
        output key is formed by rounding the probability to an integer
        percentage, for example ``0.25`` becomes ``"q25"``.  NumPy raises
        ``ValueError`` for probabilities outside the valid interval.

    Returns
    -------
    dict[str, float]
        ``n_trials``, ``mean``, sample ``sd``, ``min``, ``max``, the requested
        quantiles, bias-corrected ``skewness``, and excess ``kurtosis``.
        Statistics without enough finite observations are ``NaN``.

    Notes
    -----
    This function does not impute values, remove outliers, fit a model, or
    infer a distribution family.
    """
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    summary: dict[str, float] = {
        "n_trials": int(array.size),
        "mean": float(np.mean(array)) if array.size else float("nan"),
        "sd": float(np.std(array, ddof=1)) if array.size > 1 else float("nan"),
        "min": float(np.min(array)) if array.size else float("nan"),
        "max": float(np.max(array)) if array.size else float("nan"),
    }
    for quantile in quantiles:
        summary[f"q{int(round(quantile * 100)):02d}"] = (
            float(np.quantile(array, quantile)) if array.size else float("nan")
        )
    has_shape = array.size >= MIN_TRIALS_FOR_SHAPE
    summary["skewness"] = float(stats.skew(array, bias=False)) if has_shape else float("nan")
    summary["kurtosis"] = (
        float(stats.kurtosis(array, bias=False)) if has_shape else float("nan")
    )
    return summary


def ex_gaussian_parameters(values: Sequence[float]) -> dict[str, float]:
    """Fit the ex-Gaussian ``mu``, ``sigma``, ``tau`` decomposition.

    The ex-Gaussian is the convolution of a Gaussian (``mu``, ``sigma``) with
    an exponential (``tau``); ``tau`` isolates the right tail that a mean
    cannot separate from a shift in the body of the distribution. Fitting uses
    ``scipy.stats.exponnorm``, whose shape ``K`` equals ``tau / sigma``.

    Parameters
    ----------
    values
        Numeric observations in the unit of the desired parameters.  For
        decision times, ``mu``, ``sigma``, and ``tau`` are therefore all in
        milliseconds.  Non-finite observations are discarded.

    Returns
    -------
    dict[str, float]
        Maximum-likelihood estimates named ``exgaussian_mu``,
        ``exgaussian_sigma``, and ``exgaussian_tau``, plus the Boolean
        ``exgaussian_fitted`` flag.  Failed or intentionally skipped fits
        contain ``NaN`` parameters and ``False``.

    Notes
    -----
    The fit is skipped when fewer than :data:`MIN_TRIALS_FOR_SHAPE` finite
    observations remain or when their standard deviation is zero.  The trial
    cutoff is an explicit stability policy, not a fitted or preprint-derived
    threshold.  SciPy performs the optimization directly; this function does
    not implement a custom estimator or silently substitute another model.
    """
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size < MIN_TRIALS_FOR_SHAPE or float(np.std(array)) <= 0:
        return {
            "exgaussian_mu": float("nan"),
            "exgaussian_sigma": float("nan"),
            "exgaussian_tau": float("nan"),
            "exgaussian_fitted": False,
        }
    try:
        shape, location, scale = stats.exponnorm.fit(array)
    except Exception:  # pragma: no cover - scipy raises several optimizer errors
        return {
            "exgaussian_mu": float("nan"),
            "exgaussian_sigma": float("nan"),
            "exgaussian_tau": float("nan"),
            "exgaussian_fitted": False,
        }
    return {
        "exgaussian_mu": float(location),
        "exgaussian_sigma": float(scale),
        "exgaussian_tau": float(shape * scale),
        "exgaussian_fitted": True,
    }


def _strata(trials: pd.DataFrame) -> list[tuple[str, str, pd.Series]]:
    """Build the fixed masks used for decision-time summaries.

    Parameters
    ----------
    trials
        Task-trial rows for one subject.  The frame must contain ``condition``
        and integer ``trial_class`` columns.

    Returns
    -------
    list[tuple[str, str, pandas.Series]]
        ``(stratum_type, stratum_name, mask)`` tuples for the overall sample,
        every task condition, every trial class, and every
        condition-by-class cell.  Masks retain the input frame's index.

    Notes
    -----
    The function only constructs declared categories from
    :data:`TASK_CONDITIONS` and :data:`CLASS_NAMES`; it does not infer or
    relabel conditions from the data.
    """
    condition = trials["condition"].astype(str).str.lower()
    strata: list[tuple[str, str, pd.Series]] = [
        ("overall", "all_task_trials", pd.Series(True, index=trials.index))
    ]
    for name in TASK_CONDITIONS:
        strata.append(("condition", name.lower(), condition == name.lower()))
    for code, name in CLASS_NAMES.items():
        strata.append(("class", name, trials["trial_class"] == code))
    for name in TASK_CONDITIONS:
        for code, class_name in CLASS_NAMES.items():
            strata.append(
                (
                    "condition_class",
                    f"{name.lower()}_{class_name}",
                    (condition == name.lower()) & (trials["trial_class"] == code),
                )
            )
    return strata


def decision_time_distributions(
    features: pd.DataFrame,
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    fit_ex_gaussian: bool = True,
) -> pd.DataFrame:
    """Compute subject-level decision-time distributions by task stratum.

    Strata are the overall task sample, each condition, each trial class, and
    each condition-by-class cell, so one table serves both distributional and
    factorial descriptions.

    Parameters
    ----------
    features
        Canonical trial-feature table.  It must contain the columns required
        by :func:`~meg_tokens.behavior.trials.task_trials`, plus ``subject``,
        ``condition``, ``trial_class``, and ``dt_ms``.
    quantiles
        Quantile probabilities forwarded unchanged to
        :func:`distribution_summary`.
    fit_ex_gaussian
        Whether to append ex-Gaussian parameters for each subject and
        stratum.  When ``False``, the ex-Gaussian columns are absent rather
        than filled with ``NaN``.

    Returns
    -------
    pandas.DataFrame
        One row per subject and fixed stratum.  Identifier columns are
        ``subject``, ``stratum_type``, and ``stratum``; remaining columns are
        the outputs of :func:`distribution_summary` and, when requested,
        :func:`ex_gaussian_parameters`.  Decision-time values and fitted
        parameters are in milliseconds.

    Notes
    -----
    Non-task rows are removed by :func:`task_trials`.  Empty declared cells
    are retained with ``n_trials == 0`` and ``NaN`` statistics.  No trial is
    reassigned, imputed, winsorized, or pooled across subjects.
    """
    trials = task_trials(features)
    rows = []
    for subject, subject_trials in trials.groupby("subject", sort=True):
        for stratum_type, stratum, selection in _strata(subject_trials):
            values = finite_values(subject_trials.loc[selection, "dt_ms"])
            row = {
                "subject": subject,
                "stratum_type": stratum_type,
                "stratum": stratum,
                **distribution_summary(values, quantiles=quantiles),
            }
            if fit_ex_gaussian:
                row.update(ex_gaussian_parameters(values))
            rows.append(row)
    return pd.DataFrame(rows)


def decision_time_distribution_statistics(
    subject_distributions: pd.DataFrame,
    *,
    metrics: Sequence[str] = ("q10", "q50", "q90", "skewness", "exgaussian_tau"),
) -> pd.DataFrame:
    """Run fixed paired contrasts on subject-level distribution metrics.

    Parameters
    ----------
    subject_distributions
        Output of :func:`decision_time_distributions`, with one unique row per
        ``subject`` and ``stratum`` and columns for every requested metric.
    metrics
        Distribution-summary columns to test.  The defaults compare the 10th,
        50th, and 90th percentiles, skewness, and ex-Gaussian ``tau``.
        Requested metrics absent from ``subject_distributions`` are skipped,
        because ``exgaussian_tau`` exists only when the caller asked
        :func:`decision_time_distributions` to fit it.

    Returns
    -------
    pandas.DataFrame
        One row per available metric and fixed contrast, containing labels
        for the comparison and the fields returned by
        :func:`~meg_tokens.behavior.math.inference.paired_subject_statistics`.

    Notes
    -----
    The declared contrasts are easy versus ambiguous, easy versus misleading,
    ambiguous versus misleading, and fast versus slow.  Pairing is by subject;
    the shared inference helper removes incomplete pairs.  A contrast is
    omitted when its metric or either of its strata is absent.  The function
    performs no multiplicity correction and does not choose contrasts from the
    observed results.  Duplicate subject/stratum rows cause pandas ``pivot`` to
    raise rather than being silently aggregated.
    """
    available = [
        metric for metric in metrics if metric in subject_distributions.columns
    ]
    if not available:
        return pd.DataFrame()
    wide = subject_distributions.pivot(
        index="subject", columns="stratum", values=available
    )
    contrasts = [
        ("class", "easy", "ambiguous"),
        ("class", "easy", "misleading"),
        ("class", "ambiguous", "misleading"),
        ("condition", "fast", "slow"),
    ]
    rows = []
    for stratum_type, first, second in contrasts:
        for metric in available:
            if (metric, first) not in wide.columns or (metric, second) not in wide.columns:
                continue
            pair = wide[[(metric, first), (metric, second)]].copy()
            pair.columns = ["a", "b"]
            rows.append(
                {
                    "analysis": "decision_time_distribution",
                    "metric": metric,
                    "stratum_type": stratum_type,
                    "contrast": f"{first}_vs_{second}",
                    "label_a": first,
                    "label_b": second,
                    **paired_subject_statistics(pair["a"], pair["b"]),
                }
            )
    return pd.DataFrame(rows)


def spd_cumulative_distributions(
    features: pd.DataFrame,
    *,
    thresholds: Sequence[float] = DEFAULT_SPD_THRESHOLDS,
) -> pd.DataFrame:
    """Compute empirical cumulative logged-SPD curves by trial class.

    For each threshold the table reports the pooled trial proportion at or
    below it and the mean of the per-subject proportions with its SEM, so the
    curve can be read either way without recomputing it.

    Parameters
    ----------
    features
        Canonical trial-feature table containing task-trial markers,
        ``subject``, ``trial_class``, ``logged_spd``, and
        ``logged_spd_validated_15row``.
    thresholds
        SPD cut points at which to evaluate ``P(logged_spd <= threshold)``.
        Values are used as supplied and are not sorted, clipped, or inferred
        from the data.

    Returns
    -------
    pandas.DataFrame
        One row per logged-SPD view, declared trial class, and threshold.
        ``pooled_proportion`` weights every finite trial equally.
        ``mean_subject_proportion`` gives each subject with at least one finite
        value equal weight, and ``sem_subject_proportion`` is the sample
        standard deviation of those proportions divided by the square root of
        their count.  ``n_trials`` and ``n_subjects`` report the corresponding
        denominators.

    Notes
    -----
    Non-task rows and non-finite SPD values are excluded.  The two views are
    analyzed separately: ``all_logged`` uses every finite logged value, while
    ``validated_15row`` uses only values already validated by the feature
    pipeline.  This function does not reconstruct, validate, or guess SPD.
    It also does not perform hypothesis tests; it returns descriptive
    empirical cumulative distributions only.
    """
    trials = task_trials(features)
    rows = []
    for view, column in SPD_VIEWS.items():
        for code, class_name in CLASS_NAMES.items():
            selected = trials.loc[trials["trial_class"] == code]
            pooled = finite_values(selected[column])
            per_subject = {
                subject: finite_values(group[column])
                for subject, group in selected.groupby("subject", sort=True)
            }
            for threshold in thresholds:
                proportions = [
                    float(np.mean(values <= threshold))
                    for values in per_subject.values()
                    if values.size
                ]
                array = np.asarray(proportions, dtype=float)
                rows.append(
                    {
                        "analysis": "spd_cumulative_distribution",
                        "view": view,
                        "trial_class": code,
                        "trial_class_name": class_name,
                        "threshold": float(threshold),
                        "n_trials": int(pooled.size),
                        "n_subjects": int(array.size),
                        "pooled_proportion": (
                            float(np.mean(pooled <= threshold))
                            if pooled.size
                            else float("nan")
                        ),
                        "mean_subject_proportion": (
                            float(np.mean(array)) if array.size else float("nan")
                        ),
                        "sem_subject_proportion": (
                            float(np.std(array, ddof=1) / np.sqrt(array.size))
                            if array.size > 1
                            else float("nan")
                        ),
                    }
                )
    return pd.DataFrame(rows)
