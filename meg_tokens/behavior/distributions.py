"""Distributional descriptions of DT and SPD (roadmap Tier A1-A2).

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

from meg_tokens.behavior.metrics import paired_subject_statistics
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
    """Summarize one sample with moments and quantiles."""
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
    """Return per-subject DT distribution statistics for every stratum.

    Strata are the overall task sample, each condition, each trial class, and
    each condition-by-class cell, so that the same table serves the Tier A2
    distributional description and the Tier A3 cell breakdown.
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
    """Contrast distributional statistics across classes and conditions."""
    wide = subject_distributions.pivot(
        index="subject", columns="stratum", values=list(metrics)
    )
    contrasts = [
        ("class", "easy", "ambiguous"),
        ("class", "easy", "misleading"),
        ("class", "ambiguous", "misleading"),
        ("condition", "fast", "slow"),
    ]
    rows = []
    for stratum_type, first, second in contrasts:
        for metric in metrics:
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
                    **paired_subject_statistics(pair, "a", "b"),
                }
            )
    return pd.DataFrame(rows)


def spd_cumulative_distributions(
    features: pd.DataFrame,
    *,
    thresholds: Sequence[float] = DEFAULT_SPD_THRESHOLDS,
) -> pd.DataFrame:
    """Return cumulative SPD distributions by class for both logged views.

    For each threshold the table reports the pooled trial proportion at or
    below it and the mean of the per-subject proportions with its SEM, so the
    curve can be read either way without recomputing it.
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
