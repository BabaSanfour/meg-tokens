"""Canonical task-performance analyses over trial features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from meg_tokens.behavior.math.inference import paired_subject_statistics
from meg_tokens.behavior.schema import validate_boolean_values
from meg_tokens.behavior.trials import (
    CLASS_NAMES,
    TASK_CONDITIONS,
    finite_values,
    require_columns,
    task_trials,
)


def compare_fast_slow(features: pd.DataFrame) -> dict:
    """Summarize finite decision times in Fast and Slow task trials.

    Parameters
    ----------
    features
        Canonical trial-feature table. Only rows selected by
        :func:`~meg_tokens.behavior.trials.task_trials` contribute.

    Returns
    -------
    dict
        Finite-trial counts, counts of negative decision times, and arithmetic
        mean decision time for Fast and Slow conditions. Means are in
        milliseconds and are ``NaN`` when a condition has no finite values.

    Notes
    -----
    Negative decision times are retained in both counts and means and reported
    separately as anticipations. This is the declared pipeline policy used for
    the preprint comparison; no outlier exclusion, imputation, or condition
    inference is performed.
    """
    trials = task_trials(features)
    require_columns(trials, ["condition", "dt_ms"])
    condition = trials["condition"].astype(str).str.lower()
    fast = finite_values(trials.loc[condition == "fast", "dt_ms"])
    slow = finite_values(trials.loc[condition == "slow", "dt_ms"])
    return {
        "n_fast": len(fast),
        "n_slow": len(slow),
        "n_fast_anticipations": int((fast < 0).sum()),
        "n_slow_anticipations": int((slow < 0).sum()),
        "mean_fast": float(fast.mean()) if len(fast) else np.nan,
        "mean_slow": float(slow.mean()) if len(slow) else np.nan,
    }


def analyze_trial_classes(features: pd.DataFrame) -> dict:
    """Summarize decision times for the three declared design classes.

    Parameters
    ----------
    features
        Canonical trial-feature table. Only eligible task trials are used.

    Returns
    -------
    dict
        Arrays of finite decision times keyed by ``easy``, ``ambiguous``, and
        ``misleading``, plus nested ``counts`` and arithmetic ``means`` for the
        same classes. Times are in milliseconds; empty-class means are ``NaN``.

    Notes
    -----
    Class labels come directly from the canonical integer ``trial_class``
    field and :data:`~meg_tokens.behavior.trials.CLASS_NAMES`. Unclassified
    trials are excluded rather than assigned to a class. No values are
    imputed or winsorized.
    """
    trials = task_trials(features)
    require_columns(trials, ["trial_class", "dt_ms"])
    values = {
        name: finite_values(trials.loc[trials["trial_class"] == code, "dt_ms"])
        for code, name in CLASS_NAMES.items()
    }
    return {
        **values,
        "counts": {name: len(data) for name, data in values.items()},
        "means": {
            name: float(np.mean(data)) if len(data) else np.nan
            for name, data in values.items()
        },
    }


def compare_correct_error(
    features: pd.DataFrame,
    *,
    condition: str | None = None,
) -> dict:
    """Summarize finite decision times and accuracy by trial outcome.

    Parameters
    ----------
    features
        Canonical trial-feature table. Only eligible task trials contribute.
    condition
        Optional declared task condition, matched case-insensitively. When
        omitted, Fast and Slow trials are pooled.

    Returns
    -------
    dict
        Counts and mean decision times for correct and error trials, plus
        ``percent_correct`` on a 0--100 scale. The denominator is the number of
        trials with both finite decision time and defined correctness. Empty
        means and zero-denominator accuracy are ``NaN``.

    Raises
    ------
    ValueError
        If ``condition`` is not Fast or Slow, or a noncanonical correctness
        value is present.

    Notes
    -----
    Correctness is validated as Boolean and never interpreted through generic
    Python truthiness. No missing outcome is guessed.
    """
    trials = task_trials(features)
    require_columns(trials, ["condition", "dt_ms", "isCorrect"])
    if condition is not None:
        declared_conditions = {name.lower() for name in TASK_CONDITIONS}
        if condition.lower() not in declared_conditions:
            raise ValueError(
                f"condition must be one of {sorted(declared_conditions)}"
            )
        trials = trials.loc[
            trials["condition"].astype(str).str.lower() == condition.lower()
        ]
    validate_boolean_values(trials["isCorrect"], field="isCorrect", optional=True)
    dt = pd.to_numeric(trials["dt_ms"], errors="coerce")
    correctness = trials["isCorrect"].astype("boolean")
    valid = np.isfinite(dt) & correctness.notna()
    valid_dt = dt.loc[valid]
    valid_correctness = correctness.loc[valid].astype(bool)
    correct = valid_dt.loc[valid_correctness].to_numpy(dtype=float)
    error = valid_dt.loc[~valid_correctness].to_numpy(dtype=float)
    n_responses = len(correct) + len(error)
    return {
        "n_correct": len(correct),
        "n_error": len(error),
        "mean_correct": float(correct.mean()) if len(correct) else np.nan,
        "mean_error": float(error.mean()) if len(error) else np.nan,
        "percent_correct": (
            float(len(correct)) / n_responses * 100 if n_responses else np.nan
        ),
    }


def analyze_post_error_slowing(features: pd.DataFrame) -> dict:
    """Summarize decision time following correct and error trials.

    Parameters
    ----------
    features
        Canonical trial-feature table containing run order, started status,
        correctness, and decision time.

    Returns
    -------
    dict
        Counts and arithmetic mean decision time, in milliseconds, for trials
        immediately following a correct or error trial. Empty means are
        ``NaN``.

    Notes
    -----
    Adjacency is defined only within subject, condition, and run after sorting
    by ``run_trial_index``. Only started Fast/Slow rows participate. A pair is
    excluded when the preceding correctness value is missing or the following
    decision time is non-finite. Intervening rows are not skipped to create a
    new adjacency. This is the classical descriptive summary; the extended
    robust sequential analysis lives in ``analyses.sequential``.
    """
    require_columns(
        features,
        [
            "subject",
            "condition",
            "run",
            "run_trial_index",
            "is_started",
            "isCorrect",
            "dt_ms",
        ],
    )
    validate_boolean_values(features["is_started"], field="is_started")
    validate_boolean_values(features["isCorrect"], field="isCorrect", optional=True)
    trials = features.loc[features["is_started"].astype("boolean")].copy()
    trials = trials.loc[
        trials["condition"].astype(str).str.lower().isin({"fast", "slow"})
    ]
    post_correct = []
    post_error = []
    group_columns = ["subject", "condition", "run"]
    for _, run in trials.groupby(group_columns, sort=False):
        run = run.sort_values("run_trial_index")
        for index in range(len(run) - 1):
            is_correct = run.iloc[index]["isCorrect"]
            next_dt = pd.to_numeric(run.iloc[index + 1]["dt_ms"], errors="coerce")
            if pd.isna(is_correct) or not np.isfinite(next_dt):
                continue
            target = post_correct if is_correct else post_error
            target.append(float(next_dt))
    return {
        "n_post_correct": len(post_correct),
        "n_post_error": len(post_error),
        "mean_post_correct": (
            float(np.mean(post_correct)) if post_correct else np.nan
        ),
        "mean_post_error": float(np.mean(post_error)) if post_error else np.nan,
    }


def behavior_group_statistics(summary: pd.DataFrame) -> pd.DataFrame:
    """Compute the fixed preprint-comparison contrasts across subjects.

    Parameters
    ----------
    summary
        Canonical subject-level behavioral summary. Each row represents one
        subject and must contain Fast/Slow decision-time and error measures,
        class decision times, and class logged-SPD means for both declared SPD
        views.

    Returns
    -------
    pandas.DataFrame
        One row for each fixed contrast. Metadata identify the analysis, view,
        unit, labels, and source columns; statistical fields come directly from
        :func:`~meg_tokens.behavior.math.inference.paired_subject_statistics`.

    Notes
    -----
    The contrasts are Fast versus Slow decision time, Fast versus Slow raw
    error count, all three pairwise trial-class decision-time comparisons, and
    all three class comparisons for each logged-SPD view. Pairing is by row in
    the canonical subject summary, and a subject is excluded from a contrast
    when either value is non-finite. Error counts are not converted to rates.
    ``view="primary"`` labels analyses without an SPD sensitivity view; it does
    not trigger alternate selection logic. Contrasts are declared in advance,
    are not chosen from observed results, and receive no multiplicity
    correction.

    Raises
    ------
    ValueError
        If any canonical summary column required by the fixed contrasts is
        absent.
    """
    class_pairs = (
        ("easy", "ambiguous"),
        ("easy", "misleading"),
        ("ambiguous", "misleading"),
    )
    contrasts = [
        {
            "analysis": "decision_time",
            "contrast": "fast_vs_slow",
            "view": "primary",
            "unit": "ms",
            "label_a": "Fast",
            "label_b": "Slow",
            "column_a": "mean_fast_dt_ms",
            "column_b": "mean_slow_dt_ms",
        },
        {
            "analysis": "error_count",
            "contrast": "fast_vs_slow",
            "view": "primary",
            "unit": "trials",
            "label_a": "Fast",
            "label_b": "Slow",
            "column_a": "n_fast_error_trials",
            "column_b": "n_slow_error_trials",
        },
    ]
    for first, second in class_pairs:
        contrasts.append({
            "analysis": "decision_time",
            "contrast": f"{first}_vs_{second}",
            "view": "primary",
            "unit": "ms",
            "label_a": first.capitalize(),
            "label_b": second.capitalize(),
            "column_a": f"mean_{first}_dt_ms",
            "column_b": f"mean_{second}_dt_ms",
        })
        for view in ("all_logged", "validated_15row"):
            contrasts.append({
                "analysis": "spd",
                "contrast": f"{first}_vs_{second}",
                "view": view,
                "unit": "probability",
                "label_a": first.capitalize(),
                "label_b": second.capitalize(),
                "column_a": f"mean_{first}_spd_{view}",
                "column_b": f"mean_{second}_spd_{view}",
            })

    require_columns(
        summary,
        [
            column
            for contrast in contrasts
            for column in (contrast["column_a"], contrast["column_b"])
        ],
    )
    rows = []
    for contrast in contrasts:
        result = paired_subject_statistics(
            summary[contrast["column_a"]],
            summary[contrast["column_b"]],
        )
        rows.append({**contrast, **result})
    return pd.DataFrame(rows)
