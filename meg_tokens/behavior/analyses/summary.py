"""Assemble the canonical subject-level behavioral summary.

This module contains no independent inferential statistics. It combines the
declared performance summaries with descriptive logged-SPD fields and exposes
the stable one-row-per-subject table consumed by group and extended analyses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from meg_tokens.behavior.analyses.performance import (
    analyze_post_error_slowing,
    analyze_trial_classes,
    compare_correct_error,
    compare_fast_slow,
)
from meg_tokens.behavior.schema import validate_trial_features
from meg_tokens.behavior.trials import (
    CLASS_NAMES,
    TASK_CONDITIONS,
    finite_values,
    require_columns,
    task_trials,
)


def _summarize_logged_spd(features: pd.DataFrame) -> dict[str, dict]:
    """Build the logged-SPD fields used by the canonical subject summary.

    Parameters
    ----------
    features
        Canonical trial-feature rows for one subject. Only eligible task trials
        selected by :func:`~meg_tokens.behavior.trials.task_trials` contribute.

    Returns
    -------
    dict[str, dict]
        Summaries for ``all_logged`` and ``validated_15row``. Each contains the
        finite-trial count, arithmetic mean success probability at decision,
        and the same values for every declared trial class. Empty means are
        ``NaN``.

    Notes
    -----
    The helper summarizes SPD values already computed and validated by the
    feature pipeline. It does not reconstruct probabilities, infer trial
    classes, or substitute the all-logged value when 15-row validation fails.
    """
    trials = task_trials(features)
    views = {
        "all_logged": "logged_spd",
        "validated_15row": "logged_spd_validated_15row",
    }
    require_columns(trials, ["trial_class", *views.values()])

    summaries = {}
    for view, value_column in views.items():
        values = finite_values(trials[value_column])
        classes = {}
        for trial_class, name in CLASS_NAMES.items():
            class_values = finite_values(
                trials.loc[trials["trial_class"] == trial_class, value_column]
            )
            classes[name] = {
                "n_trials": len(class_values),
                "mean_spd": (
                    float(class_values.mean())
                    if class_values.size
                    else float("nan")
                ),
            }
        summaries[view] = {
            "n_trials": len(values),
            "mean_spd": float(values.mean()) if values.size else float("nan"),
            "classes": classes,
        }
    return summaries


def summarize_behavior(features: pd.DataFrame) -> pd.DataFrame:
    """Assemble the canonical subject-level behavioral summary.

    Parameters
    ----------
    features
        Complete canonical trial-feature table produced by
        :func:`~meg_tokens.behavior.features.build_trial_features`. It may
        contain multiple subjects, conditions, and runs, but must contain at
        least one row and a finite motor baseline, constant within each
        response side, per subject.

    Returns
    -------
    pandas.DataFrame
        One row per subject, sorted by subject identifier. Columns report:

        - the motor baseline in milliseconds;
        - started and eligible trial counts;
        - Fast/Slow, correct/error, and trial-class decision-time summaries;
        - accuracy percentages on a 0--100 scale;
        - all-logged and validated-15-row SPD summaries overall and by class;
        - classical post-correct and post-error decision-time summaries.

        Counts use the eligibility rule named by each field. Means are
        arithmetic means and unavailable values are ``NaN``.

    Raises
    ------
    ValueError
        If the canonical feature schema is invalid, no trials are supplied, a
        subject identifier is missing, an unknown condition is present, or a
        subject does not have exactly one finite motor-baseline value.

    Notes
    -----
    ``n_fast_trials`` and ``n_slow_trials`` count all started rows in those
    conditions, including no-response trials. ``n_fast_dt_trials`` and
    ``n_slow_dt_trials`` count only finite decision times among primary-
    analysis-eligible trials. Accuracy uses trials with both finite decision
    time and defined correctness, as documented by
    :func:`~meg_tokens.behavior.analyses.performance.compare_correct_error`.
    Negative decision times remain included and are counted separately as
    anticipations. This function assembles already defined measures; it does
    not infer conditions, classes, correctness, or SPD.
    """
    validate_trial_features(features)
    if features.empty:
        raise ValueError("trial-feature table must contain at least one trial")
    if features["subject"].isna().any():
        raise ValueError("subject must not be missing")
    condition_values = features["condition"].astype(str).str.lower()
    declared_conditions = {
        "rt",
        *[condition.lower() for condition in TASK_CONDITIONS],
    }
    unknown_conditions = sorted(set(condition_values) - declared_conditions)
    if unknown_conditions:
        raise ValueError(f"unknown behavioral conditions: {unknown_conditions}")

    rows = []
    for subject, subject_features in features.groupby("subject", sort=True):
        subject_features = subject_features.copy()
        condition = subject_features["condition"].astype(str).str.lower()
        started = subject_features["is_started"].astype("boolean")
        rt_trials = subject_features.loc[started & (condition == "rt")]
        fast_trials = subject_features.loc[started & (condition == "fast")]
        slow_trials = subject_features.loc[started & (condition == "slow")]

        baseline = pd.to_numeric(
            subject_features["motor_baseline_ms"], errors="raise"
        ).to_numpy(dtype=float)
        # One baseline per response side, so at most two distinct values per
        # subject (one if a hand had no RT-run estimate and fell back to the
        # pooled one). It must still be constant *within* a side: anything
        # else means the correction was applied inconsistently.
        if not np.isfinite(baseline).all():
            raise ValueError(
                f"subject {subject!r} must have a finite motor baseline on every trial"
            )
        sides = pd.to_numeric(subject_features["choice_side"], errors="coerce")
        for side, side_baselines in pd.Series(baseline).groupby(sides.to_numpy()):
            if side_baselines.nunique() != 1:
                raise ValueError(
                    f"subject {subject!r} has more than one motor baseline for "
                    f"response side {int(side)!r}"
                )
        # Reported as the mean over trials, which is the effective latency
        # removed from this subject's decision times.
        reported_baseline = float(np.mean(baseline))

        speed = compare_fast_slow(subject_features)
        accuracy = compare_correct_error(subject_features)
        fast_accuracy = compare_correct_error(subject_features, condition="Fast")
        slow_accuracy = compare_correct_error(subject_features, condition="Slow")
        classes = analyze_trial_classes(subject_features)
        spd = _summarize_logged_spd(subject_features)
        post_error = analyze_post_error_slowing(subject_features)
        all_logged_spd = spd["all_logged"]
        validated_spd = spd["validated_15row"]

        rows.append({
            "subject": subject,
            "motor_baseline_ms": reported_baseline,
            "n_rt_trials": len(rt_trials),
            "n_rt_baseline_trials": len(finite_values(rt_trials["rawRT"])),
            "n_fast_trials": len(fast_trials),
            "n_slow_trials": len(slow_trials),
            "n_never_started_trials": int((~started).sum()),
            "n_fast_dt_trials": speed["n_fast"],
            "n_slow_dt_trials": speed["n_slow"],
            "n_fast_dt_anticipations": speed["n_fast_anticipations"],
            "n_slow_dt_anticipations": speed["n_slow_anticipations"],
            "mean_fast_dt_ms": speed["mean_fast"],
            "mean_slow_dt_ms": speed["mean_slow"],
            "n_correct_trials": accuracy["n_correct"],
            "n_error_trials": accuracy["n_error"],
            "percent_correct": accuracy["percent_correct"],
            "n_fast_correct_trials": fast_accuracy["n_correct"],
            "n_fast_error_trials": fast_accuracy["n_error"],
            "fast_percent_correct": fast_accuracy["percent_correct"],
            "fast_percent_error": 100.0 - fast_accuracy["percent_correct"],
            "n_slow_correct_trials": slow_accuracy["n_correct"],
            "n_slow_error_trials": slow_accuracy["n_error"],
            "slow_percent_correct": slow_accuracy["percent_correct"],
            "slow_percent_error": 100.0 - slow_accuracy["percent_correct"],
            "mean_correct_dt_ms": accuracy["mean_correct"],
            "mean_error_dt_ms": accuracy["mean_error"],
            **{
                f"n_{name}_trials": classes["counts"][name]
                for name in CLASS_NAMES.values()
            },
            **{
                f"mean_{name}_dt_ms": classes["means"][name]
                for name in CLASS_NAMES.values()
            },
            "n_spd_all_logged": all_logged_spd["n_trials"],
            "mean_spd_all_logged": all_logged_spd["mean_spd"],
            "n_spd_validated_15row": validated_spd["n_trials"],
            "mean_spd_validated_15row": validated_spd["mean_spd"],
            **{
                f"n_{name}_spd_all_logged":
                    all_logged_spd["classes"][name]["n_trials"]
                for name in CLASS_NAMES.values()
            },
            **{
                f"mean_{name}_spd_all_logged":
                    all_logged_spd["classes"][name]["mean_spd"]
                for name in CLASS_NAMES.values()
            },
            **{
                f"n_{name}_spd_validated_15row":
                    validated_spd["classes"][name]["n_trials"]
                for name in CLASS_NAMES.values()
            },
            **{
                f"mean_{name}_spd_validated_15row":
                    validated_spd["classes"][name]["mean_spd"]
                for name in CLASS_NAMES.values()
            },
            "n_post_correct_trials": post_error["n_post_correct"],
            "n_post_error_trials": post_error["n_post_error"],
            "mean_post_correct_dt_ms": post_error["mean_post_correct"],
            "mean_post_error_dt_ms": post_error["mean_post_error"],
        })
    return pd.DataFrame(rows)
