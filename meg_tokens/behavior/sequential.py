"""Trial-to-trial history effects (roadmap Tier B6-B7).

The summary post-error measure compares every post-error trial with every
post-correct trial, which confounds the effect with slow stretches of a
session. Tier B6 instead compares each post-error trial with the trial
immediately *preceding* its error, so both sides of the contrast come from the
same moment in the session. Tier B7 adds the choice-history effects that the
same adjacency makes available: win-stay/lose-shift, side autocorrelation, and
the influence of the previous trial's class and outcome on the current DT.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from meg_tokens.behavior.metrics import paired_subject_statistics
from meg_tokens.behavior.regression import one_sample_statistics
from meg_tokens.behavior.trials import (
    CLASS_NAMES,
    TASK_CONDITIONS,
    require_columns,
)


def _ordered_task_runs(features: pd.DataFrame) -> pd.DataFrame:
    """Return started task trials with neighbours attached inside each run.

    Adjacency is defined on ``run_trial_index`` within one run, and a pair is
    kept only when the two indices are genuinely consecutive, so a never-started
    row between them breaks the chain instead of silently joining trials that
    were minutes apart.
    """
    require_columns(
        features,
        [
            "subject",
            "condition",
            "run",
            "run_trial_index",
            "dt_ms",
            "isCorrect",
            "choice_side",
            "trial_class",
            "is_started",
        ],
    )
    condition = features["condition"].astype(str).str.lower()
    trials = features.loc[
        condition.isin({name.lower() for name in TASK_CONDITIONS})
        & features["is_started"].astype(bool)
    ].copy()
    trials["condition"] = condition.loc[trials.index]
    trials["is_correct"] = trials["isCorrect"].map(_as_optional_bool)
    trials["dt"] = pd.to_numeric(trials["dt_ms"], errors="coerce")
    trials["side"] = pd.to_numeric(trials["choice_side"], errors="coerce")
    trials = trials.sort_values(["subject", "condition", "run", "run_trial_index"])

    grouped = trials.groupby(["subject", "condition", "run"], sort=False)
    position = trials["run_trial_index"].astype(float)
    for name, shift in (("previous", 1), ("next", -1)):
        adjacent = grouped.shift(shift)
        gap = position - grouped["run_trial_index"].shift(shift).astype(float)
        contiguous = gap.abs() == 1
        trials[f"{name}_dt"] = adjacent["dt"].where(contiguous)
        trials[f"{name}_is_correct"] = adjacent["is_correct"].where(contiguous)
        trials[f"{name}_side"] = adjacent["side"].where(contiguous)
        trials[f"{name}_trial_class"] = adjacent["trial_class"].where(contiguous)
    return trials


def robust_post_error_slowing(features: pd.DataFrame) -> pd.DataFrame:
    """Compare each post-error trial with the trial preceding its error.

    ``robust_pes_ms`` is ``DT(error + 1) - DT(error - 1)``. The classical
    contrast (mean post-error minus mean post-correct DT) is reported beside it
    from the same trials so the two definitions can be compared directly.
    """
    trials = _ordered_task_runs(features)
    is_error = trials["is_correct"] == False  # noqa: E712 - keeps NaN out of the mask
    rows = []
    for subject, subject_trials in trials.groupby("subject", sort=True):
        subject_errors = is_error.loc[subject_trials.index]
        for name in ("all", *[item.lower() for item in TASK_CONDITIONS]):
            selection = (
                pd.Series(True, index=subject_trials.index)
                if name == "all"
                else subject_trials["condition"] == name
            )
            errors = subject_trials.loc[selection & subject_errors]
            paired = errors.loc[errors["previous_dt"].notna() & errors["next_dt"].notna()]
            differences = (paired["next_dt"] - paired["previous_dt"]).to_numpy(dtype=float)

            following = subject_trials.loc[selection & subject_trials["previous_dt"].notna()]
            post_error = following.loc[following["previous_is_correct"] == False]  # noqa: E712
            post_correct = following.loc[following["previous_is_correct"] == True]  # noqa: E712
            rows.append(
                {
                    "subject": subject,
                    "condition": name,
                    "n_error_pairs": int(differences.size),
                    "robust_pes_ms": (
                        float(np.mean(differences)) if differences.size else float("nan")
                    ),
                    "mean_pre_error_dt_ms": (
                        float(paired["previous_dt"].mean()) if len(paired) else float("nan")
                    ),
                    "mean_post_error_dt_ms": (
                        float(paired["next_dt"].mean()) if len(paired) else float("nan")
                    ),
                    "n_post_error_trials": int(post_error["dt"].notna().sum()),
                    "n_post_correct_trials": int(post_correct["dt"].notna().sum()),
                    "classical_pes_ms": (
                        float(post_error["dt"].mean() - post_correct["dt"].mean())
                        if post_error["dt"].notna().any() and post_correct["dt"].notna().any()
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def post_error_statistics(subject_table: pd.DataFrame) -> pd.DataFrame:
    """Test both post-error definitions against zero and across conditions."""
    rows = []
    for measure in ("robust_pes_ms", "classical_pes_ms"):
        for condition, group in subject_table.groupby("condition", sort=True):
            rows.append(
                {
                    "analysis": "post_error_slowing",
                    "measure": measure,
                    "condition": condition,
                    "test": "one_sample_vs_zero",
                    **one_sample_statistics(group[measure]),
                }
            )
        wide = subject_table.pivot(index="subject", columns="condition", values=measure)
        if "fast" in wide.columns and "slow" in wide.columns:
            rows.append(
                {
                    "analysis": "post_error_slowing",
                    "measure": measure,
                    "condition": "fast_vs_slow",
                    "test": "paired_t_test",
                    **paired_subject_statistics(wide, "fast", "slow"),
                }
            )
    return pd.DataFrame(rows)


def choice_history(features: pd.DataFrame) -> pd.DataFrame:
    """Return win-stay/lose-shift, side autocorrelation, and previous-trial DT."""
    trials = _ordered_task_runs(features)
    rows = []
    for subject, subject_trials in trials.groupby("subject", sort=True):
        for name in ("all", *[item.lower() for item in TASK_CONDITIONS]):
            selected = (
                subject_trials
                if name == "all"
                else subject_trials.loc[subject_trials["condition"] == name]
            )
            linked = selected.loc[
                selected["previous_side"].notna() & selected["side"].notna()
            ]
            if linked.empty:
                continue
            repeated = (linked["side"] == linked["previous_side"]).astype(float)
            after_correct = linked["previous_is_correct"] == True  # noqa: E712
            after_error = linked["previous_is_correct"] == False  # noqa: E712
            side_now = np.where(linked["side"].to_numpy(dtype=float) == 1, 1.0, -1.0)
            side_previous = np.where(
                linked["previous_side"].to_numpy(dtype=float) == 1, 1.0, -1.0
            )
            autocorrelation = (
                float(np.corrcoef(side_now, side_previous)[0, 1])
                if side_now.size > 2 and np.std(side_now) > 0 and np.std(side_previous) > 0
                else float("nan")
            )
            row = {
                "subject": subject,
                "condition": name,
                "n_linked_trials": int(len(linked)),
                "win_stay": (
                    float(repeated[after_correct].mean()) if after_correct.any() else float("nan")
                ),
                "lose_stay": (
                    float(repeated[after_error].mean()) if after_error.any() else float("nan")
                ),
                "side_autocorrelation_lag1": autocorrelation,
            }
            row["lose_shift"] = (
                1.0 - row["lose_stay"] if np.isfinite(row["lose_stay"]) else float("nan")
            )
            dt_after = {
                "correct": linked.loc[after_correct, "dt"],
                "error": linked.loc[after_error, "dt"],
            }
            for outcome, values in dt_after.items():
                row[f"mean_dt_after_{outcome}_ms"] = (
                    float(values.mean()) if values.notna().any() else float("nan")
                )
            for code, class_name in CLASS_NAMES.items():
                values = linked.loc[linked["previous_trial_class"] == code, "dt"]
                row[f"mean_dt_after_{class_name}_ms"] = (
                    float(values.mean()) if values.notna().any() else float("nan")
                )
            rows.append(row)
    return pd.DataFrame(rows)


def choice_history_statistics(subject_table: pd.DataFrame) -> pd.DataFrame:
    """Test choice-history effects across subjects."""
    rows = []
    for condition, group in subject_table.groupby("condition", sort=True):
        rows.append(
            {
                "analysis": "choice_history",
                "measure": "win_stay_vs_lose_stay",
                "condition": condition,
                "test": "paired_t_test",
                "label_a": "win_stay",
                "label_b": "lose_stay",
                **paired_subject_statistics(group, "win_stay", "lose_stay"),
            }
        )
        rows.append(
            {
                "analysis": "choice_history",
                "measure": "side_autocorrelation_lag1",
                "condition": condition,
                "test": "one_sample_vs_zero",
                **one_sample_statistics(group["side_autocorrelation_lag1"]),
            }
        )
        rows.append(
            {
                "analysis": "choice_history",
                "measure": "dt_after_error_vs_correct",
                "condition": condition,
                "test": "paired_t_test",
                "label_a": "after_error",
                "label_b": "after_correct",
                **paired_subject_statistics(
                    group, "mean_dt_after_error_ms", "mean_dt_after_correct_ms"
                ),
            }
        )
        for first, second in (("easy", "ambiguous"), ("easy", "misleading"), ("ambiguous", "misleading")):
            rows.append(
                {
                    "analysis": "choice_history",
                    "measure": f"dt_after_{first}_vs_{second}",
                    "condition": condition,
                    "test": "paired_t_test",
                    "label_a": f"after_{first}",
                    "label_b": f"after_{second}",
                    **paired_subject_statistics(
                        group, f"mean_dt_after_{first}_ms", f"mean_dt_after_{second}_ms"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _as_optional_bool(value: object) -> object:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
        return np.nan
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    return bool(value)
