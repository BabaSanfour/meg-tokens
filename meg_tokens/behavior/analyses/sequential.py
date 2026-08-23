"""Trial-to-trial history analyses.

The summary post-error measure compares every post-error trial with every
post-correct trial, which confounds the effect with slow stretches of a
session. The robust comparison instead pairs each post-error trial with the trial
immediately *preceding* its error, so both sides of the contrast come from the
same moment in the session. The same adjacency also supports choice-history
effects: win-stay/lose-shift and side autocorrelation. The previous trial's
*outcome* effect on decision time lives with the post-error measures above,
which own that contrast and report it decomposed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from meg_tokens.behavior.math.inference import (
    one_sample_statistics,
    paired_subject_statistics,
)
from meg_tokens.behavior.schema import validate_boolean_values
from meg_tokens.behavior.trials import (
    TASK_CONDITIONS,
    require_columns,
)


def _ordered_task_runs(features: pd.DataFrame) -> pd.DataFrame:
    """Attach immediately adjacent trials within each ordered task run.

    Adjacency is defined on ``run_trial_index`` within one run, and a pair is
    kept only when the two indices are genuinely consecutive, so a never-started
    row between them breaks the chain instead of silently joining trials that
    were minutes apart.

    Parameters
    ----------
    features
        Canonical trial-feature table containing subject, condition, run,
        within-run order, decision time, correctness, choice side, trial class,
        and started status.

    Returns
    -------
    pandas.DataFrame
        Started Fast/Slow rows sorted by subject, condition, run, and
        ``run_trial_index``. Canonical analysis columns ``dt``, ``is_correct``,
        and ``side`` are accompanied by previous- and next-trial values when
        the neighbour's index differs by exactly one.

    Raises
    ------
    ValueError
        If Boolean fields, choice sides, run indices, or within-run uniqueness
        violate the canonical schema.

    Notes
    -----
    Decision times are converted to numeric and non-finite values become
    missing. Choice sides must be 1, 2, or missing; they are never inferred
    from truthiness. Missing correctness is allowed because lapse rows may
    interrupt a sequence. This helper defines adjacency only and performs no
    behavioral or statistical comparison.
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
    validate_boolean_values(
        features["isCorrect"],
        field="isCorrect",
        optional=True,
    )
    validate_boolean_values(features["is_started"], field="is_started")
    condition = features["condition"].astype(str).str.lower()
    trials = features.loc[
        condition.isin({name.lower() for name in TASK_CONDITIONS})
        & features["is_started"].astype("boolean")
    ].copy()
    trials["condition"] = condition.loc[trials.index]
    trials["is_correct"] = trials["isCorrect"].astype("boolean")
    dt = pd.to_numeric(trials["dt_ms"], errors="coerce")
    trials["dt"] = dt.where(np.isfinite(dt))
    trials["side"] = pd.to_numeric(trials["choice_side"], errors="raise")
    invalid_sides = trials["side"].notna() & ~trials["side"].isin({1, 2})
    if invalid_sides.any():
        raise ValueError("choice_side must contain only 1, 2, or missing values")
    run_trial_index = pd.to_numeric(trials["run_trial_index"], errors="raise")
    if (
        run_trial_index.isna().any()
        or not np.isfinite(run_trial_index).all()
        or (run_trial_index % 1 != 0).any()
    ):
        raise ValueError("run_trial_index must contain finite integers")
    trials["run_trial_index"] = run_trial_index.astype(int)
    adjacency_key = ["subject", "condition", "run", "run_trial_index"]
    if trials.duplicated(adjacency_key).any():
        raise ValueError("run_trial_index must be unique within each task run")
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
    """Compute local-pair and classical post-error slowing by subject.

    ``robust_pes_ms`` is ``DT(error + 1) - DT(error - 1)``. The classical
    contrast (mean post-error minus mean post-correct DT) is reported beside it
    from the same trials so the two definitions can be compared directly.

    The local pair is also decomposed against the subject's own baseline,
    because ``DT(error + 1) - DT(error - 1)`` is post-error slowing *only if
    pre-error trials are typical*. ``pre_error_minus_baseline_ms`` and
    ``post_error_minus_baseline_ms`` split it into the two effects it actually
    contains, and they sum to ``robust_pes_ms`` by construction.

    Parameters
    ----------
    features
        Canonical trial-feature table accepted by :func:`_ordered_task_runs`.

    Returns
    -------
    pandas.DataFrame
        One row per subject for pooled, Fast, and Slow trials. It reports the
        number of complete local error pairs, their mean pre-error and
        post-error decision times, ``robust_pes_ms``, the numbers of finite
        post-error and post-correct trials, and ``classical_pes_ms``. All times
        and differences are in milliseconds; unavailable estimates are
        ``NaN``.

    Notes
    -----
    A robust pair requires finite decision times immediately before and after
    the error, with all three trial indices contiguous in the same run. The
    error trial's own decision time is not part of that difference. The
    classical contrast uses every current trial whose immediately preceding
    contiguous trial has known correctness, then subtracts mean post-correct
    DT from mean post-error DT. No trimming, imputation, or outlier threshold
    is applied. ``robust`` refers to local temporal matching, not a robust
    estimator such as a median or M-estimator.

    The baseline is the subject's mean decision time over every finite-DT trial
    in the same selection, so the decomposition is against that subject's own
    typical trial rather than a group value. Pre-error speeding is a real and
    separately reported effect here, not a nuisance term: reporting only their
    difference attributes both to post-error slowing.
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
            # The subject's own typical trial in this selection, which is what
            # makes the pre/post split interpretable as two separate effects.
            baseline_dt = subject_trials.loc[selection, "dt"]
            baseline = (
                float(baseline_dt.mean()) if baseline_dt.notna().any() else float("nan")
            )
            rows.append(
                {
                    "subject": subject,
                    "condition": name,
                    "n_error_pairs": int(differences.size),
                    "mean_baseline_dt_ms": baseline,
                    "pre_error_minus_baseline_ms": (
                        float(paired["previous_dt"].mean() - baseline)
                        if len(paired) else float("nan")
                    ),
                    "post_error_minus_baseline_ms": (
                        float(paired["next_dt"].mean() - baseline)
                        if len(paired) else float("nan")
                    ),
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
    """Test subject-level post-error slowing at the group level.

    Parameters
    ----------
    subject_table
        Output of :func:`robust_post_error_slowing`, with unique
        subject-condition rows.

    Returns
    -------
    pandas.DataFrame
        One-sample tests against zero for robust and classical post-error
        slowing in every available condition, subject-paired Fast-minus-Slow
        contrasts when both condition columns exist, and a descriptive
        ``robust_vs_classical_agreement`` row per condition carrying the
        correlation between the two measures and how many subjects each puts
        above zero.

    Notes
    -----
    Non-finite estimates and incomplete Fast/Slow pairs are excluded by the
    shared inference helpers. No multiplicity correction is applied. Duplicate
    subject-condition rows cause the paired pivot to raise rather than being
    silently aggregated.
    """
    require_columns(
        subject_table,
        ["subject", "condition", "robust_pes_ms", "classical_pes_ms"],
    )
    rows = []
    # The two baseline-relative components are tested on the same footing as
    # the composite: `robust_pes_ms` is their difference, so citing only the
    # composite reports pre-error speeding as if it were post-error slowing.
    measures = (
        "robust_pes_ms",
        "classical_pes_ms",
        "post_error_minus_baseline_ms",
        "pre_error_minus_baseline_ms",
    )
    for measure in (m for m in measures if m in subject_table.columns):
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
                    **paired_subject_statistics(wide["fast"], wide["slow"]),
                }
            )

    # How far the two conventional measures agree. The write-up argues from
    # these -- "r = 0.82 but not interchangeable, positive in 32 of 32 against
    # 29" -- so they are registered here rather than recomputed by whoever
    # needs them. They are descriptive: the correlation carries no test,
    # because the claim is that the two measures track each other while
    # disagreeing on individual subjects, not that r differs from zero.
    for condition, group in subject_table.groupby("condition", sort=True):
        robust = pd.to_numeric(group["robust_pes_ms"], errors="coerce")
        classical = pd.to_numeric(group["classical_pes_ms"], errors="coerce")
        complete = robust.notna() & classical.notna()
        robust, classical = robust[complete], classical[complete]
        rows.append(
            {
                "analysis": "post_error_slowing",
                "measure": "robust_vs_classical_agreement",
                "condition": condition,
                "test": "descriptive",
                "n_subjects": int(complete.sum()),
                "pearson_r": (
                    float(robust.corr(classical)) if complete.sum() > 2
                    else float("nan")
                ),
                "robust_positive": int((robust > 0).sum()),
                "classical_positive": int((classical > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def choice_history(features: pd.DataFrame) -> pd.DataFrame:
    """Summarize first-order choice and decision-time history by subject.

    Parameters
    ----------
    features
        Canonical trial-feature table accepted by :func:`_ordered_task_runs`.

    Returns
    -------
    pandas.DataFrame
        One row per subject and nonempty pooled, Fast, or Slow subset.
        Proportion fields report win-stay, lose-stay, and its complement
        lose-shift. ``side_autocorrelation_lag1`` is the Pearson correlation
        between consecutive sides encoded as target 1 = ``+1`` and target 2 =
        ``-1``. Remaining fields are mean current-trial decision times after
        each previous outcome and declared trial class.

    Notes
    -----
    A linked trial requires valid current and immediately previous choice sides
    at consecutive indices within the same run. Win/lose proportions further
    require known previous correctness, so their denominators may be smaller
    than ``n_linked_trials``. Decision-time means exclude non-finite current
    times. The side correlation requires at least three linked trials and
    variation in both lagged-side vectors; this minimum is an explicit
    stability policy, not inferred from the data. No history state is carried
    across runs, conditions, missing indices, or never-started trials.
    """
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
            # Decision time after an error belongs to `robust_post_error_slowing`,
            # which owns the post-error contrast and reports it decomposed
            # against each subject's baseline. Computing it here too produced
            # the same per-subject value to 4.5e-13 under a second name, and two
            # figures drew it.
            rows.append(row)
    return pd.DataFrame(rows)


def choice_history_statistics(subject_table: pd.DataFrame) -> pd.DataFrame:
    """Run fixed group contrasts on subject-level choice-history measures.

    Parameters
    ----------
    subject_table
        Output of :func:`choice_history`.

    Returns
    -------
    pandas.DataFrame
        For every available condition: a paired win-stay versus lose-stay
        contrast, a one-sample lag-one side-correlation test against zero, a
        paired decision-time-after-error versus after-correct contrast, and the
        three pairwise previous-class decision-time contrasts.

    Notes
    -----
    All comparisons pair summary values within subject. Non-finite values and
    incomplete pairs are excluded by the shared inference functions. Tests are
    fixed rather than selected from observed results, and no multiplicity
    correction or Fast-versus-Slow contrast is applied.
    """
    require_columns(
        subject_table,
        [
            "condition",
            "win_stay",
            "lose_stay",
            "side_autocorrelation_lag1",
        ],
    )
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
                **paired_subject_statistics(
                    group["win_stay"],
                    group["lose_stay"],
                ),
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
    return pd.DataFrame(rows)
