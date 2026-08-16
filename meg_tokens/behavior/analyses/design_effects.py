"""Design, session, and lapse analyses.

These are the descriptive checks the class and Fast/Slow summaries do not
cover: the full condition-by-class cell breakdown with its interaction,
left/right choice balance that MEG choice cells depend on, drift across the
session, and the lapses and extreme decision times that a mean hides.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from meg_tokens.behavior.math.inference import (
    one_sample_statistics,
    paired_subject_statistics,
    repeated_measures_anova,
)
from meg_tokens.behavior.schema import validate_boolean_values
from meg_tokens.behavior.trials import (
    CLASS_NAMES,
    LAPSE_OUTCOMES,
    OUTCOME_LABELS,
    TASK_CONDITIONS,
    finite_values,
    lapse_trials,
    require_columns,
    task_trials,
)


SIDE_NAMES: Final[dict[int, str]] = {1: "left", 2: "right"}

# Robust cutoff for the extreme-DT review, in scaled median absolute
# deviations. Flagged trials are reported, never removed.
DEFAULT_MAD_THRESHOLD: Final[float] = 5.0


def condition_class_cells(features: pd.DataFrame) -> pd.DataFrame:
    """Summarize every condition-by-difficulty cell within each subject.

    Parameters
    ----------
    features
        Canonical trial-feature table. Only rows selected by ``task_trials``
        contribute.

    Returns
    -------
    pandas.DataFrame
        One row per subject, Fast/Slow condition, and easy/ambiguous/misleading
        class. Each row reports total trials, mean finite decision time in
        milliseconds, and accuracy among trials with known correctness. Empty
        cells are retained with zero trials and missing summaries.

    Raises
    ------
    ValueError
        If required fields are absent or ``isCorrect`` contains a noncanonical
        value.
    """
    trials = task_trials(features)
    require_columns(trials, ["condition", "trial_class", "dt_ms", "isCorrect"])
    validate_boolean_values(
        trials["isCorrect"],
        field="isCorrect",
        optional=True,
    )
    condition = trials["condition"].astype(str).str.lower()
    correct = trials["isCorrect"].astype("boolean")
    rows = []
    for subject, subject_trials in trials.groupby("subject", sort=True):
        subject_condition = condition.loc[subject_trials.index]
        subject_correct = correct.loc[subject_trials.index]
        for name in TASK_CONDITIONS:
            for code, class_name in CLASS_NAMES.items():
                selection = (subject_condition == name.lower()) & (
                    subject_trials["trial_class"] == code
                )
                cell = subject_trials.loc[selection]
                accuracy_values = subject_correct.loc[selection].dropna()
                dt_values = finite_values(cell["dt_ms"])
                rows.append(
                    {
                        "subject": subject,
                        "condition": name.lower(),
                        "trial_class": code,
                        "trial_class_name": class_name,
                        "n_trials": int(len(cell)),
                        "mean_dt_ms": (
                            float(np.mean(dt_values)) if dt_values.size else float("nan")
                        ),
                        "accuracy": (
                            float(accuracy_values.astype(float).mean())
                            if len(accuracy_values)
                            else float("nan")
                        ),
                    }
                )
    return pd.DataFrame(rows)


def condition_class_statistics(cells: pd.DataFrame) -> pd.DataFrame:
    """Test condition, difficulty, and their interaction within subjects.

    Parameters
    ----------
    cells
        Output of :func:`condition_class_cells`.

    Returns
    -------
    pandas.DataFrame
        Three repeated-measures ANOVA effects for each of decision time,
        log decision time, and accuracy, when all six columns exist. Rows
        report complete-subject count, degrees of freedom, F statistic,
        p-value, and partial eta squared.

    Notes
    -----
    Cell order is Fast then Slow, with class codes 1, 2, and 3 inside each
    condition. Subjects missing any cell for a measure are removed by
    ``repeated_measures_anova``. No sphericity or multiplicity correction is
    applied here.

    ``mean_dt_ms`` and ``log_mean_dt_ms`` test the same cells on different
    scales and answer different questions. The raw interaction asks whether
    Fast/Slow shifts every class by the same number of milliseconds; the log
    interaction asks whether it scales every class by the same factor. The
    condition effect is multiplicative, so only the second is the question
    worth asking -- reported together so the contrast is visible rather than
    a silent modelling choice.
    """
    factor_names = {
        "factor1": "condition",
        "factor2": "trial_class",
        "factor1xfactor2": "condition_x_trial_class",
    }
    rows = []
    for measure in ("mean_dt_ms", "log_mean_dt_ms", "accuracy"):
        source = "mean_dt_ms" if measure == "log_mean_dt_ms" else measure
        wide = cells.pivot_table(
            index="subject",
            columns=["condition", "trial_class"],
            values=source,
            sort=True,
        )
        if measure == "log_mean_dt_ms":
            # Tested on the log scale, where an interaction asks whether the
            # Fast/Slow *ratio* differs by class. The additive interaction
            # answers a question the condition effect is not: that effect is
            # multiplicative (docs/behavior.md, "Fast vs. Slow stretches
            # decision time proportionally"), so a constant ratio shows up as
            # leftover structure when the test is run on raw milliseconds.
            wide = np.log(wide.where(wide > 0))
        ordered = [
            (name.lower(), code)
            for name in TASK_CONDITIONS
            for code in CLASS_NAMES
        ]
        missing = [column for column in ordered if column not in wide.columns]
        if missing:
            continue
        matrix = wide[ordered].to_numpy(dtype=float)
        for effect in repeated_measures_anova(matrix, (len(TASK_CONDITIONS), len(CLASS_NAMES))):
            rows.append(
                {
                    "analysis": "condition_by_class",
                    "measure": measure,
                    "effect": factor_names[effect["effect"]],
                    "n_subjects": effect["n_subjects"],
                    "df_effect": effect["df_effect"],
                    "df_error": effect["df_error"],
                    "F": effect["F"],
                    "p": effect["p"],
                    "partial_eta_squared": effect["partial_eta_squared"],
                }
            )
    return pd.DataFrame(rows)


def choice_side_summary(features: pd.DataFrame) -> pd.DataFrame:
    """Summarize response-side and correct-side balance within each subject.

    Parameters
    ----------
    features
        Canonical trial-feature table containing eligible choices, correct
        targets, decision times, and nullable correctness.

    Returns
    -------
    pandas.DataFrame
        One row per subject for pooled, Fast, and Slow trials. For each side it
        reports the proportion of choices, proportion of correct targets, mean
        decision time among choices to that side, and accuracy conditional on
        choosing that side.

    Notes
    -----
    Side 1 is labeled left and side 2 right. Choice and correct-target
    proportions use every selected row as their denominator; accuracy excludes
    rows with missing correctness.
    """
    trials = task_trials(features)
    require_columns(trials, ["choice_side", "correct_side", "dt_ms", "isCorrect"])
    validate_boolean_values(
        trials["isCorrect"],
        field="isCorrect",
        optional=True,
    )
    condition = trials["condition"].astype(str).str.lower()
    correct = trials["isCorrect"].astype("boolean")
    rows = []
    for subject, subject_trials in trials.groupby("subject", sort=True):
        subject_condition = condition.loc[subject_trials.index]
        for name in ("all", *[item.lower() for item in TASK_CONDITIONS]):
            selection = (
                pd.Series(True, index=subject_trials.index)
                if name == "all"
                else subject_condition == name
            )
            selected = subject_trials.loc[selection]
            if selected.empty:
                continue
            sides = pd.to_numeric(selected["choice_side"], errors="coerce")
            correct_sides = pd.to_numeric(selected["correct_side"], errors="coerce")
            row = {
                "subject": subject,
                "condition": name,
                "n_trials": int(len(selected)),
            }
            for side, side_name in SIDE_NAMES.items():
                side_selection = sides == side
                side_trials = selected.loc[side_selection]
                dt_values = finite_values(side_trials["dt_ms"])
                accuracy_values = (
                    correct.loc[side_trials.index].dropna().astype(float)
                )
                row[f"proportion_{side_name}_choices"] = float(side_selection.mean())
                row[f"proportion_{side_name}_correct_side"] = float(
                    (correct_sides == side).mean()
                )
                row[f"mean_{side_name}_dt_ms"] = (
                    float(np.mean(dt_values)) if dt_values.size else float("nan")
                )
                row[f"accuracy_{side_name}"] = (
                    float(accuracy_values.mean()) if len(accuracy_values) else float("nan")
                )
            rows.append(row)
    return pd.DataFrame(rows)


def choice_side_statistics(summary: pd.DataFrame) -> pd.DataFrame:
    """Run subject-paired left-versus-right comparisons.

    Parameters
    ----------
    summary
        Output of :func:`choice_side_summary`.

    Returns
    -------
    pandas.DataFrame
        For each pooled or condition-specific subset, paired tests of choice
        proportion, mean decision time, and accuracy. Differences and Cohen's
        dz use left minus right; subjects missing either member are excluded
        pairwise.
    """
    rows = []
    for condition, group in summary.groupby("condition", sort=True):
        rows.append(
            {
                "analysis": "choice_side",
                "measure": "choice_proportion",
                "condition": condition,
                "test": "paired_t_test",
                "label_a": "left",
                "label_b": "right",
                **paired_subject_statistics(
                    group["proportion_left_choices"],
                    group["proportion_right_choices"],
                ),
            }
        )
        for measure, column_a, column_b in (
            ("decision_time", "mean_left_dt_ms", "mean_right_dt_ms"),
            ("accuracy", "accuracy_left", "accuracy_right"),
        ):
            rows.append(
                {
                    "analysis": "choice_side",
                    "measure": measure,
                    "condition": condition,
                    "test": "paired_t_test",
                    "label_a": "left",
                    "label_b": "right",
                    **paired_subject_statistics(
                        group[column_a],
                        group[column_b],
                    ),
                }
            )
    return pd.DataFrame(rows)


def _session_block_order(trials: pd.DataFrame) -> pd.Series:
    """Rank each subject's task blocks by when they were actually run.

    ``initial_time_ms`` is the LabVIEW session clock at trial onset and is the
    only field that recovers block order: the filenames carry a date but no
    time, and ``nTrialIndex`` restarts at 1 in each run. Fast and Slow blocks
    are interleaved within a session, so this ordering is not the same as
    sorting by condition and run number.

    Parameters
    ----------
    trials
        Eligible task trials containing subject, condition, run, and
        ``initial_time_ms``.

    Returns
    -------
    pandas.Series
        One-based chronological block rank aligned to ``trials.index``. Every
        row from the same subject-condition-run block receives the same rank.
    """
    keys = trials[["subject", "condition", "run"]]
    first_index = (
        trials.assign(_key=list(map(tuple, keys.to_numpy())))
        .groupby("_key")["initial_time_ms"]
        .min()
    )
    ranks = {}
    for subject in trials["subject"].unique():
        subject_keys = [key for key in first_index.index if key[0] == subject]
        for position, key in enumerate(
            sorted(subject_keys, key=lambda item: first_index[item]), start=1
        ):
            ranks[key] = position
    return pd.Series(
        [ranks[key] for key in map(tuple, keys.to_numpy())],
        index=trials.index,
        dtype=float,
    )


def time_on_task(features: pd.DataFrame) -> pd.DataFrame:
    """Fit DT against block order and within-block position per subject.

    The pooled fit carries a Fast/Slow indicator. Without it the block slope
    would absorb the condition difference, because the two conditions are
    interleaved rather than blocked into halves of the session.

    Parameters
    ----------
    features
        Canonical trial-feature table.

    Returns
    -------
    pandas.DataFrame
        One OLS result per subject for pooled, Fast, and Slow trials. Rows
        report usable-trial and block counts, milliseconds per chronological
        block, milliseconds per within-block trial, and whether the model was
        estimable.

    Notes
    -----
    Condition-specific models are ``DT ~ block + within-block trial``. The
    pooled model additionally includes a Slow-condition indicator. Non-finite
    rows are removed jointly; insufficient or rank-deficient designs return
    missing coefficients with ``converged=false``.
    """
    trials = task_trials(features)
    require_columns(
        trials,
        ["dt_ms", "run_trial_index", "initial_time_ms", "condition", "run"],
    )
    trials = trials.assign(block_position=_session_block_order(trials))
    condition = trials["condition"].astype(str).str.lower()
    rows = []
    for subject, subject_trials in trials.groupby("subject", sort=True):
        subject_condition = condition.loc[subject_trials.index]
        for name in ("all", *[item.lower() for item in TASK_CONDITIONS]):
            selection = (
                pd.Series(True, index=subject_trials.index)
                if name == "all"
                else subject_condition == name
            )
            selected = subject_trials.loc[selection]
            if selected.empty:
                continue
            dt = pd.to_numeric(selected["dt_ms"], errors="coerce").to_numpy(dtype=float)
            block = selected["block_position"].to_numpy(dtype=float)
            within = pd.to_numeric(selected["run_trial_index"], errors="coerce").to_numpy(
                dtype=float
            )
            columns = [np.ones_like(dt), block, within]
            selected_condition = subject_condition.loc[selected.index]
            if selected_condition.nunique() > 1:
                columns.append((selected_condition == "slow").to_numpy(dtype=float))
            design = np.column_stack(columns)
            finite = np.isfinite(dt) & np.all(np.isfinite(design), axis=1)
            design = design[finite]
            dt = dt[finite]
            fit = (
                sm.OLS(dt, design).fit()
                if len(dt) > design.shape[1]
                and np.linalg.matrix_rank(design) == design.shape[1]
                else None
            )
            rows.append(
                {
                    "subject": subject,
                    "condition": name,
                    "n_trials": len(dt),
                    "n_blocks": int(selected["block_position"].nunique()),
                    "dt_per_block_ms": (
                        float(fit.params[1])
                        if fit is not None
                        else float("nan")
                    ),
                    "dt_per_within_block_trial_ms": (
                        float(fit.params[2])
                        if fit is not None
                        else float("nan")
                    ),
                    "converged": fit is not None,
                }
            )
    return pd.DataFrame(rows)


def time_on_task_statistics(subject_fits: pd.DataFrame) -> pd.DataFrame:
    """Test subject-level session-drift coefficients at group level.

    Parameters
    ----------
    subject_fits
        Output of :func:`time_on_task`.

    Returns
    -------
    pandas.DataFrame
        One-sample tests against zero for block and within-block slopes in every
        subset, plus subject-paired Fast-minus-Slow comparisons for each slope.
    """
    terms = ("dt_per_block_ms", "dt_per_within_block_trial_ms")
    rows = []
    for condition, group in subject_fits.groupby("condition", sort=True):
        for term in terms:
            rows.append(
                {
                    "analysis": "time_on_task",
                    "term": term,
                    "condition": condition,
                    "test": "one_sample_vs_zero",
                    **one_sample_statistics(group[term]),
                }
            )
    for term in terms:
        wide = subject_fits.pivot(index="subject", columns="condition", values=term)
        if "fast" not in wide.columns or "slow" not in wide.columns:
            continue
        rows.append(
            {
                "analysis": "time_on_task",
                "term": term,
                "condition": "fast_vs_slow",
                "test": "paired_t_test",
                **paired_subject_statistics(wide["fast"], wide["slow"]),
            }
        )
    return pd.DataFrame(rows)


def condition_order_effects(features: pd.DataFrame) -> pd.DataFrame:
    """Compare the Fast/Slow DT difference between condition-order groups.

    Whether a subject started the session on Fast or Slow blocks is a
    between-subject factor, so this is an independent-samples test on the
    within-subject speed-accuracy adjustment.

    Parameters
    ----------
    features
        Canonical trial-feature table.

    Returns
    -------
    pandas.DataFrame
        One row per subject containing the chronologically first condition,
        mean finite Fast and Slow decision times, and ``Slow - Fast`` decision
        time in milliseconds.

    Notes
    -----
    First condition is determined from ``initial_time_ms`` through
    :func:`_session_block_order`, not from run numbers or filename order.
    """
    trials = task_trials(features)
    require_columns(trials, ["dt_ms", "condition", "initial_time_ms"])
    trials = trials.assign(block_position=_session_block_order(trials))
    condition = trials["condition"].astype(str).str.lower()
    rows = []
    for subject, subject_trials in trials.groupby("subject", sort=True):
        subject_condition = condition.loc[subject_trials.index]
        first_block = subject_trials.loc[
            subject_trials["block_position"] == subject_trials["block_position"].min()
        ]
        first_condition = str(first_block["condition"].iloc[0]).lower()
        means = {
            name: float(np.mean(finite_values(subject_trials.loc[subject_condition == name, "dt_ms"])))
            if (subject_condition == name).any()
            else float("nan")
            for name in (item.lower() for item in TASK_CONDITIONS)
        }
        rows.append(
            {
                "subject": subject,
                "first_condition": first_condition,
                "mean_fast_dt_ms": means.get("fast", float("nan")),
                "mean_slow_dt_ms": means.get("slow", float("nan")),
                "slow_minus_fast_dt_ms": means.get("slow", float("nan"))
                - means.get("fast", float("nan")),
            }
        )
    return pd.DataFrame(rows)


def condition_order_statistics(order_table: pd.DataFrame) -> pd.DataFrame:
    """Compare behavioral means between Fast-first and Slow-first subjects.

    Parameters
    ----------
    order_table
        Output of :func:`condition_order_effects`.

    Returns
    -------
    pandas.DataFrame
        Welch independent-samples t-tests for ``Slow - Fast`` adjustment and
        the two condition means. Each row includes group labels, sample sizes,
        means, ``a - b`` difference, statistic, p-value, and Welch degrees of
        freedom.

    Notes
    -----
    A measure is omitted unless exactly two first-condition groups exist and
    each supplies at least two finite subject values.
    """
    rows = []
    for measure in ("slow_minus_fast_dt_ms", "mean_fast_dt_ms", "mean_slow_dt_ms"):
        groups = {
            str(name): pd.to_numeric(group[measure], errors="coerce").dropna().to_numpy()
            for name, group in order_table.groupby("first_condition", sort=True)
        }
        if len(groups) != 2:
            continue
        (label_a, values_a), (label_b, values_b) = sorted(groups.items())
        if len(values_a) < 2 or len(values_b) < 2:
            continue
        result = stats.ttest_ind(values_a, values_b, equal_var=False)
        rows.append(
            {
                "analysis": "condition_order",
                "measure": measure,
                "test": "welch_t_test",
                "label_a": f"first_{label_a}",
                "label_b": f"first_{label_b}",
                "n_a": int(values_a.size),
                "n_b": int(values_b.size),
                "mean_a": float(np.mean(values_a)),
                "mean_b": float(np.mean(values_b)),
                "mean_difference": float(np.mean(values_a) - np.mean(values_b)),
                "t": float(result.statistic),
                "p": float(result.pvalue),
                "df": float(getattr(result, "df", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def lapse_summary(features: pd.DataFrame) -> pd.DataFrame:
    """Summarize no-choice lapses among started task trials.

    Parameters
    ----------
    features
        Canonical trial-feature table containing started and choice flags,
        condition, and outcome code.

    Returns
    -------
    pandas.DataFrame
        One row per subject for pooled, Fast, and Slow trials. Rows report the
        number of started trials, no-choice trials, lapse proportion, counts
        for the predefined lapse outcome codes, and a residual count for other
        no-choice outcomes.

    Notes
    -----
    The denominator is every started trial in the subset, including completed
    choices. A lapse is defined structurally as started with no usable choice;
    outcome codes describe those lapses but do not determine inclusion.
    """
    require_columns(features, ["nOutcome", "condition", "is_started", "has_choice"])
    lapses = lapse_trials(features)
    started = features.loc[
        features["condition"].astype(str).str.lower().isin(
            {name.lower() for name in TASK_CONDITIONS}
        )
        & features["is_started"].astype(bool)
    ]
    rows = []
    for subject, subject_started in started.groupby("subject", sort=True):
        subject_lapses = lapses.loc[lapses["subject"] == subject]
        condition = subject_started["condition"].astype(str).str.lower()
        for name in ("all", *[item.lower() for item in TASK_CONDITIONS]):
            selection = (
                pd.Series(True, index=subject_started.index)
                if name == "all"
                else condition == name
            )
            n_started = int(selection.sum())
            if not n_started:
                continue
            in_condition = (
                subject_lapses
                if name == "all"
                else subject_lapses.loc[
                    subject_lapses["condition"].astype(str).str.lower() == name
                ]
            )
            outcomes = pd.to_numeric(in_condition["nOutcome"], errors="coerce")
            row = {
                "subject": subject,
                "condition": name,
                "n_started_trials": n_started,
                "n_lapse_trials": int(len(in_condition)),
                "lapse_rate": float(len(in_condition)) / n_started,
            }
            for code in LAPSE_OUTCOMES:
                row[f"n_outcome_{code}_{OUTCOME_LABELS[code]}"] = int((outcomes == code).sum())
            row["n_lapse_other_outcomes"] = int(
                (~outcomes.isin(LAPSE_OUTCOMES)).sum()
            )
            rows.append(row)
    return pd.DataFrame(rows)


def extreme_decision_times(
    features: pd.DataFrame,
    *,
    mad_threshold: float = DEFAULT_MAD_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag DTs far from a subject's median without removing them.

    Returns a per-subject count table and the flagged trials themselves, so
    that an extreme value can be traced back to its run and trial rather than
    only counted.

    Parameters
    ----------
    features
        Canonical trial-feature table. Only eligible task trials with finite
        decision time contribute.
    mad_threshold
        Absolute robust-z cutoff. The default is five scaled median absolute
        deviations from the subject median.

    Returns
    -------
    counts
        Per-subject decision-time count, median, scaled MAD, extreme counts by
        direction, negative-DT count, and observed range.
    flagged
        One provenance-rich row per extreme trial, including trial identifier,
        condition, run, class, decision time, robust z score, and outcome.

    Notes
    -----
    Scaled MAD is ``1.4826 * median(abs(DT - median(DT)))``. When it is zero,
    robust z scores are defined as zero and no trial is flagged. Flagged rows
    are reported only; no observation is removed from another analysis.
    """
    trials = task_trials(features)
    require_columns(trials, ["dt_ms", "trial_id"])
    counts = []
    flagged = []
    for subject, subject_trials in trials.groupby("subject", sort=True):
        dt = pd.to_numeric(subject_trials["dt_ms"], errors="coerce")
        usable = subject_trials.loc[dt.notna()]
        values = dt.dropna().to_numpy(dtype=float)
        if not values.size:
            continue
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median))) * 1.4826
        robust_z = (
            (values - median) / mad if mad > 0 else np.zeros_like(values)
        )
        is_extreme = np.abs(robust_z) > mad_threshold
        counts.append(
            {
                "subject": subject,
                "n_dt_trials": int(values.size),
                "median_dt_ms": median,
                "mad_dt_ms": mad,
                "n_extreme_dt": int(is_extreme.sum()),
                "n_extreme_slow": int((robust_z > mad_threshold).sum()),
                "n_extreme_fast": int((robust_z < -mad_threshold).sum()),
                "n_negative_dt": int((values < 0).sum()),
                "max_dt_ms": float(values.max()),
                "min_dt_ms": float(values.min()),
            }
        )
        for position in np.flatnonzero(is_extreme):
            trial = usable.iloc[int(position)]
            flagged.append(
                {
                    "subject": subject,
                    "trial_id": trial["trial_id"],
                    "condition": trial["condition"],
                    "run": trial["run"],
                    "run_trial_index": trial["run_trial_index"],
                    "trial_class_name": trial.get("trial_class_name"),
                    "dt_ms": float(values[position]),
                    "robust_z": float(robust_z[position]),
                    "nOutcome": trial["nOutcome"],
                }
            )
    return pd.DataFrame(counts), pd.DataFrame(flagged)


def lapse_statistics(summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize lapse rates across subjects and compare conditions.

    Parameters
    ----------
    summary
        Output of :func:`lapse_summary`.

    Returns
    -------
    pandas.DataFrame
        Across-subject lapse-rate summaries for each subset and a paired
        Fast-minus-Slow test when both conditions exist. Rows also carry the
        total number of contributing lapse trials.

    Notes
    -----
    Condition summaries use the generic one-sample helper, which reports the
    mean, SEM, and a two-sided test against zero. No multiplicity correction is
    applied.
    """
    rows = []
    for condition, group in summary.groupby("condition", sort=True):
        rows.append(
            {
                "analysis": "lapses",
                "measure": "lapse_rate",
                "condition": condition,
                "test": "mean",
                "n_lapse_trials": int(group["n_lapse_trials"].sum()),
                **one_sample_statistics(group["lapse_rate"]),
            }
        )
    wide = summary.pivot(index="subject", columns="condition", values="lapse_rate")
    if "fast" in wide.columns and "slow" in wide.columns:
        rows.append(
            {
                "analysis": "lapses",
                "measure": "lapse_rate",
                "condition": "fast_vs_slow",
                "test": "paired_t_test",
                "n_lapse_trials": int(summary["n_lapse_trials"].sum()),
                **paired_subject_statistics(wide["fast"], wide["slow"]),
            }
        )
    return pd.DataFrame(rows)
