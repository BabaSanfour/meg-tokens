"""Distributional analyses of decision time and success probability.

Means alone hide the right tail that distinguishes a threshold change from a
drift change, so these helpers report subject-level quantiles and skewness for
decision time, and cumulative distributions for logged success probability at
decision.
"""

from __future__ import annotations

from typing import Final, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from meg_tokens.behavior.math.inference import (
    confidence_interval,
    one_sample_statistics,
    paired_subject_statistics,
)
from meg_tokens.behavior.trials import (
    CLASS_NAMES,
    TASK_CONDITIONS,
    classified_trials,
    finite_values,
    require_columns,
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

# Fewer trials than this cannot support a stable third or fourth moment; those
# statistics are reported as NaN rather than as noise with a number attached.
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

    Returns
    -------
    pandas.DataFrame
        One row per subject and fixed stratum.  Identifier columns are
        ``subject``, ``stratum_type``, and ``stratum``; remaining columns are
        the outputs of :func:`distribution_summary`.  Decision-time values are
        in milliseconds.

    Notes
    -----
    Non-task rows are removed by :func:`task_trials`.  Empty declared cells
    are retained with ``n_trials == 0`` and ``NaN`` statistics.  No trial is
    reassigned, imputed, winsorized, or pooled across subjects.

    Distribution shape is described by quantiles and skewness only.  A
    parametric ex-Gaussian decomposition was fitted here previously and has
    been removed: at this cohort's per-stratum trial counts (54-128) the
    three-parameter fit is not identifiable -- subject-level mu/tau and
    sigma/tau correlations ran -0.5 to -0.65 for every stratum except the
    individual trial classes, and published recovery studies put sigma's 95%
    interval at [0, 141] for a true sigma of 100 at n = 50.  Quantile
    contrasts answer the same shape questions without the identifiability
    problem, and are what the tokens-task literature reports.
    """
    trials = task_trials(features)
    rows = []
    for subject, subject_trials in trials.groupby("subject", sort=True):
        for stratum_type, stratum, selection in _strata(subject_trials):
            values = finite_values(subject_trials.loc[selection, "dt_ms"])
            rows.append({
                "subject": subject,
                "stratum_type": stratum_type,
                "stratum": stratum,
                **distribution_summary(values, quantiles=quantiles),
            })
    return pd.DataFrame(rows)


def decision_time_distribution_statistics(
    subject_distributions: pd.DataFrame,
    *,
    metrics: Sequence[str] = ("q10", "q50", "q90", "skewness"),
) -> pd.DataFrame:
    """Run fixed paired contrasts on subject-level distribution metrics.

    Parameters
    ----------
    subject_distributions
        Output of :func:`decision_time_distributions`, with one unique row per
        ``subject`` and ``stratum`` and columns for every requested metric.
    metrics
        Distribution-summary columns to test.  The defaults compare the 10th,
        50th, and 90th percentiles and skewness.  Requested metrics absent
        from ``subject_distributions`` are skipped.

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

    Every row is keyed by trial class, so the unclassified 43% is excluded by
    name rather than falling out of the per-class comparison below. The
    distributions are unchanged either way.
    """
    trials = classified_trials(features)
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


GEOMETRY_CONTRASTS: Final[tuple[tuple[str, str, str], ...]] = (
    ("condition", "fast", "slow"),
    ("class", "easy", "ambiguous"),
    ("class", "easy", "misleading"),
    ("class", "ambiguous", "misleading"),
)


def _geometry_samples(trials: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    """Decision times per ``(stratum_type, label)`` for one subject."""
    condition = trials["condition"].astype(str).str.lower()
    class_name = trials["trial_class_name"].astype(str).str.lower()
    samples: dict[tuple[str, str], np.ndarray] = {}
    for name in TASK_CONDITIONS:
        label = name.lower()
        samples[("condition", label)] = finite_values(
            trials.loc[condition == label, "dt_ms"]
        )
    for label in CLASS_NAMES.values():
        samples[("class", label)] = finite_values(
            trials.loc[class_name == label, "dt_ms"]
        )
    return samples


def decision_time_geometry(
    features: pd.DataFrame,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    min_trials: int = MIN_TRIALS_FOR_SHAPE,
) -> pd.DataFrame:
    """Per-subject geometry of one decision-time distribution against another.

    Two distributions can differ because a constant was *added* to every trial
    (a delay), because the whole process ran proportionally slower (a stretch),
    or because the distribution changed *shape*. Distinguishing the three is
    the standing audit procedure for this dataset, and it is what the report
    argues from for both the Fast/Slow contrast and the difficulty contrasts.
    It was being done by hand; this makes it a derivative.

    Parameters
    ----------
    features
        Canonical trial-feature table.
    quantiles
        Quantile probabilities compared across the pair.
    min_trials
        Minimum finite trials required on *each* side of a contrast.

    Returns
    -------
    pandas.DataFrame
        One row per subject and contrast in :data:`GEOMETRY_CONTRASTS`,
        reporting per-quantile differences and log ratios, the two sides'
        means/SDs/CVs together with what a pure delay and a pure stretch each
        predict for side ``b``, a per-subject stretch factor, and
        Kolmogorov-Smirnov p-values before and after removing a median offset
        and a median factor.

    Notes
    -----
    Differences are ``a - b`` and ratios are ``b / a``, matching
    :func:`decision_time_distribution_statistics` so the two tables can be read
    together; for ``fast`` vs ``slow`` that makes the difference negative and
    the ratio the familiar stretch above one.

    The delay and stretch predictions are each given the *best per-subject fit*
    of their own model -- that subject's own offset ``mean_b - mean_a`` or
    their own factor ``mean_b / mean_a`` -- so neither is handicapped by a
    group-level parameter. A delay leaves the SD alone and lowers the CV; a
    stretch scales the SD and leaves the CV alone, which is what makes the two
    separable.

    The stretch factor is the least-squares slope of ``b``'s quantiles on
    ``a``'s through the origin: it uses the whole quantile ladder rather than
    the means, so a pair that differs only in the tail cannot masquerade as a
    uniform stretch.

    The KS ladder is deliberately per subject and reported as p-values, not as
    a count: subtracting each side's median should remove a pure delay, and
    dividing by it should remove a pure stretch, so the *number of subjects
    still differing* after each is the discriminating quantity. Counting is
    left to :func:`decision_time_geometry_statistics`, which also records the
    number expected by chance.
    """
    trials = task_trials(features)
    require_columns(trials, ["subject", "condition", "trial_class_name", "dt_ms"])
    probabilities = tuple(float(value) for value in quantiles)
    labels = [f"q{int(round(value * 100)):02d}" for value in probabilities]

    rows = []
    for subject, subject_trials in trials.groupby("subject", sort=True):
        samples = _geometry_samples(subject_trials)
        for stratum_type, label_a, label_b in GEOMETRY_CONTRASTS:
            values_a = samples.get((stratum_type, label_a), np.array([]))
            values_b = samples.get((stratum_type, label_b), np.array([]))
            row = {
                "subject": subject,
                "stratum_type": stratum_type,
                "contrast": f"{label_a}_vs_{label_b}",
                "label_a": label_a,
                "label_b": label_b,
                "n_a": int(values_a.size),
                "n_b": int(values_b.size),
            }
            if values_a.size < min_trials or values_b.size < min_trials:
                rows.append(row)
                continue

            quantiles_a = np.quantile(values_a, probabilities)
            quantiles_b = np.quantile(values_b, probabilities)
            for label, first, second in zip(labels, quantiles_a, quantiles_b):
                row[f"delta_{label}_ms"] = float(first - second)
                row[f"log_ratio_{label}"] = (
                    float(np.log(second / first)) if first > 0 and second > 0 else np.nan
                )

            mean_a, mean_b = float(values_a.mean()), float(values_b.mean())
            sd_a = float(values_a.std(ddof=1))
            sd_b = float(values_b.std(ddof=1))
            factor = mean_b / mean_a if mean_a > 0 else np.nan
            row.update(
                {
                    "mean_a_ms": mean_a,
                    "mean_b_ms": mean_b,
                    "sd_a_ms": sd_a,
                    "sd_b_ms": sd_b,
                    "cv_a": sd_a / mean_a if mean_a > 0 else np.nan,
                    "cv_b": sd_b / mean_b if mean_b > 0 else np.nan,
                    # A delay shifts the mean and leaves the spread alone.
                    "sd_b_delay_predicted_ms": sd_a,
                    "cv_b_delay_predicted": sd_a / mean_b if mean_b > 0 else np.nan,
                    # A stretch scales both, so the CV is unchanged.
                    "sd_b_stretch_predicted_ms": sd_a * factor,
                    "cv_b_stretch_predicted": sd_a / mean_a if mean_a > 0 else np.nan,
                    "mean_factor": factor,
                    "stretch_factor": (
                        float(np.dot(quantiles_a, quantiles_b) / np.dot(quantiles_a, quantiles_a))
                        if np.dot(quantiles_a, quantiles_a) > 0
                        else np.nan
                    ),
                    "ks_p_raw": float(stats.ks_2samp(values_a, values_b).pvalue),
                }
            )
            median_a, median_b = float(np.median(values_a)), float(np.median(values_b))
            row["ks_p_median_subtracted"] = float(
                stats.ks_2samp(values_a - median_a, values_b - median_b).pvalue
            )
            row["ks_p_median_divided"] = (
                float(stats.ks_2samp(values_a / median_a, values_b / median_b).pvalue)
                if median_a > 0 and median_b > 0
                else np.nan
            )
            rows.append(row)
    return pd.DataFrame(rows)


def decision_time_geometry_statistics(
    geometry: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Group tests deciding delay versus stretch versus shape change.

    Parameters
    ----------
    geometry
        Output of :func:`decision_time_geometry`.
    alpha
        Per-subject significance level for the KS ladder counts.

    Returns
    -------
    pandas.DataFrame
        Per contrast: a ``fixed_delay`` test (do the quantile *differences*
        change across the ladder?), a ``proportional_stretch`` test (do the
        *log ratios*?), the SD and CV of side ``b`` against what each model
        predicts, the stretch factor against one, and KS ladder counts before
        and after removing a median offset and a median factor.

    Notes
    -----
    The two geometry tests are deliberately opposed. A fixed delay predicts a
    constant difference, so evidence *against* it is a reliable change in the
    difference between the extreme quantiles. A stretch predicts a constant
    ratio, so evidence against it is a reliable change in the log ratio.
    Rejecting the first and not the second is the signature of a stretch;
    rejecting both is a shape change. Because a non-rejection carries the
    claim here, both rows report an interval and must be quoted as bounds.

    ``n_expected_by_chance`` is ``alpha`` times the number of subjects tested.
    A ladder count is only interpretable against it -- with 32 subjects roughly
    1.6 will differ at ``alpha`` even when the distributions are identical.
    """
    rows: list[dict] = []
    if geometry.empty:
        return pd.DataFrame(rows)

    for (stratum_type, contrast), group in geometry.groupby(
        ["stratum_type", "contrast"], sort=True
    ):
        common = {
            "analysis": "decision_time_geometry",
            "stratum_type": stratum_type,
            "contrast": contrast,
        }

        def _add(term: str, values, *, coverage: float = 0.95, null: float = 0.0):
            statistics = one_sample_statistics(np.asarray(values, dtype=float) - null)
            low, high = confidence_interval(
                statistics["mean"], statistics["sem"], statistics["df"], coverage=coverage
            )
            rows.append({
                **common, "term": term, "test": "one_sample_vs_zero" if not null
                else "one_sample_vs_null",
                "estimate": statistics["mean"] + null,
                "ci_low": low + null, "ci_high": high + null, "coverage": coverage,
                **{key: statistics[key] for key in ("n_subjects", "t", "p", "df", "cohens_dz")},
            })

        # A delay predicts a constant difference across the ladder; a stretch
        # predicts a constant ratio. Each row is evidence *against* one model.
        if {"delta_q90_ms", "delta_q10_ms"}.issubset(group.columns):
            _add("fixed_delay_q90_minus_q10_ms",
                 group["delta_q90_ms"] - group["delta_q10_ms"])
        if {"log_ratio_q90", "log_ratio_q10"}.issubset(group.columns):
            _add("proportional_stretch_log_ratio_q90_minus_q10",
                 group["log_ratio_q90"] - group["log_ratio_q10"])

        for measure in ("sd", "cv"):
            observed = f"{measure}_b_ms" if measure == "sd" else "cv_b"
            for model in ("delay", "stretch"):
                predicted = (
                    f"sd_b_{model}_predicted_ms" if measure == "sd"
                    else f"cv_b_{model}_predicted"
                )
                if {observed, predicted}.issubset(group.columns):
                    _add(f"{measure}_observed_minus_{model}_predicted",
                         group[observed] - group[predicted])

        if "stretch_factor" in group.columns:
            _add("stretch_factor", group["stretch_factor"], null=1.0)

        for column, label in (
            ("ks_p_raw", "raw"),
            ("ks_p_median_subtracted", "median_subtracted"),
            ("ks_p_median_divided", "median_divided"),
        ):
            if column not in group.columns:
                continue
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if not len(values):
                continue
            rows.append({
                **common, "term": "ks_subjects_differing", "test": "count",
                "normalization": label,
                "n_subjects": int(len(values)),
                "estimate": float((values < alpha).sum()),
                "n_expected_by_chance": float(alpha * len(values)),
                "alpha": alpha,
            })
    return pd.DataFrame(rows)
