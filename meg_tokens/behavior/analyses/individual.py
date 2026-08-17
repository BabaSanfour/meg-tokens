"""Individual-difference and cross-species analyses.

Relates each subject's speed-accuracy adjustment, urgency, evidence
sensitivity, and accuracy to one another and to neural measures. It also
reports the behavioral statistics used in the Cisek-lab monkey work in a
comparable form.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import pandas as pd
from scipy import stats

from meg_tokens.behavior.math.inference import one_sample_statistics
from meg_tokens.behavior.trials import (
    CLASS_NAMES,
    require_columns,
)


DEFAULT_PROFILE_MEASURES: Final[tuple[str, ...]] = (
    "mean_dt_ms",
    "percent_correct",
    "sat_adjustment_ms",
    "urgency_slope_per_second",
    "criterion_slope_sum_log_lr_per_second",
    "accuracy_log_odds_per_unit",
    "lapse_rate",
)


def individual_profile(
    subject_summary: pd.DataFrame,
    *,
    urgency: pd.DataFrame | None = None,
    criterion: pd.DataFrame | None = None,
    evidence: pd.DataFrame | None = None,
    evidence_predictor: str = "sp_design_early",
    lapses: pd.DataFrame | None = None,
    neural_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assemble one explicit individual-differences row per subject.

    Parameters
    ----------
    subject_summary
        Canonical output of
        :func:`~meg_tokens.behavior.analyses.summary.summarize_behavior`.
        Subjects must be unique. Fast and Slow mean decision times are
        required; supported accuracy, class, SPD, and motor columns are copied
        when present.
    urgency
        Optional subject-level criterion fits using decision time as predictor.
        The pooled ``logged_spd`` slope becomes
        ``urgency_slope_per_second``.
    criterion
        Optional subject-level first-order chosen-target SumLogLR fits using
        decision time as predictor. The pooled slope becomes
        ``criterion_slope_sum_log_lr_per_second``.
    evidence
        Optional output of
        :func:`~meg_tokens.behavior.analyses.evidence.continuous_evidence_effects`.
        Pooled accuracy and decision-time slopes for ``evidence_predictor`` are
        joined.
    evidence_predictor
        Exact evidence predictor to select. It is never inferred from row
        order or column naming.
    lapses
        Optional subject-level lapse table. Its pooled ``lapse_rate`` is joined.
    neural_metrics
        Optional table with one row per subject. All non-key columns are joined
        unchanged for later individual-differences analyses.

    Returns
    -------
    pandas.DataFrame
        One row per subject. ``mean_dt_ms`` is the equally weighted mean of the
        subject's Fast and Slow mean decision times, not a trial-weighted mean.
        If one condition mean is missing, pandas returns the available mean; if
        both are missing, the result is ``NaN``. ``sat_adjustment_ms`` is Slow
        minus Fast mean decision time and is ``NaN`` when either is missing.
        Optional inputs contribute columns through left joins and therefore
        never remove subjects from ``subject_summary``.

    Raises
    ------
    KeyError
        If a supplied table lacks a column required for its requested merge.
    ValueError
        If a supplied table has duplicate rows for a merge key.
    """
    require_columns(subject_summary, ["subject", "mean_fast_dt_ms", "mean_slow_dt_ms"])
    if subject_summary["subject"].duplicated().any():
        raise ValueError("subject_summary must contain one row per subject")
    profile = subject_summary[
        [
            column
            for column in (
                "subject",
                "percent_correct",
                "mean_fast_dt_ms",
                "mean_slow_dt_ms",
                "fast_percent_error",
                "slow_percent_error",
                "motor_baseline_ms",
                *[f"mean_{name}_dt_ms" for name in CLASS_NAMES.values()],
                *[f"mean_{name}_spd_all_logged" for name in CLASS_NAMES.values()],
            )
            if column in subject_summary.columns
        ]
    ].copy()
    profile["mean_dt_ms"] = profile[["mean_fast_dt_ms", "mean_slow_dt_ms"]].mean(axis=1)
    profile["sat_adjustment_ms"] = profile["mean_slow_dt_ms"] - profile["mean_fast_dt_ms"]

    if urgency is not None and len(urgency):
        profile = _merge_condition_value(
            profile,
            _one_response(urgency),
            "slope",
            "urgency_slope_per_second",
        )
    if criterion is not None and len(criterion):
        profile = _merge_condition_value(
            profile,
            _one_response(
                criterion,
                "chosen_sum_log_lr_first_order_at_decision",
            ),
            "slope",
            "criterion_slope_sum_log_lr_per_second",
        )
    if evidence is not None and len(evidence):
        require_columns(evidence, ["predictor"])
        selected = evidence.loc[evidence["predictor"] == evidence_predictor]
        profile = _merge_condition_value(
            profile, selected, "accuracy_log_odds_per_unit", "accuracy_log_odds_per_unit"
        )
        profile = _merge_condition_value(
            profile, selected, "dt_slope_ms_per_unit", "dt_slope_ms_per_unit"
        )
    if lapses is not None and len(lapses):
        profile = _merge_condition_value(profile, lapses, "lapse_rate", "lapse_rate")
    if neural_metrics is not None and len(neural_metrics):
        require_columns(neural_metrics, ["subject"])
        if neural_metrics["subject"].duplicated().any():
            raise ValueError("neural_metrics must contain one row per subject")
        profile = profile.merge(neural_metrics, on="subject", how="left")
    return profile


def _one_response(table: pd.DataFrame, response: str = "logged_spd") -> pd.DataFrame:
    """Select one declared evidence scale from a criterion-fit table.

    Parameters
    ----------
    table
        Subject-level criterion fits containing a ``response`` column.
    response
        Exact response label to retain.

    Returns
    -------
    pandas.DataFrame
        Rows whose response label equals ``response``. An unknown label returns
        an empty table; it is not replaced with another evidence scale.
    """
    require_columns(table, ["response"])
    return table.loc[table["response"] == response]


def _merge_condition_value(
    profile: pd.DataFrame,
    table: pd.DataFrame,
    source_column: str,
    target_column: str,
    *,
    condition: str = "all",
) -> pd.DataFrame:
    """Left-join one condition-specific measure into a subject profile.

    Parameters
    ----------
    profile
        Table with one row per subject.
    table
        Subject-level results containing ``subject``, ``condition``, and
        ``source_column``.
    source_column
        Existing result column to select.
    target_column
        Name assigned to the measure in the returned profile.
    condition
        Exact condition label to retain before joining; pooled results use
        ``"all"``.

    Returns
    -------
    pandas.DataFrame
        A left join preserving every profile subject.

    Raises
    ------
    KeyError
        If a required input column is absent.
    ValueError
        If the selected table contains more than one row for a subject.
    """
    require_columns(profile, ["subject"])
    require_columns(table, ["subject", "condition", source_column])
    selected = table.loc[table["condition"] == condition, ["subject", source_column]]
    if selected["subject"].duplicated().any():
        raise ValueError(
            f"multiple {condition!r} rows per subject for {source_column!r}"
        )
    selected = selected.rename(columns={source_column: target_column})
    return profile.merge(selected, on="subject", how="left")


def individual_correlations(
    profile: pd.DataFrame,
    *,
    measures: Sequence[str] = DEFAULT_PROFILE_MEASURES,
) -> pd.DataFrame:
    """Compute pairwise Pearson and Spearman subject-level correlations.

    Parameters
    ----------
    profile
        One-row-per-subject table, normally returned by
        :func:`individual_profile`.
    measures
        Ordered measure names to consider. Measures absent from ``profile``
        are omitted because several default measures come from optional
        profile inputs. Their order determines the reported pair ordering.

    Returns
    -------
    pandas.DataFrame
        One row per estimable unordered measure pair, with the complete-pair
        subject count, Pearson ``r`` and p-value, Spearman ``rho`` and p-value,
        and Pearson degrees of freedom ``n_subjects - 2``.

    Notes
    -----
    Values are converted to numeric and subjects missing either member of a
    pair are excluded for that pair only. At least three complete subjects and
    nonzero variance in both measures are required; otherwise the pair is
    omitted. Correlations are descriptive associations, not causal estimates.
    No outlier removal or multiplicity correction is applied.
    """
    available = [measure for measure in measures if measure in profile.columns]
    rows = []
    for index, first in enumerate(available):
        for second in available[index + 1 :]:
            pair = profile[[first, second]].apply(pd.to_numeric, errors="coerce").dropna()
            n_subjects = int(len(pair))
            if n_subjects < 3:
                continue
            values_a = pair[first].to_numpy(dtype=float)
            values_b = pair[second].to_numpy(dtype=float)
            if np.std(values_a) == 0 or np.std(values_b) == 0:
                continue
            pearson = stats.pearsonr(values_a, values_b)
            spearman = stats.spearmanr(values_a, values_b)
            rows.append(
                {
                    "analysis": "individual_differences",
                    "measure_a": first,
                    "measure_b": second,
                    "n_subjects": n_subjects,
                    "pearson_r": float(pearson.statistic),
                    "pearson_p": float(pearson.pvalue),
                    "spearman_rho": float(spearman.statistic),
                    "spearman_p": float(spearman.pvalue),
                    "df": float(n_subjects - 2),
                }
            )
    return pd.DataFrame(rows)


def comparison_statistics(
    subject_summary: pd.DataFrame,
    *,
    criterion: pd.DataFrame | None = None,
    sequential_sampling: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize human measures that have cross-species counterparts.

    Parameters
    ----------
    subject_summary
        Canonical subject-level behavioral summary containing mean decision
        time and mean logged success probability for every declared trial
        class.
    criterion
        Optional criterion-fit table. When supplied, the pooled slope fitted on
        first-order chosen-target SumLogLR is summarized to match the canonical
        Tokens-task criterion analysis.
    sequential_sampling
        Optional accumulator-fit table from
        :func:`~meg_tokens.behavior.analyses.sequential_sampling.fit_sequential_sampling_models`.
        When supplied, the urgency-gating model's advantage over bounded
        integration and its fitted urgency growth are summarized, because those
        are the quantities the monkey work reports for the same task.

    Returns
    -------
    pandas.DataFrame
        One row per human measure, with the descriptive and one-sample
        fields returned by
        :func:`~meg_tokens.behavior.math.inference.one_sample_statistics`.

    Notes
    -----
    This function does not load monkey data, calculate a human-versus-monkey
    effect, or reproduce values from a publication. One-sample tests are
    against zero and no multiplicity correction is applied.
    """
    required = ["subject"]
    required.extend(f"mean_{name}_dt_ms" for name in CLASS_NAMES.values())
    required.extend(
        f"mean_{name}_spd_all_logged" for name in CLASS_NAMES.values()
    )
    require_columns(subject_summary, required)
    rows = []
    for name in CLASS_NAMES.values():
        column = f"mean_{name}_dt_ms"
        rows.append(
            {
                "analysis": "cross_species_comparison",
                "measure": f"decision_time_{name}_ms",
                **one_sample_statistics(subject_summary[column]),
            }
        )
        spd_column = f"mean_{name}_spd_all_logged"
        rows.append(
            {
                "analysis": "cross_species_comparison",
                "measure": f"success_probability_at_decision_{name}",
                **one_sample_statistics(subject_summary[spd_column]),
            }
        )
    if criterion is not None and len(criterion):
        # Match the first-order chosen-target SumLogLR reported by the monkey
        # work rather than the horizon-dependent exact posterior odds.
        require_columns(criterion, ["subject", "condition", "response", "slope"])
        scaled = _one_response(
            criterion,
            "chosen_sum_log_lr_first_order_at_decision",
        )
        overall = scaled.loc[scaled["condition"] == "all"]
        rows.append(
            {
                "analysis": "cross_species_comparison",
                "measure": "criterion_slope_sum_log_lr_per_second",
                **one_sample_statistics(overall["slope"]),
            }
        )
    if sequential_sampling is not None and len(sequential_sampling):
        require_columns(
            sequential_sampling,
            ["subject", "condition", "model", "delta_bic", "urgency_scale"],
        )
        urgency = sequential_sampling.loc[
            sequential_sampling["model"] == "urgency"
        ].set_index("subject")
        pooled = urgency.loc[urgency["condition"] == "all"]
        fast = urgency.loc[urgency["condition"] == "fast"]
        slow = urgency.loc[urgency["condition"] == "slow"]
        paired = fast.index.intersection(slow.index)
        rows.append(
            {
                "analysis": "cross_species_comparison",
                "measure": "urgency_minus_integrator_bic",
                **one_sample_statistics(pooled["delta_bic"]),
            }
        )
        rows.append(
            {
                "analysis": "cross_species_comparison",
                "measure": "urgency_scale_criterion_seconds",
                **one_sample_statistics(pooled["urgency_scale"]),
            }
        )
        # urgency_scale is threshold / urgency slope, so smaller is faster-rising;
        # a negative Fast-minus-Slow difference is more urgency under time
        # pressure. Entered as a per-subject difference so this row stays a
        # one-sample test against zero, like the rest of the table.
        rows.append(
            {
                "analysis": "cross_species_comparison",
                "measure": "urgency_scale_fast_minus_slow",
                **one_sample_statistics(
                    fast.loc[paired, "urgency_scale"].to_numpy(dtype=float)
                    - slow.loc[paired, "urgency_scale"].to_numpy(dtype=float)
                ),
            }
        )
    return pd.DataFrame(rows)
