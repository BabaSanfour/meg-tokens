"""Individual differences and cross-species comparison.

Covers roadmap Tier B9 (relating each subject's speed-accuracy adjustment,
urgency, evidence sensitivity, and accuracy to one another and to neural
measures) and Tier C6 (reporting the statistics the Cisek-lab monkey work
reports, in the same form, so the two can be compared).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from meg_tokens.behavior.metrics import paired_subject_statistics
from meg_tokens.behavior.regression import one_sample_statistics
from meg_tokens.behavior.trials import (
    CLASS_NAMES,
    TASK_CONDITIONS,
    finite_values,
    require_columns,
    task_trials,
)


DEFAULT_PROFILE_MEASURES: tuple[str, ...] = (
    "mean_dt_ms",
    "percent_correct",
    "sat_adjustment_ms",
    "urgency_slope_per_second",
    "criterion_slope_per_token",
    "accuracy_log_odds_per_unit",
    "lapse_rate",
)


def individual_profile(
    subject_summary: pd.DataFrame,
    *,
    urgency: Optional[pd.DataFrame] = None,
    criterion: Optional[pd.DataFrame] = None,
    evidence: Optional[pd.DataFrame] = None,
    lapses: Optional[pd.DataFrame] = None,
    neural_metrics: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Assemble one row per subject from the roadmap's subject-level measures.

    ``neural_metrics`` is any table with a ``subject`` column; its remaining
    columns are joined unchanged so that MEG measures can enter the same
    correlation matrix once they exist.
    """
    require_columns(subject_summary, ["subject", "mean_fast_dt_ms", "mean_slow_dt_ms"])
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
            _one_response(criterion),
            "slope",
            "criterion_slope_per_token",
        )
    if evidence is not None and len(evidence):
        selected = evidence.loc[evidence["predictor"] == evidence["predictor"].iloc[0]]
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
        profile = profile.merge(neural_metrics, on="subject", how="left")
    return profile


def _one_response(table: pd.DataFrame, response: str = "logged_spd") -> pd.DataFrame:
    """Keep one evidence scale from a criterion table fitted on several."""
    if "response" not in table.columns:
        return table
    return table.loc[table["response"] == response]


def _merge_condition_value(
    profile: pd.DataFrame,
    table: pd.DataFrame,
    source_column: str,
    target_column: str,
    *,
    condition: str = "all",
) -> pd.DataFrame:
    if source_column not in table.columns:
        return profile
    selected = table.loc[table["condition"] == condition, ["subject", source_column]]
    selected = selected.rename(columns={source_column: target_column})
    return profile.merge(selected, on="subject", how="left")


def individual_correlations(
    profile: pd.DataFrame,
    *,
    measures: Sequence[str] = DEFAULT_PROFILE_MEASURES,
) -> pd.DataFrame:
    """Correlate subject-level measures pairwise across subjects."""
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
    criterion: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Report this dataset in the form the monkey literature reports it.

    Each row names the published measure it is comparable to. Only values
    measured here are filled in; the corresponding monkey values are not
    reproduced in code because they are read from the cited papers.
    """
    rows = []
    for name in CLASS_NAMES.values():
        column = f"mean_{name}_dt_ms"
        if column in subject_summary.columns:
            rows.append(
                {
                    "analysis": "cross_species_comparison",
                    "measure": f"decision_time_{name}_ms",
                    "comparable_to": "DT by trial class (Cisek et al. 2009, Fig. 4)",
                    **one_sample_statistics(subject_summary[column]),
                }
            )
        spd_column = f"mean_{name}_spd_all_logged"
        if spd_column in subject_summary.columns:
            rows.append(
                {
                    "analysis": "cross_species_comparison",
                    "measure": f"success_probability_at_decision_{name}",
                    "comparable_to": "SP at decision by class (Thura et al. 2012, Fig. 3)",
                    **one_sample_statistics(subject_summary[spd_column]),
                }
            )
    if criterion is not None and len(criterion):
        # The log-odds scale is the one the monkey work reports the accuracy
        # criterion on (their SumLogLR at commitment).
        scaled = _one_response(criterion, "logged_spd_log_odds")
        overall = scaled.loc[scaled["condition"] == "all"]
        rows.append(
            {
                "analysis": "cross_species_comparison",
                "measure": "criterion_slope_log_odds_per_token",
                "comparable_to": "decline of evidence at commitment (Thura et al. 2012, Fig. 5)",
                **one_sample_statistics(overall["slope"]),
            }
        )
    return pd.DataFrame(rows)
