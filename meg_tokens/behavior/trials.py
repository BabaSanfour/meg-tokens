"""Shared trial selection for the roadmap behavioral analyses.

Every Tier A/B analysis in ``docs/behavior_analysis_roadmap.md`` consumes the
same substrate: the trial-feature table written by
``meg_tokens.workflows.behavior.analyze_behavior``. These helpers centralize
which rows each analysis is allowed to see so that the eligibility rule is
stated once instead of being re-derived per analysis.
"""

from __future__ import annotations

from typing import Final, Sequence

import numpy as np
import pandas as pd


CLASS_NAMES: Final[dict[int, str]] = {1: "easy", 2: "ambiguous", 3: "misleading"}
CLASS_CODES: Final[dict[str, int]] = {name: code for code, name in CLASS_NAMES.items()}
TASK_CONDITIONS: Final[tuple[str, ...]] = ("Fast", "Slow")

# LabVIEW outcome codes, from archive/DDM_scripts/matlab_scripts (ERROR CODES).
# 7003 (never started) is handled by started_trials and never reaches analysis.
OUTCOME_LABELS: Final[dict[int, str]] = {
    7002: "subject_did_not_start",
    7003: "center_hold_error",
    7004: "target_hold_error",
    7005: "reaction_time_too_short",
    7006: "reaction_time_too_long",
    7007: "movement_time_too_short",
    7008: "movement_time_too_long",
    7011: "delay_1_error",
    7012: "delay_2_error",
    7013: "delay_3_error",
    7014: "delay_4_error",
    7021: "incorrect_choice",
}

# Started trials on which the subject produced no usable choice. These are the
# lapses summarized by Tier A6.
LAPSE_OUTCOMES: Final[tuple[int, ...]] = (7006, 7011)


def require_columns(table: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise if a trial-feature table is missing columns an analysis needs."""
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(
            "Trial-feature table is missing required columns "
            f"{missing}. Re-run 'behavior analyze' to regenerate it."
        )


def task_trials(features: pd.DataFrame) -> pd.DataFrame:
    """Return started Fast/Slow trials on which a choice was made.

    This is the ``primary_analysis_eligible`` view already defined by the
    trial-feature table: never-started rows, RT baseline runs, and lapses are
    all excluded. Analyses that deliberately need lapses (Tier A6) select them
    with :func:`lapse_trials` instead.
    """
    require_columns(features, ["primary_analysis_eligible"])
    return features.loc[features["primary_analysis_eligible"].astype(bool)].copy()


def classified_trials(features: pd.DataFrame) -> pd.DataFrame:
    """Return task trials carrying one of the three difficulty classes."""
    trials = task_trials(features)
    require_columns(trials, ["trial_class"])
    return trials.loc[trials["trial_class"].isin(CLASS_NAMES)].copy()


def lapse_trials(features: pd.DataFrame) -> pd.DataFrame:
    """Return started task trials that received a go cue but no choice."""
    require_columns(features, ["condition", "is_started", "has_choice"])
    condition = features["condition"].astype(str).str.lower()
    return features.loc[
        condition.isin({name.lower() for name in TASK_CONDITIONS})
        & features["is_started"].astype(bool)
        & ~features["has_choice"].astype(bool)
    ].copy()


def finite_values(series: pd.Series) -> np.ndarray:
    """Return the finite numeric values of a column as a float array."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]
