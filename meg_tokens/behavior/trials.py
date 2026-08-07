"""Shared trial selection for the behavioral analysis pipeline.

Every analysis consumes the trial-feature table written by
``meg_tokens.workflows.behavior_analysis.analyze_behavior``. These helpers
centralize which rows each analysis may use so eligibility is stated once
instead of being reconstructed independently.
"""

from __future__ import annotations

from typing import Final, Sequence

import numpy as np
import pandas as pd

from meg_tokens.behavior.schema import OUTCOME_NEVER_STARTED


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

# Started trials on which the subject produced no usable choice.
LAPSE_OUTCOMES: Final[tuple[int, ...]] = (7006, 7011)


def require_columns(table: pd.DataFrame, columns: Sequence[str]) -> None:
    """Require the columns needed by one behavioral operation.

    Parameters
    ----------
    table
        Trial or feature table to inspect.
    columns
        Required column names.

    Raises
    ------
    ValueError
        If any requested column is absent. The message instructs users to
        regenerate the analysis derivative rather than attempting a fallback.
    """
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(
            "Trial-feature table is missing required columns "
            f"{missing}. Re-run 'behavior analyze' to regenerate it."
        )


def started_trials(table: pd.DataFrame) -> pd.DataFrame:
    """Select rows not marked with the never-started outcome sentinel.

    Parameters
    ----------
    table
        Stage 1 or trial-feature table containing ``nOutcome``.

    Returns
    -------
    pandas.DataFrame
        Independent copy of rows whose outcome is not
        ``OUTCOME_NEVER_STARTED``. No response, condition, or class filter is
        applied.
    """
    return table.loc[table["nOutcome"] != OUTCOME_NEVER_STARTED].copy()


def task_trials(features: pd.DataFrame) -> pd.DataFrame:
    """Return started Fast/Slow trials on which a choice was made.

    This is the ``primary_analysis_eligible`` view already defined by the
    trial-feature table: never-started rows, RT baseline runs, and lapses are
    all excluded. Analyses that deliberately need lapses select them with
    :func:`lapse_trials` instead.

    Parameters
    ----------
    features
        Canonical trial-feature table.

    Returns
    -------
    pandas.DataFrame
        Independent copy of rows whose canonical eligibility flag is true.
    """
    require_columns(features, ["primary_analysis_eligible"])
    return features.loc[features["primary_analysis_eligible"].astype(bool)].copy()


def classified_trials(features: pd.DataFrame) -> pd.DataFrame:
    """Select eligible task trials with a defined difficulty class.

    Parameters
    ----------
    features
        Canonical trial-feature table.

    Returns
    -------
    pandas.DataFrame
        Independent copy of eligible rows with class code 1, 2, or 3.

    Notes
    -----
    Class codes 1, 2, and 3 correspond to easy, ambiguous, and misleading.
    Class 0 remains available to continuous-evidence analyses but is excluded
    here. The returned table is an independent copy.
    """
    trials = task_trials(features)
    require_columns(trials, ["trial_class"])
    return trials.loc[trials["trial_class"].isin(CLASS_NAMES)].copy()


def lapse_trials(features: pd.DataFrame) -> pd.DataFrame:
    """Select started Fast/Slow trials without a usable choice.

    Parameters
    ----------
    features
        Canonical trial-feature table.

    Returns
    -------
    pandas.DataFrame
        Independent copy of rows in a task condition with ``is_started=true``
        and ``has_choice=false``. Outcome-code interpretation is left to the
        consuming analysis.
    """
    require_columns(features, ["condition", "is_started", "has_choice"])
    condition = features["condition"].astype(str).str.lower()
    return features.loc[
        condition.isin({name.lower() for name in TASK_CONDITIONS})
        & features["is_started"].astype(bool)
        & ~features["has_choice"].astype(bool)
    ].copy()


def finite_values(series: pd.Series) -> np.ndarray:
    """Extract finite numeric observations from a series.

    Parameters
    ----------
    series
        Values to convert and filter.

    Returns
    -------
    numpy.ndarray
        One-dimensional float array containing only finite observations.

    Notes
    -----
    Values are converted with ``pandas.to_numeric(errors='coerce')``; null,
    non-numeric, positive-infinite, and negative-infinite entries are removed.
    Original indices are not returned.
    """
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]
