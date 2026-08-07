"""Load and enrich canonical behavioral derivative tables.

This module is the behavioral boundary over project-level tabular I/O. It
selects the appropriate schema converters and validators but contains no TDMS
parsing or scientific analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from meg_tokens.behavior.schema import (
    behavior_table_converters,
    token_direction_converter,
    validate_behavior_table,
    validate_trial_features,
)
from meg_tokens.io import load_table

if TYPE_CHECKING:
    from meg_tokens.behavior.tdms import TdmsRunInfo


def add_run_metadata(
    table: pd.DataFrame,
    run_info: TdmsRunInfo,
    source_file: str,
) -> pd.DataFrame:
    """Attach run identity and response fields to parsed TDMS trials.

    Parameters
    ----------
    table
        Parsed trials from one TDMS file, without run-level identity columns.
    run_info
        Canonical subject, condition, run, and acquisition date parsed from the
        filename.
    source_file
        Source filename recorded verbatim for provenance.

    Returns
    -------
    pandas.DataFrame
        Copy of ``table`` with leading ``subject``, ``condition``, ``run``, and
        ``source_file`` columns plus ``rawRT`` and nullable ``isCorrect``.

    Notes
    -----
    ``rawRT`` is ``tEnterTarget - tGO`` in milliseconds and is defined only
    when ``nChoiceMade > 0``. Correctness is likewise absent on no-choice
    trials. The input table is not modified.
    """
    out = table.copy()
    out.insert(0, "subject", run_info.subject)
    out.insert(1, "condition", run_info.condition)
    out.insert(2, "run", int(run_info.run))
    out.insert(3, "source_file", source_file)
    chosen = out["nChoiceMade"] > 0
    out["rawRT"] = (out["tEnterTarget"] - out["tGO"]).where(chosen)
    out["isCorrect"] = (
        out["nChoiceMade"] == out["nCorrectChoice"]
    ).where(chosen)
    return out


def read_behavior_table(path: str | Path) -> pd.DataFrame:
    """Load one Stage 1 behavioral derivative under the strict schema.

    Parameters
    ----------
    path
        TSV derivative path.

    Returns
    -------
    pandas.DataFrame
        Deserialized and validated run table. Sequence columns are Python
        lists, booleans use canonical values, and compact token directions
        remain strings.

    Raises
    ------
    ValueError
        If serialized values or the resulting table violate the Stage 1
        contract.
    """
    table = load_table(path, converters=behavior_table_converters())
    validate_behavior_table(table)
    return table


def read_trial_features(path: str | Path) -> pd.DataFrame:
    """Load the canonical analysis-ready trial-feature derivative.

    Parameters
    ----------
    path
        TSV derivative path.

    Returns
    -------
    pandas.DataFrame
        Validated trial features with compact token directions preserved as
        text even when they contain digits only.

    Raises
    ------
    ValueError
        If required fields, canonical types, token directions, or MEG join-key
        uniqueness are invalid.
    """
    table = load_table(
        path,
        converters={"token_directions": token_direction_converter},
    )
    validate_trial_features(table)
    return table
