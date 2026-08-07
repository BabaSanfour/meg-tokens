"""Strict schemas and deserialization rules for behavioral derivatives.

Stage 1 tables contain parsed TDMS trials plus run identity. Trial-feature
tables contain the analysis-ready, MEG-joinable representation. Validation is
structural and deterministic: alternate legacy encodings are rejected rather
than inferred.
"""

from __future__ import annotations

from ast import literal_eval
from typing import Final

import numpy as np
import pandas as pd


TRIAL_COLUMNS = [
    "nTrialIndex", "sTrialClass", "sTrialClassRaw", "trial_class_source",
    "trial_class_rule", "sp_design_correct", "nInitialTime", "nChoiceMade",
    "nCorrectChoice", "tGO", "tEnterCenter", "tExitCenter", "tEnterTarget",
    "tTrialEnd", "sTokenDirs", "nTokenNum", "nTokenDir", "tTime", "nProb",
    "token_log_rows", "token_log_short", "nOutcome",
]

OUTCOME_NEVER_STARTED: Final[int] = 7003

SEQUENCE_COLUMNS: Final[dict[str, type]] = {
    "sp_design_correct": float,
    "nTokenNum": int,
    "nTokenDir": int,
    "tTime": float,
    "nProb": float,
}

TRIAL_FEATURE_COLUMNS = [
    "trial_id", "subject", "condition", "run", "block_index",
    "run_trial_index", "started_trial_index", "nTrialIndex",
    "initial_time_ms", "source_file", "trial_class", "trial_class_name",
    "sTrialClassRaw", "trial_class_source", "trial_class_rule",
    "nChoiceMade", "nCorrectChoice", "choice_side", "correct_side",
    "isCorrect", "nOutcome", "outcome_label", "motor_baseline_ms",
    "rawRT", "dt_ms", "logged_spd", "logged_spd_log_odds",
    "logged_spd_validated_15row", "evidence_at_decision",
    "decision_token_index", "token_directions", "sp_design_early",
    "sum_log_lr_design_early", "token_lead_at_decision",
    "sum_log_lr_at_decision", "sum_log_lr_saturated", "token_log_rows",
    "token_log_short", "is_started", "has_choice", "is_no_response",
    "dt_anticipation", "spd_available", "design_time_alignment_valid",
    "primary_analysis_eligible",
]


def parse_token_directions(value: object) -> list[int]:
    """Parse the canonical compact token-direction string.

    Parameters
    ----------
    value
        A non-empty string containing only ``1`` and ``2``, with one character
        per token jump.

    Returns
    -------
    list of int
        Target identifiers in jump order.

    Raises
    ------
    ValueError
        If ``value`` is not a non-empty canonical direction string or contains
        a character other than ``1`` or ``2``.
    """
    if not isinstance(value, str) or not value:
        raise ValueError("Token directions must be a non-empty string")
    if any(character not in {"1", "2"} for character in value):
        raise ValueError("Token directions must contain only 1 and 2")
    return [int(character) for character in value]


def token_direction_converter(value: object) -> str:
    """Validate a serialized token-direction field while preserving text.

    Parameters
    ----------
    value
        Raw TSV cell supplied by pandas.

    Returns
    -------
    str
        The validated compact direction string.

    Raises
    ------
    ValueError
        If the value is not a non-empty string containing only ``1`` and ``2``.

    Notes
    -----
    This converter prevents pandas from inferring compact values such as
    ``121`` as integers when loading TSV files. It returns the original text
    after applying :func:`parse_token_directions`.
    """
    parse_token_directions(value)
    return str(value)


def validate_boolean_values(
    values: pd.Series,
    *,
    field: str,
    optional: bool = False,
) -> None:
    """Validate a series of canonical booleans without coercion.

    Parameters
    ----------
    values
        Values to validate.
    field
        Field name used in error messages.
    optional
        Whether null values are allowed.

    Raises
    ------
    ValueError
        If a required value is null or a non-null value is not ``bool`` or
        ``numpy.bool_``. Numeric and string stand-ins are not accepted.
    """
    for value in values:
        if value is pd.NA or value is None or (
            isinstance(value, float) and pd.isna(value)
        ):
            if optional:
                continue
            raise ValueError(f"{field} must be boolean")
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{field} must be boolean")


def _parse_sequence_cell(
    value: object,
    *,
    field: str,
    item_type: type,
    optional: bool = False,
) -> list | None:
    """Deserialize one TSV cell into a typed list.

    Parameters
    ----------
    value
        Native sequence, serialized Python list or tuple, or optional null.
    field
        Field name used in errors.
    item_type
        Callable type applied to every sequence item.
    optional
        Whether null and blank cells return ``None``.

    Returns
    -------
    list or None
        Typed sequence, or ``None`` for an allowed missing value.

    Raises
    ------
    ValueError
        If the cell is required but missing, is not a list or tuple, cannot be
        parsed safely, or contains an item that cannot be converted.

    Notes
    -----
    Only Python list and tuple literals are accepted when the cell is text.
    Scalar values are rejected, including historical scalar-zero encodings.
    ``optional=True`` permits a null or blank cell and returns ``None``.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        if optional:
            return None
        raise ValueError(f"{field} must contain a sequence")
    if isinstance(value, str):
        if not value.strip() and optional:
            return None
        try:
            parsed = literal_eval(value)
        except (SyntaxError, ValueError) as error:
            raise ValueError(f"{field} must contain a sequence") from error
    else:
        parsed = value
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"{field} must contain a sequence")
    try:
        return [item_type(item) for item in parsed]
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{field} must contain only {item_type.__name__} values"
        ) from error


def _parse_boolean_cell(
    value: object,
    *,
    field: str,
    optional: bool = False,
) -> object:
    """Deserialize one TSV boolean without accepting numeric stand-ins.

    Parameters
    ----------
    value
        Native or serialized boolean value.
    field
        Field name used in errors.
    optional
        Whether a null or blank value becomes ``pandas.NA``.

    Returns
    -------
    bool or pandas.NA
        Canonical boolean representation.

    Raises
    ------
    ValueError
        If the value is missing when required or is not a supported boolean
        representation.

    Notes
    -----
    Native booleans and the case-insensitive strings ``true`` and ``false``
    are accepted because TSV storage is textual. Optional blank values become
    ``pandas.NA``; values such as ``0`` and ``1`` are rejected.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        if optional:
            return pd.NA
        raise ValueError(f"{field} must be boolean")
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized and optional:
            return pd.NA
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError(f"{field} must be boolean")


def behavior_table_converters() -> dict[str, object]:
    """Return strict pandas converters for serialized Stage 1 columns.

    Returns
    -------
    dict
        Column-to-callable mapping for sequence fields, canonical booleans,
        and compact token directions. The mapping is intended for
        :func:`pandas.read_csv` through the project table loader.
    """
    converters = {
        field: (
            lambda value, field=field, item_type=item_type: _parse_sequence_cell(
                value,
                field=field,
                item_type=item_type,
                optional=field == "sp_design_correct",
            )
        )
        for field, item_type in SEQUENCE_COLUMNS.items()
    }
    converters["token_log_short"] = lambda value: _parse_boolean_cell(
        value,
        field="token_log_short",
    )
    converters["isCorrect"] = lambda value: _parse_boolean_cell(
        value,
        field="isCorrect",
        optional=True,
    )
    converters["sTokenDirs"] = token_direction_converter
    return converters


def validate_behavior_table(table: pd.DataFrame) -> None:
    """Validate the complete canonical Stage 1 behavioral contract.

    Parameters
    ----------
    table
        Deserialized Stage 1 table for one behavioral run.

    Raises
    ------
    ValueError
        If required columns are absent; sequence or boolean types are invalid;
        compact directions are malformed; runtime token arrays and
        ``token_log_rows`` differ in length; the 14-row flag is inconsistent;
        trial indices are not positive and consecutive; never-started trials
        contain a go cue or choice; or chosen-trial events are out of order.

    Notes
    -----
    Validation does not repair values, reconstruct missing arrays, or discard
    trials. ``sp_design_correct`` is the only optional sequence because a
    correct target may be unavailable in the source trial.
    """
    missing = [column for column in TRIAL_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(f"Behavior table is missing required columns: {missing}")

    for field in SEQUENCE_COLUMNS:
        for row_index, value in table[field].items():
            if field == "sp_design_correct" and (
                value is None or (isinstance(value, float) and pd.isna(value))
            ):
                continue
            if not isinstance(value, (list, tuple)):
                raise ValueError(
                    f"{field} must contain a sequence at row {row_index}"
                )

    validate_boolean_values(table["token_log_short"], field="token_log_short")
    if "isCorrect" in table.columns:
        validate_boolean_values(
            table["isCorrect"],
            field="isCorrect",
            optional=True,
        )

    for row_index, value in table["sTokenDirs"].items():
        try:
            parse_token_directions(value)
        except ValueError as error:
            raise ValueError(
                f"Invalid sTokenDirs at row {row_index}: {error}"
            ) from error

    runtime_fields = ("nTokenNum", "nTokenDir", "tTime", "nProb")
    for row_index, row in table.iterrows():
        lengths = {field: len(row[field]) for field in runtime_fields}
        lengths["token_log_rows"] = int(row["token_log_rows"])
        if len(set(lengths.values())) != 1:
            raise ValueError(
                "Runtime token fields must have equal lengths at row "
                f"{row_index}: {lengths}"
            )
        if bool(row["token_log_short"]) != (int(row["token_log_rows"]) == 14):
            raise ValueError(
                "token_log_short must be true exactly for 14-row logs "
                f"at row {row_index}"
            )

    if not table.empty:
        actual_index = table["nTrialIndex"].astype(int).tolist()
        if actual_index[0] < 1:
            raise ValueError("nTrialIndex must start at 1 or above")
        expected_index = list(range(actual_index[0], actual_index[0] + len(table)))
        if actual_index != expected_index:
            raise ValueError(
                "nTrialIndex must be a gap-free consecutive sequence "
                f"(got {actual_index})"
            )

    never_started = table["nOutcome"] == OUTCOME_NEVER_STARTED
    invalid_never_started = never_started & (
        (table["tGO"] != 0) | (table["nChoiceMade"] != 0)
    )
    if invalid_never_started.any():
        rows = table.loc[invalid_never_started, "nTrialIndex"].tolist()
        raise ValueError(
            "OUTCOME_NEVER_STARTED trials must have tGO == 0 and "
            f"nChoiceMade == 0 (trials: {rows})"
        )

    chosen = table["nChoiceMade"] > 0
    invalid_order = chosen & (
        (table["tGO"] > table["tExitCenter"])
        | (table["tExitCenter"] > table["tEnterTarget"])
        | (table["tEnterTarget"] > table["tTrialEnd"])
    )
    if invalid_order.any():
        rows = table.loc[invalid_order, "nTrialIndex"].tolist()
        raise ValueError(f"Invalid event ordering for trials: {rows}")


def validate_trial_features(table: pd.DataFrame) -> None:
    """Validate the canonical analysis-ready trial-feature table.

    Parameters
    ----------
    table
        Trial-feature derivative produced by ``build_trial_features`` or loaded
        from disk.

    Raises
    ------
    ValueError
        If a required feature is absent, a boolean or token-direction field is
        non-canonical, or the MEG join key is duplicated.

    Notes
    -----
    The unique join key is ``subject``, ``condition``, ``run``, and
    ``run_trial_index``. ``isCorrect`` and ``sum_log_lr_saturated`` may be null;
    all other declared boolean flags must be present on every row.
    """
    missing = [
        column for column in TRIAL_FEATURE_COLUMNS if column not in table.columns
    ]
    if missing:
        raise ValueError(
            f"Trial-feature table is missing required columns: {missing}"
        )
    validate_boolean_values(
        table["isCorrect"],
        field="isCorrect",
        optional=True,
    )
    for field in (
        "sum_log_lr_saturated",
        "token_log_short",
        "is_started",
        "has_choice",
        "is_no_response",
        "dt_anticipation",
        "spd_available",
        "design_time_alignment_valid",
        "primary_analysis_eligible",
    ):
        validate_boolean_values(
            table[field],
            field=field,
            optional=field == "sum_log_lr_saturated",
        )
    for row_index, value in table["token_directions"].items():
        try:
            parse_token_directions(value)
        except ValueError as error:
            raise ValueError(
                f"Invalid token_directions at row {row_index}: {error}"
            ) from error
    join_key = ["subject", "condition", "run", "run_trial_index"]
    duplicated = table.duplicated(join_key, keep=False)
    if duplicated.any():
        rows = table.loc[duplicated, join_key].to_dict("records")
        raise ValueError(f"Trial-feature join keys must be unique: {rows}")
