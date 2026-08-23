"""Tests for meg_tokens.behavior.schema.

This module's job is rejecting malformed LabVIEW transcriptions, so most of it
is guards. They are grouped by the contract they enforce rather than split one
per rejected value: a table of bad inputs is both shorter and more complete
than a test each.
"""

import numpy as np
import pandas as pd
import pytest

from meg_tokens.behavior.schema import (
    CLASSIFICATION_COLUMNS,
    OUTCOME_NEVER_STARTED,
    RAW_TRIAL_COLUMNS,
    TRIAL_COLUMNS,
    TRIAL_FEATURE_COLUMNS,
    behavior_table_converters,
    parse_token_directions,
    token_direction_converter,
    validate_behavior_table,
    validate_boolean_values,
    validate_trial_features,
)

from tests.behavior.factories import raw_run, stage1_run, trial_features


def _convert(field, value):
    return behavior_table_converters()[field](value)


# --- token directions ---------------------------------------------------------


def test_token_directions_are_one_target_per_jump_and_stay_text():
    assert parse_token_directions("1221") == [1, 2, 2, 1]
    # A digits-only cell such as `121` must not become the integer 121.
    converted = token_direction_converter("121")
    assert converted == "121" and isinstance(converted, str)
    with pytest.raises(ValueError, match="only 1 and 2"):
        token_direction_converter("1231")


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (None, "non-empty string"),
        ("", "non-empty string"),
        (121, "non-empty string"),
        ("1231", "only 1 and 2"),
        ("1,2", "only 1 and 2"),
        ("12 21", "only 1 and 2"),
    ],
)
def test_parse_token_directions_rejects_impossible_targets(value, message):
    with pytest.raises(ValueError, match=message):
        parse_token_directions(value)


# --- canonical booleans and sequence converters -------------------------------


def test_a_boolean_field_accepts_only_real_booleans_and_nulls_only_when_optional():
    validate_boolean_values(pd.Series([True, False, np.bool_(True)]), field="flag")
    for value in (0, 1, "true", "False", 1.0):
        with pytest.raises(ValueError, match="flag must be boolean"):
            validate_boolean_values(pd.Series([True, value]), field="flag")

    nullable = pd.Series([True, pd.NA], dtype="boolean")
    validate_boolean_values(nullable, field="flag", optional=True)
    with pytest.raises(ValueError, match="flag must be boolean"):
        validate_boolean_values(nullable, field="flag")


def test_the_converters_deserialize_every_sequence_column_and_reject_scalars():
    assert _convert("nTokenNum", "[1, 2]") == [1, 2]
    assert _convert("nTokenDir", "[1, 2]") == [1, 2]
    assert _convert("tTime", "[1100, 1300]") == [1100.0, 1300.0]
    assert _convert("nProb", "[0.6, 0.8]") == [0.6, 0.8]
    assert _convert("sp_design_correct", "[0.6, 0.8]") == [0.6, 0.8]
    assert _convert("nTokenNum", "[]") == []

    # A trial can lack a correct target, but never its runtime token log.
    assert _convert("sp_design_correct", "") is None
    assert _convert("sp_design_correct", float("nan")) is None
    for value in ("", "0", "0.6", "nan"):
        with pytest.raises(ValueError, match="nProb must contain a sequence"):
            _convert("nProb", value)
    with pytest.raises(ValueError, match="nTokenNum must contain only int values"):
        _convert("nTokenNum", "['a']")


def test_boolean_converters_accept_serialized_text_but_not_digits():
    assert _convert("token_log_short", "true") is True
    assert _convert("token_log_short", "False") is False
    assert _convert("isCorrect", "") is pd.NA
    for value in ("1", ""):
        with pytest.raises(ValueError, match="token_log_short must be boolean"):
            _convert("token_log_short", value)


# --- Stage 1 table validation -------------------------------------------------


def test_the_stage1_contract_demands_every_declared_column_by_name():
    table = stage1_run(raw_rts=[900.0, 1100.0])
    validate_behavior_table(table)

    with pytest.raises(ValueError, match=r"missing required columns: \['tGO', 'nProb'\]"):
        validate_behavior_table(table.drop(columns=["nProb", "tGO"]))

    # The declared contract and the validator must not drift apart.
    one_row = stage1_run(raw_rts=[900.0])
    assert set(TRIAL_COLUMNS) <= set(one_row.columns)
    for column in TRIAL_COLUMNS:
        with pytest.raises(ValueError, match="missing required columns"):
            validate_behavior_table(one_row.drop(columns=[column]))


def test_the_raw_contract_is_the_stage1_contract_minus_the_classification_only():
    assert set(TRIAL_COLUMNS) - set(RAW_TRIAL_COLUMNS) == set(CLASSIFICATION_COLUMNS)
    assert len(RAW_TRIAL_COLUMNS) == len(TRIAL_COLUMNS) - len(CLASSIFICATION_COLUMNS)

    # A staged raw table has not been through classification, so requiring
    # those columns would reject a correct file.
    validate_behavior_table(raw_run(), require_classification=False)
    for column in CLASSIFICATION_COLUMNS:
        table = stage1_run(raw_rts=[900.0]).drop(columns=[column])
        validate_behavior_table(table, require_classification=False)
        with pytest.raises(ValueError, match=column):
            validate_behavior_table(table)

    # Every other rule concerns a transcribed field, so it applies to a raw
    # table exactly as it does to a derivative.
    malformed = raw_run()
    malformed.at[0, "nProb"] = 0.6
    with pytest.raises(ValueError, match="nProb must contain a sequence at row 0"):
        validate_behavior_table(malformed, require_classification=False)


def test_the_runtime_token_log_must_be_internally_consistent_and_flagged_when_short():
    scalar = stage1_run(raw_rts=[900.0])
    scalar.at[0, "nProb"] = 0.6
    with pytest.raises(ValueError, match="nProb must contain a sequence at row 0"):
        validate_behavior_table(scalar)

    ragged = stage1_run(raw_rts=[900.0])
    ragged.at[0, "nProb"] = ragged.at[0, "nProb"][:-1]
    with pytest.raises(ValueError, match="Runtime token fields must have equal lengths"):
        validate_behavior_table(ragged)

    mislabelled = stage1_run(raw_rts=[900.0])
    mislabelled["token_log_short"] = [True]
    with pytest.raises(
        ValueError, match="token_log_short must be true exactly for 14-row logs"
    ):
        validate_behavior_table(mislabelled)

    short = stage1_run(raw_rts=[900.0], probabilities=[0.6] * 14)
    assert short["token_log_short"].tolist() == [True]
    validate_behavior_table(short)

    bad_directions = stage1_run(raw_rts=[900.0, 1100.0])
    bad_directions.at[1, "sTokenDirs"] = "1231"
    with pytest.raises(ValueError, match="Invalid sTokenDirs at row 1"):
        validate_behavior_table(bad_directions)


def test_trial_index_may_start_above_one_but_must_not_skip():
    """nTrialIndex counts the whole session, so a run can begin part-way in."""
    validate_behavior_table(stage1_run(raw_rts=[900.0, 1100.0, 1000.0], first_trial_index=7))

    with pytest.raises(ValueError, match="must start at 1 or above"):
        validate_behavior_table(stage1_run(raw_rts=[900.0], first_trial_index=0))

    gapped = stage1_run(raw_rts=[900.0, 1100.0, 1000.0], first_trial_index=5)
    gapped.at[2, "nTrialIndex"] = 8
    with pytest.raises(ValueError, match="gap-free consecutive sequence"):
        validate_behavior_table(gapped)


def test_event_ordering_is_enforced_on_chosen_trials_and_nowhere_else():
    # No-response trials leave zeroed timestamps behind; they are not errors.
    validate_behavior_table(stage1_run(raw_rts=[900.0, None]))

    for column, value in (("tGO", 5000), ("tExitCenter", 5000), ("tEnterTarget", 5000)):
        table = stage1_run(raw_rts=[900.0])
        table.at[0, column] = value
        with pytest.raises(ValueError, match=r"Invalid event ordering for trials: \[1\]"):
            validate_behavior_table(table)

    never_started = stage1_run(raw_rts=[900.0, None], outcomes=[0, OUTCOME_NEVER_STARTED])
    never_started.at[1, "tGO"] = 0
    validate_behavior_table(never_started)
    for column, value in (("tGO", 1000), ("nChoiceMade", 1)):
        table = never_started.copy()
        table.at[1, column] = value
        with pytest.raises(ValueError, match="must have tGO == 0 and nChoiceMade == 0"):
            validate_behavior_table(table)


# --- trial-feature validation -------------------------------------------------


def test_the_feature_contract_demands_every_declared_column_and_canonical_flags():
    validate_trial_features(trial_features(dt_ms=[900.0, 1000.0]))

    one_row = trial_features(dt_ms=[900.0])
    assert set(TRIAL_FEATURE_COLUMNS) <= set(one_row.columns)
    for column in TRIAL_FEATURE_COLUMNS:
        with pytest.raises(ValueError, match="missing required columns"):
            validate_trial_features(one_row.drop(columns=[column]))

    for field in (
        "token_log_short", "is_started", "has_choice", "is_no_response",
        "dt_anticipation", "spd_available", "design_time_alignment_valid",
        "primary_analysis_eligible",
    ):
        table = trial_features(dt_ms=[900.0])
        table[field] = ["true"]
        with pytest.raises(ValueError, match=f"{field} must be boolean"):
            validate_trial_features(table)

    # Outcome flags are genuinely absent on some trials, so they are nullable.
    for field in ("isCorrect", "sum_log_lr_saturated"):
        table = trial_features(dt_ms=[900.0, 1000.0])
        table[field] = pd.array([True, None], dtype="boolean")
        validate_trial_features(table)

    bad_directions = trial_features(dt_ms=[900.0, 1000.0])
    bad_directions.at[1, "token_directions"] = "1231"
    with pytest.raises(ValueError, match="Invalid token_directions at row 1"):
        validate_trial_features(bad_directions)


def test_the_meg_join_key_identifies_exactly_one_trial_across_runs_and_conditions():
    with pytest.raises(ValueError, match="join keys must be unique"):
        validate_trial_features(
            trial_features(
                subject=["H01", "H01"], condition=["Fast", "Fast"],
                run=[1, 1], run_trial_index=[3, 3],
            )
        )

    # The same trial number in another run or condition is a different trial.
    validate_trial_features(
        trial_features(
            subject=["H01"] * 4,
            condition=["Fast", "Fast", "Slow", "Slow"],
            run=[1, 2, 1, 2],
            run_trial_index=[3, 3, 3, 3],
        )
    )
