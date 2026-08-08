"""Tests for meg_tokens.behavior.schema."""

import numpy as np
import pandas as pd
import pytest

from meg_tokens.behavior.schema import (
    OUTCOME_NEVER_STARTED,
    TRIAL_COLUMNS,
    TRIAL_FEATURE_COLUMNS,
    behavior_table_converters,
    parse_token_directions,
    token_direction_converter,
    validate_behavior_table,
    validate_boolean_values,
    validate_trial_features,
)

from tests.behavior.factories import stage1_run, trial_features


# --- token directions ---------------------------------------------------------


def test_parse_token_directions_returns_one_target_per_jump():
    assert parse_token_directions("1221") == [1, 2, 2, 1]


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


def test_token_direction_converter_keeps_the_compact_string_as_text():
    """A digits-only cell such as ``121`` must not become the integer 121."""
    converted = token_direction_converter("121")

    assert converted == "121"
    assert isinstance(converted, str)


def test_token_direction_converter_rejects_a_malformed_cell():
    with pytest.raises(ValueError, match="only 1 and 2"):
        token_direction_converter("1231")


# --- canonical booleans -------------------------------------------------------


def test_validate_boolean_values_accepts_python_and_numpy_booleans():
    validate_boolean_values(
        pd.Series([True, False, np.bool_(True)]),
        field="flag",
    )


@pytest.mark.parametrize("value", [0, 1, "true", "False", 1.0])
def test_validate_boolean_values_rejects_numeric_and_string_stand_ins(value):
    with pytest.raises(ValueError, match="flag must be boolean"):
        validate_boolean_values(pd.Series([True, value]), field="flag")


def test_validate_boolean_values_allows_nulls_only_when_declared_optional():
    values = pd.Series([True, pd.NA], dtype="boolean")

    validate_boolean_values(values, field="flag", optional=True)
    with pytest.raises(ValueError, match="flag must be boolean"):
        validate_boolean_values(values, field="flag")


# --- Stage 1 converters -------------------------------------------------------


def _convert(field, value):
    return behavior_table_converters()[field](value)


def test_converters_deserialize_every_declared_sequence_column():
    assert _convert("nTokenNum", "[1, 2]") == [1, 2]
    assert _convert("nTokenDir", "[1, 2]") == [1, 2]
    assert _convert("tTime", "[1100, 1300]") == [1100.0, 1300.0]
    assert _convert("nProb", "[0.6, 0.8]") == [0.6, 0.8]
    assert _convert("sp_design_correct", "[0.6, 0.8]") == [0.6, 0.8]
    assert _convert("nTokenNum", "[]") == []


def test_only_the_correct_target_profile_may_be_absent():
    """A trial can lack a correct target, but never its runtime token log."""
    assert _convert("sp_design_correct", "") is None
    assert _convert("sp_design_correct", float("nan")) is None
    with pytest.raises(ValueError, match="nProb must contain a sequence"):
        _convert("nProb", "")


@pytest.mark.parametrize("value", ["0", "0.6", "nan"])
def test_sequence_converters_reject_scalar_cells(value):
    with pytest.raises(ValueError, match="nProb must contain a sequence"):
        _convert("nProb", value)


def test_sequence_converters_reject_items_of_the_wrong_type():
    with pytest.raises(ValueError, match="nTokenNum must contain only int values"):
        _convert("nTokenNum", "['a']")


def test_boolean_converters_accept_serialized_text_but_not_digits():
    assert _convert("token_log_short", "true") is True
    assert _convert("token_log_short", "False") is False
    assert _convert("isCorrect", "") is pd.NA
    with pytest.raises(ValueError, match="token_log_short must be boolean"):
        _convert("token_log_short", "1")
    with pytest.raises(ValueError, match="token_log_short must be boolean"):
        _convert("token_log_short", "")


# --- Stage 1 table validation -------------------------------------------------


def test_validate_behavior_table_accepts_the_canonical_run():
    validate_behavior_table(stage1_run(raw_rts=[900.0, 1100.0]))


def test_validate_behavior_table_names_every_missing_column():
    table = stage1_run(raw_rts=[900.0]).drop(columns=["nProb", "tGO"])

    with pytest.raises(ValueError, match=r"missing required columns: \['tGO', 'nProb'\]"):
        validate_behavior_table(table)


def test_validate_behavior_table_declares_the_full_stage1_column_set():
    """The declared contract and the validator must not drift apart."""
    table = stage1_run(raw_rts=[900.0])

    assert set(TRIAL_COLUMNS) <= set(table.columns)
    for column in TRIAL_COLUMNS:
        with pytest.raises(ValueError, match="missing required columns"):
            validate_behavior_table(table.drop(columns=[column]))


def test_validate_behavior_table_rejects_a_scalar_in_a_sequence_field():
    table = stage1_run(raw_rts=[900.0])
    table.at[0, "nProb"] = 0.6

    with pytest.raises(ValueError, match="nProb must contain a sequence at row 0"):
        validate_behavior_table(table)


def test_runtime_token_fields_must_agree_with_the_logged_row_count():
    table = stage1_run(raw_rts=[900.0])
    table.at[0, "nProb"] = table.at[0, "nProb"][:-1]

    with pytest.raises(ValueError, match="Runtime token fields must have equal lengths"):
        validate_behavior_table(table)


def test_short_log_flag_must_mark_exactly_the_fourteen_row_logs():
    complete = stage1_run(raw_rts=[900.0])
    complete["token_log_short"] = [True]

    with pytest.raises(ValueError, match="token_log_short must be true exactly for 14-row logs"):
        validate_behavior_table(complete)

    short = stage1_run(raw_rts=[900.0], probabilities=[0.6] * 14)
    assert short["token_log_short"].tolist() == [True]
    validate_behavior_table(short)


def test_validate_behavior_table_reports_a_malformed_direction_string_by_row():
    table = stage1_run(raw_rts=[900.0, 1100.0])
    table.at[1, "sTokenDirs"] = "1231"

    with pytest.raises(ValueError, match="Invalid sTokenDirs at row 1"):
        validate_behavior_table(table)


def test_trial_index_may_start_above_one_but_must_not_skip():
    """nTrialIndex counts the whole session, so a run can begin part-way in."""
    validate_behavior_table(
        stage1_run(raw_rts=[900.0, 1100.0, 1000.0], first_trial_index=7)
    )

    with pytest.raises(ValueError, match="must start at 1 or above"):
        validate_behavior_table(stage1_run(raw_rts=[900.0], first_trial_index=0))

    gapped = stage1_run(raw_rts=[900.0, 1100.0, 1000.0], first_trial_index=5)
    gapped.at[2, "nTrialIndex"] = 8
    with pytest.raises(ValueError, match="gap-free consecutive sequence"):
        validate_behavior_table(gapped)


def test_never_started_trials_carry_no_go_cue_and_no_choice():
    table = stage1_run(raw_rts=[900.0, None], outcomes=[0, OUTCOME_NEVER_STARTED])
    table.at[1, "tGO"] = 0

    validate_behavior_table(table)


@pytest.mark.parametrize(("column", "value"), [("tGO", 1000), ("nChoiceMade", 1)])
def test_never_started_trials_reject_recorded_activity(column, value):
    table = stage1_run(raw_rts=[900.0, None], outcomes=[0, OUTCOME_NEVER_STARTED])
    table.at[1, "tGO"] = 0
    table.at[1, column] = value

    with pytest.raises(ValueError, match="must have tGO == 0 and nChoiceMade == 0"):
        validate_behavior_table(table)


@pytest.mark.parametrize(
    ("column", "value"),
    [("tGO", 5000), ("tExitCenter", 5000), ("tEnterTarget", 5000)],
)
def test_chosen_trials_must_keep_their_events_in_order(column, value):
    table = stage1_run(raw_rts=[900.0])
    table.at[0, column] = value

    with pytest.raises(ValueError, match=r"Invalid event ordering for trials: \[1\]"):
        validate_behavior_table(table)


def test_event_ordering_is_not_checked_on_trials_without_a_choice():
    """No-response trials leave zeroed timestamps behind; they are not errors."""
    validate_behavior_table(stage1_run(raw_rts=[900.0, None]))


# --- trial-feature validation -------------------------------------------------


def test_validate_trial_features_accepts_the_canonical_feature_table():
    validate_trial_features(trial_features(dt_ms=[900.0, 1000.0]))


def test_validate_trial_features_requires_every_declared_feature():
    table = trial_features(dt_ms=[900.0])

    assert set(TRIAL_FEATURE_COLUMNS) <= set(table.columns)
    for column in TRIAL_FEATURE_COLUMNS:
        with pytest.raises(ValueError, match="missing required columns"):
            validate_trial_features(table.drop(columns=[column]))


@pytest.mark.parametrize(
    "field",
    [
        "token_log_short",
        "is_started",
        "has_choice",
        "is_no_response",
        "dt_anticipation",
        "spd_available",
        "design_time_alignment_valid",
        "primary_analysis_eligible",
    ],
)
def test_required_feature_flags_reject_serialized_booleans(field):
    table = trial_features(dt_ms=[900.0])
    table[field] = ["true"]

    with pytest.raises(ValueError, match=f"{field} must be boolean"):
        validate_trial_features(table)


@pytest.mark.parametrize("field", ["isCorrect", "sum_log_lr_saturated"])
def test_outcome_flags_may_be_missing_on_a_trial(field):
    table = trial_features(dt_ms=[900.0, 1000.0])
    table[field] = pd.array([True, None], dtype="boolean")

    validate_trial_features(table)


def test_validate_trial_features_reports_a_malformed_direction_string_by_row():
    table = trial_features(dt_ms=[900.0, 1000.0])
    table.at[1, "token_directions"] = "1231"

    with pytest.raises(ValueError, match="Invalid token_directions at row 1"):
        validate_trial_features(table)


def test_the_meg_join_key_must_identify_exactly_one_trial():
    table = trial_features(
        subject=["H01", "H01"],
        condition=["Fast", "Fast"],
        run=[1, 1],
        run_trial_index=[3, 3],
    )

    with pytest.raises(ValueError, match="join keys must be unique"):
        validate_trial_features(table)


def test_the_join_key_distinguishes_runs_and_conditions():
    """The same trial number in another run or condition is a different trial."""
    validate_trial_features(
        trial_features(
            subject=["H01"] * 4,
            condition=["Fast", "Fast", "Slow", "Slow"],
            run=[1, 2, 1, 2],
            run_trial_index=[3, 3, 3, 3],
        )
    )
