"""Tests for meg_tokens.behavior.tables."""

import numpy as np
import pandas as pd
import pytest

from meg_tokens.behavior.schema import TRIAL_COLUMNS
from meg_tokens.behavior.tables import (
    add_run_metadata,
    read_behavior_table,
    read_trial_features,
)
from meg_tokens.behavior.tdms import TdmsRunInfo
from meg_tokens.io import save_table

from tests.behavior.factories import stage1_run, trial_features


RUN_INFO = TdmsRunInfo(subject="H01", condition="Slow", run="2", date="180131")


def _parsed_trials() -> pd.DataFrame:
    """Parser output: trial columns only, without run identity."""
    return stage1_run(
        raw_rts=[900.0, 1100.0, None],
        choices=[1, 2, 1],
        correct_choices=[1, 1, 1],
    )[TRIAL_COLUMNS]


# --- run metadata -------------------------------------------------------------


def test_add_run_metadata_puts_run_identity_in_front_of_the_trials():
    enriched = add_run_metadata(_parsed_trials(), RUN_INFO, "H01Slow2_180131.tdms")

    assert list(enriched.columns[:4]) == [
        "subject", "condition", "run", "source_file",
    ]
    assert enriched["subject"].tolist() == ["H01"] * 3
    assert enriched["condition"].tolist() == ["Slow"] * 3
    assert enriched["run"].tolist() == [2] * 3
    assert enriched["source_file"].tolist() == ["H01Slow2_180131.tdms"] * 3


def test_raw_rt_is_the_go_cue_to_target_latency():
    enriched = add_run_metadata(_parsed_trials(), RUN_INFO, "H01Slow2_180131.tdms")

    # tEnterTarget - tGO, in milliseconds.
    assert enriched["rawRT"].tolist()[:2] == [900.0, 1100.0]


def test_response_fields_stay_undefined_when_no_choice_was_made():
    enriched = add_run_metadata(_parsed_trials(), RUN_INFO, "H01Slow2_180131.tdms")

    assert pd.isna(enriched.loc[2, "rawRT"])
    assert pd.isna(enriched.loc[2, "isCorrect"])


def test_correctness_compares_the_chosen_and_correct_targets():
    enriched = add_run_metadata(_parsed_trials(), RUN_INFO, "H01Slow2_180131.tdms")

    # Trial 2 chose target 2 while target 1 was correct.
    assert enriched["isCorrect"].tolist()[:2] == [True, False]


def test_add_run_metadata_leaves_the_parsed_table_untouched():
    parsed = _parsed_trials()

    add_run_metadata(parsed, RUN_INFO, "H01Slow2_180131.tdms")

    assert "subject" not in parsed.columns
    assert "rawRT" not in parsed.columns


# --- reading Stage 1 derivatives ----------------------------------------------


def test_read_behavior_table_restores_sequences_and_booleans(tmp_path):
    table = stage1_run(raw_rts=[900.0, 1100.0])
    path = tmp_path / "behavior.tsv"
    save_table(path, table)

    loaded = read_behavior_table(path)

    assert loaded.at[0, "nProb"] == table.at[0, "nProb"]
    assert loaded.at[0, "tTime"] == table.at[0, "tTime"]
    assert loaded.at[0, "nTokenNum"] == table.at[0, "nTokenNum"]
    assert loaded.at[0, "sp_design_correct"] == table.at[0, "sp_design_correct"]
    log_short = loaded.at[0, "token_log_short"]
    assert isinstance(log_short, (bool, np.bool_)) and not log_short


def test_read_behavior_table_keeps_an_empty_token_log_empty(tmp_path):
    table = stage1_run(raw_rts=[900.0, None], probabilities=[])
    path = tmp_path / "behavior.tsv"
    save_table(path, table)

    loaded = read_behavior_table(path)

    assert loaded.at[1, "nProb"] == []
    assert loaded.at[1, "token_log_rows"] == 0


def test_read_behavior_table_preserves_a_digits_only_direction_string(tmp_path):
    table = stage1_run(raw_rts=[900.0], token_directions="121", probabilities=[0.6])
    path = tmp_path / "behavior.tsv"
    save_table(path, table)

    loaded = read_behavior_table(path)

    assert loaded.at[0, "sTokenDirs"] == "121"


def test_read_behavior_table_rejects_a_serialized_scalar_sequence(tmp_path):
    table = stage1_run(raw_rts=[900.0])
    table.at[0, "nProb"] = 0
    path = tmp_path / "behavior.tsv"
    save_table(path, table)

    with pytest.raises(ValueError, match="nProb must contain a sequence"):
        read_behavior_table(path)


def test_read_behavior_table_rejects_a_numeric_boolean(tmp_path):
    table = stage1_run(raw_rts=[900.0])
    table["token_log_short"] = [0]
    path = tmp_path / "behavior.tsv"
    save_table(path, table)

    with pytest.raises(ValueError, match="token_log_short must be boolean"):
        read_behavior_table(path)


def test_read_behavior_table_applies_the_stage1_contract(tmp_path):
    table = stage1_run(raw_rts=[900.0])
    table.at[0, "tGO"] = 9000
    path = tmp_path / "behavior.tsv"
    save_table(path, table)

    with pytest.raises(ValueError, match="Invalid event ordering"):
        read_behavior_table(path)


# --- reading trial features ---------------------------------------------------


def test_read_trial_features_round_trips_the_canonical_table(tmp_path):
    table = trial_features(dt_ms=[900.0, 1000.0], logged_spd=[0.8, 0.9])
    path = tmp_path / "features.tsv"
    save_table(path, table)

    loaded = read_trial_features(path)

    assert loaded["dt_ms"].tolist() == [900.0, 1000.0]
    assert loaded["logged_spd"].tolist() == [0.8, 0.9]
    assert loaded["is_started"].tolist() == [True, True]


def test_read_trial_features_keeps_compact_directions_as_text(tmp_path):
    table = trial_features(dt_ms=[900.0], token_directions=["121"])
    path = tmp_path / "features.tsv"
    save_table(path, table)

    loaded = read_trial_features(path)

    assert loaded.at[0, "token_directions"] == "121"


def test_read_trial_features_applies_the_feature_contract(tmp_path):
    table = trial_features(
        subject=["H01", "H01"],
        run_trial_index=[3, 3],
    )
    path = tmp_path / "features.tsv"
    save_table(path, table)

    with pytest.raises(ValueError, match="join keys must be unique"):
        read_trial_features(path)
