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
        raw_rts=[900.0, 1100.0, None], choices=[1, 2, 1], correct_choices=[1, 1, 1]
    )[TRIAL_COLUMNS]


def test_add_run_metadata_prepends_identity_and_derives_the_response_fields():
    parsed = _parsed_trials()
    enriched = add_run_metadata(parsed, RUN_INFO, "H01Slow2_180131.tdms")

    assert list(enriched.columns[:4]) == ["subject", "condition", "run", "source_file"]
    assert enriched["subject"].tolist() == ["H01"] * 3
    assert enriched["condition"].tolist() == ["Slow"] * 3
    assert enriched["run"].tolist() == [2] * 3
    assert enriched["source_file"].tolist() == ["H01Slow2_180131.tdms"] * 3

    # rawRT is tEnterTarget - tGO in ms, and correctness compares the chosen
    # target with the correct one. Trial 2 chose target 2 while 1 was correct.
    assert enriched["rawRT"].tolist()[:2] == [900.0, 1100.0]
    assert enriched["isCorrect"].tolist()[:2] == [True, False]
    # Both stay undefined when no choice was made.
    assert pd.isna(enriched.loc[2, "rawRT"])
    assert pd.isna(enriched.loc[2, "isCorrect"])

    # The parsed table is not modified in place.
    assert "subject" not in parsed.columns
    assert "rawRT" not in parsed.columns


def test_reading_a_stage1_derivative_restores_its_types_and_applies_the_contract(tmp_path):
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

    empty_log = tmp_path / "empty.tsv"
    save_table(empty_log, stage1_run(raw_rts=[900.0, None], probabilities=[]))
    empty = read_behavior_table(empty_log)
    assert empty.at[1, "nProb"] == []
    assert empty.at[1, "token_log_rows"] == 0

    # A digits-only direction cell must survive as text, not become an integer.
    digits = tmp_path / "digits.tsv"
    save_table(digits, stage1_run(raw_rts=[900.0], token_directions="121", probabilities=[0.6]))
    assert read_behavior_table(digits).at[0, "sTokenDirs"] == "121"


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (lambda t: t.assign(nProb=[0] * len(t)), "nProb must contain a sequence"),
        (lambda t: t.assign(token_log_short=[0] * len(t)), "token_log_short must be boolean"),
        (lambda t: t.assign(tGO=[9000] * len(t)), "Invalid event ordering"),
    ],
)
def test_reading_a_stage1_derivative_rejects_a_file_that_breaks_the_contract(
    tmp_path, corrupt, message
):
    path = tmp_path / "behavior.tsv"
    save_table(path, corrupt(stage1_run(raw_rts=[900.0])))
    with pytest.raises(ValueError, match=message):
        read_behavior_table(path)


def test_reading_trial_features_round_trips_the_table_and_applies_its_contract(tmp_path):
    path = tmp_path / "features.tsv"
    save_table(path, trial_features(dt_ms=[900.0, 1000.0], logged_spd=[0.8, 0.9]))

    loaded = read_trial_features(path)
    assert loaded["dt_ms"].tolist() == [900.0, 1000.0]
    assert loaded["logged_spd"].tolist() == [0.8, 0.9]
    assert loaded["is_started"].tolist() == [True, True]

    digits = tmp_path / "digits.tsv"
    save_table(digits, trial_features(dt_ms=[900.0], token_directions=["121"]))
    assert read_trial_features(digits).at[0, "token_directions"] == "121"

    duplicated = tmp_path / "duplicated.tsv"
    save_table(duplicated, trial_features(subject=["H01", "H01"], run_trial_index=[3, 3]))
    with pytest.raises(ValueError, match="join keys must be unique"):
        read_trial_features(duplicated)
