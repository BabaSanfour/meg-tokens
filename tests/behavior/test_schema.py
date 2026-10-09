"""Tests for meg_tokens.behavior.schema.

This module's job is rejecting malformed LabVIEW transcriptions, so most of it
is guards. They are grouped by the contract they enforce rather than split one
per rejected value: a table of bad inputs is both shorter and more complete
than a test each.
"""

import pytest

from meg_tokens.behavior.schema import validate_behavior_table, validate_trial_features

from tests.behavior.factories import stage1_run, trial_features


# --- Stage 1 table validation -------------------------------------------------


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


# --- trial-feature validation -------------------------------------------------


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
