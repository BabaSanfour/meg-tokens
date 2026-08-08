"""Tests for meg_tokens.behavior.trials."""

import numpy as np
import pandas as pd
import pytest

from meg_tokens.behavior.schema import OUTCOME_NEVER_STARTED
from meg_tokens.behavior.trials import (
    CLASS_CODES,
    CLASS_NAMES,
    LAPSE_OUTCOMES,
    OUTCOME_LABELS,
    TASK_CONDITIONS,
    classified_trials,
    finite_values,
    lapse_trials,
    require_columns,
    started_trials,
    task_trials,
)

from tests.behavior.factories import trial_features


def test_require_columns_accepts_a_table_that_has_them():
    require_columns(trial_features(dt_ms=[900.0]), ["subject", "dt_ms"])


def test_require_columns_names_the_gaps_and_the_command_that_fills_them():
    table = trial_features(dt_ms=[900.0])

    with pytest.raises(ValueError) as error:
        require_columns(table, ["dt_ms", "absent", "also_absent"])

    message = str(error.value)
    assert "['absent', 'also_absent']" in message
    assert "behavior analyze" in message


# --- selection ----------------------------------------------------------------


def test_started_trials_drops_only_the_never_started_sentinel():
    features = trial_features(
        dt_ms=[900.0, float("nan"), 1000.0],
        nOutcome=[0, OUTCOME_NEVER_STARTED, 7006],
    )

    started = started_trials(features)

    assert started["nOutcome"].tolist() == [0, 7006]


def test_task_trials_returns_the_declared_eligibility_view():
    """Eligibility is decided once by the feature table, not re-derived here."""
    features = trial_features(
        condition=["RT", "Fast", "Fast", "Slow"],
        dt_ms=[float("nan"), 900.0, float("nan"), 1300.0],
        has_choice=[True, True, False, True],
        primary_analysis_eligible=[False, True, False, True],
    )

    eligible = task_trials(features)

    assert eligible["condition"].tolist() == ["Fast", "Slow"]
    assert eligible["dt_ms"].tolist() == [900.0, 1300.0]


def test_task_trials_reports_a_feature_table_that_predates_the_flag():
    features = trial_features(dt_ms=[900.0]).drop(
        columns=["primary_analysis_eligible"]
    )

    with pytest.raises(ValueError, match="primary_analysis_eligible"):
        task_trials(features)


def test_classified_trials_keeps_the_three_design_classes_only():
    features = trial_features(
        trial_class=[1, 2, 3, 0],
        trial_class_name=["easy", "ambiguous", "misleading", "unclassified"],
        dt_ms=[900.0, 1200.0, 1100.0, 1000.0],
    )

    classified = classified_trials(features)

    assert classified["trial_class"].tolist() == [1, 2, 3]


def test_classified_trials_also_applies_the_eligibility_rule():
    features = trial_features(
        trial_class=[1, 2],
        dt_ms=[900.0, 1200.0],
        primary_analysis_eligible=[True, False],
    )

    assert classified_trials(features)["trial_class"].tolist() == [1]


def test_lapse_trials_selects_started_task_trials_without_a_choice():
    features = trial_features(
        condition=["Fast", "Fast", "RT", "Slow"],
        run_trial_index=[1, 2, 1, 1],
        is_started=[True, False, True, True],
        has_choice=[False, False, False, False],
        primary_analysis_eligible=[False, False, False, False],
        dt_ms=[float("nan")] * 4,
    )

    lapses = lapse_trials(features)

    # Never-started rows and the RT baseline run are not lapses.
    assert lapses["condition"].tolist() == ["Fast", "Slow"]
    assert lapses["run_trial_index"].tolist() == [1, 1]


def test_lapse_trials_and_task_trials_are_disjoint():
    features = trial_features(
        condition=["Fast", "Fast"],
        has_choice=[True, False],
        primary_analysis_eligible=[True, False],
        dt_ms=[900.0, float("nan")],
    )

    assert set(task_trials(features).index) & set(lapse_trials(features).index) == set()


@pytest.mark.parametrize(
    "select",
    [started_trials, task_trials, classified_trials, lapse_trials],
)
def test_selections_return_an_independent_copy(select):
    # One eligible choice trial and one lapse, so every selector returns a row.
    features = trial_features(
        dt_ms=[900.0, float("nan")],
        has_choice=[True, False],
        primary_analysis_eligible=[True, False],
    )

    selected = select(features)
    assert not selected.empty
    selected.loc[selected.index[0], "dt_ms"] = -1.0

    assert features["dt_ms"].tolist()[0] == 900.0


# --- numeric extraction -------------------------------------------------------


def test_finite_values_keeps_only_usable_observations():
    values = finite_values(
        pd.Series([900.0, None, np.inf, -np.inf, "not a number", 1100.0])
    )

    assert values.tolist() == [900.0, 1100.0]
    assert values.dtype == float


def test_finite_values_returns_an_empty_array_rather_than_failing():
    assert finite_values(pd.Series([], dtype=float)).size == 0


# --- declared vocabularies ----------------------------------------------------


def test_class_names_and_codes_are_mutual_inverses():
    assert CLASS_NAMES == {1: "easy", 2: "ambiguous", 3: "misleading"}
    assert CLASS_CODES == {name: code for code, name in CLASS_NAMES.items()}


def test_task_conditions_exclude_the_rt_baseline_run():
    assert TASK_CONDITIONS == ("Fast", "Slow")


def test_lapse_outcomes_are_labelled_started_trial_outcomes():
    assert set(LAPSE_OUTCOMES) <= set(OUTCOME_LABELS)
    assert OUTCOME_NEVER_STARTED not in LAPSE_OUTCOMES
