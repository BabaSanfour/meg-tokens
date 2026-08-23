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
    subject_condition_groups,
    task_trials,
)

from tests.behavior.factories import trial_features


def test_require_columns_names_the_gaps_and_the_command_that_fills_them():
    table = trial_features(dt_ms=[900.0])
    require_columns(table, ["subject", "dt_ms"])

    with pytest.raises(ValueError) as error:
        require_columns(table, ["dt_ms", "absent", "also_absent"])
    message = str(error.value)
    assert "['absent', 'also_absent']" in message
    assert "behavior analyze" in message


def test_each_selector_applies_exactly_the_rule_it_names():
    # started_trials drops only the never-started sentinel.
    started = started_trials(
        trial_features(
            dt_ms=[900.0, float("nan"), 1000.0],
            nOutcome=[0, OUTCOME_NEVER_STARTED, 7006],
        )
    )
    assert started["nOutcome"].tolist() == [0, 7006]

    # Eligibility is decided once by the feature table, not re-derived here.
    eligible = task_trials(
        trial_features(
            condition=["RT", "Fast", "Fast", "Slow"],
            dt_ms=[float("nan"), 900.0, float("nan"), 1300.0],
            has_choice=[True, True, False, True],
            primary_analysis_eligible=[False, True, False, True],
        )
    )
    assert eligible["condition"].tolist() == ["Fast", "Slow"]
    assert eligible["dt_ms"].tolist() == [900.0, 1300.0]

    with pytest.raises(ValueError, match="primary_analysis_eligible"):
        task_trials(trial_features(dt_ms=[900.0]).drop(columns=["primary_analysis_eligible"]))

    # classified_trials keeps the three design classes and applies eligibility.
    classified = classified_trials(
        trial_features(
            trial_class=[1, 2, 3, 0],
            trial_class_name=["easy", "ambiguous", "misleading", "unclassified"],
            dt_ms=[900.0, 1200.0, 1100.0, 1000.0],
        )
    )
    assert classified["trial_class"].tolist() == [1, 2, 3]
    assert classified_trials(
        trial_features(
            trial_class=[1, 2], dt_ms=[900.0, 1200.0],
            primary_analysis_eligible=[True, False],
        )
    )["trial_class"].tolist() == [1]

    # Never-started rows and the RT baseline run are not lapses.
    lapses = lapse_trials(
        trial_features(
            condition=["Fast", "Fast", "RT", "Slow"],
            run_trial_index=[1, 2, 1, 1],
            is_started=[True, False, True, True],
            has_choice=[False, False, False, False],
            primary_analysis_eligible=[False, False, False, False],
            dt_ms=[float("nan")] * 4,
        )
    )
    assert lapses["condition"].tolist() == ["Fast", "Slow"]
    assert lapses["run_trial_index"].tolist() == [1, 1]

    disjoint = trial_features(
        condition=["Fast", "Fast"], has_choice=[True, False],
        primary_analysis_eligible=[True, False], dt_ms=[900.0, float("nan")],
    )
    assert set(task_trials(disjoint).index) & set(lapse_trials(disjoint).index) == set()


@pytest.mark.parametrize("select", [started_trials, task_trials, classified_trials, lapse_trials])
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


def test_subject_condition_groups_yield_the_pooled_row_and_every_declared_condition():
    features = trial_features(
        subject=["H01"] * 3 + ["H02"] * 2,
        condition=["Fast", "Fast", "Slow", "Fast", "Fast"],
        run=[1, 1, 2, 1, 1],
        run_trial_index=[1, 2, 1, 1, 2],
        dt_ms=[900.0, 1000.0, 1300.0, 950.0, 1050.0],
    )

    groups = list(subject_condition_groups(features))
    assert [(subject, name) for subject, name, _ in groups] == [
        ("H01", "all"), ("H01", "fast"), ("H01", "slow"),
        ("H02", "all"), ("H02", "fast"), ("H02", "slow"),
    ]
    # "all" is every task trial of that subject, not the concatenation of the
    # declared conditions, and an empty cell is reported rather than skipped.
    assert [len(frame) for _, _, frame in groups] == [3, 2, 1, 2, 2, 0]

    # skip_empty drops the cell the subject never ran.
    kept = list(subject_condition_groups(features, skip_empty=True))
    assert [(subject, name) for subject, name, _ in kept] == [
        ("H01", "all"), ("H01", "fast"), ("H01", "slow"),
        ("H02", "all"), ("H02", "fast"),
    ]


def test_finite_values_keeps_only_usable_observations():
    values = finite_values(pd.Series([900.0, None, np.inf, -np.inf, "not a number", 1100.0]))
    assert values.tolist() == [900.0, 1100.0]
    assert values.dtype == float
    assert finite_values(pd.Series([], dtype=float)).size == 0


def test_the_declared_vocabularies_agree_with_each_other():
    assert CLASS_NAMES == {1: "easy", 2: "ambiguous", 3: "misleading"}
    assert CLASS_CODES == {name: code for code, name in CLASS_NAMES.items()}
    # The RT baseline run is not a task condition.
    assert TASK_CONDITIONS == ("Fast", "Slow")
    assert set(LAPSE_OUTCOMES) <= set(OUTCOME_LABELS)
    assert OUTCOME_NEVER_STARTED not in LAPSE_OUTCOMES
