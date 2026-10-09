"""Tests for meg_tokens.behavior.analyses.individual."""

import numpy as np
import pandas as pd
import pytest

from meg_tokens.behavior.analyses.individual import (
    comparison_statistics,
    individual_profile,
)


SUBJECTS = ["H01", "H02", "H03", "H04"]


def _subject_summary(**overrides) -> pd.DataFrame:
    columns = {
        "subject": SUBJECTS,
        "mean_fast_dt_ms": [900.0, 1000.0, 1100.0, 1200.0],
        "mean_slow_dt_ms": [1100.0, 1300.0, 1400.0, 1700.0],
        "percent_correct": [80.0, 78.0, 75.0, 70.0],
        "motor_baseline_ms": [340.0, 355.0, 360.0, 370.0],
        "mean_easy_dt_ms": [800.0, 900.0, 1000.0, 1100.0],
        "mean_ambiguous_dt_ms": [1100.0, 1250.0, 1350.0, 1600.0],
        "mean_misleading_dt_ms": [1050.0, 1200.0, 1300.0, 1500.0],
        "mean_easy_spd_all_logged": [0.90, 0.88, 0.86, 0.84],
        "mean_ambiguous_spd_all_logged": [0.72, 0.70, 0.69, 0.66],
        "mean_misleading_spd_all_logged": [0.65, 0.63, 0.60, 0.58],
    }
    columns.update(overrides)
    return pd.DataFrame(columns)


def _condition_table(column, values, condition="all", **extra):
    table = {
        "subject": SUBJECTS,
        "condition": [condition] * len(SUBJECTS),
        column: values,
    }
    table.update({key: [value] * len(SUBJECTS) for key, value in extra.items()})
    return pd.DataFrame(table)


# --- assembling the profile ---------------------------------------------------


def test_an_optional_input_joins_without_adding_or_removing_a_subject():
    neural = pd.DataFrame({"subject": SUBJECTS, "alpha_power": [1.0, 2.0, 3.0, 4.0]})

    # Neural metrics are joined unchanged for later analysis.
    profile = individual_profile(_subject_summary(), neural_metrics=neural)
    assert profile["alpha_power"].tolist() == [1.0, 2.0, 3.0, 4.0]

    lapses = pd.DataFrame({
        "subject": SUBJECTS[:2],
        "condition": ["all"] * 2,
        "lapse_rate": [0.1, 0.2],
    })

    profile = individual_profile(_subject_summary(), lapses=lapses)

    assert profile["subject"].tolist() == SUBJECTS
    assert profile["lapse_rate"].isna().tolist() == [False, False, True, True]


# --- cross-species comparison -------------------------------------------------


def test_the_criterion_slope_uses_first_order_sum_log_lr():
    """The comparable monkey measure is first-order chosen-target SumLogLR,
    not exact posterior log odds."""
    criterion = pd.concat(
        [
            _condition_table(
                "slope", [-0.05, -0.04, -0.03, -0.01], response="logged_spd"
            ),
            _condition_table(
                "slope",
                [-0.5, -0.4, -0.3, -0.1],
                response="chosen_sum_log_lr_first_order_at_decision",
            ),
        ],
        ignore_index=True,
    )

    statistics = comparison_statistics(
        _subject_summary(), criterion=criterion
    ).set_index("measure")
    row = statistics.loc["criterion_slope_sum_log_lr_per_second"]

    assert row["n_subjects"] == 4
    assert row["mean"] == pytest.approx(np.mean([-0.5, -0.4, -0.3, -0.1]))
