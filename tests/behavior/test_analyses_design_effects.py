"""Tests for meg_tokens.behavior.analyses.design_effects."""

import numpy as np
import pandas as pd
import pytest

from meg_tokens.behavior.analyses.design_effects import (
    choice_side_statistics,
    choice_side_summary,
    condition_class_cells,
    condition_class_statistics,
    condition_order_effects,
    condition_order_statistics,
    extreme_decision_times,
    lapse_statistics,
    lapse_summary,
    time_on_task,
    time_on_task_statistics,
)
from meg_tokens.behavior.trials import CLASS_NAMES

from tests.behavior.factories import trial_features


def _factorial_session(subject="H01", offset=0.0, n_subjects=1):
    """One subject's Fast and Slow blocks covering all three classes."""
    sessions = []
    for index in range(n_subjects):
        name = subject if n_subjects == 1 else f"H{index + 1:02d}"
        shift = offset + 50.0 * index
        sessions.append(
            trial_features(
                subject=[name] * 6,
                condition=["Fast"] * 3 + ["Slow"] * 3,
                run=[1, 1, 1, 2, 2, 2],
                run_trial_index=[1, 2, 3, 1, 2, 3],
                initial_time_ms=[0, 1000, 2000, 3000, 4000, 5000],
                trial_class=[1, 2, 3] * 2,
                trial_class_name=["easy", "ambiguous", "misleading"] * 2,
                dt_ms=[
                    800.0 + shift, 1200.0 + shift, 1100.0 + shift,
                    900.0 + 1.2 * shift, 1400.0 + 1.2 * shift, 1300.0 + 1.2 * shift,
                ],
                isCorrect=[True, True, False, True, False, True],
                choice_side=[1, 2, 1, 2, 1, 2],
                correct_side=[1, 2, 2, 2, 2, 2],
            )
        )
    return pd.concat(sessions, ignore_index=True)


# --- condition by class -------------------------------------------------------


def test_every_declared_cell_is_reported_for_every_subject():
    cells = condition_class_cells(_factorial_session(n_subjects=2))

    assert len(cells) == 2 * 2 * 3
    assert set(cells["condition"]) == {"fast", "slow"}
    assert set(cells["trial_class_name"]) == {"easy", "ambiguous", "misleading"}


def test_an_empty_cell_is_kept_with_a_zero_count_and_no_mean():
    features = _factorial_session()
    features = features.loc[features["trial_class"] != 3]

    cells = condition_class_cells(features).set_index(["condition", "trial_class_name"])

    assert cells.loc[("fast", "misleading"), "n_trials"] == 0
    assert np.isnan(cells.loc[("fast", "misleading"), "mean_dt_ms"])


def test_each_cell_reports_its_own_decision_time_and_accuracy():
    cells = condition_class_cells(_factorial_session()).set_index(
        ["condition", "trial_class_name"]
    )

    assert cells.loc[("fast", "easy"), "mean_dt_ms"] == pytest.approx(800.0)
    assert cells.loc[("slow", "ambiguous"), "mean_dt_ms"] == pytest.approx(1400.0)
    assert cells.loc[("fast", "misleading"), "accuracy"] == pytest.approx(0.0)
    assert cells.loc[("slow", "misleading"), "accuracy"] == pytest.approx(1.0)


def test_condition_class_cells_reject_a_noncanonical_correctness_value():
    features = _factorial_session()
    features["isCorrect"] = "true"

    with pytest.raises(ValueError, match="isCorrect must be boolean"):
        condition_class_cells(features)


def test_the_cells_feed_a_condition_by_class_anova_for_every_measure():
    cells = condition_class_cells(_factorial_session(n_subjects=4))

    statistics = condition_class_statistics(cells)

    assert set(statistics["measure"]) == {"mean_dt_ms", "log_mean_dt_ms", "accuracy"}
    assert set(statistics["effect"]) == {
        "condition", "trial_class", "condition_x_trial_class",
    }
    assert (statistics["n_subjects"] == 4).all()


def test_a_purely_multiplicative_condition_effect_interacts_on_ms_but_not_on_log():
    """The reason ``log_mean_dt_ms`` is reported alongside ``mean_dt_ms``.

    Build Slow as exactly 1.2x Fast in every class. That is a constant *ratio*
    with no interaction to find, but because the classes sit at different
    baselines it shows up as a condition-by-class interaction when the test is
    run on raw milliseconds -- the scale the effect is not on.
    """
    rng = np.random.default_rng(0)
    baselines = {1: 800.0, 2: 1400.0, 3: 1200.0}
    rows = []
    for subject in range(20):
        # Per-subject speed and per-cell noise are multiplicative, so the log
        # scale is additive by construction and the interaction there is pure
        # error. Additive noise would leave the ms interaction with no error
        # variance at all and the F undefined.
        subject_scale = float(np.exp(rng.normal(0.0, 0.15)))
        for condition, factor in (("fast", 1.0), ("slow", 1.2)):
            for code, class_name in CLASS_NAMES.items():
                rows.append(
                    {
                        "subject": f"H{subject:02d}",
                        "condition": condition,
                        "trial_class": code,
                        "trial_class_name": class_name,
                        "n_trials": 50,
                        "mean_dt_ms": (
                            baselines[code] * subject_scale * factor
                            * float(np.exp(rng.normal(0.0, 0.04)))
                        ),
                        "accuracy": 0.8,
                    }
                )

    statistics = condition_class_statistics(pd.DataFrame(rows))
    interaction = statistics.loc[
        statistics["effect"] == "condition_x_trial_class"
    ].set_index("measure")

    assert interaction.loc["mean_dt_ms", "p"] < 0.05
    assert interaction.loc["log_mean_dt_ms", "p"] > 0.05
    assert (
        interaction.loc["log_mean_dt_ms", "partial_eta_squared"]
        < interaction.loc["mean_dt_ms", "partial_eta_squared"] / 3
    )


def test_the_anova_is_skipped_when_a_cell_is_missing_for_everyone():
    features = _factorial_session(n_subjects=4)
    features = features.loc[
        ~((features["condition"] == "Slow") & (features["trial_class"] == 3))
    ]

    cells = condition_class_cells(features)

    # The cell exists with zero trials, so no subject has a complete design.
    assert condition_class_statistics(cells).empty


# --- choice side --------------------------------------------------------------


def test_choice_side_summary_reports_balance_and_side_asymmetry():
    features = trial_features(
        choice_side=[1, 1, 1, 2],
        correct_side=[1, 1, 2, 2],
        dt_ms=[900.0, 1100.0, 1000.0, 1400.0],
        isCorrect=[True, True, False, True],
    )

    summary = choice_side_summary(features)
    overall = summary.loc[summary["condition"] == "all"].iloc[0]

    assert overall["proportion_left_choices"] == pytest.approx(0.75)
    assert overall["proportion_right_choices"] == pytest.approx(0.25)
    assert overall["proportion_left_correct_side"] == pytest.approx(0.5)
    assert overall["mean_left_dt_ms"] == pytest.approx(1000.0)
    assert overall["mean_right_dt_ms"] == pytest.approx(1400.0)
    assert overall["accuracy_left"] == pytest.approx(2 / 3)
    assert overall["accuracy_right"] == pytest.approx(1.0)


def test_choice_side_is_summarized_pooled_and_within_each_condition():
    summary = choice_side_summary(_factorial_session())

    assert summary["condition"].tolist() == ["all", "fast", "slow"]
    assert summary.loc[summary["condition"] == "all", "n_trials"].iloc[0] == 6


def test_choice_side_statistics_contrast_left_against_right():
    summary = choice_side_summary(_factorial_session(n_subjects=4))

    statistics = choice_side_statistics(summary)

    assert set(statistics["measure"]) == {
        "choice_proportion", "decision_time", "accuracy",
    }
    assert set(statistics["condition"]) == {"all", "fast", "slow"}
    assert (statistics["label_a"] == "left").all()
    assert (statistics["label_b"] == "right").all()


# --- time on task -------------------------------------------------------------


def _interleaved_session(subject="H01", drift=0.0):
    """Fast and Slow blocks alternating on the session clock."""
    return trial_features(
        subject=[subject] * 8,
        condition=["Fast"] * 2 + ["Slow"] * 2 + ["Fast"] * 2 + ["Slow"] * 2,
        run=[1, 1, 1, 1, 2, 2, 2, 2],
        run_trial_index=[1, 2, 1, 2, 1, 2, 1, 2],
        initial_time_ms=[100, 200, 300, 400, 500, 600, 700, 800],
        dt_ms=[
            1000.0, 1010.0, 1200.0, 1210.0,
            1000.0 + drift, 1010.0 + drift, 1200.0 + drift, 1210.0 + drift,
        ],
    )


def test_block_order_comes_from_the_session_clock_not_the_run_number():
    """Fast run 2 was recorded after Slow run 1, so DT drift must be measured
    against the chronological block, not the per-condition run counter."""
    drift = time_on_task(_interleaved_session(drift=-100.0))
    fast = drift.loc[drift["condition"] == "fast"].iloc[0]

    assert fast["n_blocks"] == 2
    assert fast["converged"]
    assert fast["dt_per_block_ms"] < 0


def test_a_session_without_drift_reports_a_flat_block_slope():
    drift = time_on_task(_interleaved_session(drift=0.0))
    fast = drift.loc[drift["condition"] == "fast"].iloc[0]

    assert fast["dt_per_block_ms"] == pytest.approx(0.0, abs=1e-9)


def test_time_on_task_is_fitted_pooled_and_per_condition():
    drift = time_on_task(_interleaved_session(drift=-100.0))

    assert set(drift["condition"]) == {"all", "fast", "slow"}


def test_an_unestimable_design_is_reported_rather_than_guessed():
    features = trial_features(
        condition=["Fast"] * 2,
        run=[1, 1],
        run_trial_index=[1, 2],
        initial_time_ms=[100, 200],
        dt_ms=[1000.0, 1100.0],
    )

    drift = time_on_task(features)

    assert not drift["converged"].any()
    assert drift["dt_per_block_ms"].isna().all()


def test_time_on_task_statistics_test_the_slopes_and_contrast_conditions():
    fits = pd.concat(
        [
            time_on_task(_interleaved_session(f"H{index:02d}", drift=-100.0 - index))
            for index in range(1, 5)
        ],
        ignore_index=True,
    )

    statistics = time_on_task_statistics(fits)

    assert set(statistics["term"]) == {
        "dt_per_block_ms", "dt_per_within_block_trial_ms",
    }
    assert "fast_vs_slow" in set(statistics["condition"])
    drift_test = statistics.loc[
        (statistics["condition"] == "fast")
        & (statistics["term"] == "dt_per_block_ms")
    ].iloc[0]
    assert drift_test["n_subjects"] == 4
    assert drift_test["mean"] < 0


# --- condition order ----------------------------------------------------------


def test_the_first_condition_is_read_from_the_session_clock():
    order = condition_order_effects(_interleaved_session())

    assert order["first_condition"].tolist() == ["fast"]


def test_condition_order_reports_the_speed_accuracy_adjustment_per_subject():
    order = condition_order_effects(_interleaved_session()).iloc[0]

    assert order["mean_fast_dt_ms"] == pytest.approx(1005.0)
    assert order["mean_slow_dt_ms"] == pytest.approx(1205.0)
    assert order["slow_minus_fast_dt_ms"] == pytest.approx(200.0)


def _order_table(first_conditions):
    return pd.DataFrame({
        "subject": [f"H{index:02d}" for index in range(len(first_conditions))],
        "first_condition": first_conditions,
        "mean_fast_dt_ms": [900.0 + 20 * i for i in range(len(first_conditions))],
        "mean_slow_dt_ms": [1200.0 + 35 * i for i in range(len(first_conditions))],
        "slow_minus_fast_dt_ms": [
            300.0 + 15 * i for i in range(len(first_conditions))
        ],
    })


def test_condition_order_is_tested_between_subjects():
    statistics = condition_order_statistics(
        _order_table(["fast", "fast", "slow", "slow", "fast", "slow"])
    )

    assert set(statistics["measure"]) == {
        "slow_minus_fast_dt_ms", "mean_fast_dt_ms", "mean_slow_dt_ms",
    }
    row = statistics.iloc[0]
    assert row["test"] == "welch_t_test"
    assert (row["label_a"], row["label_b"]) == ("first_fast", "first_slow")
    assert row["n_a"] == 3 and row["n_b"] == 3


@pytest.mark.parametrize(
    "first_conditions",
    [["fast"] * 4, ["fast", "slow", "slow", "slow"]],
    ids=["one_group", "group_of_one"],
)
def test_condition_order_is_omitted_when_the_groups_cannot_support_a_test(
    first_conditions,
):
    assert condition_order_statistics(_order_table(first_conditions)).empty


# --- lapses -------------------------------------------------------------------


def _lapse_session(subject="H01"):
    return trial_features(
        subject=[subject] * 4,
        condition=["Fast", "Fast", "Slow", "Slow"],
        run=[1, 1, 2, 2],
        run_trial_index=[1, 2, 1, 2],
        has_choice=[True, False, True, False],
        primary_analysis_eligible=[True, False, True, False],
        nOutcome=[0, 7006, 0, 7011],
        dt_ms=[900.0, float("nan"), 1300.0, float("nan")],
    )


def test_a_lapse_is_a_started_trial_that_produced_no_choice():
    summary = lapse_summary(_lapse_session())
    overall = summary.loc[summary["condition"] == "all"].iloc[0]

    assert overall["n_started_trials"] == 4
    assert overall["n_lapse_trials"] == 2
    assert overall["lapse_rate"] == pytest.approx(0.5)


def test_lapse_outcome_codes_are_counted_under_their_labview_labels():
    summary = lapse_summary(_lapse_session())
    overall = summary.loc[summary["condition"] == "all"].iloc[0]

    assert overall["n_outcome_7006_reaction_time_too_long"] == 1
    assert overall["n_outcome_7011_delay_1_error"] == 1
    assert overall["n_lapse_other_outcomes"] == 0


def test_an_unexpected_no_choice_outcome_is_counted_separately():
    features = _lapse_session()
    features.loc[1, "nOutcome"] = 7004

    summary = lapse_summary(features)
    overall = summary.loc[summary["condition"] == "all"].iloc[0]

    assert overall["n_lapse_trials"] == 2
    assert overall["n_outcome_7006_reaction_time_too_long"] == 0
    assert overall["n_lapse_other_outcomes"] == 1


def test_never_started_trials_are_not_lapses_and_not_in_the_denominator():
    features = _lapse_session()
    features.loc[1, "is_started"] = False

    overall = lapse_summary(features).query("condition == 'all'").iloc[0]

    assert overall["n_started_trials"] == 3
    assert overall["n_lapse_trials"] == 1


def test_lapse_statistics_summarize_rates_and_contrast_the_conditions():
    summary = pd.concat(
        [_lapse_session(f"H{index:02d}") for index in range(1, 5)],
        ignore_index=True,
    )
    summary = lapse_summary(summary)

    statistics = lapse_statistics(summary)

    assert set(statistics["condition"]) == {"all", "fast", "slow", "fast_vs_slow"}
    pooled = statistics.loc[statistics["condition"] == "all"].iloc[0]
    assert pooled["n_subjects"] == 4
    assert pooled["mean"] == pytest.approx(0.5)
    assert pooled["n_lapse_trials"] == 8


# --- extreme decision times ---------------------------------------------------


def _extreme_session():
    return trial_features(
        run_trial_index=[1, 2, 3, 4, 5, 6],
        dt_ms=[900.0, 950.0, 1000.0, 1050.0, 1100.0, 20000.0],
    )


def test_an_extreme_decision_time_is_flagged_and_kept():
    counts, flagged = extreme_decision_times(_extreme_session(), mad_threshold=5.0)

    assert counts.iloc[0]["n_dt_trials"] == 6
    assert counts.iloc[0]["n_extreme_dt"] == 1
    assert counts.iloc[0]["n_extreme_slow"] == 1
    assert counts.iloc[0]["max_dt_ms"] == pytest.approx(20000.0)
    assert flagged["dt_ms"].tolist() == pytest.approx([20000.0])
    assert flagged.iloc[0]["robust_z"] > 5.0


def test_a_flagged_trial_carries_enough_provenance_to_find_it_again():
    _, flagged = extreme_decision_times(_extreme_session())
    row = flagged.iloc[0]

    assert row["trial_id"].endswith("trial-006")
    assert row["condition"] == "Fast"
    assert row["run_trial_index"] == 6
    assert row["trial_class_name"] == "easy"


def test_the_threshold_decides_what_counts_as_extreme():
    counts, flagged = extreme_decision_times(_extreme_session(), mad_threshold=1e6)

    assert counts.iloc[0]["n_extreme_dt"] == 0
    assert flagged.empty


def test_a_subject_without_decision_time_variability_flags_nothing():
    features = trial_features(run_trial_index=[1, 2, 3], dt_ms=[900.0] * 3)

    counts, flagged = extreme_decision_times(features)

    assert counts.iloc[0]["mad_dt_ms"] == pytest.approx(0.0)
    assert counts.iloc[0]["n_extreme_dt"] == 0
    assert flagged.empty


def test_negative_decision_times_are_counted_in_their_own_right():
    features = trial_features(
        run_trial_index=[1, 2, 3, 4],
        dt_ms=[-100.0, 900.0, 950.0, 1000.0],
    )

    counts, _ = extreme_decision_times(features)

    assert counts.iloc[0]["n_negative_dt"] == 1
    assert counts.iloc[0]["min_dt_ms"] == pytest.approx(-100.0)
