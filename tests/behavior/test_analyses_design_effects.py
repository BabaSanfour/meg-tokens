"""Tests for meg_tokens.behavior.analyses.design_effects."""

import numpy as np
import pandas as pd
import pytest

from meg_tokens.behavior.analyses.design_effects import (
    choice_side_statistics,
    choice_side_summary,
    condition_class_cells,
    condition_class_contrasts,
    condition_class_statistics,
    condition_order_effects,
    condition_order_statistics,
    extreme_decision_times,
    lapse_statistics,
    lapse_summary,
    time_on_task,
    time_on_task_statistics,
    within_block_decile_profile,
    within_block_decile_statistics,
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


def _both_order_groups(n_per_group=5, effect=120.0):
    """Subjects counterbalanced on which condition they met first.

    `_interleaved_session` always starts on Fast, so it yields a single order
    group and the between-group tests are skipped entirely. Each subject here
    carries a different Slow-minus-Fast effect, so the within-group tests have
    the variance an interval needs.
    """
    sessions = []
    for group_index, first in enumerate(("Fast", "Slow")):
        second = "Slow" if first == "Fast" else "Fast"
        for index in range(n_per_group):
            subject = f"H{group_index * n_per_group + index + 1:02d}"
            subject_effect = effect + 25.0 * index
            order = [first] * 2 + [second] * 2
            sessions.append(
                trial_features(
                    subject=[subject] * 4,
                    condition=order,
                    run=[1, 1, 2, 2],
                    run_trial_index=[1, 2, 1, 2],
                    initial_time_ms=[100, 200, 300, 400],
                    dt_ms=[
                        1000.0 + (subject_effect if name == "Slow" else 0.0) + offset
                        for name, offset in zip(order, (0.0, 10.0, 0.0, 10.0))
                    ],
                )
            )
    return pd.concat(sessions, ignore_index=True)


def test_condition_order_statistics_test_each_group_against_zero():
    statistics = condition_order_statistics(
        condition_order_effects(_both_order_groups())
    )
    within = statistics.loc[statistics["test"] == "one_sample_vs_zero"]

    # The within-group tests are what counterbalancing has to pass, so they
    # belong in the derivative rather than being recomputed by the figure.
    assert len(within) == 2
    assert set(within["measure"]) == {"slow_minus_fast_dt_ms"}
    assert within["n_positive"].notna().all()
    assert (within["ci_low"] < within["mean"]).all()
    assert (within["mean"] < within["ci_high"]).all()


def test_the_between_group_null_is_accompanied_by_a_bound():
    statistics = condition_order_statistics(
        condition_order_effects(_both_order_groups())
    )
    bound = statistics.loc[statistics["test"] == "welch_bound"]

    assert len(bound) == 1
    # 90%, deliberately: it is read as a bound on an effect argued to be
    # absent, so the narrower interval is the stricter claim.
    assert bound.iloc[0]["coverage"] == pytest.approx(0.90)
    assert bound.iloc[0]["ci_low"] < bound.iloc[0]["ci_high"]


def test_three_different_tests_report_on_the_same_measure():
    """So a consumer must select on `test`, not on `measure` alone."""
    statistics = condition_order_statistics(
        condition_order_effects(_both_order_groups())
    )
    rows = statistics.loc[statistics["measure"] == "slow_minus_fast_dt_ms"]

    assert len(rows) == 4
    assert set(rows["test"]) == {"welch_t_test", "welch_bound", "one_sample_vs_zero"}


# --- condition x class contrasts ----------------------------------------------


def _scaled_cells(factor=1.20, n_subjects=6, jitter=0.05):
    """Cells where Slow is ``factor`` times Fast in every class.

    A pure multiplicative condition effect: the stretch factor must recover
    ``factor`` for all three classes, whatever the class differs by in level.

    ``jitter`` spreads each subject's own factor around ``factor``
    symmetrically *in log space*, alternating sign, so the geometric mean stays
    exactly ``factor`` while the log ratios carry real between-subject
    variance. Without that variance the interval is undefined, which is a
    property of the fixture rather than of the estimator.
    """
    base = {"easy": 800.0, "ambiguous": 1400.0, "misleading": 1200.0}
    accuracy = {"easy": 0.99, "ambiguous": 0.77, "misleading": 0.37}
    rows = []
    for index in range(n_subjects):
        shift = 20.0 * index
        subject_factor = factor * np.exp(jitter * (1 if index % 2 == 0 else -1))
        for class_name, level in base.items():
            for condition in ("fast", "slow"):
                rows.append(
                    {
                        "subject": f"H{index + 1:02d}",
                        "condition": condition,
                        "trial_class_name": class_name,
                        "mean_dt_ms": (level + shift)
                        * (subject_factor if condition == "slow" else 1.0),
                        "accuracy": accuracy[class_name] + 0.005 * index,
                    }
                )
    return pd.DataFrame(rows)


def test_a_pure_multiplicative_effect_recovers_one_factor_for_every_class():
    contrasts = condition_class_contrasts(_scaled_cells(factor=1.20))
    stretch = contrasts.loc[contrasts["term"] == "stretch_factor"].set_index(
        "trial_class_name"
    )

    assert stretch["estimate"].to_numpy() == pytest.approx(1.20)
    # The additive differences are wildly unequal for the same ratio, which is
    # exactly why the factor and not the difference is the right summary.
    additive = contrasts.loc[
        contrasts["term"] == "additive_difference_ms"
    ].set_index("trial_class_name")
    assert additive.loc["easy", "estimate"] < additive.loc["ambiguous", "estimate"]


def test_a_pure_multiplicative_effect_still_shows_an_additive_interaction():
    """Which is exactly why the report tests the interaction on the log scale.

    Every class here stretches by an identical factor, so on the multiplicative
    reading there is no interaction at all. Measured in milliseconds there
    plainly is one: classes sitting at different levels move by different
    amounts. A millisecond-scale interaction test would reject a model that is
    true by construction.
    """
    # jitter=0 so the difference of differences is exact rather than averaged
    # over per-subject factors; this assertion is about the estimate, and the
    # interval is exercised by the stretch-factor test above.
    contrasts = condition_class_contrasts(_scaled_cells(factor=1.20, jitter=0.0))
    bound = contrasts.loc[contrasts["term"] == "additive_interaction_ms"].iloc[0]

    level_gap = 800.0 - 1200.0
    assert bound["estimate"] == pytest.approx(level_gap * 0.20, abs=1e-6)
    assert bound["coverage"] == pytest.approx(0.90)


def test_the_stretch_factor_interval_is_reported_on_the_factor_itself():
    contrasts = condition_class_contrasts(_scaled_cells(factor=1.20))
    row = contrasts.loc[contrasts["term"] == "stretch_factor"].iloc[0]

    # Back-transformed from log ratios, so the interval brackets the factor
    # rather than a log value, and is asymmetric about it.
    assert row["ci_low"] < row["estimate"] < row["ci_high"]
    assert row["ci_low"] > 0


def test_accuracy_is_tested_against_chance_not_zero():
    contrasts = condition_class_contrasts(_scaled_cells())
    accuracy = contrasts.loc[contrasts["term"] == "accuracy_vs_chance"].set_index(
        "trial_class_name"
    )

    # Misleading sits below .5 and easy far above it, with opposite t signs.
    assert accuracy.loc["misleading", "estimate"] < 0.5
    assert accuracy.loc["misleading", "t"] < 0
    assert accuracy.loc["easy", "t"] > 0
    # The interval is reported in accuracy units, not as a deviation from .5.
    assert 0.0 < accuracy.loc["misleading", "ci_low"] < 0.5


# --- within-block deciles -----------------------------------------------------


def _scheduled_class_session(subject="H01", n_subjects=1, positional_ms=0.0):
    """Sessions where trial class is confounded with within-block position.

    Each 20-trial block runs easy trials first and ambiguous trials second, and
    decision time is a pure function of class plus an optional genuine
    positional term. The observed profile therefore rises even when
    ``positional_ms`` is zero, and only the class adjustment can tell the two
    sources apart -- which is the property panel B of F04 rests on.
    """
    block_length = 20
    sessions = []
    for index in range(n_subjects):
        name = subject if n_subjects == 1 else f"H{index + 1:02d}"
        for run in (1, 2):
            positions = list(range(1, block_length + 1))
            classes = ["easy"] * 10 + ["ambiguous"] * 10
            sessions.append(
                trial_features(
                    subject=[name] * block_length,
                    condition=["Fast" if run == 1 else "Slow"] * block_length,
                    run=[run] * block_length,
                    run_trial_index=positions,
                    initial_time_ms=[
                        100_000 * run + 100 * position for position in positions
                    ],
                    trial_class=[1] * 10 + [2] * 10,
                    trial_class_name=classes,
                    dt_ms=[
                        (800.0 if name_ == "easy" else 1200.0)
                        + positional_ms * position
                        + 10.0 * index
                        for name_, position in zip(classes, positions)
                    ],
                )
            )
    return pd.concat(sessions, ignore_index=True)


def test_deciles_are_cut_within_each_block_for_every_condition_subset():
    profile = within_block_decile_profile(_scheduled_class_session())

    assert set(profile["condition"]) == {"all", "fast", "slow"}
    for condition in ("all", "fast", "slow"):
        subset = profile.loc[profile["condition"] == condition]
        assert sorted(subset["decile"]) == list(range(1, 11))
    # Each 20-trial block contributes two trials per decile, so the pooled
    # subset holds four and each condition holds two. Every decile is equally
    # filled, which is what cutting within the block guarantees.
    assert set(profile.loc[profile["condition"] == "all", "n_trials"]) == {4}
    assert set(profile.loc[profile["condition"] == "fast", "n_trials"]) == {2}


def test_the_class_adjustment_removes_a_purely_compositional_rise():
    profile = within_block_decile_profile(_scheduled_class_session())
    pooled = profile.loc[profile["condition"] == "all"].set_index("decile")

    # Class alone drives decision time here, so the observed profile steps from
    # the easy level to the ambiguous one ...
    assert pooled.loc[1, "dt_ms"] == pytest.approx(800.0)
    assert pooled.loc[10, "dt_ms"] == pytest.approx(1200.0)
    # ... while the adjusted profile is flat at the subject's grand mean.
    assert pooled["dt_class_adjusted_ms"].to_numpy() == pytest.approx(1000.0)


def test_the_class_adjustment_is_conservative_about_genuine_drift():
    """Confounded class and position means the adjustment over-corrects.

    Removing per-class means also removes whatever genuine positional drift
    happens to fall between the classes, because that drift is part of the
    class means. Here class and position are *perfectly* confounded, so the
    adjustment is maximally conservative: of a real 90 ms rise it keeps the
    40 ms falling within a class and attributes the rest to composition.

    The direction is what matters for the report: this estimator can
    under-state a real positional effect but never over-state one, so the
    class-adjusted slope is a lower bound and the compositional share of the
    raw slope is an upper bound.
    """
    profile = within_block_decile_profile(
        _scheduled_class_session(positional_ms=5.0)
    )
    pooled = profile.loc[profile["condition"] == "all"].set_index("decile")
    adjusted = pooled["dt_class_adjusted_ms"]
    observed = pooled["dt_ms"]

    genuine_rise = 5.0 * 18  # deciles 1 and 10 average positions 1.5 and 19.5
    assert observed.loc[10] - observed.loc[1] == pytest.approx(400.0 + genuine_rise)
    surviving = adjusted.loc[10] - adjusted.loc[1]
    assert 0 < surviving < genuine_rise
    # Exactly the within-class share: half the block spanned at 5 ms/trial.
    assert surviving == pytest.approx(5.0 * 8, abs=1e-6)


def test_class_composition_is_reported_per_decile():
    profile = within_block_decile_profile(_scheduled_class_session())
    pooled = profile.loc[profile["condition"] == "all"].set_index("decile")

    assert pooled.loc[1, "proportion_easy"] == pytest.approx(1.0)
    assert pooled.loc[10, "proportion_ambiguous"] == pytest.approx(1.0)


def test_decile_statistics_separate_observed_and_adjusted_rise():
    profile = pd.concat(
        [
            within_block_decile_profile(
                _scheduled_class_session(f"H{index:02d}", positional_ms=index)
            )
            for index in range(1, 6)
        ],
        ignore_index=True,
    )

    statistics = within_block_decile_statistics(profile)
    rise = statistics.loc[statistics["term"] == "first_to_last_rise_ms"].set_index(
        "measure"
    )

    assert rise.loc["dt_ms", "mean"] > rise.loc["dt_class_adjusted_ms", "mean"]
    assert rise.loc["dt_ms", "n_subjects"] == 5


def test_the_peak_decile_is_descriptive_and_carries_no_test():
    profile = within_block_decile_profile(_scheduled_class_session(n_subjects=2))

    statistics = within_block_decile_statistics(profile)
    peaks = statistics.loc[statistics["term"] == "peak_decile"]

    assert set(peaks["test"]) == {"descriptive"}
    assert peaks["p"].isna().all()


def test_per_decile_deviations_carry_a_bonferroni_threshold():
    profile = pd.concat(
        [
            within_block_decile_profile(
                _scheduled_class_session(f"H{index:02d}", positional_ms=index)
            )
            for index in range(1, 6)
        ],
        ignore_index=True,
    )

    statistics = within_block_decile_statistics(profile)
    deviations = statistics.loc[statistics["term"] == "linear_deviation_ms"]

    assert len(deviations) == 10
    assert deviations["p_threshold"].to_numpy() == pytest.approx(0.005)


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
