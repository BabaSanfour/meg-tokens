"""Tests for meg_tokens.behavior.analyses.design_effects.

One test per behaviour, not per assertion. The fixtures are the expensive part
— a counterbalanced twenty-subject session takes as long to build as the whole
analysis takes to run — so every claim a fixture can support is asserted on it.
"""

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


# --- condition by class -------------------------------------------------------


def test_every_declared_cell_is_reported_with_its_own_decision_time_and_accuracy():
    cells = condition_class_cells(_factorial_session(n_subjects=2))
    assert len(cells) == 2 * 2 * 3
    assert set(cells["condition"]) == {"fast", "slow"}
    assert set(cells["trial_class_name"]) == {"easy", "ambiguous", "misleading"}

    indexed = condition_class_cells(_factorial_session()).set_index(
        ["condition", "trial_class_name"]
    )
    assert indexed.loc[("fast", "easy"), "mean_dt_ms"] == pytest.approx(800.0)
    assert indexed.loc[("slow", "ambiguous"), "mean_dt_ms"] == pytest.approx(1400.0)
    assert indexed.loc[("fast", "misleading"), "accuracy"] == pytest.approx(0.0)
    assert indexed.loc[("slow", "misleading"), "accuracy"] == pytest.approx(1.0)

    # A cell nobody ran is kept with a zero count and no mean, never dropped.
    features = _factorial_session()
    sparse = condition_class_cells(
        features.loc[features["trial_class"] != 3]
    ).set_index(["condition", "trial_class_name"])
    assert sparse.loc[("fast", "misleading"), "n_trials"] == 0
    assert np.isnan(sparse.loc[("fast", "misleading"), "mean_dt_ms"])

    with pytest.raises(ValueError, match="isCorrect must be boolean"):
        condition_class_cells(_factorial_session().assign(isCorrect="true"))


def test_the_anova_covers_every_measure_and_is_skipped_when_no_subject_is_complete():
    statistics = condition_class_statistics(
        condition_class_cells(_factorial_session(n_subjects=4))
    )
    assert set(statistics["measure"]) == {"mean_dt_ms", "log_mean_dt_ms", "accuracy"}
    assert set(statistics["effect"]) == {"condition", "trial_class", "condition_x_trial_class"}
    assert (statistics["n_subjects"] == 4).all()

    features = _factorial_session(n_subjects=4)
    incomplete = features.loc[
        ~((features["condition"] == "Slow") & (features["trial_class"] == 3))
    ]
    # The cell exists with zero trials, so no subject has a complete design.
    assert condition_class_statistics(condition_class_cells(incomplete)).empty


def test_a_purely_multiplicative_condition_effect_interacts_on_ms_but_not_on_log():
    """The reason ``log_mean_dt_ms`` is reported alongside ``mean_dt_ms``.

    Slow is exactly 1.2x Fast in every class: a constant *ratio* with no
    interaction to find. Because the classes sit at different baselines it
    nonetheless shows up as a condition-by-class interaction when the test runs
    on raw milliseconds — the scale the effect is not on. Per-subject speed and
    per-cell noise are multiplicative, so the log scale is additive by
    construction and its interaction is pure error; additive noise would leave
    the ms interaction with no error variance and the F undefined.
    """
    rng = np.random.default_rng(0)
    baselines = {1: 800.0, 2: 1400.0, 3: 1200.0}
    rows = [
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
        for subject in range(20)
        for subject_scale in [float(np.exp(rng.normal(0.0, 0.15)))]
        for condition, factor in (("fast", 1.0), ("slow", 1.2))
        for code, class_name in CLASS_NAMES.items()
    ]

    interaction = condition_class_statistics(pd.DataFrame(rows))
    interaction = interaction.loc[
        interaction["effect"] == "condition_x_trial_class"
    ].set_index("measure")

    assert interaction.loc["mean_dt_ms", "p"] < 0.05
    assert interaction.loc["log_mean_dt_ms", "p"] > 0.05
    assert (
        interaction.loc["log_mean_dt_ms", "partial_eta_squared"]
        < interaction.loc["mean_dt_ms", "partial_eta_squared"] / 3
    )


# --- condition x class contrasts ----------------------------------------------


def _scaled_cells(factor=1.20, n_subjects=6, jitter=0.05):
    """Cells where Slow is ``factor`` times Fast in every class.

    A pure multiplicative condition effect: the stretch factor must recover
    ``factor`` for all three classes, whatever the class differs by in level.
    ``jitter`` spreads each subject's own factor around ``factor``
    symmetrically *in log space*, alternating sign, so the geometric mean stays
    exactly ``factor`` while the log ratios carry real between-subject variance.
    Without that variance the interval is undefined, which is a property of the
    fixture rather than of the estimator.
    """
    base = {"easy": 800.0, "ambiguous": 1400.0, "misleading": 1200.0}
    accuracy = {"easy": 0.99, "ambiguous": 0.77, "misleading": 0.37}
    return pd.DataFrame([
        {
            "subject": f"H{index + 1:02d}",
            "condition": condition,
            "trial_class_name": class_name,
            "mean_dt_ms": (level + 20.0 * index) * (subject_factor if condition == "slow" else 1.0),
            "accuracy": accuracy[class_name] + 0.005 * index,
        }
        for index in range(n_subjects)
        for subject_factor in [factor * np.exp(jitter * (1 if index % 2 == 0 else -1))]
        for class_name, level in base.items()
        for condition in ("fast", "slow")
    ])


def test_a_pure_multiplicative_effect_recovers_one_factor_and_still_interacts_additively():
    contrasts = condition_class_contrasts(_scaled_cells(factor=1.20))
    stretch = contrasts.loc[contrasts["term"] == "stretch_factor"].set_index("trial_class_name")
    assert stretch["estimate"].to_numpy() == pytest.approx(1.20)
    # The additive differences are wildly unequal for the same ratio, which is
    # exactly why the factor and not the difference is the right summary.
    additive = contrasts.loc[
        contrasts["term"] == "additive_difference_ms"
    ].set_index("trial_class_name")
    assert additive.loc["easy", "estimate"] < additive.loc["ambiguous", "estimate"]

    # Back-transformed from log ratios, so the interval brackets the factor
    # rather than a log value.
    row = contrasts.loc[contrasts["term"] == "stretch_factor"].iloc[0]
    assert row["ci_low"] < row["estimate"] < row["ci_high"]
    assert row["ci_low"] > 0

    # Every class stretches identically, so on the multiplicative reading there
    # is no interaction at all; measured in milliseconds there plainly is one.
    # jitter=0 so the difference of differences is exact.
    bound = condition_class_contrasts(_scaled_cells(factor=1.20, jitter=0.0))
    bound = bound.loc[bound["term"] == "additive_interaction_ms"].iloc[0]
    assert bound["estimate"] == pytest.approx((800.0 - 1200.0) * 0.20, abs=1e-6)
    assert bound["coverage"] == pytest.approx(0.90)


def test_accuracy_is_tested_against_chance_not_zero():
    accuracy = condition_class_contrasts(_scaled_cells())
    accuracy = accuracy.loc[accuracy["term"] == "accuracy_vs_chance"].set_index("trial_class_name")
    # Misleading sits below .5 and easy far above it, with opposite t signs.
    assert accuracy.loc["misleading", "estimate"] < 0.5
    assert accuracy.loc["misleading", "t"] < 0
    assert accuracy.loc["easy", "t"] > 0
    # The interval is in accuracy units, not as a deviation from .5.
    assert 0.0 < accuracy.loc["misleading", "ci_low"] < 0.5


# --- choice side --------------------------------------------------------------


def test_choice_side_reports_balance_asymmetry_and_a_paired_left_right_contrast():
    overall = choice_side_summary(
        trial_features(
            choice_side=[1, 1, 1, 2],
            correct_side=[1, 1, 2, 2],
            dt_ms=[900.0, 1100.0, 1000.0, 1400.0],
            isCorrect=[True, True, False, True],
        )
    )
    overall = overall.loc[overall["condition"] == "all"].iloc[0]
    assert overall["proportion_left_choices"] == pytest.approx(0.75)
    assert overall["proportion_right_choices"] == pytest.approx(0.25)
    assert overall["proportion_left_correct_side"] == pytest.approx(0.5)
    assert overall["mean_left_dt_ms"] == pytest.approx(1000.0)
    assert overall["mean_right_dt_ms"] == pytest.approx(1400.0)
    assert overall["accuracy_left"] == pytest.approx(2 / 3)
    assert overall["accuracy_right"] == pytest.approx(1.0)

    pooled = choice_side_summary(_factorial_session())
    assert pooled["condition"].tolist() == ["all", "fast", "slow"]
    assert pooled.loc[pooled["condition"] == "all", "n_trials"].iloc[0] == 6

    statistics = choice_side_statistics(choice_side_summary(_factorial_session(n_subjects=4)))
    assert set(statistics["measure"]) == {"choice_proportion", "decision_time", "accuracy"}
    assert set(statistics["condition"]) == {"all", "fast", "slow"}
    assert (statistics["label_a"] == "left").all()
    assert (statistics["label_b"] == "right").all()


# --- time on task -------------------------------------------------------------


def test_block_order_comes_from_the_session_clock_not_the_run_number():
    """Fast run 2 was recorded after Slow run 1, so DT drift must be measured
    against the chronological block, not the per-condition run counter."""
    drift = time_on_task(_interleaved_session(drift=-100.0))
    assert set(drift["condition"]) == {"all", "fast", "slow"}
    fast = drift.loc[drift["condition"] == "fast"].iloc[0]
    assert fast["n_blocks"] == 2
    assert fast["converged"]
    assert fast["dt_per_block_ms"] < 0

    flat = time_on_task(_interleaved_session(drift=0.0))
    assert flat.loc[flat["condition"] == "fast", "dt_per_block_ms"].iloc[0] == pytest.approx(
        0.0, abs=1e-9
    )

    # A single block cannot estimate a block slope, and says so.
    unestimable = time_on_task(
        trial_features(
            condition=["Fast"] * 2, run=[1, 1], run_trial_index=[1, 2],
            initial_time_ms=[100, 200], dt_ms=[1000.0, 1100.0],
        )
    )
    assert not unestimable["converged"].any()
    assert unestimable["dt_per_block_ms"].isna().all()


def test_time_on_task_statistics_test_the_slopes_and_contrast_the_conditions():
    fits = pd.concat(
        [
            time_on_task(_interleaved_session(f"H{index:02d}", drift=-100.0 - index))
            for index in range(1, 5)
        ],
        ignore_index=True,
    )
    statistics = time_on_task_statistics(fits)
    assert set(statistics["term"]) == {"dt_per_block_ms", "dt_per_within_block_trial_ms"}
    assert "fast_vs_slow" in set(statistics["condition"])
    drift_test = statistics.loc[
        (statistics["condition"] == "fast") & (statistics["term"] == "dt_per_block_ms")
    ].iloc[0]
    assert drift_test["n_subjects"] == 4
    assert drift_test["mean"] < 0


# --- within-block deciles -----------------------------------------------------


def _scheduled_class_session(subject="H01", n_subjects=1, positional_ms=0.0):
    """Sessions where trial class is confounded with within-block position.

    Each 20-trial block runs easy trials first and ambiguous trials second, and
    decision time is a pure function of class plus an optional genuine
    positional term. The observed profile therefore rises even when
    ``positional_ms`` is zero, and only the class adjustment can tell the two
    sources apart — which is the property panel B of F04 rests on.
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
                    initial_time_ms=[100_000 * run + 100 * position for position in positions],
                    trial_class=[1] * 10 + [2] * 10,
                    trial_class_name=classes,
                    dt_ms=[
                        (800.0 if name_ == "easy" else 1200.0)
                        + positional_ms * position + 10.0 * index
                        for name_, position in zip(classes, positions)
                    ],
                )
            )
    return pd.concat(sessions, ignore_index=True)


def test_deciles_are_cut_within_each_block_and_the_class_adjustment_removes_composition():
    profile = within_block_decile_profile(_scheduled_class_session())
    assert set(profile["condition"]) == {"all", "fast", "slow"}
    for condition in ("all", "fast", "slow"):
        assert sorted(profile.loc[profile["condition"] == condition, "decile"]) == list(range(1, 11))
    # Each 20-trial block contributes two trials per decile, so the pooled
    # subset holds four and each condition two. Every decile is equally filled,
    # which is what cutting within the block guarantees.
    assert set(profile.loc[profile["condition"] == "all", "n_trials"]) == {4}
    assert set(profile.loc[profile["condition"] == "fast", "n_trials"]) == {2}

    pooled = profile.loc[profile["condition"] == "all"].set_index("decile")
    # Class alone drives decision time here, so the observed profile steps from
    # the easy level to the ambiguous one ...
    assert pooled.loc[1, "dt_ms"] == pytest.approx(800.0)
    assert pooled.loc[10, "dt_ms"] == pytest.approx(1200.0)
    # ... while the adjusted profile is flat at the subject's grand mean.
    assert pooled["dt_class_adjusted_ms"].to_numpy() == pytest.approx(1000.0)
    assert pooled.loc[1, "proportion_easy"] == pytest.approx(1.0)
    assert pooled.loc[10, "proportion_ambiguous"] == pytest.approx(1.0)


def test_the_class_adjustment_understates_real_drift_rather_than_inventing_it():
    """Removing per-class means also removes whatever genuine positional drift
    falls between the classes, because that drift is part of the class means.
    Here class and position are *perfectly* confounded, so the adjustment is
    maximally conservative. The direction is what matters: this estimator can
    under-state a real positional effect but never over-state one, so the
    class-adjusted slope is a lower bound and the compositional share of the
    raw slope is an upper bound.
    """
    pooled = within_block_decile_profile(_scheduled_class_session(positional_ms=5.0))
    pooled = pooled.loc[pooled["condition"] == "all"].set_index("decile")

    genuine_rise = 5.0 * 18  # deciles 1 and 10 average positions 1.5 and 19.5
    observed = pooled["dt_ms"]
    assert observed.loc[10] - observed.loc[1] == pytest.approx(400.0 + genuine_rise)
    surviving = pooled["dt_class_adjusted_ms"].loc[10] - pooled["dt_class_adjusted_ms"].loc[1]
    assert 0 < surviving < genuine_rise
    # Exactly the within-class share: half the block spanned at 5 ms/trial.
    assert surviving == pytest.approx(5.0 * 8, abs=1e-6)


def test_decile_statistics_separate_the_two_rises_and_correct_the_per_decile_tests():
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

    rise = statistics.loc[statistics["term"] == "first_to_last_rise_ms"].set_index("measure")
    assert rise.loc["dt_ms", "mean"] > rise.loc["dt_class_adjusted_ms", "mean"]
    assert rise.loc["dt_ms", "n_subjects"] == 5

    deviations = statistics.loc[statistics["term"] == "linear_deviation_ms"]
    assert len(deviations) == 10
    assert deviations["p_threshold"].to_numpy() == pytest.approx(0.005)

    # The peak decile is descriptive: an argmax has no null to test against.
    peaks = within_block_decile_statistics(
        within_block_decile_profile(_scheduled_class_session(n_subjects=2))
    )
    peaks = peaks.loc[peaks["term"] == "peak_decile"]
    assert set(peaks["test"]) == {"descriptive"}
    assert peaks["p"].isna().all()


# --- condition order ----------------------------------------------------------


def _both_order_groups(n_per_group=5, effect=120.0):
    """Subjects counterbalanced on which condition they met first.

    ``_interleaved_session`` always starts on Fast, so it yields a single order
    group and the between-group tests are skipped entirely. Each subject here
    carries a different Slow-minus-Fast effect, so the within-group tests have
    the variance an interval needs.
    """
    sessions = []
    for group_index, first in enumerate(("Fast", "Slow")):
        second = "Slow" if first == "Fast" else "Fast"
        for index in range(n_per_group):
            order = [first] * 2 + [second] * 2
            subject_effect = effect + 25.0 * index
            sessions.append(
                trial_features(
                    subject=[f"H{group_index * n_per_group + index + 1:02d}"] * 4,
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


def _order_table(first_conditions):
    return pd.DataFrame({
        "subject": [f"H{index:02d}" for index in range(len(first_conditions))],
        "first_condition": first_conditions,
        "mean_fast_dt_ms": [900.0 + 20 * i for i in range(len(first_conditions))],
        "mean_slow_dt_ms": [1200.0 + 35 * i for i in range(len(first_conditions))],
        "slow_minus_fast_dt_ms": [300.0 + 15 * i for i in range(len(first_conditions))],
    })


def test_the_first_condition_and_the_adjustment_come_from_the_session_clock():
    order = condition_order_effects(_interleaved_session())
    assert order["first_condition"].tolist() == ["fast"]
    row = order.iloc[0]
    assert row["mean_fast_dt_ms"] == pytest.approx(1005.0)
    assert row["mean_slow_dt_ms"] == pytest.approx(1205.0)
    assert row["slow_minus_fast_dt_ms"] == pytest.approx(200.0)


def test_condition_order_reports_a_between_group_test_a_bound_and_the_within_group_tests():
    statistics = condition_order_statistics(condition_order_effects(_both_order_groups()))

    # Three different tests report on the same measure, so a consumer must
    # select on `test`, not on `measure` alone.
    rows = statistics.loc[statistics["measure"] == "slow_minus_fast_dt_ms"]
    assert len(rows) == 4
    assert set(rows["test"]) == {"welch_t_test", "welch_bound", "one_sample_vs_zero"}

    # The within-group tests are what counterbalancing has to pass, so they
    # belong in the derivative rather than being recomputed by the figure.
    within = statistics.loc[statistics["test"] == "one_sample_vs_zero"]
    assert len(within) == 2
    assert set(within["measure"]) == {"slow_minus_fast_dt_ms"}
    assert within["n_positive"].notna().all()
    assert (within["ci_low"] < within["mean"]).all()
    assert (within["mean"] < within["ci_high"]).all()

    # 90%, deliberately: it is read as a bound on an effect argued to be
    # absent, so the narrower interval is the stricter claim.
    bound = statistics.loc[statistics["test"] == "welch_bound"]
    assert len(bound) == 1
    assert bound.iloc[0]["coverage"] == pytest.approx(0.90)
    assert bound.iloc[0]["ci_low"] < bound.iloc[0]["ci_high"]

    between = condition_order_statistics(
        _order_table(["fast", "fast", "slow", "slow", "fast", "slow"])
    )
    assert set(between["measure"]) == {
        "slow_minus_fast_dt_ms", "mean_fast_dt_ms", "mean_slow_dt_ms",
    }
    first = between.iloc[0]
    assert first["test"] == "welch_t_test"
    assert (first["label_a"], first["label_b"]) == ("first_fast", "first_slow")
    assert first["n_a"] == 3 and first["n_b"] == 3


@pytest.mark.parametrize(
    "first_conditions",
    [["fast"] * 4, ["fast", "slow", "slow", "slow"]],
    ids=["one_group", "group_of_one"],
)
def test_condition_order_is_omitted_when_the_groups_cannot_support_a_test(first_conditions):
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


def test_a_lapse_is_a_started_trial_without_a_choice_counted_under_its_labview_code():
    overall = lapse_summary(_lapse_session()).query("condition == 'all'").iloc[0]
    assert overall["n_started_trials"] == 4
    assert overall["n_lapse_trials"] == 2
    assert overall["lapse_rate"] == pytest.approx(0.5)
    assert overall["n_outcome_7006_reaction_time_too_long"] == 1
    assert overall["n_outcome_7011_delay_1_error"] == 1
    assert overall["n_lapse_other_outcomes"] == 0

    # An outcome code that is not a known lapse still counts as a lapse, but
    # under its own heading rather than being folded into a named one.
    unexpected = _lapse_session()
    unexpected.loc[1, "nOutcome"] = 7004
    unexpected = lapse_summary(unexpected).query("condition == 'all'").iloc[0]
    assert unexpected["n_lapse_trials"] == 2
    assert unexpected["n_outcome_7006_reaction_time_too_long"] == 0
    assert unexpected["n_lapse_other_outcomes"] == 1

    # A never-started trial is neither a lapse nor part of the denominator.
    never_started = _lapse_session()
    never_started.loc[1, "is_started"] = False
    never_started = lapse_summary(never_started).query("condition == 'all'").iloc[0]
    assert never_started["n_started_trials"] == 3
    assert never_started["n_lapse_trials"] == 1


# --- extreme decision times ---------------------------------------------------


def _extreme_session():
    return trial_features(
        run_trial_index=[1, 2, 3, 4, 5, 6],
        dt_ms=[900.0, 950.0, 1000.0, 1050.0, 1100.0, 20000.0],
    )


def test_an_extreme_decision_time_is_flagged_with_provenance_and_never_removed():
    counts, flagged = extreme_decision_times(_extreme_session(), mad_threshold=5.0)
    assert counts.iloc[0]["n_dt_trials"] == 6
    assert counts.iloc[0]["n_extreme_dt"] == 1
    assert counts.iloc[0]["n_extreme_slow"] == 1
    assert counts.iloc[0]["max_dt_ms"] == pytest.approx(20000.0)
    assert flagged["dt_ms"].tolist() == pytest.approx([20000.0])
    assert flagged.iloc[0]["robust_z"] > 5.0
    # Enough provenance to find the trial again in the raw log.
    row = flagged.iloc[0]
    assert row["trial_id"].endswith("trial-006")
    assert row["condition"] == "Fast"
    assert row["run_trial_index"] == 6
    assert row["trial_class_name"] == "easy"


def test_what_counts_as_extreme_is_the_threshold_and_needs_variability_to_exist():
    counts, flagged = extreme_decision_times(_extreme_session(), mad_threshold=1e6)
    assert counts.iloc[0]["n_extreme_dt"] == 0
    assert flagged.empty

    flat_counts, flat_flagged = extreme_decision_times(
        trial_features(run_trial_index=[1, 2, 3], dt_ms=[900.0] * 3)
    )
    assert flat_counts.iloc[0]["mad_dt_ms"] == pytest.approx(0.0)
    assert flat_counts.iloc[0]["n_extreme_dt"] == 0
    assert flat_flagged.empty

    # Anticipations are counted in their own right rather than as slow outliers.
    negative, _ = extreme_decision_times(
        trial_features(run_trial_index=[1, 2, 3, 4], dt_ms=[-100.0, 900.0, 950.0, 1000.0])
    )
    assert negative.iloc[0]["n_negative_dt"] == 1
    assert negative.iloc[0]["min_dt_ms"] == pytest.approx(-100.0)
