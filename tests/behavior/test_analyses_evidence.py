"""Tests for meg_tokens.behavior.analyses.evidence.

Grouped one test per behaviour rather than one per assertion: a fixture that
takes a second to build is shared by every claim it can support, and claims
that only make sense together (a decomposition and the identity it must
satisfy) are asserted together.
"""

import numpy as np
import pandas as pd
import pytest

from meg_tokens.behavior.analyses.evidence import (
    conditional_accuracy_functions,
    conditional_accuracy_statistics,
    continuous_evidence_effects,
    continuous_evidence_statistics,
    criterion_decline,
    early_evidence_order_profile,
    early_evidence_order_statistics,
    early_evidence_profile,
    evidence_at_decision_responses,
    first_order_chosen_sum_log_lr,
    first_order_criterion_decline,
    reverse_correlation,
    reverse_correlation_statistics,
)
from meg_tokens.behavior.math.evidence import FIRST_ORDER_TOKEN_LOG_LR

from tests.behavior.factories import TOKEN_DIRECTIONS, trial_features


def _pooled(table):
    return table.loc[table["condition"] == "all"]


def _stratum(table, stratum="all"):
    return _pooled(table).loc[table["trial_class_name"] == stratum]


def _across_subjects(build, n=4):
    """Run one per-subject builder over four subjects and stack the results."""
    return pd.concat(
        [build(f"H{index:02d}") for index in range(1, n + 1)], ignore_index=True
    )


# --- criterion decline --------------------------------------------------------


def test_criterion_decline_recovers_a_planted_slope_in_the_declared_units():
    tokens = [2, 4, 6, 8, 10, 12]
    per_token = _pooled(
        criterion_decline(
            trial_features(
                decision_token_index=tokens,
                logged_spd=[0.5 + 0.02 * value for value in tokens],
            )
        )
    ).iloc[0]
    assert per_token["slope"] == pytest.approx(0.02)
    assert per_token["intercept"] == pytest.approx(0.5)
    assert per_token["n_trials"] == 6
    assert per_token["converged"]

    # Against dt_ms the predictor is seconds, so 1000 ms is one unit.
    per_second = _pooled(
        criterion_decline(
            trial_features(
                dt_ms=[1000.0, 2000.0, 3000.0, 4000.0],
                logged_spd=[0.9, 0.8, 0.7, 0.6],
            ),
            predictor="dt_ms",
        )
    ).iloc[0]
    assert per_second["slope"] == pytest.approx(-0.1)
    assert per_second["intercept"] == pytest.approx(1.0)


def test_criterion_decline_reports_each_cell_including_the_ones_it_cannot_fit():
    fits = criterion_decline(
        trial_features(
            condition=["Fast"] * 4 + ["Slow"] * 4,
            run=[1] * 4 + [2] * 4,
            run_trial_index=[1, 2, 3, 4] * 2,
            decision_token_index=[2, 4, 6, 8] * 2,
            logged_spd=[0.9, 0.8, 0.7, 0.6, 0.9, 0.85, 0.8, 0.75],
        )
    ).set_index("condition")
    assert set(fits.index) == {"all", "fast", "slow"}
    assert fits.loc["fast", "slope"] == pytest.approx(-0.05)
    assert fits.loc["slow", "slope"] == pytest.approx(-0.025)

    # A condition the subject never ran is a reported empty cell, not a gap.
    one_condition = criterion_decline(
        trial_features(
            condition=["Fast"] * 4,
            decision_token_index=[2, 4, 6, 8],
            logged_spd=[0.9, 0.8, 0.7, 0.6],
        )
    ).set_index("condition")
    assert one_condition.loc["slow", "n_trials"] == 0
    assert not one_condition.loc["slow", "converged"]

    # A predictor with no variance is unestimable, and says so rather than
    # reporting a zero slope.
    flat = _pooled(
        criterion_decline(
            trial_features(decision_token_index=[5, 5, 5], logged_spd=[0.8, 0.7, 0.9])
        )
    ).iloc[0]
    assert not flat["converged"]
    assert np.isnan(flat["slope"])


def test_the_two_evidence_scales_stay_distinct_and_errors_reverse_into_the_chosen_frame():
    fits = evidence_at_decision_responses(
        trial_features(
            decision_token_index=[2, 4, 6, 8], logged_spd=[0.9, 0.8, 0.7, 0.6]
        ),
        predictor="decision_token_index",
    )
    assert set(fits["response"]) == {"logged_spd", "logged_spd_log_odds"}
    assert set(fits["predictor"]) == {"decision_token_index"}
    slopes = fits.loc[fits["condition"] == "all"].set_index("response")["slope"]
    assert slopes["logged_spd"] < 0 and slopes["logged_spd_log_odds"] < 0
    assert slopes["logged_spd"] != pytest.approx(slopes["logged_spd_log_odds"])

    # First-order SumLogLR is the *chosen*-target lead, so an error flips sign.
    values = first_order_chosen_sum_log_lr(
        trial_features(token_lead_at_decision=[2, 2], isCorrect=[True, False])
    )
    assert values.tolist() == pytest.approx(
        [2 * FIRST_ORDER_TOKEN_LOG_LR, -2 * FIRST_ORDER_TOKEN_LOG_LR]
    )


def test_the_first_order_criterion_uses_continuous_time_and_alignable_trials_only():
    fits = first_order_criterion_decline(
        trial_features(
            dt_ms=[-50, 400, 800, 1200, 1600],
            decision_token_index=[0, 2, 4, 6, 8],
            token_lead_at_decision=[0, 4, 3, 99, 1],
            design_time_alignment_valid=[True, True, True, False, True],
        )
    )
    overall = _pooled(fits).iloc[0]
    # The anticipation and the unalignable trial are both excluded.
    assert overall["n_trials"] == 3
    assert overall["predictor"] == "dt_ms"
    assert overall["slope"] == pytest.approx(-2.5 * FIRST_ORDER_TOKEN_LOG_LR)


# --- reverse correlation ------------------------------------------------------


def _kernel_subject(subject, *, n_jumps=2, per_condition=False):
    if per_condition:
        return reverse_correlation(
            trial_features(
                subject=[subject] * 12,
                condition=["Fast"] * 6 + ["Slow"] * 6,
                run=[1] * 6 + [2] * 6,
                run_trial_index=[1, 2, 3, 4, 5, 6] * 2,
                decision_token_index=[3] * 12,
                choice_side=[1, 1, 1, 2, 2, 2] * 2,
                token_directions=(["1" * 15] * 3 + ["2" * 15] * 3) * 2,
            ),
            n_jumps=n_jumps,
        )
    return reverse_correlation(
        trial_features(
            subject=[subject] * 6,
            decision_token_index=[3] * 6,
            choice_side=[1, 1, 1, 2, 2, 2],
            token_directions=["1" * 15] * 3 + ["2" * 15] * 3,
        ),
        n_jumps=n_jumps,
    )


def test_the_kernel_weights_only_tokens_the_subject_had_actually_seen():
    # Every trial commits after two tokens, so jumps 3+ were never visible.
    kernels = _pooled(
        reverse_correlation(
            trial_features(
                decision_token_index=[2] * 6,
                choice_side=[1, 1, 1, 2, 2, 2],
                token_directions=["112" + "2" * 12] * 6,
            ),
            n_jumps=4,
        )
    ).set_index("jump")
    assert kernels.loc[1, "n_trials_token_seen"] == 6
    assert kernels.loc[3, "n_trials_token_seen"] == 0
    assert np.isnan(kernels.loc[3, "mean_signed_direction"])
    # Both early tokens went to target 1: +1 for subjects who chose it and -1
    # for those who chose 2, which averages to zero over an even split.
    assert kernels.loc[1, "mean_signed_direction"] == pytest.approx(0.0)

    # A token that always predicts the choice gets a kernel of exactly one.
    perfect = _pooled(_kernel_subject("H01", n_jumps=3)).set_index("jump")
    assert perfect.loc[1, "mean_signed_direction"] == pytest.approx(1.0)


def test_the_kernel_spans_the_requested_jumps_and_marks_what_it_cannot_fit():
    kernels = _pooled(
        reverse_correlation(
            trial_features(
                decision_token_index=[5] * 4,
                choice_side=[1, 1, 2, 2],
                token_directions=[TOKEN_DIRECTIONS] * 4,
            ),
            n_jumps=6,
        )
    )
    assert kernels["jump"].tolist() == [1, 2, 3, 4, 5, 6]

    # With one choice made by everyone the logistic has no variance to fit; the
    # model-free kernel is still reported.
    degenerate = _pooled(
        reverse_correlation(
            trial_features(
                decision_token_index=[5] * 4,
                choice_side=[1, 1, 1, 1],
                token_directions=[TOKEN_DIRECTIONS] * 4,
            ),
            n_jumps=3,
        )
    )
    assert not degenerate["converged"].any()
    assert degenerate["logistic_weight"].isna().all()
    assert degenerate["mean_signed_direction"].notna().all()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"choice_side": [3, 1, 2, 1]}, "choice_side must be 1 or 2"),
        ({"decision_token_index": [5, None, 5, 5]}, "decision_token_index must not be missing"),
        ({}, "n_jumps must be at least 1"),
    ],
)
def test_reverse_correlation_refuses_to_guess(overrides, message):
    columns = {"decision_token_index": [5] * 4, "choice_side": [1, 1, 2, 2]}
    columns.update(overrides)
    with pytest.raises(ValueError, match=message):
        reverse_correlation(trial_features(**columns), n_jumps=0 if not overrides else 3)


def test_kernel_statistics_register_every_quantity_a_kernel_figure_needs():
    """Both metrics at every jump, the difference's own SEM and interval, and
    the corrected threshold — a figure that had to recover any of these itself
    would compute an inferential statistic in the plotting layer."""
    statistics = reverse_correlation_statistics(
        _across_subjects(lambda s: _kernel_subject(s, n_jumps=4))
    )

    assert set(statistics["metric"]) == {"logistic_weight", "mean_signed_direction"}
    per_jump = statistics.loc[statistics["test"] != "all_jumps_vs_zero"]
    assert set(per_jump["jump"]) == {1, 2, 3, 4}
    # The across-jump summary deliberately carries no jump.
    assert statistics.loc[statistics["test"] == "all_jumps_vs_zero", "jump"].isna().all()

    signed = statistics.loc[
        (statistics["metric"] == "mean_signed_direction")
        & (statistics["condition"] == "all")
        & (statistics["jump"] == 1)
    ].iloc[0]
    assert signed["n_subjects"] == 4
    assert signed["mean"] == pytest.approx(1.0)

    # Eight independent per-jump tests need a stated threshold, and the count
    # of significant jumps is registered so the report never reduces a p column.
    assert (statistics["n_jumps_tested"] == 4).all()
    assert statistics["p_threshold"].eq(0.05 / 4).all()
    summary = statistics.loc[
        (statistics["test"] == "all_jumps_vs_zero")
        & (statistics["metric"] == "logistic_weight")
        & (statistics["condition"] == "all")
    ].iloc[0]
    assert 0 <= summary["n_jumps_significant"] <= 4

    # A paired difference needs the SEM *of the difference*: recovering it as
    # mean / t in figure code fails at exactly zero, the case a kernel figure
    # most needs to draw.
    contrast = reverse_correlation_statistics(
        _across_subjects(lambda s: _kernel_subject(s, per_condition=True))
    )
    contrast = contrast.loc[contrast["condition"] == "fast_vs_slow"]
    assert {"sem_difference", "equivalence_bound", "ci_low", "ci_high"}.issubset(
        contrast.columns
    )


# --- conditional accuracy -----------------------------------------------------


def _falling_accuracy_functions():
    return _across_subjects(
        lambda s: conditional_accuracy_functions(
            trial_features(
                subject=[s] * 6,
                dt_ms=[500.0, 700.0, 900.0, 1100.0, 1300.0, 1500.0],
                isCorrect=[True, True, True, True, False, False],
                run_trial_index=[1, 2, 3, 4, 5, 6],
            ),
            n_bins=3,
        )
    )


def _single_bin_functions():
    return conditional_accuracy_functions(
        trial_features(
            subject=["H01"] * 4 + ["H02"] * 4,
            run_trial_index=[1, 2, 3, 4] * 2,
            dt_ms=[900.0, 1000.0, 1100.0, 1200.0] * 2,
            isCorrect=[True, True, False, False] * 2,
        ),
        n_bins=1,
    )


def _composition_driven_functions():
    """Eight subjects whose pooled accuracy falls mostly because the fast bins
    hold easy trials and the slow bins ambiguous ones. Both classes appear in
    every bin, without which a subject cannot be decomposed at all, and which
    subject's ambiguous trials go wrong varies — that variation is what the
    bootstrap resamples."""
    names = (
        ["easy"] * 3 + ["ambiguous"]
        + ["easy"] * 2 + ["ambiguous"] * 2
        + ["easy"] + ["ambiguous"] * 3
    )
    subjects = []
    for index in range(8):
        correct = [name == "easy" for name in names]
        correct[3 + index % 4] = index % 2 == 0
        correct[7 + index % 4] = False
        subjects.append(
            trial_features(
                subject=[f"H{index + 1:02d}"] * len(names),
                run_trial_index=list(range(1, len(names) + 1)),
                dt_ms=[
                    400.0, 450.0, 500.0, 550.0,
                    900.0, 950.0, 1000.0, 1050.0,
                    1400.0, 1450.0, 1500.0, 1550.0,
                ],
                trial_class=[1 if name == "easy" else 2 for name in names],
                trial_class_name=names,
                isCorrect=correct,
            )
        )
    return conditional_accuracy_functions(
        pd.concat(subjects, ignore_index=True), n_bins=3
    )


def test_decision_time_bins_are_cut_inside_each_subject_and_pooled_across_classes():
    # A globally slow subject must not be pushed entirely into the slow bin.
    within_subject = _stratum(
        conditional_accuracy_functions(
            trial_features(
                subject=["H01"] * 4 + ["H02"] * 4,
                dt_ms=[500.0, 700.0, 900.0, 1100.0, 5000.0, 5200.0, 5400.0, 5600.0],
                isCorrect=[True, True, False, False] * 2,
                run_trial_index=[1, 2, 3, 4] * 2,
            ),
            n_bins=2,
        )
    )
    assert set(within_subject["dt_bin"]) == {1, 2}
    assert within_subject.loc[within_subject["dt_bin"] == 1, "accuracy"].tolist() == [1.0, 1.0]
    assert within_subject.loc[within_subject["dt_bin"] == 2, "accuracy"].tolist() == [0.0, 0.0]

    # Bins are cut on all of a subject's trials, so class rows partition the
    # pooled row and every class is read at the same decision times.
    classed = _pooled(
        conditional_accuracy_functions(
            trial_features(
                dt_ms=[500.0, 700.0, 900.0, 1100.0],
                isCorrect=[True, True, False, False],
                trial_class=[1, 2, 1, 2],
                trial_class_name=["easy", "ambiguous", "easy", "ambiguous"],
            ),
            n_bins=2,
        )
    )
    first_bin = classed.loc[classed["dt_bin"] == 1].set_index("trial_class_name")
    assert first_bin.loc["all", "n_trials"] == 2
    assert first_bin.loc[["easy", "ambiguous"], "n_trials"].tolist() == [1, 1]
    assert first_bin.loc[["easy", "ambiguous"], "proportion_of_bin"].sum() == 1.0
    assert first_bin.loc["ambiguous", "mean_dt_ms"] == pytest.approx(700.0)
    # The class-adjusted column is only defined on the pooled row.
    assert classed.loc[classed["trial_class_name"] != "all", "accuracy_class_adjusted"].isna().all()

    counts = _stratum(
        conditional_accuracy_functions(
            trial_features(dt_ms=[500.0, 700.0, 900.0, 1100.0], isCorrect=[True, True, True, False]),
            n_bins=2,
        )
    )
    assert counts["n_trials"].tolist() == [2, 2]
    assert counts.loc[counts["dt_bin"] == 1, "mean_dt_ms"].iloc[0] == pytest.approx(600.0)


def test_unclassified_trials_are_a_stratum_and_class_adjustment_removes_composition():
    unclassified = _pooled(
        conditional_accuracy_functions(
            trial_features(
                dt_ms=[500.0, 700.0, 900.0, 1100.0],
                isCorrect=[True, True, True, False],
                trial_class=[1, 0, 1, 0],
                trial_class_name=["easy", "unclassified", "easy", "unclassified"],
            ),
            n_bins=2,
        )
    )
    assert "unclassified" in set(unclassified["trial_class_name"])
    assert unclassified.loc[
        unclassified["trial_class_name"] == "unclassified", "n_trials"
    ].sum() == 2

    # Both classes are flat in time; only their bin shares differ, so observed
    # accuracy falls and the class-adjusted value does not move at all.
    adjusted = _stratum(
        conditional_accuracy_functions(
            trial_features(
                dt_ms=[500.0, 600.0, 700.0, 1000.0, 1100.0, 1200.0],
                isCorrect=[True, True, False, True, False, False],
                trial_class=[1, 1, 2, 1, 2, 2],
                trial_class_name=["easy", "easy", "ambiguous", "easy", "ambiguous", "ambiguous"],
                run_trial_index=[1, 2, 3, 4, 5, 6],
            ),
            n_bins=2,
        )
    ).sort_values("dt_bin")
    assert adjusted["accuracy"].tolist() == [pytest.approx(2 / 3), pytest.approx(1 / 3)]
    assert adjusted["accuracy_class_adjusted"].tolist() == [pytest.approx(0.5)] * 2


@pytest.mark.parametrize(
    ("columns", "n_bins", "message"),
    [
        ({"dt_ms": [500.0, 900.0], "isCorrect": ["true", "false"]}, 2, "isCorrect must be boolean"),
        ({"dt_ms": [900.0]}, 0, "n_bins must be at least 1"),
    ],
)
def test_conditional_accuracy_refuses_to_guess(columns, n_bins, message):
    with pytest.raises(ValueError, match=message):
        conditional_accuracy_functions(trial_features(**columns), n_bins=n_bins)


def test_a_subset_that_cannot_be_binned_is_omitted_rather_than_collapsed():
    features = trial_features(dt_ms=[900.0] * 3, isCorrect=[True, True, False])
    assert conditional_accuracy_functions(features, n_bins=3).empty


def test_conditional_accuracy_statistics_report_the_trend_in_both_units_against_chance():
    statistics = conditional_accuracy_statistics(_falling_accuracy_functions())
    assert set(statistics["term"]) >= {
        "mean_accuracy", "accuracy_slope_per_bin", "accuracy_slope_per_second",
    }
    selection = (
        (statistics["measure"] == "accuracy")
        & (statistics["condition"] == "all")
        & (statistics["trial_class_name"] == "all")
        & (statistics["min_cell_trials"] == 1)
    )
    per_bin = statistics.loc[selection & (statistics["term"] == "accuracy_slope_per_bin")].iloc[0]
    per_second = statistics.loc[selection & (statistics["term"] == "accuracy_slope_per_second")].iloc[0]
    assert per_bin["n_subjects"] == 4
    assert per_bin["estimate"] < 0
    # Bins are 400 ms apart here, so a per-second slope is 2.5x the per-bin one.
    assert per_second["estimate"] == pytest.approx(per_bin["estimate"] * 2.5)

    # Every slope states the bound it rules out, and at which coverage — a
    # bound silently relabelled 95% is the defect this guards.
    slopes = statistics.loc[statistics["term"] == "accuracy_slope_per_bin"]
    assert (slopes["equivalence_bound"] > 0).all()
    assert (slopes["coverage"] == 0.95).all()
    assert (slopes["equivalence_bound_coverage"] == 0.90).all()

    # A 2AFC accuracy of 0.5 carries no information; against zero it is huge.
    # Subject bin accuracies of 1.0, 0.0, 0.5, 0.5: mean exactly chance.
    chance = conditional_accuracy_statistics(
        pd.concat(
            [
                conditional_accuracy_functions(
                    trial_features(
                        subject=[f"H{index:02d}"] * 4,
                        dt_ms=[500.0, 700.0, 900.0, 1100.0],
                        isCorrect=outcome,
                        run_trial_index=[1, 2, 3, 4],
                    ),
                    n_bins=2,
                )
                for index, outcome in enumerate(
                    [
                        [True, True, True, True],
                        [False, False, False, False],
                        [True, False, True, False],
                        [False, True, False, True],
                    ],
                    start=1,
                )
            ],
            ignore_index=True,
        )
    )
    chance = chance.loc[chance["term"] == "mean_accuracy"]
    assert set(chance["test"]) == {"one_sample_vs_chance"}
    assert chance["estimate"].tolist() == [pytest.approx(0.5)] * len(chance)
    assert chance["t"].abs().max() == pytest.approx(0.0)


def test_the_change_decomposition_is_an_identity_and_names_what_dropping_unclassified_leaves():
    statistics = conditional_accuracy_statistics(_falling_accuracy_functions())
    rows = statistics.loc[
        (statistics["term"] == "accuracy_change_decomposition")
        & (statistics["condition"] == "all")
        & (statistics["trial_class_name"] == "all")
    ].set_index("measure")
    # Composition plus within-class is exact, not approximate: if it were
    # approximate the residual could absorb a mistake.
    total = rows.loc["composition", "estimate"] + rows.loc["within_class", "estimate"]
    assert total == pytest.approx(rows.loc["composition", "observed_change"])
    assert rows.loc["composition", "share_of_observed_change"] + rows.loc[
        "within_class", "share_of_observed_change"
    ] == pytest.approx(1.0)

    # The `designed_classes` row is why the figure draws four strata: it has to
    # be a registered number rather than an argument made in prose.
    crowded = _across_subjects(
        lambda s: trial_features(
            subject=[s] * 8,
            run_trial_index=list(range(1, 9)),
            dt_ms=[500.0, 600.0, 700.0, 800.0, 1200.0, 1300.0, 1400.0, 1500.0],
            trial_class=[1, 1, 1, 0, 1, 0, 0, 0],
            trial_class_name=["easy"] * 3 + ["unclassified"] + ["easy"] + ["unclassified"] * 3,
            isCorrect=[True] * 5 + [False, True, False],
        )
    )
    within = conditional_accuracy_statistics(
        conditional_accuracy_functions(crowded, n_bins=2)
    )
    within = within.loc[
        (within["term"] == "accuracy_change_decomposition")
        & (within["measure"] == "within_class")
        & (within["condition"] == "all")
    ].set_index("trial_class_name")
    assert "designed_classes" in within.index
    # Dropping unclassified drops the stratum carrying the decline, so what
    # remains runs the other way.
    assert within.loc["unclassified", "estimate"] < 0
    assert within.loc["designed_classes", "estimate"] >= 0
    assert within.loc["designed_classes", "estimate"] != pytest.approx(within.loc["all", "estimate"])


def test_a_subject_missing_a_class_from_an_end_bin_drops_out_of_the_decomposition():
    """An absent class has a proportion of zero but no accuracy, so neither
    term is defined — the subject drops out rather than contributing a zero."""
    complete = _across_subjects(
        lambda s: trial_features(
            subject=[s] * 8,
            run_trial_index=list(range(1, 9)),
            dt_ms=[500.0, 600.0, 700.0, 800.0, 1200.0, 1300.0, 1400.0, 1500.0],
            trial_class=[1, 2, 1, 2, 1, 2, 1, 2],
            trial_class_name=["easy", "ambiguous"] * 4,
            isCorrect=[True, True, True, False, True, False, False, False],
        )
    )
    partial = trial_features(
        subject=["H99"] * 8,
        run_trial_index=list(range(1, 9)),
        dt_ms=[500.0, 600.0, 700.0, 800.0, 1200.0, 1300.0, 1400.0, 1500.0],
        trial_class=[1] * 8,
        trial_class_name=["easy"] * 8,
        isCorrect=[True] * 5 + [False] * 3,
    )
    statistics = conditional_accuracy_statistics(
        conditional_accuracy_functions(
            pd.concat([complete, partial], ignore_index=True), n_bins=2
        )
    )
    row = statistics.loc[
        (statistics["term"] == "accuracy_change_decomposition")
        & (statistics["measure"] == "composition")
        & (statistics["condition"] == "all")
        & (statistics["trial_class_name"] == "all")
    ].iloc[0]
    assert row["n_subjects"] == 4


def test_a_single_realized_bin_emits_the_means_but_no_decomposition():
    statistics = conditional_accuracy_statistics(_single_bin_functions())
    assert "mean_accuracy" in set(statistics["term"])
    assert "accuracy_change_decomposition" not in set(statistics["term"])
    assert "composition_share_agreement" not in set(statistics["term"])


def test_the_two_composition_routes_reproduce_their_sources_and_only_their_difference_is_tested():
    statistics = conditional_accuracy_statistics(_composition_driven_functions())

    def agreement(measure, slope_unit="accuracy_slope_per_bin"):
        return statistics.loc[
            (statistics["term"] == "composition_share_agreement")
            & (statistics["measure"] == measure)
            & (statistics["condition"] == "all")
            & (statistics["slope_unit"] == slope_unit)
        ].iloc[0]

    # Each route must be the *same* quantity the write-up already quotes
    # elsewhere in this table, or the comparison is of something nobody claimed.
    decomposition = statistics.loc[
        (statistics["term"] == "accuracy_change_decomposition")
        & (statistics["measure"] == "composition")
        & (statistics["condition"] == "all")
        & (statistics["trial_class_name"] == "all")
    ].iloc[0]
    assert agreement("endpoint_share")["estimate"] == pytest.approx(
        decomposition["share_of_observed_change"]
    )
    slopes = statistics.loc[
        (statistics["term"] == "accuracy_slope_per_bin")
        & (statistics["condition"] == "all")
        & (statistics["trial_class_name"] == "all")
        & (statistics["min_cell_trials"] == 1)
    ].set_index("measure")
    removed = abs(
        slopes.loc["accuracy_class_adjusted", "estimate"] / slopes.loc["accuracy", "estimate"]
    )
    assert agreement("slope_share")["estimate"] == pytest.approx(1.0 - removed)
    assert agreement("endpoint_minus_slope_share")["estimate"] == pytest.approx(
        agreement("endpoint_share")["estimate"] - agreement("slope_share")["estimate"]
    )

    # A share has no null at zero — "this route attributes none of the decline
    # to composition" is not a hypothesis — so a p there would only ever be
    # quoted by mistake. The interval is the whole content of those rows.
    for measure in ("endpoint_share", "slope_share"):
        row = agreement(measure)
        assert row["test"] == "subject_bootstrap_descriptive"
        assert pd.isna(row["p"]) and pd.isna(row["equivalence_bound"])
        assert row["ci_low"] < row["estimate"] < row["ci_high"]

    # The routes agreeing to a fraction of a point is worth nothing unless the
    # comparison could have resolved a large difference, which is the bound.
    difference = agreement("endpoint_minus_slope_share")
    assert difference["test"] == "subject_bootstrap_vs_zero"
    assert difference["equivalence_bound"] == pytest.approx(
        max(abs(difference["ci_low"]), abs(difference["ci_high"])), rel=0.5
    )
    assert difference["equivalence_bound_coverage"] == 0.90
    assert difference["coverage"] == 0.95

    # The endpoint route does not depend on the slope unit; the slope route
    # does, and if the two units disagreed neither could be quoted.
    assert agreement("endpoint_share")["estimate"] == pytest.approx(
        agreement("endpoint_share", "accuracy_slope_per_second")["estimate"]
    )
    assert {"accuracy_slope_per_bin", "accuracy_slope_per_second"} <= set(
        statistics.loc[statistics["term"] == "composition_share_agreement", "slope_unit"]
    )


# --- continuous evidence ------------------------------------------------------


def test_continuous_evidence_uses_every_task_trial_and_a_declared_neutral_point():
    # Unclassified trials are kept: this analysis is the one that does not
    # depend on the class rule.
    kept = _pooled(
        continuous_evidence_effects(
            trial_features(
                trial_class=[1, 2, 0, 0],
                trial_class_name=["easy", "ambiguous", "unclassified", "unclassified"],
                sp_design_early=[0.9, 0.6, 0.5, 0.8],
                dt_ms=[800.0, 1200.0, 1400.0, 900.0],
                isCorrect=[True, True, False, True],
            ),
            predictors={"sp_design_early": 0.5},
        )
    ).iloc[0]
    assert kept["n_dt_trials"] == 4

    # Evidence strength is distance from the *declared* neutral point. These
    # trials speed up as evidence moves away from 0.5 in either direction,
    # which only a centred predictor can see.
    features = trial_features(
        sp_design_early=[0.1, 0.3, 0.5, 0.7, 0.9],
        dt_ms=[1000.0, 1200.0, 1400.0, 1200.0, 1000.0],
        isCorrect=[True, False, False, True, True],
    )
    centred = _pooled(
        continuous_evidence_effects(features, predictors={"sp_design_early": 0.5})
    ).iloc[0]
    uncentred = _pooled(
        continuous_evidence_effects(features, predictors={"sp_design_early": 0.0})
    ).iloc[0]
    assert centred["dt_slope_ms_per_unit"] == pytest.approx(-1000.0)
    assert uncentred["dt_slope_ms_per_unit"] == pytest.approx(0.0, abs=1e-9)


def test_continuous_evidence_fits_each_declared_predictor_and_marks_what_it_cannot():
    monotone = trial_features(
        sp_design_early=[0.5, 0.6, 0.7, 0.8, 0.9],
        sum_log_lr_design_early=[0.0, 0.4, 0.8, 1.2, 1.6],
        dt_ms=[1400.0, 1300.0, 1200.0, 1100.0, 1000.0],
        isCorrect=[False, True, True, True, True],
    )
    assert set(_pooled(continuous_evidence_effects(monotone))["predictor"]) == {
        "sp_design_early", "sum_log_lr_design_early",
    }
    # Distance from chance, so a 0.1 rise in SP removes 100 ms.
    overall = _pooled(
        continuous_evidence_effects(monotone, predictors={"sp_design_early": 0.5})
    ).iloc[0]
    assert overall["dt_slope_ms_per_unit"] == pytest.approx(-1000.0)

    # Accuracy rises with signed evidence without separating it perfectly,
    # which a logistic fit cannot estimate.
    accuracy = _pooled(
        continuous_evidence_effects(
            trial_features(
                sp_design_early=[0.2, 0.3, 0.4, 0.6, 0.7, 0.8],
                dt_ms=[1400.0, 1300.0, 1250.0, 1150.0, 1100.0, 1000.0],
                isCorrect=[False, True, False, False, True, True],
            ),
            predictors={"sp_design_early": 0.5},
        )
    ).iloc[0]
    assert accuracy["n_accuracy_trials"] == 6
    assert accuracy["accuracy_log_odds_per_unit"] > 0

    flat = _pooled(
        continuous_evidence_effects(
            trial_features(
                sp_design_early=[0.8, 0.8, 0.8],
                dt_ms=[900.0, 1000.0, 1100.0],
                isCorrect=[True, True, True],
            ),
            predictors={"sp_design_early": 0.5},
        )
    ).iloc[0]
    assert not flat["converged"]
    assert np.isnan(flat["dt_slope_ms_per_unit"])
    assert np.isnan(flat["accuracy_log_odds_per_unit"])


def test_continuous_evidence_statistics_test_both_coefficients_and_record_their_level_count():
    fits = _across_subjects(
        lambda s: continuous_evidence_effects(
            trial_features(
                subject=[s] * 10,
                condition=["Fast"] * 5 + ["Slow"] * 5,
                run=[1] * 5 + [2] * 5,
                run_trial_index=[1, 2, 3, 4, 5] * 2,
                sp_design_early=[0.5, 0.6, 0.7, 0.8, 0.9] * 2,
                dt_ms=(
                    [1400.0 - 100 * step * int(s[1:]) for step in range(5)]
                    + [1600.0 - 80 * step * int(s[1:]) for step in range(5)]
                ),
                isCorrect=[False, True, True, True, True] * 2,
            ),
            predictors={"sp_design_early": 0.5},
        )
    )
    statistics = continuous_evidence_statistics(fits)
    assert set(statistics["term"]) == {"dt_slope_ms_per_unit", "accuracy_log_odds_per_unit"}
    assert "fast_vs_slow" in set(statistics["condition"])
    dt_trend = statistics.loc[
        (statistics["term"] == "dt_slope_ms_per_unit") & (statistics["condition"] == "all")
    ].iloc[0]
    assert dt_trend["n_subjects"] == 4
    assert dt_trend["mean"] < 0

    # A per-unit slope over two levels is a difference of means; the count is
    # what stops it being read as a tested linear trend. Signed evidence keeps
    # four levels here; its distance from chance has two.
    coarse = trial_features(
        sp_design_early=[0.2, 0.4, 0.6, 0.8],
        dt_ms=[1400.0, 1300.0, 1200.0, 1000.0],
        isCorrect=[False, False, True, True],
    )
    effects = continuous_evidence_effects(coarse, predictors={"sp_design_early": 0.5})
    assert _pooled(effects).iloc[0]["n_signed_levels"] == 4
    assert _pooled(effects).iloc[0]["n_strength_levels"] == 2
    levels = continuous_evidence_statistics(effects).set_index(["term", "condition"])[
        "n_predictor_levels"
    ]
    assert levels[("dt_slope_ms_per_unit", "all")] == 2
    assert levels[("accuracy_log_odds_per_unit", "all")] == 4


# --- early-evidence profile and token order -----------------------------------


def _early_evidence_features(subject="H01"):
    """Two classes sharing an early-evidence level and diverging after it.

    Six trials per class-by-level cell, which is what ``SPARSE_CELL_MIN_TRIALS``
    requires before a contrast will read a cell. At the shared level (0.6) easy
    is fast and ambiguous slow; both speed up by 200 ms at 0.8, so early
    evidence and class are separable here by construction.
    """
    cell = 6
    return trial_features(
        subject=[subject] * (4 * cell),
        run_trial_index=list(range(1, 4 * cell + 1)),
        sp_design_early=([0.6] * cell + [0.8] * cell) * 2,
        trial_class=[1] * (2 * cell) + [2] * (2 * cell),
        trial_class_name=["easy"] * (2 * cell) + ["ambiguous"] * (2 * cell),
        dt_ms=(
            [1000.0, 1100.0] * (cell // 2) + [800.0, 900.0] * (cell // 2)
            + [1400.0, 1500.0] * (cell // 2) + [1200.0, 1300.0] * (cell // 2)
        ),
        isCorrect=[True] * (4 * cell - 1) + [False],
    )


def _order_features(subject="H01"):
    """One class reaching a 2-1 lead by two different token orders.

    "112..." leads with the correct target and "211..." does not: same lead
    after three tokens, opposite first jump. A correct first jump is slower
    here, as it is in the real data.
    """
    cell = 6
    return trial_features(
        subject=[subject] * (2 * cell),
        run_trial_index=list(range(1, 2 * cell + 1)),
        sp_design_early=[0.6] * (2 * cell),
        trial_class=[1] * (2 * cell),
        trial_class_name=["easy"] * (2 * cell),
        correct_side=[1] * (2 * cell),
        token_directions=["112121212121212"] * cell + ["211121212121212"] * cell,
        dt_ms=[1200.0] * cell + [1000.0] * cell,
        isCorrect=[True] * (2 * cell - 1) + [False],
    )


def test_the_profile_resolves_each_level_by_class_without_inventing_absent_cells():
    profile = _pooled(early_evidence_profile(_early_evidence_features()))
    assert sorted(profile["sp_design_early"].unique()) == [0.6, 0.8]
    shared = profile.loc[profile["sp_design_early"] == 0.6].set_index("trial_class_name")
    # The class rows partition the pooled row at every level.
    assert shared.loc["all", "n_trials"] == 12
    assert shared.loc[["easy", "ambiguous"], "n_trials"].tolist() == [6, 6]
    assert shared.loc["easy", "mean_dt_ms"] == pytest.approx(1050.0)
    assert shared.loc["ambiguous", "mean_dt_ms"] == pytest.approx(1450.0)
    # `sum_log_lr_design_early` is logit(sp), the exact posterior — not the
    # first-order SumLogLR, which is a different scale.
    level = profile.loc[profile["sp_design_early"] == 0.8].iloc[0]
    assert level["sum_log_lr_design_early"] == pytest.approx(np.log(0.8 / 0.2))

    sparse = _pooled(
        early_evidence_profile(
            trial_features(
                sp_design_early=[0.6, 0.6, 0.8, 0.8],
                trial_class=[1, 1, 2, 2],
                trial_class_name=["easy", "easy", "ambiguous", "ambiguous"],
                dt_ms=[900.0, 1000.0, 1300.0, 1400.0],
                isCorrect=[True, True, False, True],
            )
        )
    )
    assert set(sparse.loc[sparse["sp_design_early"] == 0.6, "trial_class_name"]) == {"all", "easy"}


def test_the_level_contrasts_are_within_subject_and_never_reach_a_thin_cell():
    features = _across_subjects(_early_evidence_features)
    statistics = continuous_evidence_statistics(
        continuous_evidence_effects(features), early_evidence_profile(features)
    )
    contrast = statistics.loc[
        (statistics["term"] == "class_contrast_at_level")
        & (statistics["measure"] == "mean_dt_ms")
        & (statistics["sp_design_early"] == 0.6)
    ].iloc[0]
    assert contrast["trial_class_name"] == "ambiguous_vs_easy"
    assert contrast["n_subjects"] == 4
    assert contrast["mean_difference"] == pytest.approx(400.0)

    # The honest form of dt_slope_ms_per_unit: a level-to-level contrast in
    # milliseconds, holding the class fixed.
    within = statistics.loc[
        (statistics["term"] == "evidence_contrast_within_class")
        & (statistics["measure"] == "mean_dt_ms")
        & (statistics["trial_class_name"] == "easy")
    ].iloc[0]
    assert within["sp_design_early"] == 0.8
    assert within["sp_design_early_reference"] == 0.6
    assert within["mean_difference"] == pytest.approx(-200.0)

    # A one-trial cell would silently restrict a contrast to whoever owns one,
    # so cells below the threshold are dropped before any contrast — but the
    # level is still kept in the derivative.
    thin = _across_subjects(
        lambda s: trial_features(
            subject=[s] * 13,
            run_trial_index=list(range(1, 14)),
            sp_design_early=[0.6] * 6 + [0.8] * 6 + [0.2],
            trial_class=[1] * 13,
            trial_class_name=["easy"] * 13,
            dt_ms=[1200.0] * 6 + [1000.0] * 6 + [700.0],
            isCorrect=[True] * 13,
        )
    )
    profile = early_evidence_profile(thin)
    assert 0.2 in set(profile["sp_design_early"])
    thin_within = continuous_evidence_statistics(
        continuous_evidence_effects(thin), profile
    )
    thin_within = thin_within.loc[
        (thin_within["term"] == "evidence_contrast_within_class")
        & (thin_within["measure"] == "mean_dt_ms")
    ].iloc[0]
    assert thin_within["sp_design_early_reference"] == 0.6
    assert thin_within["mean_difference"] == pytest.approx(-200.0)


def test_token_order_is_coded_against_the_correct_target_and_contrasts_hold_the_level_fixed():
    profile = _pooled(early_evidence_order_profile(_order_features()))
    assert set(profile["token_order"]) == {"CCW", "WCC"}
    assert set(profile["first_token"]) == {"correct", "wrong"}
    # Three characters, one per jump, over the window sp_design_early covers.
    assert profile["token_order"].str.len().eq(3).all()

    # Coded against the correct target, not a fixed side — otherwise the code
    # would mean opposite things on left- and right-correct trials.
    left = early_evidence_order_profile(
        _order_features().assign(correct_side=1, nCorrectChoice=1)
    )
    mirrored = _order_features().assign(correct_side=2, nCorrectChoice=2)
    mirrored["token_directions"] = mirrored["token_directions"].str.translate(
        str.maketrans("12", "21")
    )
    assert set(left["token_order"]) == set(
        early_evidence_order_profile(mirrored)["token_order"]
    )

    features = _across_subjects(_order_features)
    statistics = early_evidence_order_statistics(early_evidence_order_profile(features))
    row = statistics.loc[
        (statistics["term"] == "first_token_contrast")
        & (statistics["measure"] == "mean_dt_ms")
    ].iloc[0]
    assert row["trial_class_name"] == "easy"
    assert row["sp_design_early"] == 0.6
    assert row["n_subjects"] == 4
    # Correct-first minus wrong-first, the direction the real data shows.
    assert row["mean_difference"] == pytest.approx(200.0)

    # The fact the write-up leans on: classes at one level can hold disjoint
    # orders, so "matched on early evidence" is true only of the summary.
    only_correct_first = features.loc[features["token_directions"].str.startswith("1")]
    shares = early_evidence_order_statistics(
        early_evidence_order_profile(only_correct_first)
    )
    shares = shares.loc[shares["term"] == "order_composition"]
    assert shares["token_order"].tolist() == ["CCW"]
    assert shares["estimate"].iloc[0] == pytest.approx(1.0)
