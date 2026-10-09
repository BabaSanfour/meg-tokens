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
    criterion_decline,
    early_evidence_order_profile,
    early_evidence_order_statistics,
    evidence_at_decision_responses,
    first_order_chosen_sum_log_lr,
    reverse_correlation,
)
from meg_tokens.behavior.math.evidence import FIRST_ORDER_TOKEN_LOG_LR

from tests.behavior.factories import trial_features


def _pooled(table):
    return table.loc[table["condition"] == "all"]


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


# --- early-evidence profile and token order -----------------------------------


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
