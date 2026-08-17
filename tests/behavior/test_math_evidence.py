"""Tests for meg_tokens.behavior.math.evidence."""

from math import comb, log

import numpy as np
import pytest

from meg_tokens.behavior.math.evidence import (
    FIRST_ORDER_TOKEN_LOG_LR,
    MAX_LOG_ODDS,
    evidence_after_tokens,
    first_order_sum_log_lr_profile,
    log_posterior_odds,
    sum_log_lr_profile,
    token_lead_profile,
)


# --- token lead ---------------------------------------------------------------


def test_the_token_lead_is_the_running_count_difference():
    assert token_lead_profile([1, 1, 2, 1], target=1) == [1, 2, 1, 2]


def test_the_token_lead_flips_sign_with_the_reference_target():
    assert token_lead_profile([1, 1, 2, 1], target=2) == [-1, -2, -1, -2]


def test_the_token_lead_rejects_a_serialized_direction_string():
    with pytest.raises(TypeError, match="sequence of integer"):
        token_lead_profile("1121", target=1)


@pytest.mark.parametrize(
    ("directions", "target", "error", "message"),
    [
        ([1, 3], 1, ValueError, "only target ids 1 and 2"),
        ([1, 2.0], 1, TypeError, "integer target ids"),
        ([1, 2], 0, ValueError, "target must be 1 or 2"),
    ],
)
def test_the_token_lead_rejects_impossible_inputs(
    directions, target, error, message
):
    with pytest.raises(error, match=message):
        token_lead_profile(directions, target=target)


def test_the_token_lead_accepts_numpy_integers():
    assert token_lead_profile(np.array([1, 2, 1]), target=1) == [1, 0, 1]


# --- log posterior odds -------------------------------------------------------


def test_chance_is_zero_evidence():
    odds, saturated = log_posterior_odds(0.5)

    assert odds == pytest.approx(0.0)
    assert not saturated


def test_odds_follow_the_logit_of_the_probability():
    odds, saturated = log_posterior_odds(0.8)

    assert odds == pytest.approx(log(0.8 / 0.2))
    assert not saturated


def test_certainty_is_flagged_and_clipped_instead_of_returning_infinity():
    """Downstream regressions cannot take an infinite predictor, so the finite
    boundary is reported together with the fact that it was hit."""
    assert log_posterior_odds(1.0) == (pytest.approx(MAX_LOG_ODDS), True)
    assert log_posterior_odds(0.0) == (pytest.approx(-MAX_LOG_ODDS), True)


def test_a_probability_beyond_the_boundary_is_also_flagged():
    # 255/256 is the first Tokens-task state at the clipping boundary.
    odds, saturated = log_posterior_odds(255.0 / 256.0)

    assert odds == pytest.approx(MAX_LOG_ODDS)
    assert saturated


@pytest.mark.parametrize("value", [1.5, -0.1])
def test_odds_reject_values_outside_the_probability_scale(value):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        log_posterior_odds(value)


# --- cumulative log likelihood ratio ------------------------------------------


def test_first_order_token_log_lr_follows_the_15_token_majority_design():
    selected_event_likelihood = sum(
        comb(14, selected_remaining)
        for selected_remaining in range(7, 15)
    ) / (2**14)

    assert selected_event_likelihood == pytest.approx(0.604736328125)
    assert FIRST_ORDER_TOKEN_LOG_LR == pytest.approx(
        log(selected_event_likelihood / (1.0 - selected_event_likelihood))
    )


def test_the_evidence_profile_grows_with_the_target_majority():
    profile = sum_log_lr_profile([1] * 8 + [2] * 7, target=1)

    assert profile[0] < profile[3] < profile[7]


def test_evidence_saturates_once_the_majority_is_secured_and_stays_there():
    profile = sum_log_lr_profile([1] * 8 + [2] * 7, target=1)

    # The eighth token settles the trial; no later token can change it.
    assert profile[7] == pytest.approx(MAX_LOG_ODDS)
    assert profile[-1] == pytest.approx(MAX_LOG_ODDS)


def test_the_evidence_profile_is_finite_throughout():
    profile = sum_log_lr_profile([2] * 8 + [1] * 7, target=1)

    assert all(abs(value) <= MAX_LOG_ODDS for value in profile)
    assert profile[7] == pytest.approx(-MAX_LOG_ODDS)


def test_the_evidence_profile_rejects_a_serialized_direction_string():
    with pytest.raises(TypeError, match="sequence of integer"):
        sum_log_lr_profile("1121", target=1)


def test_first_order_sum_log_lr_is_a_fixed_multiple_of_token_lead():
    profile = first_order_sum_log_lr_profile([1, 2, 1, 1], target=1)

    assert profile == pytest.approx([
        FIRST_ORDER_TOKEN_LOG_LR,
        0.0,
        FIRST_ORDER_TOKEN_LOG_LR,
        2 * FIRST_ORDER_TOKEN_LOG_LR,
    ])


def test_first_order_sum_log_lr_does_not_saturate_with_the_horizon():
    profile = first_order_sum_log_lr_profile([1] * 8 + [2] * 7, target=1)

    assert profile[7] == pytest.approx(8 * FIRST_ORDER_TOKEN_LOG_LR)
    assert profile[-1] == pytest.approx(FIRST_ORDER_TOKEN_LOG_LR)


# --- selecting evidence at a token count --------------------------------------


def test_no_observed_token_selects_the_declared_prior():
    assert evidence_after_tokens([0.6, 0.7], 0, prior=0.5) == 0.5
    assert evidence_after_tokens([2, 1], 0, prior=0.0) == 0.0


def test_the_token_count_is_one_based_over_the_profile():
    assert evidence_after_tokens([0.6, 0.7], 1, prior=0.5) == 0.6
    assert evidence_after_tokens([0.6, 0.7], 2, prior=0.5) == 0.7


def test_selecting_evidence_rejects_a_count_outside_the_profile():
    with pytest.raises(ValueError, match="non-negative"):
        evidence_after_tokens([0.6, 0.7], -1, prior=0.5)
    with pytest.raises(ValueError, match="cannot exceed"):
        evidence_after_tokens([0.6, 0.7], 3, prior=0.5)


def test_selecting_evidence_rejects_a_non_integer_count():
    with pytest.raises(TypeError, match="n_tokens must be an integer"):
        evidence_after_tokens([0.6, 0.7], 1.0, prior=0.5)
