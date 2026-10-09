"""Tests for meg_tokens.behavior.math.evidence."""

from math import comb, log

import numpy as np
import pytest

from meg_tokens.behavior.math.evidence import (
    FIRST_ORDER_TOKEN_LOG_LR,
    MAX_LOG_ODDS,
    first_order_sum_log_lr_profile,
    log_posterior_odds,
    sum_log_lr_profile,
    token_lead_profile,
)


def test_the_token_lead_is_the_running_count_difference_from_the_reference_target():
    assert token_lead_profile([1, 1, 2, 1], target=1) == [1, 2, 1, 2]
    assert token_lead_profile([1, 1, 2, 1], target=2) == [-1, -2, -1, -2]
    assert token_lead_profile(np.array([1, 2, 1]), target=1) == [1, 0, 1]


def test_log_posterior_odds_are_the_logit_clipped_at_a_flagged_boundary():
    assert log_posterior_odds(0.5) == (pytest.approx(0.0), False)
    assert log_posterior_odds(0.8) == (pytest.approx(log(0.8 / 0.2)), False)
    # Downstream regressions cannot take an infinite predictor, so the finite
    # boundary is reported together with the fact that it was hit.
    assert log_posterior_odds(1.0) == (pytest.approx(MAX_LOG_ODDS), True)
    assert log_posterior_odds(0.0) == (pytest.approx(-MAX_LOG_ODDS), True)
    # 255/256 is the first Tokens-task state at the clipping boundary.
    assert log_posterior_odds(255.0 / 256.0) == (pytest.approx(MAX_LOG_ODDS), True)

    for value in (1.5, -0.1):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            log_posterior_odds(value)


def test_the_first_order_increment_follows_the_fifteen_token_majority_design():
    selected_event_likelihood = sum(
        comb(14, selected_remaining) for selected_remaining in range(7, 15)
    ) / (2**14)
    assert selected_event_likelihood == pytest.approx(0.604736328125)
    assert FIRST_ORDER_TOKEN_LOG_LR == pytest.approx(
        log(selected_event_likelihood / (1.0 - selected_event_likelihood))
    )


def test_the_exact_posterior_profile_grows_then_saturates_finitely_at_the_horizon():
    profile = sum_log_lr_profile([1] * 8 + [2] * 7, target=1)
    assert profile[0] < profile[3] < profile[7]
    # The eighth token settles the trial; no later token can change it.
    assert profile[7] == pytest.approx(MAX_LOG_ODDS)
    assert profile[-1] == pytest.approx(MAX_LOG_ODDS)

    against = sum_log_lr_profile([2] * 8 + [1] * 7, target=1)
    assert all(abs(value) <= MAX_LOG_ODDS for value in against)
    assert against[7] == pytest.approx(-MAX_LOG_ODDS)

    with pytest.raises(TypeError, match="sequence of integer"):
        sum_log_lr_profile("1121", target=1)


def test_the_first_order_profile_is_a_fixed_multiple_of_the_lead_and_never_saturates():
    assert first_order_sum_log_lr_profile([1, 2, 1, 1], target=1) == pytest.approx([
        FIRST_ORDER_TOKEN_LOG_LR,
        0.0,
        FIRST_ORDER_TOKEN_LOG_LR,
        2 * FIRST_ORDER_TOKEN_LOG_LR,
    ])
    # The horizon is invisible to this scale, which is the whole point of
    # reporting it beside the exact posterior.
    profile = first_order_sum_log_lr_profile([1] * 8 + [2] * 7, target=1)
    assert profile[7] == pytest.approx(8 * FIRST_ORDER_TOKEN_LOG_LR)
    assert profile[-1] == pytest.approx(FIRST_ORDER_TOKEN_LOG_LR)
