import pytest

from meg_tokens.behavior.success_probability import (
    align_design_profile_to_runtime,
    classify_design_profile,
    classify_design_profile_with_rule,
    design_spd_at_decision,
    implied_target_counts,
    probability_at_decision,
    success_probability,
    success_probability_profile,
)


def test_success_probability_matches_known_first_jump_values():
    assert success_probability(1, 0, 14) == pytest.approx(0.604736328125)
    assert success_probability(0, 1, 14) == pytest.approx(0.395263671875)


def test_success_probability_profile_reaches_the_known_winner():
    profile = success_probability_profile("112112211121122", target=1)

    assert len(profile) == 15
    assert profile[0] == pytest.approx(0.604736328125)
    assert profile[-1] == 1.0


def test_probability_at_decision_uses_last_available_token_state():
    probability, index = probability_at_decision(
        [0.6, 0.7, 0.8],
        [1200, 1400, 1600],
        decision_time=1500,
    )

    assert probability == 0.7
    assert index == 1


def test_probability_at_decision_returns_prejump_probability_before_first_token():
    probability, index = probability_at_decision(
        [0.6],
        [1200],
        decision_time=1100,
    )

    assert probability == 0.5
    assert index is None


def test_probability_at_decision_rejects_unsorted_times():
    with pytest.raises(ValueError, match="strictly increasing"):
        probability_at_decision(
            [0.6, 0.7],
            [1200, 1200],
            decision_time=1300,
        )


def test_design_time_alignment_rejects_14_row_runtime_log():
    profile = success_probability_profile("112112211121122", target=1)

    with pytest.raises(ValueError, match="complete 15-row runtime log"):
        align_design_profile_to_runtime(
            profile,
            list(range(14)),
            token_log_rows=14,
            token_log_short=True,
        )


def test_design_spd_uses_complete_runtime_alignment():
    profile = success_probability_profile("112112211121122", target=1)
    probability, index = design_spd_at_decision(
        profile,
        list(range(100, 1600, 100)),
        decision_time=250,
        token_log_rows=15,
        token_log_short=False,
    )

    assert probability == profile[1]
    assert index == 1


def test_implied_target_counts_accepts_a_shifted_legal_profile():
    profile = success_probability_profile("121121122112212", target=1)

    counts = implied_target_counts(profile[1:], first_jump=2)

    assert counts[-1] == 8
    assert len(counts) == 14


def test_implied_target_counts_rejects_disconnected_states():
    after_two_with_zero_target_tokens = success_probability(0, 2, 13)
    after_three_with_two_target_tokens = success_probability(2, 1, 12)
    with pytest.raises(ValueError, match="legal token-by-token path"):
        implied_target_counts(
            [after_two_with_zero_target_tokens, after_three_with_two_target_tokens],
            first_jump=2,
        )


@pytest.mark.parametrize(
    ("directions", "expected"),
    [
        ("111111112222222", 1),
        ("121121122112212", 2),
        ("222111111112222", 3),
    ],
)
def test_classify_design_profile(directions, expected):
    target = 1 if directions.count("1") > directions.count("2") else 2
    profile = success_probability_profile(directions, target=target)

    assert classify_design_profile(profile) == expected


def test_classify_design_profile_reports_sp11_boundary_rule():
    profile = success_probability_profile("212112111111212", target=1)

    assert classify_design_profile_with_rule(profile) == (
        2,
        "ambiguous_sp11_boundary",
    )
