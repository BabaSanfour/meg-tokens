import pytest

from meg_tokens.behavior.math.probability import (
    classify_design_profile,
    probability_at_decision,
    success_probability,
    success_probability_profile,
    validate_probability_path,
)


def test_success_probability_matches_known_first_jump_values():
    assert success_probability(1, 0, 14) == pytest.approx(0.604736328125)
    assert success_probability(0, 1, 14) == pytest.approx(0.395263671875)


def test_probability_math_rejects_serialized_or_non_integer_inputs():
    with pytest.raises(TypeError, match="integers"):
        success_probability(1.0, 0, 14)
    with pytest.raises(TypeError, match="sequence of integer"):
        success_probability_profile("112112211121122", target=1)


def test_success_probability_profile_reaches_the_known_winner():
    profile = success_probability_profile(
        [int(value) for value in "112112211121122"],
        target=1,
    )

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


def test_validate_probability_path_accepts_a_shifted_legal_profile():
    profile = success_probability_profile(
        [int(value) for value in "121121122112212"],
        target=1,
    )

    assert validate_probability_path(profile[1:], first_jump=2) is None


def test_validate_probability_path_rejects_disconnected_states():
    after_two_with_zero_target_tokens = success_probability(0, 2, 13)
    after_three_with_two_target_tokens = success_probability(2, 1, 12)
    with pytest.raises(ValueError, match="legal token-by-token path"):
        validate_probability_path(
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
    profile = success_probability_profile(
        [int(value) for value in directions],
        target=target,
    )

    trial_class, _ = classify_design_profile(profile)
    assert trial_class == expected


def test_classify_design_profile_reports_sp11_boundary_rule():
    profile = success_probability_profile(
        [int(value) for value in "212112111111212"],
        target=1,
    )

    assert classify_design_profile(profile) == (
        2,
        "ambiguous_sp11_boundary",
    )
