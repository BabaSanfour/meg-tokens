"""Tests for meg_tokens.behavior.math.probability."""

import pytest

from meg_tokens.behavior.math.probability import (
    TOTAL_TOKENS,
    WINNING_TOKENS,
    classify_design_profile,
    probability_at_decision,
    success_probability,
    success_probability_profile,
    validate_probability_path,
)


def _directions(text):
    return [int(character) for character in text]


def test_the_success_probability_is_the_closed_form_equation_1_state():
    # 1 token gained of the 8 needed, 14 still to fall.
    assert success_probability(1, 0, 14) == pytest.approx(0.604736328125)
    assert success_probability(0, 1, 14) == pytest.approx(0.395263671875)
    # A secured majority is certain, a lost one impossible, an even split with
    # one token left a coin flip.
    assert success_probability(WINNING_TOKENS, 0, 7) == 1.0
    assert success_probability(0, WINNING_TOKENS, 7) == 0.0
    assert success_probability(7, 7, 1) == pytest.approx(0.5)
    # The two targets partition the outcome space at every reachable state.
    for target_tokens in range(0, 8):
        other = 7 - target_tokens
        assert success_probability(target_tokens, other, TOTAL_TOKENS - 7) + success_probability(
            other, target_tokens, TOTAL_TOKENS - 7
        ) == pytest.approx(1.0)


@pytest.mark.parametrize("counts", [(1.0, 0, 14), (True, 0, 14), (1, "0", 14)])
def test_success_probability_rejects_non_integer_counts(counts):
    with pytest.raises(TypeError, match="integers"):
        success_probability(*counts)


def test_success_probability_rejects_counts_that_are_not_one_trial():
    with pytest.raises(ValueError, match="must be non-negative"):
        success_probability(-1, 2, 14)
    with pytest.raises(ValueError, match=f"must sum to {TOTAL_TOKENS}"):
        success_probability(1, 0, 13)


def test_the_profile_reports_one_state_per_observed_jump_referenced_to_its_target():
    profile = success_probability_profile(_directions("112112211121122"), target=1)
    assert len(profile) == 15
    assert profile[0] == pytest.approx(0.604736328125)
    assert profile[-1] == 1.0
    # A partially observed trial is legal.
    assert len(success_probability_profile([1, 2, 1], target=1)) == 3
    # Referencing to the other target complements every state.
    other = success_probability_profile(_directions("112112211121122"), target=2)
    for one, two in zip(profile, other):
        assert one + two == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("directions", "target", "error", "message"),
    [
        # A compact "1121" must be parsed by the schema layer, not iterated
        # here as characters.
        ("112112211121122", 1, TypeError, "sequence of integer"),
        ([1, 2, 3], 1, ValueError, "only target ids 1 and 2"),
        ([1, 2.0], 1, TypeError, "integer target ids"),
        ([1] * 16, 1, ValueError, "more than 15 tokens"),
        ([1, 2], 3, ValueError, "target must be 1 or 2"),
    ],
)
def test_the_profile_rejects_impossible_inputs(directions, target, error, message):
    with pytest.raises(error, match=message):
        success_probability_profile(directions, target=target)


def test_path_validation_accepts_a_legal_walk_and_rejects_everything_else():
    profile = success_probability_profile(_directions("121121122112212"), target=1)
    assert validate_probability_path(profile) is None

    # A 14-row log starts at jump 2; read as jump 1 it is not a legal path.
    assert validate_probability_path(profile[1:], first_jump=2) is None
    with pytest.raises(ValueError, match="not an Equation-1 state"):
        validate_probability_path(profile[1:], first_jump=1)

    with pytest.raises(ValueError, match="not an Equation-1 state at jump 1"):
        validate_probability_path([0.5])

    # Two states that no single jump can join.
    with pytest.raises(ValueError, match="legal token-by-token path"):
        validate_probability_path(
            [success_probability(0, 2, 13), success_probability(2, 1, 12)], first_jump=2
        )

    with pytest.raises(ValueError, match="cannot extend beyond jump 15"):
        validate_probability_path(profile, first_jump=2)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"first_jump": 1.0}, TypeError, "first_jump must be an integer"),
        ({"first_jump": 0}, ValueError, "at least 1"),
        ({"atol": -1e-6}, ValueError, "finite non-negative"),
    ],
)
def test_path_validation_rejects_invalid_settings(kwargs, error, message):
    with pytest.raises(error, match=message):
        validate_probability_path([0.604736328125], **kwargs)


@pytest.mark.parametrize(
    ("directions", "expected"),
    [
        ("111111112222222", (1, "easy_sp2_sp5_sp8")),
        ("121121122112212", (2, "ambiguous_sp2_sp3_sp5")),
        ("222111111112222", (3, "misleading_sp3")),
        # Its third-jump SP falls just under the misleading cut-off, so rule
        # order is what decides the class.
        ("212112111111212", (2, "ambiguous_sp11_boundary")),
        ("111122221111111", (0, "unclassified")),
    ],
)
def test_classify_design_profile_names_the_rule_it_matched(directions, expected):
    target = 1 if directions.count("1") > directions.count("2") else 2
    profile = success_probability_profile(_directions(directions), target=target)
    assert classify_design_profile(profile) == expected


def test_classification_requires_a_complete_well_formed_design_profile():
    with pytest.raises(ValueError, match="exactly 15 SP values"):
        classify_design_profile(success_probability_profile(_directions("11112222"), target=1))

    for value in (1.5, float("nan"), -0.1):
        profile = [0.5] * 15
        profile[4] = value
        with pytest.raises(ValueError, match=r"finite and in \[0, 1\]"):
            classify_design_profile(profile)


def test_the_decision_reads_the_last_token_already_on_screen():
    times = [1200, 1400, 1600]
    assert probability_at_decision([0.6, 0.7, 0.8], times, decision_time=1500) == (0.7, 1)
    # A token landing exactly at the decision counts as seen.
    assert probability_at_decision([0.6, 0.7, 0.8], times, decision_time=1400) == (0.7, 1)
    # After every token, the final state.
    assert probability_at_decision([0.6, 0.7, 0.8], times, decision_time=9000) == (0.8, 2)
    # Before the first token, the declared prior and no index.
    assert probability_at_decision([0.6], [1200], decision_time=1100) == (0.5, None)
    assert probability_at_decision(
        [0.6], [1200], decision_time=1100, initial_probability=0.4
    ) == (0.4, None)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"probabilities": [0.6], "token_times": [1200, 1400]}, "equal lengths"),
        ({"probabilities": [0.6, 0.7], "token_times": [1200, 1200]}, "strictly increasing"),
        ({"probabilities": [0.6, 0.7], "token_times": [1400, 1200]}, "strictly increasing"),
        (
            {"probabilities": [0.6, 1.7], "token_times": [1200, 1400]},
            r"probabilities must be finite and in \[0, 1\]",
        ),
        ({"probabilities": [0.6], "token_times": [float("inf")]}, "token_times must be finite"),
    ],
)
def test_the_decision_lookup_rejects_an_inconsistent_token_log(kwargs, message):
    with pytest.raises(ValueError, match=message):
        probability_at_decision(decision_time=1500, **kwargs)


def test_the_decision_lookup_rejects_an_undefined_decision_time():
    with pytest.raises(ValueError, match="decision_time must be finite"):
        probability_at_decision([0.6], [1200], decision_time=float("nan"))
