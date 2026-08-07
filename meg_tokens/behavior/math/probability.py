"""Pure success-probability mathematics for the 15-token Tokens task.

This module implements five distinct operations: calculate one exact success
probability, build its token-by-token profile, validate a recorded probability
path, classify a complete designed profile, and select the probability
available at a decision time. Serialized table values and runtime-log policy
are handled outside the mathematics layer.
"""

from __future__ import annotations

from bisect import bisect_right
from math import comb, isclose, isfinite
from numbers import Integral
from typing import Final, Sequence


TOTAL_TOKENS: Final[int] = 15
WINNING_TOKENS: Final[int] = 8


def _directions(directions: Sequence[int]) -> list[int]:
    """Validate and copy an in-memory token-direction sequence."""
    if isinstance(directions, (str, bytes)):
        raise TypeError("directions must be a sequence of integer target ids")
    values = list(directions)
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in values
    ):
        raise TypeError("directions must contain integer target ids")
    if any(int(value) not in (1, 2) for value in values):
        raise ValueError("directions must contain only target ids 1 and 2")
    if len(values) > TOTAL_TOKENS:
        raise ValueError(
            f"directions cannot contain more than {TOTAL_TOKENS} tokens"
        )
    return [int(value) for value in values]


def success_probability(
    target_tokens: int,
    other_tokens: int,
    remaining_tokens: int,
) -> float:
    """Calculate the target's exact probability of winning the trial.

    Parameters
    ----------
    target_tokens
        Tokens already at the target of interest.
    other_tokens
        Tokens already at the other target.
    remaining_tokens
        Tokens that have not yet jumped. Each is assumed independently equally
        likely to reach either target.

    Returns
    -------
    float
        Probability that the target finishes with at least eight of the 15
        tokens.

    Raises
    ------
    TypeError
        If any count is not an integer.
    ValueError
        If a count is negative or the three counts do not sum to 15.

    Notes
    -----
    The result is the upper tail of a binomial distribution with success
    probability 0.5, matching the Tokens-task success-probability equation.
    """
    counts = (target_tokens, other_tokens, remaining_tokens)
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in counts
    ):
        raise TypeError("Token counts must be integers")
    if any(value < 0 for value in counts):
        raise ValueError("Token counts must be non-negative")
    if sum(counts) != TOTAL_TOKENS:
        raise ValueError(f"Token counts must sum to {TOTAL_TOKENS}")

    needed = max(0, WINNING_TOKENS - int(target_tokens))
    remaining = int(remaining_tokens)
    if needed > remaining:
        return 0.0
    favorable = sum(comb(remaining, value) for value in range(needed, remaining + 1))
    return favorable / (2**remaining)


def success_probability_profile(
    directions: Sequence[int],
    *,
    target: int,
) -> list[float]:
    """Build a target-referenced success-probability profile.

    Parameters
    ----------
    directions
        Integer target identifier for each observed token jump. Serialized
        strings must be parsed by the schema layer before calling this function.
    target
        Target whose eventual success probability is calculated; ``1`` or
        ``2``.

    Returns
    -------
    list of float
        Success probability immediately after each supplied token jump. The
        output length equals the input length and excludes the pre-jump prior.

    Raises
    ------
    TypeError
        If directions are serialized text or contain non-integer values.
    ValueError
        If ``target`` or a direction is not ``1`` or ``2``, or more than 15
        directions are supplied.
    """
    if target not in (1, 2):
        raise ValueError("target must be 1 or 2")
    parsed = _directions(directions)

    target_count = 0
    other_count = 0
    profile: list[float] = []
    for jump, direction in enumerate(parsed, start=1):
        if direction == target:
            target_count += 1
        else:
            other_count += 1
        profile.append(
            success_probability(
                target_count,
                other_count,
                TOTAL_TOKENS - jump,
            )
        )
    return profile


def validate_probability_path(
    probabilities: Sequence[float],
    *,
    first_jump: int = 1,
    atol: float = 1e-6,
) -> None:
    """Validate that recorded probabilities follow at least one legal path.

    Parameters
    ----------
    probabilities
        Recorded target-referenced success probabilities in jump order.
    first_jump
        One-based jump corresponding to the first probability. Use ``2`` for a
        14-row log that begins after the second token.
    atol
        Absolute tolerance used to match recorded values to exact legal states.

    Raises
    ------
    TypeError
        If ``first_jump`` is not an integer.
    ValueError
        If the requested jumps fall outside the 15-token trial, ``atol`` is
        invalid, a probability is not a legal state, or adjacent states cannot
        be connected by one token jump.

    Notes
    -----
    Saturated probabilities can correspond to more than one target-token
    count. This function deliberately returns no inferred counts: it verifies
    that a legal path exists without choosing among indistinguishable paths or
    guessing a nearest state.
    """
    if isinstance(first_jump, bool) or not isinstance(first_jump, Integral):
        raise TypeError("first_jump must be an integer")
    if first_jump < 1:
        raise ValueError("first_jump must be at least 1")
    if first_jump + len(probabilities) - 1 > TOTAL_TOKENS:
        raise ValueError(
            f"probability path cannot extend beyond jump {TOTAL_TOKENS}"
        )
    if not isfinite(float(atol)) or atol < 0:
        raise ValueError("atol must be a finite non-negative value")

    previous_counts: set[int] | None = None
    for offset, probability in enumerate(probabilities):
        jump = first_jump + offset
        candidates = [
            target_count
            for target_count in range(jump + 1)
            if abs(
                success_probability(
                    target_count,
                    jump - target_count,
                    TOTAL_TOKENS - jump,
                )
                - float(probability)
            )
            <= atol
        ]
        if not candidates:
            raise ValueError(
                f"Probability {probability} is not an Equation-1 state at jump {jump}"
            )
        if previous_counts is None:
            previous_counts = set(candidates)
            continue
        current_counts = {
            candidate
            for candidate in candidates
            if any(
                candidate - previous in (0, 1)
                for previous in previous_counts
            )
        }
        if not current_counts:
            raise ValueError("Probability states do not form a legal token-by-token path")
        previous_counts = current_counts


def classify_design_profile(
    profile: Sequence[float],
    *,
    atol: float = 1e-6,
) -> tuple[int, str]:
    """Classify one complete correct-target design profile.

    Parameters
    ----------
    profile
        Fifteen success probabilities, one after each designed token jump,
        referenced to the trial's correct target.
    atol
        Absolute tolerance used only for the rules requiring equality at 0.5
        or 1.0.

    Returns
    -------
    trial_class
        ``1`` for easy, ``2`` for ambiguous, ``3`` for misleading, or ``0``
        when none of the predefined rules matches.
    rule
        Stable identifier of the rule that matched, or ``"unclassified"``.

    Raises
    ------
    ValueError
        If the profile does not contain exactly 15 finite probabilities in
        ``[0, 1]`` or ``atol`` is invalid.

    Notes
    -----
    Rules are evaluated in order because the classes are not expressed as a
    single exhaustive partition. The function reports the matching rule so
    classification remains auditable.
    """
    values = [float(value) for value in profile]
    if len(values) != TOTAL_TOKENS:
        raise ValueError(
            f"A design profile must contain exactly {TOTAL_TOKENS} SP values"
        )
    if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("Design probabilities must be finite and in [0, 1]")
    if not isfinite(float(atol)) or atol < 0:
        raise ValueError("atol must be a finite non-negative value")
    sp2, sp3, sp5, sp8, sp11 = (
        values[index] for index in (1, 2, 4, 7, 10)
    )
    if sp2 > 0.60 and sp5 > 0.75 and sp8 > 0.75:
        return 1, "easy_sp2_sp5_sp8"
    if (
        isclose(sp2, 0.50, rel_tol=0.0, abs_tol=atol)
        and 0.40 < sp3 < 0.65
        and 0.35 < sp5 < 0.65
    ):
        return 2, "ambiguous_sp2_sp3_sp5"
    if (
        isclose(sp2, 0.50, rel_tol=0.0, abs_tol=atol)
        and 0.38 < sp3 < 0.40
        and 0.35 < sp5 < 0.65
        and isclose(sp11, 1.0, rel_tol=0.0, abs_tol=atol)
    ):
        return 2, "ambiguous_sp11_boundary"
    if sp3 < 0.40:
        return 3, "misleading_sp3"
    return 0, "unclassified"


def probability_at_decision(
    probabilities: Sequence[float],
    token_times: Sequence[float],
    *,
    decision_time: float,
    initial_probability: float = 0.5,
) -> tuple[float, int | None]:
    """Select the last probability available when a decision was made.

    Parameters
    ----------
    probabilities
        Probability after each logged token jump.
    token_times
        Strictly increasing time of each corresponding jump, in the same units
        and reference frame as ``decision_time``.
    decision_time
        Time at which the subject committed to a target.
    initial_probability
        Pre-jump probability returned when commitment precedes the first token.

    Returns
    -------
    probability
        Last probability whose token time is less than or equal to the decision
        time, or ``initial_probability`` before the first jump.
    index
        Zero-based index of the selected logged state, or ``None`` for the
        pre-jump state.

    Raises
    ------
    ValueError
        If probabilities and times have different lengths, contain non-finite
        values, times are not strictly increasing, or a probability lies
        outside ``[0, 1]``.

    Notes
    -----
    Equality belongs to the observed state: a token occurring exactly at the
    decision time is considered available.
    """
    if len(probabilities) != len(token_times):
        raise ValueError("probabilities and token_times must have equal lengths")
    probabilities = [float(value) for value in probabilities]
    token_times = [float(value) for value in token_times]
    if not isfinite(float(decision_time)):
        raise ValueError("decision_time must be finite")
    if any(not isfinite(value) for value in token_times):
        raise ValueError("token_times must be finite")
    if any(right <= left for left, right in zip(token_times, token_times[1:])):
        raise ValueError("token_times must be strictly increasing")
    if not isfinite(float(initial_probability)) or not 0.0 <= initial_probability <= 1.0:
        raise ValueError("initial_probability must be in [0, 1]")
    if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError("probabilities must be finite and in [0, 1]")

    position = bisect_right(token_times, decision_time) - 1
    if position < 0:
        return float(initial_probability), None
    return float(probabilities[position]), position
