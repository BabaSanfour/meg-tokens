"""Success-probability profiles and probability-at-decision helpers."""

from __future__ import annotations

from bisect import bisect_right
from math import comb, isclose
from typing import Sequence


def success_probability(
    target_tokens: int,
    other_tokens: int,
    remaining_tokens: int,
) -> float:
    """Return the probability that the target finishes with at least 8 tokens."""
    counts = (target_tokens, other_tokens, remaining_tokens)
    if any(int(value) != value or value < 0 for value in counts):
        raise ValueError("Token counts must be non-negative integers")
    if sum(counts) != 15:
        raise ValueError("Token counts must sum to 15")

    needed = max(0, 8 - int(target_tokens))
    remaining = int(remaining_tokens)
    if needed > remaining:
        return 0.0
    favorable = sum(comb(remaining, value) for value in range(needed, remaining + 1))
    return favorable / (2**remaining)


def success_probability_profile(
    directions: Sequence[int] | str,
    *,
    target: int,
    total_tokens: int = 15,
) -> list[float]:
    """Compute target-referenced SP after each observed token direction."""
    if target not in (1, 2):
        raise ValueError("target must be 1 or 2")
    parsed = [int(value) for value in directions]
    if len(parsed) > total_tokens:
        raise ValueError("directions cannot contain more than total_tokens")
    if any(value not in (1, 2) for value in parsed):
        raise ValueError("directions must contain only 1 and 2")
    if total_tokens != 15:
        raise ValueError("The Tokens task requires exactly 15 total tokens")

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
                total_tokens - jump,
            )
        )
    return profile


def implied_target_counts(
    probabilities: Sequence[float],
    *,
    first_jump: int = 1,
    atol: float = 1e-6,
) -> list[int]:
    """Infer target-token counts when probabilities form a legal SP path.

    Raises ``ValueError`` if a probability is not an Equation-1 state at its
    proposed jump or if adjacent states cannot be connected by one token jump.
    """
    if first_jump < 1:
        raise ValueError("first_jump must be at least 1")
    if first_jump + len(probabilities) - 1 > 15:
        raise ValueError("probability path cannot extend beyond jump 15")

    paths: list[list[int]] = []
    for offset, probability in enumerate(probabilities):
        jump = first_jump + offset
        candidates = [
            target_count
            for target_count in range(jump + 1)
            if abs(
                success_probability(
                    target_count,
                    jump - target_count,
                    15 - jump,
                )
                - float(probability)
            )
            <= atol
        ]
        if not candidates:
            raise ValueError(
                f"Probability {probability} is not an Equation-1 state at jump {jump}"
            )
        if not paths:
            paths = [[candidate] for candidate in candidates]
            continue
        paths = [
            [*path, candidate]
            for path in paths
            for candidate in candidates
            if candidate - path[-1] in (0, 1)
        ]
        if not paths:
            raise ValueError("Probability states do not form a legal token-by-token path")
    return paths[0]


def classify_design_profile_with_rule(
    profile: Sequence[float],
    *,
    atol: float = 1e-6,
) -> tuple[int, str]:
    """Classify a correct-target design profile and name the rule used."""
    if len(profile) != 15:
        raise ValueError("A design profile must contain exactly 15 SP values")
    sp2, sp3, sp5, sp8, sp11 = (
        float(profile[index]) for index in (1, 2, 4, 7, 10)
    )
    if sp2 > 0.60 and sp5 > 0.75 and sp8 > 0.75:
        return 1, "easy_sp2_sp5_sp8"
    if isclose(sp2, 0.50, abs_tol=atol) and 0.40 < sp3 < 0.65 and 0.35 < sp5 < 0.65:
        return 2, "ambiguous_sp2_sp3_sp5"
    if (
        isclose(sp2, 0.50, abs_tol=atol)
        and 0.38 < sp3 < 0.40
        and 0.35 < sp5 < 0.65
        and isclose(sp11, 1.0, abs_tol=atol)
    ):
        return 2, "ambiguous_sp11_boundary"
    if sp3 < 0.40:
        return 3, "misleading_sp3"
    return 0, "unclassified"


def classify_design_profile(profile: Sequence[float], *, atol: float = 1e-6) -> int:
    """Classify a complete correct-target design profile.

    Returns 1 (easy), 2 (ambiguous), 3 (misleading), or 0 (unclassified).
    """
    trial_class, _ = classify_design_profile_with_rule(profile, atol=atol)
    return trial_class


def probability_at_decision(
    probabilities: Sequence[float],
    token_times: Sequence[float],
    *,
    decision_time: float,
    initial_probability: float = 0.5,
) -> tuple[float, int | None]:
    """Return the last probability available at or before decision time.

    The returned index is ``None`` when the decision precedes the first token;
    in that case the pre-jump probability is returned.
    """
    if len(probabilities) != len(token_times):
        raise ValueError("probabilities and token_times must have equal lengths")
    if any(right <= left for left, right in zip(token_times, token_times[1:])):
        raise ValueError("token_times must be strictly increasing")
    if not 0.0 <= initial_probability <= 1.0:
        raise ValueError("initial_probability must be in [0, 1]")
    if any(not 0.0 <= float(value) <= 1.0 for value in probabilities):
        raise ValueError("probabilities must be in [0, 1]")

    position = bisect_right(token_times, decision_time) - 1
    if position < 0:
        return float(initial_probability), None
    return float(probabilities[position]), position


def align_design_profile_to_runtime(
    design_profile: Sequence[float],
    token_times: Sequence[float],
    *,
    token_log_rows: int,
    token_log_short: bool,
) -> tuple[list[float], list[float]]:
    """Return a design SP profile and runtime times only when alignment is valid."""
    if token_log_short or token_log_rows != 15:
        raise ValueError(
            "Design-derived time-resolved SP requires a complete 15-row runtime log"
        )
    if len(design_profile) != 15 or len(token_times) != 15:
        raise ValueError("Design profile and runtime token times must each have 15 values")
    return (
        [float(value) for value in design_profile],
        [float(value) for value in token_times],
    )


def design_spd_at_decision(
    design_profile: Sequence[float],
    token_times: Sequence[float],
    *,
    decision_time: float,
    token_log_rows: int,
    token_log_short: bool,
) -> tuple[float, int | None]:
    """Compute design-derived SPD only for a validated complete runtime log."""
    profile, times = align_design_profile_to_runtime(
        design_profile,
        token_times,
        token_log_rows=token_log_rows,
        token_log_short=token_log_short,
    )
    return probability_at_decision(
        profile,
        times,
        decision_time=decision_time,
    )
