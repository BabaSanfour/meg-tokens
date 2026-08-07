"""Pure evidence mathematics for the Tokens task."""

from __future__ import annotations

from math import log
from typing import Final, Sequence

import numpy as np

from meg_tokens.behavior.math.probability import success_probability_profile


MAX_LOG_ODDS: Final[float] = log(255.0)


def _validated_directions(directions: Sequence[int]) -> list[int]:
    """Return integer directions after validating the pure-math contract."""
    if isinstance(directions, (str, bytes)):
        raise TypeError("directions must be a sequence of integer target ids")
    values = list(directions)
    if any(not isinstance(value, (int, np.integer)) for value in values):
        raise TypeError("directions must contain integer target ids")
    if any(int(value) not in (1, 2) for value in values):
        raise ValueError("directions must contain only target ids 1 and 2")
    return [int(value) for value in values]


def token_lead_profile(
    directions: Sequence[int],
    *,
    target: int,
) -> list[int]:
    """Compute the running token-count difference in favor of one target.

    Parameters
    ----------
    directions
        Integer target identifier for every token jump. Parsing serialized
        table values is outside this pure-math function.
    target
        Target used as the positive evidence direction; must be ``1`` or ``2``.

    Returns
    -------
    list of int
        At each jump, the number of tokens at ``target`` minus the number at
        the other target. The result has the same length as ``directions``.

    Raises
    ------
    ValueError
        If ``target`` or a direction is not ``1`` or ``2``.
    TypeError
        If ``directions`` is a string or contains non-integer values.
    """
    if target not in (1, 2):
        raise ValueError("target must be 1 or 2")
    values = _validated_directions(directions)
    lead = 0
    profile = []
    for direction in values:
        lead += 1 if int(direction) == target else -1
        profile.append(lead)
    return profile


def log_posterior_odds(probability: float) -> tuple[float, bool]:
    """Transform a success probability to finite natural-log odds.

    Parameters
    ----------
    probability
        Target-referenced success probability in the closed interval
        ``[0, 1]``.

    Returns
    -------
    log_odds
        ``log(p / (1 - p))``, clipped to ``[-log(255), log(255)]``.
    saturated
        ``True`` when the probability implies infinite log odds or when the
        finite result reaches the clipping boundary.

    Raises
    ------
    ValueError
        If ``probability`` lies outside ``[0, 1]``.

    Notes
    -----
    The finite boundary avoids infinities at probabilities zero and one. The
    returned flag preserves whether clipping occurred so callers need not
    infer it from the numeric value.
    """
    value = float(probability)
    if not 0.0 <= value <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    if value <= 0.0:
        return -MAX_LOG_ODDS, True
    if value >= 1.0:
        return MAX_LOG_ODDS, True
    odds = log(value / (1.0 - value))
    if abs(odds) >= MAX_LOG_ODDS:
        return float(np.sign(odds) * MAX_LOG_ODDS), True
    return odds, False


def sum_log_lr_profile(
    directions: Sequence[int],
    *,
    target: int,
) -> list[float]:
    """Compute target-referenced log-odds evidence after every token jump.

    Parameters
    ----------
    directions
        Complete or partial sequence of token destinations.
    target
        Target treated as the successful outcome; must be ``1`` or ``2``.

    Returns
    -------
    list of float
        Natural-log odds of the target's eventual success after each observed
        token. Values are finite because :func:`log_posterior_odds` clips
        saturated states.

    Notes
    -----
    Success probabilities are calculated under the Tokens-task rule of 15
    total tokens and a majority threshold of eight. No value is emitted for
    the pre-jump prior.
    """
    profile = success_probability_profile(
        _validated_directions(directions),
        target=target,
    )
    return [log_posterior_odds(value)[0] for value in profile]


def evidence_after_tokens(
    profile: Sequence[float],
    n_tokens: int,
    *,
    prior: float,
) -> float:
    """Select evidence from a per-jump profile at a requested token count.

    Parameters
    ----------
    profile
        Evidence values indexed by observed jump, with element zero
        representing the value after the first token.
    n_tokens
        Number of observed tokens. Zero selects ``prior``; positive values
        must not exceed the profile length.
    prior
        Evidence before any token has jumped.

    Returns
    -------
    float
        Evidence after exactly ``n_tokens`` observations, or the prior before
        the first jump.

    Raises
    ------
    TypeError
        If ``n_tokens`` is not an integer.
    ValueError
        If ``n_tokens`` is negative or exceeds the profile length.
    """
    if not isinstance(n_tokens, (int, np.integer)):
        raise TypeError("n_tokens must be an integer")
    if n_tokens < 0:
        raise ValueError("n_tokens must be non-negative")
    if n_tokens == 0:
        return float(prior)
    if n_tokens > len(profile):
        raise ValueError("n_tokens cannot exceed the evidence profile length")
    return float(profile[n_tokens - 1])
