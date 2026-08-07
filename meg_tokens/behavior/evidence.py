"""Continuous evidence, criterion decline, and urgency (roadmap Tier B1-B5, C3).

Trial class and SPD summarize a trial with one label and one number. The
analyses here keep evidence continuous and time-resolved: cumulative
log-likelihood evidence from the token directions (B1), the decline of
evidence at decision as more tokens are observed (B2) or as time passes (C3,
the explicit urgency parameters), the temporal weighting of tokens on choice
(B3), accuracy across DT quantiles (B4), and regressions that retain the
unclassified 60% of trials instead of discarding them (B5).
"""

from __future__ import annotations

from math import log
from typing import Final, Sequence

import numpy as np
import pandas as pd

from meg_tokens.behavior.metrics import paired_subject_statistics
from meg_tokens.behavior.regression import (
    fit_linear,
    fit_logistic,
    one_sample_statistics,
)
from meg_tokens.behavior.success_probability import success_probability_profile
from meg_tokens.behavior.trials import TASK_CONDITIONS, require_columns, task_trials


# Success probability is the exact posterior that the target wins (Equation 1),
# so with equal priors the cumulative log-likelihood ratio for that target is
# its log posterior odds. SP hits exactly 0 or 1 once a target has secured or
# lost the majority; those states are certainty, whose log odds are infinite.
# The smallest non-degenerate SP the 15-token design can reach is 1/256 (a
# target holding no tokens with 8 still to fall), so saturated states are
# reported at +/- log(255) and flagged rather than propagated as infinities
# that would silently drop the most decisive trials from every regression.
MAX_LOG_ODDS: Final[float] = log(255.0)

# Jumps entering the reverse-correlation kernel. The median decision falls
# well before the last token, so later jumps are unseen on most trials and
# contribute no usable variance.
DEFAULT_KERNEL_JUMPS: Final[int] = 8
DEFAULT_ACCURACY_BINS: Final[int] = 5


def parse_token_directions(value: object) -> list[int]:
    """Parse a designed ``sTokenDirs`` sequence into per-jump target ids."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    directions = [int(character) for character in str(value).strip() if character.isdigit()]
    if any(direction not in (1, 2) for direction in directions):
        raise ValueError(f"Token directions must contain only 1 and 2: {value!r}")
    return directions


def token_lead_profile(directions: Sequence[int] | str, *, target: int) -> list[int]:
    """Return the running token count difference in favour of ``target``.

    This is the sufficient statistic behind Equation 1: it is always finite and
    is a monotone transform of the success probability at the same jump.
    """
    if target not in (1, 2):
        raise ValueError("target must be 1 or 2")
    lead = 0
    profile = []
    for direction in parse_token_directions(directions) if isinstance(directions, str) else directions:
        lead += 1 if int(direction) == target else -1
        profile.append(lead)
    return profile


def log_posterior_odds(probability: float) -> tuple[float, bool]:
    """Return log odds for one success probability and whether it saturated."""
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


def sum_log_lr_profile(directions: Sequence[int] | str, *, target: int) -> list[float]:
    """Return cumulative log-likelihood evidence for ``target`` per jump."""
    parsed = parse_token_directions(directions) if isinstance(directions, str) else list(directions)
    profile = success_probability_profile(parsed, target=target)
    return [log_posterior_odds(value)[0] for value in profile]


def evidence_after_tokens(profile: Sequence[float], n_tokens: int, *, prior: float) -> float:
    """Return a profile value after ``n_tokens`` jumps, or the prior at zero."""
    if n_tokens <= 0:
        return float(prior)
    index = min(int(n_tokens), len(profile)) - 1
    if index < 0:
        return float(prior)
    return float(profile[index])


def _subject_condition_groups(trials: pd.DataFrame):
    for subject, subject_trials in trials.groupby("subject", sort=True):
        condition = subject_trials["condition"].astype(str).str.lower()
        yield subject, "all", subject_trials
        for name in TASK_CONDITIONS:
            yield subject, name.lower(), subject_trials.loc[condition == name.lower()]


def criterion_decline(
    features: pd.DataFrame,
    *,
    predictor: str = "decision_token_index",
    response: str = "logged_spd",
) -> pd.DataFrame:
    """Fit evidence at decision against elapsed evidence or elapsed time.

    A negative slope means the subject accepted progressively weaker evidence
    the longer a trial ran, which is the behavioral signature of urgency
    gating. With ``predictor="decision_token_index"`` this is the Tier B2
    accuracy-criterion decline; with ``predictor="dt_ms"`` the intercept and
    slope are the Tier C3 urgency parameters in evidence units per second.
    """
    trials = task_trials(features)
    require_columns(trials, [predictor, response, "subject", "condition"])
    scale = 1000.0 if predictor == "dt_ms" else 1.0
    rows = []
    for subject, condition, selected in _subject_condition_groups(trials):
        x = pd.to_numeric(selected[predictor], errors="coerce").to_numpy(dtype=float) / scale
        y = pd.to_numeric(selected[response], errors="coerce").to_numpy(dtype=float)
        design = np.column_stack([np.ones_like(x), x])
        fit = fit_linear(design, y, ["intercept", "slope"])
        rows.append(
            {
                "subject": subject,
                "condition": condition,
                "predictor": predictor,
                "response": response,
                "n_trials": fit.n_observations,
                "intercept": fit.coefficient("intercept") if fit.converged else float("nan"),
                "slope": fit.coefficient("slope") if fit.converged else float("nan"),
                "slope_se": (
                    float(fit.standard_errors[1]) if fit.converged else float("nan")
                ),
                "converged": fit.converged,
            }
        )
    return pd.DataFrame(rows)


def criterion_decline_statistics(subject_fits: pd.DataFrame) -> pd.DataFrame:
    """Test criterion slopes against zero and between Fast and Slow."""
    rows = []
    for (predictor, response), fits in subject_fits.groupby(
        ["predictor", "response"], sort=True
    ):
        for condition, group in fits.groupby("condition", sort=True):
            for term in ("intercept", "slope"):
                rows.append(
                    {
                        "analysis": "criterion_decline",
                        "predictor": predictor,
                        "response": response,
                        "term": term,
                        "condition": condition,
                        "test": "one_sample_vs_zero",
                        **one_sample_statistics(group[term]),
                    }
                )
        wide = fits.pivot(index="subject", columns="condition", values=["intercept", "slope"])
        for term in ("intercept", "slope"):
            if (term, "fast") not in wide.columns or (term, "slow") not in wide.columns:
                continue
            pair = wide[[(term, "fast"), (term, "slow")]].copy()
            pair.columns = ["fast", "slow"]
            rows.append(
                {
                    "analysis": "criterion_decline",
                    "predictor": predictor,
                    "response": response,
                    "term": term,
                    "condition": "fast_vs_slow",
                    "test": "paired_t_test",
                    **paired_subject_statistics(pair, "fast", "slow"),
                }
            )
    return pd.DataFrame(rows)


def urgency_parameters(features: pd.DataFrame) -> pd.DataFrame:
    """Return per-subject urgency intercept and slope by condition (Tier C3)."""
    return criterion_decline(features, predictor="dt_ms", response="logged_spd")


def evidence_at_decision_responses(
    features: pd.DataFrame,
    *,
    predictor: str,
    responses: Sequence[str] = ("logged_spd", "logged_spd_log_odds"),
) -> pd.DataFrame:
    """Fit the criterion against one predictor on every evidence scale.

    Success probability is bounded and compresses near 1, so a slope measured
    on it is not directly comparable with the log-odds criterion the
    urgency-gating literature reports. Both are fitted and stacked.
    """
    return pd.concat(
        [
            criterion_decline(features, predictor=predictor, response=response)
            for response in responses
        ],
        ignore_index=True,
    )


def reverse_correlation(
    features: pd.DataFrame,
    *,
    n_jumps: int = DEFAULT_KERNEL_JUMPS,
) -> pd.DataFrame:
    """Estimate how strongly each token jump influenced the eventual choice.

    Each trial contributes one signed predictor per jump: ``+1`` when that
    token went to target 1, ``-1`` when it went to target 2, and ``0`` when the
    token fell after the subject had already committed and therefore could not
    have been seen. The response is the chosen target. The fitted logistic
    weights are the psychophysical kernel; ``mean_signed_direction`` reports
    the same data model-free, as the average direction of that jump relative to
    the target the subject chose.
    """
    trials = task_trials(features)
    require_columns(
        trials,
        ["token_directions", "choice_side", "decision_token_index", "subject", "condition"],
    )
    jumps = [f"jump{index:02d}" for index in range(1, n_jumps + 1)]
    rows = []
    for subject, condition, selected in _subject_condition_groups(trials):
        if selected.empty:
            continue
        predictors = np.zeros((len(selected), n_jumps), dtype=float)
        toward_choice = np.full((len(selected), n_jumps), np.nan)
        choices = np.zeros(len(selected), dtype=float)
        for row_index, (_, trial) in enumerate(selected.iterrows()):
            directions = parse_token_directions(trial["token_directions"])
            seen = int(pd.to_numeric(trial["decision_token_index"], errors="coerce") or 0)
            choice = int(trial["choice_side"])
            choices[row_index] = 1.0 if choice == 1 else 0.0
            for jump_index in range(min(n_jumps, len(directions), seen)):
                signed = 1.0 if directions[jump_index] == 1 else -1.0
                predictors[row_index, jump_index] = signed
                toward_choice[row_index, jump_index] = (
                    1.0 if directions[jump_index] == choice else -1.0
                )
        design = np.column_stack([np.ones(len(selected)), predictors])
        fit = fit_logistic(design, choices, ["intercept", *jumps])
        weights = fit.as_dict() if fit.converged else {}
        for jump_index, jump in enumerate(jumps):
            column = toward_choice[:, jump_index]
            observed = column[np.isfinite(column)]
            rows.append(
                {
                    "subject": subject,
                    "condition": condition,
                    "jump": jump_index + 1,
                    "n_trials": int(len(selected)),
                    "n_trials_token_seen": int(observed.size),
                    "logistic_weight": weights.get(jump, float("nan")),
                    "mean_signed_direction": (
                        float(np.mean(observed)) if observed.size else float("nan")
                    ),
                    "converged": fit.converged,
                }
            )
    return pd.DataFrame(rows)


def reverse_correlation_statistics(kernels: pd.DataFrame) -> pd.DataFrame:
    """Test kernel weights against zero and compare Fast with Slow per jump."""
    rows = []
    for metric in ("logistic_weight", "mean_signed_direction"):
        for (condition, jump), group in kernels.groupby(["condition", "jump"], sort=True):
            rows.append(
                {
                    "analysis": "reverse_correlation",
                    "metric": metric,
                    "condition": condition,
                    "jump": int(jump),
                    "test": "one_sample_vs_zero",
                    **one_sample_statistics(group[metric]),
                }
            )
        for jump, group in kernels.groupby("jump", sort=True):
            wide = group.pivot(index="subject", columns="condition", values=metric)
            if "fast" not in wide.columns or "slow" not in wide.columns:
                continue
            rows.append(
                {
                    "analysis": "reverse_correlation",
                    "metric": metric,
                    "condition": "fast_vs_slow",
                    "jump": int(jump),
                    "test": "paired_t_test",
                    **paired_subject_statistics(wide, "fast", "slow"),
                }
            )
    return pd.DataFrame(rows)


def conditional_accuracy_functions(
    features: pd.DataFrame,
    *,
    n_bins: int = DEFAULT_ACCURACY_BINS,
) -> pd.DataFrame:
    """Return accuracy across within-subject DT quantile bins per condition.

    A conditional accuracy function that falls with DT indicates a declining
    criterion; one that rises indicates that slow trials are simply the hard
    ones. Bin edges are per subject and condition so that between-subject
    differences in overall speed cannot shift trials between bins.
    """
    trials = task_trials(features)
    require_columns(trials, ["dt_ms", "isCorrect", "subject", "condition"])
    rows = []
    for subject, condition, selected in _subject_condition_groups(trials):
        values = pd.to_numeric(selected["dt_ms"], errors="coerce")
        correct = selected["isCorrect"].map(_as_bool)
        usable = selected.loc[values.notna() & correct.notna()]
        if usable.empty:
            continue
        usable_dt = pd.to_numeric(usable["dt_ms"], errors="coerce")
        try:
            bins = pd.qcut(usable_dt, n_bins, labels=False, duplicates="drop")
        except ValueError:  # fewer distinct values than requested bins
            continue
        usable_correct = usable["isCorrect"].map(_as_bool).astype(float)
        for bin_index in sorted(set(bins.dropna().astype(int))):
            selection = bins == bin_index
            rows.append(
                {
                    "subject": subject,
                    "condition": condition,
                    "dt_bin": int(bin_index) + 1,
                    "n_trials": int(selection.sum()),
                    "mean_dt_ms": float(usable_dt[selection].mean()),
                    "accuracy": float(usable_correct[selection].mean()),
                }
            )
    return pd.DataFrame(rows)


def conditional_accuracy_statistics(functions: pd.DataFrame) -> pd.DataFrame:
    """Summarize conditional accuracy per bin and test its slope across bins."""
    rows = []
    for (condition, dt_bin), group in functions.groupby(["condition", "dt_bin"], sort=True):
        summary = one_sample_statistics(group["accuracy"])
        rows.append(
            {
                "analysis": "conditional_accuracy",
                "condition": condition,
                "dt_bin": int(dt_bin),
                "test": "mean_accuracy",
                "mean_dt_ms": float(group["mean_dt_ms"].mean()),
                **summary,
            }
        )
    slopes = []
    for (subject, condition), group in functions.groupby(["subject", "condition"], sort=True):
        ordered = group.sort_values("dt_bin")
        design = np.column_stack(
            [np.ones(len(ordered)), ordered["dt_bin"].to_numpy(dtype=float)]
        )
        fit = fit_linear(design, ordered["accuracy"].to_numpy(dtype=float), ["intercept", "slope"])
        slopes.append(
            {
                "subject": subject,
                "condition": condition,
                "slope": fit.coefficient("slope") if fit.converged else float("nan"),
            }
        )
    slope_table = pd.DataFrame(slopes)
    for condition, group in slope_table.groupby("condition", sort=True):
        rows.append(
            {
                "analysis": "conditional_accuracy",
                "condition": condition,
                "dt_bin": pd.NA,
                "test": "accuracy_slope_across_bins",
                "mean_dt_ms": float("nan"),
                **one_sample_statistics(group["slope"]),
            }
        )
    return pd.DataFrame(rows)


def continuous_evidence_effects(
    features: pd.DataFrame,
    *,
    predictors: Sequence[str] = ("sp_design_early", "sum_log_lr_design_early"),
) -> pd.DataFrame:
    """Regress DT and accuracy on continuous early evidence (Tier B5).

    Unlike the class analyses these use every task trial with a designed token
    sequence, including the unclassified random trials that the threshold rule
    discards. Decision time is regressed on evidence *strength* (distance from
    chance, which is what should speed a decision) and accuracy on *signed*
    evidence toward the correct target.
    """
    trials = task_trials(features)
    require_columns(trials, ["dt_ms", "isCorrect", *predictors])
    rows = []
    for subject, condition, selected in _subject_condition_groups(trials):
        if selected.empty:
            continue
        for predictor in predictors:
            signed = pd.to_numeric(selected[predictor], errors="coerce").to_numpy(dtype=float)
            centre = 0.5 if predictor.startswith("sp_") else 0.0
            signed = signed - centre
            strength = np.abs(signed)
            dt = pd.to_numeric(selected["dt_ms"], errors="coerce").to_numpy(dtype=float)
            dt_fit = fit_linear(
                np.column_stack([np.ones_like(strength), strength]),
                dt,
                ["intercept", "evidence_strength"],
            )
            correct = selected["isCorrect"].map(_as_bool)
            valid = correct.notna().to_numpy()
            accuracy_fit = fit_logistic(
                np.column_stack([np.ones(valid.sum()), signed[valid]]),
                correct[valid].astype(float).to_numpy(),
                ["intercept", "evidence_signed"],
            )
            rows.append(
                {
                    "subject": subject,
                    "condition": condition,
                    "predictor": predictor,
                    "n_dt_trials": dt_fit.n_observations,
                    "dt_intercept_ms": (
                        dt_fit.coefficient("intercept") if dt_fit.converged else float("nan")
                    ),
                    "dt_slope_ms_per_unit": (
                        dt_fit.coefficient("evidence_strength")
                        if dt_fit.converged
                        else float("nan")
                    ),
                    "n_accuracy_trials": accuracy_fit.n_observations,
                    "accuracy_log_odds_per_unit": (
                        accuracy_fit.coefficient("evidence_signed")
                        if accuracy_fit.converged
                        else float("nan")
                    ),
                    "converged": bool(dt_fit.converged and accuracy_fit.converged),
                }
            )
    return pd.DataFrame(rows)


def continuous_evidence_statistics(subject_fits: pd.DataFrame) -> pd.DataFrame:
    """Test continuous-evidence coefficients across subjects."""
    rows = []
    terms = ("dt_slope_ms_per_unit", "accuracy_log_odds_per_unit")
    for (predictor, condition), group in subject_fits.groupby(
        ["predictor", "condition"], sort=True
    ):
        for term in terms:
            rows.append(
                {
                    "analysis": "continuous_evidence",
                    "predictor": predictor,
                    "term": term,
                    "condition": condition,
                    "test": "one_sample_vs_zero",
                    **one_sample_statistics(group[term]),
                }
            )
    for predictor, group in subject_fits.groupby("predictor", sort=True):
        for term in terms:
            wide = group.pivot(index="subject", columns="condition", values=term)
            if "fast" not in wide.columns or "slow" not in wide.columns:
                continue
            rows.append(
                {
                    "analysis": "continuous_evidence",
                    "predictor": predictor,
                    "term": term,
                    "condition": "fast_vs_slow",
                    "test": "paired_t_test",
                    **paired_subject_statistics(wide, "fast", "slow"),
                }
            )
    return pd.DataFrame(rows)


def _as_bool(value: object) -> object:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
        return pd.NA
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NA
    return bool(value)
