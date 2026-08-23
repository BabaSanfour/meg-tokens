"""Continuous and time-resolved evidence analyses.

Trial class and SPD summarize a trial with one label and one number. The
analyses here keep evidence continuous and time-resolved: cumulative
log-likelihood evidence from the token directions, the decline of evidence at
decision as more tokens are observed or as time passes, the temporal weighting
of tokens on choice, accuracy across decision-time quantiles, and regressions
that retain trials outside the discrete design classes.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Mapping, Sequence
from itertools import combinations
from typing import Final

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    HessianInversionWarning,
    PerfectSeparationError,
    PerfectSeparationWarning,
)

from meg_tokens.behavior.math.inference import (
    confidence_interval,
    equivalence_bound,
    one_sample_statistics,
    paired_subject_statistics,
    repeated_measures_anova,
    subject_bootstrap,
)
from meg_tokens.behavior.math.evidence import FIRST_ORDER_TOKEN_LOG_LR
from meg_tokens.behavior.math.probability import (
    TOTAL_TOKENS,
    success_probability,
    success_probability_profile,
)
from meg_tokens.behavior.schema import (
    parse_token_directions,
    validate_boolean_values,
)
from meg_tokens.behavior.trials import (
    CHANCE_ACCURACY,
    CLASS_NAMES,
    TASK_CONDITIONS,
    require_columns,
    task_trials,
)


# Explicit output-resolution policies for the reverse-correlation and
# conditional-accuracy analyses. They are defaults, not inferred quantities.
DEFAULT_KERNEL_JUMPS: Final[int] = 8
DEFAULT_ACCURACY_BINS: Final[int] = 5

# Conditional accuracy is resolved by trial class as well as by decision-time
# bin, because the pooled function is dominated by *which trials* land in each
# bin. "unclassified" is a stratum here rather than a discard: it is 43% of
# task trials and the largest single occupant of every bin.
UNCLASSIFIED_CLASS: Final[str] = "unclassified"
POOLED_STRATUM: Final[str] = "all"
ACCURACY_STRATA: Final[tuple[str, ...]] = (
    POOLED_STRATUM, *CLASS_NAMES.values(), UNCLASSIFIED_CLASS,
)
# Sensitivity threshold for the per-class slopes only. A class-by-bin cell can
# hold as few as one trial (misleading in the fastest bin), where accuracy is
# 0 or 1 by construction. Dropping those cells is *not* the primary estimate --
# it selects on decision time, which is the axis under test -- so both are
# reported and the figure draws the unselected one.
SPARSE_CELL_MIN_TRIALS: Final[int] = 5

# Early evidence is read after three jumps, and that is not an arbitrary cut:
# it is the earliest point the classification rule uses (`SP3 < 0.40` defines
# misleading) and 71% of trials are still running there, against 45% by jump 5
# and 17% by jump 8. It is also the reason the predictor is *not* continuous.
# Three tokens admit exactly four splits (3-0, 2-1, 1-2, 0-3), so
# `sp_design_early` takes exactly four values and its absolute deviation from
# chance takes two. A per-unit slope fitted on it is a difference between two
# means wearing a regression's clothes; the count is registered next to every
# such coefficient as `n_predictor_levels` so a reader can see that. The three
# jumps are baked into `sp_design_early` upstream, so there is no cut constant
# to set here -- only the column to read it from.
EARLY_EVIDENCE_PREDICTOR: Final[str] = "sp_design_early"
DEFAULT_EVIDENCE_PREDICTORS: Final[dict[str, float]] = {
    "sp_design_early": 0.5,
    "sum_log_lr_design_early": 0.0,
}
FIRST_ORDER_CRITERION_RESPONSE: Final[str] = (
    "chosen_sum_log_lr_first_order_at_decision"
)

INVALID_LOGIT_WARNINGS: Final[tuple[type[Warning], ...]] = (
    ConvergenceWarning,
    HessianInversionWarning,
    PerfectSeparationWarning,
)


def _subject_condition_groups(
    trials: pd.DataFrame,
) -> Iterator[tuple[object, str, pd.DataFrame]]:
    """Yield pooled and condition-specific task trials for every subject.

    Parameters
    ----------
    trials
        Task-trial table containing ``subject`` and ``condition`` columns.

    Yields
    ------
    tuple[object, str, pandas.DataFrame]
        Subject identifier, normalized group label, and the corresponding
        rows. Each subject is yielded once as ``"all"`` and once for every
        condition declared in :data:`TASK_CONDITIONS`.

    Notes
    -----
    Declared condition groups are yielded even when empty. This keeps group
    construction identical across all evidence analyses and lets each caller
    decide whether an empty cell should be retained or omitted.
    """
    for subject, subject_trials in trials.groupby("subject", sort=True):
        condition = subject_trials["condition"].astype(str).str.lower()
        yield subject, "all", subject_trials
        for name in TASK_CONDITIONS:
            label = name.lower()
            yield subject, label, subject_trials.loc[condition == label]


def _distinct_levels(values, decimals: int = 12) -> int:
    """How many distinct values a predictor actually takes.

    Rounded before counting: subtracting a centre from a probability grid
    leaves representation noise, so ``|0.2 - 0.5|`` and ``|0.8 - 0.5|`` differ
    in the last bit and an exact-equality count would report three levels
    where the design has two.
    """
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return int(len(np.unique(np.round(array, decimals))))


def _one_sample_summary(values, *, null: float = 0.0, coverage: float = 0.95) -> dict:
    """Mean, its interval, and the test of that mean against ``null``.

    ``estimate`` and the interval are on the measured scale, so an accuracy
    tested against chance still reads as an accuracy; ``t``, ``p`` and
    ``cohens_dz`` refer to the difference from ``null``.
    """
    statistics = one_sample_statistics(np.asarray(values, dtype=float) - null)
    low, high = confidence_interval(
        statistics["mean"], statistics["sem"], statistics["df"], coverage=coverage
    )
    return {
        "estimate": statistics["mean"] + null,
        "ci_low": low + null,
        "ci_high": high + null,
        "coverage": coverage,
        **{k: statistics[k] for k in ("n_subjects", "t", "p", "df", "cohens_dz")},
    }


def criterion_decline(
    features: pd.DataFrame,
    *,
    predictor: str = "decision_token_index",
    response: str = "logged_spd",
    require_design_time_alignment: bool = False,
) -> pd.DataFrame:
    """Fit each subject's evidence criterion as a function of trial duration.

    A negative slope means the subject accepted progressively weaker evidence
    the longer a trial ran, which is the behavioral signature of urgency
    gating. With ``predictor="decision_token_index"`` this measures criterion
    decline across observed tokens; with ``predictor="dt_ms"`` the intercept
    and slope are urgency parameters in evidence units per second.

    Parameters
    ----------
    features
        Canonical trial-feature table. Only eligible Fast and Slow task trials
        returned by :func:`~meg_tokens.behavior.trials.task_trials` are used.
    predictor
        Numeric feature used on the horizontal axis. ``dt_ms`` is converted to
        seconds before fitting; other predictors retain their stored units.
    response
        Numeric evidence-at-decision feature to model.
    require_design_time_alignment
        If ``True``, retain only trials whose complete 15-row token log makes
        the designed token sequence and commitment index directly alignable.
        This is required for first-order SumLogLR at a particular token time.

    Returns
    -------
    pandas.DataFrame
        One ordinary-least-squares fit per subject for pooled trials and for
        each task condition. Columns contain the fitted intercept, slope,
        slope standard error, usable-trial count, and convergence status.

    Notes
    -----
    The fitted model is ``response = intercept + slope * predictor``. Rows
    with non-finite predictor or response values are excluded. A fit is
    reported only when the design has full rank and more observations than
    coefficients; otherwise its estimates are ``NaN`` and ``converged`` is
    ``False``. Empty declared condition cells are retained in the output. No
    imputation, outlier removal, or alternative model is applied.
    """
    trials = task_trials(features)
    if require_design_time_alignment:
        require_columns(trials, ["design_time_alignment_valid"])
        validate_boolean_values(
            trials["design_time_alignment_valid"],
            field="design_time_alignment_valid",
        )
        trials = trials.loc[trials["design_time_alignment_valid"].astype(bool)]
    require_columns(trials, [predictor, response, "subject", "condition"])
    scale = 1000.0 if predictor == "dt_ms" else 1.0
    rows = []
    for subject, condition, selected in _subject_condition_groups(trials):
        x = (
            pd.to_numeric(selected[predictor], errors="coerce").to_numpy(dtype=float)
            / scale
        )
        y = pd.to_numeric(selected[response], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        design = np.column_stack([np.ones_like(x), x])
        converged = len(y) > design.shape[1] and (
            np.linalg.matrix_rank(design) == design.shape[1]
        )
        fit = sm.OLS(y, design).fit() if converged else None
        rows.append(
            {
                "subject": subject,
                "condition": condition,
                "predictor": predictor,
                "response": response,
                "n_trials": len(y),
                "intercept": (
                    float(fit.params[0]) if fit is not None else float("nan")
                ),
                "slope": float(fit.params[1]) if fit is not None else float("nan"),
                "slope_se": float(fit.bse[1]) if fit is not None else float("nan"),
                "converged": fit is not None,
            }
        )
    return pd.DataFrame(rows)


def criterion_decline_statistics(subject_fits: pd.DataFrame) -> pd.DataFrame:
    """Run group tests on subject-level criterion intercepts and slopes.

    Parameters
    ----------
    subject_fits
        Output of :func:`criterion_decline` or
        :func:`evidence_at_decision_responses`.

    Returns
    -------
    pandas.DataFrame
        One-sample tests against zero for every condition and paired Fast-minus-
        Slow tests for each predictor, response, and fitted term. Pairing uses
        subject identifiers and therefore excludes unpaired observations.

    Notes
    -----
    Intercepts and slopes are tested separately. No multiplicity correction is
    applied. Duplicate subject/condition rows cause the pivot operation to
    raise instead of being aggregated silently.
    """
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
                    **paired_subject_statistics(pair["fast"], pair["slow"]),
                }
            )
    return pd.DataFrame(rows)


def evidence_at_decision_responses(
    features: pd.DataFrame,
    *,
    predictor: str,
    responses: Sequence[str] = ("logged_spd", "logged_spd_log_odds"),
    require_design_time_alignment: bool = False,
) -> pd.DataFrame:
    """Fit the criterion against one predictor on every evidence scale.

    Success probability is bounded and compresses near 1, so a slope measured
    on it is not directly comparable with the log-odds criterion the
    urgency-gating literature reports. Both are fitted and stacked.

    Parameters
    ----------
    features
        Canonical trial-feature table.
    predictor
        Feature against which every requested evidence response is fitted.
    responses
        Evidence columns to fit independently. By default these are logged
        success probability and its natural-log-odds transform.
    require_design_time_alignment
        Forwarded to :func:`criterion_decline`; use it for responses derived
        from the designed token present at the commitment index.

    Returns
    -------
    pandas.DataFrame
        Concatenated subject-level fits, with the ``response`` column identifying
        the evidence scale used for each row.

    Notes
    -----
    Each response is fitted independently with the exact model and eligibility
    rules documented by :func:`criterion_decline`. The function does not
    transform a response or assume that the requested columns share units.
    """
    return pd.concat(
        [
            criterion_decline(
                features,
                predictor=predictor,
                response=response,
                require_design_time_alignment=require_design_time_alignment,
            )
            for response in responses
        ],
        ignore_index=True,
    )


def first_order_chosen_sum_log_lr(features: pd.DataFrame) -> pd.Series:
    """Return Cisek et al. (2009) first-order chosen-target SumLogLR.

    The canonical trial-feature table stores token lead against the correct
    target so predictors remain response-independent. Criterion estimation is
    the deliberate exception: Cisek et al. (2009) define evidence relative to
    the selected target. Correct choices preserve the stored sign; errors
    reverse it. The fixed first-order log likelihood per token then converts
    token lead to SumLogLR without the exact posterior's shrinking-horizon
    transformation.
    """
    require_columns(features, ["token_lead_at_decision", "isCorrect"])
    validate_boolean_values(
        features["isCorrect"], field="isCorrect", optional=True
    )
    lead = pd.to_numeric(features["token_lead_at_decision"], errors="coerce")
    correct = features["isCorrect"].astype("boolean")
    sign = pd.Series(np.nan, index=features.index, dtype=float)
    sign.loc[correct == True] = 1.0  # noqa: E712 - nullable Boolean comparison
    sign.loc[correct == False] = -1.0  # noqa: E712
    return lead * sign * FIRST_ORDER_TOKEN_LOG_LR


def first_order_criterion_decline(
    features: pd.DataFrame,
    *,
    predictor: str = "dt_ms",
) -> pd.DataFrame:
    """Fit Cisek et al. (2009) SumLogLR against continuous decision time.

    Trials without a complete token log are excluded because the token present
    at commitment cannot be aligned unambiguously. Commitments before the first
    token are also excluded: their zero SumLogLR is an anticipation, not an
    evidence criterion. The default fit is trial-level OLS against decision
    time in seconds; any 200-ms binning belongs only to the report display.
    """
    require_columns(features, ["decision_token_index"])
    decision_token = pd.to_numeric(features["decision_token_index"], errors="coerce")
    post_first_token = decision_token.gt(0).fillna(False)
    enriched = features.copy()
    enriched[FIRST_ORDER_CRITERION_RESPONSE] = first_order_chosen_sum_log_lr(
        enriched
    )
    return criterion_decline(
        enriched.loc[post_first_token],
        predictor=predictor,
        response=FIRST_ORDER_CRITERION_RESPONSE,
        require_design_time_alignment=True,
    )


DISCRETIZATION_OVERSHOOT_THRESHOLDS: Final[tuple[float, ...]] = (
    0.70, 0.75, 0.80, 0.85, 0.90,
)
POSTERIOR_EVIDENCE_SENSITIVITY_VARIANTS: Final[tuple[str, ...]] = (
    "baseline",
    "token0_excluded",
    "complete_log_only",
    "three_second_horizon",
)


def posterior_grid_by_token_index() -> pd.DataFrame:
    """Enumerate every exact success-probability state reachable at each token index.

    Returns
    -------
    pandas.DataFrame
        One row per ``(token_index, target_tokens)`` state, with
        ``other_tokens``, ``remaining_tokens``, and the exact
        :func:`~meg_tokens.behavior.math.probability.success_probability` for
        that state.

    Notes
    -----
    This is the task design's full state space, independent of any subject
    or trial: it does not read data and is deterministic given the fixed
    15-token, 8-token-majority rule. It is the reference grid used to
    quantify how much a fixed evidence criterion's apparent time slope owes
    to the posterior discretizing as the token horizon shrinks, rather than
    to a genuine change in the criterion.
    """
    rows = []
    for token_index in range(1, TOTAL_TOKENS + 1):
        remaining = TOTAL_TOKENS - token_index
        for target_tokens in range(token_index + 1):
            other_tokens = token_index - target_tokens
            rows.append(
                {
                    "token_index": token_index,
                    "target_tokens": target_tokens,
                    "other_tokens": other_tokens,
                    "remaining_tokens": remaining,
                    "success_probability": success_probability(
                        target_tokens, other_tokens, remaining
                    ),
                }
            )
    return pd.DataFrame(rows)


def posterior_grid_step_summary(grid: pd.DataFrame) -> pd.DataFrame:
    """Summarize the discrete spacing between reachable posterior states.

    Parameters
    ----------
    grid
        Output of :func:`posterior_grid_by_token_index`.

    Returns
    -------
    pandas.DataFrame
        One row per token index, with the count of distinct reachable
        success-probability values and the minimum, median, and maximum gap
        between adjacent ones.

    Notes
    -----
    A coarser grid (fewer states, larger gaps) at later token indices is what
    lets a fixed decision threshold acquire a larger apparent overshoot late
    in a trial; see :func:`discretization_overshoot_fits` for the
    counterfactual this motivates.
    """
    require_columns(grid, ["token_index", "success_probability"])
    rows = []
    for token_index, group in grid.groupby("token_index", sort=True):
        values = np.sort(group["success_probability"].unique())
        gaps = np.diff(values)
        rows.append(
            {
                "token_index": int(token_index),
                "n_reachable_states": int(len(values)),
                "min_step": float(gaps.min()) if len(gaps) else float("nan"),
                "median_step": float(np.median(gaps)) if len(gaps) else float("nan"),
                "max_step": float(gaps.max()) if len(gaps) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def discretization_overshoot_fits(
    features: pd.DataFrame,
    *,
    thresholds: Sequence[float] = DISCRETIZATION_OVERSHOOT_THRESHOLDS,
) -> pd.DataFrame:
    """Fit criterion decline to a fixed-threshold replay of each trial's design.

    Parameters
    ----------
    features
        Canonical trial-feature table.
    thresholds
        Candidate fixed success-probability thresholds. For each one, every
        eligible trial's own designed token sequence
        (:func:`~meg_tokens.behavior.math.probability.success_probability_profile`)
        is walked token by token until the leading side's exact probability
        first reaches the threshold; that token index and probability become
        one synthetic ``(decision_token_index, logged_spd)`` observation.
        Trials whose design never reaches a threshold contribute no
        observation for it.

    Returns
    -------
    pandas.DataFrame
        :func:`criterion_decline` output (predictor
        ``decision_token_index``, response ``logged_spd``) for the synthetic
        data at every threshold, stacked with a leading ``threshold`` column.

    Notes
    -----
    The simulated stopping rule is exactly time-invariant: it reacts only to
    the current exact probability, never to elapsed time or token count
    directly. Any slope this analysis returns is therefore attributable
    entirely to the posterior grid coarsening
    (:func:`posterior_grid_step_summary`) as the token horizon shrinks, not to
    a genuine urgency effect. It is a discretization-only counterfactual to
    compare the real, subject-fitted slope against -- not itself a claim
    about behavior.
    """
    trials = task_trials(features)
    require_columns(trials, ["token_directions", "subject", "condition"])
    fit_tables = []
    for threshold in thresholds:
        synthetic_rows = []
        for _, trial in trials.iterrows():
            try:
                directions = parse_token_directions(trial["token_directions"])
            except (TypeError, ValueError):
                continue
            if not directions:
                continue
            profile = success_probability_profile(directions, target=1)
            decided = None
            for token_index, probability in enumerate(profile, start=1):
                leading = max(probability, 1.0 - probability)
                if leading >= threshold:
                    decided = (token_index, leading)
                    break
            if decided is None:
                continue
            token_index, decided_sp = decided
            synthetic_rows.append(
                {
                    "subject": trial["subject"],
                    "condition": trial["condition"],
                    "decision_token_index": token_index,
                    "logged_spd": decided_sp,
                    "primary_analysis_eligible": True,
                }
            )
        if not synthetic_rows:
            continue
        fits = criterion_decline(
            pd.DataFrame(synthetic_rows),
            predictor="decision_token_index",
            response="logged_spd",
        )
        fits.insert(0, "threshold", threshold)
        fit_tables.append(fits)
    return pd.concat(fit_tables, ignore_index=True) if fit_tables else pd.DataFrame()


def discretization_overshoot_statistics(fits: pd.DataFrame) -> pd.DataFrame:
    """Run :func:`criterion_decline_statistics` separately for each threshold.

    Parameters
    ----------
    fits
        Output of :func:`discretization_overshoot_fits`.

    Returns
    -------
    pandas.DataFrame
        Group statistics for every threshold, stacked with a leading
        ``threshold`` column.
    """
    require_columns(fits, ["threshold"])
    stats_tables = []
    for threshold, group in fits.groupby("threshold", sort=True):
        group_stats = criterion_decline_statistics(group.drop(columns="threshold"))
        group_stats.insert(0, "threshold", threshold)
        stats_tables.append(group_stats)
    return pd.concat(stats_tables, ignore_index=True) if stats_tables else pd.DataFrame()


def posterior_evidence_sensitivity(
    features: pd.DataFrame,
    *,
    predictor: str = "dt_ms",
    response: str = "logged_spd",
) -> pd.DataFrame:
    """Refit the exact-posterior criterion under three deferred sensitivities.

    Parameters
    ----------
    features
        Canonical trial-feature table.
    predictor, response
        Forwarded to :func:`criterion_decline` for every variant.

    Returns
    -------
    pandas.DataFrame
        :func:`criterion_decline` output for four variants, stacked with a
        leading ``sensitivity`` column: ``baseline`` (the unrestricted
        eligible-trial fit, matching the current ``urgency``/``F10``
        derivative), ``token0_excluded`` (commitments before the first
        token removed), ``complete_log_only`` (restricted to trials with a
        complete 15-row token log), and ``three_second_horizon`` (restricted
        to ``dt_ms <= 3000``).

    Notes
    -----
    These are the three sensitivities the audit plan named as needing to be
    checked "only when they answer a real ambiguity" -- this function runs
    all three unconditionally so the comparison is available; whether any of
    them changes the reported conclusion is an interpretive judgment left to
    the write-up, not decided here.
    """
    require_columns(
        features,
        ["decision_token_index", "dt_ms", "design_time_alignment_valid"],
    )
    decision_token = pd.to_numeric(features["decision_token_index"], errors="coerce")
    complete_log = features["design_time_alignment_valid"].astype(bool)
    dt_ms = pd.to_numeric(features["dt_ms"], errors="coerce")
    variants = {
        "baseline": features,
        "token0_excluded": features.loc[decision_token.gt(0).fillna(False)],
        "complete_log_only": features.loc[complete_log],
        "three_second_horizon": features.loc[dt_ms <= 3000.0],
    }
    fit_tables = []
    for name in POSTERIOR_EVIDENCE_SENSITIVITY_VARIANTS:
        fits = criterion_decline(variants[name], predictor=predictor, response=response)
        fits.insert(0, "sensitivity", name)
        fit_tables.append(fits)
    return pd.concat(fit_tables, ignore_index=True)


def posterior_evidence_sensitivity_statistics(fits: pd.DataFrame) -> pd.DataFrame:
    """Run :func:`criterion_decline_statistics` separately for each sensitivity.

    Parameters
    ----------
    fits
        Output of :func:`posterior_evidence_sensitivity`.

    Returns
    -------
    pandas.DataFrame
        Group statistics for every variant, stacked with a leading
        ``sensitivity`` column.
    """
    require_columns(fits, ["sensitivity"])
    stats_tables = []
    for name in POSTERIOR_EVIDENCE_SENSITIVITY_VARIANTS:
        group = fits.loc[fits["sensitivity"] == name]
        if group.empty:
            continue
        group_stats = criterion_decline_statistics(group.drop(columns="sensitivity"))
        group_stats.insert(0, "sensitivity", name)
        stats_tables.append(group_stats)
    return pd.concat(stats_tables, ignore_index=True) if stats_tables else pd.DataFrame()


def posterior_evidence_criterion_agreement(
    urgency_fits: pd.DataFrame,
    criterion_fits: pd.DataFrame,
) -> pd.DataFrame:
    """Correlate subject-level F10 coefficients against F09 coefficients.

    Parameters
    ----------
    urgency_fits
        Output of :func:`evidence_at_decision_responses` (the ``urgency``
        derivative), containing both ``logged_spd`` and
        ``logged_spd_log_odds`` responses.
    criterion_fits
        Output of :func:`first_order_criterion_decline` (the
        ``criteriondecline`` derivative).

    Returns
    -------
    pandas.DataFrame
        One row per ``(response, term)`` pair, with the complete-pair subject
        count and Pearson/Spearman correlation between the pooled-condition
        (``condition == "all"``) subject coefficients of the two analyses.

    Notes
    -----
    At least three complete subjects and nonzero variance in both measures
    are required; otherwise the pair is omitted. A correlation here is
    convergence between two distinct evidence measures, not proof that they
    are interchangeable -- the exact posterior and the first-order SumLogLR
    are fitted on different response scales with different eligibility
    rules.
    """
    require_columns(
        urgency_fits,
        ["subject", "condition", "response", "converged", "intercept", "slope"],
    )
    require_columns(
        criterion_fits,
        ["subject", "condition", "converged", "intercept", "slope"],
    )
    criterion_pooled = criterion_fits.loc[
        (criterion_fits["condition"] == "all") & criterion_fits["converged"]
    ].set_index("subject")
    rows = []
    for response, response_fits in urgency_fits.groupby("response", sort=True):
        urgency_pooled = response_fits.loc[
            (response_fits["condition"] == "all") & response_fits["converged"]
        ].set_index("subject")
        common = urgency_pooled.index.intersection(criterion_pooled.index)
        for term in ("intercept", "slope"):
            pair = pd.DataFrame(
                {
                    "criterion": criterion_pooled.loc[common, term],
                    "urgency": urgency_pooled.loc[common, term],
                }
            ).apply(pd.to_numeric, errors="coerce").dropna()
            n_subjects = int(len(pair))
            if n_subjects < 3:
                continue
            values_criterion = pair["criterion"].to_numpy(dtype=float)
            values_urgency = pair["urgency"].to_numpy(dtype=float)
            if np.std(values_criterion) == 0 or np.std(values_urgency) == 0:
                continue
            pearson = stats.pearsonr(values_criterion, values_urgency)
            spearman = stats.spearmanr(values_criterion, values_urgency)
            rows.append(
                {
                    "response": response,
                    "term": term,
                    "n_subjects": n_subjects,
                    "pearson_r": float(pearson.statistic),
                    "pearson_p": float(pearson.pvalue),
                    "spearman_rho": float(spearman.statistic),
                    "spearman_p": float(spearman.pvalue),
                }
            )
    return pd.DataFrame(rows)


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

    Parameters
    ----------
    features
        Canonical trial-feature table containing designed token directions,
        choice, and the number of tokens observed before commitment.
    n_jumps
        Positive number of sequential token positions included in the kernel.
        Eight is an explicit output-resolution default, not a quantity inferred
        from the supplied data.

    Returns
    -------
    pandas.DataFrame
        One row per subject, condition, and jump. ``logistic_weight`` is the
        fitted choice coefficient; ``mean_signed_direction`` is the mean token
        direction relative to the eventual choice; ``n_trials_token_seen``
        reports the observations contributing to that jump.

    Notes
    -----
    The logistic response is target 1 versus target 2. Tokens occurring after
    commitment enter the regression design as zero and are excluded from the
    model-free mean. A logistic model is reported only when both choices are
    present, the design is full rank, and statsmodels converges without a
    separation, Hessian, or convergence warning. Failed models produce
    ``NaN`` weights; they are not replaced with another estimator. Invalid or
    missing choices and decision-token indices raise rather than being guessed.
    """
    if n_jumps < 1:
        raise ValueError("n_jumps must be at least 1")
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
            seen_value = pd.to_numeric(
                trial["decision_token_index"], errors="raise"
            )
            if pd.isna(seen_value):
                raise ValueError("decision_token_index must not be missing")
            seen = int(seen_value)
            choice = int(trial["choice_side"])
            if choice not in (1, 2):
                raise ValueError("choice_side must be 1 or 2")
            choices[row_index] = 1.0 if choice == 1 else 0.0
            for jump_index in range(min(n_jumps, len(directions), seen)):
                signed = 1.0 if directions[jump_index] == 1 else -1.0
                predictors[row_index, jump_index] = signed
                toward_choice[row_index, jump_index] = (
                    1.0 if directions[jump_index] == choice else -1.0
                )
        design = np.column_stack([np.ones(len(selected)), predictors])
        fit = None
        if (
            len(choices) > design.shape[1]
            and len(np.unique(choices)) == 2
            and np.linalg.matrix_rank(design) == design.shape[1]
        ):
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    candidate = sm.Logit(choices, design).fit(disp=False)
                invalid = any(
                    isinstance(item.message, INVALID_LOGIT_WARNINGS)
                    for item in caught
                )
                if (
                    not invalid
                    and candidate.mle_retvals.get("converged", False)
                ):
                    fit = candidate
            except (np.linalg.LinAlgError, PerfectSeparationError):
                pass
        weights = (
            dict(zip(["intercept", *jumps], fit.params))
            if fit is not None
            else {}
        )
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
                    "converged": fit is not None,
                }
            )
    return pd.DataFrame(rows)


def reverse_correlation_statistics(kernels: pd.DataFrame) -> pd.DataFrame:
    """Run group tests on the psychophysical kernel at every jump.

    Parameters
    ----------
    kernels
        Output of :func:`reverse_correlation`.

    Returns
    -------
    pandas.DataFrame
        One-sample tests against zero and subject-paired Fast-minus-Slow tests,
        computed separately for logistic weights and model-free signed means,
        plus one ``all_jumps_vs_zero`` row per metric and condition summarizing
        whether every jump departs from zero.

    Notes
    -----
    The Fast-minus-Slow rows carry ``sem_difference`` -- the standard error of
    the *paired difference*, which is the quantity an error bar on a difference
    needs. ``paired_subject_statistics`` reports ``sem_a`` and ``sem_b``, the
    standard errors of the two conditions, and neither is it. Without this
    column a caller has to recover it as ``mean_difference / t``, which divides
    by zero exactly when the difference is null -- the case a kernel figure
    most needs to draw. They also carry an ``equivalence_bound``, so a jump
    where the conditions do not differ states what it excludes.

    Every test is at a single jump. ``p_threshold`` carries the Bonferroni
    level for the number of jumps tested, so a figure marking per-jump
    significance can use a corrected threshold rather than 0.05; ``p`` itself
    is uncorrected.

    The ``all_jumps_vs_zero`` row exists so the report layer never has to
    reduce a column of p-values itself: it records how many jumps were tested,
    how many cleared the corrected threshold, and the largest p among them.

    Non-finite subject estimates are excluded by the shared inference helpers.
    Duplicate subject/condition/jump rows cause the paired pivot to raise
    instead of being aggregated silently.
    """
    rows = []
    n_jumps = int(kernels["jump"].nunique())
    threshold = 0.05 / n_jumps if n_jumps else float("nan")
    for metric in ("logistic_weight", "mean_signed_direction"):
        for (condition, jump), group in kernels.groupby(["condition", "jump"], sort=True):
            rows.append(
                {
                    "analysis": "reverse_correlation",
                    "metric": metric,
                    "condition": condition,
                    "jump": int(jump),
                    "test": "one_sample_vs_zero",
                    "n_jumps_tested": n_jumps,
                    "p_threshold": threshold,
                    **one_sample_statistics(group[metric]),
                }
            )
        # One row per condition standing in for "every weight differs from
        # zero", so the figure states it instead of deriving it.
        for condition, group in kernels.groupby("condition", sort=True):
            per_jump = [
                one_sample_statistics(cell[metric])["p"]
                for _, cell in group.groupby("jump", sort=True)
            ]
            finite = [value for value in per_jump if np.isfinite(value)]
            rows.append({
                "analysis": "reverse_correlation", "metric": metric,
                "condition": condition, "test": "all_jumps_vs_zero",
                "n_jumps_tested": len(per_jump), "p_threshold": threshold,
                "n_jumps_significant": sum(value < threshold for value in finite),
                "p": max(finite) if finite else float("nan"),
            })
        for jump, group in kernels.groupby("jump", sort=True):
            wide = group.pivot(index="subject", columns="condition", values=metric)
            if "fast" not in wide.columns or "slow" not in wide.columns:
                continue
            differences = (wide["fast"] - wide["slow"]).dropna()
            paired = paired_subject_statistics(wide["fast"], wide["slow"])
            rows.append(
                {
                    "analysis": "reverse_correlation",
                    "metric": metric,
                    "condition": "fast_vs_slow",
                    "jump": int(jump),
                    "test": "paired_t_test",
                    "n_jumps_tested": n_jumps,
                    "p_threshold": threshold,
                    **paired,
                    # The SEM of the difference, which is what an error bar on
                    # a difference is; sem_a and sem_b are not it. The 95%
                    # interval is registered beside it so the report layer
                    # never has to reach for a t quantile.
                    **_one_sample_summary(differences),
                    "sem_difference": (
                        float(differences.std(ddof=1) / np.sqrt(len(differences)))
                        if len(differences) > 1 else float("nan")
                    ),
                    **equivalence_bound(differences),
                }
            )
    return pd.DataFrame(rows)


def conditional_accuracy_functions(
    features: pd.DataFrame,
    *,
    n_bins: int = DEFAULT_ACCURACY_BINS,
) -> pd.DataFrame:
    """Return accuracy across within-subject DT quantile bins, by trial class.

    A conditional accuracy function that falls with DT is usually read as a
    declining criterion. On this design that reading is unsafe pooled, because
    the bins do not hold the same *stimuli*: easy trials are fast and crowd the
    first bin, ambiguous ones are slow and crowd the last. This function
    therefore resolves each bin by trial class and also returns a
    class-adjusted accuracy, so the compositional and the within-class parts of
    the pooled function can be separated instead of conflated.

    Bin edges are per subject and condition, so that between-subject
    differences in overall speed cannot shift trials between bins. They are cut
    on *all* of that subject's trials in the subset, not within class, so every
    class is read at the same decision times and the class rows partition the
    pooled row exactly.

    Parameters
    ----------
    features
        Canonical trial-feature table with decision time, correctness, and
        trial class.
    n_bins
        Positive requested number of equal-count decision-time bins within each
        subject and condition. Five is an explicit resolution default.
        Duplicate quantile edges may reduce the realized count.

    Returns
    -------
    pandas.DataFrame
        One row per subject, condition subset (``all``, ``fast``, ``slow``),
        realized bin, and stratum. ``trial_class_name`` is ``"all"`` for the
        pooled row and a class name -- including ``"unclassified"`` -- for each
        stratum row. Every row carries its trial count, its own mean decision
        time, its accuracy, and its share of the bin; the pooled rows
        additionally carry class-adjusted accuracy.

    Notes
    -----
    Missing decision times and correctness values are excluded. Correctness
    values are validated rather than coerced. Bins are constructed with
    :func:`pandas.qcut`; subsets that cannot form a bin are omitted. Bin
    numbers therefore denote within-subject quantile order, not common
    decision-time boundaries across subjects.

    The class adjustment matches
    :func:`~meg_tokens.behavior.analyses.design_effects.within_block_decile_profile`:
    each trial has its subject's mean accuracy *for its own class* removed and
    that subject's grand mean added back, so the gap between the observed and
    adjusted profiles *is* the compositional contribution. It is computed once
    per subject over all their trials, not separately within each condition
    subset, so a subject's adjusted values do not depend on which subset they
    are being averaged into. Unclassified trials keep their own level rather
    than being dropped, since they are 43% of the data and roughly 42% of every
    bin.

    The adjustment removes only *between-class* composition. Evidence strength
    still varies within a class -- most of all within ``unclassified``, which
    is a residual category rather than a designed one -- so a residual
    adjusted slope remains an upper bound on any genuine accuracy-versus-time
    relation, never a clean estimate of one.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    trials = task_trials(features)
    require_columns(
        trials, ["dt_ms", "isCorrect", "subject", "condition", "trial_class_name"]
    )
    validate_boolean_values(trials["isCorrect"], field="isCorrect", optional=True)
    trials = trials.assign(
        _dt=pd.to_numeric(trials["dt_ms"], errors="coerce"),
        _correct=trials["isCorrect"].astype("boolean").astype(float),
        _class=trials["trial_class_name"].astype(str),
    )
    trials = trials.loc[trials["_dt"].notna() & trials["_correct"].notna()].copy()
    # Per subject, over every one of their trials -- see the class-adjustment
    # note above for why this is not recomputed inside each condition subset.
    trials["_adjusted"] = (
        trials["_correct"]
        - trials.groupby(["subject", "_class"])["_correct"].transform("mean")
        + trials.groupby("subject")["_correct"].transform("mean")
    )

    rows = []
    for subject, condition, selected in _subject_condition_groups(trials):
        if selected.empty:
            continue
        try:
            bins = pd.qcut(selected["_dt"], n_bins, labels=False, duplicates="drop")
        except ValueError:  # fewer distinct values than requested bins
            continue
        for bin_index in sorted(set(bins.dropna().astype(int))):
            cell = selected.loc[bins == bin_index]
            for stratum in ACCURACY_STRATA:
                stratum_cell = (
                    cell if stratum == POOLED_STRATUM
                    else cell.loc[cell["_class"] == stratum]
                )
                if stratum_cell.empty:
                    continue
                rows.append(
                    {
                        "subject": subject,
                        "condition": condition,
                        "dt_bin": int(bin_index) + 1,
                        "trial_class_name": stratum,
                        "n_trials": int(len(stratum_cell)),
                        "proportion_of_bin": float(len(stratum_cell) / len(cell)),
                        "mean_dt_ms": float(stratum_cell["_dt"].mean()),
                        "accuracy": float(stratum_cell["_correct"].mean()),
                        "accuracy_class_adjusted": (
                            float(stratum_cell["_adjusted"].mean())
                            if stratum == POOLED_STRATUM
                            else float("nan")
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _accuracy_slopes(
    functions: pd.DataFrame,
    *,
    measure: str,
    predictor: str,
    predictor_scale: float,
    min_cell_trials: int,
) -> pd.DataFrame:
    """Per-subject least-squares slope of ``measure`` across decision-time bins.

    One row per subject, condition subset and stratum. ``predictor_scale``
    divides the predictor before fitting, which is how the per-second slope is
    expressed in the units of the axis the figure draws (``mean_dt_ms`` over
    1000) rather than in per-millisecond values nothing quotes.

    Cells holding fewer than ``min_cell_trials`` trials are dropped before
    fitting. A subject needs three surviving bins and a full-rank
    intercept-plus-predictor design; anything less yields NaN, which the
    group test then excludes.
    """
    usable = functions.loc[
        (functions["n_trials"] >= min_cell_trials)
        & np.isfinite(pd.to_numeric(functions[measure], errors="coerce"))
    ]
    rows = []
    for (subject, condition, stratum), group in usable.groupby(
        ["subject", "condition", "trial_class_name"], sort=True
    ):
        ordered = group.sort_values("dt_bin")
        predictor_values = (
            ordered[predictor].to_numpy(dtype=float) / predictor_scale
        )
        design = np.column_stack([np.ones(len(ordered)), predictor_values])
        response = ordered[measure].to_numpy(dtype=float)
        fit = (
            sm.OLS(response, design).fit()
            if len(response) > design.shape[1]
            and np.linalg.matrix_rank(design) == design.shape[1]
            else None
        )
        rows.append(
            {
                "subject": subject,
                "condition": condition,
                "trial_class_name": stratum,
                "slope": float(fit.params[1]) if fit is not None else float("nan"),
            }
        )
    return pd.DataFrame(rows)


# Each slope is registered in two units: per decision-time quintile, which is
# how the bins are defined, and per second of that bin's mean decision time,
# which is the unit of the axis the figure draws. Reporting only the first on
# a millisecond axis is what made the old annotation unreadable.
_SLOPE_UNITS: Final[tuple[tuple[str, str, float], ...]] = (
    ("accuracy_slope_per_bin", "dt_bin", 1.0),
    ("accuracy_slope_per_second", "mean_dt_ms", 1000.0),
)

# Every stratum except the pooled one, in the order the report draws them.
_CLASS_STRATA: Final[tuple[str, ...]] = (*CLASS_NAMES.values(), UNCLASSIFIED_CLASS)


def _change_decomposition(functions: pd.DataFrame) -> pd.DataFrame:
    """Split each subject's first-to-last accuracy change into composition and
    within-class parts.

    For one class, ``p_last * a_last - p_first * a_first`` is identically

        (p_last - p_first) * (a_last + a_first) / 2      <- composition
      + (p_last + p_first) / 2 * (a_last - a_first)      <- within class

    so summing both terms over classes reproduces that subject's own observed
    change exactly. There is no residual for a mistake to hide in, and because
    the split is per subject it can be tested across subjects like everything
    else here rather than being read off group means -- which is not the same
    number, since the mean of a product is not the product of means.

    A subject missing a class from either end bin cannot be decomposed: an
    absent class has a proportion of zero but no accuracy, so neither term is
    defined. Those subjects are dropped, and the row records how many remain.
    """
    rows = []
    for condition, group in functions.groupby("condition", sort=True):
        bins = sorted(group["dt_bin"].unique())
        if len(bins) < 2:
            continue
        first, last = bins[0], bins[-1]
        share = group.pivot_table(
            index=["subject", "dt_bin"], columns="trial_class_name",
            values="proportion_of_bin",
        )
        accuracy = group.pivot_table(
            index=["subject", "dt_bin"], columns="trial_class_name", values="accuracy",
        )
        strata = [name for name in _CLASS_STRATA if name in share.columns]
        for subject in share.index.get_level_values("subject").unique():
            try:
                p_first, p_last = share.loc[(subject, first), strata], share.loc[(subject, last), strata]
                a_first, a_last = accuracy.loc[(subject, first), strata], accuracy.loc[(subject, last), strata]
            except KeyError:
                continue
            if p_first.isna().any() or p_last.isna().any():
                continue
            if a_first.isna().any() or a_last.isna().any():
                continue
            composition = (p_last - p_first) * (a_last + a_first) / 2.0
            within = (p_last + p_first) / 2.0 * (a_last - a_first)
            rows.append({
                "subject": subject, "condition": condition,
                **{f"composition_{name}": float(composition[name]) for name in strata},
                **{f"within_class_{name}": float(within[name]) for name in strata},
                "composition": float(composition.sum()),
                "within_class": float(within.sum()),
                "observed_change": float(composition.sum() + within.sum()),
                # The classes the design names, i.e. what would remain if the
                # unclassified stratum were dropped from the figure.
                "within_class_designed": float(
                    within[[n for n in strata if n != UNCLASSIFIED_CLASS]].sum()
                ),
            })
    return pd.DataFrame(rows)


# The agreement between the two composition routes is a ratio of subject means
# against another such ratio, so it is bootstrapped over subjects rather than
# t-tested. Both settings are fixed: a derivative that changes on re-run would
# make every number the write-up quotes unverifiable.
COMPOSITION_SHARE_RESAMPLES: Final[int] = 20000
COMPOSITION_SHARE_SEED: Final[int] = 0


def _mean_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Ratio of two means over the last axis, NaN on a vanishing denominator.

    Reducing over the last axis is what lets one expression serve both the
    point estimate and a whole array of bootstrap resamples. Dividing by a mean
    of exactly zero yields NaN rather than an infinity, which the bootstrap
    then drops instead of propagating into a percentile.
    """
    top = np.mean(numerator, axis=-1)
    bottom = np.mean(denominator, axis=-1)
    return np.divide(
        top, bottom, out=np.full(np.shape(bottom), np.nan), where=bottom != 0.0
    )


def _composition_share_agreement(functions: pd.DataFrame) -> list[dict]:
    """Test whether the endpoint and slope routes to the composition share agree.

    Two independent constructions answer "how much of the conditional accuracy
    function is composition". The **endpoint** route splits the first-to-last
    change and takes the composition part's share of it; the **slope** route
    refits a line over all five bins before and after the class adjustment and
    takes the share of the slope the adjustment removes. They are different
    quantities on different supports and need not have matched.

    Both are ratios of subject means, so neither has a per-subject value to
    average -- a subject whose own observed change is near zero has an
    exploding share, and one such subject exists in each route here. The
    sampling unit is therefore resampled instead: one index array drives both
    routes per replicate, which keeps them paired and lets their difference be
    interval-estimated. See :func:`~meg_tokens.behavior.math.inference.subject_bootstrap`.

    Returns
    -------
    list of dict
        Three rows per condition and slope unit: each route's share with its
        own bootstrap interval, then the endpoint-minus-slope difference with
        an interval, an achieved significance level and an equivalence bound.
        The bound is the row that matters. A difference indistinguishable from
        zero says only that the test could not separate the routes, and the
        bound says how large a separation it would have missed -- which is the
        difference between "the routes agree" and "the routes were never
        measured precisely enough to disagree".

    Notes
    -----
    The endpoint share does not depend on the slope unit, so its row repeats
    across the two unit blocks. Each block is a complete comparison that way,
    which is worth one duplicated value.

    Per-class slopes are not involved: the routes are compared on the pooled
    stratum, which is the one the write-up's share is quoted for.
    """
    rows: list[dict] = []
    decomposition = _change_decomposition(functions)
    if decomposition.empty:
        return rows
    for term, predictor, scale in _SLOPE_UNITS:
        slopes = {
            measure: _accuracy_slopes(
                functions, measure=measure, predictor=predictor,
                predictor_scale=scale, min_cell_trials=1,
            )
            for measure in ("accuracy", "accuracy_class_adjusted")
        }
        for condition, group in decomposition.groupby("condition", sort=True):
            change = group.set_index("subject")[["composition", "observed_change"]]
            pooled = {
                measure: table.loc[
                    (table["condition"] == condition)
                    & (table["trial_class_name"] == POOLED_STRATUM)
                ].set_index("subject")["slope"]
                for measure, table in slopes.items()
            }
            subjects = change.index
            for series in pooled.values():
                subjects = subjects.intersection(series.dropna().index)
            if len(subjects) < 2:
                continue
            composition = change.loc[subjects, "composition"].to_numpy(dtype=float)
            observed_change = change.loc[subjects, "observed_change"].to_numpy(
                dtype=float
            )
            observed = pooled["accuracy"].loc[subjects].to_numpy(dtype=float)
            adjusted = (
                pooled["accuracy_class_adjusted"].loc[subjects].to_numpy(dtype=float)
            )

            def endpoint_share(index: np.ndarray) -> np.ndarray:
                return _mean_ratio(composition[index], observed_change[index])

            def slope_share(index: np.ndarray) -> np.ndarray:
                # The ratio of the two *group* slopes, which is the quantity
                # the write-up quotes. Taking magnitudes subject by subject
                # instead would fold the handful of subjects whose accuracy
                # rises with decision time into the group decline and inflate
                # both slopes.
                removed = np.abs(_mean_ratio(adjusted[index], observed[index]))
                # The adjustment removes that fraction of the slope; the share
                # composition accounts for is what is left.
                return 1.0 - removed

            statistics = (
                ("endpoint_share", endpoint_share, False),
                ("slope_share", slope_share, False),
                (
                    "endpoint_minus_slope_share",
                    lambda index: endpoint_share(index) - slope_share(index),
                    True,
                ),
            )
            for measure, statistic, tests_zero in statistics:
                summary = subject_bootstrap(
                    statistic, len(subjects),
                    n_resamples=COMPOSITION_SHARE_RESAMPLES,
                    seed=COMPOSITION_SHARE_SEED,
                )
                if not tests_zero:
                    # A share has no null at zero: "this route attributes none
                    # of the decline to composition" is not a hypothesis anyone
                    # holds, and a p against it would only ever be quoted by
                    # mistake. The interval is the whole content of these rows.
                    summary = {
                        key: value for key, value in summary.items()
                        if key not in {"p", "equivalence_bound",
                                       "equivalence_bound_coverage"}
                    }
                rows.append({
                    "analysis": "conditional_accuracy",
                    "term": "composition_share_agreement",
                    "measure": measure, "condition": condition,
                    "trial_class_name": POOLED_STRATUM,
                    "slope_unit": term, "min_cell_trials": 1,
                    "test": (
                        "subject_bootstrap_vs_zero" if tests_zero
                        else "subject_bootstrap_descriptive"
                    ),
                    **summary,
                })
    return rows


def conditional_accuracy_statistics(functions: pd.DataFrame) -> pd.DataFrame:
    """Test conditional accuracy, its trend, and the composition behind it.

    Parameters
    ----------
    functions
        Output of :func:`conditional_accuracy_functions`.

    Returns
    -------
    pandas.DataFrame
        Rows keyed by ``term``: per-bin accuracy against chance
        (``mean_accuracy``), each class's share of each bin
        (``class_proportion`` and its first-to-last change), the split of the
        first-to-last change into composition and within-class parts
        (``accuracy_change_decomposition``), the test that this split and the
        class-adjusted slope reach the same composition share
        (``composition_share_agreement``), the linear
        accuracy trend across bins in two units
        (``accuracy_slope_per_bin``, ``accuracy_slope_per_second``), and the
        Fast-minus-Slow contrast on each trend
        (``accuracy_slope_fast_minus_slow``). Every row carries its stratum in
        ``trial_class_name`` and, where the distinction exists, its measure in
        ``measure`` and its cell threshold in ``min_cell_trials``.

    Notes
    -----
    Accuracy is tested against **chance**, not zero: the task is a
    two-alternative forced choice, so a bin at 0.5 carries no information and a
    bin below it -- which misleading trials are, by construction -- is a real
    result. ``estimate`` is the accuracy itself and ``ci_low``/``ci_high``
    bracket it on the accuracy scale; ``t``, ``p`` and ``cohens_dz`` refer to
    the difference from chance.

    Slope rows carry a 95% interval **and** an ``equivalence_bound``: the
    smallest slope a two-one-sided-tests check at alpha = .05 excludes, read
    off the 90% interval, whose coverage is stated in
    ``equivalence_bound_coverage`` so the two are never confused. It is
    reported for every stratum so a flat conditional accuracy function can be
    quoted as a bound rather than as "n.s.".

    The decomposition and the class-adjusted slope answer the same question in
    different units and need not match to the digit: one splits the change
    between the first and last bin, the other refits a line over all five. The
    ``share_of_observed_change`` on the composition row is comparable to
    ``1 - |adjusted slope| / |observed slope|``; quoting either is fine,
    quoting one as though it were the other is not. Whether they actually
    agree is tested rather than eyeballed -- ``composition_share_agreement``
    carries both shares, their difference, and the bound on that difference,
    which is what says how much the closeness of the two numbers is worth.
    The ``designed_classes`` row is the within-class part with
    ``unclassified`` removed, which is what the figure would be claiming if it
    drew only the three named classes.

    Per-class slopes are reported twice, at ``min_cell_trials`` of 1 and
    :data:`SPARSE_CELL_MIN_TRIALS`. The threshold is a sensitivity check, not
    the primary estimate: dropping thin cells selects on decision time, which
    is the axis under test.

    Subject slopes require at least three realized bins and a full-rank
    design. No multiplicity correction is applied.

    Easy trials sit at ceiling in the fast bins -- nearly every subject scores
    1.0 -- so SciPy reports a precision-loss warning on those accuracy-versus-
    chance tests and their t statistics run into the hundreds. That is the
    stimulus class behaving as designed, not a computation to fix: read those
    cells as "at ceiling" and do not quote the t.
    """
    rows: list[dict] = []

    for (condition, stratum, dt_bin), group in functions.groupby(
        ["condition", "trial_class_name", "dt_bin"], sort=True
    ):
        rows.append({
            "analysis": "conditional_accuracy", "term": "mean_accuracy",
            "condition": condition, "trial_class_name": stratum,
            "dt_bin": int(dt_bin), "test": "one_sample_vs_chance",
            "mean_dt_ms": float(group["mean_dt_ms"].mean()),
            "mean_n_trials": float(group["n_trials"].mean()),
            **_one_sample_summary(group["accuracy"], null=CHANCE_ACCURACY),
        })
        if stratum != POOLED_STRATUM:
            rows.append({
                "analysis": "conditional_accuracy", "term": "class_proportion",
                "condition": condition, "trial_class_name": stratum,
                "dt_bin": int(dt_bin), "test": "descriptive",
                "n_subjects": int(group["proportion_of_bin"].notna().sum()),
                "estimate": float(group["proportion_of_bin"].mean()),
            })

    # How much the bins differ in *what they hold*: the claim that the pooled
    # function is composition rests on this moving, so it is tested rather
    # than left as two numbers read off the per-bin rows.
    for (condition, stratum), group in functions.groupby(
        ["condition", "trial_class_name"], sort=True
    ):
        # The pooled stratum is every trial in the bin, so its share is 1.0 in
        # every bin by construction. A row saying so would carry a NaN test.
        if stratum == POOLED_STRATUM:
            continue
        wide = group.pivot_table(
            index="subject", columns="dt_bin", values="proportion_of_bin"
        )
        if wide.shape[1] < 2:
            continue
        changes = (wide[wide.columns[-1]] - wide[wide.columns[0]]).dropna()
        rows.append({
            "analysis": "conditional_accuracy",
            "term": "class_proportion_first_to_last",
            "condition": condition, "trial_class_name": stratum,
            "test": "one_sample_vs_zero",
            "first_bin_proportion": float(wide[wide.columns[0]].mean()),
            "last_bin_proportion": float(wide[wide.columns[-1]].mean()),
            **_one_sample_summary(changes),
        })

    # Where the first-to-last change actually comes from. The class-adjusted
    # slope answers the same question with one number; this attributes it by
    # class, which is what says whether any single stratum is carrying the
    # residual -- and one is.
    decomposition = _change_decomposition(functions)
    # An empty result has no columns at all, so grouping it raises rather than
    # yielding nothing. It is reached whenever no condition realized two bins,
    # or every subject was missing a class from an end bin.
    if decomposition.empty:
        decomposition = pd.DataFrame(columns=["condition"])
    for condition, group in decomposition.groupby("condition", sort=True):
        for part in ("composition", "within_class"):
            rows.append({
                "analysis": "conditional_accuracy",
                "term": "accuracy_change_decomposition", "measure": part,
                "condition": condition, "trial_class_name": POOLED_STRATUM,
                "test": "one_sample_vs_zero",
                "observed_change": float(group["observed_change"].mean()),
                "share_of_observed_change": float(
                    group[part].mean() / group["observed_change"].mean()
                ),
                **_one_sample_summary(group[part]),
                **equivalence_bound(group[part]),
            })
            for stratum in _CLASS_STRATA:
                column = f"{part}_{stratum}"
                if column not in group:
                    continue
                rows.append({
                    "analysis": "conditional_accuracy",
                    "term": "accuracy_change_decomposition", "measure": part,
                    "condition": condition, "trial_class_name": stratum,
                    "test": "one_sample_vs_zero",
                    **_one_sample_summary(group[column]),
                    **equivalence_bound(group[column]),
                })
        # What the within-class part would be if `unclassified` were dropped.
        # It is the justification for drawing that stratum at all, so it is a
        # registered row rather than an argument made in prose.
        rows.append({
            "analysis": "conditional_accuracy",
            "term": "accuracy_change_decomposition", "measure": "within_class",
            "condition": condition, "trial_class_name": "designed_classes",
            "test": "one_sample_vs_zero",
            **_one_sample_summary(group["within_class_designed"]),
            **equivalence_bound(group["within_class_designed"]),
        })

    # Whether the two routes to the composition share -- this split and the
    # class-adjusted slope below -- actually land on the same number.
    rows.extend(_composition_share_agreement(functions))

    for term, predictor, scale in _SLOPE_UNITS:
        for measure in ("accuracy", "accuracy_class_adjusted"):
            for min_cell_trials in (1, SPARSE_CELL_MIN_TRIALS):
                slopes = _accuracy_slopes(
                    functions, measure=measure, predictor=predictor,
                    predictor_scale=scale, min_cell_trials=min_cell_trials,
                )
                if slopes.empty:
                    continue
                for (condition, stratum), group in slopes.groupby(
                    ["condition", "trial_class_name"], sort=True
                ):
                    rows.append({
                        "analysis": "conditional_accuracy", "term": term,
                        "measure": measure, "condition": condition,
                        "trial_class_name": stratum,
                        "min_cell_trials": min_cell_trials,
                        "test": "one_sample_vs_zero",
                        # ci_low/ci_high/coverage are the 95% interval; the
                        # bound is read off the 90% one and says so in its own
                        # coverage column rather than overwriting them.
                        **_one_sample_summary(group["slope"]),
                        **equivalence_bound(group["slope"]),
                    })

                # Fast minus Slow on the same slope, reported as a bound: the
                # condition contrast on this figure is a null the write-up
                # argues from, and "n.s." would not say how large a difference
                # the test could have missed.
                wide = slopes.pivot_table(
                    index=["subject", "trial_class_name"], columns="condition",
                    values="slope",
                )
                if not {"fast", "slow"}.issubset(wide.columns):
                    continue
                for stratum, group in wide.groupby("trial_class_name", sort=True):
                    differences = (group["fast"] - group["slow"]).dropna()
                    rows.append({
                        "analysis": "conditional_accuracy",
                        "term": "accuracy_slope_fast_minus_slow",
                        "measure": measure, "condition": "fast_vs_slow",
                        "trial_class_name": stratum,
                        "min_cell_trials": min_cell_trials,
                        "slope_unit": term,
                        "test": "one_sample_vs_zero",
                        # The interval here *is* the 90% one the bound is read
                        # off, since this row exists to state the bound.
                        **_one_sample_summary(differences, coverage=0.90),
                        **equivalence_bound(differences),
                    })

    return pd.DataFrame(rows)


def continuous_evidence_effects(
    features: pd.DataFrame,
    *,
    predictors: Mapping[str, float] = DEFAULT_EVIDENCE_PREDICTORS,
) -> pd.DataFrame:
    """Regress DT and accuracy on continuous early evidence.

    Unlike the class analyses these use every task trial with a designed token
    sequence, including the unclassified random trials that the threshold rule
    discards. Decision time is regressed on evidence *strength* (distance from
    chance, which is what should speed a decision) and accuracy on *signed*
    evidence toward the correct target.

    Parameters
    ----------
    features
        Canonical trial-feature table. All eligible task trials are retained,
        irrespective of discrete trial-class assignment.
    predictors
        Mapping from each correct-target evidence column to its explicitly
        declared neutral point.  Subtracting that point gives signed evidence
        toward the correct target; its absolute value is evidence strength.

    Returns
    -------
    pandas.DataFrame
        Per-subject and per-condition linear fits of decision time on absolute
        evidence strength and logistic fits of correctness on signed evidence,
        including observation counts and convergence status.

    Notes
    -----
    Each predictor is fitted separately. These coefficients describe
    association; the function does not infer a causal effect of evidence.
    Decision time is fitted with OLS as
    ``dt_ms = intercept + slope * abs(evidence - centre)``. Correctness is
    fitted with a statsmodels logistic regression on signed
    ``evidence - centre``. A fit requires a full-rank design and more trials
    than coefficients; logistic fits additionally require both outcomes and
    clean convergence. Failed fits remain ``NaN`` and are not replaced. The
    ``converged`` field is true only when both models succeeded.
    """
    trials = task_trials(features)
    require_columns(trials, ["dt_ms", "isCorrect", *predictors])
    rows = []
    for subject, condition, selected in _subject_condition_groups(trials):
        if selected.empty:
            continue
        for predictor, centre in predictors.items():
            signed = pd.to_numeric(selected[predictor], errors="coerce").to_numpy(dtype=float)
            signed = signed - centre
            strength = np.abs(signed)
            dt = pd.to_numeric(selected["dt_ms"], errors="coerce").to_numpy(dtype=float)
            dt_valid = np.isfinite(strength) & np.isfinite(dt)
            dt_design = np.column_stack([
                np.ones(dt_valid.sum()),
                strength[dt_valid],
            ])
            dt_values = dt[dt_valid]
            dt_fit = (
                sm.OLS(dt_values, dt_design).fit()
                if len(dt_values) > dt_design.shape[1]
                and np.linalg.matrix_rank(dt_design) == dt_design.shape[1]
                else None
            )
            validate_boolean_values(
                selected["isCorrect"],
                field="isCorrect",
                optional=True,
            )
            correct = selected["isCorrect"].astype("boolean")
            accuracy_valid = correct.notna().to_numpy() & np.isfinite(signed)
            accuracy_design = np.column_stack([
                np.ones(accuracy_valid.sum()),
                signed[accuracy_valid],
            ])
            accuracy_values = (
                correct[accuracy_valid].astype(float).to_numpy()
            )
            accuracy_fit = None
            if (
                len(accuracy_values) > accuracy_design.shape[1]
                and len(np.unique(accuracy_values)) == 2
                and np.linalg.matrix_rank(accuracy_design)
                == accuracy_design.shape[1]
            ):
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        candidate = sm.Logit(
                            accuracy_values,
                            accuracy_design,
                        ).fit(disp=False)
                    invalid = any(
                        isinstance(item.message, INVALID_LOGIT_WARNINGS)
                        for item in caught
                    )
                    if (
                        not invalid
                        and candidate.mle_retvals.get("converged", False)
                    ):
                        accuracy_fit = candidate
                except (np.linalg.LinAlgError, PerfectSeparationError):
                    pass
            rows.append(
                {
                    "subject": subject,
                    "condition": condition,
                    "predictor": predictor,
                    # Three tokens admit four splits, so signed evidence has
                    # four levels and its absolute value two. A fit on two
                    # levels is a difference of means, not a trend; recording
                    # each fit's own count keeps that visible in the derivative
                    # itself. The DT fit uses `strength`, the accuracy fit
                    # `signed`, so the two counts differ and are kept apart.
                    "n_strength_levels": _distinct_levels(strength),
                    "n_signed_levels": _distinct_levels(signed),
                    "n_dt_trials": len(dt_values),
                    "dt_intercept_ms": (
                        float(dt_fit.params[0])
                        if dt_fit is not None
                        else float("nan")
                    ),
                    "dt_slope_ms_per_unit": (
                        float(dt_fit.params[1])
                        if dt_fit is not None
                        else float("nan")
                    ),
                    "n_accuracy_trials": len(accuracy_values),
                    "accuracy_log_odds_per_unit": (
                        float(accuracy_fit.params[1])
                        if accuracy_fit is not None
                        else float("nan")
                    ),
                    "converged": bool(
                        dt_fit is not None and accuracy_fit is not None
                    ),
                }
            )
    return pd.DataFrame(rows)


EARLY_EVIDENCE_TOKENS: Final[int] = 3


def early_evidence_order_profile(features: pd.DataFrame) -> pd.DataFrame:
    """Decision time and accuracy by early-evidence level, class, and the order
    of the first three tokens.

    :func:`early_evidence_profile` treats an evidence level as a single
    quantity, and for the *strength* of the evidence it is one. For the
    sequence it is not: a 2-1 lead can be reached as correct-correct-wrong,
    correct-wrong-correct or wrong-correct-correct, and the psychophysical
    kernel weights the first jump about twice the third. Two classes at the
    same level can therefore hold disjoint sets of orders and still be
    described as matched on early evidence.

    They do. This table exists so that claim, and the size of the order effect
    behind it, are registered rather than asserted.

    Parameters
    ----------
    features
        Canonical trial-feature table with token directions, the correct
        target, decision time, correctness and trial class.

    Returns
    -------
    pandas.DataFrame
        One row per subject, condition subset, early-evidence level, trial
        class and token order, with the trial count, mean decision time and
        accuracy. ``token_order`` is a string of ``C``/``W`` -- one character
        per jump, relative to the *correct* target, so it is comparable across
        trials whichever side was correct.

    Notes
    -----
    Order is read over :data:`EARLY_EVIDENCE_TOKENS` jumps, the same window
    ``sp_design_early`` summarizes, so the two describe exactly the same
    tokens. Unclassified trials are retained: they are the only class that
    spans every order at a level, which is what makes the order effect
    measurable inside a single class rather than confounded with it.
    """
    trials = task_trials(features)
    require_columns(
        trials,
        ["dt_ms", "isCorrect", "trial_class_name", "token_directions",
         "correct_side", EARLY_EVIDENCE_PREDICTOR],
    )
    validate_boolean_values(trials["isCorrect"], field="isCorrect", optional=True)
    orders = []
    # `astype(str)` because a direction string is all digits and survives a
    # TSV round trip as an integer; the parser then validates it as usual.
    for directions, correct in zip(
        trials["token_directions"].astype(str), trials["correct_side"]
    ):
        jumps = parse_token_directions(directions)[:EARLY_EVIDENCE_TOKENS]
        orders.append("".join("C" if jump == int(correct) else "W" for jump in jumps))
    trials = trials.assign(
        _dt=pd.to_numeric(trials["dt_ms"], errors="coerce"),
        _correct=trials["isCorrect"].astype("boolean").astype(float),
        _class=trials["trial_class_name"].astype(str),
        _level=pd.to_numeric(trials[EARLY_EVIDENCE_PREDICTOR], errors="coerce"),
        _order=orders,
    )
    trials = trials.loc[
        trials["_dt"].notna() & trials["_correct"].notna() & trials["_level"].notna()
    ]

    rows = []
    for subject, condition, selected in _subject_condition_groups(trials):
        if selected.empty:
            continue
        for (level, stratum, order), cell in selected.groupby(
            ["_level", "_class", "_order"], sort=True
        ):
            rows.append({
                "subject": subject,
                "condition": condition,
                "sp_design_early": float(level),
                "trial_class_name": stratum,
                "token_order": order,
                "first_token": "correct" if order.startswith("C") else "wrong",
                "n_trials": int(len(cell)),
                "mean_dt_ms": float(cell["_dt"].mean()),
                "accuracy": float(cell["_correct"].mean()),
            })
    return pd.DataFrame(rows)


def early_evidence_order_statistics(order_profile: pd.DataFrame) -> pd.DataFrame:
    """Test what token order does at a fixed evidence level.

    Parameters
    ----------
    order_profile
        Output of :func:`early_evidence_order_profile`.

    Returns
    -------
    pandas.DataFrame
        ``order_composition`` rows giving each class's share of each order at
        each level -- the descriptive fact that classes at the same level can
        hold disjoint orders -- and ``first_token_contrast`` rows testing, one
        class and level at a time, decision time and accuracy when the first
        jump favours the correct target against when it does not.

    Notes
    -----
    The contrast is on the *first* jump because that is where the
    psychophysical kernel puts the most weight, and it is computed **within a
    class at a fixed level**, so the lead after three tokens and the class
    label are both held constant and only the order varies. That is what makes
    it an estimate of the order effect rather than another class comparison.

    Cells below :data:`SPARSE_CELL_MIN_TRIALS` are dropped before contrasting,
    on the same grounds as elsewhere in this module.
    """
    rows: list[dict] = []
    pooled = order_profile.loc[order_profile["condition"] == POOLED_STRATUM]

    for (level, stratum, order), group in pooled.groupby(
        ["sp_design_early", "trial_class_name", "token_order"], sort=True
    ):
        total = pooled.loc[
            (pooled["sp_design_early"] == level)
            & (pooled["trial_class_name"] == stratum), "n_trials"
        ].sum()
        rows.append({
            "analysis": "early_evidence_order", "term": "order_composition",
            "sp_design_early": float(level), "trial_class_name": stratum,
            "token_order": order, "test": "descriptive",
            "n_trials": int(group["n_trials"].sum()),
            "estimate": float(group["n_trials"].sum() / total) if total else float("nan"),
            "n_subjects": int(group["subject"].nunique()),
        })

    testable = pooled.loc[pooled["n_trials"] >= SPARSE_CELL_MIN_TRIALS]
    for (level, stratum), group in testable.groupby(
        ["sp_design_early", "trial_class_name"], sort=True
    ):
        for measure in ("mean_dt_ms", "accuracy"):
            wide = group.pivot_table(
                index="subject", columns="first_token", values=measure, aggfunc="mean",
            )
            if not {"correct", "wrong"}.issubset(wide.columns):
                continue
            paired = wide[["correct", "wrong"]].dropna()
            if len(paired) < 2:
                continue
            rows.append({
                "analysis": "early_evidence_order",
                "term": "first_token_contrast", "measure": measure,
                "sp_design_early": float(level), "trial_class_name": stratum,
                "test": "paired_t_test",
                **paired_subject_statistics(paired["correct"], paired["wrong"]),
                **equivalence_bound(paired["correct"] - paired["wrong"]),
            })
    return pd.DataFrame(rows)


def early_evidence_profile(features: pd.DataFrame) -> pd.DataFrame:
    """Decision time and accuracy at each early-evidence level, by trial class.

    The per-subject fits in :func:`continuous_evidence_effects` summarize the
    early-evidence relation with one coefficient. On this design that summary
    is misleading in two ways this table exists to expose.

    First, early evidence is a **four-level factor, not a continuum**. Three
    tokens admit exactly four splits, so ``sp_design_early`` takes exactly four
    values -- 0.194, 0.387, 0.613, 0.806, the correct target trailing 0-3 and
    1-2 and leading 2-1 and 3-0 -- and a coefficient "per unit" is a contrast
    between two of them. Resolving the levels shows what the coefficient
    averages over.

    Second, and the reason the table is also keyed by trial class: **the same
    early evidence does not imply the same trial**. The classification rule
    reads the success-probability profile at jumps 5, 8 and 11 as well as 3, so
    trials that look identical after three tokens diverge afterwards. Splitting
    each level by class measures how much of the trial the early read leaves
    undetermined -- which is exactly the within-class heterogeneity that makes
    F10's class-adjusted residual an upper bound rather than an estimate.

    Parameters
    ----------
    features
        Canonical trial-feature table. Every eligible task trial contributes,
        including the unclassified random trials the threshold rule discards;
        they are a stratum here, not a discard.

    Returns
    -------
    pandas.DataFrame
        One row per subject, condition subset (``all``, ``fast``, ``slow``),
        early-evidence level and stratum, with the trial count, mean decision
        time and accuracy. ``trial_class_name`` is ``"all"`` for the pooled row
        and a class name -- including ``"unclassified"`` -- otherwise. Each row
        also carries the level's ``sum_log_lr_design_early``.

    Notes
    -----
    ``sum_log_lr_design_early`` is exactly ``logit(sp_design_early)``: the
    **exact posterior** log odds after three tokens, not the first-order
    SumLogLR of Cisek et al. (2009). The two are different scales that disagree
    in sign on this dataset (see the criterion-scale comparison), so the column
    is reported beside the probability rather than labelled "SumLogLR".

    Not every class reaches every level, and that is a property of the stimulus
    set rather than missing data: no random sequence leaves the correct target
    trailing after three tokens, so unclassified trials occupy only the two
    positive levels, and every trial at a negative level is misleading by
    construction. Cells that do not exist are absent rather than zero-filled.
    """
    trials = task_trials(features)
    require_columns(
        trials,
        ["dt_ms", "isCorrect", "trial_class_name", EARLY_EVIDENCE_PREDICTOR,
         "sum_log_lr_design_early"],
    )
    validate_boolean_values(trials["isCorrect"], field="isCorrect", optional=True)
    trials = trials.assign(
        _dt=pd.to_numeric(trials["dt_ms"], errors="coerce"),
        _correct=trials["isCorrect"].astype("boolean").astype(float),
        _class=trials["trial_class_name"].astype(str),
        _level=pd.to_numeric(trials[EARLY_EVIDENCE_PREDICTOR], errors="coerce"),
        _log_odds=pd.to_numeric(trials["sum_log_lr_design_early"], errors="coerce"),
    )
    trials = trials.loc[
        trials["_dt"].notna() & trials["_correct"].notna() & trials["_level"].notna()
    ]

    rows = []
    for subject, condition, selected in _subject_condition_groups(trials):
        if selected.empty:
            continue
        for level, cell in selected.groupby("_level", sort=True):
            for stratum in ACCURACY_STRATA:
                part = (
                    cell if stratum == POOLED_STRATUM
                    else cell.loc[cell["_class"] == stratum]
                )
                if part.empty:
                    continue
                rows.append(
                    {
                        "subject": subject,
                        "condition": condition,
                        "sp_design_early": float(level),
                        "sum_log_lr_design_early": float(part["_log_odds"].iloc[0]),
                        "trial_class_name": stratum,
                        "n_trials": int(len(part)),
                        "mean_dt_ms": float(part["_dt"].mean()),
                        "accuracy": float(part["_correct"].mean()),
                    }
                )
    return pd.DataFrame(rows)


_EARLY_EVIDENCE_MEASURES: Final[tuple[tuple[str, str], ...]] = (
    ("mean_dt_ms", "decision time"),
    ("accuracy", "accuracy"),
)


def _level_matrix(profile: pd.DataFrame, level: float, strata: Sequence[str],
                  measure: str) -> tuple[list[str], np.ndarray]:
    """Complete-subject ``(n_subjects, n_strata)`` matrix at one evidence level."""
    subset = profile.loc[
        (profile["condition"] == POOLED_STRATUM)
        & (profile["sp_design_early"] == level)
        & (profile["trial_class_name"].isin(strata))
    ]
    wide = subset.pivot_table(
        index="subject", columns="trial_class_name", values=measure
    )
    wide = wide.reindex(columns=[s for s in strata if s in wide.columns])
    wide = wide.dropna(axis=0, how="any")
    return list(wide.columns), wide.to_numpy(dtype=float)


def continuous_evidence_statistics(
    subject_fits: pd.DataFrame,
    profile: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Group tests on the early-evidence coefficients and the level profile.

    Parameters
    ----------
    subject_fits
        Output of :func:`continuous_evidence_effects`.
    profile
        Output of :func:`early_evidence_profile`. When omitted only the
        coefficient rows are produced.

    Returns
    -------
    pandas.DataFrame
        Rows keyed by ``term``. The coefficient rows
        (``dt_slope_ms_per_unit``, ``accuracy_log_odds_per_unit``) test each
        fit against zero and carry the Fast-minus-Slow contrast as a bound.
        The profile rows report each level-by-class cell
        (``level_mean_dt_ms``, ``level_accuracy``), the contrast between
        classes at a level where several coexist
        (``class_contrast_at_level``, ``class_omnibus_at_level``), and the
        contrast between levels *within* a class
        (``evidence_contrast_within_class``).

    Notes
    -----
    The coefficient rows carry ``n_predictor_levels``: the number of distinct
    predictor values the fit actually saw. It is 4 for signed evidence and 2
    for its absolute deviation from chance, so ``dt_slope_ms_per_unit`` is a
    two-point contrast and must not be read as a tested linear trend. The
    contrast itself, in milliseconds, is registered as
    ``evidence_contrast_within_class``, which is the quantity to quote.

    The class contrast at a level is the load-bearing test: it asks whether
    trials that are indistinguishable after three tokens behave alike, and it
    is a within-subject comparison, so it cannot be explained by which subjects
    contribute which classes. The Fast-minus-Slow rows are reported as bounds
    rather than p-values because the figure argues from those nulls.

    No multiplicity correction is applied across levels, classes or measures.
    """
    rows: list[dict] = []
    # Each coefficient's own predictor: the decision-time fit is on absolute
    # strength, the accuracy fit on signed evidence, so they see a different
    # number of levels. 2 for the first, 4 for the second -- printed on every
    # row so a coefficient is never read as a tested linear trend.
    terms = {
        "dt_slope_ms_per_unit": "n_strength_levels",
        "accuracy_log_odds_per_unit": "n_signed_levels",
    }

    def level_count(group: pd.DataFrame, column: str) -> float:
        return float(group[column].max()) if column in group else float("nan")

    for (predictor, condition), group in subject_fits.groupby(
        ["predictor", "condition"], sort=True
    ):
        for term, level_column in terms.items():
            rows.append({
                "analysis": "continuous_evidence",
                "predictor": predictor, "term": term, "condition": condition,
                "test": "one_sample_vs_zero",
                "n_predictor_levels": level_count(group, level_column),
                **one_sample_statistics(group[term]),
            })
    for predictor, group in subject_fits.groupby("predictor", sort=True):
        for term, level_column in terms.items():
            wide = group.pivot(index="subject", columns="condition", values=term)
            if "fast" not in wide.columns or "slow" not in wide.columns:
                continue
            differences = (wide["fast"] - wide["slow"]).dropna()
            rows.append({
                "analysis": "continuous_evidence",
                "predictor": predictor, "term": term, "condition": "fast_vs_slow",
                "test": "paired_t_test",
                "n_predictor_levels": level_count(group, level_column),
                **paired_subject_statistics(wide["fast"], wide["slow"]),
                # A bare p on this row would assert a null the figure argues
                # from without saying how large a difference it could miss.
                **equivalence_bound(differences),
            })

    if profile is None or profile.empty:
        return pd.DataFrame(rows)

    pooled = profile.loc[profile["condition"] == POOLED_STRATUM]
    levels = sorted(pooled["sp_design_early"].unique())
    # Descriptive rows keep every cell; contrasts do not. The 0-3 level holds
    # 12 trials in the whole dataset -- a subject who has one contributes an
    # accuracy of 0 or 1 -- so a contrast reaching it would silently restrict
    # itself to the seven subjects who happen to own one. Cells below
    # SPARSE_CELL_MIN_TRIALS are therefore dropped before any contrast, which
    # is also why `misleading` ends up with no within-class contrast: it
    # effectively occupies a single level.
    testable = pooled.loc[pooled["n_trials"] >= SPARSE_CELL_MIN_TRIALS]

    for (level, stratum), group in pooled.groupby(
        ["sp_design_early", "trial_class_name"], sort=True
    ):
        for measure, _ in _EARLY_EVIDENCE_MEASURES:
            rows.append({
                "analysis": "continuous_evidence",
                "term": f"level_{measure}", "measure": measure,
                "condition": POOLED_STRATUM, "sp_design_early": float(level),
                "trial_class_name": stratum, "test": "descriptive",
                "mean_n_trials": float(group["n_trials"].mean()),
                "estimate": float(group[measure].mean()),
                "n_subjects": int(group[measure].notna().sum()),
            })

    # Does the same early evidence imply the same trial? Every level holding
    # more than one class answers it; the +1-lead level, which holds 69% of
    # trials and three classes, answers it best.
    for level in levels:
        present = sorted(
            testable.loc[(testable["sp_design_early"] == level)
                         & (testable["trial_class_name"] != POOLED_STRATUM),
                         "trial_class_name"].unique()
        )
        if len(present) < 2:
            continue
        for measure, _ in _EARLY_EVIDENCE_MEASURES:
            names, matrix = _level_matrix(testable, level, present, measure)
            if matrix.shape[0] < 2 or matrix.shape[1] < 2:
                continue
            for first, second in combinations(range(len(names)), 2):
                differences = matrix[:, first] - matrix[:, second]
                rows.append({
                    "analysis": "continuous_evidence",
                    "term": "class_contrast_at_level", "measure": measure,
                    "condition": POOLED_STRATUM, "sp_design_early": float(level),
                    "trial_class_name": f"{names[first]}_vs_{names[second]}",
                    "test": "paired_t_test",
                    **paired_subject_statistics(matrix[:, first], matrix[:, second]),
                    # By the +3 lead the classes converge in accuracy; that
                    # null is part of the figure's claim, so it carries what
                    # it excludes rather than a bare p.
                    **equivalence_bound(differences),
                })
            if matrix.shape[1] >= 3:
                effect = repeated_measures_anova(matrix, (matrix.shape[1],))[0]
                rows.append({
                    "analysis": "continuous_evidence",
                    "term": "class_omnibus_at_level", "measure": measure,
                    "condition": POOLED_STRATUM, "sp_design_early": float(level),
                    "trial_class_name": "+".join(names),
                    "test": "repeated_measures_anova",
                    "n_subjects": effect["n_subjects"],
                    "df_effect": effect["df_effect"], "df_error": effect["df_error"],
                    "F": effect["F"], "p": effect["p"],
                    "partial_eta_squared": effect["partial_eta_squared"],
                    "estimate": float(matrix.mean(axis=0).max()
                                      - matrix.mean(axis=0).min()),
                })

    # And the converse: holding the class fixed, does the early read still
    # move behaviour? This is the honest form of `dt_slope_ms_per_unit` --
    # a contrast between two levels, in the measure's own units.
    for stratum, group in testable.groupby("trial_class_name", sort=True):
        stratum_levels = sorted(group["sp_design_early"].unique())
        if len(stratum_levels) < 2:
            continue
        low, high = stratum_levels[0], stratum_levels[-1]
        for measure, _ in _EARLY_EVIDENCE_MEASURES:
            wide = group.pivot_table(
                index="subject", columns="sp_design_early", values=measure
            )
            if not {low, high}.issubset(wide.columns):
                continue
            paired = wide[[high, low]].dropna()
            if len(paired) < 2:
                continue
            rows.append({
                "analysis": "continuous_evidence",
                "term": "evidence_contrast_within_class", "measure": measure,
                "condition": POOLED_STRATUM, "trial_class_name": stratum,
                "sp_design_early": float(high), "sp_design_early_reference": float(low),
                "test": "paired_t_test",
                **paired_subject_statistics(paired[high], paired[low]),
                # Within ambiguous the early read moves neither measure. The
                # figure argues from that, so the row states the bound.
                **equivalence_bound(paired[high] - paired[low]),
            })
    return pd.DataFrame(rows)


CRITERION_SCALES: Final[dict[str, str]] = {
    "first_order": FIRST_ORDER_CRITERION_RESPONSE,
    "exact_log_odds": "logged_spd_log_odds",
}
CRITERION_TIME_BIN_S: Final[float] = 0.2
# Two bin rules, because drawing and testing have different needs.
#
# Display keeps any bin held by at least half the cohort. Beyond ~3 s the
# bins thin to single-digit trial counts -- the 3.3 s bin is 16 trials across
# 7 subjects -- so a per-subject mean there is noise, not a criterion.
DISPLAY_BIN_SUBJECT_FRACTION: Final[float] = 0.5
# The omnibus is a repeated-measures ANOVA and so is listwise: a subject
# missing any tested bin drops out entirely. At the display rule that costs
# 22 of 32 subjects and keeps only those with unusually wide decision-time
# distributions, which is a worse bias than the narrower window. Requiring
# near-complete bins instead tests 0.5-2.1 s, holding 88% of the trials, on
# essentially the whole cohort.
TEST_BIN_SUBJECT_FRACTION: Final[float] = 0.9


def _matched_criterion_trials(features: pd.DataFrame) -> pd.DataFrame:
    """The trials on which both evidence scales are defined.

    A 14-row token log cannot be aligned to the token present at commitment,
    so the criterion is undefined there rather than merely noisy; and a
    commitment before the first token has no sensory evidence to score. Both
    scales are therefore fitted on this single set, which is what makes the
    comparison between them attributable to the scale rather than to the
    sample.
    """
    require_columns(features, ["decision_token_index", "token_log_rows"])
    enriched = features.copy()
    enriched[FIRST_ORDER_CRITERION_RESPONSE] = first_order_chosen_sum_log_lr(enriched)
    complete_log = pd.to_numeric(
        enriched["token_log_rows"], errors="coerce"
    ).eq(15).fillna(False)
    post_first_token = pd.to_numeric(
        enriched["decision_token_index"], errors="coerce"
    ).gt(0).fillna(False)
    return enriched.loc[complete_log & post_first_token]


def criterion_scale_comparison(features: pd.DataFrame) -> pd.DataFrame:
    """Fit the evidence criterion against decision time on both scales.

    The criterion is how much evidence favouring the chosen target a subject
    had in hand at commitment. Whether it declines with time -- the classic
    behavioural signature of urgency gating -- depends on how "evidence" is
    scored, and the two available scorings disagree:

    ``first_order``
        Cisek et al. (2009, Eq. 22) SumLogLR: the chosen-target token lead
        times ``log(0.604736 / 0.395264)``. It is blind to how many tokens
        remain.
    ``exact_log_odds``
        The log odds of the design's exact success probability at commitment,
        which accounts for the tokens still to come.

    The two are not interchangeable, because the trial has a fixed 15-token
    horizon. A lead of 3 is the same SumLogLR at token 5 and at token 13, but
    the true probability of winning rises from 0.83 to 1.00 as the chance to
    overturn it disappears. A subject holding one *constant* posterior
    threshold must therefore accept a progressively smaller token lead, which
    appears as a declining first-order criterion even though the policy never
    changed. Fitting both on the same trials is what separates a policy change
    from this scale artifact.

    Parameters
    ----------
    features
        Canonical trial-feature table.

    Returns
    -------
    pandas.DataFrame
        One row per subject, condition subset and scale, with the intercept
        and slope of a trial-level OLS against decision time in seconds.

    Notes
    -----
    Both scales are log-odds quantities, so their slopes are directly
    comparable; the raw success probability is deliberately not fitted here,
    since it is bounded at 1 and compresses exactly where late commitments
    live.
    """
    trials = _matched_criterion_trials(features)
    frames = []
    for scale, response in CRITERION_SCALES.items():
        fits = criterion_decline(
            trials,
            predictor="dt_ms",
            response=response,
            require_design_time_alignment=True,
        )
        frames.append(fits.assign(scale=scale))
    return pd.concat(frames, ignore_index=True)


def criterion_time_profile(features: pd.DataFrame) -> pd.DataFrame:
    """Bin the criterion by decision time, per subject, on both scales.

    Parameters
    ----------
    features
        Canonical trial-feature table.

    Returns
    -------
    pandas.DataFrame
        One row per subject, scale and time bin, with the mean criterion and
        the trial count. Bins are ``CRITERION_TIME_BIN_S`` wide and labelled by
        their centre.

    Notes
    -----
    This exists for two reasons the fitted slope cannot serve: the figure draws
    it, and :func:`criterion_scale_comparison_statistics` tests whether the
    profile departs from the straight line the slope summarises. The report's
    claim that the early bins are non-linear was previously an eyeball
    judgement; binning here makes it testable.
    """
    trials = _matched_criterion_trials(features)
    seconds = pd.to_numeric(trials["dt_ms"], errors="coerce") / 1000.0
    trials = trials.assign(
        _bin=(np.floor(seconds / CRITERION_TIME_BIN_S) + 0.5) * CRITERION_TIME_BIN_S
    )
    rows = []
    for scale, response in CRITERION_SCALES.items():
        values = pd.to_numeric(trials[response], errors="coerce")
        frame = trials.assign(_value=values).dropna(subset=["_value", "_bin"])
        grouped = frame.groupby(["subject", "_bin"], sort=True)["_value"]
        for (subject, centre), series in grouped:
            rows.append({
                "subject": subject,
                "scale": scale,
                "time_bin_center_s": float(centre),
                "n_trials": int(series.size),
                "criterion": float(series.mean()),
            })
    return pd.DataFrame(rows)


def criterion_scale_comparison_statistics(
    fits: pd.DataFrame,
    profile: pd.DataFrame,
) -> pd.DataFrame:
    """Group tests for the two-scale criterion comparison.

    Parameters
    ----------
    fits
        Output of :func:`criterion_scale_comparison`.
    profile
        Output of :func:`criterion_time_profile`.

    Returns
    -------
    pandas.DataFrame
        Per scale: the pooled slope against zero, the Fast-minus-Slow contrast
        with a 90% interval and the smallest difference it excludes, and an
        omnibus test of departure from a straight line, plus the between-scale
        contrast of subject slopes.

    Notes
    -----
    The Fast/Slow row is reported as a bound, not as a p-value. A
    non-significant contrast here is uninformative unless the reader can see
    how large a difference the test could have detected -- for the first-order
    scale that bound is most of the pooled effect itself.

    Linearity is tested on deviations: each subject's binned profile has its
    own least-squares line in time removed, and the residuals are submitted to
    a one-factor repeated-measures ANOVA. Testing the means directly would
    reject on the slope, which is not in question. Only bins holding at least
    :data:`TEST_BIN_SUBJECT_FRACTION` of the cohort are tested; the figure draws
    a wider window under :data:`DISPLAY_BIN_SUBJECT_FRACTION`, so the tested
    range is narrower than the drawn one and the row reports ``n_bins``.
    """
    rows: list[dict] = []
    for scale in CRITERION_SCALES:
        scale_fits = fits.loc[fits["scale"] == scale]
        pooled = scale_fits.loc[scale_fits["condition"] == "all", "slope"]
        rows.append({
            "analysis": "criterion_scale", "scale": scale, "term": "slope",
            "condition": "all", "test": "one_sample_vs_zero",
            **_one_sample_summary(pooled),
        })

        wide = scale_fits.pivot_table(index="subject", columns="condition", values="slope")
        if {"fast", "slow"}.issubset(wide.columns):
            differences = (wide["fast"] - wide["slow"]).dropna()
            rows.append({
                "analysis": "criterion_scale", "scale": scale, "term": "fast_minus_slow",
                "condition": "fast_vs_slow", "test": "one_sample_vs_zero",
                # The interval here *is* the 90% one the bound is read off,
                # since this row exists to state the bound.
                **_one_sample_summary(differences, coverage=0.90),
                **equivalence_bound(differences),
            })

        # Departure from a straight line, on the binned profile.
        subset = profile.loc[profile["scale"] == scale]
        wide = subset.pivot_table(
            index="subject", columns="time_bin_center_s", values="criterion"
        )
        threshold = int(np.ceil(TEST_BIN_SUBJECT_FRACTION * wide.shape[0]))
        complete = wide.loc[:, wide.notna().sum() >= threshold]
        complete = complete.dropna(axis=0, how="any")
        if complete.shape[0] >= 2 and complete.shape[1] >= 3:
            centres = complete.columns.to_numpy(dtype=float)
            matrix = complete.to_numpy(dtype=float)
            design = np.column_stack([np.ones_like(centres), centres])
            coefficients = np.linalg.lstsq(design, matrix.T, rcond=None)[0]
            deviations = matrix - (design @ coefficients).T
            effect = repeated_measures_anova(deviations, (complete.shape[1],))[0]
            rows.append({
                "analysis": "criterion_scale", "scale": scale,
                "term": "nonlinearity_omnibus", "condition": "all",
                "test": "repeated_measures_anova",
                "n_subjects": effect["n_subjects"], "n_bins": complete.shape[1],
                "df_effect": effect["df_effect"], "df_error": effect["df_error"],
                "F": effect["F"], "p": effect["p"],
                "partial_eta_squared": effect["partial_eta_squared"],
            })

    # The comparison the merged figure exists to make.
    wide = fits.loc[fits["condition"] == "all"].pivot_table(
        index="subject", columns="scale", values="slope"
    )
    if set(CRITERION_SCALES).issubset(wide.columns):
        first, exact = wide["first_order"], wide["exact_log_odds"]
        paired = paired_subject_statistics(first, exact)
        rows.append({
            "analysis": "criterion_scale", "scale": "first_order_vs_exact_log_odds",
            "term": "slope_difference", "condition": "all", "test": "paired_t_test",
            "estimate": paired["mean_difference"], **paired,
        })
        both = pd.concat([first, exact], axis=1).dropna()
        if len(both) >= 3:
            r, p = stats.pearsonr(both["first_order"], both["exact_log_odds"])
            rows.append({
                "analysis": "criterion_scale", "scale": "first_order_vs_exact_log_odds",
                "term": "slope_correlation", "condition": "all", "test": "pearson",
                "n_subjects": int(len(both)), "estimate": float(r), "p": float(p),
            })
    return pd.DataFrame(rows)
