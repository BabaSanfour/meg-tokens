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
from typing import Final

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    HessianInversionWarning,
    PerfectSeparationError,
    PerfectSeparationWarning,
)

from meg_tokens.behavior.math.inference import (
    one_sample_statistics,
    paired_subject_statistics,
)
from meg_tokens.behavior.schema import (
    parse_token_directions,
    validate_boolean_values,
)
from meg_tokens.behavior.trials import TASK_CONDITIONS, require_columns, task_trials


# Explicit output-resolution policies for the reverse-correlation and
# conditional-accuracy analyses. They are defaults, not inferred quantities.
DEFAULT_KERNEL_JUMPS: Final[int] = 8
DEFAULT_ACCURACY_BINS: Final[int] = 5
DEFAULT_EVIDENCE_PREDICTORS: Final[dict[str, float]] = {
    "sp_design_early": 0.5,
    "sum_log_lr_design_early": 0.0,
}

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


def criterion_decline(
    features: pd.DataFrame,
    *,
    predictor: str = "decision_token_index",
    response: str = "logged_spd",
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
        computed separately for logistic weights and model-free signed means.

    Notes
    -----
    Non-finite subject estimates are excluded by the shared inference helpers.
    Tests are performed independently at every jump without multiplicity
    correction. Duplicate subject/condition/jump rows cause the paired pivot
    to raise instead of being aggregated silently.
    """
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
                    **paired_subject_statistics(wide["fast"], wide["slow"]),
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

    Parameters
    ----------
    features
        Canonical trial-feature table with decision time and correctness.
    n_bins
        Positive requested number of equal-count decision-time bins within each
        subject and condition. Five is an explicit resolution default.
        Duplicate quantile edges may reduce the realized count.

    Returns
    -------
    pandas.DataFrame
        Accuracy, mean decision time, and trial count for every realized bin in
        the pooled, Fast, and Slow subsets of each subject.

    Notes
    -----
    Missing decision times and correctness values are excluded. Correctness
    values are validated rather than coerced. Bins are constructed separately
    within every subject and condition with :func:`pandas.qcut`; subsets that
    cannot form a bin are omitted. Bin numbers therefore denote within-subject
    quantile order, not common decision-time boundaries across subjects.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    trials = task_trials(features)
    require_columns(trials, ["dt_ms", "isCorrect", "subject", "condition"])
    rows = []
    for subject, condition, selected in _subject_condition_groups(trials):
        values = pd.to_numeric(selected["dt_ms"], errors="coerce")
        validate_boolean_values(
            selected["isCorrect"],
            field="isCorrect",
            optional=True,
        )
        correct = selected["isCorrect"].astype("boolean")
        usable = selected.loc[values.notna() & correct.notna()]
        if usable.empty:
            continue
        usable_dt = pd.to_numeric(usable["dt_ms"], errors="coerce")
        try:
            bins = pd.qcut(usable_dt, n_bins, labels=False, duplicates="drop")
        except ValueError:  # fewer distinct values than requested bins
            continue
        usable_correct = usable["isCorrect"].astype("boolean").astype(float)
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
    """Summarize conditional accuracy and its change across decision-time bins.

    Parameters
    ----------
    functions
        Output of :func:`conditional_accuracy_functions`.

    Returns
    -------
    pandas.DataFrame
        Across-subject accuracy summaries for each condition and bin, followed
        by one-sample tests of subject-level linear accuracy slopes against
        zero. The slope uses ordinal bin number, not milliseconds.

    Notes
    -----
    The per-bin rows use the shared one-sample summary, including its test of
    accuracy against zero; the slope rows test whether the across-bin linear
    trend differs from zero. Subject slopes require at least three realized
    bins and a full-rank intercept-plus-bin design. No multiplicity correction
    or Fast-versus-Slow slope contrast is applied here.
    """
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
        accuracy = ordered["accuracy"].to_numpy(dtype=float)
        fit = (
            sm.OLS(accuracy, design).fit()
            if len(accuracy) > design.shape[1]
            and np.linalg.matrix_rank(design) == design.shape[1]
            else None
        )
        slopes.append(
            {
                "subject": subject,
                "condition": condition,
                "slope": float(fit.params[1]) if fit is not None else float("nan"),
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


def continuous_evidence_statistics(subject_fits: pd.DataFrame) -> pd.DataFrame:
    """Run group tests on subject-level continuous-evidence coefficients.

    Parameters
    ----------
    subject_fits
        Output of :func:`continuous_evidence_effects`.

    Returns
    -------
    pandas.DataFrame
        One-sample tests against zero for decision-time and accuracy slopes in
        every condition, plus subject-paired Fast-minus-Slow comparisons.

    Notes
    -----
    Pairing is by subject and incomplete pairs are excluded by the shared
    inference helper. Tests are independent across predictors, coefficients,
    and conditions; no multiplicity correction is applied. Duplicate
    subject/condition rows cause the paired pivot to raise rather than being
    silently aggregated.
    """
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
                    **paired_subject_statistics(wide["fast"], wide["slow"]),
                }
            )
    return pd.DataFrame(rows)
