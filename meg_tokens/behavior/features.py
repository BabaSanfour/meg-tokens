"""Construct the canonical trial-level behavioral feature table.

Stage 1 tables preserve recorded and parsed run data. This module performs the
single deterministic transformation from those tables to analysis-ready,
MEG-joinable features. It does not select trials or run group statistics.
Unless stated otherwise, all times are milliseconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from meg_tokens.behavior.math.evidence import (
    evidence_after_tokens,
    log_posterior_odds,
    token_lead_profile,
)
from meg_tokens.behavior.math.probability import (
    probability_at_decision,
    success_probability_profile,
)
from meg_tokens.behavior.schema import (
    OUTCOME_NEVER_STARTED,
    parse_token_directions,
    validate_trial_features,
)
from meg_tokens.behavior.trials import OUTCOME_LABELS


def calculate_motor_baseline(rt_dfs: list[pd.DataFrame]) -> float:
    """Calculate one subject's motor baseline from chosen RT trials.

    Parameters
    ----------
    rt_dfs
        Canonical Stage 1 tables for the subject's RT runs. Each table must
        contain ``rawRT`` in milliseconds; missing responses may be null.

    Returns
    -------
    float
        Mean of each run's own mean finite ``rawRT``, in milliseconds.

    Raises
    ------
    KeyError
        If a non-empty table lacks ``rawRT``.
    ValueError
        If no valid RT response exists across the supplied tables.

    Notes
    -----
    Runs, not trials, contribute equally: each run is reduced to its own mean
    before those means are averaged, so a run with more usable responses does
    not pull the baseline toward itself.  This matches the preprint's
    estimator (``00_44_Behavior_Trial_Types.ipynb`` averages per-run means),
    and it differs from a trial-level pooled mean whenever the RT runs hold
    unequal numbers of responses.  Runs with no usable response contribute
    nothing.  The function does not estimate missing responses or apply
    outlier rejection.
    """
    run_means = []
    for table in rt_dfs:
        if table.empty:
            continue
        values = pd.to_numeric(table["rawRT"]).dropna()
        if not values.empty:
            run_means.append(float(values.mean()))
    if not run_means:
        raise ValueError("No valid RT trials are available for the motor baseline")
    return float(np.mean(run_means))


def calculate_motor_baseline_by_side(rt_dfs: list[pd.DataFrame]) -> dict[int, float]:
    """Calculate one subject's motor baseline separately for each response side.

    Parameters
    ----------
    rt_dfs
        Canonical Stage 1 tables for the subject's RT runs, containing
        ``rawRT`` in milliseconds and ``nChoiceMade`` as the response side.

    Returns
    -------
    dict
        Response side (1 = left, 2 = right) to that side's baseline in
        milliseconds, averaged over runs exactly as
        :func:`calculate_motor_baseline` does. A side with no usable response
        in any run is absent from the mapping; callers fall back to the
        pooled baseline for it.

    Raises
    ------
    ValueError
        If no valid RT response exists across the supplied tables.

    Notes
    -----
    The two hands are not equally fast. Across this cohort the RT runs give
    left 442.1 ms and right 456.0 ms, a 13.9 ms difference (t(31) = -3.29,
    p = .003) in a task with no deliberation in it -- so it is motor, and a
    single pooled baseline leaves it inside ``dt``. Subtracting the matching
    hand's baseline removes it: the left-minus-right asymmetry in decision
    time falls from -22.8 ms (p = .023) to -8.9 ms (p = .39) while the
    Fast/Slow effect is unchanged at +127.1 ms (was +127.4).

    This departs from the preprint, which pools both hands; recorded in
    ``docs/behavior.md`` under "Deliberate deviations from the preprint's
    analysis".
    """
    per_side: dict[int, list[float]] = {}
    for table in rt_dfs:
        if table.empty:
            continue
        sides = pd.to_numeric(table["nChoiceMade"], errors="coerce")
        values = pd.to_numeric(table["rawRT"], errors="coerce")
        for side in (1, 2):
            usable = values.loc[(sides == side) & values.notna()]
            if not usable.empty:
                per_side.setdefault(side, []).append(float(usable.mean()))
    if not per_side:
        raise ValueError("No valid RT trials are available for the motor baseline")
    return {side: float(np.mean(means)) for side, means in per_side.items()}


def calculate_decision_times(
    table: pd.DataFrame,
    motor_baseline: float,
) -> pd.Series:
    """Calculate decision time for every trial with a recorded response.

    Parameters
    ----------
    table
        Canonical Stage 1 run table containing ``rawRT`` in milliseconds.
    motor_baseline
        Subject-level mean RT latency in milliseconds.

    Returns
    -------
    pandas.Series
        ``rawRT - motor_baseline`` for rows with non-null ``rawRT``. The
        original row index is preserved; no entry is returned for a null
        response.

    Notes
    -----
    Negative values are retained as anticipations and flagged later in the
    trial-feature table rather than silently removed.
    """
    raw_rt = pd.to_numeric(table["rawRT"])
    return raw_rt.loc[raw_rt.notna()] - motor_baseline


def logged_spd_at_decision(
    trial: pd.Series,
    motor_baseline: float,
) -> tuple[float, int | None]:
    """Select logged success probability at a trial's decision time.

    Parameters
    ----------
    trial
        Canonical Stage 1 trial containing parallel ``nProb`` and ``tTime``
        sequences plus ``tEnterTarget``. Logged probabilities are already
        referenced to the chosen target by the acquisition file.
    motor_baseline
        Subject-level motor latency subtracted from ``tEnterTarget`` so the
        commitment time uses the token-log time frame.

    Returns
    -------
    probability
        Last logged probability available at or before commitment, or 0.5 if
        commitment precedes the first token.
    index
        Zero-based index of that token-log row, or ``None`` for the pre-token
        state.

    Raises
    ------
    ValueError
        If the probability and time sequences violate the strict alignment,
        ordering, finiteness, or probability-range contract.
    """
    return probability_at_decision(
        trial["nProb"],
        trial["tTime"],
        decision_time=float(trial["tEnterTarget"]) - float(motor_baseline),
    )


def _design_evidence(
    token_directions: object,
    *,
    target: int,
    n_tokens_at_decision: int,
) -> dict[str, object]:
    """Derive correct-target evidence from one designed token sequence.

    Parameters
    ----------
    token_directions
        Canonical compact direction string with one ``1`` or ``2`` per token.
    target
        Reference target, normally ``nCorrectChoice``.
    n_tokens_at_decision
        Number of designed tokens observed by commitment. Zero selects the
        pre-token priors of 0.5 probability and zero token lead.

    Returns
    -------
    dict
        Early evidence after jump three and evidence at commitment on success-
        probability, log-odds, and token-lead scales. ``sum_log_lr_saturated``
        reports finite log-odds clipping.

    Notes
    -----
    An invalid target or a design shorter than three tokens yields explicitly
    missing features. Structural parsing errors are not guessed or suppressed.
    """
    blank = {
        "sp_design_early": float("nan"),
        "sum_log_lr_design_early": float("nan"),
        "token_lead_at_decision": pd.NA,
        "sum_log_lr_at_decision": float("nan"),
        "sum_log_lr_saturated": pd.NA,
    }
    if target not in (1, 2):
        return blank
    directions = parse_token_directions(token_directions)
    if len(directions) < 3:
        return blank
    profile = success_probability_profile(directions, target=target)
    lead = token_lead_profile(directions, target=target)
    early_odds, _ = log_posterior_odds(profile[2])
    probability = evidence_after_tokens(
        profile,
        n_tokens_at_decision,
        prior=0.5,
    )
    odds, saturated = log_posterior_odds(probability)
    return {
        "sp_design_early": float(profile[2]),
        "sum_log_lr_design_early": early_odds,
        "token_lead_at_decision": int(
            evidence_after_tokens(lead, n_tokens_at_decision, prior=0.0)
        ),
        "sum_log_lr_at_decision": odds,
        "sum_log_lr_saturated": bool(saturated),
    }


def build_trial_features(
    by_subject: dict[str, list[pd.DataFrame]],
    motor_baselines: dict[str, float],
    motor_baselines_by_side: dict[str, dict[int, float]] | None = None,
) -> pd.DataFrame:
    """Build one canonical feature row for every staged behavioral trial.

    Parameters
    ----------
    by_subject
        Mapping from canonical subject identifier to that subject's Stage 1
        run tables. Tables must already carry run metadata and pass the Stage 1
        schema.
    motor_baselines
        Pooled motor latency in milliseconds for every subject in
        ``by_subject``. Used as the fallback when a subject has no usable RT
        response on the hand a task trial was answered with.
    motor_baselines_by_side
        Optional per-hand motor latency, from
        :func:`calculate_motor_baseline_by_side`. When supplied, each trial is
        corrected with the baseline for the hand that answered it, which
        removes the motor speed difference between hands from ``dt_ms`` and
        from the commitment time that ``logged_spd`` is read at. Omitted, the
        pooled baseline is used for every trial, reproducing the previous
        behaviour.

    Returns
    -------
    pandas.DataFrame
        Validated trial-feature table. Its MEG join key is ``subject``,
        ``condition``, ``run``, and one-based ``run_trial_index``.

    Raises
    ------
    KeyError
        If a subject lacks a motor baseline or a required Stage 1 field.
    ValueError
        If a probability lookup, designed-evidence calculation, or final
        trial-feature schema validation fails.

    Notes
    -----
    Every input row is preserved, including never-started and no-response
    trials. ``decision_token_index`` is one-based in the output; zero denotes
    commitment before the first token. Decision time and logged SPD are only
    calculated for chosen Fast/Slow trials. Eligibility flags express later
    selection policy without deleting observations.
    """
    rows = []
    class_names = {
        0: "unclassified",
        1: "easy",
        2: "ambiguous",
        3: "misleading",
    }
    for subject, tables in sorted(by_subject.items()):
        pooled_baseline = motor_baselines[subject]
        side_baselines = (motor_baselines_by_side or {}).get(subject, {})
        for table in tables:
            condition = str(table["condition"].iloc[0])
            run = int(table["run"].iloc[0])
            started_index = 0
            for run_trial_index, (_, trial) in enumerate(
                table.iterrows(),
                start=1,
            ):
                is_started = int(trial["nOutcome"]) != OUTCOME_NEVER_STARTED
                if is_started:
                    started_index += 1
                raw_rt = pd.to_numeric(
                    pd.Series([trial["rawRT"]]),
                    errors="coerce",
                ).iloc[0]
                has_choice = bool(pd.notna(raw_rt))
                is_task = condition.lower() in {"fast", "slow"}
                # The hand that answered this trial, so its own motor latency
                # is removed rather than the average of both hands'.
                trial_side = int(trial["nChoiceMade"])
                motor_baseline = side_baselines.get(trial_side, pooled_baseline)
                dt = (
                    float(raw_rt) - motor_baseline
                    if is_task and has_choice
                    else float("nan")
                )

                spd = float("nan")
                decision_token_index = pd.NA
                if is_task and has_choice:
                    spd, zero_based_index = logged_spd_at_decision(
                        trial,
                        motor_baseline,
                    )
                    decision_token_index = (
                        0 if zero_based_index is None else zero_based_index + 1
                    )
                log_rows = int(trial["token_log_rows"])
                trial_class = int(trial["sTrialClass"])
                choice_side = int(trial["nChoiceMade"])
                correct_side = int(trial["nCorrectChoice"])
                tokens_seen = (
                    int(decision_token_index)
                    if pd.notna(decision_token_index)
                    else 0
                )
                outcome = int(trial["nOutcome"])
                rows.append({
                    "trial_id": (
                        f"sub-{subject}_task-tokens_run-{run}_desc-"
                        f"{condition.lower()}_trial-{run_trial_index:03d}"
                    ),
                    "subject": subject,
                    "condition": condition,
                    "run": run,
                    "block_index": run,
                    "run_trial_index": run_trial_index,
                    "started_trial_index": (
                        started_index if is_started else pd.NA
                    ),
                    "nTrialIndex": int(trial["nTrialIndex"]),
                    "initial_time_ms": int(trial["nInitialTime"]),
                    "source_file": trial["source_file"],
                    "trial_class": trial_class,
                    "trial_class_name": class_names.get(
                        trial_class,
                        "unclassified",
                    ),
                    "sTrialClassRaw": trial["sTrialClassRaw"],
                    "trial_class_source": trial["trial_class_source"],
                    "trial_class_rule": trial["trial_class_rule"],
                    "nChoiceMade": choice_side,
                    "nCorrectChoice": correct_side,
                    "choice_side": choice_side if has_choice else pd.NA,
                    "correct_side": (
                        correct_side if correct_side in (1, 2) else pd.NA
                    ),
                    "isCorrect": trial["isCorrect"],
                    "nOutcome": outcome,
                    "outcome_label": OUTCOME_LABELS.get(outcome, "completed"),
                    "motor_baseline_ms": motor_baseline,
                    "rawRT": raw_rt,
                    "dt_ms": dt,
                    "logged_spd": spd,
                    "logged_spd_log_odds": (
                        log_posterior_odds(spd)[0]
                        if pd.notna(spd)
                        else float("nan")
                    ),
                    "logged_spd_validated_15row": (
                        spd if log_rows == 15 else float("nan")
                    ),
                    "evidence_at_decision": (
                        spd - 0.5 if pd.notna(spd) else float("nan")
                    ),
                    "decision_token_index": decision_token_index,
                    "token_directions": trial["sTokenDirs"],
                    **_design_evidence(
                        trial["sTokenDirs"],
                        target=correct_side,
                        n_tokens_at_decision=tokens_seen if has_choice else 0,
                    ),
                    "token_log_rows": log_rows,
                    "token_log_short": trial["token_log_short"],
                    "is_started": is_started,
                    "has_choice": has_choice,
                    "is_no_response": is_started and not has_choice,
                    "dt_anticipation": bool(pd.notna(dt) and dt < 0),
                    "spd_available": bool(pd.notna(spd)),
                    "design_time_alignment_valid": bool(
                        is_task and has_choice and log_rows == 15
                    ),
                    "primary_analysis_eligible": bool(
                        is_task and is_started and has_choice
                    ),
                })
    features = pd.DataFrame(rows)
    validate_trial_features(features)
    return features
