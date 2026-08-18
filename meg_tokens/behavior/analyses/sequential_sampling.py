"""Mechanistic sequential-sampling models for the token task.

Cisek's urgency-gating model is not a drift-diffusion model with a shrinking
bound. Its decision variable is the product of a *low-pass-filtered* estimate
of the momentary evidence and a growing urgency signal, compared with a
*constant* firing threshold (Thura et al. 2012, Eqs. 23, 25 and 26)::

    tau dw/dt = -w + dS/dt + noise        low-pass filter, tau = 100-250 ms
    x(t) = f[S(t)] * u(t)                 decision variable, u(t) = beta * t
    decide when |x(t)| >= T               T constant

Because the threshold is constant and urgency multiplies, ``|f[S(t)] * u(t)|
>= T`` is the same statement as ``|f[S(t)]| >= T / u(t)``: a leaky filter of
the evidence against a hyperbolically declining criterion. That identity is
what makes the model solvable as a first-passage problem, and it is the only
sense in which a declining bound belongs here — the process being bounded is a
filter with a 200 ms memory, not an integrator. The comparison model is the
bounded integrator of Thura et al. 2012, Eq. 27: the same evidence, integrated
without leak, against a fixed bound. Carland et al. 2019 (Box 1) places the
two at opposite corners of a plane spanned by the integration time constant
and the urgency slope.

Writing the linear urgency as ``u(t) = t + urgency_onset_s`` rather than
``1 + rate * t`` matters in practice: these data want the published
zero-intercept form, which the second parameterization can only approach by
sending both its parameters to infinity along a ridge.

Both models are driven by the trial's own token-by-token evidence trajectory.
This matters: with evidence held constant within a trial the two are
algebraically equivalent (Cisek et al. 2009, Eqs. 3-4), so a fit that reduces
the trial to one scalar cannot test urgency gating at all.

``pyddm`` is used rather than HSSM because HSSM's likelihoods — analytic or
network-approximated — assume a drift that is constant within a trial, and
neither the token-by-token evidence trajectory nor the leaky filter can be
expressed through them without training a new likelihood network. ``pyddm``
solves the Fokker-Planck equation directly for an arbitrary ``drift(x, t,
conditions)`` and ``bound(t)``, which is what this model needs. The cost is
that these are maximum-likelihood point fits, so the group layer is a normal
population model rather than a posterior, and the criteria are AIC and BIC
rather than WAIC or LOO.
The public :func:`fit_sequential_sampling_models` function retains the original
two-model API (``ddm`` and ``urgency``) for backwards compatibility.  The
complete Thura et al. (2012) comparison is exposed by
:func:`fit_mechanistic_model_set`, which adds a collapsing-bound integrator and
an additive-urgency sensitivity model.  All four models consume the same
trial-level token path, eligibility mask, decision-time origin, likelihood and
motor correction; only the state equation or boundary differs.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from itertools import combinations
import json
from typing import Final, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, stats

from meg_tokens.behavior.analyses.evidence import _subject_condition_groups
from meg_tokens.behavior.math.inference import (
    one_sample_statistics,
    paired_subject_statistics,
)
from meg_tokens.behavior.schema import (
    parse_token_directions,
    validate_boolean_values,
)
from meg_tokens.behavior.trials import require_columns, task_trials


# Task timing: the tokens jump one by one every 200 ms, and decision time is
# measured from the first jump, so the evidence step at jump k lands at 0.2 k.
TOKEN_INTERVAL_S: Final[float] = 0.2

# Low-pass filter time constant. Cisek et al. 2009 (model 6) fitted this task
# with 200 ms; the same filter is 100 ms in Thura et al. 2012 and 250 ms in
# Carland et al. 2015. It is held at the published value for this task rather
# than fitted, so that the urgency model cannot win by nesting the integrator:
# a free time constant recovers the bounded integrator as its long-tau limit.
FILTER_TAU_S: Final[float] = 0.2
MIXTURE_COEF: Final[float] = 0.02

# Keep the historical default stable for existing pooled derivatives and report
# builders.  The complete prespecified comparison is ``MECHANISTIC_MODELS``.
SSM_MODELS: Final[tuple[str, ...]] = ("ddm", "urgency")
MECHANISTIC_MODELS: Final[tuple[str, ...]] = (
    "ddm",
    "urgency",
    "collapsing",
    "additive_urgency",
)
MODEL_PARAMETERS: Final[dict[str, tuple[str, ...]]] = {
    "ddm": ("drift_scale", "bound", "nondecision_s"),
    "urgency": (
        "drift_scale",
        "urgency_scale",
        "urgency_onset_s",
        "nondecision_s",
    ),
    # A hyperbolically collapsing bound is the simplest adaptive-bound
    # alternative that remains identifiable with the available trial counts.
    "collapsing": ("drift_scale", "bound", "collapse_rate", "nondecision_s"),
    # Additive urgency is deliberately a sensitivity model: a time-dependent
    # drive is added to the leaky evidence estimate before a fixed threshold.
    "additive_urgency": (
        "drift_scale",
        "additive_scale",
        "bound",
        "nondecision_s",
    ),
}
PARAMETER_RANGES: Final[dict[str, tuple[float, float]]] = {
    "drift_scale": (0.0, 5.0),
    "bound": (0.1, 5.0),
    "urgency_scale": (0.01, 2.0),
    "urgency_onset_s": (0.0, 2.0),
    "collapse_rate": (0.0, 5.0),
    "additive_scale": (0.0, 2.0),
    # Wide enough for the integrator to buy all the dead time it wants. Decision
    # times already have each subject's motor baseline subtracted, so 1 s is far
    # beyond anything plausible -- but the integrator fits this task by absorbing
    # the urgency it cannot express into non-decision time, and capping that
    # would hand the comparison to the urgency model by construction.
    "nondecision_s": (0.0, 1.0),
}

# Explicit fitting policies. The solver step is both the diffusion grid and the
# bin width of the likelihood, and a four-parameter accumulator is not
# identifiable from a handful of trials.
SOLVER_STEP_S: Final[float] = 0.01
EVIDENCE_AFTER_LAST: str = "hold"

# Configuration names are stable derivative/provenance identifiers.  The
# cluster workflow runs one subject × one configuration per array task so the
# full robustness grid is restartable rather than a monolithic 2,304-fit job.
ROBUSTNESS_CONFIGURATIONS: Final[tuple[str, ...]] = (
    "baseline",
    "tau_100ms",
    "tau_300ms",
    "solver_20ms",
    "post_horizon_evidence_zero",
    "expanded_bounds",
)

# Plotting resolution for the fitted time courses. Coarser than the solver step
# because a curve does not need the solver's grid, and the table is written per
# subject, condition, model and trial class.
TIME_COURSE_STEP_S: Final[float] = 0.02
MINIMUM_FIT_TRIALS: Final[int] = 50

# Differential evolution starts from a random population, so the search is
# seeded to make a derivative reproducible from the same trial table.
FIT_SEED: Final[int] = 0

# Central-difference step for the observed-information Hessian. Non-decision
# time shifts the predicted density in whole solver steps, so the step must
# span several of them or the estimated curvature is exactly zero.
HESSIAN_STEP: Final[float] = 0.05


def _lead_path(directions: str, correct_target: int) -> tuple[int, ...]:
    """Build the running token lead of the correct target, jump by jump.

    Parameters
    ----------
    directions
        Compact token-direction string, one ``1`` or ``2`` per jump.
    correct_target
        Identifier of the target that was ultimately correct.

    Returns
    -------
    tuple of int
        Lead after each jump, positive when the correct target is ahead.

    Notes
    -----
    The token lead is the sensory-evidence signal of the monkey work: Cisek et
    al. 2009 define ``SumLogLR`` over token movements and note that it "simply
    amounts to counting the number of tokens which move in each direction",
    and Thura and Cisek 2014 state it is proportional to the difference in
    tokens between the targets. Referencing it to the correct rather than the
    chosen target keeps the regressor independent of the response it predicts.
    """
    lead = 0
    path = []
    for target in parse_token_directions(directions):
        lead += 1 if target == correct_target else -1
        path.append(lead)
    return tuple(path)


def empirical_lead_paths(
    features: pd.DataFrame,
    *,
    n_paths: int = 8,
) -> tuple[tuple[int, ...], ...]:
    """Return frequent empirical token paths in the correct-target frame.

    Recovery simulations need realistic evidence histories, but the raw token
    strings use the experiment's target labels.  A raw string therefore cannot
    be passed to :func:`_synthetic_frame` without also carrying its actual
    ``nCorrectChoice`` value: forcing every path to target 1 silently mirrors
    trials whose correct target was 2.  This helper canonicalizes each trial
    first, then selects frequent *lead-path tuples* without using the choice
    outcome.
    """
    if n_paths < 1:
        raise ValueError("n_paths must be positive")
    trials = task_trials(features)
    require_columns(trials, ["token_directions", "nCorrectChoice"])
    counts: Counter[tuple[int, ...]] = Counter()
    for directions, target in zip(
        trials["token_directions"], trials["nCorrectChoice"], strict=False
    ):
        try:
            target_value = int(target)
        except (TypeError, ValueError):
            continue
        if target_value not in {1, 2}:
            continue
        path = _lead_path(str(directions), target_value)
        if path:
            counts[path] += 1
    return tuple(path for path, _ in counts.most_common(n_paths))


def _evidence(t: float, lead_path: tuple[int, ...]) -> float:
    """Evaluate the token lead at one moment of one trial."""
    jump = int(t / TOKEN_INTERVAL_S)
    if jump <= 0:
        return 0.0
    if jump >= len(lead_path) and EVIDENCE_AFTER_LAST == "zero":
        return 0.0
    return float(lead_path[min(jump, len(lead_path)) - 1])


def _additive_drive(
    t: float,
    lead_path: tuple[int, ...],
    additive_scale: float,
) -> float:
    """Return the label-symmetric additive urgency sensitivity drive.

    In a two-race model a common urgency input is identical in both channels
    and cancels from their difference.  A signed one-dimensional implementation
    can only represent the closest alternative: a time-growing drive toward
    the currently favoured evidence direction.  It is not the primary Thura
    model and is retained solely as a transparent sensitivity analysis.
    """
    evidence = _evidence(t, lead_path)
    return float(additive_scale) * max(float(t), 0.0) * float(np.sign(evidence))


def _decision_frame(trials: pd.DataFrame) -> pd.DataFrame:
    """Reduce task trials to decision times, choices, and evidence paths.

    Parameters
    ----------
    trials
        Eligible task trials for one subject and condition.

    Returns
    -------
    pandas.DataFrame
        Columns ``rt`` (seconds), ``correct`` (0 or 1), ``lead_path``,
        ``trial_class`` and ``trial_id``, holding only rows the accumulator can
        score.

    Notes
    -----
    Anticipations (``dt_ms`` at or below zero) are dropped: a first-passage
    density has no support before the process starts. They are retained and
    flagged everywhere else in the pipeline. The evidence path is a tuple so
    that ``pyddm`` treats trials sharing a token sequence as one condition and
    solves the model once for each distinct sequence.
    """
    validate_boolean_values(trials["isCorrect"], field="isCorrect", optional=True)
    frame = pd.DataFrame(
        {
            "rt": pd.to_numeric(trials["dt_ms"], errors="coerce") / 1000.0,
            "correct": trials["isCorrect"].astype("boolean"),
            "lead_path": [
                _lead_path(directions, int(target))
                for directions, target in zip(
                    trials["token_directions"], trials["nCorrectChoice"]
                )
            ],
            "trial_class": trials["trial_class_name"].astype(str).to_numpy(),
            "trial_id": trials["trial_id"].astype(str).to_numpy(),
        }
    )
    frame = frame.loc[
        np.isfinite(frame["rt"]) & (frame["rt"] > 0.0) & frame["correct"].notna()
    ]
    return frame.astype({"correct": int}).reset_index(drop=True)


def _build_model(model: str, t_dur: float, values: dict[str, float] | None = None):
    """Construct one accumulator, either free for fitting or held at values.

    Parameters
    ----------
    model
        ``"ddm"`` for the bounded integrator, ``"urgency"`` for the primary
        Thura et al. model, ``"collapsing"`` for a collapsing-bound integrator,
        or ``"additive_urgency"`` for the additive-urgency sensitivity model.
    t_dur
        Integration horizon in seconds; it must exceed every decision time.
    values
        Parameter values to hold fixed. When omitted every parameter is left
        free within :data:`PARAMETER_RANGES`.

    Returns
    -------
    pyddm.Model
        Accumulator with unit noise, so that drift and bound are expressed in
        noise units.

    Notes
    -----
    The urgency model's drift is the low-pass filter written as a
    state-dependent drift, ``(drift_scale * evidence - x) / FILTER_TAU_S``,
    and its criterion is ``urgency_scale / (t + urgency_onset_s)``, the
    threshold divided by a linear urgency signal ``u(t) = t +
    urgency_onset_s``. Only the threshold-to-slope ratio and the
    intercept-to-slope ratio are identifiable, which is what those two
    parameters are: ``urgency_scale`` in criterion-seconds and
    ``urgency_onset_s`` in seconds. ``urgency_onset_s = 0`` is the published
    form ``u(t) = beta * t`` exactly, and a large onset is a criterion that
    barely falls, so both ends of the urgency question sit inside the range.
    The denominator is floored at one token interval because nothing can be
    decided before the first token moves, and an unfloored ``1/t`` criterion
    would demand an unbounded diffusion grid. PyDDM's uniform lapse mixture is
    set explicitly to :data:`MIXTURE_COEF` (0.02) for every model, rather than
    relying on a library default; it is identical across models and therefore
    cannot by itself favour one family.
    """
    import pyddm

    if model not in MODEL_PARAMETERS:
        raise ValueError(f"unknown sequential-sampling model: {model!r}")

    parameters = {
        name: PARAMETER_RANGES[name] if values is None else float(values[name])
        for name in MODEL_PARAMETERS[model]
    }

    def integrator_drift(t, lead_path, drift_scale):
        return drift_scale * _evidence(t, lead_path)

    def filter_drift(t, x, lead_path, drift_scale):
        return (drift_scale * _evidence(t, lead_path) - x) / FILTER_TAU_S

    def fixed_bound(bound):
        return bound

    def urgency_bound(t, urgency_scale, urgency_onset_s):
        return urgency_scale / max(t + urgency_onset_s, TOKEN_INTERVAL_S)

    def collapsing_bound(t, bound, collapse_rate):
        return bound / (1.0 + collapse_rate * max(float(t), 0.0))

    def additive_drift(t, x, lead_path, drift_scale, additive_scale):
        # A common additive drive cancels in the signed difference of two race
        # channels.  The only defensible scalar sensitivity is therefore a
        # drive in the *current evidence-preference* direction, not the
        # post-hoc correct-target direction.  ``_additive_drive`` is odd under
        # target relabelling, which is tested below and recorded as a caveat.
        return (
            (drift_scale * _evidence(t, lead_path) - x) / FILTER_TAU_S
            + _additive_drive(t, lead_path, additive_scale)
        )

    if model == "ddm":
        drift, bound = integrator_drift, fixed_bound
    elif model == "urgency":
        drift, bound = filter_drift, urgency_bound
    elif model == "collapsing":
        drift, bound = integrator_drift, collapsing_bound
    else:
        drift, bound = additive_drift, fixed_bound
    return pyddm.gddm(
        drift=drift,
        noise=1.0,
        bound=bound,
        nondecision=lambda nondecision_s: nondecision_s,
        parameters=parameters,
        conditions=["lead_path"],
        T_dur=t_dur,
        dt=SOLVER_STEP_S,
        dx=SOLVER_STEP_S,
        mixture_coef=MIXTURE_COEF,
    )


def _standard_errors(
    model: str,
    sample,
    estimates: dict[str, float],
    t_dur: float,
) -> dict[str, float]:
    """Derive parameter standard errors from the observed information matrix.

    Parameters
    ----------
    model
        Fitted model family.
    sample
        ``pyddm`` sample the model was fitted to.
    estimates
        Maximum-likelihood parameter values.
    t_dur
        Integration horizon used for the fit.

    Returns
    -------
    dict
        One standard error per parameter, ``NaN`` where the information matrix
        is singular or implies a non-positive variance.

    Notes
    -----
    The Hessian of the negative log-likelihood is taken by central differences
    because ``pyddm`` fits by differential evolution and returns no derivative.
    A non-positive variance means the estimate sits against a range boundary or
    on a flat likelihood; it is reported as missing rather than replaced.
    """
    import pyddm

    names = MODEL_PARAMETERS[model]
    centre = np.array([estimates[name] for name in names])

    def loss(values: np.ndarray) -> float:
        return float(
            pyddm.get_model_loss(
                _build_model(model, t_dur, dict(zip(names, values))),
                sample,
                lossfunction=pyddm.LossLikelihood,
            )
        )

    hessian = np.empty((len(names), len(names)))
    for i in range(len(names)):
        for j in range(i, len(names)):
            shift_i = np.zeros(len(names))
            shift_j = np.zeros(len(names))
            shift_i[i] = HESSIAN_STEP
            shift_j[j] = HESSIAN_STEP
            second = (
                loss(centre + shift_i + shift_j)
                - loss(centre + shift_i - shift_j)
                - loss(centre - shift_i + shift_j)
                + loss(centre - shift_i - shift_j)
            ) / (4.0 * HESSIAN_STEP**2)
            hessian[i, j] = hessian[j, i] = second
    try:
        variances = np.diag(np.linalg.inv(hessian))
    except np.linalg.LinAlgError:
        return {name: float("nan") for name in names}
    return {
        name: float(np.sqrt(variance)) if variance > 0 else float("nan")
        for name, variance in zip(names, variances)
    }


def _fit_one(
    model: str,
    frame: pd.DataFrame,
    *,
    n_starts: int = 1,
    seed: int = FIT_SEED,
    compute_uncertainty: bool = True,
) -> dict[str, float | str | int | bool]:
    """Fit one accumulator with deterministic multi-start maximum likelihood.

    ``pyddm``'s differential-evolution solver is global but stochastic.  A
    small, explicitly seeded start set makes the result auditable and exposes
    whether the reported solution is reproducible across starts.  The best
    objective value is retained; all starts and boundary hits are persisted in
    the returned diagnostics rather than silently discarded.
    """
    import pyddm

    sample = pyddm.Sample.from_pandas_dataframe(
        frame[["rt", "correct", "lead_path"]],
        rt_column_name="rt",
        choice_column_name="correct",
    )
    t_dur = float(np.ceil(frame["rt"].max() + 0.5))
    n_starts = max(1, int(n_starts))
    candidates = []
    fit_errors = []
    for start in range(n_starts):
        try:
            fitted = pyddm.fit_adjust_model(
                sample,
                _build_model(model, t_dur),
                lossfunction=pyddm.LossLikelihood,
                fitparams={"seed": int(seed + start)},
                verbose=False,
            )
            objective = float(fitted.fitresult.value())
            candidates.append((objective, fitted, start))
        except Exception as error:  # pragma: no cover - solver-specific failures
            fit_errors.append(f"start_{start}:{type(error).__name__}:{error}")
    if not candidates:
        return {
            "log_likelihood": float("nan"),
            "t_dur_s": t_dur,
            "n_starts": n_starts,
            "best_start": -1,
            "optimizer_success": False,
            "boundary_hit": False,
            "boundary_parameters": "",
            "start_objectives": json.dumps({}, sort_keys=True),
            "start_converged": json.dumps({}, sort_keys=True),
            "fit_error": " | ".join(fit_errors),
        }
    _, fitted, best_start = min(candidates, key=lambda value: value[0])
    estimates = {
        name: float(value)
        for name, value in zip(
            fitted.get_model_parameter_names(), fitted.get_model_parameters()
        )
    }
    standard_errors = (
        _standard_errors(model, sample, estimates, t_dur)
        if compute_uncertainty
        else {name: float("nan") for name in MODEL_PARAMETERS[model]}
    )
    boundary_parameters = [
        name
        for name in MODEL_PARAMETERS[model]
        if (
            abs(estimates[name] - PARAMETER_RANGES[name][0])
            <= 0.01 * max(PARAMETER_RANGES[name][1] - PARAMETER_RANGES[name][0], 1e-12)
            or abs(PARAMETER_RANGES[name][1] - estimates[name])
            <= 0.01 * max(PARAMETER_RANGES[name][1] - PARAMETER_RANGES[name][0], 1e-12)
        )
    ]
    start_objectives = {str(start): float(objective) for objective, _, start in candidates}
    start_converged = {str(start): bool(np.isfinite(objective)) for objective, _, start in candidates}
    return {
        "log_likelihood": -float(fitted.fitresult.value()),
        "t_dur_s": t_dur,
        "n_starts": n_starts,
        "best_start": int(best_start),
        "optimizer_success": True,
        "boundary_hit": bool(boundary_parameters),
        "boundary_parameters": ",".join(boundary_parameters),
        "start_objectives": json.dumps(start_objectives, sort_keys=True),
        "start_converged": json.dumps(start_converged, sort_keys=True),
        "fit_error": " | ".join(fit_errors),
        **estimates,
        **{f"{name}_se": value for name, value in standard_errors.items()},
    }


def _fit_all_cells(
    cells: list[tuple[object, str, pd.DataFrame]],
    *,
    n_jobs: int,
    models: tuple[str, ...] = SSM_MODELS,
    n_starts: int = 1,
    compute_uncertainty: bool = True,
) -> list[dict[str, dict[str, float] | None]]:
    """Fit both models in every cell, optionally in worker processes.

    Parameters
    ----------
    cells
        Subject, condition, and decision frame for every cell to fit.
    n_jobs
        Worker processes to use; ``1`` stays in this process and a negative
        value uses every available CPU.

    Returns
    -------
    list of dict
        One mapping of model name to fit result per cell, in the order the
        cells were given. A cell with too few trials maps every model to
        ``None``.

    Notes
    -----
    The unit of work is one model in one cell, not one cell, because the
    urgency model takes several times longer than the integrator and pairing
    them would leave workers idle. Cells are fitted independently, so this is
    the level at which a cluster allocation scales: 32 CPUs occupy 32 of the
    192 fits at once.
    """
    tasks = [
        (index, model, frame)
        for index, (_, _, frame) in enumerate(cells)
        for model in models
        if len(frame) >= MINIMUM_FIT_TRIALS
    ]
    results: list[dict[str, dict[str, float] | None]] = [
        {model: None for model in models} for _ in cells
    ]
    if n_jobs == 1:
        for index, model, frame in tasks:
            results[index][model] = _fit_one(model, frame, n_starts=n_starts, compute_uncertainty=compute_uncertainty)
        return results
    with ProcessPoolExecutor(max_workers=None if n_jobs < 0 else n_jobs) as pool:
        submitted = {
            pool.submit(_fit_one, model, frame, n_starts=n_starts, compute_uncertainty=compute_uncertainty): (index, model)
            for index, model, frame in tasks
        }
        for future in as_completed(submitted):
            index, model = submitted[future]
            results[index][model] = future.result()
    return results


def fit_sequential_sampling_models(
    features: pd.DataFrame,
    *,
    n_jobs: int = 1,
    models: tuple[str, ...] = SSM_MODELS,
    n_starts: int = 1,
    compute_uncertainty: bool = True,
) -> pd.DataFrame:
    """Fit the urgency-gating and bounded-integrator models per subject.

    All requested models see the same trial-by-trial token evidence and differ
    only in how they use it.  The default keeps the historical two-model fit;
    pass ``models=MECHANISTIC_MODELS`` for the complete Thura et al. comparison.

    Parameters
    ----------
    features
        Canonical trial-feature table. Only eligible Fast and Slow task trials
        returned by :func:`~meg_tokens.behavior.trials.task_trials` are used.
    n_jobs
        Number of worker processes for the fits. ``1`` runs them in this
        process; a negative value uses every available CPU.
    models
        Model names from :data:`MODEL_PARAMETERS`. The fixed-bound ``ddm``
        must be included because all information criteria are reported
        relative to it.
    n_starts
        Number of independently seeded differential-evolution fits per cell.

    Returns
    -------
    pandas.DataFrame
        One row per subject, condition, and model, holding the fitted
        parameters and their standard errors, the log-likelihood, Akaike and
        Bayesian information criteria, both criteria relative to the integrator
        fit of the same cell, and the number of distinct token sequences the
        cell contains.

    Raises
    ------
    ValueError
        If ``n_jobs`` is zero.

    Notes
    -----
    Fitting is maximum likelihood by differential evolution over the whole
    decision-time distribution, per subject and condition; the group layer is
    :func:`population_parameters`. Cells with fewer than
    :data:`MINIMUM_FIT_TRIALS` scorable trials are reported with ``NaN``
    estimates and ``converged`` false rather than fitted. The two models are
    not nested — one is a filter and the other an integrator — so the criterion
    difference is an information-criterion comparison, not a likelihood-ratio
    test. The search is seeded, so refitting the same table reproduces it, in
    any number of processes: every fit is independent, so ``n_jobs`` changes
    the wall time and nothing else. Each likelihood evaluation solves the
    diffusion once per distinct token sequence in the cell, which is what makes
    these fits expensive enough to want the workers.
    """
    if n_jobs == 0:
        raise ValueError("n_jobs must not be zero")
    models = tuple(models)
    if not models or "ddm" not in models:
        raise ValueError("models must be non-empty and include the fixed ddm baseline")
    unknown = set(models).difference(MODEL_PARAMETERS)
    if unknown:
        raise ValueError(f"unknown sequential-sampling models: {sorted(unknown)}")
    trials = task_trials(features)
    require_columns(
        trials,
        [
            "dt_ms",
            "isCorrect",
            "token_directions",
            "nCorrectChoice",
            "trial_class_name",
            "trial_id",
            "subject",
            "condition",
        ],
    )
    parameter_columns = list(
        dict.fromkeys(
            column
            for model in models
            for name in MODEL_PARAMETERS[model]
            for column in (name, f"{name}_se")
        )
    )
    # Preserve optimizer provenance in the persisted cell table.  These fields
    # are intentionally separate from ``converged``: a fit can have a valid
    # best solution while showing boundary pressure, missing curvature, or
    # disagreement among seeded starts.
    diagnostic_columns = (
        "n_starts",
        "best_start",
        "optimizer_success",
        "boundary_hit",
        "boundary_parameters",
        "start_objectives",
        "start_converged",
        "fit_error",
    )
    cells = [
        (subject, condition, _decision_frame(selected))
        for subject, condition, selected in _subject_condition_groups(trials)
    ]
    fitted = _fit_all_cells(
        cells,
        n_jobs=n_jobs,
        models=models,
        n_starts=n_starts,
        compute_uncertainty=compute_uncertainty,
    )
    rows = []
    for (subject, condition, frame), fits in zip(cells, fitted):
        criteria = {}
        for model, fit in fits.items():
            n_parameters = len(MODEL_PARAMETERS[model])
            criteria[model] = (
                {
                    "aic": 2 * n_parameters - 2 * fit["log_likelihood"],
                    "bic": n_parameters * np.log(len(frame))
                    - 2 * fit["log_likelihood"],
                }
                if fit is not None
                else {"aic": float("nan"), "bic": float("nan")}
            )
        for model, fit in fits.items():
            failed_diagnostics = {
                "n_starts": int(n_starts),
                "best_start": -1,
                "optimizer_success": False,
                "boundary_hit": False,
                "boundary_parameters": "",
                "start_objectives": "{}",
                "start_converged": "{}",
                "fit_error": "insufficient_trials",
            }
            rows.append(
                {
                    "subject": subject,
                    "condition": condition,
                    "model": model,
                    "n_trials": int(len(frame)),
                    "n_token_sequences": int(frame["lead_path"].nunique()),
                    "n_parameters": len(MODEL_PARAMETERS[model]),
                    "log_likelihood": (
                        fit["log_likelihood"] if fit is not None else float("nan")
                    ),
                    "t_dur_s": (
                        fit["t_dur_s"] if fit is not None else float("nan")
                    ),
                    **criteria[model],
                    "delta_aic": criteria[model]["aic"] - criteria["ddm"]["aic"],
                    "delta_bic": criteria[model]["bic"] - criteria["ddm"]["bic"],
                    "converged": bool(fit is not None and fit.get("optimizer_success", True)),
                    **{
                        column: (
                            fit.get(column, float("nan"))
                            if fit is not None
                            else float("nan")
                        )
                        for column in parameter_columns
                    },
                    **{
                        column: (
                            fit.get(column, float("nan"))
                            if fit is not None
                            else failed_diagnostics[column]
                        )
                        for column in diagnostic_columns
                    },
                }
            )
    return pd.DataFrame(rows)


def fit_mechanistic_model_set(
    features: pd.DataFrame,
    *,
    n_jobs: int = 1,
    n_starts: int = 3,
    compute_uncertainty: bool = True,
) -> pd.DataFrame:
    """Fit the prespecified four-model Thura et al. comparison.

    This named wrapper prevents a cluster command from accidentally reverting
    to the historical two-model prototype.  ``n_starts=3`` is the default for
    the final analysis; it can be reduced only for a smoke test.
    """
    return fit_sequential_sampling_models(
        features,
        n_jobs=n_jobs,
        models=MECHANISTIC_MODELS,
        n_starts=n_starts,
        compute_uncertainty=compute_uncertainty,
    )


def model_comparison_statistics(fits: pd.DataFrame) -> pd.DataFrame:
    """Test the urgency-gating advantage over the bounded integrator.

    This is the species-generalization test: urgency gating beats integration
    in Cisek's monkeys and human subjects on this task, and the same
    comparison is made here on the human decision-time distributions.

    Parameters
    ----------
    fits
        Output of :func:`fit_sequential_sampling_models`.

    Returns
    -------
    pandas.DataFrame
        One row per condition and information criterion, holding how many
        subjects each model wins and a one-sample test of the per-subject
        criterion difference against zero.

    Notes
    -----
    The difference is urgency minus integrator, so a negative mean favors
    urgency gating. Subjects whose cell was not fitted contribute no difference
    and are excluded by the shared inference helper. Both models are fitted to
    identical trials, so the difference is paired by construction; no
    multiplicity correction is applied across conditions.
    """
    urgency = fits.loc[fits["model"] == "urgency"]
    rows = []
    for condition, group in urgency.groupby("condition", sort=True):
        for criterion in ("aic", "bic"):
            difference = group[f"delta_{criterion}"]
            rows.append(
                {
                    "analysis": "ssm_model_comparison",
                    "condition": condition,
                    "criterion": criterion,
                    "test": "one_sample_vs_zero",
                    "n_subjects_favoring_urgency": int((difference < 0).sum()),
                    "n_subjects_favoring_ddm": int((difference > 0).sum()),
                    **one_sample_statistics(difference),
                }
            )
    return pd.DataFrame(rows)


def mechanistic_model_statistics(fits: pd.DataFrame) -> pd.DataFrame:
    """Compare every candidate model with the fixed-bound ``ddm`` baseline.

    The returned differences are candidate minus ``ddm`` (negative values
    favour the candidate).  This keeps the historical ``ssmcomparisonstats``
    schema while making the collapsing-bound and additive-urgency sensitivity
    comparisons explicit and paired by subject/condition.
    """
    if fits.empty:
        return pd.DataFrame()
    baseline = fits.loc[fits["model"] == "ddm"].set_index(
        ["subject", "condition"]
    )
    rows = []
    for model in sorted(set(fits["model"]) - {"ddm"}):
        candidate = fits.loc[fits["model"] == model].set_index(
            ["subject", "condition"]
        )
        for condition in sorted(set(candidate.index.get_level_values("condition"))):
            candidate_condition = candidate.loc[
                candidate.index.get_level_values("condition") == condition
            ]
            baseline_condition = baseline.loc[
                baseline.index.get_level_values("condition") == condition
            ]
            common = baseline_condition.index.intersection(candidate_condition.index)
            for criterion in ("aic", "bic"):
                difference = (
                    candidate_condition.loc[common, criterion]
                    - baseline_condition.loc[common, criterion]
                ).replace([np.inf, -np.inf], np.nan).dropna()
                rows.append(
                    {
                        "analysis": "ssm_mechanistic_model_comparison",
                        "model": model,
                        "condition": condition,
                        "criterion": criterion,
                        "test": "one_sample_vs_zero",
                        "n_subjects_favoring_candidate": int((difference < 0).sum()),
                        "n_subjects_favoring_ddm": int((difference > 0).sum()),
                        **one_sample_statistics(difference),
                    }
                )
    return pd.DataFrame(rows)


def eligibility_audit(features: pd.DataFrame) -> pd.DataFrame:
    """Persist the exact trial counts entering a mechanistic fit.

    Counts are deliberately derived from the canonical flags rather than from
    model-specific rows.  This makes exclusions (never-started, lapses,
    non-task runs, missing/non-positive decision times and short token logs)
    visible without silently changing the project's primary eligibility rule.
    """
    require_columns(
        features,
        ["primary_analysis_eligible", "condition", "is_started", "has_choice", "dt_ms"],
    )
    condition = features["condition"].astype(str).str.lower()
    task = condition.isin({"fast", "slow"}).fillna(False).to_numpy(dtype=bool)
    started = features["is_started"].fillna(False).astype(bool).to_numpy()
    chosen = features["has_choice"].fillna(False).astype(bool).to_numpy()
    eligible = features["primary_analysis_eligible"].fillna(False).astype(bool).to_numpy()
    dt = pd.to_numeric(features["dt_ms"], errors="coerce")
    checks = {
        "all_feature_rows": np.ones(len(features), dtype=bool),
        "task_condition": task,
        "started_task": task & started,
        "chosen_task": task & started & chosen,
        "primary_analysis_eligible": eligible,
        "finite_dt": eligible & np.isfinite(dt.to_numpy(dtype=float)),
        "positive_dt_for_first_passage": eligible & np.isfinite(dt.to_numpy(dtype=float)) & (dt.to_numpy(dtype=float) > 0),
    }
    previous = np.ones(len(features), dtype=bool)
    rows = []
    for criterion, mask in checks.items():
        mask = np.asarray(mask, dtype=bool)
        retained = previous & mask
        retained_subjects = features.loc[retained, "subject"].nunique() if "subject" in features else np.nan
        retained_cells = features.loc[retained, ["subject", "condition"]].drop_duplicates().shape[0] if {"subject", "condition"}.issubset(features.columns) else np.nan
        rows.append({
            "criterion": criterion,
            "n_rows": int(mask.sum()),
            "n_retained_cumulative": int(retained.sum()),
            "n_excluded_at_step": int(previous.sum() - retained.sum()),
            "retention_fraction_cumulative": float(retained.mean()) if len(retained) else float("nan"),
            "n_subjects_retained": int(retained_subjects) if np.isfinite(retained_subjects) else np.nan,
            "n_subject_condition_cells_retained": int(retained_cells) if np.isfinite(retained_cells) else np.nan,
        })
        previous = retained
    # Short logs are retained by the project's primary eligibility flag, but
    # they cannot support the validated 15-row design-alignment derivative.
    # Record this as a diagnostic rather than silently applying a new exclusion
    # to the mechanistic fit.  This makes the distinction auditable in the
    # output without changing the project's prespecified trial population.
    if "token_log_short" in features:
        short = features["token_log_short"].fillna(False).astype(bool).to_numpy()
        rows.append({
            "criterion": "diagnostic_short_token_log",
            "n_rows": int(short.sum()),
            "n_retained_cumulative": int(previous.sum()),
            "n_excluded_at_step": 0,
            "retention_fraction_cumulative": float(previous.mean()) if len(previous) else float("nan"),
            "n_subjects_retained": int(features.loc[previous, "subject"].nunique()) if "subject" in features else np.nan,
            "n_subject_condition_cells_retained": int(features.loc[previous, ["subject", "condition"]].drop_duplicates().shape[0]) if {"subject", "condition"}.issubset(features.columns) else np.nan,
            "diagnostic_n_short": int(short.sum()),
        })
    return pd.DataFrame(rows)


def heldout_model_evaluation(
    features: pd.DataFrame,
    *,
    models: tuple[str, ...] = MECHANISTIC_MODELS,
    folds: int = 2,
    n_starts: int = 1,
    return_predictions: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Fit each model on deterministic folds and score the held-out fold.

    Fold assignment is deterministic and stratified by observed response
    within each subject/condition cell.  The same primary-eligibility and
    first-passage rules as the full fit are used.  The solver horizon is the
    maximum RT in *both* train and held-out data.  Set ``return_predictions``
    to receive ``(aggregate, per_trial)``; otherwise only the aggregate table
    is returned.
    """
    if folds < 2:
        raise ValueError("folds must be at least 2")
    trials = task_trials(features)
    require_columns(
        trials,
        ["dt_ms", "isCorrect", "token_directions", "nCorrectChoice", "trial_class_name", "trial_id", "subject", "condition"],
    )
    rows = []
    trial_rows = []
    for subject, condition, selected in _subject_condition_groups(trials):
        frame = _decision_frame(selected)
        frame = frame.copy()
        if not frame.empty:
            fold_values = np.full(len(frame), -1, dtype=int)
            for _, response_group in frame.groupby("correct", sort=True):
                ordered = sorted(
                    response_group.index,
                    key=lambda index: hashlib.sha1(
                        str(frame.loc[index, "trial_id"]).encode("utf-8")
                    ).hexdigest(),
                )
                for rank, index in enumerate(ordered):
                    fold_values[frame.index.get_loc(index)] = rank % folds
            frame["fold"] = fold_values
        for fold in range(folds):
            if frame.empty:
                train = frame.copy()
                test = frame.copy()
            else:
                train = frame.loc[frame["fold"] != fold].drop(columns="fold")
                test = frame.loc[frame["fold"] == fold].drop(columns="fold")
            for model_name in models:
                failure = None
                fit = None
                if len(train) < MINIMUM_FIT_TRIALS or test.empty:
                    failure = "insufficient_train_or_test_trials"
                else:
                    fit = _fit_one(
                        model_name,
                        train,
                        n_starts=n_starts,
                        compute_uncertainty=False,
                    )
                    if not fit.get("optimizer_success", True):
                        failure = fit.get("fit_error", "optimizer_failed")
                if failure is not None:
                    rows.append({
                        "subject": subject,
                        "condition": condition,
                        "model": model_name,
                        "fold": fold,
                        "n_train": int(len(train)),
                        "n_test": int(len(test)),
                        "heldout_log_likelihood": float("nan"),
                        "heldout_log_likelihood_per_trial": float("nan"),
                        "observed_accuracy": float(test["correct"].mean()) if len(test) else float("nan"),
                        "predicted_accuracy": float("nan"),
                        "observed_mean_rt_s": float(test["rt"].mean()) if len(test) else float("nan"),
                        "observed_q50_rt_s": float(np.quantile(test["rt"], 0.5)) if len(test) else float("nan"),
                        "observed_q90_rt_s": float(np.quantile(test["rt"], 0.9)) if len(test) else float("nan"),
                        "predicted_q50_rt_s": float("nan"),
                        "predicted_q90_rt_s": float("nan"),
                        "solver_horizon_s": float("nan"),
                        "n_starts": int(fit.get("n_starts", n_starts)) if fit else int(n_starts),
                        "best_start": int(fit.get("best_start", -1)) if fit else -1,
                        "optimizer_success": False,
                        "boundary_hit": bool(fit.get("boundary_hit", False)) if fit else False,
                        "boundary_parameters": fit.get("boundary_parameters", "") if fit else "",
                        "start_objectives": fit.get("start_objectives", "{}") if fit else "{}",
                        "start_converged": fit.get("start_converged", "{}") if fit else "{}",
                        "fit_error": str(failure),
                        "converged": False,
                    })
                    continue
                sample = __import__("pyddm").Sample.from_pandas_dataframe(
                    test[["rt", "correct", "lead_path"]],
                    rt_column_name="rt", choice_column_name="correct",
                )
                horizon = float(np.ceil(max(train["rt"].max(), test["rt"].max()) + 0.5))
                fitted_model = _build_model(model_name, horizon, {
                    name: float(fit[name]) for name in MODEL_PARAMETERS[model_name]
                })
                loss = float(__import__("pyddm").get_model_loss(
                    fitted_model, sample, lossfunction=__import__("pyddm").LossLikelihood
                ))
                solution_by_path = {
                    path: fitted_model.solve(conditions={"lead_path": path})
                    for path in test["lead_path"].unique()
                }
                predicted_accuracy = np.array([
                    solution_by_path[path].prob("correct")
                    for path in test["lead_path"]
                ], dtype=float)
                observed_rt = test["rt"].to_numpy(dtype=float)
                predicted_quantiles = _predicted_quantiles(
                    fitted_model, test["lead_path"].value_counts(normalize=True).to_dict(), (0.5, 0.9)
                )
                rows.append({
                    "subject": subject,
                    "condition": condition,
                    "model": model_name,
                    "fold": fold,
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "heldout_log_likelihood": -loss,
                    "heldout_log_likelihood_per_trial": -loss / len(test),
                    "observed_accuracy": float(test["correct"].mean()),
                    "predicted_accuracy": float(predicted_accuracy.mean()),
                    "observed_mean_rt_s": float(observed_rt.mean()),
                    "observed_q50_rt_s": float(np.quantile(observed_rt, 0.5)),
                    "observed_q90_rt_s": float(np.quantile(observed_rt, 0.9)),
                    "predicted_q50_rt_s": predicted_quantiles[0.5],
                    "predicted_q90_rt_s": predicted_quantiles[0.9],
                    "solver_horizon_s": horizon,
                    "boundary_hit": bool(fit.get("boundary_hit", False)),
                    "n_starts": int(fit.get("n_starts", n_starts)),
                    "best_start": int(fit.get("best_start", -1)),
                    "optimizer_success": bool(fit.get("optimizer_success", True)),
                    "boundary_parameters": fit.get("boundary_parameters", ""),
                    "start_objectives": fit.get("start_objectives", "{}"),
                    "start_converged": fit.get("start_converged", "{}"),
                    "fit_error": fit.get("fit_error", ""),
                    "converged": True,
                })
                for trial, pred_acc in zip(test.to_dict("records"), predicted_accuracy):
                    solution = solution_by_path[trial["lead_path"]]
                    choice = "correct" if int(trial["correct"]) else "error"
                    density = float(
                        np.interp(
                            float(trial["rt"]),
                            np.asarray(fitted_model.t_domain(), dtype=float),
                            np.asarray(solution.pdf(choice), dtype=float),
                            left=0.0,
                            right=0.0,
                        )
                    )
                    trial_rows.append({
                        "subject": subject,
                        "condition": condition,
                        "model": model_name,
                        "fold": fold,
                        "trial_id": trial["trial_id"],
                        "dt_s": trial["rt"],
                        "observed_correct": trial["correct"],
                        "predicted_accuracy": float(pred_acc),
                        "heldout_log_score": float(np.log(max(density, 1e-12))),
                        "solver_horizon_s": horizon,
                    })
    aggregate = pd.DataFrame(rows)
    predictions = pd.DataFrame(trial_rows)
    return (aggregate, predictions) if return_predictions else aggregate


def _heldout_subject_scores(heldout: pd.DataFrame) -> pd.DataFrame:
    """Return weighted subject × condition × model held-out scores.

    Fold log likelihoods are summed and divided by the total held-out trial
    count within subject × condition × model (equivalently, fold means are
    weighted by ``n_test``).  The inferential unit is therefore a subject, not
    a fold or a trial.  Legacy tables without summed likelihood/count columns
    fall back to an unweighted fold mean.  Shared by every held-out
    comparison so the weighting can only be defined once.
    """
    required = {
        "subject", "condition", "model", "fold",
        "heldout_log_likelihood_per_trial",
    }
    require_columns(heldout, sorted(required))
    if heldout.empty:
        return pd.DataFrame(columns=["subject", "condition", "model", "score"])
    scored = heldout.assign(
        score=pd.to_numeric(
            heldout["heldout_log_likelihood_per_trial"], errors="coerce"
        )
    )
    if {"heldout_log_likelihood", "n_test"}.issubset(scored.columns):
        # A fold's per-trial score is not an equally weighted subject
        # estimate when folds have different sizes.  Reconstruct the
        # subject-level score from summed held-out log likelihood and trial
        # count; this is the same as weighting fold means by n_test.
        scored["heldout_ll"] = pd.to_numeric(
            scored["heldout_log_likelihood"], errors="coerce"
        )
        scored["n_test_numeric"] = pd.to_numeric(
            scored["n_test"], errors="coerce"
        )
        scored = scored.loc[
            np.isfinite(scored["heldout_ll"])
            & np.isfinite(scored["n_test_numeric"])
            & (scored["n_test_numeric"] > 0)
        ]
        folded = (
            scored.groupby(["subject", "condition", "model"], as_index=False)
            .agg(
                heldout_ll=("heldout_ll", "sum"),
                n_test_numeric=("n_test_numeric", "sum"),
            )
        )
        folded["score"] = folded["heldout_ll"] / folded["n_test_numeric"]
    else:
        folded = (
            scored.groupby(["subject", "condition", "model"], as_index=False)["score"]
            .mean()
        )
    return folded[["subject", "condition", "model", "score"]]


def _paired_score_contrast(
    a_scores: pd.Series, b_scores: pd.Series, condition: str
) -> dict[str, object] | None:
    """One-sample paired contrast (a − b) for one condition, or ``None``.

    ``None`` is returned when no subject has both scores after alignment.
    Shared by every held-out comparison so the CI/favouring-count logic is
    defined once regardless of what "a" and "b" represent.
    """
    a_condition = a_scores.loc[a_scores.index.get_level_values("condition") == condition]
    b_condition = b_scores.loc[b_scores.index.get_level_values("condition") == condition]
    common = a_condition.index.intersection(b_condition.index)
    differences = (
        a_condition.loc[common] - b_condition.loc[common]
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if differences.empty:
        return None
    summary = one_sample_statistics(differences)
    n = int(summary["n_subjects"])
    if n > 1 and np.isfinite(summary["sem"]):
        half_width = float(stats.t.ppf(0.975, n - 1) * summary["sem"])
    else:
        half_width = float("nan")
    return {
        "n_subjects_favoring_a": int((differences > 0).sum()),
        "n_subjects_favoring_b": int((differences < 0).sum()),
        "ci95_low": float(summary["mean"] - half_width),
        "ci95_high": float(summary["mean"] + half_width),
        **summary,
    }


def heldout_model_statistics(heldout: pd.DataFrame) -> pd.DataFrame:
    """Compare held-out candidate scores with the fixed-bound baseline.

    The inferential unit is a subject, not a fold or a trial (see
    :func:`_heldout_subject_scores`), and the output remains
    condition-stratified.  Positive candidate-minus-DDM values favour the
    candidate.
    """
    folded = _heldout_subject_scores(heldout)
    if folded.empty:
        return pd.DataFrame()
    indexed = folded.set_index(["subject", "condition"])
    baseline = indexed.loc[indexed["model"] == "ddm", "score"]
    rows = []
    for model in sorted(set(folded["model"]) - {"ddm"}):
        candidate = indexed.loc[indexed["model"] == model, "score"]
        for condition in sorted(set(candidate.index.get_level_values("condition"))):
            contrast = _paired_score_contrast(candidate, baseline, condition)
            if contrast is None:
                continue
            rows.append({
                "analysis": "ssm_heldout_model_comparison",
                "model": model,
                "condition": condition,
                "criterion": "heldout_log_likelihood_per_trial",
                "test": "one_sample_vs_zero",
                "direction": "candidate_minus_ddm_positive_favors_candidate",
                "n_subjects_favoring_candidate": contrast.pop("n_subjects_favoring_a"),
                "n_subjects_favoring_ddm": contrast.pop("n_subjects_favoring_b"),
                **contrast,
            })
    return pd.DataFrame(rows)


def heldout_pairwise_model_statistics(heldout: pd.DataFrame) -> pd.DataFrame:
    """Compare every pair of candidates' held-out scores directly.

    :func:`heldout_model_statistics` only contrasts each candidate against
    the fixed-bound baseline.  Two candidates that both beat ``ddm`` are not
    thereby shown to differ from each other — e.g. urgency gating and the
    collapsing bound must be compared directly, not inferred by eye from two
    separate baseline contrasts.  Reuses the identical weighted subject-level
    score and one-sample-vs-zero paired contrast as
    :func:`heldout_model_statistics`, for every unordered pair of models
    present (including the pairs already covered against ``ddm``, so this
    table is a complete pairwise view on its own).
    """
    folded = _heldout_subject_scores(heldout)
    if folded.empty:
        return pd.DataFrame()
    indexed = folded.set_index(["subject", "condition"])
    models = sorted(set(folded["model"]))
    rows = []
    for model_a, model_b in combinations(models, 2):
        a_scores = indexed.loc[indexed["model"] == model_a, "score"]
        b_scores = indexed.loc[indexed["model"] == model_b, "score"]
        conditions = sorted(
            set(a_scores.index.get_level_values("condition"))
            & set(b_scores.index.get_level_values("condition"))
        )
        for condition in conditions:
            contrast = _paired_score_contrast(a_scores, b_scores, condition)
            if contrast is None:
                continue
            rows.append({
                "analysis": "ssm_heldout_pairwise_comparison",
                "model_a": model_a,
                "model_b": model_b,
                "condition": condition,
                "criterion": "heldout_log_likelihood_per_trial",
                "test": "one_sample_vs_zero",
                "direction": "a_minus_b_positive_favors_model_a",
                "n_subjects_favoring_model_a": contrast.pop("n_subjects_favoring_a"),
                "n_subjects_favoring_model_b": contrast.pop("n_subjects_favoring_b"),
                **contrast,
            })
    return pd.DataFrame(rows)


def heldout_fold_audit(
    heldout: pd.DataFrame,
    *,
    expected_folds: int | None = None,
) -> pd.DataFrame:
    """Persist fold/count completeness for each subject-cell-model fit."""
    require_columns(
        heldout,
        ["subject", "condition", "model", "fold", "n_train", "n_test"],
    )
    if heldout.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in heldout.groupby(
        ["subject", "condition", "model"], sort=True
    ):
        observed = int(group["fold"].nunique())
        rows.append({
            "analysis": "ssm_heldout_fold_audit",
            "subject": keys[0],
            "condition": keys[1],
            "model": keys[2],
            "expected_folds": int(expected_folds) if expected_folds is not None else np.nan,
            "n_folds_observed": observed,
            "folds_complete": bool(
                expected_folds is None or observed == int(expected_folds)
            ),
            "n_converged": int(group["converged"].fillna(False).astype(bool).sum())
            if "converged" in group else 0,
            "n_failed": int((~group["converged"].fillna(False).astype(bool)).sum())
            if "converged" in group else int(len(group)),
            "n_cells_observed": int(len(group)),
            "n_test_total": int(pd.to_numeric(group["n_test"], errors="coerce").sum()),
            "n_train_min": int(pd.to_numeric(group["n_train"], errors="coerce").min()),
            "n_train_max": int(pd.to_numeric(group["n_train"], errors="coerce").max()),
            "fold_ids": json.dumps(sorted(pd.to_numeric(group["fold"], errors="coerce").dropna().astype(int).unique().tolist())),
        })
    return pd.DataFrame(rows)


def _predicted_quantiles(
    model,
    path_weights: dict[tuple[int, ...], float],
    quantiles: tuple[float, ...],
    choice: str = "both",
) -> dict[float, float]:
    """Compute mixture quantiles from model first-passage densities."""
    times = np.asarray(model.t_domain(), dtype=float)
    density = np.zeros(times.size, dtype=float)
    for path, weight in path_weights.items():
        solution = model.solve(conditions={"lead_path": path})
        if choice == "both":
            density += float(weight) * (
                np.asarray(solution.pdf("correct"), dtype=float)
                + np.asarray(solution.pdf("error"), dtype=float)
            )
        else:
            density += float(weight) * np.asarray(solution.pdf(choice), dtype=float)
    mass = float(np.sum(density) * model.dt)
    if mass <= 0:
        return {q: float("nan") for q in quantiles}
    cumulative = np.cumsum(density) * model.dt / mass
    return {q: float(np.interp(q, cumulative, times)) for q in quantiles}


def matched_sequence_diagnostic(
    features: pd.DataFrame,
    *,
    early_jumps: int = 3,
    convergence_jump: int = 6,
    max_late_trajectory_distance: float = 2.0,
    min_trials: int = 5,
) -> pd.DataFrame:
    """Match opposite early histories after a prespecified convergence point.

    Exact suffix matching immediately after the early window is impossible for
    this fixed-length design when both paths end at the same correct-target
    lead: opposite early signs cannot have the same intervening evidence and
    still converge.  The primary diagnostic therefore matches stimulus-only
    paths on (i) opposite early signs with equal absolute lead, (ii) equal
    running lead at jump 6 (600 ms, three filter time constants after the
    early window), and (iii) a maximum absolute late lead-trajectory distance
    of two tokens.  A deterministic greedy one-to-one match minimizes the
    trajectory RMSE, breaking ties by trial ID.  Both decisions must occur
    strictly after jump 6.  Response, correctness, and final outcome are
    never used to select pairs. ``min_trials`` is retained for compatibility
    with the earlier exact-suffix prototype but is not a selection gate;
    :func:`matched_sequence_audit` reports cell-level retention instead.
    """
    if early_jumps < 1:
        raise ValueError("early_jumps must be positive")
    if convergence_jump <= early_jumps:
        raise ValueError("convergence_jump must be after early_jumps")
    if max_late_trajectory_distance < 0:
        raise ValueError("max_late_trajectory_distance must be non-negative")
    trials = task_trials(features)
    require_columns(
        trials,
        ["token_directions", "nCorrectChoice", "dt_ms", "decision_token_index", "isCorrect", "subject", "condition", "trial_id"],
    )
    rows = []
    for (subject, condition), group in trials.groupby(["subject", "condition"], sort=True):
        records = []
        for trial in group.to_dict("records"):
            directions = str(trial["token_directions"])
            target = int(trial["nCorrectChoice"])
            path = _lead_path(directions, target)
            if len(path) < convergence_jump:
                continue
            decision_token_index = pd.to_numeric(
                pd.Series([trial["decision_token_index"]]), errors="coerce"
            ).iloc[0]
            early_lead = int(path[early_jumps - 1])
            if early_lead == 0:
                continue
            late_trajectory = tuple(path[convergence_jump - 1 :])
            records.append({
                **trial,
                "early_lead": early_lead,
                "early_bias": "for" if early_lead > 0 else "against",
                "match_key": (abs(early_lead), int(path[convergence_jump - 1])),
                "convergence_lead": int(path[convergence_jump - 1]),
                "late_trajectory": late_trajectory,
                "late_lead": int(path[-1]),
                "decision_token_index": float(decision_token_index)
                if np.isfinite(decision_token_index) else float("nan"),
                "lead_path": path,
            })
        frame = pd.DataFrame(records)
        if frame.empty:
            continue
        # Greedy 1:1 matching uses only stimulus-derived fields and trial IDs.
        # ``min_trials`` is retained for API compatibility but is not used as
        # a stimulus-selection gate; retention is reported at cell level.
        pair_counter = 0
        n_for_total = int((frame["early_bias"] == "for").sum())
        n_against_total = int((frame["early_bias"] == "against").sum())
        for match_key, candidates in frame.groupby("match_key", sort=True):
            for_records = sorted(
                candidates.loc[candidates["early_bias"] == "for"].to_dict("records"),
                key=lambda row: str(row["trial_id"]),
            )
            against_records = sorted(
                candidates.loc[candidates["early_bias"] == "against"].to_dict("records"),
                key=lambda row: str(row["trial_id"]),
            )
            if not for_records or not against_records:
                continue
            pair_candidates = []
            for biased_for in for_records:
                distances = []
                for biased_against in against_records:
                    left = np.asarray(biased_for["late_trajectory"], dtype=float)
                    right = np.asarray(biased_against["late_trajectory"], dtype=float)
                    if left.shape != right.shape:
                        continue
                    absolute_distance = float(np.max(np.abs(left - right)))
                    if absolute_distance <= max_late_trajectory_distance:
                        rmse = float(np.sqrt(np.mean((left - right) ** 2)))
                        distances.append((rmse, str(biased_against["trial_id"]), absolute_distance, biased_against))
                if distances:
                    for rmse, against_id, absolute_distance, biased_against in distances:
                        pair_candidates.append(
                            (rmse, absolute_distance, str(biased_for["trial_id"]), against_id, biased_for, biased_against)
                        )
            # Sort all possible edges globally.  Tracking IDs, rather than
            # mutable list positions, makes the greedy result stable after a
            # prior against trial has been accepted.
            used_for: set[str] = set()
            used_against: set[str] = set()
            accepted_pairs = []
            for rmse, absolute_distance, for_id, against_id, biased_for, biased_against in sorted(
                pair_candidates, key=lambda row: (row[0], row[1], row[2], row[3])
            ):
                if for_id in used_for or against_id in used_against:
                    continue
                used_for.add(for_id)
                used_against.add(against_id)
                accepted_pairs.append((rmse, absolute_distance, biased_for, biased_against))
            n_stimulus_pairs = len(accepted_pairs)
            n_post_pairs = sum(
                int(
                    np.isfinite(biased_for["decision_token_index"])
                    and np.isfinite(biased_against["decision_token_index"])
                    and biased_for["decision_token_index"] > convergence_jump
                    and biased_against["decision_token_index"] > convergence_jump
                )
                for _, _, biased_for, biased_against in accepted_pairs
            )
            for rmse, absolute_distance, biased_for, biased_against in accepted_pairs:
                pair_id = (
                    f"{subject}-{str(condition).lower()}-"
                    f"p{pair_counter:05d}"
                )
                pair_counter += 1
                post_convergence = bool(
                    np.isfinite(biased_for["decision_token_index"])
                    and np.isfinite(biased_against["decision_token_index"])
                    and biased_for["decision_token_index"] > convergence_jump
                    and biased_against["decision_token_index"] > convergence_jump
                )
                matching_retention = float(2 * n_stimulus_pairs / len(frame)) if len(frame) else float("nan")
                for bias, record in (("for", biased_for), ("against", biased_against)):
                    dt = float(pd.to_numeric(pd.Series([record["dt_ms"]]), errors="coerce").iloc[0])
                    accuracy = float(pd.to_numeric(pd.Series([record["isCorrect"]]), errors="coerce").iloc[0])
                    rows.append({
                        "subject": subject,
                        "condition": condition,
                        "pair_id": pair_id,
                        "match_suffix": json.dumps(list(record["late_trajectory"])),
                        "early_lead_abs": int(match_key[0]),
                        "early_jumps": early_jumps,
                        "convergence_jump": convergence_jump,
                        "convergence_lead": int(match_key[1]),
                        "early_bias": bias,
                        "n_trials": 1,
                        "mean_dt_ms": dt,
                        "mean_accuracy": accuracy,
                        "mean_early_lead": float(record["early_lead"]),
                        "mean_late_lead": float(record["late_lead"]),
                        "n_for": 1,
                        "n_against": 1,
                        "matching_retention": matching_retention,
                        "post_convergence_eligible": post_convergence,
                        "n_stimulus_pairs_subject_condition": n_stimulus_pairs,
                        "n_post_convergence_pairs_subject_condition": n_post_pairs,
                        "post_convergence_pair_fraction": float(n_post_pairs / n_stimulus_pairs)
                        if n_stimulus_pairs else float("nan"),
                        "n_candidates_for": int(len(for_records)),
                        "n_candidates_against": int(len(against_records)),
                        "n_candidates_for_total": n_for_total,
                        "n_candidates_against_total": n_against_total,
                        "n_input_trials_subject_condition": int(len(frame)),
                        "matched_trial_id": str(record["trial_id"]),
                        "decision_token_index": record["decision_token_index"],
                        "lead_path": json.dumps(list(record["lead_path"])),
                        "trajectory_rmse": rmse,
                        "late_trajectory_max_distance": absolute_distance,
                        "max_late_trajectory_distance": max_late_trajectory_distance,
                        "matching_rule": "stimulus_only_equal_early_abs_and_jump6_lead_greedy_rmse",
                        "decision_after_convergence": post_convergence,
                    })
    return pd.DataFrame(rows)


def matched_sequence_audit(
    matched: pd.DataFrame,
    features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Collapse matched-pair retention/balance diagnostics by cell."""
    if matched.empty and features is None:
        return pd.DataFrame()
    input_counts: dict[tuple[object, object], int] = {}
    if features is not None:
        trials = task_trials(features)
        input_counts = {
            (subject, condition): int(len(group))
            for (subject, condition), group in trials.groupby(
                ["subject", "condition"], sort=True
            )
        }
    matched_groups = {
        key: group for key, group in matched.groupby(
            ["subject", "condition"], sort=True
        )
    } if not matched.empty else {}
    rows = []
    for key in sorted(set(input_counts) | set(matched_groups), key=lambda item: (str(item[0]), str(item[1]))):
        subject, condition = key
        group = matched_groups.get(key, pd.DataFrame())
        if group.empty:
            rows.append({
                "analysis": "ssm_matched_sequence_audit",
                "subject": subject,
                "condition": condition,
                "early_jumps": 3,
                "convergence_jump": 6,
                "max_late_trajectory_distance": 2.0,
                "n_input_trials": input_counts[key],
                "n_stimulus_pairs": 0,
                "n_post_convergence_pairs": 0,
                "post_convergence_pair_fraction": float("nan"),
                "n_post_convergence_trials": 0,
                "stimulus_pair_retention": 0.0,
                "post_convergence_pair_retention": 0.0,
                "paired_bias_count_ratio": float("nan"),
                "early_lead_abs_mean_difference": float("nan"),
                "convergence_lead_mean_difference": float("nan"),
                "trajectory_rmse_mean": float("nan"),
                "trajectory_rmse_max": float("nan"),
                "late_trajectory_max_distance_max": float("nan"),
                "matching_rule": "stimulus_only_equal_early_abs_and_jump6_lead_greedy_rmse",
            })
            continue
        pairs = group.drop_duplicates("pair_id")
        row = pairs.iloc[0]
        n_stimulus_pairs = int(len(pairs))
        n_post_pairs = int(
            pairs["post_convergence_eligible"].fillna(False).astype(bool).sum()
        )
        pair_pivot = group.pivot_table(
            index="pair_id", columns="early_bias",
            values=["early_lead_abs", "convergence_lead"], aggfunc="first",
        )
        early_difference = (
            pair_pivot[("early_lead_abs", "against")]
            - pair_pivot[("early_lead_abs", "for")]
        ).dropna() if ("early_lead_abs", "against") in pair_pivot and ("early_lead_abs", "for") in pair_pivot else pd.Series(dtype=float)
        convergence_difference = (
            pair_pivot[("convergence_lead", "against")]
            - pair_pivot[("convergence_lead", "for")]
        ).dropna() if ("convergence_lead", "against") in pair_pivot and ("convergence_lead", "for") in pair_pivot else pd.Series(dtype=float)
        rows.append({
            "analysis": "ssm_matched_sequence_audit",
            "subject": subject,
            "condition": condition,
            "early_jumps": int(row.get("early_jumps", 3)),
            "convergence_jump": int(row["convergence_jump"]),
            "max_late_trajectory_distance": float(row["max_late_trajectory_distance"]),
            "n_input_trials": int(row["n_input_trials_subject_condition"]),
            "n_stimulus_pairs": n_stimulus_pairs,
            "n_post_convergence_pairs": n_post_pairs,
            "post_convergence_pair_fraction": float(n_post_pairs / n_stimulus_pairs)
            if n_stimulus_pairs else float("nan"),
            "n_post_convergence_trials": int(n_post_pairs * 2),
            "stimulus_pair_retention": float(
                2 * n_stimulus_pairs / max(int(row["n_input_trials_subject_condition"]), 1)
            ),
            "post_convergence_pair_retention": float(
                2 * n_post_pairs / max(int(row["n_input_trials_subject_condition"]), 1)
            ),
            "paired_bias_count_ratio": float(
                len(group.loc[group["early_bias"] == "against"])
                / max(len(group.loc[group["early_bias"] == "for"]), 1)
            ),
            "early_lead_abs_mean_difference": float(early_difference.mean()) if len(early_difference) else float("nan"),
            "convergence_lead_mean_difference": float(convergence_difference.mean()) if len(convergence_difference) else float("nan"),
            "trajectory_rmse_mean": float(
                pd.to_numeric(pairs["trajectory_rmse"], errors="coerce").mean()
            ),
            "trajectory_rmse_max": float(
                pd.to_numeric(pairs["trajectory_rmse"], errors="coerce").max()
            ),
            "late_trajectory_max_distance_max": float(
                pd.to_numeric(pairs["late_trajectory_max_distance"], errors="coerce").max()
            ),
            "matching_rule": row["matching_rule"],
        })
    return pd.DataFrame(rows)


def matched_sequence_statistics(matched: pd.DataFrame) -> pd.DataFrame:
    """Test early-bias differences on subject-level matched summaries."""
    if matched.empty:
        return pd.DataFrame()
    if "post_convergence_eligible" in matched:
        matched = matched.loc[matched["post_convergence_eligible"].fillna(False).astype(bool)]
    if matched.empty:
        return pd.DataFrame()
    rows = []
    for condition, group in matched.groupby("condition", sort=True):
        pivot = group.pivot_table(
            index=["subject", "pair_id"],
            columns="early_bias",
            values=["mean_dt_ms", "mean_accuracy"],
            aggfunc="mean",
        )
        for measure in ("mean_dt_ms", "mean_accuracy"):
            if (measure, "for") not in pivot or (measure, "against") not in pivot:
                continue
            pair_differences = (pivot[(measure, "against")] - pivot[(measure, "for")]).dropna()
            differences = pair_differences.groupby(level="subject").mean()
            rows.append({
                "analysis": "matched_sequence",
                "condition": condition,
                "measure": measure,
                "n_subjects": int(differences.index.get_level_values("subject").nunique()),
                "n_pairs": int(len(pair_differences)),
                "mean_against_minus_for": float(differences.mean()) if len(differences) else float("nan"),
                **one_sample_statistics(differences),
            })
    return pd.DataFrame(rows)


def _matched_path(value: object) -> tuple[int, ...]:
    """Decode the JSON path persisted by the matched-sequence derivative."""
    if isinstance(value, tuple):
        return tuple(int(item) for item in value)
    if isinstance(value, list):
        return tuple(int(item) for item in value)
    try:
        decoded = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid matched lead path: {value!r}") from error
    return tuple(int(item) for item in decoded)


def matched_sequence_model_predictions(
    features: pd.DataFrame,
    fits: pd.DataFrame,
    matched: pd.DataFrame,
) -> pd.DataFrame:
    """Predict behavior for the exact retained matched histories.

    For each fitted subject × condition × model × pair × early-bias row, the
    correct probability and mean decision time are conditioned on first
    passage after the prespecified convergence jump.  No new matching occurs
    here: this table is keyed to the already retained stimulus-only pairs.
    """
    if matched.empty or fits.empty:
        return pd.DataFrame()
    require_columns(matched, ["subject", "condition", "pair_id", "early_bias", "lead_path", "convergence_jump"])
    rows = []
    for fit in fits.loc[fits["converged"].fillna(False).astype(bool)].to_dict("records"):
        fit_condition = str(fit["condition"]).lower()
        selected = matched.loc[
            (matched["subject"].astype(str) == str(fit["subject"]))
            & (matched["condition"].astype(str).str.lower() == fit_condition)
        ]
        if "post_convergence_eligible" in selected:
            selected = selected.loc[
                selected["post_convergence_eligible"].fillna(False).astype(bool)
            ]
        if selected.empty:
            continue
        values = {name: float(fit[name]) for name in MODEL_PARAMETERS[str(fit["model"])]}
        model = _build_model(str(fit["model"]), float(fit["t_dur_s"]), values)
        for record in selected.to_dict("records"):
            path = _matched_path(record["lead_path"])
            solution = model.solve(conditions={"lead_path": path})
            times = np.asarray(model.t_domain(), dtype=float)
            after = times >= float(record["convergence_jump"]) * TOKEN_INTERVAL_S
            correct_density = np.asarray(solution.pdf("correct"), dtype=float)
            error_density = np.asarray(solution.pdf("error"), dtype=float)
            total_density = correct_density + error_density
            total_mass = float(np.sum(total_density[after]) * model.dt)
            correct_mass = float(np.sum(correct_density[after]) * model.dt)
            mean_time = float(
                np.sum(times[after] * total_density[after]) * model.dt / total_mass
            ) if total_mass > 0 else float("nan")
            rows.append({
                "analysis": "ssm_matched_sequence_model_prediction",
                "subject": fit["subject"],
                "condition": fit_condition,
                "model": fit["model"],
                "pair_id": record["pair_id"],
                "early_bias": record["early_bias"],
                "convergence_jump": int(record["convergence_jump"]),
                "lead_path": json.dumps(list(path)),
                "n_trials": int(record["n_trials"]),
                "predicted_survival_after_convergence": total_mass,
                "predicted_accuracy_after_convergence": correct_mass / total_mass if total_mass > 0 else float("nan"),
                "predicted_mean_rt_after_convergence_s": mean_time,
            })
    return pd.DataFrame(rows)


def matched_sequence_model_statistics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize against-minus-for predictions on retained matched pairs."""
    if predictions.empty:
        return pd.DataFrame()
    rows = []
    for (condition, model), group in predictions.groupby(["condition", "model"], sort=True):
        pivot = group.pivot_table(
            index=["subject", "pair_id"], columns="early_bias",
            values=["predicted_accuracy_after_convergence", "predicted_mean_rt_after_convergence_s"],
            aggfunc="mean",
        )
        for measure in ("predicted_accuracy_after_convergence", "predicted_mean_rt_after_convergence_s"):
            if (measure, "for") not in pivot or (measure, "against") not in pivot:
                continue
            pair_differences = (pivot[(measure, "against")] - pivot[(measure, "for")]).dropna()
            differences = pair_differences.groupby(level="subject").mean()
            rows.append({
                "analysis": "ssm_matched_sequence_model",
                "condition": condition,
                "model": model,
                "measure": measure,
                "n_subjects": int(differences.index.nunique()),
                "n_pairs": int(len(pair_differences)),
                "mean_against_minus_for": float(differences.mean()) if len(differences) else float("nan"),
                **one_sample_statistics(differences),
            })
    return pd.DataFrame(rows)


def boundary_convergence_audit(fits: pd.DataFrame) -> pd.DataFrame:
    """Summarize convergence, boundary hits, and missing-information rates."""
    if fits.empty:
        return pd.DataFrame()
    rows = []
    for (condition, model), group in fits.groupby(["condition", "model"], sort=True):
        row = {
            "analysis": "ssm_boundary_convergence_audit",
            "condition": condition,
            "model": model,
            "n_cells": int(len(group)),
            "n_converged": int(group["converged"].fillna(False).astype(bool).sum()),
            "n_boundary_hit": int(group.get("boundary_hit", pd.Series(False, index=group.index)).fillna(False).astype(bool).sum()),
            "boundary_rate": float(group.get("boundary_hit", pd.Series(False, index=group.index)).fillna(False).astype(bool).mean()),
        }
        for parameter in MODEL_PARAMETERS.get(str(model), ()):
            se = pd.to_numeric(group.get(f"{parameter}_se", pd.Series(np.nan, index=group.index)), errors="coerce")
            row[f"{parameter}_se_missing"] = int(se.isna().sum())
            row[f"{parameter}_near_lower"] = int((pd.to_numeric(group.get(parameter, pd.Series(np.nan, index=group.index)), errors="coerce") - PARAMETER_RANGES[parameter][0]).abs().le(0.01 * (PARAMETER_RANGES[parameter][1] - PARAMETER_RANGES[parameter][0])).sum())
            row[f"{parameter}_near_upper"] = int((PARAMETER_RANGES[parameter][1] - pd.to_numeric(group.get(parameter, pd.Series(np.nan, index=group.index)), errors="coerce")).abs().le(0.01 * (PARAMETER_RANGES[parameter][1] - PARAMETER_RANGES[parameter][0])).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _synthetic_frame(
    model_name: str,
    values: dict[str, float],
    *,
    n_per_sequence: int,
    seed: int,
    lead_paths: tuple[tuple[int, ...], ...] | None = None,
) -> pd.DataFrame:
    """Generate a small synthetic frame from one known model.

    ``lead_paths`` are already in the correct-target frame.  Keeping this
    function ignorant of raw target labels prevents recovery from silently
    reorienting target-2 trials as if target 1 were correct.
    """
    import pyddm

    lead_paths = lead_paths or tuple(
        _lead_path(sequence, correct_target=1)
        for sequence in (
            "111211121112111",
            "121212121212121",
            "211211211221122",
            "222122212221222",
        )
    )
    if any(
        not isinstance(path, tuple)
        or not path
        or any(not isinstance(value, (int, np.integer)) for value in path)
        for path in lead_paths
    ):
        raise TypeError("lead_paths must be non-empty tuples of integer leads")
    model = _build_model(model_name, 4.0, values)
    rows = []
    for sequence_index, path in enumerate(lead_paths):
        solution = model.solve(conditions={"lead_path": path})
        sampled = solution.resample(n_per_sequence, seed=seed + sequence_index)
        for times, correct in ((sampled.choice_upper, True), (sampled.choice_lower, False)):
            for index, time in enumerate(times):
                rows.append({
                    "rt": float(time),
                    "correct": int(correct),
                    "lead_path": path,
                    "trial_class": "synthetic",
                    "trial_id": f"synthetic-{model_name}-{sequence_index}-{correct}-{index}",
                })
    return pd.DataFrame(rows)


def _recovery_truth_values(
    model_name: str,
    *,
    repetition: int,
    repetitions: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Construct deterministic interior, Latin-hypercube-like truths."""
    n_design = max(int(repetitions), 1)
    values: dict[str, float] = {}
    fractions: dict[str, float] = {}
    for parameter_index, parameter in enumerate(MODEL_PARAMETERS[model_name]):
        # A seeded permutation gives every parameter every interior stratum
        # exactly once when the production array covers all repetitions.  A
        # stable SHA seed avoids Python's process-randomized ``hash()`` and
        # keeps local/cluster truth points identical.
        permutation_seed = int.from_bytes(
            hashlib.sha256(
                f"{seed}:{model_name}:{parameter}".encode("utf-8")
            ).digest()[:8],
            "little",
        ) % (2**32)
        permutation = np.random.default_rng(permutation_seed).permutation(n_design)
        position = int(permutation[repetition])
        fraction = 0.2 + 0.6 * ((position + 0.5) / n_design)
        lower, upper = PARAMETER_RANGES[parameter]
        values[parameter] = float(lower + fraction * (upper - lower))
        fractions[parameter] = float(fraction)
    return values, fractions


def parameter_recovery(
    *,
    models: tuple[str, ...] = MECHANISTIC_MODELS,
    repetitions: int = 4,
    n_per_sequence: int = 40,
    n_starts: int = 2,
    seed: int = FIT_SEED,
    representative_paths: tuple[tuple[int, ...], ...] | None = None,
    repetition_indices: Sequence[int] | None = None,
    truth_design_repetitions: int | None = None,
) -> pd.DataFrame:
    """Recover known parameters from simulated trial paths.

    ``repetition_indices`` lets a Slurm array run one truth point at a time.
    ``truth_design_repetitions`` keeps the Latin-hypercube-like design fixed
    across those independent tasks, so task 7 of a 12-point array has exactly
    the same truth and seed as repetition 7 of a local pooled run.
    """
    design_repetitions = int(truth_design_repetitions or repetitions)
    if design_repetitions < 1:
        raise ValueError("truth_design_repetitions must be positive")
    indices = tuple(
        range(int(repetitions))
        if repetition_indices is None
        else (int(index) for index in repetition_indices)
    )
    if any(index < 0 or index >= design_repetitions for index in indices):
        raise ValueError("repetition_indices must lie within the truth design")
    rows = []
    for repetition in indices:
        for model_name in models:
            values, fractions = _recovery_truth_values(
                model_name,
                repetition=repetition,
                repetitions=design_repetitions,
                seed=seed,
            )
            simulation_seed = int(seed + repetition * 100)
            frame = _synthetic_frame(
                model_name,
                values,
                n_per_sequence=n_per_sequence,
                seed=simulation_seed,
                lead_paths=representative_paths,
            )
            estimate = _fit_one(model_name, frame, n_starts=n_starts, compute_uncertainty=False)
            for parameter in MODEL_PARAMETERS[model_name]:
                rows.append({
                    "analysis": "ssm_parameter_recovery",
                    "repetition": repetition,
                    "true_model": model_name,
                    "parameter": parameter,
                    "true_value": values[parameter],
                    "estimated_value": estimate.get(parameter, np.nan),
                    "truth_design_index": repetition,
                    "truth_design_repetitions": design_repetitions,
                    "truth_design_fraction": fractions[parameter],
                    "simulation_seed": simulation_seed,
                    "n_per_sequence": n_per_sequence,
                    "n_starts": n_starts,
                    "fit_seed": seed,
                    "converged": bool(estimate.get("optimizer_success", False)),
                    "boundary_hit": bool(estimate.get("boundary_hit", False)),
                    "optimizer_success": bool(estimate.get("optimizer_success", False)),
                    "best_start": int(estimate.get("best_start", -1)),
                    "start_objectives": estimate.get("start_objectives", "{}"),
                    "start_converged": estimate.get("start_converged", "{}"),
                    "fit_error": estimate.get("fit_error", ""),
                })
    return pd.DataFrame(rows)


def model_recovery(
    *,
    models: tuple[str, ...] = MECHANISTIC_MODELS,
    repetitions: int = 4,
    n_per_sequence: int = 40,
    n_starts: int = 2,
    seed: int = FIT_SEED,
    representative_paths: tuple[tuple[int, ...], ...] | None = None,
    repetition_indices: Sequence[int] | None = None,
    truth_design_repetitions: int | None = None,
) -> pd.DataFrame:
    """Fit every candidate to data simulated from every candidate model.

    See :func:`parameter_recovery` for the repetition-indexed array contract.
    """
    design_repetitions = int(truth_design_repetitions or repetitions)
    if design_repetitions < 1:
        raise ValueError("truth_design_repetitions must be positive")
    indices = tuple(
        range(int(repetitions))
        if repetition_indices is None
        else (int(index) for index in repetition_indices)
    )
    if any(index < 0 or index >= design_repetitions for index in indices):
        raise ValueError("repetition_indices must lie within the truth design")
    rows = []
    for repetition in indices:
        for true_model in models:
            true_values, truth_fractions = _recovery_truth_values(
                true_model,
                repetition=repetition,
                repetitions=design_repetitions,
                seed=seed,
            )
            simulation_seed = int(seed + repetition * 100)
            frame = _synthetic_frame(
                true_model,
                true_values,
                n_per_sequence=n_per_sequence,
                seed=simulation_seed,
                lead_paths=representative_paths,
            )
            for fitted_model in models:
                estimate = _fit_one(fitted_model, frame, n_starts=n_starts, compute_uncertainty=False)
                ll = float(estimate.get("log_likelihood", np.nan))
                rows.append({
                    "analysis": "ssm_model_recovery",
                    "repetition": repetition,
                    "true_model": true_model,
                    "fitted_model": fitted_model,
                    "truth_design_index": repetition,
                    "truth_design_repetitions": design_repetitions,
                    "truth_design_fraction": json.dumps(truth_fractions, sort_keys=True),
                    "simulation_seed": simulation_seed,
                    "n_per_sequence": n_per_sequence,
                    "n_starts": n_starts,
                    "fit_seed": seed,
                    "log_likelihood": ll,
                    "bic": (len(MODEL_PARAMETERS[fitted_model]) * np.log(len(frame)) - 2 * ll) if np.isfinite(ll) else np.nan,
                    "converged": bool(estimate.get("optimizer_success", False)),
                    "optimizer_success": bool(estimate.get("optimizer_success", False)),
                    "best_start": int(estimate.get("best_start", -1)),
                    "start_objectives": estimate.get("start_objectives", "{}"),
                    "start_converged": estimate.get("start_converged", "{}"),
                    "fit_error": estimate.get("fit_error", ""),
                })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["selected_model"] = result.groupby(["repetition", "true_model"])["bic"].transform(lambda values: result.loc[values.index, "fitted_model"].iloc[int(np.nanargmin(values.to_numpy(dtype=float)))] if np.isfinite(values).any() else "")
    result["recovery_correct"] = result["selected_model"] == result["true_model"]
    return result


def parameter_recovery_statistics(recovery: pd.DataFrame) -> pd.DataFrame:
    """Summarize parameter-recovery bias, error, and fit diagnostics."""
    required = {
        "true_model", "parameter", "true_value", "estimated_value",
        "converged", "boundary_hit",
    }
    require_columns(recovery, sorted(required))
    rows = []
    for (true_model, parameter), group in recovery.groupby(
        ["true_model", "parameter"], sort=True
    ):
        true_value = pd.to_numeric(group["true_value"], errors="coerce")
        estimated = pd.to_numeric(group["estimated_value"], errors="coerce")
        finite = true_value.notna() & estimated.notna()
        errors = (estimated[finite] - true_value[finite]).to_numpy(dtype=float)
        if errors.size:
            correlation = (
                float(np.corrcoef(
                    true_value[finite].to_numpy(dtype=float),
                    estimated[finite].to_numpy(dtype=float),
                )[0, 1])
                if errors.size > 1
                and np.std(true_value[finite]) > 0
                and np.std(estimated[finite]) > 0
                else float("nan")
            )
            bias = float(np.mean(errors))
            rmse = float(np.sqrt(np.mean(errors ** 2)))
        else:
            correlation = bias = rmse = float("nan")
        rows.append({
            "analysis": "ssm_parameter_recovery_summary",
            "true_model": true_model,
            "parameter": parameter,
            "n_recovery_cases": int(len(group)),
            "n_finite_estimates": int(finite.sum()),
            "bias": bias,
            "rmse": rmse,
            "correlation": correlation,
            "convergence_rate": float(
                group["converged"].fillna(False).astype(bool).mean()
            ),
            "boundary_rate": float(
                group["boundary_hit"].fillna(False).astype(bool).mean()
            ),
        })
    return pd.DataFrame(rows)


def model_recovery_confusion(recovery: pd.DataFrame) -> pd.DataFrame:
    """Return long-form selected-model counts/rates by true model."""
    require_columns(recovery, ["repetition", "true_model", "selected_model"])
    if recovery.empty:
        return pd.DataFrame()
    selected = recovery.drop_duplicates(
        ["repetition", "true_model"]
    )[["repetition", "true_model", "selected_model"]]
    true_models = sorted(set(recovery["true_model"].astype(str)))
    fitted_models = sorted(
        set(recovery["selected_model"].dropna().astype(str))
        | set(MECHANISTIC_MODELS)
    )
    rows = []
    for true_model in true_models:
        group = selected.loc[selected["true_model"].astype(str) == true_model]
        n_repetitions = int(len(group))
        counts = group["selected_model"].astype(str).value_counts()
        for fitted_model in fitted_models:
            count = int(counts.get(fitted_model, 0))
            rows.append({
                "analysis": "ssm_model_recovery_confusion",
                "true_model": true_model,
                "selected_model": fitted_model,
                "n_repetitions": n_repetitions,
                "n_selected": count,
                "selection_rate": float(count / n_repetitions)
                if n_repetitions else float("nan"),
            })
    return pd.DataFrame(rows)


def _integrate_timecourse_density(times: np.ndarray, density: np.ndarray) -> float:
    """Integrate a downsampled time-course density in seconds."""
    finite = np.isfinite(times) & np.isfinite(density)
    times = np.asarray(times[finite], dtype=float)
    density = np.clip(np.asarray(density[finite], dtype=float), 0.0, None)
    if not len(times):
        return float("nan")
    order = np.argsort(times)
    times = times[order]
    density = density[order]
    unique, inverse = np.unique(times, return_inverse=True)
    if len(unique) != len(times):
        density = np.asarray(
            [density[inverse == index].mean() for index in range(len(unique))],
            dtype=float,
        )
        times = unique
    if len(times) == 1:
        return float(density[0] * TIME_COURSE_STEP_S)
    return float(np.sum(0.5 * (density[:-1] + density[1:]) * np.diff(times)))


def _timecourse_density_quantiles(
    times: np.ndarray,
    density: np.ndarray,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> tuple[float, dict[float, float]]:
    """Return mass and outcome-conditional quantiles from time-course rows."""
    finite = np.isfinite(times) & np.isfinite(density)
    times = np.asarray(times[finite], dtype=float)
    density = np.clip(np.asarray(density[finite], dtype=float), 0.0, None)
    if not len(times):
        return float("nan"), {quantile: float("nan") for quantile in quantiles}
    order = np.argsort(times)
    times = times[order]
    density = density[order]
    unique, inverse = np.unique(times, return_inverse=True)
    if len(unique) != len(times):
        density = np.asarray(
            [density[inverse == index].mean() for index in range(len(unique))],
            dtype=float,
        )
        times = unique
    mass = _integrate_timecourse_density(times, density)
    if not np.isfinite(mass) or mass <= 0:
        return mass, {quantile: float("nan") for quantile in quantiles}
    if len(times) == 1:
        cumulative = np.asarray([mass], dtype=float)
    else:
        increments = 0.5 * (density[:-1] + density[1:]) * np.diff(times)
        cumulative = np.concatenate(([0.0], np.cumsum(increments)))
        cumulative[-1] = mass
    return mass, {
        quantile: float(np.interp(quantile * mass, cumulative, times))
        for quantile in quantiles
    }


def _distribution_metrics_from_timecourse(course: pd.DataFrame) -> dict[str, object]:
    """Compute masses and conditional quantiles from one time-course cell."""
    grouped = (
        course.assign(_time=pd.to_numeric(course["time_s"], errors="coerce"))
        .groupby("_time", as_index=False)[
            ["predicted_density_correct", "predicted_density_error"]
        ]
        .mean()
        .dropna(subset=["_time"])
    )
    times = grouped["_time"].to_numpy(dtype=float)
    correct_mass, correct_quantiles = _timecourse_density_quantiles(
        times,
        pd.to_numeric(grouped["predicted_density_correct"], errors="coerce").to_numpy(dtype=float),
    )
    error_mass, error_quantiles = _timecourse_density_quantiles(
        times,
        pd.to_numeric(grouped["predicted_density_error"], errors="coerce").to_numpy(dtype=float),
    )
    return {
        "predicted_correct_mass": correct_mass,
        "predicted_error_mass": error_mass,
        "predicted_decision_mass": (
            float(correct_mass + error_mass)
            if np.isfinite(correct_mass) and np.isfinite(error_mass)
            else float("nan")
        ),
        # This preserves the existing PyDDM convention: ``prob('correct')``
        # is the probability of a correct first passage, rather than a
        # renormalization conditional on a decision within the finite horizon.
        "predicted_accuracy": correct_mass,
        "predicted_quantiles": {
            "correct": correct_quantiles,
            "error": error_quantiles,
        },
    }


def fitted_distribution_checks(
    features: pd.DataFrame,
    fits: pd.DataFrame,
    *,
    timecourse: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compare observed and fitted correct/error RT quantiles by cell/class.

    ``features`` retain their acquisition labels (``Fast``/``Slow``), while
    fitted tables use normalized labels (``all``/``fast``/``slow``).  The
    shared subject/condition grouping helper is therefore used for both the
    pooled and condition-specific frames.  When the primary ``ssmtimecourse``
    derivative is supplied, its already-computed density rows are integrated
    and no PyDDM paths are solved again.  Raw decision-time quantiles always
    come directly from the trial features.  Without ``timecourse`` the legacy
    model-solving fallback remains available for small standalone uses.
    """
    trials = task_trials(features)
    require_columns(trials, ["subject", "condition", "isCorrect", "dt_ms"])
    frames = {
        (str(subject), str(condition).lower()): _decision_frame(selected)
        for subject, condition, selected in _subject_condition_groups(trials)
    }
    timecourse_lookup: dict[tuple[str, str, str, str], pd.DataFrame] = {}
    if timecourse is not None:
        require_columns(
            timecourse,
            [
                "subject", "condition", "model", "trial_class", "time_s",
                "predicted_density_correct", "predicted_density_error",
            ],
        )
        normalized = timecourse.copy()
        normalized["_subject"] = normalized["subject"].astype(str)
        normalized["_condition"] = normalized["condition"].astype(str).str.lower()
        normalized["_model"] = normalized["model"].astype(str)
        normalized["_trial_class"] = normalized["trial_class"].astype(str)
        timecourse_lookup = {
            (str(subject), str(condition), str(model), str(trial_class)): group
            for (subject, condition, model, trial_class), group in normalized.groupby(
                ["_subject", "_condition", "_model", "_trial_class"], sort=False
            )
        }

    rows = []
    quantiles = (0.1, 0.5, 0.9)
    for fit in fits.loc[fits["converged"].fillna(False).astype(bool)].to_dict("records"):
        subject = str(fit["subject"])
        condition = str(fit["condition"]).lower()
        frame = frames.get((subject, condition))
        if frame is None or frame.empty:
            continue
        model_name = str(fit["model"])
        model = None
        solutions = {}
        if timecourse is None:
            values = {name: float(fit[name]) for name in MODEL_PARAMETERS[model_name]}
            model = _build_model(model_name, float(fit["t_dur_s"]), values)
        for trial_class, group in _class_groups(frame):
            if group.empty:
                continue
            source = "ssmtimecourse" if timecourse is not None else "model_solve"
            metrics = None
            if timecourse is not None:
                course = timecourse_lookup.get(
                    (subject, condition, model_name, str(trial_class))
                )
                if course is None or course.empty:
                    continue
                metrics = _distribution_metrics_from_timecourse(course)
            else:
                weights = group["lead_path"].value_counts(normalize=True).to_dict()
                solutions = {
                    path: model.solve(conditions={"lead_path": path})
                    for path in weights
                }
                correct_mass = float(
                    sum(weight * solutions[path].prob("correct") for path, weight in weights.items())
                )
                error_mass = float(
                    sum(weight * solutions[path].prob("error") for path, weight in weights.items())
                )
                metrics = {
                    "predicted_correct_mass": correct_mass,
                    "predicted_error_mass": error_mass,
                    "predicted_decision_mass": correct_mass + error_mass,
                    "predicted_accuracy": correct_mass,
                    "predicted_quantiles": {
                        "correct": _predicted_quantiles(model, weights, quantiles, choice="correct"),
                        "error": _predicted_quantiles(model, weights, quantiles, choice="error"),
                    },
                }
            for outcome, code in (("correct", 1), ("error", 0)):
                observed = group.loc[group["correct"] == code, "rt"].to_numpy(dtype=float)
                observed = observed[np.isfinite(observed)]
                for quantile in quantiles:
                    rows.append({
                        "analysis": "ssm_fitted_distribution_check",
                        "subject": subject,
                        "condition": condition,
                        "model": model_name,
                        "trial_class": trial_class,
                        "outcome": outcome,
                        "quantile": quantile,
                        "n_observed": int(len(observed)),
                        "observed_rt_s": (
                            float(np.quantile(observed, quantile))
                            if len(observed) else float("nan")
                        ),
                        "predicted_rt_s": metrics["predicted_quantiles"][outcome][quantile],
                        "predicted_correct_mass": metrics["predicted_correct_mass"],
                        "predicted_error_mass": metrics["predicted_error_mass"],
                        "predicted_decision_mass": metrics["predicted_decision_mass"],
                        "predicted_accuracy": metrics["predicted_accuracy"],
                        "prediction_source": source,
                    })
    return pd.DataFrame(rows)


def exclusion_robustness_audit(features: pd.DataFrame) -> pd.DataFrame:
    """Report retention for complete-log and canonical-alignment sensitivities.

    The primary analysis eligibility is never replaced silently.  These two
    views are explicit sensitivity populations for a later refit: a complete
    15-row token log, and the canonical design-time alignment flag.
    """
    trials = task_trials(features)
    require_columns(trials, ["subject", "condition", "token_log_rows"])
    complete = pd.to_numeric(trials["token_log_rows"], errors="coerce").eq(15)
    if "design_time_alignment_valid" in trials:
        alignment = trials["design_time_alignment_valid"].fillna(False).astype(bool)
    else:
        alignment = pd.Series(False, index=trials.index)
    views = {
        "complete_token_log": complete,
        "canonical_alignment_valid": alignment,
    }
    rows = []
    for rule, mask in views.items():
        retained = trials.loc[mask]
        cells = retained[["subject", "condition"]].drop_duplicates()
        counts = retained.groupby(["subject", "condition"]).size()
        rows.append({
            "analysis": "ssm_exclusion_robustness_audit",
            "rule": rule,
            "criterion": (
                "token_log_rows==15"
                if rule == "complete_token_log"
                else "design_time_alignment_valid==True"
            ),
            "n_primary_task_trials": int(len(trials)),
            "n_retained_trials": int(len(retained)),
            "retention_fraction": float(len(retained) / len(trials))
            if len(trials) else float("nan"),
            "n_subjects_retained": int(retained["subject"].nunique()),
            "n_subject_condition_cells": int(len(cells)),
            "n_cells_below_minimum": int((counts < MINIMUM_FIT_TRIALS).sum()),
            "minimum_fit_trials": MINIMUM_FIT_TRIALS,
        })
    symmetric_difference = complete ^ alignment
    rows.append({
        "analysis": "ssm_exclusion_robustness_audit",
        "rule": "complete_token_log_alignment_equivalence",
        "criterion": "token_log_rows==15 versus design_time_alignment_valid",
        "n_primary_task_trials": int(len(trials)),
        "n_retained_trials": int((complete | alignment).sum()),
        "retention_fraction": float((complete | alignment).mean()) if len(trials) else float("nan"),
        "n_subjects_retained": int(trials.loc[complete | alignment, "subject"].nunique()),
        "n_subject_condition_cells": int(
            trials.loc[complete | alignment, ["subject", "condition"]].drop_duplicates().shape[0]
        ),
        "n_cells_below_minimum": int(
            trials.loc[complete | alignment].groupby(["subject", "condition"]).size().lt(MINIMUM_FIT_TRIALS).sum()
        ),
        "minimum_fit_trials": MINIMUM_FIT_TRIALS,
        "n_complete_token_log": int(complete.sum()),
        "n_alignment_valid": int(alignment.sum()),
        "n_intersection": int((complete & alignment).sum()),
        "n_symmetric_difference": int(symmetric_difference.sum()),
        "equivalent": bool(not symmetric_difference.any()),
    })
    return pd.DataFrame(rows)


def influential_subject_sensitivity(
    fits: pd.DataFrame,
    *,
    excluded_subject: str = "H20",
) -> pd.DataFrame:
    """Recompute group model contrasts after excluding one fitted subject."""
    require_columns(fits, ["subject", "condition", "model"])
    filtered = fits.loc[fits["subject"].astype(str) != str(excluded_subject)].copy()
    result = mechanistic_model_statistics(filtered)
    if result.empty:
        return result
    result.insert(1, "sensitivity_excluded_subject", str(excluded_subject))
    result.insert(2, "n_subjects_retained", int(filtered["subject"].nunique()))
    result.insert(3, "analysis_scope", "already_fit_subject_results_no_refit")
    return result


def robustness_grid(
    features: pd.DataFrame,
    *,
    n_jobs: int = 1,
    n_starts: int = 1,
    configurations: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Run the prespecified bounds/tau/grid/overshoot robustness refits.

    ``configurations`` is optional for backwards-compatible local use.  The
    cluster workflow supplies one configuration per subject-array task.  The
    returned table retains every subject × condition × model fit row and its
    active parameter bounds; group summaries are derived separately.
    """
    global FILTER_TAU_S, SOLVER_STEP_S, EVIDENCE_AFTER_LAST
    baseline_tau, baseline_step, baseline_after = FILTER_TAU_S, SOLVER_STEP_S, EVIDENCE_AFTER_LAST
    baseline_ranges = dict(PARAMETER_RANGES)
    configuration_specs = (
        ("baseline", 0.2, 0.01, "hold", 1.0),
        ("tau_100ms", 0.1, 0.01, "hold", 1.0),
        ("tau_300ms", 0.3, 0.01, "hold", 1.0),
        ("solver_20ms", 0.2, 0.02, "hold", 1.0),
        # This is evidence-horizon handling, not boundary overshoot. PyDDM's
        # absorbing boundary solver handles crossings numerically at dx; no
        # separate overshoot correction is applied.
        ("post_horizon_evidence_zero", 0.2, 0.01, "zero", 1.0),
        ("expanded_bounds", 0.2, 0.01, "hold", 2.0),
    )
    wanted = set(configurations or ROBUSTNESS_CONFIGURATIONS)
    unknown = wanted.difference(name for name, *_ in configuration_specs)
    if unknown:
        raise ValueError(f"unknown robustness configurations: {sorted(unknown)}")
    rows = []
    try:
        for name, tau, step, after_last, bound_multiplier in configuration_specs:
            if name not in wanted:
                continue
            FILTER_TAU_S, SOLVER_STEP_S, EVIDENCE_AFTER_LAST = tau, step, after_last
            PARAMETER_RANGES.update(baseline_ranges)
            for parameter, (lower, upper) in baseline_ranges.items():
                if name == "expanded_bounds" and parameter not in {"urgency_scale", "urgency_onset_s", "collapse_rate"}:
                    continue
                PARAMETER_RANGES[parameter] = (lower, upper * bound_multiplier)
            fits = fit_mechanistic_model_set(
                features,
                n_jobs=n_jobs,
                n_starts=n_starts,
                compute_uncertainty=False,
            )
            fits = fits.copy()
            fits.insert(0, "analysis", "ssm_robustness_grid")
            fits.insert(1, "configuration", name)
            fits["filter_tau_s"] = tau
            fits["solver_step_s"] = step
            fits["evidence_after_last"] = after_last
            fits["bound_multiplier"] = bound_multiplier
            fits["parameter_ranges"] = json.dumps(
                {parameter: list(bounds) for parameter, bounds in PARAMETER_RANGES.items()},
                sort_keys=True,
            )
            for parameter, (lower, upper) in PARAMETER_RANGES.items():
                fits[f"{parameter}_lower_bound"] = lower
                fits[f"{parameter}_upper_bound"] = upper
            fits["boundary_crossing"] = "absorbing_numerical_grid"
            fits["overshoot_correction"] = "none"
            rows.append(fits)
    finally:
        FILTER_TAU_S, SOLVER_STEP_S, EVIDENCE_AFTER_LAST = baseline_tau, baseline_step, baseline_after
        PARAMETER_RANGES.clear(); PARAMETER_RANGES.update(baseline_ranges)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def robustness_statistics(
    robustness: pd.DataFrame,
    *,
    baseline_configuration: str = "baseline",
) -> pd.DataFrame:
    """Summarize full robustness rows and paired changes from baseline."""
    required = {
        "configuration", "subject", "condition", "model", "delta_bic",
        "converged", "boundary_hit",
    }
    require_columns(robustness, sorted(required))
    if robustness.empty:
        return pd.DataFrame()
    rows = []
    for (configuration, condition, model), group in robustness.groupby(
        ["configuration", "condition", "model"], sort=True
    ):
        values = pd.to_numeric(group["delta_bic"], errors="coerce")
        finite = values.replace([np.inf, -np.inf], np.nan).dropna()
        targeted_near_upper = {}
        targeted_flags = []
        for parameter in ("urgency_scale", "urgency_onset_s", "collapse_rate"):
            values_parameter = pd.to_numeric(
                group.get(parameter, pd.Series(np.nan, index=group.index)),
                errors="coerce",
            )
            upper = pd.to_numeric(
                group.get(
                    f"{parameter}_upper_bound",
                    pd.Series(PARAMETER_RANGES[parameter][1], index=group.index),
                ),
                errors="coerce",
            )
            lower = pd.to_numeric(
                group.get(
                    f"{parameter}_lower_bound",
                    pd.Series(PARAMETER_RANGES[parameter][0], index=group.index),
                ),
                errors="coerce",
            )
            near_upper = (upper - values_parameter).le(
                0.01 * (upper - lower)
            ) & values_parameter.notna()
            targeted_near_upper[parameter] = int(near_upper.sum())
            targeted_flags.append(near_upper)
        ndt = pd.to_numeric(
            group.get("nondecision_s", pd.Series(np.nan, index=group.index)),
            errors="coerce",
        )
        ndt_lower = pd.to_numeric(
            group.get(
                "nondecision_s_lower_bound",
                pd.Series(PARAMETER_RANGES["nondecision_s"][0], index=group.index),
            ),
            errors="coerce",
        )
        ndt_upper = pd.to_numeric(
            group.get(
                "nondecision_s_upper_bound",
                pd.Series(PARAMETER_RANGES["nondecision_s"][1], index=group.index),
            ),
            errors="coerce",
        )
        ndt_near_lower = (ndt - ndt_lower).le(
            0.01 * (ndt_upper - ndt_lower)
        ) & ndt.notna()
        rows.append({
            "analysis": "ssm_robustness_summary",
            "configuration": configuration,
            "condition": condition,
            "model": model,
            "criterion": "delta_bic",
            "n_subjects": int(group["subject"].nunique()),
            "n_converged": int(group["converged"].fillna(False).astype(bool).sum()),
            "convergence_rate": float(group["converged"].fillna(False).astype(bool).mean()),
            "n_boundary_hit": int(group["boundary_hit"].fillna(False).astype(bool).sum()),
            "boundary_rate": float(group["boundary_hit"].fillna(False).astype(bool).mean()),
            **{f"{parameter}_near_upper": count for parameter, count in targeted_near_upper.items()},
            "n_targeted_near_upper": int(
                np.logical_or.reduce([flag.to_numpy(dtype=bool) for flag in targeted_flags]).sum()
            ) if targeted_flags else 0,
            "nondecision_s_near_lower": int(ndt_near_lower.sum()),
            "n_subjects_favoring_candidate": int((finite < 0).sum()),
            "n_subjects_favoring_ddm": int((finite > 0).sum()),
            **one_sample_statistics(finite),
        })
    baseline = robustness.loc[
        robustness["configuration"] == baseline_configuration,
        ["subject", "condition", "model", "delta_bic"],
    ].rename(columns={"delta_bic": "baseline_delta_bic"})
    alternatives = robustness.loc[
        robustness["configuration"] != baseline_configuration,
        ["configuration", "subject", "condition", "model", "delta_bic"],
    ]
    paired = alternatives.merge(
        baseline, on=["subject", "condition", "model"], how="inner"
    )
    for (configuration, condition, model), group in paired.groupby(
        ["configuration", "condition", "model"], sort=True
    ):
        differences = (
            pd.to_numeric(group["delta_bic"], errors="coerce")
            - pd.to_numeric(group["baseline_delta_bic"], errors="coerce")
        ).replace([np.inf, -np.inf], np.nan).dropna()
        rows.append({
            "analysis": "ssm_robustness_paired_sensitivity",
            "configuration": configuration,
            "condition": condition,
            "model": model,
            "criterion": "delta_bic_change_vs_baseline",
            "n_subjects": int(len(differences)),
            "n_converged": np.nan,
            "convergence_rate": np.nan,
            "n_boundary_hit": np.nan,
            "boundary_rate": np.nan,
            "n_near_upper_or_boundary": np.nan,
            "n_subjects_favoring_candidate": int((differences < 0).sum()),
            "n_subjects_favoring_ddm": int((differences > 0).sum()),
            **one_sample_statistics(differences),
        })
    return pd.DataFrame(rows)


def exclusion_robustness_statistics(
    strict_fits: pd.DataFrame,
    primary_fits: pd.DataFrame,
    *,
    audit: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compare strict complete-log/alignment fits with primary fits."""
    required = {"subject", "condition", "model", "delta_bic", "converged", "boundary_hit"}
    require_columns(strict_fits, sorted(required))
    require_columns(primary_fits, sorted(required))
    merged = strict_fits.merge(
        primary_fits[["subject", "condition", "model", "delta_bic"]].rename(
            columns={"delta_bic": "primary_delta_bic"}
        ),
        on=["subject", "condition", "model"], how="inner",
    )
    rows = []
    for (condition, model), group in merged.groupby(["condition", "model"], sort=True):
        if model == "ddm":
            continue
        differences = (
            pd.to_numeric(group["delta_bic"], errors="coerce")
            - pd.to_numeric(group["primary_delta_bic"], errors="coerce")
        ).replace([np.inf, -np.inf], np.nan).dropna()
        row = {
            "analysis": "ssm_exclusion_robustness_summary",
            "rule": "complete_token_log_alignment",
            "condition": condition,
            "model": model,
            "criterion": "strict_delta_bic_minus_primary_delta_bic",
            "n_subjects": int(len(differences)),
            "n_converged": int(group["converged"].fillna(False).astype(bool).sum()),
            "convergence_rate": float(group["converged"].fillna(False).astype(bool).mean()),
            "n_boundary_hit": int(group["boundary_hit"].fillna(False).astype(bool).sum()),
            "boundary_rate": float(group["boundary_hit"].fillna(False).astype(bool).mean()),
            "n_subjects_favoring_strict": int((differences < 0).sum()),
            "n_subjects_favoring_primary": int((differences > 0).sum()),
            **one_sample_statistics(differences),
        }
        if audit is not None and not audit.empty:
            equivalence = audit.loc[
                audit["rule"] == "complete_token_log_alignment_equivalence"
            ]
            if len(equivalence):
                census = equivalence.iloc[0]
                row.update({
                    "mask_equivalent": bool(census.get("equivalent", False)),
                    "n_complete_token_log": census.get("n_complete_token_log", np.nan),
                    "n_alignment_valid": census.get("n_alignment_valid", np.nan),
                    "n_mask_intersection": census.get("n_intersection", np.nan),
                    "n_mask_symmetric_difference": census.get("n_symmetric_difference", np.nan),
                })
        rows.append(row)
    return pd.DataFrame(rows)


def urgency_condition_contrast(fits: pd.DataFrame) -> pd.DataFrame:
    """Test whether fitted urgency growth differs between Fast and Slow blocks.

    Cisek's account makes urgency the mechanism that carries time pressure, so
    the speed instruction should change how fast urgency grows. Cisek et al.
    2009 found exactly that in this task with a model-free proxy: the slope and
    intercept of evidence at decision against decision time both differed
    between fast and slow blocks.

    Parameters
    ----------
    fits
        Output of :func:`fit_sequential_sampling_models`.

    Returns
    -------
    pandas.DataFrame
        One row per urgency-model parameter, holding the paired Fast-minus-Slow
        contrast across subjects.

    Notes
    -----
    ``urgency_scale`` is the threshold over the urgency slope, so a *negative*
    Fast-minus-Slow difference is *faster*-rising urgency under time pressure.
    Every parameter is contrasted, not only the urgency ones: a shift in drift
    scale or non-decision time instead is the alternative account of the same
    speed-accuracy change, and it is only visible if reported. Subjects missing
    either condition's fit are dropped pairwise by the shared inference helper.
    """
    urgency = fits.loc[fits["model"] == "urgency"].set_index("subject")
    fast = urgency.loc[urgency["condition"] == "fast"]
    slow = urgency.loc[urgency["condition"] == "slow"]
    paired = fast.index.intersection(slow.index)
    rows = []
    for parameter in MODEL_PARAMETERS["urgency"]:
        rows.append(
            {
                "analysis": "ssm_urgency_condition_contrast",
                "parameter": parameter,
                "test": "paired_fast_minus_slow",
                **paired_subject_statistics(
                    fast.loc[paired, parameter], slow.loc[paired, parameter]
                ),
            }
        )
    return pd.DataFrame(rows)


def fitted_predictions(
    features: pd.DataFrame,
    fits: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate each fitted model, for plotting and for reuse as a regressor.

    Parameters
    ----------
    features
        Canonical trial-feature table, the same one the models were fitted to.
    fits
        Output of :func:`fit_sequential_sampling_models`, including
        ``t_dur_s``.

    Returns
    -------
    time_courses
        One row per subject, condition, model, trial class, and time point,
        holding the decision criterion, the mean decision variable, and the
        predicted and observed decision-time densities for each response.
    trial_predictions
        One row per trial and model, holding the criterion and decision
        variable the fitted model puts at that trial's decision time, and the
        accuracy it predicts for that trial's token sequence.

    Notes
    -----
    Both are evaluated once while the fitted model is in hand, because
    refitting to draw a curve costs half an hour per subject. Time courses are
    resolved by trial class because that is where the models disagree and where
    the monkey work plots them: pooling classes averages a misleading trial's
    negative early evidence against an easy trial's positive evidence and
    reports their cancellation. The criterion does not depend on the class and
    repeats across them, as do the observed densities across models, so that
    one row is a complete plotting record. ``trial_predictions`` carries the
    canonical trial key, which is what makes the model-derived criterion usable
    as a regressor joined against source-space features. Cells that were not
    fitted contribute no rows.
    """
    trials = task_trials(features)
    frames = {
        (subject, condition): _decision_frame(selected)
        for subject, condition, selected in _subject_condition_groups(trials)
    }
    time_course_rows = []
    trial_rows = []
    for fit in fits.loc[fits["converged"].astype(bool)].to_dict("records"):
        frame = frames.get((fit["subject"], fit["condition"]))
        if frame is None or frame.empty:
            continue
        courses, predictions = _cell_predictions(fit, frame)
        time_course_rows.extend(courses)
        trial_rows.extend(predictions)
    return pd.DataFrame(time_course_rows), pd.DataFrame(trial_rows)


def _cell_predictions(fit: dict, frame: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """Evaluate one fitted subject, condition and model over its own trials."""
    model_name = str(fit["model"])
    values = {name: float(fit[name]) for name in MODEL_PARAMETERS[model_name]}
    model = _build_model(model_name, float(fit["t_dur_s"]), values)
    times = np.asarray(model.t_domain(), dtype=float)
    bound = model.get_dependence("bound")
    criterion = np.array(
        [float(bound.get_bound(t=float(time), conditions={})) for time in times]
    )

    solutions = {}
    trajectories = {}
    for path in frame["lead_path"].unique():
        solutions[path] = model.solve(conditions={"lead_path": path})
        trajectories[path] = _mean_decision_variable(model_name, values, path, times)

    stride = max(1, int(round(TIME_COURSE_STEP_S / SOLVER_STEP_S)))
    edges = np.append(times, times[-1] + SOLVER_STEP_S)
    keys = {"subject": fit["subject"], "condition": fit["condition"], "model": model_name}
    course_rows = []
    for trial_class, group in _class_groups(frame):
        weights = group["lead_path"].value_counts() / len(group)
        predicted = {
            choice: sum(
                weight * np.asarray(solutions[path].pdf(choice))
                for path, weight in weights.items()
            )
            for choice in ("correct", "error")
        }
        decision_variable = sum(
            weight * trajectories[path] for path, weight in weights.items()
        )
        observed = {}
        for choice, wanted in (("correct", 1), ("error", 0)):
            counted, _ = np.histogram(
                group.loc[group["correct"] == wanted, "rt"].to_numpy(dtype=float),
                bins=edges,
            )
            observed[choice] = counted / (len(group) * SOLVER_STEP_S)
        for index in range(0, times.size, stride):
            course_rows.append(
                {
                    **keys,
                    "trial_class": trial_class,
                    "n_trials": int(len(group)),
                    "time_s": float(times[index]),
                    "criterion": float(criterion[index]),
                    "mean_decision_variable": float(decision_variable[index]),
                    "predicted_density_correct": float(predicted["correct"][index]),
                    "predicted_density_error": float(predicted["error"][index]),
                    "observed_density_correct": float(observed["correct"][index]),
                    "observed_density_error": float(observed["error"][index]),
                }
            )

    trial_rows = []
    for trial in frame.to_dict("records"):
        index = min(int(trial["rt"] / SOLVER_STEP_S), times.size - 1)
        trial_rows.append(
            {
                "trial_id": trial["trial_id"],
                **keys,
                "trial_class": trial["trial_class"],
                "dt_ms": float(trial["rt"] * 1000.0),
                "criterion_at_decision": float(criterion[index]),
                "decision_variable_at_decision": float(
                    trajectories[trial["lead_path"]][index]
                ),
                "predicted_accuracy": float(
                    solutions[trial["lead_path"]].prob("correct")
                ),
            }
        )
    return course_rows, trial_rows


def _class_groups(frame: pd.DataFrame):
    """Yield the pooled frame and then each declared trial class within it."""
    yield "all", frame
    for trial_class in sorted(frame["trial_class"].unique()):
        yield trial_class, frame.loc[frame["trial_class"] == trial_class]


def _mean_decision_variable(
    model: str,
    values: dict[str, float],
    lead_path: tuple[int, ...],
    times: np.ndarray,
) -> np.ndarray:
    """Integrate one model's noise-free decision variable over the time grid.

    Notes
    -----
    Forward Euler on the same step the solver uses. Without noise the
    integrator's trajectory is the running integral of the evidence and the
    urgency model's is the low-pass filter of it, which is the contrast the
    trajectory figure exists to show.
    """
    trajectory = np.empty(times.size)
    state = 0.0
    for index, time in enumerate(times):
        evidence = values["drift_scale"] * _evidence(float(time), lead_path)
        if model in ("ddm", "collapsing"):
            rate = evidence
        elif model == "urgency":
            rate = (evidence - state) / FILTER_TAU_S
        else:
            rate = (evidence - state) / FILTER_TAU_S + _additive_drive(
                float(time), lead_path, values["additive_scale"]
            )
        state += rate * SOLVER_STEP_S
        trajectory[index] = state
    return trajectory


def _population_normal(
    estimates: np.ndarray,
    errors: np.ndarray,
) -> tuple[float, float, float]:
    """Fit a normal population distribution to estimates with known errors.

    Parameters
    ----------
    estimates
        Subject-level maximum-likelihood values.
    errors
        Their standard errors, one per estimate.

    Returns
    -------
    population_mean
        Precision-weighted population mean.
    population_mean_error
        Standard error of that mean.
    between_subject_sd
        Population standard deviation net of estimation error.

    Notes
    -----
    Each estimate is treated as its subject's true value plus independent
    normal estimation error, giving the marginal variance
    ``between_subject_sd ** 2 + error ** 2``. The mean has a closed form given
    the population standard deviation, so only the latter is optimized. A
    population standard deviation of zero is a genuine solution and means the
    spread between subjects is no wider than their estimation error.
    """
    if estimates.size < 2:
        return float("nan"), float("nan"), float("nan")
    variances = errors**2

    def negative_log_marginal(between_sd: float) -> float:
        total = variances + between_sd**2
        weights = 1.0 / total
        mean = float(np.sum(weights * estimates) / np.sum(weights))
        return 0.5 * float(np.sum(np.log(total) + (estimates - mean) ** 2 / total))

    result = optimize.minimize_scalar(
        negative_log_marginal,
        bounds=(0.0, 10.0 * float(np.std(estimates, ddof=1)) + 1.0),
        method="bounded",
    )
    between_sd = float(result.x)
    weights = 1.0 / (variances + between_sd**2)
    mean = float(np.sum(weights * estimates) / np.sum(weights))
    return mean, float(np.sqrt(1.0 / np.sum(weights))), between_sd


def population_parameters(
    fits: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pool the subject-level fits under a normal population model.

    Parameters
    ----------
    fits
        Output of :func:`fit_sequential_sampling_models`.

    Returns
    -------
    subject_estimates
        One row per subject, condition, model, and parameter, holding the
        subject's own estimate, its standard error, the population-informed
        estimate, and the weight the subject's own data carry in it.
    population
        One row per condition, model, and parameter, holding the population
        mean, its standard error, the between-subject standard deviation, and
        a test of the mean against zero.

    Notes
    -----
    Every subject's estimate is shrunk toward the population mean in proportion
    to its own uncertainty, so a poorly constrained subject moves further than
    a well constrained one. Fitting the population by maximum marginal
    likelihood over the subject-level estimates and their standard errors is
    empirical Bayes, not a hierarchical posterior: it conditions on the
    estimation errors rather than propagating their own uncertainty, which a
    joint sampler would do. Subjects whose cell was not fitted, or whose
    standard error is missing, are excluded from both tables. The population
    mean is tested against zero, which is a meaningful null only for the
    urgency rate: every parameter is constrained non-negative, so the other
    tests restate that constraint.
    """
    subject_rows = []
    population_rows = []
    for (condition, model), group in fits.groupby(["condition", "model"], sort=True):
        for parameter in MODEL_PARAMETERS[model]:
            estimates = pd.to_numeric(group[parameter], errors="coerce").to_numpy(
                dtype=float
            )
            errors = pd.to_numeric(group[f"{parameter}_se"], errors="coerce").to_numpy(
                dtype=float
            )
            usable = np.isfinite(estimates) & np.isfinite(errors) & (errors > 0)
            mean, mean_error, between_sd = _population_normal(
                estimates[usable], errors[usable]
            )
            z_statistic = mean / mean_error if mean_error > 0 else float("nan")
            population_rows.append(
                {
                    "analysis": "ssm_population",
                    "condition": condition,
                    "model": model,
                    "parameter": parameter,
                    "n_subjects": int(usable.sum()),
                    "population_mean": mean,
                    "population_mean_se": mean_error,
                    "between_subject_sd": between_sd,
                    "z": z_statistic,
                    "p": (
                        float(2.0 * stats.norm.sf(abs(z_statistic)))
                        if np.isfinite(z_statistic)
                        else float("nan")
                    ),
                }
            )
            if not np.isfinite(mean):
                continue
            for subject, estimate, error in zip(
                group.loc[usable, "subject"], estimates[usable], errors[usable]
            ):
                weight = between_sd**2 / (between_sd**2 + error**2)
                subject_rows.append(
                    {
                        "subject": subject,
                        "condition": condition,
                        "model": model,
                        "parameter": parameter,
                        "estimate": float(estimate),
                        "standard_error": float(error),
                        "population_informed_estimate": float(
                            mean + weight * (estimate - mean)
                        ),
                        "own_data_weight": float(weight),
                    }
                )
    return pd.DataFrame(subject_rows), pd.DataFrame(population_rows)
