"""Urgency gating against bounded integration, fitted per subject.

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
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Final

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

SSM_MODELS: Final[tuple[str, ...]] = ("ddm", "urgency")
MODEL_PARAMETERS: Final[dict[str, tuple[str, ...]]] = {
    "ddm": ("drift_scale", "bound", "nondecision_s"),
    "urgency": (
        "drift_scale",
        "urgency_scale",
        "urgency_onset_s",
        "nondecision_s",
    ),
}
PARAMETER_RANGES: Final[dict[str, tuple[float, float]]] = {
    "drift_scale": (0.0, 5.0),
    "bound": (0.1, 5.0),
    "urgency_scale": (0.01, 2.0),
    "urgency_onset_s": (0.0, 2.0),
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


def _evidence(t: float, lead_path: tuple[int, ...]) -> float:
    """Evaluate the token lead at one moment of one trial."""
    jump = int(t / TOKEN_INTERVAL_S)
    if jump <= 0:
        return 0.0
    return float(lead_path[min(jump, len(lead_path)) - 1])


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
        ``"ddm"`` for the bounded integrator or ``"urgency"`` for the
        urgency-gating model.
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
    would demand an unbounded diffusion grid. ``pyddm``'s default two-percent
    uniform contaminant is kept: it bounds the likelihood of outlying decision
    times away from zero, and being identical in both models it cannot favor
    either.
    """
    import pyddm

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

    drift = integrator_drift if model == "ddm" else filter_drift
    bound = fixed_bound if model == "ddm" else urgency_bound
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


def _fit_one(model: str, frame: pd.DataFrame) -> dict[str, float]:
    """Fit one accumulator to one subject's decisions by maximum likelihood."""
    import pyddm

    sample = pyddm.Sample.from_pandas_dataframe(
        frame[["rt", "correct", "lead_path"]],
        rt_column_name="rt",
        choice_column_name="correct",
    )
    t_dur = float(np.ceil(frame["rt"].max() + 0.5))
    fitted = pyddm.fit_adjust_model(
        sample,
        _build_model(model, t_dur),
        lossfunction=pyddm.LossLikelihood,
        fitparams={"seed": FIT_SEED},
        verbose=False,
    )
    estimates = {
        name: float(value)
        for name, value in zip(
            fitted.get_model_parameter_names(), fitted.get_model_parameters()
        )
    }
    errors = _standard_errors(model, sample, estimates, t_dur)
    return {
        "log_likelihood": -float(fitted.fitresult.value()),
        "t_dur_s": t_dur,
        **estimates,
        **{f"{name}_se": value for name, value in errors.items()},
    }


def _fit_all_cells(
    cells: list[tuple[object, str, pd.DataFrame]],
    *,
    n_jobs: int,
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
        for model in SSM_MODELS
        if len(frame) >= MINIMUM_FIT_TRIALS
    ]
    results: list[dict[str, dict[str, float] | None]] = [
        {model: None for model in SSM_MODELS} for _ in cells
    ]
    if n_jobs == 1:
        for index, model, frame in tasks:
            results[index][model] = _fit_one(model, frame)
        return results
    with ProcessPoolExecutor(max_workers=None if n_jobs < 0 else n_jobs) as pool:
        submitted = {
            pool.submit(_fit_one, model, frame): (index, model)
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
) -> pd.DataFrame:
    """Fit the urgency-gating and bounded-integrator models per subject.

    Both models see the same trial-by-trial token evidence and differ only in
    how they use it: the integrator accumulates it without leak against a fixed
    bound, while the urgency model low-pass filters it and compares the result
    with a criterion that falls as urgency grows.

    Parameters
    ----------
    features
        Canonical trial-feature table. Only eligible Fast and Slow task trials
        returned by :func:`~meg_tokens.behavior.trials.task_trials` are used.
    n_jobs
        Number of worker processes for the fits. ``1`` runs them in this
        process; a negative value uses every available CPU.

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
            for model in SSM_MODELS
            for name in MODEL_PARAMETERS[model]
            for column in (name, f"{name}_se")
        )
    )
    cells = [
        (subject, condition, _decision_frame(selected))
        for subject, condition, selected in _subject_condition_groups(trials)
    ]
    fitted = _fit_all_cells(cells, n_jobs=n_jobs)
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
                    "converged": fit is not None,
                    **{
                        column: (
                            fit.get(column, float("nan"))
                            if fit is not None
                            else float("nan")
                        )
                        for column in parameter_columns
                    },
                }
            )
    return pd.DataFrame(rows)


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
        rate = evidence if model == "ddm" else (evidence - state) / FILTER_TAU_S
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
