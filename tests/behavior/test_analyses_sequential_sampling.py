"""Tests for meg_tokens.behavior.analyses.sequential_sampling.

The evidence trajectory and the group layers are tested on written-out tables
and run in milliseconds. Fitting an accumulator is a differential-evolution
search over a diffusion solve and takes minutes per subject, so the recovery
test that checks the fit itself runs only when ``MEG_TOKENS_RUN_MODEL_FITS``
is set:

    MEG_TOKENS_RUN_MODEL_FITS=1 python -m pytest \
        tests/behavior/test_analyses_sequential_sampling.py
"""

import os

import numpy as np
import pandas as pd
import pytest

from meg_tokens.behavior.analyses.sequential_sampling import (
    MODEL_PARAMETERS,
    TIME_COURSE_STEP_S,
    TOKEN_INTERVAL_S,
    _evidence,
    _lead_path,
    fit_sequential_sampling_models,
    fitted_predictions,
    model_comparison_statistics,
    population_parameters,
    urgency_condition_contrast,
)

from tests.behavior.factories import trial_features


requires_model_fits = pytest.mark.skipif(
    not os.environ.get("MEG_TOKENS_RUN_MODEL_FITS"),
    reason=(
        "Set MEG_TOKENS_RUN_MODEL_FITS=1 to run the accumulator fits, which "
        "take minutes per subject."
    ),
)


def _fits(**columns) -> pd.DataFrame:
    """Build a fit table of the shape ``fit_sequential_sampling_models`` returns."""
    length = len(next(iter(columns.values())))
    defaults = {
        "subject": [f"H{index:02d}" for index in range(1, length + 1)],
        "condition": ["all"] * length,
        "model": ["urgency"] * length,
        "n_trials": [200] * length,
        "n_token_sequences": [60] * length,
        "n_parameters": [4] * length,
        "log_likelihood": [-100.0] * length,
        "aic": [208.0] * length,
        "bic": [221.0] * length,
        "delta_aic": [-10.0 - index for index in range(length)],
        "delta_bic": [-8.0 - index for index in range(length)],
        "converged": [True] * length,
    }
    for parameter in MODEL_PARAMETERS["urgency"]:
        defaults[parameter] = [1.0] * length
        defaults[f"{parameter}_se"] = [0.1] * length
    defaults.update(columns)
    return pd.DataFrame(defaults)


# --- evidence trajectory ------------------------------------------------------


def test_the_evidence_path_is_the_running_lead_of_the_correct_target():
    assert _lead_path("1121", correct_target=1) == (1, 2, 1, 2)
    assert _lead_path("1121", correct_target=2) == (-1, -2, -1, -2)


def test_evidence_steps_once_per_token_and_holds_after_the_last_jump():
    path = (1, 2, 1)

    assert _evidence(0.0, path) == 0.0
    assert _evidence(TOKEN_INTERVAL_S - 0.01, path) == 0.0
    assert _evidence(TOKEN_INTERVAL_S, path) == 1.0
    assert _evidence(2 * TOKEN_INTERVAL_S, path) == 2.0
    assert _evidence(10 * TOKEN_INTERVAL_S, path) == 1.0


# --- fitting ------------------------------------------------------------------


def test_a_cell_with_too_few_trials_is_reported_rather_than_fitted():
    """Four free parameters are not identifiable from a handful of trials."""
    features = trial_features(dt_ms=[900.0, 1000.0, 1100.0, 1200.0])

    fits = fit_sequential_sampling_models(features)

    assert len(fits) == 6  # pooled, Fast, and Slow, each with both models
    assert set(fits["model"]) == {"ddm", "urgency"}
    assert not fits["converged"].any()
    assert fits["drift_scale"].isna().all()
    assert fits.loc[fits["condition"] == "all", "n_trials"].eq(4).all()
    assert fits.loc[fits["condition"] == "slow", "n_trials"].eq(0).all()


def test_trials_sharing_a_token_sequence_are_counted_once():
    """The distinct sequences drive the solver cost, so the fit table reports them."""
    features = trial_features(
        dt_ms=[900.0, 1000.0, 1100.0, 1200.0],
        token_directions=["112112112112112"] * 2 + ["221221221221221"] * 2,
        nCorrectChoice=[1, 1, 1, 1],
    )

    fits = fit_sequential_sampling_models(features)

    assert fits.loc[fits["condition"] == "all", "n_token_sequences"].eq(2).all()


@requires_model_fits
def test_the_fit_recovers_a_planted_urgency_signal():
    """Decisions simulated from filtered evidence gated by a growing urgency are
    fitted back to it, and the comparison selects urgency over integration."""
    from meg_tokens.behavior.analyses.sequential_sampling import _build_model

    planted = {
        "drift_scale": 0.8,
        "nondecision_s": 0.1,
        "urgency_scale": 0.6,
        "urgency_onset_s": 0.15,
    }
    model = _build_model("urgency", 4.0, planted)
    sequences = ["111211121112111", "121212121212121", "211211211221122"]
    decision_times = []
    correct = []
    directions = []
    for sequence in sequences:
        path = _lead_path(sequence, correct_target=1)
        solution = model.solve(conditions={"lead_path": path})
        sample = solution.resample(400, seed=1)
        outcomes = ((sample.choice_upper, True), (sample.choice_lower, False))
        for times, outcome in outcomes:
            decision_times.extend(float(value) * 1000.0 for value in times)
            correct.extend([outcome] * len(times))
            directions.extend([sequence] * len(times))
    features = trial_features(
        dt_ms=decision_times,
        isCorrect=correct,
        token_directions=directions,
        nCorrectChoice=[1] * len(decision_times),
    )

    fits = fit_sequential_sampling_models(features).set_index(["condition", "model"])

    urgency = fits.loc[("all", "urgency")]
    assert urgency["urgency_scale"] == pytest.approx(
        planted["urgency_scale"], abs=0.2
    )
    assert urgency["drift_scale"] == pytest.approx(planted["drift_scale"], abs=0.3)
    assert urgency["urgency_onset_s"] == pytest.approx(
        planted["urgency_onset_s"], abs=0.2
    )
    assert urgency["delta_bic"] < 0
    assert urgency["urgency_scale_se"] > 0


# --- fitted predictions -------------------------------------------------------


def _plottable_fits() -> pd.DataFrame:
    """Fitted values for both models, of the shape the prediction step reads."""
    shared = {
        "subject": "H01",
        "condition": "all",
        "converged": True,
        "t_dur_s": 3.0,
        "drift_scale": 0.5,
        "bound": 1.2,
        "nondecision_s": 0.05,
        "urgency_scale": 0.6,
        "urgency_onset_s": 0.1,
    }
    return pd.DataFrame(
        [{**shared, "model": model} for model in ("ddm", "urgency")]
    )


def _plottable_features() -> pd.DataFrame:
    """Easy and misleading trials whose evidence paths are mirror images."""
    return trial_features(
        dt_ms=list(np.linspace(400.0, 2000.0, 60)),
        isCorrect=[True] * 40 + [False] * 20,
        token_directions=["112112112112112"] * 30 + ["221221221221221"] * 30,
        nCorrectChoice=[1] * 60,
        trial_class_name=["easy"] * 30 + ["misleading"] * 30,
    )


def test_the_criterion_is_flat_for_the_integrator_and_falls_for_urgency():
    courses, _ = fitted_predictions(_plottable_features(), _plottable_fits())
    criterion = courses.groupby("model")["criterion"]

    assert criterion.nunique()["ddm"] == 1
    assert criterion.max()["urgency"] > criterion.min()["urgency"]
    assert courses["time_s"].diff().dropna().max() == pytest.approx(
        TIME_COURSE_STEP_S
    )


def test_trajectories_are_resolved_by_trial_class():
    """Pooling classes averages misleading evidence against easy evidence; these
    paths are mirror images, so the pooled trajectory cancels and the per-class
    ones do not."""
    courses, _ = fitted_predictions(_plottable_features(), _plottable_fits())
    urgency = courses.loc[courses["model"] == "urgency"].set_index("trial_class")

    assert set(courses["trial_class"]) == {"all", "easy", "misleading"}
    assert urgency.loc["easy", "mean_decision_variable"].max() > 0.5
    assert urgency.loc["misleading", "mean_decision_variable"].min() < -0.5
    assert urgency.loc["all", "mean_decision_variable"].abs().max() < 1e-9


def test_the_predicted_decision_time_density_integrates_to_one():
    courses, _ = fitted_predictions(_plottable_features(), _plottable_fits())
    pooled = courses.loc[
        (courses["model"] == "urgency") & (courses["trial_class"] == "all")
    ]
    mass = (
        pooled[["predicted_density_correct", "predicted_density_error"]].to_numpy().sum()
        * TIME_COURSE_STEP_S
    )

    assert mass == pytest.approx(1.0, abs=0.02)


def test_each_trial_prediction_carries_the_join_key():
    _, predictions = fitted_predictions(_plottable_features(), _plottable_fits())

    assert set(predictions["model"]) == {"ddm", "urgency"}
    assert predictions["trial_id"].notna().all()
    assert len(predictions) == 120  # 60 trials x 2 models
    assert predictions["predicted_accuracy"].between(0.0, 1.0).all()


def test_a_cell_that_was_not_fitted_contributes_no_prediction_rows():
    fits = _plottable_fits().assign(converged=False)

    courses, predictions = fitted_predictions(_plottable_features(), fits)

    assert courses.empty
    assert predictions.empty


# --- model comparison ---------------------------------------------------------


def test_the_comparison_counts_the_subjects_each_model_wins():
    fits = pd.concat(
        [
            _fits(delta_bic=[-12.0, -3.0, 4.0, np.nan], delta_aic=[-14.0] * 4),
            _fits(model=["ddm"] * 4, delta_bic=[0.0] * 4, delta_aic=[0.0] * 4),
        ],
        ignore_index=True,
    )

    statistics = model_comparison_statistics(fits).set_index("criterion")

    assert statistics.loc["bic", "n_subjects_favoring_urgency"] == 2
    assert statistics.loc["bic", "n_subjects_favoring_ddm"] == 1
    # The non-converged subject contributes no difference to the group test.
    assert statistics.loc["bic", "n_subjects"] == 3
    assert statistics.loc["bic", "mean"] == pytest.approx((-12.0 - 3.0 + 4.0) / 3)


def test_the_comparison_is_reported_for_every_condition():
    fits = pd.concat(
        [_fits(condition=[condition] * 3) for condition in ("all", "fast", "slow")],
        ignore_index=True,
    )

    statistics = model_comparison_statistics(fits)

    assert set(statistics["condition"]) == {"all", "fast", "slow"}
    assert set(statistics["criterion"]) == {"aic", "bic"}


# --- Fast versus Slow urgency -------------------------------------------------


def test_the_condition_contrast_pairs_each_subject_across_blocks():
    fits = pd.concat(
        [
            _fits(
                condition=["fast"] * 3,
                urgency_scale=[1.4, 1.2, 1.6],
            ),
            _fits(
                condition=["slow"] * 3,
                urgency_scale=[1.0, 0.9, 1.1],
            ),
        ],
        ignore_index=True,
    )

    contrast = urgency_condition_contrast(fits).set_index("parameter")

    assert contrast.loc["urgency_scale", "n_subjects"] == 3
    assert contrast.loc["urgency_scale", "mean_difference"] == pytest.approx(
        np.mean([0.4, 0.3, 0.5])
    )
    # Every fitted parameter is contrasted, not only the urgency scale.
    assert set(contrast.index) == set(MODEL_PARAMETERS["urgency"])


def test_a_subject_fitted_in_only_one_condition_is_left_out_of_the_contrast():
    fits = pd.concat(
        [
            _fits(condition=["fast"] * 3, urgency_scale=[1.4, 1.2, 1.6]),
            _fits(condition=["slow"] * 2, urgency_scale=[1.0, 0.9]),
        ],
        ignore_index=True,
    )

    contrast = urgency_condition_contrast(fits).set_index("parameter")

    assert contrast.loc["urgency_scale", "n_subjects"] == 2


# --- population layer ---------------------------------------------------------


def test_the_population_mean_and_spread_are_estimated_from_the_subject_fits():
    """With negligible estimation error the population reduces to the subjects'
    own mean and their maximum-likelihood standard deviation."""
    estimates = [0.8, 1.0, 1.2]
    fits = _fits(urgency_scale=estimates, urgency_scale_se=[0.001] * 3)

    _, population = population_parameters(fits)
    scale = population.set_index("parameter").loc["urgency_scale"]

    assert scale["n_subjects"] == 3
    assert scale["population_mean"] == pytest.approx(1.0)
    assert scale["between_subject_sd"] == pytest.approx(
        np.sqrt(np.mean((np.array(estimates) - 1.0) ** 2)), rel=1e-3
    )


def test_an_uncertain_subject_is_pulled_further_toward_the_population():
    fits = _fits(
        urgency_scale=[1.0, 1.0, 1.0, 2.0, 2.0],
        urgency_scale_se=[0.05, 0.05, 0.05, 0.05, 0.8],
    )

    estimates, _ = population_parameters(fits)
    scales = estimates.loc[estimates["parameter"] == "urgency_scale"].set_index("subject")

    certain = scales.loc["H04"]
    uncertain = scales.loc["H05"]
    assert uncertain["own_data_weight"] < certain["own_data_weight"]
    assert abs(uncertain["population_informed_estimate"] - 2.0) > abs(
        certain["population_informed_estimate"] - 2.0
    )


def test_a_subject_without_a_usable_standard_error_is_left_out():
    fits = _fits(urgency_scale=[1.0, 1.2, np.nan], urgency_scale_se=[0.1, 0.1, np.nan])

    estimates, population = population_parameters(fits)

    assert population.set_index("parameter").loc["urgency_scale", "n_subjects"] == 2
    assert set(estimates.loc[estimates["parameter"] == "urgency_scale", "subject"]) == {
        "H01",
        "H02",
    }


def test_a_parameter_no_subject_could_be_fitted_for_is_still_reported():
    fits = _fits(urgency_scale=[np.nan] * 3, urgency_scale_se=[np.nan] * 3)

    estimates, population = population_parameters(fits)

    scale = population.set_index("parameter").loc["urgency_scale"]
    assert scale["n_subjects"] == 0
    assert np.isnan(scale["population_mean"])
    assert estimates.loc[estimates["parameter"] == "urgency_scale"].empty
