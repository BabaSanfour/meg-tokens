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
    MECHANISTIC_MODELS,
    MODEL_PARAMETERS,
    PARAMETER_RANGES,
    MIXTURE_COEF,
    TIME_COURSE_STEP_S,
    TOKEN_INTERVAL_S,
    _additive_drive,
    _build_model,
    _evidence,
    empirical_lead_paths,
    exclusion_robustness_audit,
    _recovery_truth_values,
    _lead_path,
    fit_sequential_sampling_models,
    eligibility_audit,
    matched_sequence_diagnostic,
    matched_sequence_audit,
    matched_sequence_statistics,
    mechanistic_model_statistics,
    heldout_model_statistics,
    heldout_pairwise_model_statistics,
    heldout_model_evaluation,
    heldout_fold_audit,
    fitted_predictions,
    model_comparison_statistics,
    model_recovery_confusion,
    parameter_recovery_statistics,
    parameter_recovery,
    model_recovery,
    robustness_statistics,
    exclusion_robustness_statistics,
    fitted_distribution_checks,
    population_parameters,
    urgency_condition_contrast,
)
from meg_tokens.workflows.thura2012 import (
    _analysis_scope,
    _validate_heldout_coverage,
    _validate_primary_fit_diagnostics,
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


def test_complete_model_set_has_explicit_four_model_schema():
    features = trial_features(dt_ms=[900.0, 1000.0, 1100.0, 1200.0])
    fits = fit_sequential_sampling_models(features, models=MECHANISTIC_MODELS)
    assert set(fits["model"]) == set(MECHANISTIC_MODELS)
    assert len(fits) == 12
    assert {"collapse_rate", "additive_scale"}.issubset(fits.columns)
    assert {
        "n_starts", "best_start", "optimizer_success", "boundary_hit",
        "boundary_parameters", "start_objectives", "start_converged", "fit_error",
    }.issubset(fits.columns)
    assert fits.loc[fits["model"] == "ddm", "fit_error"].eq("insufficient_trials").all()
    assert not fits["converged"].any()


def test_all_model_builders_use_the_explicit_lapse_overlay():
    """The PyDDM lapse coefficient is part of the reproducible model policy."""
    assert MIXTURE_COEF == pytest.approx(0.02)
    for model_name, parameters in MODEL_PARAMETERS.items():
        values = {
            parameter: sum(PARAMETER_RANGES[parameter]) / 2
            for parameter in parameters
        }
        model = _build_model(model_name, 3.0, values)
        overlays = model.get_dependence("overlay").overlays
        assert overlays[-1].name == "easy_mixture_model"


def test_eligibility_audit_preserves_primary_trial_counts():
    features = trial_features(
        dt_ms=[900.0, np.nan, -10.0],
        primary_analysis_eligible=[True, False, True],
        has_choice=[True, False, True],
        is_started=[True, True, True],
    )
    audit = eligibility_audit(features).set_index("criterion")
    assert audit.loc["all_feature_rows", "n_rows"] == 3
    assert audit.loc["primary_analysis_eligible", "n_rows"] == 2
    assert audit.loc["positive_dt_for_first_passage", "n_rows"] == 1
    assert "diagnostic_short_token_log" in audit.index


def test_subject_fit_provenance_labels_state_grid_units_not_seconds(tmp_path):
    from meg_tokens.workflows.behavior_characterization import _ssm_fit_metadata

    metadata = _ssm_fit_metadata(
        models=MECHANISTIC_MODELS,
        n_jobs=8,
        n_starts=3,
        features_path=tmp_path / "trialfeatures.tsv",
        excluded=set(),
    )
    assert metadata["solver"]["state_grid_units"] == "decision-variable/noise units (not seconds)"
    assert "dx_s" not in metadata["solver"]
    assert metadata["n_jobs"] == 8
    assert metadata["n_starts"] == 3
    assert metadata["source_tree_sha256"]


def test_primary_fit_aggregation_rejects_stale_one_start_diagnostics():
    table = pd.DataFrame(
        {
            "n_starts": [3],
            "optimizer_success": [True],
            "boundary_hit": [False],
            "boundary_parameters": [""],
            "start_objectives": ['{"0": 1.0, "1": 2.0, "2": 3.0}'],
            "start_converged": ['{"0": true, "1": true, "2": true}'],
            "fit_error": [""],
        }
    )
    _validate_primary_fit_diagnostics(table, subject="H01")

    stale = table.copy()
    stale.loc[0, "n_starts"] = 1
    with pytest.raises(ValueError, match="n_starts=3"):
        _validate_primary_fit_diagnostics(stale, subject="H01")

    incomplete = table.copy()
    incomplete.loc[0, "start_objectives"] = '{"0": 1.0}'
    with pytest.raises(ValueError, match="start_objectives has 1 starts"):
        _validate_primary_fit_diagnostics(incomplete, subject="H01")


def test_thura_sidecar_scope_labels_population_and_simulation_stages():
    assert _analysis_scope("ssmcomparison")[0] == "thura2012_mechanistic_evaluation"
    assert "complete-log" not in _analysis_scope("ssmexclusionrefit")[1]
    assert "token_log_rows==15" in _analysis_scope("ssmexclusionrefit")[1]
    assert "synthetic simulations" in _analysis_scope("ssmparameterrecovery")[1]
    assert "robustness configuration" in _analysis_scope("ssmrobustnessstats")[1]


def test_heldout_coverage_rejects_duplicate_or_missing_fold_cells():
    rows = [
        {"subject": "H01", "condition": condition, "model": model, "fold": fold}
        for condition in ("all", "fast", "slow")
        for model in MECHANISTIC_MODELS
        for fold in range(3)
    ]
    table = pd.DataFrame(rows)
    _validate_heldout_coverage(table, subjects=["H01"], folds=3)
    with pytest.raises(ValueError, match="held-out coverage is incomplete"):
        _validate_heldout_coverage(table.iloc[:-1], subjects=["H01"], folds=3)


def test_fitted_distribution_checks_normalizes_conditions_and_reuses_timecourse():
    features = trial_features(
        subject=["H01"] * 6,
        condition=["Fast", "Fast", "Fast", "Slow", "Slow", "Slow"],
        dt_ms=[100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
        isCorrect=[True, False, True, False, False, True],
    )
    fits = pd.DataFrame({
        "subject": ["H01"] * 3,
        "condition": ["all", "fast", "slow"],
        "model": ["ddm"] * 3,
        "converged": [True] * 3,
    })
    timecourse_rows = []
    for condition in ("all", "fast", "slow"):
        for trial_class in ("all", "easy"):
            for time_s, correct_density, error_density in (
                (0.0, 0.0, 0.0),
                (0.1, 2.0, 1.0),
                (0.2, 2.0, 1.0),
                (0.3, 0.0, 0.0),
            ):
                timecourse_rows.append({
                    "subject": "H01",
                    "condition": condition,
                    "model": "ddm",
                    "trial_class": trial_class,
                    "time_s": time_s,
                    "predicted_density_correct": correct_density,
                    "predicted_density_error": error_density,
                })
    checks = fitted_distribution_checks(
        features, fits, timecourse=pd.DataFrame(timecourse_rows)
    )
    assert not checks.empty
    assert set(checks["condition"]) == {"all", "fast", "slow"}
    assert set(checks["outcome"]) == {"correct", "error"}
    assert set(checks["trial_class"]) == {"all", "easy"}
    assert checks["prediction_source"].eq("ssmtimecourse").all()
    all_correct_median = checks.loc[
        (checks["condition"] == "all")
        & (checks["trial_class"] == "all")
        & (checks["outcome"] == "correct")
        & (checks["quantile"] == 0.5)
    ].iloc[0]
    assert all_correct_median["n_observed"] == 3
    assert all_correct_median["observed_rt_s"] == pytest.approx(0.3)
    assert all_correct_median["predicted_correct_mass"] == pytest.approx(0.4)
    assert all_correct_median["predicted_error_mass"] == pytest.approx(0.2)
    assert np.isfinite(all_correct_median["predicted_rt_s"])


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


def test_mechanistic_comparison_reports_each_alternative_against_ddm():
    base = _fits(model=["ddm"] * 2, delta_bic=[0.0, 0.0], delta_aic=[0.0, 0.0])
    alternatives = pd.concat(
        [
            _fits(model=[model] * 2, delta_bic=[-2.0, 2.0], delta_aic=[-3.0, 3.0])
            for model in MECHANISTIC_MODELS if model != "ddm"
        ],
        ignore_index=True,
    )
    stats_table = mechanistic_model_statistics(pd.concat([base, alternatives]))
    assert set(stats_table["model"]) == set(MECHANISTIC_MODELS) - {"ddm"}
    assert set(stats_table["criterion"]) == {"aic", "bic"}


def test_mechanistic_comparison_is_condition_stratified_and_subject_paired():
    baseline = _fits(
        model=["ddm"] * 4,
        condition=["fast", "fast", "slow", "slow"],
        subject=["H01", "H02", "H01", "H02"],
        aic=[100.0] * 4,
        bic=[100.0] * 4,
    )
    urgency = _fits(
        model=["urgency"] * 4,
        condition=["fast", "fast", "slow", "slow"],
        subject=["H01", "H02", "H01", "H02"],
        aic=[90.0] * 4,
        bic=[90.0] * 4,
    )
    stats_table = mechanistic_model_statistics(pd.concat([baseline, urgency]))
    assert set(stats_table["condition"]) == {"fast", "slow"}
    assert stats_table["n_subjects"].eq(2).all()


def test_matched_sequence_diagnostic_does_not_match_on_outcome():
    # Opposite early histories converge at jump 4, share the same post-
    # convergence suffix, and are decided after the prespecified jump 6.
    suffix = "11111122222"
    features = trial_features(
        dt_ms=[900.0] * 10,
        token_directions=["1122" + suffix] * 5 + ["2211" + suffix] * 5,
        nCorrectChoice=[1] * 10,
        decision_token_index=[np.nan, 4] + [7] * 8,
        isCorrect=[True, False] * 5,
    )
    matched = matched_sequence_diagnostic(features, early_jumps=3, min_trials=1)
    assert set(matched["early_bias"]) == {"for", "against"}
    assert matched["n_trials"].min() >= 1
    assert matched["pair_id"].nunique() == 5
    assert matched["convergence_jump"].eq(6).all()
    assert matched["decision_after_convergence"].eq(
        matched["post_convergence_eligible"]
    ).all()
    assert matched["decision_token_index"].isna().any()
    assert matched["matching_rule"].str.startswith("stimulus_only").all()
    statistics = matched_sequence_statistics(matched)
    assert statistics["n_subjects"].eq(1).all()
    audit = matched_sequence_audit(matched, features)
    assert audit.iloc[0]["n_stimulus_pairs"] == 5
    assert audit.iloc[0]["n_post_convergence_pairs"] == 3
    assert audit.iloc[0]["post_convergence_pair_fraction"] == pytest.approx(0.6)


def test_additive_urgency_drive_is_symmetric_under_target_relabeling():
    path = _lead_path("112212212212212", correct_target=1)
    mirrored = tuple(-value for value in path)
    times = np.linspace(0.2, 2.0, 10)
    assert all(
        _additive_drive(float(time), path, 0.7)
        == pytest.approx(-_additive_drive(float(time), mirrored, 0.7))
        for time in times
    )


def test_empirical_recovery_paths_use_each_trial_actual_correct_target():
    directions = ["112211221122112", "112211221122112"]
    features = trial_features(
        dt_ms=[900.0, 900.0],
        token_directions=directions,
        nCorrectChoice=[1, 2],
    )
    paths = empirical_lead_paths(features)
    assert paths[0] == _lead_path(directions[0], correct_target=1)
    assert paths[1] == tuple(-value for value in paths[0])
    assert all(isinstance(path, tuple) for path in paths)


def test_heldout_statistics_are_subject_paired_and_condition_stratified():
    rows = []
    for subject, offset in (("H01", 0.1), ("H02", -0.1)):
        for condition in ("fast", "slow"):
            for fold in (0, 1):
                n_test = 1 if fold == 0 else 3
                rows.extend([
                    {
                        "subject": subject, "condition": condition,
                        "model": "ddm", "fold": fold,
                        "n_test": n_test,
                        "heldout_log_likelihood": -1.0 * n_test,
                        "heldout_log_likelihood_per_trial": -1.0,
                    },
                    {
                        "subject": subject, "condition": condition,
                        "model": "urgency", "fold": fold,
                        "n_test": n_test,
                        "heldout_log_likelihood": (-1.0 + offset) * n_test,
                        "heldout_log_likelihood_per_trial": -1.0 + offset,
                    },
                ])
    stats_table = heldout_model_statistics(pd.DataFrame(rows))
    assert set(stats_table["condition"]) == {"fast", "slow"}
    assert stats_table["n_subjects"].eq(2).all()
    assert stats_table["criterion"].eq("heldout_log_likelihood_per_trial").all()


def test_heldout_statistics_weight_fold_scores_by_test_trial_count():
    rows = pd.DataFrame([
        {"subject": "H01", "condition": "fast", "model": "ddm", "fold": 0,
         "n_test": 1, "heldout_log_likelihood": -1.0,
         "heldout_log_likelihood_per_trial": -1.0},
        {"subject": "H01", "condition": "fast", "model": "ddm", "fold": 1,
         "n_test": 3, "heldout_log_likelihood": -6.0,
         "heldout_log_likelihood_per_trial": -2.0},
        {"subject": "H01", "condition": "fast", "model": "urgency", "fold": 0,
         "n_test": 1, "heldout_log_likelihood": -0.5,
         "heldout_log_likelihood_per_trial": -0.5},
        {"subject": "H01", "condition": "fast", "model": "urgency", "fold": 1,
         "n_test": 3, "heldout_log_likelihood": -3.0,
         "heldout_log_likelihood_per_trial": -1.0},
    ])
    stats_table = heldout_model_statistics(rows)
    # Candidate score is -3.5/4, baseline is -7/4: Δ = 0.875.
    assert stats_table.iloc[0]["mean"] == pytest.approx(0.875)


def test_heldout_pairwise_statistics_compares_two_candidates_directly():
    rows = []
    for subject, offset in (("H01", 0.1), ("H02", -0.1)):
        for condition in ("fast", "slow"):
            for fold in (0, 1):
                n_test = 1 if fold == 0 else 3
                rows.extend([
                    {
                        "subject": subject, "condition": condition,
                        "model": "ddm", "fold": fold,
                        "n_test": n_test,
                        "heldout_log_likelihood": -1.0 * n_test,
                        "heldout_log_likelihood_per_trial": -1.0,
                    },
                    {
                        "subject": subject, "condition": condition,
                        "model": "urgency", "fold": fold,
                        "n_test": n_test,
                        "heldout_log_likelihood": (-1.0 + offset) * n_test,
                        "heldout_log_likelihood_per_trial": -1.0 + offset,
                    },
                    {
                        "subject": subject, "condition": condition,
                        "model": "collapsing", "fold": fold,
                        "n_test": n_test,
                        "heldout_log_likelihood": -1.0 * n_test,
                        "heldout_log_likelihood_per_trial": -1.0,
                    },
                ])
    pairwise = heldout_pairwise_model_statistics(pd.DataFrame(rows))
    # Every unordered pair among {ddm, urgency, collapsing} in both
    # conditions: 3 pairs x 2 conditions.
    assert len(pairwise) == 6
    assert set(zip(pairwise["model_a"], pairwise["model_b"])) == {
        ("collapsing", "ddm"), ("collapsing", "urgency"), ("ddm", "urgency"),
    }
    urgency_vs_collapsing = pairwise.loc[
        (pairwise["model_a"] == "collapsing") & (pairwise["model_b"] == "urgency")
    ]
    assert urgency_vs_collapsing["n_subjects"].eq(2).all()
    assert urgency_vs_collapsing["criterion"].eq("heldout_log_likelihood_per_trial").all()
    # collapsing is identical to ddm here, so ddm-vs-urgency and
    # collapsing-vs-urgency must give the same contrast, just with the sign
    # flipped by which model is "a".
    ddm_vs_urgency = pairwise.loc[
        (pairwise["model_a"] == "ddm") & (pairwise["model_b"] == "urgency")
    ].set_index("condition")["mean"]
    collapsing_vs_urgency_mean = urgency_vs_collapsing.set_index("condition")["mean"]
    pd.testing.assert_series_equal(
        ddm_vs_urgency.sort_index(), collapsing_vs_urgency_mean.sort_index(),
        check_names=False,
    )


def test_heldout_pairwise_statistics_weight_fold_scores_by_test_trial_count():
    rows = pd.DataFrame([
        {"subject": "H01", "condition": "fast", "model": "ddm", "fold": 0,
         "n_test": 1, "heldout_log_likelihood": -1.0,
         "heldout_log_likelihood_per_trial": -1.0},
        {"subject": "H01", "condition": "fast", "model": "ddm", "fold": 1,
         "n_test": 3, "heldout_log_likelihood": -6.0,
         "heldout_log_likelihood_per_trial": -2.0},
        {"subject": "H01", "condition": "fast", "model": "urgency", "fold": 0,
         "n_test": 1, "heldout_log_likelihood": -0.5,
         "heldout_log_likelihood_per_trial": -0.5},
        {"subject": "H01", "condition": "fast", "model": "urgency", "fold": 1,
         "n_test": 3, "heldout_log_likelihood": -3.0,
         "heldout_log_likelihood_per_trial": -1.0},
    ])
    pairwise = heldout_pairwise_model_statistics(rows)
    # Same weighted scores as the ddm-baseline test: urgency -3.5/4,
    # ddm -7/4, Δ(urgency - ddm) = 0.875.
    row = pairwise.loc[
        (pairwise["model_a"] == "ddm") & (pairwise["model_b"] == "urgency")
    ].iloc[0]
    assert row["mean"] == pytest.approx(-0.875)
    assert row["n_subjects_favoring_model_b"] == 1
    assert row["n_subjects_favoring_model_a"] == 0


def test_heldout_evaluation_persists_failed_small_cells_and_fold_audit():
    features = trial_features(dt_ms=[900.0, 1000.0, 1100.0, 1200.0])
    heldout = heldout_model_evaluation(
        features, models=("ddm",), folds=3, n_starts=1
    )
    assert len(heldout) == 9  # all/fast/slow × model × three folds
    assert not heldout["converged"].any()
    assert heldout["fit_error"].eq("insufficient_train_or_test_trials").all()
    audit = heldout_fold_audit(heldout, expected_folds=3)
    assert audit["folds_complete"].all()
    assert audit["n_failed"].eq(3).all()


def test_recovery_truth_design_varies_interior_parameters_deterministically():
    first, first_fractions = _recovery_truth_values(
        "urgency", repetition=0, repetitions=4, seed=0
    )
    second, second_fractions = _recovery_truth_values(
        "urgency", repetition=1, repetitions=4, seed=0
    )
    assert any(first[name] != second[name] for name in first)
    assert first_fractions != second_fractions
    for parameter, value in first.items():
        lower, upper = PARAMETER_RANGES[parameter]
        assert lower < value < upper


def test_recovery_truth_design_is_a_permutation_at_production_size():
    designs = [
        _recovery_truth_values("urgency", repetition=index, repetitions=12, seed=0)[0]
        for index in range(12)
    ]
    for parameter in MODEL_PARAMETERS["urgency"]:
        values = {design[parameter] for design in designs}
        assert len(values) == 12


def test_recovery_array_repetition_index_has_complete_machine_rows(monkeypatch):
    import meg_tokens.behavior.analyses.sequential_sampling as sequential_sampling

    monkeypatch.setattr(
        sequential_sampling,
        "_synthetic_frame",
        lambda *args, **kwargs: pd.DataFrame({"rt": [0.5] * 20}),
    )

    def fake_fit(model, frame, **kwargs):
        return {
            **{parameter: 1.0 for parameter in MODEL_PARAMETERS[model]},
            "log_likelihood": -10.0,
            "optimizer_success": True,
            "boundary_hit": False,
        }

    monkeypatch.setattr(sequential_sampling, "_fit_one", fake_fit)
    parameter = parameter_recovery(
        repetitions=12, repetition_indices=[7], truth_design_repetitions=12,
    )
    model = model_recovery(
        repetitions=12, repetition_indices=[7], truth_design_repetitions=12,
    )
    assert len(parameter) == sum(len(values) for values in MODEL_PARAMETERS.values())
    assert len(model) == len(MECHANISTIC_MODELS) ** 2
    assert set(parameter["repetition"]) == {7}
    assert set(model["repetition"]) == {7}


def test_robustness_statistics_preserves_config_condition_model_and_paired_change():
    rows = []
    for subject in ("H01", "H02"):
        for configuration, offset in (("baseline", 0.0), ("expanded_bounds", -1.0)):
            for condition in ("all", "fast", "slow"):
                for model in MECHANISTIC_MODELS:
                    rows.append({
                        "subject": subject, "configuration": configuration,
                        "condition": condition, "model": model,
                        "delta_bic": offset if model == "urgency" else 0.0,
                        "converged": True, "boundary_hit": False,
                        "boundary_parameters": "",
                        "urgency_scale": 2.0 if model == "urgency" else np.nan,
                        "urgency_scale_lower_bound": 0.01,
                        "urgency_scale_upper_bound": 2.0,
                        "nondecision_s": 0.0,
                        "nondecision_s_lower_bound": 0.0,
                        "nondecision_s_upper_bound": 1.0,
                    })
    summary = robustness_statistics(pd.DataFrame(rows))
    assert set(summary["analysis"]) == {
        "ssm_robustness_summary", "ssm_robustness_paired_sensitivity",
    }
    assert summary["configuration"].isin({"baseline", "expanded_bounds"}).all()
    assert "urgency_scale_near_upper" in summary


def test_exclusion_statistics_compare_within_population_delta_bic():
    primary = _fits(
        model=["ddm", "urgency"], subject=["H01", "H01"],
        condition=["all", "all"], delta_bic=[0.0, -2.0],
        boundary_hit=[False, False],
    )
    strict = primary.copy()
    strict["delta_bic"] = [0.0, -3.0]
    summary = exclusion_robustness_statistics(strict, primary)
    assert set(summary["model"]) == {"urgency"}
    assert summary.iloc[0]["mean"] == pytest.approx(-1.0)


def test_recovery_summaries_are_machine_readable():
    parameter = pd.DataFrame([
        {"true_model": "ddm", "parameter": "bound", "true_value": 1.0,
         "estimated_value": 1.1, "converged": True, "boundary_hit": False},
        {"true_model": "ddm", "parameter": "bound", "true_value": 1.0,
         "estimated_value": 0.9, "converged": True, "boundary_hit": True},
    ])
    summary = parameter_recovery_statistics(parameter)
    assert {"bias", "rmse", "correlation", "convergence_rate", "boundary_rate"}.issubset(summary.columns)
    model = pd.DataFrame([
        {"repetition": 0, "true_model": "ddm", "fitted_model": "ddm", "selected_model": "ddm"},
        {"repetition": 0, "true_model": "ddm", "fitted_model": "urgency", "selected_model": "ddm"},
        {"repetition": 1, "true_model": "ddm", "fitted_model": "ddm", "selected_model": "urgency"},
    ])
    confusion = model_recovery_confusion(model)
    assert confusion["n_repetitions"].eq(2).all()
    assert confusion["n_selected"].sum() == 2


def test_exclusion_robustness_audit_reports_retention_and_minimum_cells():
    features = trial_features(
        dt_ms=[900.0, 900.0, 900.0],
        token_log_rows=[15, 14, 15],
        design_time_alignment_valid=[True, False, True],
    )
    audit = exclusion_robustness_audit(features)
    assert set(audit["rule"]) == {
        "complete_token_log", "canonical_alignment_valid",
        "complete_token_log_alignment_equivalence",
    }
    assert audit["n_primary_task_trials"].eq(3).all()
    assert audit.loc[
        audit["rule"] == "complete_token_log_alignment_equivalence", "equivalent"
    ].item()


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
