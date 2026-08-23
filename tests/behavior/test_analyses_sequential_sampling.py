"""Tests for meg_tokens.behavior.analyses.sequential_sampling.

The evidence trajectory and the group layers are tested on written-out tables
and run in milliseconds. Fitting an accumulator is a differential-evolution
search over a diffusion solve and takes minutes per subject, so the recovery
test that checks the fit itself runs only when ``MEG_TOKENS_RUN_MODEL_FITS``
is set:

    MEG_TOKENS_RUN_MODEL_FITS=1 python -m pytest \
        tests/behavior/test_analyses_sequential_sampling.py

Tests are grouped one per contract rather than one per assertion: the held-out
scoring rule, for instance, is one behaviour whether it is read through the
baseline comparison, the pairwise comparison, or the per-subject differences.
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
    heldout_pairwise_subject_differences,
    heldout_model_evaluation,
    heldout_fold_audit,
    fitted_predictions,
    model_comparison_statistics,
    model_recovery_confusion,
    parameter_recovery_statistics,
    parameter_recovery,
    model_recovery,
    robustness_statistics,
    exclusion_evidence_decomposition,
    fitted_distribution_check_statistics,
    heldout_contrast_robustness,
    exclusion_robustness_statistics,
    fitted_distribution_checks,
    urgency_condition_contrast,
)
from meg_tokens.workflows.sequential_sampling import (
    _analysis_scope,
    _ssm_fit_metadata,
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


def _fold_rows(models=("ddm", "urgency"), *, subjects=("H01", "H02"),
               conditions=("fast", "slow"), offsets=(0.1, -0.1)):
    """Held-out fold scores with unequal fold sizes (n_test 1 then 3).

    The candidate beats the baseline by ``offset`` per trial, so any scoring
    rule that weights folds correctly recovers that offset exactly.
    """
    rows = []
    for subject, offset in zip(subjects, offsets):
        for condition in conditions:
            for fold in (0, 1):
                n_test = 1 if fold == 0 else 3
                for model in models:
                    per_trial = -1.0 + (offset if model == "urgency" else 0.0)
                    rows.append({
                        "subject": subject, "condition": condition,
                        "model": model, "fold": fold, "n_test": n_test,
                        "heldout_log_likelihood": per_trial * n_test,
                        "heldout_log_likelihood_per_trial": per_trial,
                    })
    return pd.DataFrame(rows)


_UNEQUAL_FOLDS = pd.DataFrame([
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


# --- evidence trajectory ------------------------------------------------------


def test_the_evidence_path_is_the_signed_running_lead_of_the_correct_target():
    assert _lead_path("1121", correct_target=1) == (1, 2, 1, 2)
    assert _lead_path("1121", correct_target=2) == (-1, -2, -1, -2)

    # Evidence steps once per token and holds after the last jump.
    path = (1, 2, 1)
    assert _evidence(0.0, path) == 0.0
    assert _evidence(TOKEN_INTERVAL_S - 0.01, path) == 0.0
    assert _evidence(TOKEN_INTERVAL_S, path) == 1.0
    assert _evidence(2 * TOKEN_INTERVAL_S, path) == 2.0
    assert _evidence(10 * TOKEN_INTERVAL_S, path) == 1.0

    # Recovery paths use each trial's *actual* correct target, so the same
    # token string mirrors when the correct target flips.
    directions = ["112211221122112"] * 2
    paths = empirical_lead_paths(
        trial_features(
            dt_ms=[900.0, 900.0],
            token_directions=directions,
            nCorrectChoice=[1, 2],
        )
    )
    assert paths[0] == _lead_path(directions[0], correct_target=1)
    assert paths[1] == tuple(-value for value in paths[0])

    # The additive drive is odd under target relabelling, which is what stops
    # it from ever reading the post-hoc correct target.
    mirrored = tuple(-value for value in _lead_path("112212212212212", correct_target=1))
    original = _lead_path("112212212212212", correct_target=1)
    assert all(
        _additive_drive(float(t), original, 0.7)
        == pytest.approx(-_additive_drive(float(t), mirrored, 0.7))
        for t in np.linspace(0.2, 2.0, 10)
    )


# --- fitting contract ---------------------------------------------------------


def test_a_cell_too_thin_to_identify_four_parameters_is_reported_not_fitted():
    features = trial_features(dt_ms=[900.0, 1000.0, 1100.0, 1200.0])

    fits = fit_sequential_sampling_models(features)

    # Pooled, Fast and Slow, each with every prespecified model. Written
    # against the constant rather than a literal: this assertion used to read
    # 6 and so pinned the retired two-model default in place.
    assert len(fits) == 3 * len(MECHANISTIC_MODELS)
    assert set(fits["model"]) == set(MECHANISTIC_MODELS)
    assert not fits["converged"].any()
    assert fits["drift_scale"].isna().all()
    assert fits.loc[fits["condition"] == "all", "n_trials"].eq(4).all()
    assert fits.loc[fits["condition"] == "slow", "n_trials"].eq(0).all()
    assert fits.loc[fits["model"] == "ddm", "fit_error"].eq("insufficient_trials").all()

    # The four-model schema carries both alternatives' parameters and the
    # multi-start diagnostics the aggregation gate reads.
    assert {"collapse_rate", "additive_scale"}.issubset(fits.columns)
    assert {
        "n_starts", "best_start", "optimizer_success", "boundary_hit",
        "boundary_parameters", "start_objectives", "start_converged", "fit_error",
    }.issubset(fits.columns)

    # Distinct sequences drive the solver cost, so the fit table reports them.
    shared = fit_sequential_sampling_models(
        trial_features(
            dt_ms=[900.0, 1000.0, 1100.0, 1200.0],
            token_directions=["112112112112112"] * 2 + ["221221221221221"] * 2,
            nCorrectChoice=[1, 1, 1, 1],
        )
    )
    assert shared.loc[shared["condition"] == "all", "n_token_sequences"].eq(2).all()


def test_every_model_builder_carries_the_explicit_lapse_overlay():
    """The PyDDM lapse coefficient is part of the reproducible model policy."""
    assert MIXTURE_COEF == pytest.approx(0.02)
    for model_name, parameters in MODEL_PARAMETERS.items():
        values = {
            parameter: sum(PARAMETER_RANGES[parameter]) / 2 for parameter in parameters
        }
        model = _build_model(model_name, 3.0, values)
        assert model.get_dependence("overlay").overlays[-1].name == "easy_mixture_model"


@requires_model_fits
def test_the_fit_recovers_a_planted_urgency_signal():
    """Decisions simulated from filtered evidence gated by a growing urgency are
    fitted back to it, and the comparison selects urgency over integration."""
    planted = {
        "drift_scale": 0.8,
        "nondecision_s": 0.1,
        "urgency_scale": 0.6,
        "urgency_onset_s": 0.15,
    }
    model = _build_model("urgency", 4.0, planted)
    decision_times, correct, directions = [], [], []
    for sequence in ("111211121112111", "121212121212121", "211211211221122"):
        solution = model.solve(conditions={"lead_path": _lead_path(sequence, correct_target=1)})
        sample = solution.resample(400, seed=1)
        for times, outcome in ((sample.choice_upper, True), (sample.choice_lower, False)):
            decision_times.extend(float(value) * 1000.0 for value in times)
            correct.extend([outcome] * len(times))
            directions.extend([sequence] * len(times))

    fits = fit_sequential_sampling_models(
        trial_features(
            dt_ms=decision_times,
            isCorrect=correct,
            token_directions=directions,
            nCorrectChoice=[1] * len(decision_times),
        )
    ).set_index(["condition", "model"])

    urgency = fits.loc[("all", "urgency")]
    assert urgency["urgency_scale"] == pytest.approx(planted["urgency_scale"], abs=0.2)
    assert urgency["drift_scale"] == pytest.approx(planted["drift_scale"], abs=0.3)
    assert urgency["urgency_onset_s"] == pytest.approx(planted["urgency_onset_s"], abs=0.2)
    assert urgency["delta_bic"] < 0
    assert urgency["urgency_scale_se"] > 0


# --- provenance and acceptance gates ------------------------------------------


def test_the_fit_provenance_and_acceptance_gates_reject_a_stale_or_partial_run(tmp_path):
    metadata = _ssm_fit_metadata(
        models=MECHANISTIC_MODELS,
        n_jobs=8,
        n_starts=3,
        features_path=tmp_path / "trialfeatures.tsv",
        excluded=set(),
    )
    # The solver grid is in decision-variable units; labelling it in seconds
    # is how a reader mis-reads the resolution by orders of magnitude.
    assert metadata["solver"]["state_grid_units"] == "decision-variable/noise units (not seconds)"
    assert "dx_s" not in metadata["solver"]
    assert metadata["n_jobs"] == 8
    assert metadata["n_starts"] == 3
    assert metadata["source_tree_sha256"]

    # Every sidecar scope names the population it describes, so a simulation
    # table is never read as a subject result.
    assert _analysis_scope("ssmcomparison")[0] == "thura2012_mechanistic_evaluation"
    assert "complete-log" not in _analysis_scope("ssmexclusionrefit")[1]
    assert "token_log_rows==15" in _analysis_scope("ssmexclusionrefit")[1]
    assert "synthetic simulations" in _analysis_scope("ssmparameterrecovery")[1]
    assert "robustness configuration" in _analysis_scope("ssmrobustnessstats")[1]

    diagnostics = pd.DataFrame({
        "n_starts": [3],
        "optimizer_success": [True],
        "boundary_hit": [False],
        "boundary_parameters": [""],
        "start_objectives": ['{"0": 1.0, "1": 2.0, "2": 3.0}'],
        "start_converged": ['{"0": true, "1": true, "2": true}'],
        "fit_error": [""],
    })
    _validate_primary_fit_diagnostics(diagnostics, subject="H01")
    stale = diagnostics.assign(n_starts=[1])
    with pytest.raises(ValueError, match="n_starts=3"):
        _validate_primary_fit_diagnostics(stale, subject="H01")
    incomplete = diagnostics.assign(start_objectives=['{"0": 1.0}'])
    with pytest.raises(ValueError, match="start_objectives has 1 starts"):
        _validate_primary_fit_diagnostics(incomplete, subject="H01")

    coverage = pd.DataFrame([
        {"subject": "H01", "condition": condition, "model": model, "fold": fold}
        for condition in ("all", "fast", "slow")
        for model in MECHANISTIC_MODELS
        for fold in range(3)
    ])
    _validate_heldout_coverage(coverage, subjects=["H01"], folds=3)
    with pytest.raises(ValueError, match="held-out coverage is incomplete"):
        _validate_heldout_coverage(coverage.iloc[:-1], subjects=["H01"], folds=3)


def test_the_eligibility_and_exclusion_audits_state_what_each_rule_costs():
    audit = eligibility_audit(
        trial_features(
            dt_ms=[900.0, np.nan, -10.0],
            primary_analysis_eligible=[True, False, True],
            has_choice=[True, False, True],
            is_started=[True, True, True],
        )
    ).set_index("criterion")
    assert audit.loc["all_feature_rows", "n_rows"] == 3
    assert audit.loc["primary_analysis_eligible", "n_rows"] == 2
    assert audit.loc["positive_dt_for_first_passage", "n_rows"] == 1
    assert "diagnostic_short_token_log" in audit.index

    exclusion = exclusion_robustness_audit(
        trial_features(
            dt_ms=[900.0, 900.0, 900.0],
            token_log_rows=[15, 14, 15],
            design_time_alignment_valid=[True, False, True],
        )
    )
    assert set(exclusion["rule"]) == {
        "complete_token_log", "canonical_alignment_valid",
        "complete_token_log_alignment_equivalence",
    }
    assert exclusion["n_primary_task_trials"].eq(3).all()
    assert exclusion.loc[
        exclusion["rule"] == "complete_token_log_alignment_equivalence", "equivalent"
    ].item()


# --- fitted predictions -------------------------------------------------------


def _plottable_fits() -> pd.DataFrame:
    shared = {
        "subject": "H01", "condition": "all", "converged": True, "t_dur_s": 3.0,
        "drift_scale": 0.5, "bound": 1.2, "nondecision_s": 0.05,
        "urgency_scale": 0.6, "urgency_onset_s": 0.1,
    }
    return pd.DataFrame([{**shared, "model": model} for model in ("ddm", "urgency")])


def _plottable_features() -> pd.DataFrame:
    """Easy and misleading trials whose evidence paths are mirror images."""
    return trial_features(
        dt_ms=list(np.linspace(400.0, 2000.0, 60)),
        isCorrect=[True] * 40 + [False] * 20,
        token_directions=["112112112112112"] * 30 + ["221221221221221"] * 30,
        nCorrectChoice=[1] * 60,
        trial_class_name=["easy"] * 30 + ["misleading"] * 30,
    )


def test_fitted_predictions_separate_the_two_mechanisms_and_resolve_them_by_class():
    courses, predictions = fitted_predictions(_plottable_features(), _plottable_fits())

    # The integrator's criterion is a fixed bound; urgency's declines.
    criterion = courses.groupby("model")["criterion"]
    assert criterion.nunique()["ddm"] == 1
    assert criterion.max()["urgency"] > criterion.min()["urgency"]
    assert courses["time_s"].diff().dropna().max() == pytest.approx(TIME_COURSE_STEP_S)

    # Pooling classes averages misleading evidence against easy evidence; these
    # paths are mirror images, so the pooled trajectory cancels and the
    # per-class ones do not.
    urgency = courses.loc[courses["model"] == "urgency"].set_index("trial_class")
    assert set(courses["trial_class"]) == {"all", "easy", "misleading"}
    assert urgency.loc["easy", "mean_decision_variable"].max() > 0.5
    assert urgency.loc["misleading", "mean_decision_variable"].min() < -0.5
    assert urgency.loc["all", "mean_decision_variable"].abs().max() < 1e-9

    pooled = courses.loc[
        (courses["model"] == "urgency") & (courses["trial_class"] == "all")
    ]
    mass = (
        pooled[["predicted_density_correct", "predicted_density_error"]].to_numpy().sum()
        * TIME_COURSE_STEP_S
    )
    assert mass == pytest.approx(1.0, abs=0.02)

    # Trial-level rows carry the MEG join key and a probability.
    assert set(predictions["model"]) == {"ddm", "urgency"}
    assert predictions["trial_id"].notna().all()
    assert len(predictions) == 120  # 60 trials x 2 models
    assert predictions["predicted_accuracy"].between(0.0, 1.0).all()

    # A cell that was not fitted contributes nothing rather than a guess.
    unfitted_courses, unfitted_predictions = fitted_predictions(
        _plottable_features(), _plottable_fits().assign(converged=False)
    )
    assert unfitted_courses.empty and unfitted_predictions.empty


# --- model comparison ---------------------------------------------------------


def test_the_in_sample_comparison_counts_wins_and_stratifies_by_condition():
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

    by_condition = model_comparison_statistics(
        pd.concat(
            [_fits(condition=[condition] * 3) for condition in ("all", "fast", "slow")],
            ignore_index=True,
        )
    )
    assert set(by_condition["condition"]) == {"all", "fast", "slow"}
    assert set(by_condition["criterion"]) == {"aic", "bic"}


def test_the_mechanistic_comparison_reports_every_alternative_against_the_integrator():
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

    paired = mechanistic_model_statistics(
        pd.concat([
            _fits(model=["ddm"] * 4, condition=["fast", "fast", "slow", "slow"],
                  subject=["H01", "H02", "H01", "H02"], aic=[100.0] * 4, bic=[100.0] * 4),
            _fits(model=["urgency"] * 4, condition=["fast", "fast", "slow", "slow"],
                  subject=["H01", "H02", "H01", "H02"], aic=[90.0] * 4, bic=[90.0] * 4),
        ])
    )
    assert set(paired["condition"]) == {"fast", "slow"}
    assert paired["n_subjects"].eq(2).all()


# --- held-out adjudication ----------------------------------------------------


def test_every_heldout_score_weights_folds_by_test_trial_count():
    """Unequal folds: a subject score is n_test-weighted, never a fold mean.

    A figure drawing per-subject dots as an unweighted fold mean would put them
    under an interval fitted to a different estimator, so all three entry
    points into the held-out scores must agree.
    """
    # Candidate -3.5/4, baseline -7/4, so the weighted contrast is 0.875.
    assert heldout_model_statistics(_UNEQUAL_FOLDS).iloc[0]["mean"] == pytest.approx(0.875)

    pairwise = heldout_pairwise_model_statistics(_UNEQUAL_FOLDS)
    row = pairwise.loc[
        (pairwise["model_a"] == "ddm") & (pairwise["model_b"] == "urgency")
    ].iloc[0]
    assert row["mean"] == pytest.approx(-0.875)
    assert row["n_subjects_favoring_model_b"] == 1
    assert row["n_subjects_favoring_model_a"] == 0

    # Fold sizes 10 and 90, so the two weightings differ by a wide margin.
    lopsided = pd.DataFrame([
        {"subject": "H01", "condition": "all", "model": model, "fold": fold,
         "n_test": n_test, "heldout_log_likelihood": per_trial * n_test,
         "heldout_log_likelihood_per_trial": per_trial}
        for fold, (n_test, urgency_score) in enumerate([(10, 0.0), (90, -1.0)])
        for model, per_trial in (("urgency", urgency_score), ("ddm", -2.0))
    ])
    difference = heldout_pairwise_subject_differences(lopsided)
    difference = difference.loc[
        (difference["model_a"] == "urgency") & (difference["model_b"] == "ddm")
    ]
    assert len(difference) == 1
    # Weighted: (0*10 + -1*90)/100 = -0.9 against ddm's -2.0.
    assert difference["difference"].iloc[0] == pytest.approx(1.1)
    # The unweighted fold mean would have been (0 + -1)/2 = -0.5 -> 1.5.
    assert difference["difference"].iloc[0] != pytest.approx(1.5)


def test_heldout_statistics_are_subject_paired_condition_stratified_and_symmetric():
    stats_table = heldout_model_statistics(_fold_rows())
    assert set(stats_table["condition"]) == {"fast", "slow"}
    assert stats_table["n_subjects"].eq(2).all()
    assert stats_table["criterion"].eq("heldout_log_likelihood_per_trial").all()

    pairwise = heldout_pairwise_model_statistics(
        _fold_rows(models=("ddm", "urgency", "collapsing"))
    )
    # Every unordered pair among three models in both conditions.
    assert len(pairwise) == 6
    assert set(zip(pairwise["model_a"], pairwise["model_b"])) == {
        ("collapsing", "ddm"), ("collapsing", "urgency"), ("ddm", "urgency"),
    }
    # collapsing is identical to ddm here, so the two contrasts against urgency
    # must agree.
    versus_collapsing = pairwise.loc[
        (pairwise["model_a"] == "collapsing") & (pairwise["model_b"] == "urgency")
    ]
    assert versus_collapsing["n_subjects"].eq(2).all()
    pd.testing.assert_series_equal(
        pairwise.loc[
            (pairwise["model_a"] == "ddm") & (pairwise["model_b"] == "urgency")
        ].set_index("condition")["mean"].sort_index(),
        versus_collapsing.set_index("condition")["mean"].sort_index(),
        check_names=False,
    )

    # Per-subject differences carry both orientations, exactly negated.
    differences = heldout_pairwise_subject_differences(
        pd.DataFrame([
            {"subject": subject, "condition": "all", "model": model, "fold": 0,
             "n_test": 50, "heldout_log_likelihood": score * 50,
             "heldout_log_likelihood_per_trial": score}
            for subject, offset in (("H01", 0.0), ("H02", 0.1))
            for model, score in (("urgency", -0.5 + offset), ("ddm", -0.8 + offset))
        ])
    )
    forward = differences.loc[
        (differences["model_a"] == "urgency") & (differences["model_b"] == "ddm"), "difference"
    ].to_numpy()
    reverse = differences.loc[
        (differences["model_a"] == "ddm") & (differences["model_b"] == "urgency"), "difference"
    ].to_numpy()
    assert forward == pytest.approx(-reverse)
    assert forward == pytest.approx([0.3, 0.3])


def test_a_heldout_cell_too_small_to_fit_is_persisted_with_its_fold_audit():
    heldout = heldout_model_evaluation(
        trial_features(dt_ms=[900.0, 1000.0, 1100.0, 1200.0]),
        models=("ddm",), folds=3, n_starts=1,
    )
    assert len(heldout) == 9  # all/fast/slow x model x three folds
    assert not heldout["converged"].any()
    assert heldout["fit_error"].eq("insufficient_train_or_test_trials").all()
    audit = heldout_fold_audit(heldout, expected_folds=3)
    assert audit["folds_complete"].all()
    assert audit["n_failed"].eq(3).all()


def test_heldout_contrast_robustness_reports_subsets_influence_and_pooling():
    heldout = pd.DataFrame([
        {"subject": subject, "condition": "all", "model": model, "fold": fold,
         "n_test": 10, "heldout_log_likelihood_per_trial": score}
        for subject, offset in zip([f"H{i:02d}" for i in range(1, 9)], range(8))
        for model, score in (("urgency", -1.0 + 0.01 * offset), ("collapsing", -1.1))
        for fold in range(3)
    ])
    fits = _fits(
        model=["urgency"] * 8, subject=[f"H{i:02d}" for i in range(1, 9)],
        condition=["all"] * 8, delta_bic=[-100.0] * 8,
        urgency_scale=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        urgency_scale_se=[np.nan, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    )
    fits["urgency_scale_lower_bound"] = 0.01
    fits["urgency_scale_upper_bound"] = 2.0

    summary = heldout_contrast_robustness(heldout, fits)

    assert set(summary["analysis"]) == {
        "ssm_heldout_identifiability_subset",
        "ssm_heldout_influence",
        "ssm_heldout_condition_pooling",
    }
    subsets = summary.loc[
        summary["analysis"] == "ssm_heldout_identifiability_subset"
    ].set_index("subset")
    # The two subjects pinned at the upper bound drop out of the identified
    # subset, and the one missing an SE from the SE subset.
    assert subsets.loc["all_subjects", "n_subjects"] == 8
    assert subsets.loc["urgency_scale_not_pinned", "n_subjects"] == 6
    assert subsets.loc["finite_observed_information_se", "n_subjects"] == 7
    influence = summary.loc[summary["analysis"] == "ssm_heldout_influence"]
    assert "least_favourable_subject" in influence.columns
    assert influence["sign_test_p"].notna().any()


# --- matched sequences and distribution checks --------------------------------


def test_matched_sequences_are_paired_on_stimulus_alone_never_on_outcome():
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
    assert matched["decision_after_convergence"].eq(matched["post_convergence_eligible"]).all()
    assert matched["decision_token_index"].isna().any()
    assert matched["matching_rule"].str.startswith("stimulus_only").all()
    assert matched_sequence_statistics(matched)["n_subjects"].eq(1).all()

    audit = matched_sequence_audit(matched, features).iloc[0]
    assert audit["n_stimulus_pairs"] == 5
    assert audit["n_post_convergence_pairs"] == 3
    assert audit["post_convergence_pair_fraction"] == pytest.approx(0.6)


def test_the_distribution_check_summary_keeps_the_pooled_trial_class_separate():
    """``trial_class`` carries its own ``all`` level; collapsing over it would
    average the pooled row together with the classes composing it."""
    summary = fitted_distribution_check_statistics(
        pd.DataFrame([
            {"subject": "H01", "condition": "all", "model": "urgency",
             "trial_class": trial_class, "outcome": "correct", "quantile": 0.5,
             "observed_rt_s": observed, "predicted_rt_s": 1.0}
            for trial_class, observed in (("all", 1.0), ("easy", 0.5), ("misleading", 1.5))
        ])
    )
    assert summary.loc[summary["trial_class"] == "all"].iloc[0]["mean_residual_s"] == pytest.approx(0.0)
    assert set(summary["trial_class"]) == {"all", "easy", "misleading"}


def test_distribution_checks_normalize_conditions_and_reuse_the_persisted_timecourse():
    features = trial_features(
        subject=["H01"] * 6,
        condition=["Fast", "Fast", "Fast", "Slow", "Slow", "Slow"],
        dt_ms=[100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
        isCorrect=[True, False, True, False, False, True],
    )
    fits = pd.DataFrame({
        "subject": ["H01"] * 3, "condition": ["all", "fast", "slow"],
        "model": ["ddm"] * 3, "converged": [True] * 3,
    })
    timecourse = pd.DataFrame([
        {"subject": "H01", "condition": condition, "model": "ddm",
         "trial_class": trial_class, "time_s": time_s,
         "predicted_density_correct": correct, "predicted_density_error": error}
        for condition in ("all", "fast", "slow")
        for trial_class in ("all", "easy")
        for time_s, correct, error in ((0.0, 0.0, 0.0), (0.1, 2.0, 1.0), (0.2, 2.0, 1.0), (0.3, 0.0, 0.0))
    ])

    checks = fitted_distribution_checks(features, fits, timecourse=timecourse)

    assert not checks.empty
    assert set(checks["condition"]) == {"all", "fast", "slow"}
    assert set(checks["outcome"]) == {"correct", "error"}
    assert set(checks["trial_class"]) == {"all", "easy"}
    assert checks["prediction_source"].eq("ssmtimecourse").all()
    median = checks.loc[
        (checks["condition"] == "all") & (checks["trial_class"] == "all")
        & (checks["outcome"] == "correct") & (checks["quantile"] == 0.5)
    ].iloc[0]
    assert median["n_observed"] == 3
    assert median["observed_rt_s"] == pytest.approx(0.3)
    assert median["predicted_correct_mass"] == pytest.approx(0.4)
    assert median["predicted_error_mass"] == pytest.approx(0.2)
    assert np.isfinite(median["predicted_rt_s"])


# --- recovery -----------------------------------------------------------------


def test_the_recovery_truth_design_is_a_deterministic_interior_permutation():
    first, first_fractions = _recovery_truth_values("urgency", repetition=0, repetitions=4, seed=0)
    second, second_fractions = _recovery_truth_values("urgency", repetition=1, repetitions=4, seed=0)
    assert any(first[name] != second[name] for name in first)
    assert first_fractions != second_fractions
    for parameter, value in first.items():
        lower, upper = PARAMETER_RANGES[parameter]
        assert lower < value < upper

    # At production size every parameter takes each of its twelve levels once.
    designs = [
        _recovery_truth_values("urgency", repetition=index, repetitions=12, seed=0)[0]
        for index in range(12)
    ]
    for parameter in MODEL_PARAMETERS["urgency"]:
        assert len({design[parameter] for design in designs}) == 12


def test_one_recovery_repetition_writes_complete_machine_readable_rows(monkeypatch):
    import meg_tokens.behavior.analyses.sequential_sampling as sequential_sampling

    monkeypatch.setattr(
        sequential_sampling, "_synthetic_frame",
        lambda *args, **kwargs: pd.DataFrame({"rt": [0.5] * 20}),
    )
    monkeypatch.setattr(
        sequential_sampling, "_fit_one",
        lambda model, frame, **kwargs: {
            **{parameter: 1.0 for parameter in MODEL_PARAMETERS[model]},
            "log_likelihood": -10.0, "optimizer_success": True, "boundary_hit": False,
        },
    )

    parameter = parameter_recovery(repetitions=12, repetition_indices=[7], truth_design_repetitions=12)
    model = model_recovery(repetitions=12, repetition_indices=[7], truth_design_repetitions=12)
    assert len(parameter) == sum(len(values) for values in MODEL_PARAMETERS.values())
    assert len(model) == len(MECHANISTIC_MODELS) ** 2
    assert set(parameter["repetition"]) == {7}
    assert set(model["repetition"]) == {7}

    summary = parameter_recovery_statistics(
        pd.DataFrame([
            {"true_model": "ddm", "parameter": "bound", "true_value": 1.0,
             "estimated_value": estimate, "converged": True, "boundary_hit": hit}
            for estimate, hit in ((1.1, False), (0.9, True))
        ])
    )
    assert {"bias", "rmse", "correlation", "convergence_rate", "boundary_rate"}.issubset(
        summary.columns
    )
    confusion = model_recovery_confusion(
        pd.DataFrame([
            {"repetition": 0, "true_model": "ddm", "fitted_model": "ddm", "selected_model": "ddm"},
            {"repetition": 0, "true_model": "ddm", "fitted_model": "urgency", "selected_model": "ddm"},
            {"repetition": 1, "true_model": "ddm", "fitted_model": "ddm", "selected_model": "urgency"},
        ])
    )
    assert confusion["n_repetitions"].eq(2).all()
    assert confusion["n_selected"].sum() == 2


# --- robustness and exclusion sensitivity -------------------------------------


def test_robustness_statistics_preserve_configuration_condition_model_and_pairing():
    rows = [
        {"subject": subject, "configuration": configuration, "condition": condition,
         "model": model, "delta_bic": offset if model == "urgency" else 0.0,
         "converged": True, "boundary_hit": False, "boundary_parameters": "",
         "urgency_scale": 2.0 if model == "urgency" else np.nan,
         "urgency_scale_lower_bound": 0.01, "urgency_scale_upper_bound": 2.0,
         "nondecision_s": 0.0, "nondecision_s_lower_bound": 0.0,
         "nondecision_s_upper_bound": 1.0}
        for subject in ("H01", "H02")
        for configuration, offset in (("baseline", 0.0), ("expanded_bounds", -1.0))
        for condition in ("all", "fast", "slow")
        for model in MECHANISTIC_MODELS
    ]
    summary = robustness_statistics(pd.DataFrame(rows))
    assert set(summary["analysis"]) == {
        "ssm_robustness_summary", "ssm_robustness_paired_sensitivity",
    }
    assert summary["configuration"].isin({"baseline", "expanded_bounds"}).all()
    assert "urgency_scale_near_upper" in summary


def test_the_exclusion_decomposition_separates_lost_trials_from_lost_evidence():
    """A strict subset that only loses trials must show no excess and no
    per-trial change; one that also gains evidence per trial must show both."""
    primary = _fits(
        model=["ddm", "urgency"], subject=["H01", "H01"], condition=["all", "all"],
        n_trials=[400, 400], n_parameters=[3, 5],
        log_likelihood=[-500.0, -400.0], delta_bic=[0.0, -187.0],
    )
    # Half the trials, identical evidence per trial: delta_ll halves.
    shrunk = _fits(
        model=["ddm", "urgency"], subject=["H01", "H01"], condition=["all", "all"],
        n_trials=[200, 200], n_parameters=[3, 5],
        log_likelihood=[-250.0, -200.0], delta_bic=[0.0, -89.4],
    )
    summary = exclusion_evidence_decomposition(shrunk, primary)
    decomposition = summary.loc[
        summary["analysis"] == "ssm_exclusion_evidence_decomposition"
    ].iloc[0]
    # Raw delta_bic roughly halves, but that is the trial count, not the
    # effect: the expectation absorbs it and the excess is ~0.
    assert decomposition["mean_expected_strict_delta_bic"] == pytest.approx(
        decomposition["mean_strict_delta_bic"], abs=1.0
    )
    assert decomposition["mean"] == pytest.approx(0.0, abs=1.0)
    assert decomposition["per_trial_change_mean"] == pytest.approx(0.0, abs=0.01)
    composition = summary.loc[
        summary["analysis"] == "ssm_exclusion_condition_composition"
    ].iloc[0]
    assert composition["retention_fraction"] == pytest.approx(0.5)
    assert composition["share_of_primary"] == pytest.approx(1.0)

    # `all` is one of the conditions, so summing rows would double-count.
    def _set(n_fast, n_slow, n_all, ll_scale):
        return pd.DataFrame([
            {"subject": "H01", "condition": condition, "model": model,
             "n_trials": n, "n_parameters": k, "log_likelihood": ll,
             "delta_bic": 0.0 if model == "ddm" else -10.0}
            for condition, n in (("all", n_all), ("fast", n_fast), ("slow", n_slow))
            for model, k, ll in (("ddm", 3, -n * 1.0), ("urgency", 5, -n * ll_scale))
        ])

    shares = exclusion_evidence_decomposition(_set(90, 30, 120, 0.8), _set(100, 100, 200, 0.8))
    shares = shares.loc[
        shares["analysis"] == "ssm_exclusion_condition_composition"
    ].set_index("condition")
    assert shares.loc["fast", "share_of_strict"] == pytest.approx(0.75)
    assert shares.loc["slow", "share_of_strict"] == pytest.approx(0.25)
    assert shares.loc["fast", "share_of_primary"] == pytest.approx(0.5)


def test_the_exclusion_refit_is_compared_within_the_same_population():
    primary = _fits(
        model=["ddm", "urgency"], subject=["H01", "H01"], condition=["all", "all"],
        delta_bic=[0.0, -2.0], boundary_hit=[False, False],
    )
    strict = primary.copy()
    strict["delta_bic"] = [0.0, -3.0]
    summary = exclusion_robustness_statistics(strict, primary)
    assert set(summary["model"]) == {"urgency"}
    assert summary.iloc[0]["mean"] == pytest.approx(-1.0)


# --- Fast versus Slow urgency -------------------------------------------------


def test_the_condition_contrast_pairs_each_subject_across_blocks():
    fits = pd.concat(
        [
            _fits(condition=["fast"] * 3, urgency_scale=[1.4, 1.2, 1.6]),
            _fits(condition=["slow"] * 3, urgency_scale=[1.0, 0.9, 1.1]),
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

    # A subject fitted in only one condition has no pair and is left out.
    unpaired = pd.concat(
        [
            _fits(condition=["fast"] * 3, urgency_scale=[1.4, 1.2, 1.6]),
            _fits(condition=["slow"] * 2, urgency_scale=[1.0, 0.9]),
        ],
        ignore_index=True,
    )
    assert urgency_condition_contrast(unpaired).set_index("parameter").loc[
        "urgency_scale", "n_subjects"
    ] == 2
