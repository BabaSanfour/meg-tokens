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
    TOKEN_INTERVAL_S,
    _additive_drive,
    _build_model,
    _evidence,
    empirical_lead_paths,
    _recovery_truth_values,
    _lead_path,
    fit_sequential_sampling_models,
    matched_sequence_diagnostic,
    matched_sequence_audit,
    matched_sequence_statistics,
    heldout_model_statistics,
    heldout_pairwise_model_statistics,
    heldout_pairwise_subject_differences,
    parameter_recovery,
    model_recovery,
    exclusion_robustness_statistics,
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


def test_both_recovery_studies_read_the_same_truth_for_a_repetition(monkeypatch):
    """Parameter recovery and model recovery are two questions about one
    simulation, so repetition 7 has to mean the same truth vector and the same
    dataset in both. Nothing downstream can detect a divergence: each study is
    internally consistent whichever truth it used, and they are published as
    separate tables. The check is here because it is the only place it fits.
    """
    import meg_tokens.behavior.analyses.sequential_sampling as sequential_sampling

    simulated = []

    def _record(model, values, *, n_per_sequence, seed, lead_paths):
        simulated.append((model, dict(values), seed))
        return pd.DataFrame({"rt": [0.5] * 20})

    monkeypatch.setattr(sequential_sampling, "_synthetic_frame", _record)
    monkeypatch.setattr(
        sequential_sampling, "_fit_one",
        lambda model, frame, **kwargs: {
            **{parameter: 1.0 for parameter in MODEL_PARAMETERS[model]},
            "log_likelihood": -10.0, "optimizer_success": True, "boundary_hit": False,
        },
    )

    parameter = parameter_recovery(
        repetitions=12, repetition_indices=[7], truth_design_repetitions=12
    )
    from_parameter, simulated[:] = list(simulated), []
    model = model_recovery(
        repetitions=12, repetition_indices=[7], truth_design_repetitions=12
    )

    assert from_parameter == simulated
    # And the tables agree on what they say that truth was.
    for column in ("truth_design_index", "truth_design_repetitions", "simulation_seed"):
        assert set(parameter[column]) == set(model[column])
    assert set(parameter["truth_source"]) == set(model["truth_source"])


# --- robustness and exclusion sensitivity -------------------------------------


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
