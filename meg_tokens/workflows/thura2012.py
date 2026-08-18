"""Cluster-side evaluation stages for the Thura et al. (2012) comparison."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
import subprocess
import sys
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from meg_tokens.behavior.analyses.sequential_sampling import (
    EVIDENCE_AFTER_LAST,
    FILTER_TAU_S,
    FIT_SEED,
    MECHANISTIC_MODELS,
    MIXTURE_COEF,
    MODEL_PARAMETERS,
    PARAMETER_RANGES,
    SOLVER_STEP_S,
    TOKEN_INTERVAL_S,
    boundary_convergence_audit,
    empirical_lead_paths,
    eligibility_audit,
    exclusion_robustness_audit,
    fit_mechanistic_model_set,
    fitted_distribution_checks,
    heldout_model_evaluation,
    heldout_fold_audit,
    heldout_model_statistics,
    heldout_pairwise_model_statistics,
    influential_subject_sensitivity,
    matched_sequence_diagnostic,
    matched_sequence_audit,
    matched_sequence_model_predictions,
    matched_sequence_model_statistics,
    matched_sequence_statistics,
    model_recovery,
    model_recovery_confusion,
    parameter_recovery,
    parameter_recovery_statistics,
    mechanistic_model_statistics,
    model_comparison_statistics,
    urgency_condition_contrast,
    population_parameters,
    robustness_grid,
    robustness_statistics,
    exclusion_robustness_statistics,
ROBUSTNESS_CONFIGURATIONS,
)
from meg_tokens.behavior.tables import read_trial_features
from meg_tokens.core import ProjectConfig, WorkflowResult, normalize_subject_id
from meg_tokens.io import (
    DerivativeLayout,
    load_table,
    require_file,
    save_table,
    sidecar_path,
)


EXCLUSION_ROBUSTNESS_RULES = (
    "complete_token_log_alignment",
)


def _package_version(name: str) -> str:
    """Return an installed package version without making provenance fragile."""
    try:
        return version(name)
    except PackageNotFoundError:
        return "unavailable"


def _analysis_scope(name: str) -> tuple[str, str]:
    """Describe the population/simulation scope represented by one derivative."""
    if name.startswith("ssmexclusion"):
        if name in {"ssmexclusionrefit", "ssmexclusioncompletetokenlogalignment"}:
            eligibility = (
                "primary_analysis_eligible & token_log_rows==15 & "
                "design_time_alignment_valid"
            )
        else:
            eligibility = (
                "primary_analysis_eligible; complete-token-log and "
                "canonical-alignment masks are reported as explicit audit/sensitivity views"
            )
        return (
            "thura2012_exclusion_robustness",
            eligibility,
        )
    if "recovery" in name:
        return (
            "thura2012_recovery",
            "synthetic simulations from empirical correct-target lead paths; no real-trial exclusions",
        )
    if name.startswith("ssmrobustness"):
        return (
            "thura2012_robustness",
            "canonical primary_analysis_eligible; named robustness configuration is stored in the table",
        )
    return (
        "thura2012_mechanistic_evaluation",
        "canonical primary_analysis_eligible; non-positive dt excluded only by first-passage scoring",
    )


def _selected_features(
    project: ProjectConfig,
    subjects: Optional[Sequence[str]],
) -> tuple[pd.DataFrame, Path, list[str], list[str]]:
    layout = DerivativeLayout(project.bids_root, task=project.task)
    path = layout.behavior_trial_features()
    features = read_trial_features(require_file(path, purpose="trial features"))
    excluded = set(project.subject_exclusions)
    if excluded:
        features = features.loc[~features["subject"].isin(excluded)]
    if subjects:
        wanted = {normalize_subject_id(subject) for subject in subjects}
        features = features.loc[features["subject"].isin(wanted)]
    if features.empty:
        raise ValueError("No trial-feature rows remain for Thura et al. analysis")
    return features, path, sorted(features["subject"].astype(str).unique()), sorted(excluded)


def _write(
    layout,
    name: str,
    table: pd.DataFrame,
    *,
    inputs: list[Path],
    subjects: list[str],
    excluded: list[str] | None = None,
    subject_level: bool = False,
    provenance: Mapping[str, Any] | None = None,
):
    path = (
        layout.behavior_subject_analysis(subjects[0], name)
        if subject_level and len(subjects) == 1
        else layout.behavior_analysis(name)
    )
    n_starts = (
        sorted(pd.to_numeric(table["n_starts"], errors="coerce").dropna().unique().tolist())
        if "n_starts" in table
        else []
    )
    stage, eligibility_scope = _analysis_scope(name)
    metadata = {
        "stage": stage,
        "analysis": name,
        "subjects": subjects,
        "excluded_subjects": excluded or [],
        "input_files": [str(value) for value in inputs],
        "models": list(MECHANISTIC_MODELS),
        "model_parameters": {model: list(parameters) for model, parameters in MODEL_PARAMETERS.items()},
        "parameter_ranges": {name: list(bounds) for name, bounds in PARAMETER_RANGES.items()},
        "condition_parameterization": "parameters are fit separately within each subject × condition × model cell; pooled statistics are subject-level",
        "timing": {
            "token_interval_s": TOKEN_INTERVAL_S,
            "decision_time_origin": "first token jump; dt_ms is motor-baseline-corrected",
            "evidence_after_last_token": EVIDENCE_AFTER_LAST,
        },
        "solver": {
            "dt_s": SOLVER_STEP_S,
            "state_grid_step": SOLVER_STEP_S,
            "state_grid_units": "decision-variable/noise units (not seconds)",
            "boundary": "absorbing finite-grid first passage",
            "overshoot_correction": "none; no separate correction beyond the numerical grid",
        },
        "likelihood": {
            "loss": "PyDDM LossLikelihood",
            "mixture_coef": MIXTURE_COEF,
            "mixture_distribution": "uniform contaminant",
        },
        "filter_tau_s": FILTER_TAU_S,
        "fit_seed": FIT_SEED,
        "simulation_seeds": sorted(
            pd.to_numeric(table["simulation_seed"], errors="coerce")
            .dropna().astype(int).unique().tolist()
        ) if "simulation_seed" in table else [],
        "recovery_repetition_indices": sorted(
            pd.to_numeric(table["repetition"], errors="coerce")
            .dropna().astype(int).unique().tolist()
        ) if "repetition" in table else [],
        "truth_design_repetitions": sorted(
            pd.to_numeric(table["truth_design_repetitions"], errors="coerce")
            .dropna().astype(int).unique().tolist()
        ) if "truth_design_repetitions" in table else [],
        "table_parameter_ranges": sorted(
            table["parameter_ranges"].dropna().astype(str).unique().tolist()
        ) if "parameter_ranges" in table else [],
        "recovery_truth_design": (
            "deterministic interior Latin-hypercube-like fractions; see "
            "truth_design_index/truth_design_fraction columns"
            if "truth_design_index" in table else None
        ),
        "n_starts": n_starts,
        "uncertainty": "observed-information finite-difference Hessian for final full-data fits; disabled for held-out/recovery/robustness candidates",
        "eligibility": eligibility_scope,
        "source": "Thura et al. (2012), doi:10.1152/jn.01071.2011",
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip(),
        "git_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
        ),
        "source_tree_sha256": _source_tree_sha256(),
        "python": sys.version,
        "software": {
            package: _package_version(package)
            for package in ("pyddm", "numpy", "pandas", "scipy")
        },
    }
    if provenance:
        metadata.update(dict(provenance))
    save_table(
        path,
        table.reset_index(drop=True),
        metadata=metadata,
    )
    return path


def _source_tree_sha256() -> str:
    """Hash the source/docs/scripts that define this analysis.

    The cluster deliberately receives an uncommitted working tree.  A commit
    field alone would therefore misidentify the code that generated a result;
    this deterministic manifest hash is stored alongside the dirty flag.
    """
    root = Path(__file__).resolve().parents[2]
    paths = sorted(
        path
        for base in (root / "meg_tokens", root / "scripts")
        if base.exists()
        for path in base.rglob("*")
        if path.is_file() and path.suffix in {".py", ".sh"}
    )
    paths.extend(
        path
        for path in (root / "docs" / "behavior.md", root / "docs" / "behavior_reporting_session_handoff.md")
        if path.is_file()
    )
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_heldout_coverage(
    heldout: pd.DataFrame,
    *,
    subjects: Sequence[str],
    folds: int,
) -> None:
    """Require one explicit held-out row for every subject/cell/model/fold."""
    required = {"subject", "condition", "model", "fold"}
    missing_columns = sorted(required.difference(heldout.columns))
    if missing_columns:
        raise ValueError(f"held-out derivative is missing coverage columns: {missing_columns}")
    conditions = ("all", "fast", "slow")
    expected = {
        (str(subject), condition, model, fold)
        for subject in subjects
        for condition in conditions
        for model in MECHANISTIC_MODELS
        for fold in range(int(folds))
    }
    observed: set[tuple[str, str, str, int]] = set()
    malformed = 0
    for row in heldout[["subject", "condition", "model", "fold"]].itertuples(index=False):
        try:
            fold = int(row.fold)
        except (TypeError, ValueError):
            malformed += 1
            continue
        observed.add((str(row.subject), str(row.condition).lower(), str(row.model), fold))
    if malformed or observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            "held-out coverage is incomplete: "
            f"expected {len(expected)} keys, observed {len(observed)}, "
            f"malformed={malformed}, missing={missing[:8]}, extra={extra[:8]}"
        )


def evaluate_thura2012(
    project: ProjectConfig,
    *,
    subjects: Optional[Sequence[str]] = None,
    folds: int = 3,
    n_starts: int = 1,
    recovery_repetitions: int = 0,
) -> WorkflowResult:
    """Run held-out evaluation, diagnostics, matching, recovery and checks."""
    layout = DerivativeLayout(project.bids_root, task=project.task)
    features, features_path, selected_subjects, excluded = _selected_features(project, subjects)
    fit_path = layout.behavior_analysis("ssmcomparison")
    fits = load_table(require_file(fit_path, purpose="complete mechanistic model fits"))
    fits = fits.loc[fits["subject"].astype(str).isin(selected_subjects)].copy()
    expected = len(selected_subjects) * 3 * len(MECHANISTIC_MODELS)
    if len(fits) != expected:
        raise ValueError(f"expected {expected} complete fit rows for selected subjects, found {len(fits)}")
    timecourse_path = layout.behavior_analysis("ssmtimecourse")
    primary_timecourse = load_table(
        require_file(timecourse_path, purpose="primary fitted time courses")
    )
    primary_timecourse = primary_timecourse.loc[
        primary_timecourse["subject"].astype(str).isin(selected_subjects)
    ].copy()
    aggregate, predictions = heldout_model_evaluation(
        features,
        models=MECHANISTIC_MODELS,
        folds=folds,
        n_starts=n_starts,
        return_predictions=True,
    )
    _validate_heldout_coverage(aggregate, subjects=selected_subjects, folds=folds)
    matched = matched_sequence_diagnostic(features)
    matched_predictions = matched_sequence_model_predictions(features, fits, matched)
    representative_paths = empirical_lead_paths(features)
    eligibility = eligibility_audit(features)
    boundary = boundary_convergence_audit(fits)
    if subjects is not None and len(selected_subjects) == 1:
        # Subject-scoped array outputs must remain attributable when they are
        # concatenated by the post-array merge stage.
        eligibility.insert(0, "subject", selected_subjects[0])
        boundary.insert(0, "subject", selected_subjects[0])
    outputs = []
    tables = {
        "ssmeligibilityaudit": eligibility,
        "ssmexclusionrobustness": exclusion_robustness_audit(features),
        "ssmboundaryaudit": boundary,
        "ssmmechanisticstats": mechanistic_model_statistics(fits),
        "ssmheldout": aggregate,
        "ssmheldoutfoldaudit": heldout_fold_audit(aggregate, expected_folds=folds),
        "ssmheldoutstats": heldout_model_statistics(aggregate),
        "ssmheldoutpredictions": predictions,
        "ssmdistributionchecks": fitted_distribution_checks(
            features, fits, timecourse=primary_timecourse
        ),
        "ssmmatchedsequence": matched,
        "ssmmatchedsequenceaudit": matched_sequence_audit(matched, features),
        "ssmmatchedsequencestats": matched_sequence_statistics(matched),
        "ssmmatchedsequencepredictions": matched_predictions,
        "ssmmatchedsequencepredstats": matched_sequence_model_statistics(matched_predictions),
    }
    if recovery_repetitions > 0:
        parameter = parameter_recovery(
                repetitions=recovery_repetitions,
                n_starts=n_starts,
                representative_paths=representative_paths,
            )
        model = model_recovery(
                repetitions=recovery_repetitions,
                n_starts=n_starts,
                representative_paths=representative_paths,
            )
        tables.update({
            "ssmparameterrecovery": parameter,
            "ssmparameterrecoverystats": parameter_recovery_statistics(parameter),
            "ssmmodelrecovery": model,
            "ssmmodelrecoverystats": model_recovery_confusion(model),
        })
    for name, table in tables.items():
        outputs.append(
            _write(
                layout,
                name,
                table,
                inputs=[features_path, fit_path, timecourse_path],
                subjects=selected_subjects,
                excluded=excluded,
                subject_level=subjects is not None,
            )
        )
    return WorkflowResult(
        stage="thura2012_mechanistic_evaluation",
        inputs=(features_path, fit_path, timecourse_path),
        outputs=tuple(outputs),
        settings={
            "subjects": selected_subjects,
            "models": list(MECHANISTIC_MODELS),
            "folds": folds,
            "n_starts": n_starts,
            "recovery_repetitions": recovery_repetitions,
            "reward_rate_claim": False,
            "reward_rate_reason": "payoff and verified intertrial timing are not present in the trial-feature contract",
        },
    )


_EVALUATION_TABLES = (
    "ssmeligibilityaudit",
    "ssmexclusionrobustness",
    "ssmboundaryaudit",
    "ssmmechanisticstats",
    "ssmheldout",
    "ssmheldoutfoldaudit",
    "ssmheldoutstats",
    "ssmheldoutpairwise",
    "ssmheldoutpredictions",
    "ssmdistributionchecks",
    "ssmmatchedsequence",
    "ssmmatchedsequenceaudit",
    "ssmmatchedsequencestats",
    "ssmmatchedsequencepredictions",
    "ssmmatchedsequencepredstats",
)


def aggregate_mechanistic_evaluation(
    project: ProjectConfig,
    *,
    folds: int = 3,
) -> WorkflowResult:
    """Merge subject-array diagnostics and recompute group-level statistics."""
    layout = DerivativeLayout(project.bids_root, task=project.task)
    features, features_path, subjects, excluded = _selected_features(project, None)
    fit_path = layout.behavior_analysis("ssmcomparison")
    fits = load_table(require_file(fit_path, purpose="complete mechanistic model fits"))
    fits = fits.loc[fits["subject"].astype(str).isin(subjects)].copy()
    expected = len(subjects) * 3 * len(MECHANISTIC_MODELS)
    if len(fits) != expected:
        raise ValueError(f"expected {expected} complete fit rows, found {len(fits)}")
    timecourse_path = layout.behavior_analysis("ssmtimecourse")
    primary_timecourse = load_table(
        require_file(timecourse_path, purpose="primary fitted time courses")
    )
    primary_timecourse = primary_timecourse.loc[
        primary_timecourse["subject"].astype(str).isin(subjects)
    ].copy()
    # Pool only subject-level raw records.  Group summaries are recomputed
    # below so single-subject inferential rows are never concatenated as if
    # they were group results.  Some subjects legitimately have no retained
    # post-convergence matched pairs, and their prediction TSV is empty; those
    # empty inputs remain provenance inputs but contribute no rows.
    recomputed = {
        "ssmexclusionrobustness",
        "ssmboundaryaudit",
        "ssmmechanisticstats",
        "ssmheldoutfoldaudit",
        "ssmheldoutstats",
        "ssmheldoutpairwise",
        "ssmdistributionchecks",
        "ssmmatchedsequenceaudit",
        "ssmmatchedsequencestats",
        "ssmmatchedsequencepredstats",
    }
    pooled: dict[str, list[pd.DataFrame]] = {
        name: [] for name in _EVALUATION_TABLES if name not in recomputed
    }
    inputs = [features_path, fit_path, timecourse_path]
    for subject in subjects:
        for name in _EVALUATION_TABLES:
            if name in recomputed:
                continue
            path = layout.behavior_subject_analysis(subject, name)
            if not path.is_file():
                raise FileNotFoundError(f"missing subject evaluation derivative: {path}")
            try:
                table = load_table(path)
            except pd.errors.EmptyDataError:
                table = pd.DataFrame()
            inputs.append(path)
            if table.empty:
                continue
            if "subject" not in table and name in {"ssmeligibilityaudit", "ssmboundaryaudit"}:
                table.insert(0, "subject", subject)
            pooled[name].append(table)
    tables = {
        name: (
            pd.concat(values, ignore_index=True)
            if values
            else pd.DataFrame()
        )
        for name, values in pooled.items()
    }
    expected_heldout_rows = len(subjects) * 3 * len(MECHANISTIC_MODELS) * folds
    if len(tables["ssmheldout"]) != expected_heldout_rows:
        raise ValueError(
            f"expected {expected_heldout_rows} held-out fold/model rows, "
            f"found {len(tables['ssmheldout'])}"
        )
    _validate_heldout_coverage(
        tables["ssmheldout"], subjects=subjects, folds=folds
    )
    tables["ssmboundaryaudit"] = boundary_convergence_audit(fits)
    tables["ssmexclusionrobustness"] = exclusion_robustness_audit(features)
    tables["ssmmechanisticstats"] = mechanistic_model_statistics(fits)
    tables["ssmdistributionchecks"] = fitted_distribution_checks(
        features,
        fits,
        timecourse=primary_timecourse,
    )
    tables["ssmheldoutfoldaudit"] = heldout_fold_audit(
        tables["ssmheldout"], expected_folds=folds
    )
    tables["ssmheldoutstats"] = heldout_model_statistics(tables["ssmheldout"])
    tables["ssmheldoutpairwise"] = heldout_pairwise_model_statistics(tables["ssmheldout"])
    tables["ssmmatchedsequencestats"] = matched_sequence_statistics(tables["ssmmatchedsequence"])
    tables["ssmmatchedsequenceaudit"] = matched_sequence_audit(
        tables["ssmmatchedsequence"], features
    )
    tables["ssmmatchedsequencepredstats"] = matched_sequence_model_statistics(
        tables["ssmmatchedsequencepredictions"]
    )
    outputs = []
    for name, table in tables.items():
        outputs.append(_write(layout, name, table, inputs=inputs, subjects=subjects, excluded=excluded))
    return WorkflowResult(
        stage="thura2012_mechanistic_evaluation_aggregation",
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        settings={
            "subjects": subjects,
            "tables": list(tables),
            "models": list(MECHANISTIC_MODELS),
            "folds": folds,
            "expected_heldout_rows": expected_heldout_rows,
        },
    )


def _validate_primary_fit_diagnostics(
    fit_table: pd.DataFrame,
    *,
    subject: str,
    required_n_starts: int = 3,
) -> None:
    """Reject smoke/legacy fits before production pooling.

    The subject array is deliberately restartable, so an old one-start smoke
    derivative can otherwise look structurally complete (12 rows) and silently
    contaminate the group table.  Production pooling requires every cell to
    carry the full requested restart diagnostics, including one objective and
    convergence flag per start.
    """
    required_columns = {
        "n_starts",
        "optimizer_success",
        "boundary_hit",
        "boundary_parameters",
        "start_objectives",
        "start_converged",
        "fit_error",
    }
    missing = sorted(required_columns.difference(fit_table.columns))
    if missing:
        raise ValueError(f"incomplete mechanistic fit for {subject}: missing diagnostics {missing}")
    starts = pd.to_numeric(fit_table["n_starts"], errors="coerce")
    if starts.isna().any() or not starts.eq(required_n_starts).all():
        observed = sorted(starts.dropna().astype(int).unique().tolist())
        raise ValueError(
            f"incomplete mechanistic fit for {subject}: expected n_starts={required_n_starts}, "
            f"observed {observed}"
        )
    for row_index, row in fit_table.iterrows():
        for field in ("start_objectives", "start_converged"):
            raw = row[field]
            try:
                values = json.loads(raw) if isinstance(raw, str) else raw
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"incomplete mechanistic fit for {subject} row {row_index}: invalid {field}"
                ) from error
            if not isinstance(values, dict) or len(values) != required_n_starts:
                count = len(values) if isinstance(values, dict) else 0
                raise ValueError(
                    f"incomplete mechanistic fit for {subject} row {row_index}: "
                    f"{field} has {count} starts, expected {required_n_starts}"
                )


def _subject_fit_provenance(path: Path, *, subject: str) -> tuple[str, str]:
    """Read and validate the source identity stored beside one fit table."""
    metadata_path = sidecar_path(path)
    if not metadata_path.is_file():
        raise ValueError(f"missing fit provenance sidecar for {subject}: {metadata_path}")
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {})
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid fit provenance sidecar for {subject}: {metadata_path}") from error
    source_hash = metadata.get("source_tree_sha256")
    commit = metadata.get("git_commit")
    if not isinstance(source_hash, str) or not source_hash:
        raise ValueError(f"fit provenance sidecar for {subject} has no source_tree_sha256")
    if not isinstance(commit, str) or not commit:
        raise ValueError(f"fit provenance sidecar for {subject} has no git_commit")
    return source_hash, commit


def aggregate_mechanistic_fits(project: ProjectConfig) -> WorkflowResult:
    """Pool only complete subject SSM outputs; never rewrite other analyses."""
    layout = DerivativeLayout(project.bids_root, task=project.task)
    features, features_path, subjects, excluded = _selected_features(project, None)
    names = ("ssmcomparison", "ssmtimecourse", "ssmtrialpredictions")
    pooled: dict[str, list[pd.DataFrame]] = {name: [] for name in names}
    inputs = [features_path]
    missing = []
    subject_provenance: dict[str, tuple[str, str]] = {}
    for subject in subjects:
        fit_path = layout.behavior_subject_analysis(subject, "ssmcomparison")
        if not fit_path.is_file():
            missing.append(subject)
            continue
        fit_table = load_table(fit_path)
        expected = len(("all", "fast", "slow")) * len(MECHANISTIC_MODELS)
        expected_keys = {
            (condition, model)
            for condition in ("all", "fast", "slow")
            for model in MECHANISTIC_MODELS
        }
        observed_keys = set(zip(fit_table.get("condition", []), fit_table.get("model", [])))
        if (
            len(fit_table) != expected
            or observed_keys != expected_keys
        ):
            raise ValueError(f"incomplete mechanistic fit for {subject}: expected {expected} rows")
        _validate_primary_fit_diagnostics(fit_table, subject=subject, required_n_starts=3)
        subject_provenance[subject] = _subject_fit_provenance(fit_path, subject=subject)
        for name in names:
            path = layout.behavior_subject_analysis(subject, name)
            if not path.is_file():
                raise FileNotFoundError(f"missing {name} for {subject}: {path}")
            pooled[name].append(load_table(path))
            inputs.append(path)
    if missing:
        raise ValueError(f"subject fit array incomplete; missing subjects: {missing}")
    tables = {name: pd.concat(values, ignore_index=True) for name, values in pooled.items()}
    fits = tables["ssmcomparison"]
    expected_total = len(subjects) * 3 * len(MECHANISTIC_MODELS)
    if len(fits) != expected_total:
        raise ValueError(f"expected {expected_total} fit rows, found {len(fits)}")
    provenance_values = set(subject_provenance.values())
    if len(provenance_values) != 1:
        raise ValueError(
            "subject mechanistic fits were generated by different source trees: "
            f"{subject_provenance}"
        )
    accepted_hash, accepted_commit = next(iter(provenance_values))
    accepted_provenance = {
        "accepted_subject_count": len(subject_provenance),
        "accepted_subject_source_tree_sha256": accepted_hash,
        "accepted_subject_git_commit": accepted_commit,
        "primary_fit_required_n_starts": 3,
    }
    outputs = []
    for name, table in tables.items():
        outputs.append(
            _write(
                layout,
                name,
                table,
                inputs=inputs,
                subjects=subjects,
                excluded=excluded,
                provenance=accepted_provenance,
            )
        )
    stats_table = pd.concat(
        [mechanistic_model_statistics(fits), model_comparison_statistics(fits), urgency_condition_contrast(fits)],
        ignore_index=True,
    )
    outputs.append(
        _write(
            layout,
            "ssmcomparisonstats",
            stats_table,
            inputs=inputs,
            subjects=subjects,
            excluded=excluded,
            provenance=accepted_provenance,
        )
    )
    outputs.append(
        _write(
            layout,
            "ssmexclusionsensitivity",
            influential_subject_sensitivity(fits, excluded_subject="H20"),
            inputs=inputs,
            subjects=subjects,
            excluded=excluded,
            provenance=accepted_provenance,
        )
    )
    estimates, population = population_parameters(fits)
    outputs.append(
        _write(
            layout,
            "ssmpopulation",
            estimates,
            inputs=inputs,
            subjects=subjects,
            excluded=excluded,
            provenance=accepted_provenance,
        )
    )
    outputs.append(
        _write(
            layout,
            "ssmpopulationstats",
            population,
            inputs=inputs,
            subjects=subjects,
            excluded=excluded,
            provenance=accepted_provenance,
        )
    )
    return WorkflowResult(
        stage="thura2012_mechanistic_fit_aggregation",
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        settings={
            "subjects": subjects,
            "n_fit_rows": int(len(fits)),
            "models": list(MECHANISTIC_MODELS),
            **accepted_provenance,
        },
    )


def run_recovery_stage(
    project: ProjectConfig,
    *,
    repetitions: int = 12,
    n_starts: int = 2,
    repetition_index: int | None = None,
) -> WorkflowResult:
    """Run one or all deterministic recovery truth points on a compute node.

    The Slurm array passes ``repetition_index`` so each task writes unique
    repetition-scoped derivatives.  ``aggregate_recovery_stage`` validates
    and pools them only after every requested repetition is present.
    """
    if repetitions < 8:
        raise ValueError("recovery requires at least 8 truth repetitions")
    if repetition_index is not None and not 0 <= repetition_index < repetitions:
        raise ValueError("repetition_index must lie within repetitions")
    layout = DerivativeLayout(project.bids_root, task=project.task)
    features, features_path, subjects, excluded = _selected_features(project, None)
    representative_paths = empirical_lead_paths(features)
    inputs = [features_path]
    indices = (repetition_index,) if repetition_index is not None else None
    parameter = parameter_recovery(
        repetitions=repetitions,
        n_starts=n_starts,
        representative_paths=representative_paths,
        repetition_indices=indices,
        truth_design_repetitions=repetitions,
    )
    model = model_recovery(
        repetitions=repetitions,
        n_starts=n_starts,
        representative_paths=representative_paths,
        repetition_indices=indices,
        truth_design_repetitions=repetitions,
    )
    expected_parameter_rows = sum(
        len(MODEL_PARAMETERS[model_name]) for model_name in MECHANISTIC_MODELS
    ) * (1 if repetition_index is not None else repetitions)
    expected_model_rows = len(MECHANISTIC_MODELS) ** 2 * (
        1 if repetition_index is not None else repetitions
    )
    if len(parameter) != expected_parameter_rows or len(model) != expected_model_rows:
        raise ValueError(
            "recovery row-count validation failed: "
            f"parameter={len(parameter)}/{expected_parameter_rows}, "
            f"model={len(model)}/{expected_model_rows}"
        )
    suffix = f"rep{repetition_index:03d}" if repetition_index is not None else ""
    parameter_name = "ssmparameterrecovery" + suffix
    parameter_stats_name = "ssmparameterrecoverystats" + suffix
    model_name = "ssmmodelrecovery" + suffix
    model_stats_name = "ssmmodelrecoverystats" + suffix
    outputs = [
        _write(layout, parameter_name, parameter, inputs=inputs, subjects=subjects, excluded=excluded),
        _write(
            layout,
            parameter_stats_name,
            parameter_recovery_statistics(parameter),
            inputs=inputs,
            subjects=subjects,
            excluded=excluded,
        ),
        _write(layout, model_name, model, inputs=inputs, subjects=subjects, excluded=excluded),
        _write(
            layout,
            model_stats_name,
            model_recovery_confusion(model),
            inputs=inputs,
            subjects=subjects,
            excluded=excluded,
        ),
    ]
    return WorkflowResult(
        stage="thura2012_recovery",
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        settings={
            "repetitions": repetitions,
            "repetition_index": repetition_index,
            "n_starts": n_starts,
            "expected_parameter_rows": expected_parameter_rows,
            "expected_model_rows": expected_model_rows,
            "models": list(MECHANISTIC_MODELS),
        },
    )


def aggregate_recovery_stage(
    project: ProjectConfig,
    *,
    repetitions: int = 12,
) -> WorkflowResult:
    """Validate and merge repetition-scoped recovery array derivatives."""
    if repetitions < 8:
        raise ValueError("recovery requires at least 8 truth repetitions")
    layout = DerivativeLayout(project.bids_root, task=project.task)
    features, features_path, subjects, excluded = _selected_features(project, None)
    parameter_tables = []
    model_tables = []
    inputs = [features_path]
    expected_parameter_rows = sum(
        len(MODEL_PARAMETERS[model_name]) for model_name in MECHANISTIC_MODELS
    )
    expected_model_rows = len(MECHANISTIC_MODELS) ** 2
    for repetition in range(repetitions):
        parameter_path = layout.behavior_analysis(
            f"ssmparameterrecoveryrep{repetition:03d}"
        )
        model_path = layout.behavior_analysis(
            f"ssmmodelrecoveryrep{repetition:03d}"
        )
        if not parameter_path.is_file() or not model_path.is_file():
            missing = [str(path) for path in (parameter_path, model_path) if not path.is_file()]
            raise FileNotFoundError(f"missing recovery repetition {repetition}: {missing}")
        parameter = load_table(parameter_path)
        model = load_table(model_path)
        if len(parameter) != expected_parameter_rows:
            raise ValueError(
                f"recovery repetition {repetition} has {len(parameter)} parameter rows; "
                f"expected {expected_parameter_rows}"
            )
        if len(model) != expected_model_rows:
            raise ValueError(
                f"recovery repetition {repetition} has {len(model)} model rows; "
                f"expected {expected_model_rows}"
            )
        if set(pd.to_numeric(parameter["repetition"], errors="coerce")) != {repetition}:
            raise ValueError(f"parameter recovery repetition label mismatch for {repetition}")
        if set(pd.to_numeric(model["repetition"], errors="coerce")) != {repetition}:
            raise ValueError(f"model recovery repetition label mismatch for {repetition}")
        if set(parameter["true_model"]) != set(MECHANISTIC_MODELS):
            raise ValueError(f"parameter recovery model coverage mismatch for {repetition}")
        if set(model["true_model"]) != set(MECHANISTIC_MODELS) or set(model["fitted_model"]) != set(MECHANISTIC_MODELS):
            raise ValueError(f"model recovery coverage mismatch for {repetition}")
        parameter_keys = set(
            zip(parameter["true_model"].astype(str), parameter["parameter"].astype(str), strict=False)
        )
        expected_parameter_keys = {
            (model_name, parameter_name)
            for model_name in MECHANISTIC_MODELS
            for parameter_name in MODEL_PARAMETERS[model_name]
        }
        if parameter_keys != expected_parameter_keys:
            raise ValueError(f"parameter recovery key coverage mismatch for {repetition}")
        model_keys = set(
            zip(model["true_model"].astype(str), model["fitted_model"].astype(str), strict=False)
        )
        expected_model_keys = {
            (true_model, fitted_model)
            for true_model in MECHANISTIC_MODELS
            for fitted_model in MECHANISTIC_MODELS
        }
        if model_keys != expected_model_keys:
            raise ValueError(f"model recovery key coverage mismatch for {repetition}")
        parameter_tables.append(parameter)
        model_tables.append(model)
        inputs.extend([parameter_path, model_path])
    parameter = pd.concat(parameter_tables, ignore_index=True)
    model = pd.concat(model_tables, ignore_index=True)
    if len(parameter) != repetitions * expected_parameter_rows or len(model) != repetitions * expected_model_rows:
        raise ValueError("pooled recovery row-count validation failed")
    outputs = [
        _write(layout, "ssmparameterrecovery", parameter, inputs=inputs, subjects=subjects, excluded=excluded),
        _write(layout, "ssmparameterrecoverystats", parameter_recovery_statistics(parameter), inputs=inputs, subjects=subjects, excluded=excluded),
        _write(layout, "ssmmodelrecovery", model, inputs=inputs, subjects=subjects, excluded=excluded),
        _write(layout, "ssmmodelrecoverystats", model_recovery_confusion(model), inputs=inputs, subjects=subjects, excluded=excluded),
    ]
    return WorkflowResult(
        stage="thura2012_recovery_aggregation",
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        settings={
            "repetitions": repetitions,
            "expected_parameter_rows": int(len(parameter)),
            "expected_model_rows": int(len(model)),
            "models": list(MECHANISTIC_MODELS),
        },
    )


def run_robustness_stage(
    project: ProjectConfig,
    *,
    subject: str | None = None,
    configuration: str | None = None,
    n_jobs: int = 1,
    n_starts: int = 1,
) -> WorkflowResult:
    """Run the prespecified numerical/modeling robustness grid on a node.

    This is deliberately separate from the fit and evaluation stages: each
    configuration refits the complete model set and can be restarted without
    rewriting pooled fits or predictive derivatives.
    """
    layout = DerivativeLayout(project.bids_root, task=project.task)
    if subject and configuration not in ROBUSTNESS_CONFIGURATIONS:
        raise ValueError(
            "subject-scoped robustness requires one of "
            f"{list(ROBUSTNESS_CONFIGURATIONS)}"
        )
    selected = [subject] if subject else None
    features, features_path, subjects, excluded = _selected_features(project, selected)
    requested = (configuration,) if configuration else None
    table = robustness_grid(
        features,
        n_jobs=n_jobs,
        n_starts=n_starts,
        configurations=requested,
    )
    if subject:
        name = "ssmrobustness" + (configuration or "all").replace("_", "")
        output_subject_level = True
    else:
        name = "ssmrobustness"
        output_subject_level = False
    output = _write(
        layout,
        name,
        table,
        inputs=[features_path],
        subjects=subjects,
        excluded=excluded,
        subject_level=output_subject_level,
    )
    return WorkflowResult(
        stage="thura2012_robustness",
        inputs=(features_path,),
        outputs=(output,),
        settings={
            "n_jobs": n_jobs,
            "n_starts": n_starts,
            "configurations": sorted(table["configuration"].unique()) if not table.empty else [],
            "subject": subject,
            "models": list(MECHANISTIC_MODELS),
        },
    )


def run_exclusion_robustness_stage(
    project: ProjectConfig,
    *,
    subject: str | None = None,
    rule: str | None = None,
    n_jobs: int = 1,
    n_starts: int = 1,
) -> WorkflowResult:
    """Refit the four-model set under explicit data-quality sensitivities."""
    layout = DerivativeLayout(project.bids_root, task=project.task)
    if rule is not None and rule not in EXCLUSION_ROBUSTNESS_RULES:
        raise ValueError(f"unknown exclusion robustness rule: {rule!r}")
    selected = [subject] if subject else None
    features, features_path, subjects, excluded = _selected_features(project, selected)
    eligible = features["primary_analysis_eligible"].fillna(False).astype(bool)
    complete = pd.to_numeric(features["token_log_rows"], errors="coerce").eq(15)
    alignment = features.get(
        "design_time_alignment_valid", pd.Series(False, index=features.index)
    ).fillna(False).astype(bool)
    if rule == "complete_token_log_alignment":
        # The intersection is the strict quality view.  The audit derivative
        # records whether complete-log and canonical-alignment masks are
        # exactly equivalent in this dataset, avoiding duplicate refits when
        # they are.
        keep = eligible & complete & alignment
    else:
        keep = eligible
    filtered = features.loc[keep].copy()
    table = fit_mechanistic_model_set(
        filtered,
        n_jobs=n_jobs,
        n_starts=n_starts,
        compute_uncertainty=False,
    )
    table.insert(0, "robustness_rule", rule or "primary_eligibility")
    table["n_retained_trials"] = int(len(filtered))
    table["n_retained_subjects"] = int(filtered["subject"].nunique())
    name = "ssmexclusion" + (rule or "primaryeligibility").replace("_", "")
    output = _write(
        layout,
        name,
        table,
        inputs=[features_path],
        subjects=subjects,
        excluded=excluded,
        subject_level=subject is not None and len(subjects) == 1,
    )
    return WorkflowResult(
        stage="thura2012_exclusion_robustness",
        inputs=(features_path,),
        outputs=(output,),
        settings={
            "rule": rule or "primary_eligibility",
            "subjects": subjects,
            "n_retained_trials": int(len(filtered)),
            "n_starts": n_starts,
            "models": list(MECHANISTIC_MODELS),
        },
    )


def aggregate_exclusion_robustness_stage(project: ProjectConfig) -> WorkflowResult:
    """Merge subject × rule exclusion-refit outputs with row-count checks."""
    layout = DerivativeLayout(project.bids_root, task=project.task)
    features, features_path, subjects, excluded = _selected_features(project, None)
    tables = []
    inputs = [features_path]
    for subject in subjects:
        for rule in EXCLUSION_ROBUSTNESS_RULES:
            name = "ssmexclusion" + rule.replace("_", "")
            path = layout.behavior_subject_analysis(subject, name)
            if not path.is_file():
                raise FileNotFoundError(f"missing exclusion robustness output: {path}")
            table = load_table(path)
            if "subject" not in table:
                table.insert(0, "subject", subject)
            tables.append(table)
            inputs.append(path)
    merged = pd.concat(tables, ignore_index=True)
    expected = len(subjects) * len(EXCLUSION_ROBUSTNESS_RULES) * len(MECHANISTIC_MODELS) * 3
    if len(merged) != expected:
        raise ValueError(f"expected {expected} exclusion-refit rows, found {len(merged)}")
    primary_path = layout.behavior_analysis("ssmcomparison")
    primary = load_table(require_file(primary_path, purpose="primary mechanistic fits"))
    inputs.append(primary_path)
    audit = exclusion_robustness_audit(features)
    output = _write(
        layout, "ssmexclusionrefit", merged, inputs=inputs,
        subjects=subjects, excluded=excluded,
    )
    stats_output = _write(
        layout,
        "ssmexclusionrobustnessstats",
        exclusion_robustness_statistics(merged, primary, audit=audit),
        inputs=inputs,
        subjects=subjects,
        excluded=excluded,
    )
    return WorkflowResult(
        stage="thura2012_exclusion_robustness_aggregation",
        inputs=tuple(inputs),
        outputs=(output, stats_output),
        settings={"subjects": subjects, "n_rows": int(len(merged)), "strict_rule": "complete_token_log_alignment"},
    )


def aggregate_robustness_stage(project: ProjectConfig) -> WorkflowResult:
    """Merge the subject × configuration robustness array outputs."""
    layout = DerivativeLayout(project.bids_root, task=project.task)
    features, features_path, subjects, excluded = _selected_features(project, None)
    tables = []
    inputs = [features_path]
    for subject in subjects:
        for configuration in ROBUSTNESS_CONFIGURATIONS:
            name = "ssmrobustness" + configuration.replace("_", "")
            path = layout.behavior_subject_analysis(subject, name)
            if not path.is_file():
                raise FileNotFoundError(f"missing robustness array output: {path}")
            table = load_table(path)
            table.insert(0, "subject", subject) if "subject" not in table else None
            tables.append(table)
            inputs.append(path)
    merged = pd.concat(tables, ignore_index=True)
    expected = (
        len(subjects) * len(ROBUSTNESS_CONFIGURATIONS) * 3
        * len(MECHANISTIC_MODELS)
    )
    if len(merged) != expected:
        raise ValueError(f"expected {expected} robustness rows, found {len(merged)}")
    if set(merged["configuration"]) != set(ROBUSTNESS_CONFIGURATIONS):
        raise ValueError("robustness configuration coverage is incomplete")
    if set(merged["model"]) != set(MECHANISTIC_MODELS):
        raise ValueError("robustness model coverage is incomplete")
    if set(merged["condition"]) != {"all", "fast", "slow"}:
        raise ValueError("robustness condition coverage is incomplete")
    output = _write(
        layout,
        "ssmrobustness",
        merged,
        inputs=inputs,
        subjects=subjects,
        excluded=excluded,
    )
    stats_output = _write(
        layout,
        "ssmrobustnessstats",
        robustness_statistics(merged),
        inputs=inputs,
        subjects=subjects,
        excluded=excluded,
    )
    return WorkflowResult(
        stage="thura2012_robustness_aggregation",
        inputs=tuple(inputs),
        outputs=(output, stats_output),
        settings={
            "subjects": subjects,
            "n_rows": int(len(merged)),
            "models": list(MECHANISTIC_MODELS),
            "conditions": ["all", "fast", "slow"],
            "configurations": list(ROBUSTNESS_CONFIGURATIONS),
        },
    )
