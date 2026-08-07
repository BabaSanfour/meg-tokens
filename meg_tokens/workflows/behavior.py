"""Behavior ingestion and summary workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from meg_tokens.behavior.metrics import (
    analyze_logged_spd,
    analyze_post_error_slowing,
    analyze_trial_classes,
    behavior_group_statistics,
    calculate_motor_baseline,
    compare_correct_error,
    compare_fast_slow,
    logged_spd_at_decision,
)
from meg_tokens.behavior.evidence import (
    evidence_after_tokens,
    log_posterior_odds,
    parse_token_directions,
    token_lead_profile,
)
from meg_tokens.behavior.success_probability import success_probability_profile
from meg_tokens.behavior.tdms import (
    OUTCOME_NEVER_STARTED,
    TdmsRunInfo,
    add_run_metadata,
    parse_tdms_file,
    parse_tdms_filename,
    started_trials,
    validate_behavior_dataframe,
)
from meg_tokens.behavior.trials import OUTCOME_LABELS
from meg_tokens.core import ProjectConfig, WorkflowResult, normalize_subject_id
from meg_tokens.io import DerivativeLayout, save_table


def _subject_input_dir(root: Path, subject: str) -> Path:
    canonical = normalize_subject_id(subject)
    candidates = [
        root / subject,
        root / canonical,
        root / f"H{int(canonical[1:])}",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Subject input directory does not exist for {subject} under {root}"
    )


def ingest_subject_behavior(
    subject: str,
    *,
    input_root: str | Path,
    output_root: str | Path,
    dry_run: bool = False,
    ignore_files: Sequence[str] = (),
    infer_random_classes: bool = True,
) -> tuple[dict[str, object], ...]:
    """Parse every standard TDMS run for one subject.

    Every ``*.tdms`` file under the subject directory must either match the
    canonical ``H<subject><Condition><run>_<YYMMDD>.tdms`` naming pattern or
    have its filename listed in ``ignore_files``. A file that matches
    neither is a data-quality problem, not something to skip quietly, so it
    raises instead of being dropped. The same guard applies to duplicate
    ``(subject, condition, run)`` combinations, which would otherwise
    silently overwrite one another's derivative output.
    """
    input_root = Path(input_root)
    subject_dir = _subject_input_dir(input_root, subject)
    layout = DerivativeLayout(output_root)
    ignore = set(ignore_files)

    candidates = sorted(subject_dir.glob("*.tdms"))
    parsed: list[tuple[Path, TdmsRunInfo]] = []
    unmatched: list[Path] = []
    for input_path in candidates:
        if input_path.name in ignore:
            continue
        try:
            run_info = parse_tdms_filename(input_path.name)
        except ValueError:
            unmatched.append(input_path)
            continue
        parsed.append((input_path, run_info))

    if unmatched:
        raise ValueError(
            "Refusing to silently skip .tdms files with non-canonical "
            f"names under {subject_dir}: "
            f"{[path.name for path in unmatched]}. Rename them to match "
            "H<subject><Condition><run>_<YYMMDD>.tdms, or pass their exact "
            "filenames via ignore_files (behavior_ignore_files in the "
            "project TOML) to exclude them explicitly."
        )

    seen: dict[tuple[str, str, str], Path] = {}
    for input_path, run_info in parsed:
        key = (run_info.subject, run_info.condition, run_info.run)
        if key in seen:
            raise ValueError(
                f"Duplicate TDMS run for subject={key[0]} "
                f"condition={key[1]} run={key[2]}: found in both "
                f"{seen[key].name} and {input_path.name}. Resolve the "
                "collision before ingesting."
            )
        seen[key] = input_path

    records = []
    for input_path, run_info in parsed:
        output_path = layout.behavior(
            subject=run_info.subject,
            run=run_info.run,
            condition=run_info.condition,
        )
        trial_count = 0
        if not dry_run:
            table = add_run_metadata(
                parse_tdms_file(
                    str(input_path),
                    infer_random_classes=infer_random_classes,
                ),
                run_info,
                input_path.name,
            )
            validate_behavior_dataframe(table)
            save_table(
                output_path,
                table,
                metadata={
                    "stage": "behavior_parsing",
                    "subject": run_info.subject,
                    "condition": run_info.condition,
                    "run": run_info.run,
                    "source_file": input_path.name,
                    "source_date": run_info.date,
                },
            )
            trial_count = len(table)
        records.append(
            {
                "subject": run_info.subject,
                "condition": run_info.condition,
                "run": run_info.run,
                "input": str(input_path),
                "output": str(output_path),
                "trials": trial_count,
            }
        )
    return tuple(records)


def ingest_behavior(
    project: ProjectConfig,
    *,
    subjects: Optional[Sequence[str]] = None,
    dry_run: bool = False,
) -> WorkflowResult:
    """Ingest selected subjects from the configured behavioral source root."""
    if project.behavior_root is None:
        raise ValueError("Project configuration requires behavior_root for ingestion")
    if not project.behavior_root.is_dir():
        raise FileNotFoundError(
            f"Behavior input root does not exist: {project.behavior_root}"
        )

    selected = list(subjects) if subjects else sorted(
        path.name
        for path in project.behavior_root.iterdir()
        if path.is_dir() and path.name.upper().startswith("H")
    )
    if not selected:
        raise FileNotFoundError(
            f"No subject directories were found under {project.behavior_root}"
        )

    records = []
    for subject in selected:
        records.extend(
            ingest_subject_behavior(
                subject,
                input_root=project.behavior_root,
                output_root=project.bids_root,
                dry_run=dry_run,
                ignore_files=project.behavior_ignore_files,
                infer_random_classes=project.infer_random_classes,
            )
        )
    return WorkflowResult(
        stage="behavior_ingestion",
        inputs=tuple(Path(record["input"]) for record in records),
        outputs=tuple(Path(record["output"]) for record in records),
        settings={
            "subjects": [normalize_subject_id(subject) for subject in selected],
            "dry_run": dry_run,
            "n_runs": len(records),
            "infer_random_classes": project.infer_random_classes,
        },
    )


def _condition_runs(tables: Sequence[pd.DataFrame], condition: str) -> list[pd.DataFrame]:
    return [
        table
        for table in tables
        if not table.empty
        and str(table["condition"].iloc[0]).lower() == condition.lower()
    ]


def _started_condition_runs(
    tables: Sequence[pd.DataFrame],
    condition: str,
) -> list[pd.DataFrame]:
    """Return analysis views containing only trials that received a go cue."""
    return [
        started_trials(table)
        for table in _condition_runs(tables, condition)
    ]


def _design_evidence(
    token_directions: object,
    *,
    target: int,
    n_tokens_at_decision: int,
) -> dict[str, object]:
    """Return designed evidence for one trial, referenced to ``target``.

    Evidence is derived from the designed token sequence rather than the
    runtime log so that it exists for every trial, including the short 14-row
    logs that forbid design-to-runtime time alignment. ``early`` is the state
    after three jumps, the horizon the preprint's misleading rule uses.
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
    probability = evidence_after_tokens(profile, n_tokens_at_decision, prior=0.5)
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


def _trial_feature_table(
    by_subject: dict[str, list[pd.DataFrame]],
    motor_baselines: dict[str, float],
) -> pd.DataFrame:
    """Build one MEG-joinable feature row per staged behavior trial."""
    rows = []
    class_names = {0: "unclassified", 1: "easy", 2: "ambiguous", 3: "misleading"}
    for subject, tables in sorted(by_subject.items()):
        motor_baseline = motor_baselines[subject]
        for table in tables:
            condition = str(table["condition"].iloc[0])
            run = int(table["run"].iloc[0])
            started_index = 0
            for run_trial_index, (_, trial) in enumerate(table.iterrows(), start=1):
                is_started = int(trial["nOutcome"]) != OUTCOME_NEVER_STARTED
                if is_started:
                    started_index += 1
                raw_rt = pd.to_numeric(pd.Series([trial["rawRT"]]), errors="coerce").iloc[0]
                has_choice = bool(pd.notna(raw_rt))
                is_task = condition.lower() in {"fast", "slow"}
                dt = float(raw_rt) - motor_baseline if is_task and has_choice else float("nan")

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
                    int(decision_token_index) if pd.notna(decision_token_index) else 0
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
                    "started_trial_index": started_index if is_started else pd.NA,
                    "nTrialIndex": int(trial["nTrialIndex"]),
                    # LabVIEW session clock at trial onset. It is the only
                    # field that orders blocks within a session: the filenames
                    # carry a date but no time, and nTrialIndex restarts at 1
                    # in every run. Fast and Slow blocks are interleaved, so
                    # session-drift analyses need this ordering to exist.
                    "initial_time_ms": int(trial["nInitialTime"]),
                    "source_file": trial["source_file"],
                    "trial_class": trial_class,
                    "trial_class_name": class_names.get(trial_class, "unclassified"),
                    "sTrialClassRaw": trial["sTrialClassRaw"],
                    "trial_class_source": trial["trial_class_source"],
                    "trial_class_rule": trial["trial_class_rule"],
                    "nChoiceMade": choice_side,
                    "nCorrectChoice": correct_side,
                    "choice_side": choice_side if has_choice else pd.NA,
                    "correct_side": correct_side if correct_side in (1, 2) else pd.NA,
                    "isCorrect": trial["isCorrect"],
                    "nOutcome": outcome,
                    "outcome_label": OUTCOME_LABELS.get(outcome, "completed"),
                    "motor_baseline_ms": motor_baseline,
                    "rawRT": raw_rt,
                    "dt_ms": dt,
                    "logged_spd": spd,
                    # Chosen-target evidence on the log-odds scale, which is
                    # the scale the urgency-gating literature reports the
                    # accuracy criterion on.
                    "logged_spd_log_odds": (
                        log_posterior_odds(spd)[0] if pd.notna(spd) else float("nan")
                    ),
                    "logged_spd_validated_15row": spd if log_rows == 15 else float("nan"),
                    "evidence_at_decision": spd - 0.5 if pd.notna(spd) else float("nan"),
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
    return pd.DataFrame(rows)


def analyze_behavior(
    project: ProjectConfig,
    *,
    subjects: Optional[Sequence[str]] = None,
) -> WorkflowResult:
    """Compute one validated behavioral summary row per subject."""
    layout = DerivativeLayout(
        project.bids_root,
        pipeline=project.pipeline,
        task=project.task,
    )
    paths = layout.behavior_tables(subjects=subjects)
    excluded_subjects = set(project.subject_exclusions)
    by_subject: dict[str, list[pd.DataFrame]] = {}
    source_paths: dict[str, list[Path]] = {}
    for path in paths:
        table = pd.read_csv(path, sep="\t")
        validate_behavior_dataframe(table)
        if table.empty:
            continue
        subject = normalize_subject_id(str(table["subject"].iloc[0]))
        if subject in excluded_subjects:
            continue
        by_subject.setdefault(subject, []).append(table)
        source_paths.setdefault(subject, []).append(path)
    if not by_subject:
        raise ValueError("Behavior derivatives do not contain any trials")

    rows = []
    motor_baselines = {}
    for subject, tables in sorted(by_subject.items()):
        n_never_started = sum(
            int((table["nOutcome"] == OUTCOME_NEVER_STARTED).sum())
            for table in tables
        )
        rt_runs = _started_condition_runs(tables, "RT")
        fast_runs = _started_condition_runs(tables, "Fast")
        slow_runs = _started_condition_runs(tables, "Slow")
        task_runs = fast_runs + slow_runs
        motor_baseline = calculate_motor_baseline(rt_runs)
        motor_baselines[subject] = motor_baseline
        speed = compare_fast_slow(fast_runs, slow_runs, motor_baseline)
        accuracy = compare_correct_error(task_runs, motor_baseline)
        fast_accuracy = compare_correct_error(fast_runs, motor_baseline)
        slow_accuracy = compare_correct_error(slow_runs, motor_baseline)
        classes = analyze_trial_classes(task_runs, motor_baseline)
        spd = analyze_logged_spd(task_runs, motor_baseline)
        post_error = analyze_post_error_slowing(task_runs, motor_baseline)
        all_logged_spd = spd["all_logged"]
        validated_spd = spd["validated_15row"]
        rows.append(
            {
                "subject": subject,
                "motor_baseline_ms": motor_baseline,
                "n_rt_trials": sum(len(table) for table in rt_runs),
                "n_rt_baseline_trials": sum(
                    int(table["rawRT"].notna().sum()) for table in rt_runs
                ),
                "n_fast_trials": sum(len(table) for table in fast_runs),
                "n_slow_trials": sum(len(table) for table in slow_runs),
                "n_never_started_trials": n_never_started,
                "n_fast_dt_trials": speed["n_fast"],
                "n_slow_dt_trials": speed["n_slow"],
                "n_fast_dt_anticipations": speed["n_fast_anticipations"],
                "n_slow_dt_anticipations": speed["n_slow_anticipations"],
                "mean_fast_dt_ms": speed["mean_fast"],
                "mean_slow_dt_ms": speed["mean_slow"],
                "n_correct_trials": accuracy["n_correct"],
                "n_error_trials": accuracy["n_error"],
                "percent_correct": accuracy["percent_correct"],
                "n_fast_correct_trials": fast_accuracy["n_correct"],
                "n_fast_error_trials": fast_accuracy["n_error"],
                "fast_percent_correct": fast_accuracy["percent_correct"],
                "fast_percent_error": 100.0 - fast_accuracy["percent_correct"],
                "n_slow_correct_trials": slow_accuracy["n_correct"],
                "n_slow_error_trials": slow_accuracy["n_error"],
                "slow_percent_correct": slow_accuracy["percent_correct"],
                "slow_percent_error": 100.0 - slow_accuracy["percent_correct"],
                "mean_correct_dt_ms": accuracy["mean_correct"],
                "mean_error_dt_ms": accuracy["mean_error"],
                "n_easy_trials": classes["counts"]["easy"],
                "n_ambiguous_trials": classes["counts"]["ambiguous"],
                "n_misleading_trials": classes["counts"]["misleading"],
                "mean_easy_dt_ms": classes["means"]["easy"],
                "mean_ambiguous_dt_ms": classes["means"]["ambiguous"],
                "mean_misleading_dt_ms": classes["means"]["misleading"],
                "n_spd_all_logged": all_logged_spd["n_trials"],
                "mean_spd_all_logged": all_logged_spd["mean_spd"],
                "n_spd_validated_15row": validated_spd["n_trials"],
                "mean_spd_validated_15row": validated_spd["mean_spd"],
                **{
                    f"n_{name}_spd_all_logged": all_logged_spd["classes"][name]["n_trials"]
                    for name in ("easy", "ambiguous", "misleading")
                },
                **{
                    f"mean_{name}_spd_all_logged": all_logged_spd["classes"][name]["mean_spd"]
                    for name in ("easy", "ambiguous", "misleading")
                },
                **{
                    f"n_{name}_spd_validated_15row": validated_spd["classes"][name]["n_trials"]
                    for name in ("easy", "ambiguous", "misleading")
                },
                **{
                    f"mean_{name}_spd_validated_15row": validated_spd["classes"][name]["mean_spd"]
                    for name in ("easy", "ambiguous", "misleading")
                },
                "n_post_correct_trials": post_error["n_post_correct"],
                "n_post_error_trials": post_error["n_post_error"],
                "mean_post_correct_dt_ms": post_error["mean_post_correct"],
                "mean_post_error_dt_ms": post_error["mean_post_error"],
            }
        )

    subject_summary = pd.DataFrame(rows)
    output_path = layout.behavior_summary()
    save_table(
        output_path,
        subject_summary,
        metadata={
            "stage": "behavior_analysis",
            "subjects": sorted(by_subject),
            "excluded_subjects": sorted(excluded_subjects),
            "spd": {
                "source": "logged chosen-target nProb paired with tTime",
                "views": ["all_logged", "validated_15row"],
                "short_log_design_alignment": "forbidden",
            },
            "input_files": [
                str(path)
                for subject in sorted(source_paths)
                for path in source_paths[subject]
            ],
        },
    )
    group_path = layout.behavior_group_statistics()
    save_table(
        group_path,
        behavior_group_statistics(subject_summary),
        metadata={
            "stage": "behavior_group_statistics",
            "subjects": sorted(by_subject),
            "excluded_subjects": sorted(excluded_subjects),
            "test": "paired_t_test",
            "effect_size": "cohens_dz",
            "spd_views": ["all_logged", "validated_15row"],
            "subject_summary": str(output_path),
        },
    )
    trial_features_path = layout.behavior_trial_features()
    save_table(
        trial_features_path,
        _trial_feature_table(by_subject, motor_baselines),
        metadata={
            "stage": "behavior_trial_features",
            "subjects": sorted(by_subject),
            "excluded_subjects": sorted(excluded_subjects),
            "join_key": ["subject", "condition", "run", "run_trial_index"],
            "dt": "rawRT - subject motor_baseline_ms; task trials only",
            "spd": "logged chosen-target nProb at the motor-corrected decision time",
            "evidence_at_decision": "logged_spd - 0.5",
            "design_evidence": (
                "correct-target success probability and log posterior odds from "
                "the designed sTokenDirs sequence; 'early' is after three jumps"
            ),
            "input_files": [str(path) for path in paths],
        },
    )
    return WorkflowResult(
        stage="behavior_analysis",
        inputs=tuple(paths),
        outputs=(output_path, group_path, trial_features_path),
        settings={
            "subjects": sorted(by_subject),
            "excluded_subjects": sorted(excluded_subjects),
            "n_subjects": len(by_subject),
        },
    )
