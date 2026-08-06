"""Behavior ingestion and summary workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from meg_tokens.behavior.metrics import (
    analyze_logged_spd,
    analyze_post_error_slowing,
    analyze_trial_classes,
    calculate_motor_baseline,
    compare_correct_error,
    compare_fast_slow,
)
from meg_tokens.behavior.tdms import (
    OUTCOME_NEVER_STARTED,
    TdmsRunInfo,
    add_run_metadata,
    parse_tdms_file,
    parse_tdms_filename,
    validate_behavior_dataframe,
)
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
                parse_tdms_file(str(input_path)),
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
        table.loc[table["nOutcome"] != OUTCOME_NEVER_STARTED].copy()
        for table in _condition_runs(tables, condition)
    ]


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
    by_subject: dict[str, list[pd.DataFrame]] = {}
    source_paths: dict[str, list[Path]] = {}
    for path in paths:
        table = pd.read_csv(path, sep="\t")
        validate_behavior_dataframe(table)
        if table.empty:
            continue
        subject = normalize_subject_id(str(table["subject"].iloc[0]))
        by_subject.setdefault(subject, []).append(table)
        source_paths.setdefault(subject, []).append(path)
    if not by_subject:
        raise ValueError("Behavior derivatives do not contain any trials")

    rows = []
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
        speed = compare_fast_slow(fast_runs, slow_runs, motor_baseline)
        accuracy = compare_correct_error(task_runs, motor_baseline)
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
                "n_fast_trials": sum(len(table) for table in fast_runs),
                "n_slow_trials": sum(len(table) for table in slow_runs),
                "n_never_started_trials": n_never_started,
                "mean_fast_dt_ms": speed["mean_fast"],
                "mean_slow_dt_ms": speed["mean_slow"],
                "fast_vs_slow_t": speed["t_stat"],
                "fast_vs_slow_p": speed["p_value"],
                "percent_correct": accuracy["percent_correct"],
                "mean_correct_dt_ms": accuracy["mean_correct"],
                "mean_error_dt_ms": accuracy["mean_error"],
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
                "mean_post_correct_dt_ms": post_error["mean_post_correct"],
                "mean_post_error_dt_ms": post_error["mean_post_error"],
            }
        )

    output_path = layout.behavior_summary()
    save_table(
        output_path,
        pd.DataFrame(rows),
        metadata={
            "stage": "behavior_analysis",
            "subjects": sorted(by_subject),
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
    return WorkflowResult(
        stage="behavior_analysis",
        inputs=tuple(paths),
        outputs=(output_path,),
        settings={"subjects": sorted(by_subject), "n_subjects": len(by_subject)},
    )
