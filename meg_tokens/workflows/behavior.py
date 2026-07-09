"""Behavior ingestion and summary workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from meg_tokens.behavior.metrics import (
    analyze_post_error_slowing,
    analyze_trial_classes,
    calculate_motor_baseline,
    compare_correct_error,
    compare_fast_slow,
)
from meg_tokens.behavior.tdms import (
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
) -> tuple[dict[str, object], ...]:
    """Parse every standard TDMS run for one subject."""
    input_root = Path(input_root)
    subject_dir = _subject_input_dir(input_root, subject)
    layout = DerivativeLayout(output_root)
    records = []

    for input_path in sorted(subject_dir.glob("*.tdms")):
        try:
            run_info = parse_tdms_filename(input_path.name)
        except ValueError:
            continue
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
        rt_runs = _condition_runs(tables, "RT")
        fast_runs = _condition_runs(tables, "Fast")
        slow_runs = _condition_runs(tables, "Slow")
        task_runs = fast_runs + slow_runs
        motor_baseline = calculate_motor_baseline(rt_runs)
        speed = compare_fast_slow(fast_runs, slow_runs, motor_baseline)
        accuracy = compare_correct_error(task_runs, motor_baseline)
        classes = analyze_trial_classes(task_runs, motor_baseline)
        post_error = analyze_post_error_slowing(task_runs, motor_baseline)
        rows.append(
            {
                "subject": subject,
                "motor_baseline_ms": motor_baseline,
                "n_rt_trials": sum(len(table) for table in rt_runs),
                "n_fast_trials": sum(len(table) for table in fast_runs),
                "n_slow_trials": sum(len(table) for table in slow_runs),
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
