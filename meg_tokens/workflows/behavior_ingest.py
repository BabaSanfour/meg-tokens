"""Behavioral TDMS ingestion workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from meg_tokens.behavior.schema import validate_behavior_table
from meg_tokens.behavior.tables import add_run_metadata
from meg_tokens.behavior.tdms import (
    TdmsRunInfo,
    parse_tdms_file,
    parse_tdms_filename,
)
from meg_tokens.core import ProjectConfig, WorkflowResult, normalize_subject_id
from meg_tokens.io import DerivativeLayout, save_table


def _subject_input_dir(root: Path, subject: str) -> Path:
    subject_dir = root / normalize_subject_id(subject)
    if not subject_dir.is_dir():
        raise FileNotFoundError(f"Subject input directory does not exist for {subject} under {root}")
    return subject_dir


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
            validate_behavior_table(table)
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
