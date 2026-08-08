"""Stage 1: raw BIDS behavior -> analysis-ready behavioral derivatives.

Reads the raw behavioral layer Stage 0 stages under ``BIDS/sub-*/beh/``,
not the ``.tdms`` containers, so the derivative is derived from the raw
BIDS layer in the ordinary BIDS sense rather than from a proprietary
format sitting outside the dataset. One consequence worth stating: Stage 0
must have run first. That is a real ordering dependency, accepted because
the alternative -- two independent readers of the LabVIEW container, and a
raw layer nothing consumes -- is worse.

What this stage adds on top of the raw log is exactly what makes it a
derivative rather than a copy: trial classes inferred from the designed
profile (:mod:`meg_tokens.behavior.classification`), run identity and the
response fields derived from it (``rawRT``, ``isCorrect``), and validation
against the full Stage 1 contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from meg_tokens.behavior.classification import classify_trials
from meg_tokens.behavior.schema import validate_behavior_table
from meg_tokens.behavior.tables import add_run_metadata, read_raw_behavior_table
from meg_tokens.behavior.tdms import TdmsRunInfo
from meg_tokens.behavior.tdms_bids import raw_behavior_files, read_raw_behavior_sidecar
from meg_tokens.core import ProjectConfig, WorkflowResult, normalize_subject_id
from meg_tokens.io import DerivativeLayout, save_table


def _run_info(table_path: Path) -> tuple[TdmsRunInfo, str]:
    """Run identity for one staged table, from the sidecar Stage 0 wrote."""
    metadata = read_raw_behavior_sidecar(table_path)
    missing = [key for key in ("subject", "condition", "run") if key not in metadata]
    if missing:
        raise ValueError(
            f"Staged behavior sidecar for {table_path.name} is missing {missing}. "
            "Re-run `meg-tokens meg apply-raw-staging` to rewrite it."
        )
    run_info = TdmsRunInfo(
        subject=normalize_subject_id(str(metadata["subject"])),
        condition=str(metadata["condition"]),
        run=str(metadata["run"]),
        date=str(metadata.get("acquisition_date", "")),
    )
    return run_info, str(metadata.get("source_file", table_path.name))


def ingest_subject_behavior(
    subject: str,
    *,
    bids_root: str | Path,
    output_root: str | Path,
    dry_run: bool = False,
    infer_random_classes: bool = True,
) -> tuple[dict[str, object], ...]:
    """Build Stage 1 derivatives for every staged run of one subject.

    Raises ``FileNotFoundError`` when the subject has no staged raw
    behavior: that is a missing prerequisite, not an empty result, and
    silently producing nothing would look identical to success.
    """
    layout = DerivativeLayout(output_root)
    table_paths = raw_behavior_files(bids_root, subject)
    if not table_paths:
        raise FileNotFoundError(
            f"No staged raw behavior for {normalize_subject_id(subject)} under "
            f"{Path(bids_root)}. Stage 1 reads the raw BIDS layer, so run "
            "`meg-tokens meg stage-raw` and `meg-tokens meg apply-raw-staging` first."
        )

    records = []
    for table_path in table_paths:
        run_info, source_file = _run_info(table_path)
        output_path = layout.behavior(
            subject=run_info.subject,
            run=run_info.run,
            condition=run_info.condition,
        )
        trial_count = 0
        if not dry_run:
            table = add_run_metadata(
                classify_trials(
                    read_raw_behavior_table(table_path),
                    infer_random_classes=infer_random_classes,
                ),
                run_info,
                source_file,
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
                    "source_file": source_file,
                    "source_date": run_info.date,
                    "raw_behavior_table": str(table_path),
                    "infer_random_classes": infer_random_classes,
                },
            )
            trial_count = len(table)
        records.append(
            {
                "subject": run_info.subject,
                "condition": run_info.condition,
                "run": run_info.run,
                "input": str(table_path),
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
    """Ingest selected subjects from the staged raw behavioral layer."""
    bids_root = project.bids_root
    if not bids_root.is_dir():
        raise FileNotFoundError(
            f"BIDS root does not exist: {bids_root}. Run `meg-tokens meg stage-raw` "
            "and `meg-tokens meg apply-raw-staging` before ingesting."
        )

    selected = list(subjects) if subjects else sorted(
        path.name.removeprefix("sub-")
        for path in bids_root.glob("sub-*")
        if path.is_dir() and (path / "beh").is_dir() and path.name != "sub-emptyroom"
    )
    if not selected:
        raise FileNotFoundError(
            f"No staged subject behavior was found under {bids_root}. Run "
            "`meg-tokens meg apply-raw-staging` first."
        )

    records = []
    for subject in selected:
        records.extend(
            ingest_subject_behavior(
                subject,
                bids_root=bids_root,
                output_root=bids_root,
                dry_run=dry_run,
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
