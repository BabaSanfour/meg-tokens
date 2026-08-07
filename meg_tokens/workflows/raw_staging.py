"""Stage 0 raw BIDSification workflow: plan, then apply, media -> BIDS/.

``plan_raw_staging`` is read-only beyond writing a reviewable manifest.
``apply_raw_staging`` reads that manifest (fresh or hand-edited) and
materializes the BIDS raw layers -- see
``docs/meg_t0_7_raw_bidsification_plan.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from meg_tokens.behavior.tdms import FILENAME_RE, find_subject_directory
from meg_tokens.behavior.tdms_bids import write_beh_bids
from meg_tokens.core import ProjectConfig, RawStagingConfig, WorkflowResult, normalize_subject_id
from meg_tokens.io import DerivativeLayout, save_table
from meg_tokens.meg.bids_raw import write_emptyroom_bids, write_meg_bids
from meg_tokens.meg.preprocessing import convert_ctf_headshape_to_pos
from meg_tokens.meg.raw_staging import MatchResult, match_media_to_behavior

MANIFEST_COLUMNS = [
    "subject", "kind", "condition", "run", "date", "media_path", "match_method",
    "meg_start_pulse_count", "behavior_trial_count", "count_agreement",
    "action", "note",
]


def _result_to_row(result: MatchResult) -> dict:
    return {
        "subject": result.subject,
        "kind": result.kind,
        "condition": result.condition or "",
        "run": result.run if result.run is not None else "",
        "date": result.date,
        "media_path": str(result.media_path) if result.media_path else "",
        "match_method": result.match_method,
        "meg_start_pulse_count": (
            result.meg_start_pulse_count if result.meg_start_pulse_count is not None else ""
        ),
        "behavior_trial_count": (
            result.behavior_trial_count if result.behavior_trial_count is not None else ""
        ),
        "count_agreement": result.count_agreement,
        "action": result.action,
        "note": result.note,
    }


def plan_raw_staging(
    project: ProjectConfig,
    *,
    subjects: Sequence[str],
    media_root: Optional[str | Path] = None,
    settings: RawStagingConfig = RawStagingConfig(),
) -> WorkflowResult:
    """Match raw media sessions to behavior runs and write a reviewable manifest.

    Reads only the raw media and ``tdms/`` -- Stage 0 has no dependency on
    ``behavior ingest`` or any other stage having run first. Read-only
    beyond the manifest itself: never touches ``BIDS/``. Run
    ``apply_raw_staging`` against the manifest this writes (optionally
    hand-edited first) to materialize the raw BIDS layers.
    """
    root = Path(media_root) if media_root is not None else project.raw_meg_root
    normalized_subjects = [normalize_subject_id(subject) for subject in subjects]

    rows = []
    for subject in normalized_subjects:
        results = match_media_to_behavior(
            root,
            project.behavior_root,
            subject,
            settings=settings,
        )
        for result in results:
            if result.match_method == "known_override":
                print(
                    f"{result.subject} {result.condition}{result.run}: matched via "
                    f"a documented session mapping, not duration/order/count alone "
                    f"(KNOWN_SESSION_OVERRIDES -- see "
                    "docs/meg_t0_7_raw_bidsification_plan.md 'H01/H05 resolution')"
                )
            elif result.count_agreement == "mismatch" and result.action == "stage":
                print(
                    f"{result.subject} {result.condition}{result.run}: staging despite a "
                    "Start-pulse-count mismatch -- documented KNOWN_TRAILING_TRIAL_MISMATCHES "
                    "case (see docs/behavior_qc_report.md section 3)"
                )
        rows.extend(_result_to_row(result) for result in results)

    table = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    layout = DerivativeLayout(project.bids_root, pipeline=project.pipeline, task=project.task)
    manifest_path = layout.raw_staging_manifest()
    save_table(
        manifest_path,
        table,
        metadata={
            "stage": "raw_staging_plan",
            "media_root": str(root),
            "subjects": normalized_subjects,
            "slowfast_nominal_duration": settings.slowfast_nominal_duration,
            "rt_nominal_duration": settings.rt_nominal_duration,
            "duration_tolerance": settings.duration_tolerance,
            "count_tolerance": settings.count_tolerance,
        },
    )

    # Plain dict/list summaries (not a DataFrame) so callers such as the
    # CLI can report results without doing their own table analysis --
    # scientific/derivative-naming logic belongs in workflows, not the CLI.
    summary_by_subject = {
        subject: {
            action: int(count)
            for action, count in table[table["subject"] == subject]["action"].value_counts().items()
        }
        for subject in normalized_subjects
    }
    review_rows = table[table["action"] == "review"].to_dict(orient="records")

    return WorkflowResult(
        stage="raw_staging_plan",
        inputs=(root,),
        outputs=(manifest_path,),
        settings={
            "subjects": normalized_subjects,
            "media_root": str(root),
            "summary_by_subject": summary_by_subject,
            "review_rows": review_rows,
        },
    )


def _matching_tdms_files(project: ProjectConfig, subject: str) -> list[Path]:
    """Every strictly-named TDMS run file for one subject.

    Skips anything not matching the project's TDMS filename contract
    (``meg_tokens.behavior.tdms.FILENAME_RE`` -- the same pattern
    ``behavior ingest`` requires) rather than raising: unlike ingestion,
    this raw-BIDS export has nothing that depends on a complete run set,
    so an odd non-run file (e.g. a `temp_*.tdms` scratch export) is simply
    not part of what gets staged.

    Reuses ``find_subject_directory`` rather than assuming the on-disk
    directory is named exactly ``normalize_subject_id(subject)``: in the
    real dataset ``H01``'s TDMS folder is literally ``H1``, not ``H01``.
    """
    try:
        subject_dir = find_subject_directory(project.behavior_root, subject)
    except FileNotFoundError:
        return []
    return sorted(
        path for path in subject_dir.glob("*.tdms") if FILENAME_RE.match(path.name)
    )


def _write_headshape(project: ProjectConfig, subject: str, eeg_path: Path) -> Path:
    pos_path = (
        Path(project.bids_root) / f"sub-{subject}" / "meg" / f"sub-{subject}_headshape.pos"
    )
    convert_ctf_headshape_to_pos(str(eeg_path), str(pos_path))
    return pos_path


def apply_raw_staging(
    project: ProjectConfig,
    *,
    manifest_path: Optional[str | Path] = None,
    subjects: Optional[Sequence[str]] = None,
) -> WorkflowResult:
    """Materialize BIDS/sub-*/meg (MEG + empty-room) and BIDS/sub-*/beh from a manifest.

    Only ``kind == "run"``/``"noise"``/``"headshape"`` rows with
    ``action == "stage"`` are materialized for MEG; nothing here
    recomputes the plan, so a manually reviewed and edited manifest is
    applied exactly as saved. Behavioral raw-BIDS export runs for every
    requested subject's TDMS files independently of MEG matching (see
    ``_matching_tdms_files``), since parsing a TDMS filename needs no
    raw-MEG session at all.
    """
    layout = DerivativeLayout(project.bids_root, pipeline=project.pipeline, task=project.task)
    path = Path(manifest_path) if manifest_path is not None else layout.raw_staging_manifest()
    if not path.is_file():
        raise FileNotFoundError(
            f"No raw-staging manifest at {path} -- run `meg-tokens meg stage-raw` first."
        )
    table = pd.read_csv(path, sep="\t", dtype={"condition": str, "date": str})

    subject_filter = (
        {normalize_subject_id(subject) for subject in subjects} if subjects else None
    )
    staged = table[table["action"] == "stage"]
    if subject_filter is not None:
        staged = staged[staged["subject"].isin(subject_filter)]

    inputs = [path]
    outputs: list[Path] = []
    empty_room_by_subject: dict = {}

    for _, row in staged[staged["kind"] == "noise"].iterrows():
        media_path = Path(row["media_path"])
        bids_path = write_emptyroom_bids(
            media_path, date=str(row["date"]), bids_root=project.bids_root, overwrite=True
        )
        empty_room_by_subject[row["subject"]] = bids_path
        outputs.append(bids_path.fpath)
        inputs.append(media_path)

    for _, row in staged[staged["kind"] == "run"].iterrows():
        subject = row["subject"]
        # overwrite=True: real CTF head digitization varies by a
        # sub-millimeter, run-to-run fitting noise (confirmed against real
        # data), but BIDS's coordsystem.json is subject+acquisition-scoped,
        # not per-run, so mne_bids refuses on the second run of the same
        # acq unless told to overwrite. Harmless here -- this project's own
        # coregistration uses a separately-managed -trans.fif, never
        # coordsystem.json -- and the manifest review step already is this
        # workflow's safety checkpoint, not a second implicit one here.
        bids_path = write_meg_bids(
            row["media_path"],
            subject=subject,
            condition=row["condition"],
            run=int(row["run"]),
            bids_root=project.bids_root,
            task=project.task,
            empty_room_bids_path=empty_room_by_subject.get(subject),
            overwrite=True,
        )
        outputs.append(bids_path.fpath)
        inputs.append(Path(row["media_path"]))

    for _, row in staged[staged["kind"] == "headshape"].iterrows():
        pos_path = _write_headshape(project, row["subject"], Path(row["media_path"]))
        outputs.append(pos_path)
        inputs.append(Path(row["media_path"]))

    staged_subjects = sorted(set(staged["subject"])) if subject_filter is None else sorted(subject_filter)
    for subject in staged_subjects:
        for tdms_path in _matching_tdms_files(project, subject):
            beh_path = write_beh_bids(tdms_path, bids_root=project.bids_root, task=project.task)
            outputs.append(beh_path)
            inputs.append(tdms_path)

    return WorkflowResult(
        stage="raw_staging_apply",
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        settings={"manifest": str(path)},
    )
