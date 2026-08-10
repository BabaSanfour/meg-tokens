"""Stage 0 raw BIDSification workflow: plan, then apply, raw trees -> BIDS/.

``plan_raw_staging`` is read-only beyond writing a reviewable manifest.
``apply_raw_staging`` reads that manifest (fresh or hand-edited) and
materializes the BIDS raw layers -- see
``docs/data_contract.md``, "Stage 0: Raw BIDSification".
"""

from __future__ import annotations

import math
from dataclasses import asdict, fields
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from meg_tokens.behavior.tdms import FILENAME_RE, parse_tdms_filename
from meg_tokens.behavior.tdms_bids import write_beh_bids
from meg_tokens.core import ProjectConfig, RawStagingConfig, WorkflowResult, normalize_subject_id
from meg_tokens.io import DerivativeLayout, save_table
from meg_tokens.meg.anat_bids import write_anat_bids
from meg_tokens.meg.meg_bids import write_emptyroom_bids, write_meg_bids
from meg_tokens.meg.preprocessing import convert_ctf_headshape_to_pos
from meg_tokens.meg.raw_staging import MatchResult, match_raw_to_behavior

# The manifest's columns are MatchResult's fields, in its field order --
# declared once, in the dataclass, rather than repeated here.
MANIFEST_COLUMNS = [field.name for field in fields(MatchResult)]


def _manifest_value(value: object) -> object:
    """Render one ``MatchResult`` field for the TSV.

    Blanks rather than ``None``/``NaN`` so the file round-trips through
    ``pandas.read_csv`` cleanly for every column, numeric ones included
    (``apply_raw_staging`` re-reads it verbatim). Floats are rounded to the
    precision the decision is actually made at -- correct and incorrect
    matches differ by three orders of magnitude, so further digits would
    imply precision the measurement does not carry.
    """
    if value is None:
        return ""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return "inf" if math.isinf(value) else round(value, 4)
    return value


def _result_to_row(result: MatchResult) -> dict:
    return {name: _manifest_value(getattr(result, name)) for name in MANIFEST_COLUMNS}


def _notable_message(result: MatchResult) -> Optional[str]:
    """What about this row deserves the operator's attention, if anything.

    Only rows resolved by something other than plain evidence agreement are
    called out, so the printed plan stays short enough that the exceptions
    are visible rather than buried in a per-run log.
    """
    if "DISAGREES with KNOWN_SESSION_OVERRIDES" in result.note:
        return (
            "the inter-trial-interval fingerprint contradicts KNOWN_SESSION_OVERRIDES "
            "-- flagged for review, NOT staged. See the manifest note."
        )
    if result.match_method == "known_override":
        return (
            "fingerprint did not resolve this run; matched via the documented session "
            "mapping (KNOWN_SESSION_OVERRIDES -- see docs/data_contract.md "
            "'H01 and H05')"
        )
    if result.count_agreement == "mismatch" and result.action == "stage":
        return (
            "staging despite a Start-pulse-count mismatch -- documented "
            "KNOWN_TRAILING_TRIAL_MISMATCHES case (see docs/meg.md "
            "section 3)"
        )
    return None


def plan_raw_staging(
    project: ProjectConfig,
    *,
    subjects: Sequence[str],
    settings: RawStagingConfig = RawStagingConfig(),
) -> WorkflowResult:
    """Match raw acquisition sessions to behavior runs and write a reviewable manifest.

    Both roots come from the project configuration
    (``ProjectConfig.raw_meg_root`` and ``.behavior_root``), so a plan is
    reproducible from the TOML alone. Reads only those two trees -- Stage 0
    has no dependency on ``behavior ingest`` or any other stage having run
    first -- and is read-only beyond the manifest itself: it never touches
    ``BIDS/``. Run ``apply_raw_staging`` against the manifest this writes
    (optionally hand-edited first) to materialize the raw BIDS layers.
    """
    root = project.raw_meg_root
    normalized_subjects = [normalize_subject_id(subject) for subject in subjects]

    rows = []
    for subject in normalized_subjects:
        results = match_raw_to_behavior(
            root,
            project.behavior_root,
            subject,
            subjects_dir=project.subjects_dir,
            settings=settings,
        )
        for result in results:
            message = _notable_message(result)
            if message:
                print(f"{result.subject} {result.condition}{result.run}: {message}")
        rows.extend(_result_to_row(result) for result in results)

    table = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    layout = DerivativeLayout(project.bids_root, task=project.task)
    manifest_path = layout.raw_staging_manifest()
    save_table(
        manifest_path,
        table,
        # asdict, not a hand-listed copy: every RawStagingConfig knob that
        # influenced this plan is recorded, including ones added later.
        metadata={
            "stage": "raw_staging_plan",
            "raw_root": str(root),
            "subjects": normalized_subjects,
            **asdict(settings),
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
            "raw_root": str(root),
            "summary_by_subject": summary_by_subject,
            "review_rows": review_rows,
        },
    )


def _matching_tdms_files(project: ProjectConfig, subject: str) -> list[Path]:
    """Every TDMS run file for one subject, or a refusal to guess.

    This is the only gate between the raw logs and the dataset: Stage 1 now
    reads the staged BIDS tables rather than the ``.tdms`` files, so a log
    dropped here is a log that never reaches any analysis. Both guards that
    used to live in ``behavior ingest`` therefore apply at this boundary
    instead of downstream of it.

    A filename not matching the project's TDMS contract
    (``meg_tokens.behavior.tdms.FILENAME_RE``) raises rather than being
    skipped -- a misnamed real run and a scratch export look identical to a
    glob, and only one of them is safe to ignore. Excluding a file is an
    explicit act: name it in ``behavior_ignore_files`` in the project TOML,
    where the reason can be recorded beside it.

    Two files claiming the same ``(condition, run)`` likewise raise: they
    resolve to one BIDS path, so the second would silently overwrite the
    first and the loss would be invisible afterwards.
    """
    subject = normalize_subject_id(subject)
    subject_dir = Path(project.behavior_root) / subject
    ignored = set(project.behavior_ignore_files)

    candidates = [
        path for path in sorted(subject_dir.glob("*.tdms")) if path.name not in ignored
    ]
    unmatched = [path.name for path in candidates if not FILENAME_RE.match(path.name)]
    if unmatched:
        raise ValueError(
            f"Refusing to silently skip .tdms files with non-canonical names under "
            f"{subject_dir}: {unmatched}. Rename them to match "
            "H<subject><Condition><run>_<YYMMDD>.tdms, or list their exact filenames "
            "in behavior_ignore_files in the project TOML to exclude them explicitly."
        )

    seen: dict[tuple[str, str], Path] = {}
    for path in candidates:
        run_info = parse_tdms_filename(path.name)
        key = (run_info.condition, run_info.run)
        if key in seen:
            raise ValueError(
                f"Duplicate TDMS run for {subject} {key[0]}{key[1]}: both "
                f"{seen[key].name} and {path.name} stage to the same BIDS path. "
                "Resolve the collision before staging."
            )
        seen[key] = path
    return candidates


def _write_headshape(project: ProjectConfig, subject: str, eeg_path: Path) -> Path:
    """Convert one subject's digitized headshape to BIDS/sub-*/meg/*_headshape.pos.

    A real format conversion, not a raw copy -- BIDS has no standard slot
    for a bare digitization file outside a raw recording's own metadata,
    so it's kept named/located predictably next to that subject's MEG data
    for the coregistration step to find later.
    """
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
    """Materialize the BIDS raw layer -- meg, beh and anat -- from a manifest.

    Only ``kind == "run"``/``"noise"``/``"headshape"``/``"anat"`` rows with
    ``action == "stage"`` are materialized; nothing here
    recomputes the plan, so a manually reviewed and edited manifest is
    applied exactly as saved. Behavioral raw-BIDS export runs for every
    requested subject's TDMS files independently of MEG matching (see
    ``_matching_tdms_files``), since parsing a TDMS filename needs no
    raw-MEG session at all.
    """
    layout = DerivativeLayout(project.bids_root, task=project.task)
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
        source_path = Path(row["source_path"])
        bids_path = write_emptyroom_bids(
            source_path, date=str(row["date"]), bids_root=project.bids_root, overwrite=True
        )
        empty_room_by_subject[row["subject"]] = bids_path
        outputs.append(bids_path.fpath)
        inputs.append(source_path)

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
            row["source_path"],
            subject=subject,
            condition=row["condition"],
            run=int(row["run"]),
            bids_root=project.bids_root,
            task=project.task,
            empty_room_bids_path=empty_room_by_subject.get(subject),
            overwrite=True,
        )
        outputs.append(bids_path.fpath)
        inputs.append(Path(row["source_path"]))

    for _, row in staged[staged["kind"] == "headshape"].iterrows():
        pos_path = _write_headshape(project, row["subject"], Path(row["source_path"]))
        outputs.append(pos_path)
        inputs.append(Path(row["source_path"]))

    for _, row in staged[staged["kind"] == "anat"].iterrows():
        # overwrite=True is unconditionally safe here: BIDS-anat holds exactly
        # one T1w per subject (no acq/run entities), so there is none of the
        # per-run coordsystem.json conflict the MEG rows have to reason about.
        anat_path = write_anat_bids(
            row["subject"],
            row["source_path"],
            bids_root=project.bids_root,
            overwrite=True,
        )
        outputs.append(anat_path.fpath)
        inputs.append(Path(row["source_path"]))

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
