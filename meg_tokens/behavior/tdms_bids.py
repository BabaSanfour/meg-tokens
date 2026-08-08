"""Raw BIDS behavioral layer: minimally-parsed TDMS logs under BIDS/sub-*/beh/.

A ``.tdms`` file is a LabVIEW container, not a table: nothing outside this
project can read a trial out of it. This layer is therefore a transcription,
not an analysis -- the same trials the acquisition software wrote, in a
format a reader with no ``nptdms`` and no ``meg_tokens`` can open, sitting
next to the MEG run they were recorded with.

This layer is the input to
:func:`meg_tokens.workflows.behavior_ingest.ingest_behavior`, whose
``derivatives/sub-*/beh/*_beh.tsv`` output adds the analysis choices --
trial-class inference and provenance columns -- that make it a derivative
rather than raw. The two are meant to differ: a row here can be compared
against its derivative counterpart precisely because only one of the two
has had judgement applied to it.

``write_beh_bids`` reuses the same tested low-level parser
(:func:`meg_tokens.behavior.tdms.parse_tdms_file`) rather than a separate
raw-only reader -- one parser means one place where the TDMS event-block
grammar and its validations live, and no second implementation to drift.
The parser transcribes only what LabVIEW wrote, so this layer carries
``RAW_TRIAL_COLUMNS``: no ``sTrialClass``, no ``trial_class_source``, no
``trial_class_rule``. Those record which class a trial logged as ``'x'``
belongs to, which is inferred rather than observed (see
``docs/behavioral_pipeline.md``, Findings: trial-classification reference
frame) and so is not something a raw file is entitled to assert -- not even
as a placeholder saying it declined to.

What the layer does keep is everything that inference *consumes*: the
recorded label ``sTrialClassRaw`` and the designed profile
``sp_design_correct``. That is what makes the derivative reproducible from
here: :func:`meg_tokens.workflows.behavior_ingest.ingest_behavior` reads
these files, applies
:func:`meg_tokens.behavior.classification.classify_trials`, and needs no
access to the original ``.tdms`` container.
"""

from __future__ import annotations

import json
from pathlib import Path

from mne_bids import BIDSPath

from meg_tokens.behavior.tdms import parse_tdms_file, parse_tdms_filename
from meg_tokens.core import normalize_subject_id
from meg_tokens.io import save_table, sidecar_path


def write_beh_bids(tdms_path: str | Path, *, bids_root: str | Path, task: str = "tokens") -> Path:
    """Write one TDMS run as a raw-BIDS behavioral table.

    Entities are read from the TDMS basename alone
    (``parse_tdms_filename``), never from a matched MEG session: a
    behavioral run is fully identified by its own filename, so this export
    runs for every one of a subject's logs regardless of whether Stage 0
    could match it to a recording -- an unmatched or ``ambiguous`` MEG row
    never costs the dataset its behavior. They mirror the MEG layer's
    choices so the two sit side by side under one subject: condition as
    ``acq-<condition>`` (``desc`` is derivatives-only and illegal on raw
    data, which is exactly what distinguishes this file from the
    ``desc-<condition>`` Stage 1 derivative of the same run), singular
    ``task``, and no ``ses``.

    A basename that does not match the project's TDMS filename contract
    raises ``ValueError`` from ``parse_tdms_filename`` rather than being
    skipped -- there are no entities to write it under. Callers that expect
    stray files (a ``temp_*.tdms`` scratch export) filter on
    ``meg_tokens.behavior.tdms.FILENAME_RE`` first; see
    ``meg_tokens.workflows.raw_staging._matching_tdms_files``.

    The written table carries a ``meg_tokens.io.save_table`` JSON sidecar
    recording the source path and ``stage="raw_bids"``, so the file states
    on its own that it is the untouched log rather than the
    inference-applied derivative. There is no ``overwrite`` switch: the
    output is a pure function of one input file, so rewriting it can only
    reproduce it.

    Returns the written ``*_beh.tsv`` path.
    """
    tdms_path = Path(tdms_path)
    run_info = parse_tdms_filename(tdms_path.name)
    table = parse_tdms_file(str(tdms_path))

    bids_path = BIDSPath(
        subject=run_info.subject,
        task=task,
        acquisition=run_info.condition.lower(),
        run=run_info.run,
        datatype="beh",
        suffix="beh",
        extension=".tsv",
        root=Path(bids_root),
    )
    save_table(
        bids_path.fpath,
        table,
        metadata={
            "stage": "raw_bids",
            "subject": run_info.subject,
            "condition": run_info.condition,
            "run": int(run_info.run),
            "acquisition_date": run_info.date,
            "source_file": str(tdms_path),
        },
    )
    return bids_path.fpath


def raw_behavior_files(bids_root: str | Path, subject: str, *, task: str = "tokens") -> list[Path]:
    """Every staged raw behavioral table for one subject, in entity order.

    Locates what ``write_beh_bids`` wrote, so Stage 1 can consume the raw
    BIDS layer rather than re-reading the ``.tdms`` containers. An empty
    list means Stage 0 has not staged this subject yet -- callers report
    that as a missing prerequisite, since it is not distinguishable here
    from a subject with no behavioral runs at all.
    """
    subject = normalize_subject_id(subject)
    beh_dir = Path(bids_root) / f"sub-{subject}" / "beh"
    return sorted(beh_dir.glob(f"sub-{subject}_task-{task}_*_beh.tsv"))


def read_raw_behavior_sidecar(table_path: str | Path) -> dict:
    """Run identity recorded beside one staged raw behavioral table.

    ``write_beh_bids`` writes subject, condition, run, acquisition date and
    the source path into the JSON sidecar, so a consumer recovers run
    identity from what was recorded at staging time rather than by
    re-parsing BIDS entities (which lowercase the condition and drop the
    date).
    """
    path = sidecar_path(Path(table_path))
    if not path.is_file():
        raise FileNotFoundError(f"No sidecar beside staged behavior table: {path}")
    with path.open() as stream:
        return json.load(stream).get("metadata", {})
