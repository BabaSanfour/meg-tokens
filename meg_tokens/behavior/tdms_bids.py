"""Raw BIDS behavioral layer: minimally-parsed TDMS logs under BIDS/sub-*/beh/.

A ``.tdms`` file is a LabVIEW container, not a table: nothing outside this
project can read a trial out of it. This layer is therefore a transcription,
not an analysis -- the same trials the acquisition software wrote, in a
format a reader with no ``nptdms`` and no ``meg_tokens`` can open, sitting
next to the MEG run they were recorded with.

This is additive and does not touch
:func:`meg_tokens.workflows.behavior_ingest.ingest_behavior` /
:func:`meg_tokens.workflows.behavior_analysis.analyze_behavior` or their
``derivatives/sub-*/beh/*_beh.tsv`` output. That pipeline already
makes real analysis choices (trial-class inference, provenance columns) on
top of the raw log, so it is correctly a derivative, not raw, and stays
exactly as it is today -- see ``docs/meg_t0_7_raw_bidsification_plan.md``.
The two outputs are meant to coexist and to differ: a row here can be
compared against its derivative counterpart precisely because only one of
the two has had judgement applied to it.

``write_beh_bids`` reuses the same tested low-level parser
(:func:`meg_tokens.behavior.tdms.parse_tdms_file`) rather than a separate
raw-only reader -- one parser means one place where the TDMS event-block
grammar and its validations live, and no second implementation to drift.
The one thing it changes is ``infer_random_classes=False``, the existing
switch that already draws this project's line between "what LabVIEW logged"
and "what we inferred" (see
``docs/behavior_t0_1_nprob_trial_class.md`` section 3b): with it off, a raw
``'x'`` trial class stays unresolved and ``trial_class_rule`` records
``"inference_disabled"`` instead of a rule this file is not entitled to
apply.
"""

from __future__ import annotations

from pathlib import Path

from mne_bids import BIDSPath

from meg_tokens.behavior.tdms import parse_tdms_file, parse_tdms_filename
from meg_tokens.io import save_table


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
    recording the source path and ``infer_random_classes=False``, so the
    file states on its own that it is the untouched log rather than the
    inference-applied derivative. There is no ``overwrite`` switch: the
    output is a pure function of one input file, so rewriting it can only
    reproduce it.

    Returns the written ``*_beh.tsv`` path.
    """
    tdms_path = Path(tdms_path)
    run_info = parse_tdms_filename(tdms_path.name)
    table = parse_tdms_file(str(tdms_path), infer_random_classes=False)

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
            "infer_random_classes": False,
        },
    )
    return bids_path.fpath
