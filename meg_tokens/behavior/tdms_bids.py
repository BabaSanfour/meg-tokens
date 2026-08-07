"""Raw BIDS behavioral layer: minimally-parsed TDMS logs under BIDS/sub-*/beh/.

This is additive and does not touch :mod:`meg_tokens.workflows.behavior`'s
``ingest_behavior``/``analyze_behavior`` or their
``derivatives/meg-tokens/sub-*/beh/*_beh.tsv`` output. That pipeline already
makes real analysis choices (trial-class inference, provenance columns) on
top of the raw log, so it is correctly a derivative, not raw, and stays
exactly as it is today -- see ``docs/meg_t0_7_raw_bidsification_plan.md``.

``write_beh_bids`` reuses the same tested low-level parser
(:func:`meg_tokens.behavior.tdms.parse_tdms_file`) with
``infer_random_classes=False``, the one existing switch that already
distinguishes "what LabVIEW logged" from "what we inferred" -- and writes it
under BIDS-legal raw entities (``acq-<condition>_run-<N>``, no ``desc-``,
which is derivatives-only).
"""

from __future__ import annotations

from pathlib import Path

from mne_bids import BIDSPath

from meg_tokens.behavior.tdms import parse_tdms_file, parse_tdms_filename
from meg_tokens.io import save_table


def write_beh_bids(tdms_path: str | Path, *, bids_root: str | Path, task: str = "tokens") -> Path:
    """Write one TDMS run as a raw-BIDS behavioral table.

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
