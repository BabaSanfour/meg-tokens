"""The BIDS-MEG raw layer: writing it, and finding it again.

``write_meg_bids`` / ``write_emptyroom_bids`` are Stage 0's writer half --
``meg.raw_staging`` decides which numbered session is which run,
``workflows.raw_staging`` applies the reviewed manifest, this performs the
copy. ``mne_bids.write_raw_bids`` does the copying rather than ``copytree``
because a CTF ``.ds`` is not relocatable: every file inside must share the
directory's basename. It also emits the standard sidecars; what this module
adds is the BIDS entities, the trigger events under this project's
subject-aware codes, and the empty-room link.

Everything else is discovery, used by the later stages rather than by Stage 0.
The run-grid walkers iterate :data:`RAW_MEG_RUNS` rather than globbing, so a
recording missing anywhere in the chain is a *named* absence.

Subject labels arrive canonical: the CLI normalizes them once on entry and
the staged tree is canonical by construction, so nothing here re-checks.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import mne
from mne_bids import BIDSPath, write_raw_bids

if TYPE_CHECKING:
    from meg_tokens.io import DerivativeLayout

from meg_tokens.core import RunSpec

#: Every recording the Tokens protocol acquires per subject, in protocol order.
RAW_MEG_RUNS = (
    ("Fast", 1), ("Fast", 2), ("Fast", 3), ("Fast", 4),
    ("RT", 1), ("RT", 2),
    ("Slow", 1), ("Slow", 2), ("Slow", 3), ("Slow", 4),
)


def raw_meg_path(
    bids_root: str | Path,
    subject: str,
    condition: str,
    run: int | str,
    *,
    task: str = "tokens",
) -> Path:
    """Where Stage 0 stages one recording."""
    return (
        Path(bids_root)
        / f"sub-{subject}"
        / "meg"
        / f"sub-{subject}_task-{task}_acq-{condition.lower()}_run-{run}_meg.ds"
    )


def raw_meg_sessions(
    bids_root: str | Path,
    subject: str,
    *,
    task: str = "tokens",
) -> list[tuple[RunSpec, Path]]:
    """Every staged recording for one subject, with its identity.

    Walks :data:`RAW_MEG_RUNS` rather than globbing, so a subject short of a
    session is short of a *named* one -- H05 has no RT run 1. An empty list
    means Stage 0 has not staged this subject at all.
    """
    return [
        (RunSpec(subject=subject, run=str(run), condition=condition), path)
        for condition, run in RAW_MEG_RUNS
        if (path := raw_meg_path(bids_root, subject, condition, run, task=task)).is_dir()
    ]


def headshape_path(bids_root: str | Path, subject: str) -> Path:
    """Where Stage 0 stages a subject's head shape."""
    return Path(bids_root) / f"sub-{subject}" / "meg" / f"sub-{subject}_headshape.eeg"


def staged_subjects(bids_root: str | Path) -> list[str]:
    """Every subject with a staged MEG tree, in order.

    ``sub-emptyroom`` is session-nested (``sub-emptyroom/ses-<date>/meg``), so
    requiring a subject-level ``meg/`` already excludes it.
    """
    root = Path(bids_root)
    return sorted(
        path.name.removeprefix("sub-")
        for path in root.glob("sub-*")
        if path.is_dir() and (path / "meg").is_dir()
    )


def epoched_meg_runs(
    layout: "DerivativeLayout",
    subject: str,
    *,
    alignment: str,
    stream: str,
) -> list[tuple[RunSpec, Path]]:
    """Every epoched run of one subject for one alignment and stream, in protocol order.

    The third counterpart to :func:`raw_meg_sessions`, walking the same
    :data:`RAW_MEG_RUNS` grid, so the property holds all the way down the
    chain: a run absent from the source stage is a named run that failed to
    epoch, not an entry the discovery quietly never produced.

    A run whose rejection kept no epoch (16 files, all H10's) is not returned:
    its file and sidecar stay as the record of why, but MNE reads an empty
    epoch file back without its metadata columns, so there is nothing an
    analysis could take from it.
    """
    runs = []
    for condition, run in RAW_MEG_RUNS:
        path = layout.epochs(
            subject=subject,
            run=str(run),
            condition=condition,
            alignment=alignment,
            stream=stream,
        )
        if path.is_file() and json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["n_epochs"]:
            runs.append(
                (RunSpec(subject=subject, run=str(run), condition=condition), path)
            )
    return runs


def _acquisitions(scans: Path) -> dict[str, datetime]:
    """Each MEG file's acquisition time from a BIDS ``scans.tsv``."""
    with scans.open(encoding="utf-8-sig") as handle:
        return {
            row["filename"]: datetime.fromisoformat(row["acq_time"].replace("Z", "+00:00"))
            for row in csv.DictReader(handle, delimiter="\t")
            if row["filename"].startswith("meg/")
        }


def nearest_emptyroom(bids_root: str | Path, subject: str) -> tuple[Path, float]:
    """The empty room acquired nearest a subject's first run, and how many hours away.

    Noise was acquired once per recording day, so for 31 subjects this is
    their day's own session, the one Stage 0 recorded as ``AssociatedEmptyRoom``.
    H05 was recorded on a day without one and takes the nearest: the same room
    and sensors, a day later.
    """
    root = Path(bids_root)
    scans = root / f"sub-{subject}" / f"sub-{subject}_scans.tsv"
    if not scans.is_file():
        raise FileNotFoundError(f"No {scans.name} for {subject} under {root}")
    start = min(_acquisitions(scans).values())
    candidates = [
        (abs(acquired - start), session / filename)
        for session in sorted((root / "sub-emptyroom").glob("ses-*"))
        for filename, acquired in _acquisitions(session / f"sub-emptyroom_{session.name}_scans.tsv").items()
    ]
    if not candidates:
        raise FileNotFoundError(f"No staged empty room under {root / 'sub-emptyroom'}")
    distance, path = min(candidates)
    return path, distance.total_seconds() / 3600


# Trigger-channel codes; the 319 recordings carry these two and no others.
# H06's Start and Go are swapped relative to every other subject, so a
# subject-blind mapping would label every H06 go pulse Start in the written table.
TRIGGER_CODES = {"Start": 262144, "Go": 524288}
SUBJECT_TRIGGER_CODES = {"H06": {"Start": 524288, "Go": 262144}}


def trigger_codes(subject: str) -> dict[str, int]:
    """Every trigger's code for this subject.

    Both at once: ``write_raw_bids`` refuses a code in ``events`` it has no
    name for.
    """
    return {**TRIGGER_CODES, **SUBJECT_TRIGGER_CODES.get(subject, {})}


def write_meg_bids(
    raw_path: str | Path,
    *,
    subject: str,
    condition: str,
    run: int | str,
    bids_root: str | Path,
    task: str = "tokens",
    empty_room_bids_path: Optional[BIDSPath] = None,
    overwrite: bool = False,
) -> BIDSPath:
    """Copy one matched raw session into the BIDS-MEG raw layer.

    Entities follow ``docs/data_contract.md``'s "Stage 0" section: the
    condition becomes ``acq-<condition>`` (``Slow2`` -> ``acq-slow_run-2``)
    because ``desc`` -- the entity that reads most naturally here -- is
    derivatives-only in BIDS and illegal on raw data; ``task`` stays the
    singular ``tokens`` across conditions, since Slow/Fast/RT are timing
    variants of one task rather than three tasks; and no ``ses`` entity is
    used, because every subject was acquired on a single date.

    ``*_events.tsv`` is derived from the real trigger channel
    (``mne.find_events`` + :func:`trigger_codes`), not
    from the TDMS-derived behavior table -- an objective record of the MEG
    acquisition's own trigger pulses, complementary to (not a duplicate
    of) ``BIDS/sub-*/beh``. Passing ``event_id`` explicitly is what keeps
    the ``trial_type`` labels consistent with this project's own codes.

    ``empty_room_bids_path`` must already exist under this same
    ``bids_root``: ``write_raw_bids`` stores it as a root-relative path in
    ``*_meg.json``'s ``AssociatedEmptyRoom`` and raises ``FileNotFoundError``
    if the file is not there, which is why the workflow materializes every
    ``noise`` manifest row before any ``run`` row. ``None`` simply omits the
    field, so a subject whose recording day has no noise session still
    stages.

    ``overwrite`` defaults to ``False`` but the workflow always passes
    ``True``: ``coordsystem.json`` is subject+``acq``-scoped while real head
    digitization varies run to run by sub-millimeter fitting noise, so the
    second run of an ``acq`` would otherwise be refused over a difference
    that means nothing here (this project coregisters from a
    separately-managed ``-trans.fif``, never from ``coordsystem.json``).
    """
    bids_path = BIDSPath(
        subject=subject,
        task=task,
        acquisition=condition.lower(),
        run=str(run),
        datatype="meg",
        root=Path(bids_root),
    )
    raw = mne.io.read_raw_ctf(str(raw_path), preload=False, verbose=False)
    events = mne.find_events(raw, verbose=False)
    # CTF's MarkerFile.mrk is read into raw.annotations automatically (the
    # same Start/Go pulses as the trigger channel). write_raw_bids writes the
    # *union* of `events` and raw.annotations, which double-counts them; the
    # trigger-channel `events` above is the authoritative record, and the one
    # epoching reads back from `events.tsv`, so drop the annotations per
    # mne_bids' own documented fix for this. Without this
    # line H02's acq-slow run-1 wrote each Start/Go pulse twice; with it the
    # table is the trigger channel exactly (111 rows, 56 Start / 55 Go, no
    # duplicated sample+trial_type) -- pinned in tests/test_meg_bids.py.
    raw.set_annotations(None)
    write_raw_bids(
        raw,
        bids_path=bids_path,
        events=events,
        event_id=trigger_codes(subject),
        empty_room=empty_room_bids_path,
        overwrite=overwrite,
        verbose=False,
    )
    return bids_path


def write_emptyroom_bids(
    noise_path: str | Path,
    *,
    date: str,
    bids_root: str | Path,
    task: str = "noise",
    overwrite: bool = False,
) -> BIDSPath:
    """Copy one empty-room recording under the standard BIDS ``sub-emptyroom`` tree.

    Empty-room noise was acquired once per recording day (the legacy
    ``NOISE_noise_<YYYYMMDD>_01.ds`` convention) and is shared by whichever
    subject(s) were recorded that day, so it belongs to no subject. BIDS
    reserves ``sub-emptyroom`` with the acquisition date as the session
    label for exactly this, and mne_bids only recognises a recording as
    empty-room -- skipping ``*_events.tsv``, which has no meaning without a
    task -- when ``subject == "emptyroom"`` and ``task == "noise"``; hence
    those two fixed values.

    ``date`` (``YYYYMMDD``) therefore does double duty: the ``ses`` entity,
    and the join key ``write_meg_bids``'s ``empty_room_bids_path`` uses to
    point a subject's runs back here. It is not taken on trust --
    ``write_raw_bids`` raises if it disagrees with the recording's own
    ``meas_date``, so a mislabeled manifest row fails loudly instead of
    associating the wrong day's noise.

    This tree is what ``meg_tokens.workflows.sources``' noise covariance reads,
    reached through :func:`nearest_emptyroom`.
    """
    bids_path = BIDSPath(
        subject="emptyroom",
        session=date,
        task=task,
        datatype="meg",
        root=Path(bids_root),
    )
    raw = mne.io.read_raw_ctf(str(noise_path), preload=False, verbose=False)
    write_raw_bids(
        raw,
        bids_path=bids_path,
        overwrite=overwrite,
        verbose=False,
    )
    return bids_path
