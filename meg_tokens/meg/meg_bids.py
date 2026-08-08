"""BIDS-MEG raw layer writer: real copies of matched CTF sessions.

The writer half of Stage 0. ``meg_tokens.meg.raw_staging`` decides *which*
numbered raw session is which behavioral run and records that in the
manifest; ``meg_tokens.workflows.raw_staging`` applies the reviewed
manifest; this module performs the copy itself.

Copies are real and one-directional: the raw acquisition tree and ``tdms/``
stay untouched read-only sources, and ``BIDS/`` is an independent copy that
can be deleted and rebuilt from them at any time.

``mne_bids.write_raw_bids`` does the copying rather than a plain
``shutil.copytree`` because a CTF session is not relocatable by copying
alone -- every file inside a ``.ds`` must share the enclosing directory's
basename, so giving the directory a BIDS name means renaming its contents
too -- and because it also emits the standard sidecars (``channels.tsv``,
``*_meg.json``, ``coordsystem.json``, ``dataset_description.json``,
``participants.tsv``) from the recording itself. What this module adds is
only what mne_bids cannot know: the BIDS entities, the real trigger events
written to ``*_events.tsv`` under this project's subject-aware event codes,
and the empty-room link.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import mne
from mne_bids import BIDSPath, write_raw_bids

from meg_tokens.core import normalize_subject_id
from meg_tokens.meg.epoching import DEFAULT_EVENT_IDS, get_event_id

mne.set_log_level("ERROR")


def _combined_event_id(subject: str) -> dict:
    """Every alignment's trigger code in one dict, for this one subject.

    All four alignments are merged rather than passing whichever one a later
    analysis happens to align to, because ``write_raw_bids`` requires a name
    for *every* code appearing in ``events`` and refuses the write otherwise
    -- and because ``*_events.tsv`` is a record of the acquisition, which
    should not depend on any downstream epoching choice.

    Built through ``get_event_id(..., subject)`` rather than read from
    ``DEFAULT_EVENT_IDS`` directly so the raw layer's ``trial_type`` labels
    agree with what ``meg_tokens.meg.epoching`` uses everywhere else: H06's
    Start and Go codes are swapped relative to every other subject
    (``SUBJECT_EVENT_OVERRIDES``), so a subject-blind mapping would silently
    label every H06 Go pulse ``Start`` in the written table.
    """
    event_id: dict = {}
    for alignment in DEFAULT_EVENT_IDS:
        event_id.update(get_event_id(alignment, subject))
    return event_id


def write_meg_bids(
    raw_path: str | Path,
    *,
    subject: str,
    condition: str,
    run: int,
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
    (``mne.find_events`` + ``meg_tokens.meg.epoching.get_event_id``), not
    from the TDMS-derived behavior table -- an objective record of the MEG
    acquisition's own trigger pulses, complementary to (not a duplicate
    of) ``BIDS/sub-*/beh``. Passing ``event_id`` explicitly is what keeps
    the ``trial_type`` labels consistent with this project's own codes; see
    ``_combined_event_id``.

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
    subject = normalize_subject_id(subject)
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
    # CTF's MarkerFile.mrk is read into raw.annotations automatically (a
    # subset of the real trigger channel -- Start/Go only, not Enter/
    # Feedback). write_raw_bids writes the *union* of `events` and
    # raw.annotations, which double-counts the overlap; the STIM-channel
    # `events` above is the complete, authoritative record (matching what
    # meg_tokens.meg.epoching uses everywhere else), so drop the
    # annotations per mne_bids' own documented fix for this. Without this
    # line H02's acq-slow run-1 wrote each Start/Go pulse twice; with it the
    # table is the trigger channel exactly (111 rows, 56 Start / 55 Go, no
    # duplicated sample+trial_type) -- pinned in tests/test_meg_bids.py.
    raw.set_annotations(None)
    write_raw_bids(
        raw,
        bids_path=bids_path,
        events=events,
        event_id=_combined_event_id(subject),
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

    This tree does not feed ``meg_tokens.workflows.sources``' noise-covariance
    lookup, which still reads ``ProjectConfig.noise_dir`` directly.
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
