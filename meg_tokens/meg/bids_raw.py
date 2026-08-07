"""BIDS-MEG raw layer writer.

Real copies via ``mne_bids.write_raw_bids`` (``symlink=False``) -- the raw
CTF media and ``tdms/`` stay untouched read-only sources; ``BIDS/`` is an
independent copy. ``write_raw_bids`` handles the CTF internal-file
renaming (every file inside a ``.ds`` must share the enclosing directory's
basename) and all the standard sidecars (``channels.tsv``, ``*_meg.json``,
``coordsystem.json``, ``dataset_description.json``, ``participants.tsv``)
itself, so this module only supplies entities, real trigger events (for
``*_events.tsv``), and the empty-room link.
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

    ``*_events.tsv`` is derived from the real trigger channel
    (``mne.find_events`` + ``meg_tokens.meg.epoching.get_event_id``), not
    from the TDMS-derived behavior table -- an objective record of the MEG
    acquisition's own trigger pulses, complementary to (not a duplicate
    of) ``BIDS/sub-*/beh``.
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
    # annotations per mne_bids' own documented fix for this.
    raw.set_annotations(None)
    write_raw_bids(
        raw,
        bids_path=bids_path,
        events=events,
        event_id=_combined_event_id(subject),
        empty_room=empty_room_bids_path,
        symlink=False,
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

    ``date`` (``YYYYMMDD``) is the acquisition date shared with the real
    subject(s) recorded that day, used as the BIDS session label so
    ``write_meg_bids``'s ``empty_room_bids_path`` can link back to it.
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
        symlink=False,
        overwrite=overwrite,
        verbose=False,
    )
    return bids_path
