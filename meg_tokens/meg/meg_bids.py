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

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import mne
from mne_bids import BIDSPath, write_raw_bids

if TYPE_CHECKING:
    from meg_tokens.io import DerivativeLayout

from meg_tokens.core import RunSpec, normalize_subject_id
from meg_tokens.meg.epoching import DEFAULT_EVENT_IDS, get_event_id

mne.set_log_level("ERROR")


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
    """Where Stage 0 stages one recording. ``subject`` must be canonical."""
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
    """Where Stage 0 stages a subject's head shape. ``subject`` must be canonical."""
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


def preprocessed_meg_runs(
    layout: "DerivativeLayout",
    subject: str,
    *,
    processing: Optional[str] = None,
) -> list[tuple[RunSpec, Path]]:
    """Every preprocessed run of one subject, with its identity, in protocol order.

    The derivative-side counterpart to :func:`raw_meg_sessions`, and it walks
    the same :data:`RAW_MEG_RUNS` grid for the same reason: a run that failed
    to preprocess is then a *named* absence rather than a shorter list nobody
    notices. Globbing the derivative directory cannot tell those apart.

    ``processing`` defaults to preferring ``clean`` over ``filt``, matching
    what ``preprocess_run`` writes with and without ICA, so epoching consumes
    the more processed derivative when both exist.
    """
    subject = normalize_subject_id(subject)
    candidates = (processing,) if processing else ("clean", "filt")
    runs = []
    for condition, run in RAW_MEG_RUNS:
        for candidate in candidates:
            path = layout.preprocessed_raw(
                subject=subject,
                run=str(run),
                condition=condition,
                processing=candidate,
            )
            if path.is_file():
                runs.append(
                    (RunSpec(subject=subject, run=str(run), condition=condition), path)
                )
                break
    return runs


def epoched_meg_runs(
    layout: "DerivativeLayout",
    subject: str,
    *,
    alignment: str,
) -> list[tuple[RunSpec, Path]]:
    """Every epoched run of one subject for one alignment, in protocol order.

    The third counterpart to :func:`raw_meg_sessions`, walking the same
    :data:`RAW_MEG_RUNS` grid, so the property holds all the way down the
    chain: a run absent from the source stage is a named run that failed to
    epoch, not an entry the discovery quietly never produced.
    """
    subject = normalize_subject_id(subject)
    runs = []
    for condition, run in RAW_MEG_RUNS:
        path = layout.epochs(
            subject=subject,
            run=str(run),
            condition=condition,
            alignment=alignment,
        )
        if path.is_file():
            runs.append(
                (RunSpec(subject=subject, run=str(run), condition=condition), path)
            )
    return runs


def associated_emptyroom(
    bids_root: str | Path,
    subject: str,
    *,
    task: str = "tokens",
) -> Path:
    """The empty-room recording this subject's runs were staged against.

    ``write_meg_bids`` already recorded the association, as the root-relative
    ``AssociatedEmptyRoom`` in each run's ``*_meg.json``, so this reads what
    Stage 0 decided rather than re-deriving it from acquisition dates.

    Noise was acquired once per recording day and every run of a subject
    resolves to the same session, but that is a property of the data rather
    than a guarantee of the format: a subject whose runs disagree is refused
    instead of being given whichever one sorted first.
    """
    subject = normalize_subject_id(subject)
    root = Path(bids_root)
    sessions = raw_meg_sessions(root, subject, task=task)
    if not sessions:
        raise FileNotFoundError(
            f"No staged raw MEG for {subject} under {root}. "
            "Run `meg-tokens meg apply-raw-staging` first."
        )

    recorded = set()
    for _, path in sessions:
        sidecar = path.with_suffix(".json")
        if not sidecar.is_file():
            continue
        associated = json.loads(sidecar.read_text(encoding="utf-8")).get(
            "AssociatedEmptyRoom"
        )
        if associated:
            recorded.add(str(associated))

    if not recorded:
        raise FileNotFoundError(
            f"{subject} has no AssociatedEmptyRoom in any run sidecar: no "
            "empty-room session was acquired on its recording day. Empty-room "
            "noise covariance is not available for this subject."
        )
    if len(recorded) > 1:
        raise ValueError(
            f"{subject}'s runs name {len(recorded)} different empty-room "
            f"sessions ({sorted(recorded)}); a per-subject noise covariance "
            "cannot stand for all of them."
        )

    # BIDS stores it root-relative with a leading separator.
    noise_path = root / str(recorded.pop()).lstrip("/")
    if not noise_path.is_dir():
        raise FileNotFoundError(
            f"{subject}'s associated empty room {noise_path} is not staged. "
            "Run `meg-tokens meg apply-raw-staging` to materialize it."
        )
    return noise_path


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

    This tree is what ``meg_tokens.workflows.sources``' noise covariance reads,
    reached through :func:`associated_emptyroom`.
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
