import os

import pandas as pd
import pytest

from meg_tokens.meg.meg_bids import _combined_event_id, write_emptyroom_bids, write_meg_bids


def test_combined_event_id_covers_all_alignments():
    event_id = _combined_event_id("H02")
    assert event_id == {
        "Start": 262144,
        "Go": 524288,
        "Enter": 1048576,
        "Feedback": 2097152,
    }


def test_combined_event_id_respects_subject_overrides():
    # H06 has Start/Go codes swapped relative to every other subject (see
    # meg_tokens.meg.epoching.SUBJECT_EVENT_OVERRIDES).
    event_id = _combined_event_id("H06")
    assert event_id["Go"] == 262144
    assert event_id["Start"] == 524288


@pytest.fixture()
def real_raw_root():
    data_root = os.environ.get("MEG_TOKENS_DATA_ROOT")
    if not data_root:
        pytest.skip("Set MEG_TOKENS_DATA_ROOT to the project's data_root to run this test.")
    root = os.path.join(data_root, "raw")
    session = os.path.join(root, "H02_DDM-tthiery_20180213_03.ds")
    noise = os.path.join(root, "NOISE_noise_20180213_01.ds")
    if not (os.path.exists(session) and os.path.exists(noise)):
        pytest.skip(f"Real raw MEG sessions not available under {root}")
    return root, session, noise


def test_write_meg_bids_and_write_emptyroom_bids_real_media_integration(tmp_path, real_raw_root):
    _, session, noise = real_raw_root
    bids_root = tmp_path / "BIDS"

    er_path = write_emptyroom_bids(noise, date="20180213", bids_root=bids_root)
    bids_path = write_meg_bids(
        session,
        subject="H02",
        condition="Slow",
        run=1,
        bids_root=bids_root,
        empty_room_bids_path=er_path,
    )

    from mne_bids import read_raw_bids

    raw = read_raw_bids(bids_path, verbose=False)
    assert raw.n_times == 378000
    assert raw.info["sfreq"] == 1200.0

    # A real copy, not a reference back to the media.
    ds_dir = bids_path.fpath
    assert ds_dir.is_dir() and not ds_dir.is_symlink()
    res4_files = list(ds_dir.glob("*.res4"))
    assert res4_files and not res4_files[0].is_symlink()
    total_size = sum(f.stat().st_size for f in bids_root.rglob("*") if f.is_file())
    assert total_size > 100_000_000  # a real ~500MB session was actually copied

    channels = pd.read_csv(
        bids_path.copy().update(suffix="channels", extension=".tsv").fpath, sep="\t"
    )
    assert {"MEGREFMAG", "MEGREFGRADAXIAL"} & set(channels["type"])

    meg_json_path = bids_path.copy().update(suffix="meg", extension=".json").fpath
    import json

    payload = json.loads(meg_json_path.read_text())
    assert payload["AssociatedEmptyRoom"] == (
        "/sub-emptyroom/ses-20180213/meg/sub-emptyroom_ses-20180213_task-noise_meg.ds"
    )

    # Regression: write_raw_bids writes the *union* of the passed `events`
    # and raw.annotations (CTF's MarkerFile.mrk, read in automatically) --
    # without dropping the annotations first this produced double-counted
    # rows (each real trigger pulse appearing twice).
    events = pd.read_csv(
        bids_path.copy().update(suffix="events", extension=".tsv").fpath, sep="\t"
    )
    assert len(events) == 111
    assert events["trial_type"].value_counts().to_dict() == {"Start": 56, "Go": 55}
    assert events.duplicated(subset=["sample", "trial_type"]).sum() == 0


def test_importing_the_module_does_not_change_mne_logging():
    """A library module must not reconfigure logging for the whole process.

    Five workflow modules import this one for a path helper; setting the log
    level at module scope meant any of those imports silently discarded a
    caller's own logging choice.
    """
    import subprocess
    import sys

    probe = (
        "import mne, sys; before = mne.utils.logger.level;"
        "import meg_tokens.meg.meg_bids, meg_tokens.meg.anat_bids;"
        "sys.exit(0 if mne.utils.logger.level == before else 1)"
    )
    assert subprocess.run([sys.executable, "-c", probe]).returncode == 0


def test_a_missing_sidecar_is_not_reported_as_a_missing_recording(tmp_path):
    """Absent sidecars and sidecars naming no noise are different facts.

    Only the second says anything about the acquisition, and H05 -- the one
    subject with no empty room -- is exactly where that distinction matters.
    """
    from meg_tokens.meg.meg_bids import associated_emptyroom, raw_meg_path

    staged = raw_meg_path(tmp_path, "H01", "Fast", 1)
    staged.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="no sidecar"):
        associated_emptyroom(tmp_path, "H01")

    staged.with_suffix(".json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="no empty-room session was acquired"):
        associated_emptyroom(tmp_path, "H01")
