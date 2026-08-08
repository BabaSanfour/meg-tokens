import os

import pandas as pd
import pytest

from meg_tokens.behavior.schema import CLASSIFICATION_COLUMNS, RAW_TRIAL_COLUMNS
from meg_tokens.behavior.tdms_bids import (
    raw_behavior_files,
    read_raw_behavior_sidecar,
    write_beh_bids,
)

EVENTS_STR = """sTaskType: 'TokensMvt'
dDate: '2018/02/13 02:20:52.738 PM'
nInitialTime: 6232233
nTrialIndex: 1
sNeuralFilename: 'H2Slow1'
nChoiceMade: 2
nCorrectChoice: 2
nSelectedTarget: 4
nUnselectedTarget: 1
tEnterCenter: 3
tClick: 3
tGO: 1041
tExitCenter: 2484
tEnterTarget: 2484
tTrialEnd: 3676
nOutcome: 0
sTrialClass: 'x'
sTokenDirs: '221221122212211'
Tokens.Data: (
[tTime: 1260, nTokenNum: 1, nTokenDir: 2, nProb: 0.70947265625000, nTokenX: 575, nTokenY: 474],
[tTime: 1460, nTokenNum: 2, nTokenDir: 2, nProb: 0.81234567890123, nTokenX: 576, nTokenY: 475]
)
"""


class _FakeGroup:
    def __init__(self, name, properties):
        self.name = name
        self.properties = properties


class _FakeTdmsFile:
    def __init__(self, groups):
        self._groups = groups

    def groups(self):
        return self._groups


def test_write_beh_bids_uses_bids_entities_and_omits_classification(tmp_path, monkeypatch):
    fake = _FakeTdmsFile([_FakeGroup("TrialData0", {"Events": EVENTS_STR})])
    monkeypatch.setattr("meg_tokens.behavior.tdms.TdmsFile", lambda path: fake)

    tdms_path = tmp_path / "H02Slow1_180213.tdms"
    tdms_path.write_text("")  # content is irrelevant; TdmsFile is mocked
    bids_root = tmp_path / "BIDS"

    written = write_beh_bids(tdms_path, bids_root=bids_root)

    assert written == (
        bids_root / "sub-H02" / "beh" / "sub-H02_task-tokens_acq-slow_run-1_beh.tsv"
    )
    df = pd.read_csv(written, sep="\t")
    assert len(df) == 1
    # The raw layer asserts no class -- not even a placeholder saying it
    # declined to. It keeps what inference consumes, so the derivative stays
    # reproducible from here.
    assert list(df.columns) == RAW_TRIAL_COLUMNS
    assert set(CLASSIFICATION_COLUMNS).isdisjoint(df.columns)
    assert df.loc[0, "sTrialClassRaw"] == "x"
    assert df.loc[0, "sp_design_correct"]

    sidecar = written.with_suffix(".json")
    assert sidecar.exists()


def test_write_beh_bids_rejects_non_standard_filename(tmp_path):
    tdms_path = tmp_path / "temp_180214.tdms"
    tdms_path.write_text("")
    with pytest.raises(ValueError):
        write_beh_bids(tdms_path, bids_root=tmp_path / "BIDS")


def test_raw_behavior_files_finds_every_staged_run_for_one_subject(tmp_path, monkeypatch):
    fake = _FakeTdmsFile([_FakeGroup("TrialData0", {"Events": EVENTS_STR})])
    monkeypatch.setattr("meg_tokens.behavior.tdms.TdmsFile", lambda path: fake)
    bids_root = tmp_path / "BIDS"

    for name in ("H02Slow1_180213.tdms", "H02Fast1_180213.tdms"):
        path = tmp_path / name
        path.write_text("")
        write_beh_bids(path, bids_root=bids_root)

    assert [p.name for p in raw_behavior_files(bids_root, "h2")] == [
        "sub-H02_task-tokens_acq-fast_run-1_beh.tsv",
        "sub-H02_task-tokens_acq-slow_run-1_beh.tsv",
    ]
    # A subject Stage 0 never staged is indistinguishable from one with no runs.
    assert raw_behavior_files(bids_root, "H09") == []


def test_read_raw_behavior_sidecar_recovers_run_identity(tmp_path, monkeypatch):
    fake = _FakeTdmsFile([_FakeGroup("TrialData0", {"Events": EVENTS_STR})])
    monkeypatch.setattr("meg_tokens.behavior.tdms.TdmsFile", lambda path: fake)
    tdms_path = tmp_path / "H02Slow1_180213.tdms"
    tdms_path.write_text("")

    written = write_beh_bids(tdms_path, bids_root=tmp_path / "BIDS")
    metadata = read_raw_behavior_sidecar(written)

    # Recovered from the sidecar, not re-derived from (lowercased, dateless)
    # BIDS entities.
    assert metadata["subject"] == "H02"
    assert metadata["condition"] == "Slow"
    assert metadata["run"] == 1
    assert metadata["acquisition_date"] == "180213"
    assert metadata["source_file"] == str(tdms_path)


def test_read_raw_behavior_sidecar_raises_without_one(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_raw_behavior_sidecar(tmp_path / "no-such_beh.tsv")


def test_write_beh_bids_real_tdms_file_integration():
    real_root = os.environ.get("MEG_TOKENS_TDMS_ROOT")
    if not real_root:
        pytest.skip("Set MEG_TOKENS_TDMS_ROOT to the behavioral log root to run this test.")
    real_path = os.path.join(real_root, "H02", "H02Slow1_180213.tdms")
    if not os.path.exists(real_path):
        pytest.skip(f"Real TDMS data not available at {real_path}")

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        written = write_beh_bids(real_path, bids_root=tmp)
        df = pd.read_csv(written, sep="\t")
        assert not df.empty
        # A real run stages as pure transcription: no class asserted, and
        # both inputs classification later reads still present.
        assert list(df.columns) == RAW_TRIAL_COLUMNS
        assert set(CLASSIFICATION_COLUMNS).isdisjoint(df.columns)
        assert set(df["sTrialClassRaw"].astype(str)) <= {"x", "e", "a", "m", "r"}
        assert df["sp_design_correct"].notna().any()
