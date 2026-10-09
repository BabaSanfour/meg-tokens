import pandas as pd

from meg_tokens.behavior.schema import CLASSIFICATION_COLUMNS, RAW_TRIAL_COLUMNS
from meg_tokens.behavior.tdms_bids import read_raw_behavior_sidecar, write_beh_bids

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
