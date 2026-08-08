import os

import pandas as pd
import pytest

from meg_tokens.behavior.tdms_bids import write_beh_bids

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


def test_write_beh_bids_uses_bids_entities_and_disables_inference(tmp_path, monkeypatch):
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
    # 'x' trials are raw/unresolved without inference -- sTrialClass stays 0
    # and the provenance column records why, exactly like
    # parse_tdms_file(..., infer_random_classes=False) documents.
    assert df.loc[0, "sTrialClass"] == 0
    assert df.loc[0, "trial_class_rule"] == "inference_disabled"

    sidecar = written.with_suffix(".json")
    assert sidecar.exists()


def test_write_beh_bids_rejects_non_standard_filename(tmp_path):
    tdms_path = tmp_path / "temp_180214.tdms"
    tdms_path.write_text("")
    with pytest.raises(ValueError):
        write_beh_bids(tdms_path, bids_root=tmp_path / "BIDS")


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
        # infer_random_classes=False means no row can go through the
        # design-profile inference path (trial_class_source == "inferred").
        assert (df["trial_class_source"] != "inferred").all()
