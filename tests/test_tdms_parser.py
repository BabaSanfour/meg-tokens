"""Transcription tests for the TDMS parser.

The parser transcribes what LabVIEW wrote and derives the designed
success-probability profile; it assigns no trial class. Everything about
which class a trial belongs to now lives in
``tests/behavior/test_classification.py``, next to the module that decides
it.
"""

import os
import re
import pytest
import pandas as pd
from meg_tokens.behavior.math.probability import success_probability_profile
from meg_tokens.behavior.schema import (
    CLASSIFICATION_COLUMNS,
    RAW_TRIAL_COLUMNS,
    parse_token_directions,
)
from meg_tokens.behavior.tdms import (
    parse_single_trial,
    parse_tdms_file,
)

# Sample Events string mimicking the real TDMS structure
EVENTS_STR = """sTaskType: 'TokensMvt'
dDate: '2018/01/31 04:20:52.738 PM'
nInitialTime: 6232233
nTrialIndex: 1
sNeuralFilename: 'H1Slow2'
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


def make_events(
    prob_values,
    *,
    token_dirs="221221122212211",
    correct_choice=2,
    trial_class="x",
):
    steps = [
        f"[tTime: {1000 + i*200}, nTokenNum: {i+1}, nTokenDir: 2, nProb: {p}]"
        for i, p in enumerate(prob_values)
    ]
    lines = EVENTS_STR.splitlines()
    idx = lines.index("Tokens.Data: (")
    new_lines = lines[:idx+1] + steps + [")"]
    return (
        "\n".join(new_lines)
        .replace("nCorrectChoice: 2", f"nCorrectChoice: {correct_choice}")
        .replace("sTrialClass: 'x'", f"sTrialClass: '{trial_class}'")
        .replace("sTokenDirs: '221221122212211'", f"sTokenDirs: '{token_dirs}'")
    )

def test_parse_single_trial():
    trial_dict = parse_single_trial(EVENTS_STR)
    
    assert trial_dict['nTrialIndex'] == 1
    assert trial_dict['nChoiceMade'] == 2
    assert trial_dict['nCorrectChoice'] == 2
    assert trial_dict['tGO'] == 1041
    assert trial_dict['tEnterTarget'] == 2484
    assert trial_dict['tTrialEnd'] == 3676
    assert trial_dict['sTokenDirs'] == '221221122212211'
    assert trial_dict['nTokenNum'] == [1, 2]
    assert trial_dict['nTokenDir'] == [2, 2]
    assert trial_dict['tTime'] == [1260, 1460]
    assert trial_dict['nProb'] == [0.70947265625000, 0.81234567890123]
    
    assert trial_dict['sTrialClassRaw'] == 'x'
    assert len(trial_dict['sp_design_correct']) == 15
    assert trial_dict['token_log_rows'] == 2
    assert trial_dict['token_log_short'] is False

    # Trial classes are a derivative-stage judgement; the parser emits the
    # raw label and the profile inference reads, and nothing more.
    assert set(trial_dict) == set(RAW_TRIAL_COLUMNS)
    assert set(CLASSIFICATION_COLUMNS).isdisjoint(trial_dict)


def test_parse_single_trial_raises_on_missing_structural_field():
    events_str = "\n".join(
        line for line in EVENTS_STR.splitlines() if not line.startswith("tGO:")
    )

    with pytest.raises(ValueError, match=re.escape("missing required field(s) ['tGO']")):
        parse_single_trial(events_str)


def test_parse_single_trial_raises_on_unrecognized_trial_class():
    events_str = EVENTS_STR.replace("sTrialClass: 'x'", "sTrialClass: 'q'")

    with pytest.raises(ValueError, match="unrecognized sTrialClass value 'q'"):
        parse_single_trial(events_str)


def test_parse_single_trial_allows_missing_tenter_target_when_skipped():
    events_str = "\n".join(
        line for line in EVENTS_STR.splitlines() if not line.startswith("tEnterTarget:")
    ).replace("nChoiceMade: 2", "nChoiceMade: 0")

    trial_dict = parse_single_trial(events_str)

    assert trial_dict['nChoiceMade'] == 0
    assert trial_dict['tEnterTarget'] == 0

@pytest.mark.parametrize("raw_label", ["e", "a", "m", "x", "r", "2"])
def test_recorded_trial_label_is_transcribed_verbatim(raw_label):
    """Every accepted label survives as text, and none of them buys the row a
    class: reading a class out of a label is the derivative stage's call."""
    events_str = make_events([0.9] * 15, trial_class=raw_label)
    trial_dict = parse_single_trial(events_str)

    assert trial_dict['sTrialClassRaw'] == raw_label
    assert set(CLASSIFICATION_COLUMNS).isdisjoint(trial_dict)


def test_design_profile_comes_from_the_token_sequence_not_the_logged_path():
    """``sp_design_correct`` is what class inference later reads. It is derived
    from the designed sequence and the correct target, so a logged probability
    path that disagrees with it cannot leak into a class."""
    events_str = make_events([0.1] * 15)
    trial_dict = parse_single_trial(events_str)

    assert trial_dict['sp_design_correct'] == success_probability_profile(
        parse_token_directions("221221122212211"), target=2
    )
    assert trial_dict['sp_design_correct'] != trial_dict['nProb']


def test_trial_without_a_correct_target_has_no_design_profile():
    events_str = EVENTS_STR.replace("nChoiceMade: 2", "nChoiceMade: 0").replace(
        "nCorrectChoice: 2", "nCorrectChoice: 0"
    )
    trial_dict = parse_single_trial(events_str)

    assert trial_dict['sp_design_correct'] is None


def test_design_profile_is_identical_for_14_and_15_log_rows():
    """A truncated runtime log shortens nProb but not the designed profile,
    which is why a 14-row trial classifies exactly like a 15-row one."""
    kwargs = {
        "token_dirs": "212112111111212",
        "correct_choice": 1,
    }
    short = parse_single_trial(make_events([0.5] * 14, **kwargs))
    complete = parse_single_trial(make_events([0.5] * 15, **kwargs))

    assert short['sp_design_correct'] == complete['sp_design_correct']
    assert len(short['sp_design_correct']) == 15
    assert short['token_log_rows'] == 14
    assert short['token_log_short'] is True
    assert complete['token_log_rows'] == 15
    assert complete['token_log_short'] is False

class _FakeGroup:
    def __init__(self, name, properties):
        self.name = name
        self.properties = properties


class _FakeTdmsFile:
    def __init__(self, groups):
        self._groups = groups

    def groups(self):
        return self._groups


def test_parse_tdms_file_raises_on_group_missing_events_property(monkeypatch):
    fake = _FakeTdmsFile([_FakeGroup("TrialData0", {})])
    monkeypatch.setattr("meg_tokens.behavior.tdms.TdmsFile", lambda path: fake)

    with pytest.raises(
        ValueError, match=re.escape("no 'Events' property: ['TrialData0']")
    ):
        parse_tdms_file("fake.tdms")


def test_parse_tdms_file_raises_when_no_trial_groups_found(monkeypatch):
    fake = _FakeTdmsFile([_FakeGroup("Settings", {})])
    monkeypatch.setattr("meg_tokens.behavior.tdms.TdmsFile", lambda path: fake)

    with pytest.raises(ValueError, match="no TrialData groups with trial data found"):
        parse_tdms_file("fake.tdms")


def test_parse_tdms_file_wraps_trial_error_with_file_path(monkeypatch):
    bad_events = "\n".join(
        line for line in EVENTS_STR.splitlines() if not line.startswith("tGO:")
    )
    fake = _FakeTdmsFile([_FakeGroup("TrialData0", {"Events": bad_events})])
    monkeypatch.setattr("meg_tokens.behavior.tdms.TdmsFile", lambda path: fake)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "fake.tdms: Trial 1: Events block is missing required field(s) "
            "['tGO']"
        ),
    ):
        parse_tdms_file("fake.tdms")


def test_parse_tdms_file_wraps_dataframe_validation_error_with_file_path(monkeypatch):
    fake = _FakeTdmsFile(
        [
            _FakeGroup("TrialData0", {"Events": EVENTS_STR}),
            _FakeGroup("TrialData1", {"Events": EVENTS_STR}),  # duplicate index 1
        ]
    )
    monkeypatch.setattr("meg_tokens.behavior.tdms.TdmsFile", lambda path: fake)

    with pytest.raises(
        ValueError,
        match=re.escape("fake.tdms: nTrialIndex must be a gap-free consecutive sequence"),
    ):
        parse_tdms_file("fake.tdms")


def test_parse_real_tdms_integration():
    real_root = os.environ.get("MEG_TOKENS_TDMS_ROOT")
    if not real_root:
        pytest.skip("Set MEG_TOKENS_TDMS_ROOT to the behavioral log root to run this test.")
    real_path = os.path.join(real_root, "H02", "H02Slow1_180213.tdms")
    if not os.path.exists(real_path):
        pytest.skip(f"Real TDMS data not available at {real_path}")

    df = parse_tdms_file(real_path)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == RAW_TRIAL_COLUMNS
    # TrialData indices should be sequential starting at 1
    assert df['nTrialIndex'].iloc[0] == 1
    # A real run records labels the acquisition software wrote, and no
    # column this project inferred.
    assert set(df['sTrialClassRaw']) <= {'x', 'e', 'a', 'm', 'r'}
    assert set(CLASSIFICATION_COLUMNS).isdisjoint(df.columns)


def test_parse_single_trial_reads_scientific_notation_nprob():
    """LabVIEW writes near-0/near-1 probabilities in scientific notation.

    A digits-and-dots pattern silently drops the exponent, turning 5.46875E-1
    into 5.46875 -- an impossible success probability that then corrupts trial
    classification. Verified on the real dataset: 870 trials were affected.
    """
    events_str = make_events(["5.46875E-1", "8.984375E-1", "-2.22044604925031E-16"])
    trial_dict = parse_single_trial(events_str)

    assert trial_dict["nProb"] == pytest.approx([0.546875, 0.8984375, 0.0])


def test_parse_single_trial_rejects_out_of_range_nprob():
    events_str = make_events(["0.5", "1.75", "0.25"])

    with pytest.raises(ValueError, match=r"nProb values outside \[0, 1\]"):
        parse_single_trial(events_str)


def test_parse_single_trial_rejects_desynced_token_fields():
    """A Tokens.Data row matching only some patterns would silently misalign
    every subsequent index-based criterion."""
    events_str = make_events(["0.5", "0.6"]).replace(
        "[tTime: 1200, nTokenNum: 2, nTokenDir: 2, nProb: 0.6]",
        "[tTime: 1200, nTokenNum: 2, nTokenDir: 2, nProb: nan]",
    )

    with pytest.raises(ValueError, match="Tokens.Data field lengths differ"):
        parse_single_trial(events_str)
