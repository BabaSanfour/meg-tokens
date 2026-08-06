import os
import re
import pytest
import pandas as pd
from meg_tokens.behavior.tdms import (
    OUTCOME_NEVER_STARTED,
    parse_single_trial,
    parse_tdms_file,
    validate_behavior_dataframe,
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


def test_never_started_outcome_matches_labview_contract():
    assert OUTCOME_NEVER_STARTED == 7003


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
    assert trial_dict['sTrialClass'] == 1
    assert trial_dict['trial_class_source'] == 'inferred'
    assert trial_dict['trial_class_rule'] == 'easy_sp2_sp5_sp8'
    assert len(trial_dict['sp_design_correct']) == 15
    assert trial_dict['token_log_rows'] == 2
    assert trial_dict['token_log_short'] is False


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

@pytest.mark.parametrize(("raw_label", "expected"), [("e", 1), ("a", 2), ("m", 3)])
def test_designed_trial_label_is_preserved(raw_label, expected):
    events_str = make_events([0.9] * 15, trial_class=raw_label)
    trial_dict = parse_single_trial(events_str)

    assert trial_dict['sTrialClassRaw'] == raw_label
    assert trial_dict['sTrialClass'] == expected
    assert trial_dict['trial_class_source'] == 'design'
    assert trial_dict['trial_class_rule'] == 'recorded_label'


def test_random_trial_uses_design_profile_instead_of_logged_probability():
    events_str = make_events([0.1] * 15)
    trial_dict = parse_single_trial(events_str)

    assert trial_dict['sTrialClass'] == 1
    assert trial_dict['trial_class_rule'] == 'easy_sp2_sp5_sp8'


def test_random_sp11_boundary_is_ambiguous():
    events_str = make_events(
        [0.5] * 15,
        token_dirs="212112111111212",
        correct_choice=1,
    )
    trial_dict = parse_single_trial(events_str)

    assert trial_dict['sTrialClass'] == 2
    assert trial_dict['trial_class_rule'] == 'ambiguous_sp11_boundary'


def test_random_unmatched_sequence_remains_unclassified():
    events_str = make_events(
        [0.5] * 15,
        token_dirs="111122221111111",
        correct_choice=1,
    )
    trial_dict = parse_single_trial(events_str)

    assert trial_dict['sTrialClass'] == 0
    assert trial_dict['trial_class_source'] == 'unclassified'
    assert trial_dict['trial_class_rule'] == 'unclassified'


def test_random_trial_without_correct_target_remains_unclassified():
    events_str = EVENTS_STR.replace("nChoiceMade: 2", "nChoiceMade: 0").replace(
        "nCorrectChoice: 2", "nCorrectChoice: 0"
    )
    trial_dict = parse_single_trial(events_str)

    assert trial_dict['sTrialClass'] == 0
    assert trial_dict['sp_design_correct'] is None
    assert trial_dict['trial_class_source'] == 'unclassified'
    assert trial_dict['trial_class_rule'] == 'no_correct_target'


def test_random_classification_is_identical_for_14_and_15_log_rows():
    kwargs = {
        "token_dirs": "212112111111212",
        "correct_choice": 1,
    }
    short = parse_single_trial(make_events([0.5] * 14, **kwargs))
    complete = parse_single_trial(make_events([0.5] * 15, **kwargs))

    assert short['sTrialClass'] == complete['sTrialClass'] == 2
    assert short['trial_class_rule'] == complete['trial_class_rule']
    assert short['sp_design_correct'] == complete['sp_design_correct']
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
    real_path = os.environ.get(
        "MEG_TOKENS_TDMS_ROOT",
        "/Users/hamzaabdelhedi/Projects/data/meg-tokens/tdms",
    ) + "/H1/H1Slow1_180131.tdms"
    if os.path.exists(real_path):
        df = parse_tdms_file(real_path)
        
        # Verify dataframe structure
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        expected_cols = [
            'nTrialIndex', 'sTrialClass', 'sTrialClassRaw',
            'trial_class_source', 'trial_class_rule', 'sp_design_correct',
            'nInitialTime', 'nChoiceMade',
            'nCorrectChoice', 'tGO', 'tEnterTarget', 'tTrialEnd',
            'sTokenDirs', 'nTokenNum', 'nTokenDir', 'tTime', 'nProb',
            'token_log_rows', 'token_log_short', 'nOutcome'
        ]
        assert list(df.columns) == expected_cols
        
        # TrialData indices should be sequential starting at 1
        assert df['nTrialIndex'].iloc[0] == 1
    else:
        pytest.skip("Real integration TDMS file not accessible.")


def test_validate_behavior_dataframe_rejects_invalid_event_order():
    df = pd.DataFrame({
        'nTrialIndex': [1],
        'sTrialClass': [1],
        'sTrialClassRaw': ['e'],
        'trial_class_source': ['design'],
        'trial_class_rule': ['recorded_label'],
        'sp_design_correct': [[0.6]],
        'nInitialTime': [0],
        'nChoiceMade': [1],
        'nCorrectChoice': [1],
        'tGO': [2000],
        'tEnterTarget': [1000],
        'tTrialEnd': [2500],
        'sTokenDirs': ['121'],
        'nTokenNum': [[1]],
        'nTokenDir': [[1]],
        'tTime': [[1100]],
        'nProb': [[0.6]],
        'token_log_rows': [1],
        'token_log_short': [False],
        'nOutcome': [0],
    })

    with pytest.raises(ValueError, match="Invalid event ordering"):
        validate_behavior_dataframe(df)


def _make_behavior_df(indices):
    n = len(indices)
    return pd.DataFrame({
        'nTrialIndex': indices,
        'sTrialClass': [1] * n,
        'sTrialClassRaw': ['e'] * n,
        'trial_class_source': ['design'] * n,
        'trial_class_rule': ['recorded_label'] * n,
        'sp_design_correct': [[0.6]] * n,
        'nInitialTime': list(range(n)),
        'nChoiceMade': [1] * n,
        'nCorrectChoice': [1] * n,
        'tGO': [500] * n,
        'tEnterTarget': [1000] * n,
        'tTrialEnd': [1500] * n,
        'sTokenDirs': ['121'] * n,
        'nTokenNum': [[1]] * n,
        'nTokenDir': [[1]] * n,
        'tTime': [[600]] * n,
        'nProb': [[0.6]] * n,
        'token_log_rows': [1] * n,
        'token_log_short': [False] * n,
        'nOutcome': [0] * n,
    })


def test_validate_behavior_dataframe_accepts_consecutive_run_not_starting_at_1():
    """Real files show nTrialIndex is a session-scoped counter, not reset per
    .tdms file -- a run starting above 1 is a benign offset, not corruption.
    """
    df = _make_behavior_df([7, 8, 9])
    validate_behavior_dataframe(df)  # should not raise


def test_validate_behavior_dataframe_rejects_index_below_1():
    df = _make_behavior_df([0, 1, 2])
    with pytest.raises(ValueError, match="nTrialIndex must start at 1 or above"):
        validate_behavior_dataframe(df)


def test_validate_behavior_dataframe_rejects_gap_in_sequence():
    df = _make_behavior_df([5, 6, 8])
    with pytest.raises(ValueError, match="gap-free consecutive sequence"):
        validate_behavior_dataframe(df)


def test_validate_behavior_dataframe_accepts_consistent_never_started_trial():
    df = _make_behavior_df([1])
    df.loc[0, ["nOutcome", "tGO", "nChoiceMade"]] = [
        OUTCOME_NEVER_STARTED,
        0,
        0,
    ]

    validate_behavior_dataframe(df)


@pytest.mark.parametrize(
    ("column", "value"),
    [("tGO", 500), ("nChoiceMade", 1)],
)
def test_validate_behavior_dataframe_rejects_inconsistent_never_started_trial(
    column,
    value,
):
    df = _make_behavior_df([1])
    df.loc[0, ["nOutcome", "tGO", "nChoiceMade"]] = [
        OUTCOME_NEVER_STARTED,
        0,
        0,
    ]
    df.loc[0, column] = value

    with pytest.raises(ValueError, match="must have tGO == 0 and nChoiceMade == 0"):
        validate_behavior_dataframe(df)


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
