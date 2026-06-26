import os
import pytest
import pandas as pd
from meg_tokens.utils.tdms_parser import parse_single_trial, parse_tdms_file

# Sample Mock Events string mimicking the real TDMS structure
MOCK_EVENTS_STR = """sTaskType: 'TokensMvt'
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

def make_mock_events(prob_values):
    steps = [
        f"[tTime: {1000 + i*200}, nTokenNum: {i+1}, nTokenDir: 2, nProb: {p}]"
        for i, p in enumerate(prob_values)
    ]
    lines = MOCK_EVENTS_STR.splitlines()
    idx = lines.index("Tokens.Data: (")
    new_lines = lines[:idx+1] + steps + [")"]
    return "\n".join(new_lines)

def test_parse_single_trial():
    trial_dict = parse_single_trial(MOCK_EVENTS_STR)
    
    assert trial_dict['nTrialIndex'] == 1
    assert trial_dict['nChoiceMade'] == 2
    assert trial_dict['nCorrectChoice'] == 2
    assert trial_dict['tGO'] == 1041
    assert trial_dict['tEnterTarget'] == 2484
    assert trial_dict['tTrialEnd'] == 3676
    assert trial_dict['sTokenDirs'] == '221221122212211'
    assert trial_dict['tTime'] == [1260, 1460]
    assert trial_dict['nProb'] == [0.70947265625000, 0.81234567890123]
    
    # 'x' maps to 0
    assert trial_dict['sTrialClass'] == 0

def test_trial_class_override_rule_1():
    # Test override rule 1: nProb[1] > 0.6 and nProb[4] > 0.75 and nProb[7] > 0.75
    prob_values = [0.5, 0.7, 0.5, 0.5, 0.8, 0.5, 0.5, 0.8]
    events_str = make_mock_events(prob_values)
    trial_dict = parse_single_trial(events_str)
    assert trial_dict['sTrialClass'] == 1

def test_trial_class_override_rule_2():
    # Test override rule 2: nProb[1] == 0.5 and nProb[2] in (0.38, 0.65) and nProb[4] > 0.35
    prob_values = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    events_str = make_mock_events(prob_values)
    trial_dict = parse_single_trial(events_str)
    assert trial_dict['sTrialClass'] == 2

def test_trial_class_override_rule_3():
    # Test override rule 3: nProb[2] < 0.4
    prob_values = [0.5, 0.5, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5]
    events_str = make_mock_events(prob_values)
    trial_dict = parse_single_trial(events_str)
    assert trial_dict['sTrialClass'] == 3

def test_parse_real_tdms_integration():
    real_path = '/media/karim/cc197cfe-12fc-4d55-b0a8-4f52a93ef003/DDM/tdms/H1/H1Slow1_180131.tdms'
    if os.path.exists(real_path):
        df = parse_tdms_file(real_path)
        
        # Verify dataframe structure
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        expected_cols = [
            'nTrialIndex', 'sTrialClass', 'nInitialTime', 'nChoiceMade',
            'nCorrectChoice', 'tGO', 'tEnterTarget', 'tTrialEnd',
            'sTokenDirs', 'tTime', 'nProb'
        ]
        assert list(df.columns) == expected_cols
        
        # TrialData indices should be sequential starting at 1
        assert df['nTrialIndex'].iloc[0] == 1
    else:
        pytest.skip("Real integration TDMS file not accessible.")
