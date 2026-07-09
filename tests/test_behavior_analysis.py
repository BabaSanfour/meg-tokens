import numpy as np
import pandas as pd
import pytest
from meg_tokens.behavior.metrics import (
    calculate_motor_baseline,
    calculate_decision_times,
    compare_fast_slow,
    analyze_trial_classes,
    compare_correct_error,
    analyze_post_error_slowing
)

@pytest.fixture
def sample_behavior_data():
    # Mock behavioral dataframes for testing
    rt_df1 = pd.DataFrame({
        'nChoiceMade': [1, 2, 0, 1], # 0 is skipped
        'tGO': [1000, 1000, 1000, 1000],
        'tEnterTarget': [1300, 1400, 1500, 1350] # Raw RTs: 300, 400, (skipped), 350
    })
    
    rt_df2 = pd.DataFrame({
        'nChoiceMade': [1, 2],
        'tGO': [1000, 1000],
        'tEnterTarget': [1320, 1430] # Raw RTs: 320, 430
    })
    
    # Fast condition run (subject decision-making is fast)
    fast_df = pd.DataFrame({
        'nTrialIndex': [1, 2, 3, 4],
        'sTrialClass': [1, 2, 3, 1],
        'nChoiceMade': [1, 2, 1, 2],
        'nCorrectChoice': [1, 2, 2, 2], # Trial 3 is error
        'tGO': [1000, 1000, 1000, 1000],
        'tEnterTarget': [2000, 2200, 2100, 2050] # Raw RTs: 1000, 1200, 1100, 1050
    })
    
    # Slow condition run
    slow_df = pd.DataFrame({
        'nTrialIndex': [1, 2, 3, 4],
        'sTrialClass': [1, 2, 3, 1],
        'nChoiceMade': [1, 2, 1, 1], # Trial 4 is error
        'nCorrectChoice': [1, 2, 1, 2],
        'tGO': [1000, 1000, 1000, 1000],
        'tEnterTarget': [2500, 2800, 2600, 2700] # Raw RTs: 1500, 1800, 1600, 1700
    })
    
    return [rt_df1, rt_df2], fast_df, slow_df

def test_calculate_motor_baseline(sample_behavior_data):
    rt_dfs, _, _ = sample_behavior_data
    baseline = calculate_motor_baseline(rt_dfs)
    
    # Valid trials in rt_df1: (1300-1000)=300, (1400-1000)=400, (1350-1000)=350. Mean = 350
    # Valid trials in rt_df2: (1320-1000)=320, (1430-1000)=430. Mean = 375
    # Total mean of [300, 400, 350, 320, 430] = 360
    assert pytest.approx(baseline) == 360.0

def test_calculate_decision_times(sample_behavior_data):
    _, fast_df, _ = sample_behavior_data
    baseline = 360.0
    
    dt = calculate_decision_times(fast_df, baseline)
    # Raw RTs: 1000, 1200, 1100, 1050
    # DTs: 640, 840, 740, 690
    assert list(dt) == [640.0, 840.0, 740.0, 690.0]

def test_compare_fast_slow(sample_behavior_data):
    _, fast_df, slow_df = sample_behavior_data
    baseline = 360.0
    
    res = compare_fast_slow([fast_df], [slow_df], baseline, n_permutations=100)
    
    # Fast DT mean = (640+840+740+690)/4 = 727.5
    # Slow DT mean = ((1500-360)+(1800-360)+(1600-360)+(1700-360))/4 = (1140+1440+1240+1340)/4 = 1290.0
    assert pytest.approx(res['mean_fast']) == 727.5
    assert pytest.approx(res['mean_slow']) == 1290.0
    assert 't_stat' in res
    assert 'p_value' in res

def test_analyze_trial_classes(sample_behavior_data):
    _, fast_df, _ = sample_behavior_data
    baseline = 360.0
    
    res = analyze_trial_classes([fast_df], baseline)
    # Trial 1 (Class 1, DT=640)
    # Trial 2 (Class 2, DT=840)
    # Trial 3 (Class 3, DT=740)
    # Trial 4 (Class 1, DT=690)
    assert pytest.approx(res['means']['easy']) == 665.0 # (640+690)/2
    assert pytest.approx(res['means']['ambiguous']) == 840.0
    assert pytest.approx(res['means']['misleading']) == 740.0

def test_compare_correct_error(sample_behavior_data):
    _, fast_df, _ = sample_behavior_data
    baseline = 360.0
    
    res = compare_correct_error([fast_df], baseline, n_permutations=100)
    
    # Correct choices: trial 1, 2, 4 (DTs: 640, 840, 690). Mean = 723.33
    # Error choices: trial 3 (DT: 740). Mean = 740.0
    # Accuracy: 3 / 4 * 100 = 75.0%
    assert pytest.approx(res['mean_correct']) == 723.3333333333334
    assert pytest.approx(res['mean_error']) == 740.0
    assert pytest.approx(res['percent_correct']) == 75.0

def test_analyze_post_error_slowing(sample_behavior_data):
    _, fast_df, _ = sample_behavior_data
    baseline = 360.0
    
    res = analyze_post_error_slowing([fast_df], baseline)
    
    # Fast trials:
    # 0: Correct (made=1, correct=1), DT next = 2200 - 1000 - 360 = 840 (Post-Correct)
    # 1: Correct (made=2, correct=2), DT next = 2100 - 1000 - 360 = 740 (Post-Correct)
    # 2: Error (made=1, correct=2), DT next = 2050 - 1000 - 360 = 690 (Post-Error)
    # 3: Correct (last, skipped next)
    assert pytest.approx(res['mean_post_correct']) == 790.0 # (840+740)/2
    assert pytest.approx(res['mean_post_error']) == 690.0
