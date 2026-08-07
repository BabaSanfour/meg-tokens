import numpy as np
import pandas as pd
import pytest
from meg_tokens.behavior.analyses.performance import (
    compare_fast_slow,
    analyze_trial_classes,
    compare_correct_error,
    analyze_post_error_slowing,
)
from meg_tokens.behavior.features import (
    calculate_decision_times,
    calculate_motor_baseline,
)
from meg_tokens.behavior.math.inference import paired_subject_statistics

@pytest.fixture
def sample_behavior_data():
    # Mock behavioral dataframes for testing
    rt_df1 = pd.DataFrame({
        'nChoiceMade': [1, 2, 0, 1], # 0 is skipped
        'tGO': [1000, 1000, 1000, 1000],
        'tEnterTarget': [1300, 1400, 1500, 1350],
        'rawRT': [300, 400, None, 350],
        'isCorrect': [True, True, None, True],
    })
    
    rt_df2 = pd.DataFrame({
        'nChoiceMade': [1, 2],
        'tGO': [1000, 1000],
        'tEnterTarget': [1320, 1430],
        'rawRT': [320, 430],
        'isCorrect': [True, True],
    })
    
    # Fast condition run (subject decision-making is fast)
    fast_df = pd.DataFrame({
        'nTrialIndex': [1, 2, 3, 4],
        'sTrialClass': [1, 2, 3, 1],
        'nChoiceMade': [1, 2, 1, 2],
        'nCorrectChoice': [1, 2, 2, 2], # Trial 3 is error
        'tGO': [1000, 1000, 1000, 1000],
        'tEnterTarget': [2000, 2200, 2100, 2050],
        'rawRT': [1000, 1200, 1100, 1050],
        'isCorrect': [True, True, False, True],
    })
    
    # Slow condition run
    slow_df = pd.DataFrame({
        'nTrialIndex': [1, 2, 3, 4],
        'sTrialClass': [1, 2, 3, 1],
        'nChoiceMade': [1, 2, 1, 1], # Trial 4 is error
        'nCorrectChoice': [1, 2, 1, 2],
        'tGO': [1000, 1000, 1000, 1000],
        'tEnterTarget': [2500, 2800, 2600, 2700],
        'rawRT': [1500, 1800, 1600, 1700],
        'isCorrect': [True, True, True, False],
    })
    
    return [rt_df1, rt_df2], fast_df, slow_df


def _task_features(fast_df, slow_df=None, *, baseline=360.0):
    tables = [("Fast", 1, fast_df)]
    if slow_df is not None:
        tables.append(("Slow", 2, slow_df))
    rows = []
    for condition, run, table in tables:
        for index, trial in table.reset_index(drop=True).iterrows():
            rows.append({
                "subject": "H01",
                "condition": condition,
                "run": run,
                "run_trial_index": index + 1,
                "primary_analysis_eligible": pd.notna(trial["rawRT"]),
                "is_started": True,
                "dt_ms": (
                    float(trial["rawRT"]) - baseline
                    if pd.notna(trial["rawRT"])
                    else np.nan
                ),
                "trial_class": int(trial["sTrialClass"]),
                "isCorrect": trial["isCorrect"],
            })
    return pd.DataFrame(rows)

def test_calculate_motor_baseline(sample_behavior_data):
    rt_dfs, _, _ = sample_behavior_data
    baseline = calculate_motor_baseline(rt_dfs)
    
    # Valid trials in rt_df1: (1300-1000)=300, (1400-1000)=400, (1350-1000)=350. Mean = 350
    # Valid trials in rt_df2: (1320-1000)=320, (1430-1000)=430. Mean = 375
    # Total mean of [300, 400, 350, 320, 430] = 360
    assert pytest.approx(baseline) == 360.0


def test_empty_performance_summaries_are_explicit():
    with pytest.raises(ValueError, match="No valid RT trials"):
        calculate_motor_baseline([])

    empty = pd.DataFrame(columns=[
        "subject", "condition", "run", "run_trial_index",
        "primary_analysis_eligible", "is_started", "dt_ms",
        "trial_class", "isCorrect",
    ])
    speed = compare_fast_slow(empty)
    accuracy = compare_correct_error(empty)
    classes = analyze_trial_classes(empty)
    post_error = analyze_post_error_slowing(empty)

    assert (speed["n_fast"], speed["n_slow"]) == (0, 0)
    assert (accuracy["n_correct"], accuracy["n_error"]) == (0, 0)
    assert classes["counts"] == {"easy": 0, "ambiguous": 0, "misleading": 0}
    assert (post_error["n_post_correct"], post_error["n_post_error"]) == (0, 0)
    assert all(
        np.isnan(value)
        for value in (
            speed["mean_fast"], speed["mean_slow"],
            accuracy["mean_correct"], accuracy["mean_error"],
            accuracy["percent_correct"],
            *classes["means"].values(),
            post_error["mean_post_correct"], post_error["mean_post_error"],
        )
    )


def test_paired_subject_statistics_reports_requested_values():
    summary = pd.DataFrame({
        "first": [1.0, 2.0, 4.0],
        "second": [2.0, 4.0, 8.0],
    })

    result = paired_subject_statistics(summary["first"], summary["second"])

    assert result["n_subjects"] == 3
    assert result["df"] == 2
    assert result["mean_a"] == pytest.approx(7 / 3)
    assert result["mean_b"] == pytest.approx(14 / 3)
    assert result["sem_a"] == pytest.approx(np.std([1, 2, 4], ddof=1) / np.sqrt(3))
    assert result["t"] == pytest.approx(result["cohens_dz"] * np.sqrt(3))
    assert 0 <= result["p"] <= 1

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
    
    res = compare_fast_slow(_task_features(fast_df, slow_df, baseline=baseline))
    
    # Fast DT mean = (640+840+740+690)/4 = 727.5
    # Slow DT mean = ((1500-360)+(1800-360)+(1600-360)+(1700-360))/4 = (1140+1440+1240+1340)/4 = 1290.0
    assert pytest.approx(res['mean_fast']) == 727.5
    assert pytest.approx(res['mean_slow']) == 1290.0
    assert res['n_fast_anticipations'] == 0
    assert res['n_slow_anticipations'] == 0

def test_analyze_trial_classes(sample_behavior_data):
    _, fast_df, _ = sample_behavior_data
    baseline = 360.0
    
    res = analyze_trial_classes(_task_features(fast_df, baseline=baseline))
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
    
    res = compare_correct_error(_task_features(fast_df, baseline=baseline))
    
    # Correct choices: trial 1, 2, 4 (DTs: 640, 840, 690). Mean = 723.33
    # Error choices: trial 3 (DT: 740). Mean = 740.0
    # Accuracy: 3 / 4 * 100 = 75.0%
    assert pytest.approx(res['mean_correct']) == 723.3333333333334
    assert pytest.approx(res['mean_error']) == 740.0
    assert pytest.approx(res['percent_correct']) == 75.0

def test_analyze_post_error_slowing(sample_behavior_data):
    _, fast_df, _ = sample_behavior_data
    baseline = 360.0
    
    res = analyze_post_error_slowing(_task_features(fast_df, baseline=baseline))
    
    # Fast trials:
    # 0: Correct (made=1, correct=1), DT next = 2200 - 1000 - 360 = 840 (Post-Correct)
    # 1: Correct (made=2, correct=2), DT next = 2100 - 1000 - 360 = 740 (Post-Correct)
    # 2: Error (made=1, correct=2), DT next = 2050 - 1000 - 360 = 690 (Post-Error)
    # 3: Correct (last, skipped next)
    assert pytest.approx(res['mean_post_correct']) == 790.0 # (840+740)/2
    assert pytest.approx(res['mean_post_error']) == 690.0


def test_performance_uses_canonical_rt_and_correctness_fields():
    df = pd.DataFrame({
        'sTrialClass': [1, 2, 3],
        'rawRT': [100.0, 200.0, 300.0],
        'isCorrect': [True, False, True],
        # Deliberately conflict with the canonical fields.
        'nChoiceMade': [1, 1, 1],
        'nCorrectChoice': [2, 2, 2],
        'tGO': [0.0, 0.0, 0.0],
        'tEnterTarget': [900.0, 900.0, 900.0],
    })

    assert calculate_motor_baseline([df]) == pytest.approx(200.0)
    assert list(calculate_decision_times(df, 50.0)) == [50.0, 150.0, 250.0]
    features = _task_features(df, baseline=0.0)
    assert analyze_trial_classes(features)['means'] == {
        'easy': 100.0,
        'ambiguous': 200.0,
        'misleading': 300.0,
    }

    correct_error = compare_correct_error(features)
    assert correct_error['mean_correct'] == pytest.approx(200.0)
    assert correct_error['mean_error'] == pytest.approx(200.0)
    assert correct_error['percent_correct'] == pytest.approx(2 / 3 * 100)

    post_error = analyze_post_error_slowing(features)
    assert post_error['mean_post_correct'] == pytest.approx(200.0)
    assert post_error['mean_post_error'] == pytest.approx(300.0)
