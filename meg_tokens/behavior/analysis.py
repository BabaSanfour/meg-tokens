import numpy as np
import pandas as pd
from scipy import stats


def calculate_motor_baseline(rt_dfs: list) -> float:
    """
    Calculates the baseline motor response time (reaction time) from RT runs.
    Baseline RT = Mean(tEnterTarget - tGO) across all RT trials.
    """
    rt_values = []
    for df in rt_dfs:
        if df.empty:
            continue
        # Drop trials where choice was skipped (nChoiceMade == 0)
        valid_df = df[df['nChoiceMade'] > 0]
        rt_values.extend((valid_df['tEnterTarget'] - valid_df['tGO']).values)
        
    return float(np.mean(rt_values)) if rt_values else 0.0

def calculate_decision_times(df: pd.DataFrame, motor_baseline: float) -> pd.Series:
    """
    Computes decision times (DT) for each valid trial in a dataframe.
    DT = (tEnterTarget - tGO) - motor_baseline
    """
    # Drop trials where choice was skipped (nChoiceMade == 0)
    valid_mask = df['nChoiceMade'] > 0
    dt = (df.loc[valid_mask, 'tEnterTarget'] - df.loc[valid_mask, 'tGO']) - motor_baseline
    return dt

def compare_fast_slow(
    fast_dfs: list,
    slow_dfs: list,
    motor_baseline: float,
    n_permutations: int = 1000
) -> dict:
    """
    Compares decision times between Fast and Slow runs using Welch's t-test.

    The legacy scripts downsampled groups before testing. The refactor keeps all
    real trials and uses a standard unequal-variance test to avoid random trial
    selection.
    """
    dt_fast = []
    for df in fast_dfs:
        dt_fast.extend(calculate_decision_times(df, motor_baseline).values)
        
    dt_slow = []
    for df in slow_dfs:
        dt_slow.extend(calculate_decision_times(df, motor_baseline).values)
        
    dt_fast = np.array(dt_fast)
    dt_slow = np.array(dt_slow)
    
    p_val = 1.0
    t_stat = 0.0
    if len(dt_fast) > 0 and len(dt_slow) > 0:
        if len(dt_fast) <= 1 or len(dt_slow) <= 1:
            t_stat = 0.0
            p_val = 1.0
        else:
            res = stats.ttest_ind(dt_slow, dt_fast, equal_var=False, nan_policy="omit")
            t_stat = float(res.statistic)
            p_val = float(res.pvalue)
        
    return {
        'mean_fast': float(np.mean(dt_fast)) if len(dt_fast) > 0 else 0.0,
        'mean_slow': float(np.mean(dt_slow)) if len(dt_slow) > 0 else 0.0,
        't_stat': t_stat,
        'p_value': p_val
    }

def analyze_trial_classes(dfs: list, motor_baseline: float) -> dict:
    """
    Groups decision times by trial complexity class:
    1 = Easy, 2 = Ambiguous, 3 = Misleading.
    """
    classes_dt = {1: [], 2: [], 3: []}
    for df in dfs:
        if df.empty:
            continue
        valid_df = df[df['nChoiceMade'] > 0].copy()
        valid_df['DT'] = (valid_df['tEnterTarget'] - valid_df['tGO']) - motor_baseline
        
        for c in [1, 2, 3]:
            class_trials = valid_df[valid_df['sTrialClass'] == c]
            classes_dt[c].extend(class_trials['DT'].values)
            
    return {
        'easy': np.array(classes_dt[1]),
        'ambiguous': np.array(classes_dt[2]),
        'misleading': np.array(classes_dt[3]),
        'means': {
            'easy': float(np.mean(classes_dt[1])) if classes_dt[1] else 0.0,
            'ambiguous': float(np.mean(classes_dt[2])) if classes_dt[2] else 0.0,
            'misleading': float(np.mean(classes_dt[3])) if classes_dt[3] else 0.0
        }
    }

def compare_correct_error(dfs: list, motor_baseline: float, n_permutations: int = 1000) -> dict:
    """
    Compares decision times on Correct choices vs. Error trials.

    Run-level means are compared with Welch's t-test when enough runs are
    available. This avoids random run subsampling while preserving the intended
    correct-versus-error behavior.
    """
    dt_correct = []
    dt_error = []
    
    # Calculate subject run-level averages to replicate legacy grouping
    run_correct_means = []
    run_error_means = []
    
    for df in dfs:
        if df.empty:
            continue
        valid_df = df[df['nChoiceMade'] > 0].copy()
        valid_df['DT'] = (valid_df['tEnterTarget'] - valid_df['tGO']) - motor_baseline
        
        correct_trials = valid_df[valid_df['nChoiceMade'] == valid_df['nCorrectChoice']]
        error_trials = valid_df[valid_df['nChoiceMade'] != valid_df['nCorrectChoice']]
        
        if not correct_trials.empty:
            dt_correct.extend(correct_trials['DT'].values)
            run_correct_means.append(float(np.mean(correct_trials['DT'])))
        if not error_trials.empty:
            dt_error.extend(error_trials['DT'].values)
            run_error_means.append(float(np.mean(error_trials['DT'])))
            
    p_val = 1.0
    t_stat = 0.0
    if len(run_correct_means) > 0 and len(run_error_means) > 0:
        if len(run_correct_means) <= 1 or len(run_error_means) <= 1:
            t_stat = 0.0
            p_val = 1.0
        else:
            res = stats.ttest_ind(run_correct_means, run_error_means, equal_var=False, nan_policy="omit")
            t_stat = float(res.statistic)
            p_val = float(res.pvalue)
        
    return {
        'mean_correct': float(np.mean(dt_correct)) if dt_correct else 0.0,
        'mean_error': float(np.mean(dt_error)) if dt_error else 0.0,
        'percent_correct': float(len(dt_correct)) / (len(dt_correct) + len(dt_error)) * 100 if (dt_correct or dt_error) else 0.0,
        't_stat': t_stat,
        'p_value': p_val
    }

def analyze_post_error_slowing(dfs: list, motor_baseline: float) -> dict:
    """
    Analyzes post-error slowing: compares decision times on trials immediately
    following a correct trial (post-correct) vs. immediately following an error trial (post-error).
    """
    post_correct_dts = []
    post_error_dts = []
    
    run_pc_means = []
    run_pe_means = []
    
    for df in dfs:
        if df.empty or len(df) < 2:
            continue
            
        pc_temp = []
        pe_temp = []
        
        # Iterate over trials (excluding the last trial of the run)
        for i in range(len(df) - 1):
            made_current = df.iloc[i]['nChoiceMade']
            correct_current = df.iloc[i]['nCorrectChoice']
            
            # Skip current trial if it was skipped
            if made_current == 0:
                continue
                
            # Next trial details
            made_next = df.iloc[i + 1]['nChoiceMade']
            if made_next == 0:
                continue  # Skip next trial if it was skipped
                
            t_enter_next = df.iloc[i + 1]['tEnterTarget']
            t_go_next = df.iloc[i + 1]['tGO']
            dt_next = (t_enter_next - t_go_next) - motor_baseline
            
            if made_current == correct_current:
                pc_temp.append(dt_next)
                post_correct_dts.append(dt_next)
            else:
                pe_temp.append(dt_next)
                post_error_dts.append(dt_next)
                
        if pc_temp:
            run_pc_means.append(float(np.mean(pc_temp)))
        if pe_temp:
            run_pe_means.append(float(np.mean(pe_temp)))
            
    # Paired t-test comparing post-correct vs post-error run averages
    p_val = 1.0
    t_stat = 0.0
    if len(run_pc_means) == len(run_pe_means) and len(run_pc_means) > 1:
        res = stats.ttest_rel(run_pc_means, run_pe_means)
        t_stat = float(res.statistic)
        p_val = float(res.pvalue)
        
    return {
        'mean_post_correct': float(np.mean(post_correct_dts)) if post_correct_dts else 0.0,
        'mean_post_error': float(np.mean(post_error_dts)) if post_error_dts else 0.0,
        't_stat': t_stat,
        'p_value': p_val
    }
