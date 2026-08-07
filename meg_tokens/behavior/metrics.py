"""Behavioral metrics for the Tokens task."""

from ast import literal_eval

import numpy as np
import pandas as pd
from scipy import stats

from meg_tokens.behavior.success_probability import probability_at_decision


def calculate_motor_baseline(rt_dfs: list) -> float:
    """
    Calculates the baseline motor response time (reaction time) from RT runs.
    Baseline RT = mean canonical rawRT across all chosen RT trials.
    """
    rt_values = []
    for df in rt_dfs:
        if df.empty:
            continue
        rt_values.extend(pd.to_numeric(df['rawRT']).dropna().values)

    if not rt_values:
        raise ValueError("No valid RT trials are available for the motor baseline")
    return float(np.mean(rt_values))

def calculate_decision_times(df: pd.DataFrame, motor_baseline: float) -> pd.Series:
    """
    Computes decision times (DT) for each valid trial in a dataframe.
    DT = rawRT - motor_baseline.
    """
    raw_rt = pd.to_numeric(df['rawRT'])
    return raw_rt.loc[raw_rt.notna()] - motor_baseline


def paired_subject_statistics(
    summary: pd.DataFrame,
    column_a: str,
    column_b: str,
) -> dict:
    """Summarize a paired subject-level contrast with Cohen's dz."""
    pairs = summary[[column_a, column_b]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    values_a = pairs[column_a].to_numpy(dtype=float)
    values_b = pairs[column_b].to_numpy(dtype=float)
    n_subjects = len(pairs)

    def mean_sem(values: np.ndarray) -> tuple[float, float]:
        if not len(values):
            return np.nan, np.nan
        mean = float(np.mean(values))
        sem = (
            float(np.std(values, ddof=1) / np.sqrt(len(values)))
            if len(values) > 1 else np.nan
        )
        return mean, sem

    mean_a, sem_a = mean_sem(values_a)
    mean_b, sem_b = mean_sem(values_b)
    t_stat = p_value = effect_size = np.nan
    if n_subjects > 1:
        result = stats.ttest_rel(values_a, values_b)
        t_stat = float(result.statistic)
        p_value = float(result.pvalue)
        differences = values_a - values_b
        difference_sd = float(np.std(differences, ddof=1))
        if difference_sd > 0:
            effect_size = float(np.mean(differences) / difference_sd)

    return {
        "n_subjects": n_subjects,
        "mean_a": mean_a,
        "sem_a": sem_a,
        "mean_b": mean_b,
        "sem_b": sem_b,
        "mean_difference": mean_a - mean_b,
        "t": t_stat,
        "p": p_value,
        "df": float(n_subjects - 1) if n_subjects > 1 else np.nan,
        "cohens_dz": effect_size,
    }


def behavior_group_statistics(summary: pd.DataFrame) -> pd.DataFrame:
    """Compute the planned paired group contrasts from subject summaries."""
    class_pairs = (
        ("easy", "ambiguous"),
        ("easy", "misleading"),
        ("ambiguous", "misleading"),
    )
    contrasts = [
        {
            "analysis": "decision_time",
            "contrast": "fast_vs_slow",
            "view": "primary",
            "unit": "ms",
            "label_a": "Fast",
            "label_b": "Slow",
            "column_a": "mean_fast_dt_ms",
            "column_b": "mean_slow_dt_ms",
        },
        {
            "analysis": "error_count",
            "contrast": "fast_vs_slow",
            "view": "primary",
            "unit": "trials",
            "label_a": "Fast",
            "label_b": "Slow",
            "column_a": "n_fast_error_trials",
            "column_b": "n_slow_error_trials",
        },
    ]
    for first, second in class_pairs:
        contrasts.append({
            "analysis": "decision_time",
            "contrast": f"{first}_vs_{second}",
            "view": "primary",
            "unit": "ms",
            "label_a": first.capitalize(),
            "label_b": second.capitalize(),
            "column_a": f"mean_{first}_dt_ms",
            "column_b": f"mean_{second}_dt_ms",
        })
        for view in ("all_logged", "validated_15row"):
            contrasts.append({
                "analysis": "spd",
                "contrast": f"{first}_vs_{second}",
                "view": view,
                "unit": "probability",
                "label_a": first.capitalize(),
                "label_b": second.capitalize(),
                "column_a": f"mean_{first}_spd_{view}",
                "column_b": f"mean_{second}_spd_{view}",
            })

    rows = []
    for contrast in contrasts:
        result = paired_subject_statistics(
            summary,
            contrast["column_a"],
            contrast["column_b"],
        )
        rows.append({**contrast, **result})
    return pd.DataFrame(rows)


def _float_sequence(value: object, *, field: str) -> list[float]:
    parsed = literal_eval(value) if isinstance(value, str) else value
    if isinstance(parsed, np.ndarray):
        parsed = parsed.tolist()
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"{field} must contain a sequence")
    return [float(item) for item in parsed]


def _boolean_value(value: object, *, field: str) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
        raise ValueError(f"{field} must be boolean")
    return bool(value)


def logged_spd_at_decision(
    trial: pd.Series,
    motor_baseline: float,
) -> tuple[float, int | None]:
    """Return logged chosen-target SPD and its zero-based token index."""
    probabilities = _float_sequence(trial['nProb'], field='nProb')
    token_times = _float_sequence(trial['tTime'], field='tTime')
    log_rows = int(trial['token_log_rows'])
    log_short = _boolean_value(
        trial['token_log_short'],
        field='token_log_short',
    )
    if len(probabilities) != log_rows or len(token_times) != log_rows:
        raise ValueError(
            "nProb, tTime, and token_log_rows must describe the same rows"
        )
    if log_short != (log_rows == 14):
        raise ValueError("token_log_short must be true exactly for 14-row logs")
    return probability_at_decision(
        probabilities,
        token_times,
        decision_time=float(trial['tEnterTarget']) - float(motor_baseline),
    )


def analyze_logged_spd(dfs: list[pd.DataFrame], motor_baseline: float) -> dict:
    """Summarize logged chosen-target SPD for all logs and 15-row sensitivity."""
    records: list[tuple[int, float, int]] = []
    for df in dfs:
        if df.empty:
            continue
        for _, trial in df.loc[df['nChoiceMade'] > 0].iterrows():
            log_rows = int(trial['token_log_rows'])
            spd, _ = logged_spd_at_decision(trial, motor_baseline)
            records.append((int(trial['sTrialClass']), spd, log_rows))

    class_names = {1: 'easy', 2: 'ambiguous', 3: 'misleading'}

    def summarize(selected: list[tuple[int, float, int]]) -> dict:
        values = [spd for _, spd, _ in selected]
        classes = {}
        for trial_class, name in class_names.items():
            class_values = [
                spd for observed_class, spd, _ in selected
                if observed_class == trial_class
            ]
            classes[name] = {
                'n_trials': len(class_values),
                'mean_spd': float(np.mean(class_values)) if class_values else np.nan,
            }
        return {
            'n_trials': len(values),
            'mean_spd': float(np.mean(values)) if values else np.nan,
            'classes': classes,
        }

    return {
        'all_logged': summarize(records),
        'validated_15row': summarize(
            [record for record in records if record[2] == 15]
        ),
    }

def compare_fast_slow(
    fast_dfs: list,
    slow_dfs: list,
    motor_baseline: float,
) -> dict:
    """Summarize decision times in Fast and Slow runs."""
    dt_fast = []
    for df in fast_dfs:
        dt_fast.extend(calculate_decision_times(df, motor_baseline).values)
        
    dt_slow = []
    for df in slow_dfs:
        dt_slow.extend(calculate_decision_times(df, motor_baseline).values)
        
    dt_fast = np.array(dt_fast)
    dt_slow = np.array(dt_slow)
    
    return {
        'n_fast': len(dt_fast),
        'n_slow': len(dt_slow),
        'n_fast_anticipations': int((dt_fast < 0).sum()),
        'n_slow_anticipations': int((dt_slow < 0).sum()),
        'mean_fast': float(np.mean(dt_fast)) if len(dt_fast) > 0 else np.nan,
        'mean_slow': float(np.mean(dt_slow)) if len(dt_slow) > 0 else np.nan,
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
        valid_df = df.loc[df['rawRT'].notna()].copy()
        valid_df['DT'] = pd.to_numeric(valid_df['rawRT']) - motor_baseline
        
        for c in [1, 2, 3]:
            class_trials = valid_df[valid_df['sTrialClass'] == c]
            classes_dt[c].extend(class_trials['DT'].values)
            
    return {
        'easy': np.array(classes_dt[1]),
        'ambiguous': np.array(classes_dt[2]),
        'misleading': np.array(classes_dt[3]),
        'counts': {
            'easy': len(classes_dt[1]),
            'ambiguous': len(classes_dt[2]),
            'misleading': len(classes_dt[3]),
        },
        'means': {
            'easy': float(np.mean(classes_dt[1])) if classes_dt[1] else np.nan,
            'ambiguous': float(np.mean(classes_dt[2])) if classes_dt[2] else np.nan,
            'misleading': float(np.mean(classes_dt[3])) if classes_dt[3] else np.nan,
        }
    }

def compare_correct_error(dfs: list, motor_baseline: float) -> dict:
    """Summarize decision times and counts for correct and error trials."""
    dt_correct = []
    dt_error = []
    
    for df in dfs:
        if df.empty:
            continue
        valid_df = df.loc[df['rawRT'].notna() & df['isCorrect'].notna()].copy()
        valid_df['DT'] = pd.to_numeric(valid_df['rawRT']) - motor_baseline
        correctness = valid_df['isCorrect'].map(
            lambda value: _boolean_value(value, field='isCorrect')
        )
        
        correct_trials = valid_df.loc[correctness]
        error_trials = valid_df.loc[~correctness]
        
        if not correct_trials.empty:
            dt_correct.extend(correct_trials['DT'].values)
        if not error_trials.empty:
            dt_error.extend(error_trials['DT'].values)

    n_responses = len(dt_correct) + len(dt_error)
        
    return {
        'n_correct': len(dt_correct),
        'n_error': len(dt_error),
        'mean_correct': float(np.mean(dt_correct)) if dt_correct else np.nan,
        'mean_error': float(np.mean(dt_error)) if dt_error else np.nan,
        'percent_correct': float(len(dt_correct)) / n_responses * 100 if n_responses else np.nan,
    }

def analyze_post_error_slowing(dfs: list, motor_baseline: float) -> dict:
    """Summarize DT after correct and error trials."""
    post_correct_dts = []
    post_error_dts = []
    
    for df in dfs:
        if df.empty or len(df) < 2:
            continue

        # Iterate over trials (excluding the last trial of the run)
        for i in range(len(df) - 1):
            is_correct = df.iloc[i]['isCorrect']
            if pd.isna(is_correct):
                continue
                
            raw_rt_next = df.iloc[i + 1]['rawRT']
            if pd.isna(raw_rt_next):
                continue
            dt_next = float(raw_rt_next) - motor_baseline
            
            if _boolean_value(is_correct, field='isCorrect'):
                post_correct_dts.append(dt_next)
            else:
                post_error_dts.append(dt_next)
                
    return {
        'n_post_correct': len(post_correct_dts),
        'n_post_error': len(post_error_dts),
        'mean_post_correct': float(np.mean(post_correct_dts)) if post_correct_dts else np.nan,
        'mean_post_error': float(np.mean(post_error_dts)) if post_error_dts else np.nan,
    }
