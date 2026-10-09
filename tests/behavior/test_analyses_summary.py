"""Tests for meg_tokens.behavior.analyses.summary."""

import pandas as pd

from meg_tokens.behavior.analyses.summary import summarize_behavior

from tests.behavior.factories import trial_features


def _session(**overrides) -> pd.DataFrame:
    """Two RT trials plus four Fast and four Slow task trials for one subject."""
    columns = {
        "condition": ["RT"] * 2 + ["Fast"] * 4 + ["Slow"] * 4,
        "run": [1] * 2 + [2] * 4 + [3] * 4,
        "run_trial_index": [1, 2] + [1, 2, 3, 4] * 2,
        "rawRT": [340.0, 360.0] + [1250.0, 1450.0, 1350.0, 1300.0]
        + [1650.0, 1850.0, 1750.0, 1700.0],
        "dt_ms": [float("nan")] * 2
        + [900.0, 1100.0, 1000.0, 950.0]
        + [1300.0, 1500.0, 1400.0, 1350.0],
        "trial_class": [0, 0] + [1, 2, 3, 1] * 2,
        "trial_class_name": ["unclassified"] * 2
        + ["easy", "ambiguous", "misleading", "easy"] * 2,
        "isCorrect": [True, True] + [True, True, False, True] * 2,
        "logged_spd": [float("nan")] * 2 + [0.9, 0.7, 0.6, 0.8] * 2,
        "primary_analysis_eligible": [False] * 2 + [True] * 8,
    }
    columns.update(overrides)
    return trial_features(**columns)


def test_trial_counts_separate_started_trials_from_usable_decision_times():
    untimed = _session()
    untimed.loc[2, ["dt_ms", "logged_spd"]] = float("nan")
    untimed.loc[2, "primary_analysis_eligible"] = False

    summary = summarize_behavior(untimed).iloc[0]

    # The Fast run still has four started trials, but only three timed ones.
    assert summary["n_fast_trials"] == 4
    assert summary["n_fast_dt_trials"] == 3
    assert summary["n_slow_trials"] == summary["n_slow_dt_trials"] == 4

    never_started = _session()
    never_started.loc[2, "is_started"] = False
    never_started.loc[2, "primary_analysis_eligible"] = False
    never_started.loc[2, "dt_ms"] = float("nan")

    summary = summarize_behavior(never_started).iloc[0]

    assert summary["n_never_started_trials"] == 1
    assert summary["n_fast_trials"] == 3

    # The RT run contributes baseline trials and no decision times.
    summary = summarize_behavior(_session()).iloc[0]
    assert summary["n_rt_trials"] == 2
    assert summary["n_rt_baseline_trials"] == 2
