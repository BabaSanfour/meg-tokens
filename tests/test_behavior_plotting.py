import numpy as np
import pytest
import matplotlib.pyplot as plt
import pandas as pd

from meg_tokens.reports.behavior import (
    plot_fast_slow_distributions,
    plot_trial_class_distributions,
    plot_comparison_bars
)
from meg_tokens.io import DerivativeLayout, save_table, sidecar_path
from meg_tokens.reports.behavior_summary import run_behavior_plotting

def test_plot_fast_slow_distributions():
    dt_fast = np.linspace(600, 800, 50)
    dt_slow = np.linspace(1000, 1400, 50)
    
    fig = plot_fast_slow_distributions(dt_fast, dt_slow, title="Test Fast vs Slow")
    
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_trial_class_distributions():
    easy = np.linspace(520, 680, 30)
    ambig = np.linspace(700, 900, 30)
    misleading = np.linspace(880, 1120, 30)
    
    fig = plot_trial_class_distributions(easy, ambig, misleading, title="Test Difficulty")
    
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_comparison_bars():
    raw_a = np.linspace(1050, 1350, 10)
    raw_b = np.linspace(1220, 1580, 10)
    
    fig = plot_comparison_bars(
        mean_a=1200.0, sem_a=50.0,
        mean_b=1400.0, sem_b=60.0,
        raw_a=raw_a, raw_b=raw_b,
        label_a="Correct", label_b="Error",
        title="Test Correct vs Error"
    )
    
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_behavior_report_reads_staged_tsv_and_writes_derivative(tmp_path):
    layout = DerivativeLayout(tmp_path)
    for condition, values in (("Fast", [500.0, 600.0]), ("Slow", [900.0, 1000.0])):
        path = layout.behavior(subject="H01", run=f"{condition}1", condition=condition)
        n_trials = len(values)
        save_table(
            path,
            pd.DataFrame({
                "subject": ["H01"] * n_trials,
                "condition": [condition] * n_trials,
                "run": [1] * n_trials,
                "source_file": [f"H01{condition}1_180131.tdms"] * n_trials,
                "nTrialIndex": list(range(1, n_trials + 1)),
                "sTrialClass": [1] * n_trials,
                "sTrialClassRaw": ["e"] * n_trials,
                "trial_class_source": ["design"] * n_trials,
                "trial_class_rule": ["recorded_label"] * n_trials,
                "sp_design_correct": [[0.6]] * n_trials,
                "nInitialTime": list(range(n_trials)),
                "nChoiceMade": [1] * n_trials,
                "nCorrectChoice": [1] * n_trials,
                "tGO": [1000.0] * n_trials,
                "tEnterCenter": [0.0] * n_trials,
                "tExitCenter": [1000.0 + value for value in values],
                "tEnterTarget": [1000.0 + value for value in values],
                "tTrialEnd": [1200.0 + value for value in values],
                "sTokenDirs": ["1"] * n_trials,
                "nTokenNum": [[1]] * n_trials,
                "nTokenDir": [[1]] * n_trials,
                "tTime": [[1100.0]] * n_trials,
                "nProb": [[0.6]] * n_trials,
                "token_log_rows": [1] * n_trials,
                "token_log_short": [False] * n_trials,
                "nOutcome": [0] * n_trials,
                "rawRT": values,
                "isCorrect": [True] * n_trials,
            }),
            metadata={"subject": "H01", "condition": condition},
        )

    outputs = run_behavior_plotting(str(tmp_path), str(tmp_path), ["H01"])

    assert len(outputs) == 1
    assert outputs[0].is_file()
    assert sidecar_path(outputs[0]).is_file()
