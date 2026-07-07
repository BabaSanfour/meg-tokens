import numpy as np
import pytest
import matplotlib.pyplot as plt
from meg_tokens.behavior.plotting import (
    plot_fast_slow_distributions,
    plot_trial_class_distributions,
    plot_comparison_bars
)

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
