import numpy as np
import pytest
import matplotlib.pyplot as plt
from meg_tokens.behavior.plotting import (
    plot_fast_slow_distributions,
    plot_trial_class_distributions,
    plot_comparison_bars
)

def test_plot_fast_slow_distributions():
    dt_fast = np.random.normal(700, 100, 50)
    dt_slow = np.random.normal(1200, 150, 50)
    
    fig = plot_fast_slow_distributions(dt_fast, dt_slow, title="Test Fast vs Slow")
    
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_trial_class_distributions():
    easy = np.random.normal(600, 80, 30)
    ambig = np.random.normal(800, 100, 30)
    misleading = np.random.normal(1000, 120, 30)
    
    fig = plot_trial_class_distributions(easy, ambig, misleading, title="Test Difficulty")
    
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_comparison_bars():
    raw_a = np.random.normal(1200, 150, 10)
    raw_b = np.random.normal(1400, 180, 10)
    
    fig = plot_comparison_bars(
        mean_a=1200.0, sem_a=50.0,
        mean_b=1400.0, sem_b=60.0,
        raw_a=raw_a, raw_b=raw_b,
        label_a="Correct", label_b="Error",
        title="Test Correct vs Error"
    )
    
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
