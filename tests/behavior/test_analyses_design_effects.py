"""Tests for meg_tokens.behavior.analyses.design_effects.

One test per behaviour, not per assertion. The fixtures are the expensive part
— a counterbalanced twenty-subject session takes as long to build as the whole
analysis takes to run — so every claim a fixture can support is asserted on it.
"""

import numpy as np
import pandas as pd
import pytest

from meg_tokens.behavior.analyses.design_effects import (
    condition_class_contrasts,
    condition_class_statistics,
    time_on_task,
    within_block_decile_profile,
)
from meg_tokens.behavior.trials import CLASS_NAMES

from tests.behavior.factories import trial_features


def _interleaved_session(subject="H01", drift=0.0):
    """Fast and Slow blocks alternating on the session clock."""
    return trial_features(
        subject=[subject] * 8,
        condition=["Fast"] * 2 + ["Slow"] * 2 + ["Fast"] * 2 + ["Slow"] * 2,
        run=[1, 1, 1, 1, 2, 2, 2, 2],
        run_trial_index=[1, 2, 1, 2, 1, 2, 1, 2],
        initial_time_ms=[100, 200, 300, 400, 500, 600, 700, 800],
        dt_ms=[
            1000.0, 1010.0, 1200.0, 1210.0,
            1000.0 + drift, 1010.0 + drift, 1200.0 + drift, 1210.0 + drift,
        ],
    )


# --- condition by class -------------------------------------------------------


def test_a_purely_multiplicative_condition_effect_interacts_on_ms_but_not_on_log():
    """The reason ``log_mean_dt_ms`` is reported alongside ``mean_dt_ms``.

    Slow is exactly 1.2x Fast in every class: a constant *ratio* with no
    interaction to find. Because the classes sit at different baselines it
    nonetheless shows up as a condition-by-class interaction when the test runs
    on raw milliseconds — the scale the effect is not on. Per-subject speed and
    per-cell noise are multiplicative, so the log scale is additive by
    construction and its interaction is pure error; additive noise would leave
    the ms interaction with no error variance and the F undefined.
    """
    rng = np.random.default_rng(0)
    baselines = {1: 800.0, 2: 1400.0, 3: 1200.0}
    rows = [
        {
            "subject": f"H{subject:02d}",
            "condition": condition,
            "trial_class": code,
            "trial_class_name": class_name,
            "n_trials": 50,
            "mean_dt_ms": (
                baselines[code] * subject_scale * factor
                * float(np.exp(rng.normal(0.0, 0.04)))
            ),
            "accuracy": 0.8,
        }
        for subject in range(20)
        for subject_scale in [float(np.exp(rng.normal(0.0, 0.15)))]
        for condition, factor in (("fast", 1.0), ("slow", 1.2))
        for code, class_name in CLASS_NAMES.items()
    ]

    interaction = condition_class_statistics(pd.DataFrame(rows))
    interaction = interaction.loc[
        interaction["effect"] == "condition_x_trial_class"
    ].set_index("measure")

    assert interaction.loc["mean_dt_ms", "p"] < 0.05
    assert interaction.loc["log_mean_dt_ms", "p"] > 0.05
    assert (
        interaction.loc["log_mean_dt_ms", "partial_eta_squared"]
        < interaction.loc["mean_dt_ms", "partial_eta_squared"] / 3
    )


# --- condition x class contrasts ----------------------------------------------


def _scaled_cells(factor=1.20, n_subjects=6, jitter=0.05):
    """Cells where Slow is ``factor`` times Fast in every class.

    A pure multiplicative condition effect: the stretch factor must recover
    ``factor`` for all three classes, whatever the class differs by in level.
    ``jitter`` spreads each subject's own factor around ``factor``
    symmetrically *in log space*, alternating sign, so the geometric mean stays
    exactly ``factor`` while the log ratios carry real between-subject variance.
    Without that variance the interval is undefined, which is a property of the
    fixture rather than of the estimator.
    """
    base = {"easy": 800.0, "ambiguous": 1400.0, "misleading": 1200.0}
    accuracy = {"easy": 0.99, "ambiguous": 0.77, "misleading": 0.37}
    return pd.DataFrame([
        {
            "subject": f"H{index + 1:02d}",
            "condition": condition,
            "trial_class_name": class_name,
            "mean_dt_ms": (level + 20.0 * index) * (subject_factor if condition == "slow" else 1.0),
            "accuracy": accuracy[class_name] + 0.005 * index,
        }
        for index in range(n_subjects)
        for subject_factor in [factor * np.exp(jitter * (1 if index % 2 == 0 else -1))]
        for class_name, level in base.items()
        for condition in ("fast", "slow")
    ])


def test_accuracy_is_tested_against_chance_not_zero():
    accuracy = condition_class_contrasts(_scaled_cells())
    accuracy = accuracy.loc[accuracy["term"] == "accuracy_vs_chance"].set_index("trial_class_name")
    # Misleading sits below .5 and easy far above it, with opposite t signs.
    assert accuracy.loc["misleading", "estimate"] < 0.5
    assert accuracy.loc["misleading", "t"] < 0
    assert accuracy.loc["easy", "t"] > 0
    # The interval is in accuracy units, not as a deviation from .5.
    assert 0.0 < accuracy.loc["misleading", "ci_low"] < 0.5


# --- time on task -------------------------------------------------------------


def test_block_order_comes_from_the_session_clock_not_the_run_number():
    """Fast run 2 was recorded after Slow run 1, so DT drift must be measured
    against the chronological block, not the per-condition run counter."""
    drift = time_on_task(_interleaved_session(drift=-100.0))
    assert set(drift["condition"]) == {"all", "fast", "slow"}
    fast = drift.loc[drift["condition"] == "fast"].iloc[0]
    assert fast["n_blocks"] == 2
    assert fast["converged"]
    assert fast["dt_per_block_ms"] < 0

    flat = time_on_task(_interleaved_session(drift=0.0))
    assert flat.loc[flat["condition"] == "fast", "dt_per_block_ms"].iloc[0] == pytest.approx(
        0.0, abs=1e-9
    )

    # A single block cannot estimate a block slope, and says so.
    unestimable = time_on_task(
        trial_features(
            condition=["Fast"] * 2, run=[1, 1], run_trial_index=[1, 2],
            initial_time_ms=[100, 200], dt_ms=[1000.0, 1100.0],
        )
    )
    assert not unestimable["converged"].any()
    assert unestimable["dt_per_block_ms"].isna().all()


# --- within-block deciles -----------------------------------------------------


def _scheduled_class_session(subject="H01", n_subjects=1, positional_ms=0.0):
    """Sessions where trial class is confounded with within-block position.

    Each 20-trial block runs easy trials first and ambiguous trials second, and
    decision time is a pure function of class plus an optional genuine
    positional term. The observed profile therefore rises even when
    ``positional_ms`` is zero, and only the class adjustment can tell the two
    sources apart — which is the property panel B of F04 rests on.
    """
    block_length = 20
    sessions = []
    for index in range(n_subjects):
        name = subject if n_subjects == 1 else f"H{index + 1:02d}"
        for run in (1, 2):
            positions = list(range(1, block_length + 1))
            classes = ["easy"] * 10 + ["ambiguous"] * 10
            sessions.append(
                trial_features(
                    subject=[name] * block_length,
                    condition=["Fast" if run == 1 else "Slow"] * block_length,
                    run=[run] * block_length,
                    run_trial_index=positions,
                    initial_time_ms=[100_000 * run + 100 * position for position in positions],
                    trial_class=[1] * 10 + [2] * 10,
                    trial_class_name=classes,
                    dt_ms=[
                        (800.0 if name_ == "easy" else 1200.0)
                        + positional_ms * position + 10.0 * index
                        for name_, position in zip(classes, positions)
                    ],
                )
            )
    return pd.concat(sessions, ignore_index=True)


def test_the_class_adjustment_understates_real_drift_rather_than_inventing_it():
    """Removing per-class means also removes whatever genuine positional drift
    falls between the classes, because that drift is part of the class means.
    Here class and position are *perfectly* confounded, so the adjustment is
    maximally conservative. The direction is what matters: this estimator can
    under-state a real positional effect but never over-state one, so the
    class-adjusted slope is a lower bound and the compositional share of the
    raw slope is an upper bound.
    """
    pooled = within_block_decile_profile(_scheduled_class_session(positional_ms=5.0))
    pooled = pooled.loc[pooled["condition"] == "all"].set_index("decile")

    genuine_rise = 5.0 * 18  # deciles 1 and 10 average positions 1.5 and 19.5
    observed = pooled["dt_ms"]
    assert observed.loc[10] - observed.loc[1] == pytest.approx(400.0 + genuine_rise)
    surviving = pooled["dt_class_adjusted_ms"].loc[10] - pooled["dt_class_adjusted_ms"].loc[1]
    assert 0 < surviving < genuine_rise
    # Exactly the within-class share: half the block spanned at 5 ms/trial.
    assert surviving == pytest.approx(5.0 * 8, abs=1e-6)
