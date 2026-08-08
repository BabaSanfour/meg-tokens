"""Opt-in regression against the preprint's N=28 behavioral results."""

import os

import numpy as np
import pandas as pd
import pytest

from meg_tokens.behavior.analyses.performance import behavior_group_statistics
from meg_tokens.core import ProjectConfig
from meg_tokens.io import DerivativeLayout


def test_real_behavior_group_statistics_match_preprint():
    config_path = os.environ.get("MEG_TOKENS_REAL_CONFIG")
    if not config_path:
        pytest.skip("Set MEG_TOKENS_REAL_CONFIG to an N=28 project TOML")

    project = ProjectConfig.from_toml(config_path)
    summary_path = DerivativeLayout(
        project.bids_root,
        task=project.task,
    ).behavior_summary()
    if not summary_path.is_file():
        pytest.skip("Run behavior ingest and behavior analyze for the real dataset first")

    summary = pd.read_csv(summary_path, sep="\t")
    assert len(project.subject_exclusions) == 4
    assert len(summary) == 28
    results = behavior_group_statistics(summary).set_index(
        ["analysis", "contrast", "view"]
    )

    published = {
        ("decision_time", "easy_vs_ambiguous", "primary"):
            {"mean_a": 1028, "sem_a": 59, "mean_b": 1405, "sem_b": 74,
             "t": -15.04, "p": 1.19e-14},
        ("decision_time", "easy_vs_misleading", "primary"):
            {"mean_a": 1028, "sem_a": 59, "mean_b": 1433, "sem_b": 79,
             "t": -13.10, "p": 3.25e-13},
        ("decision_time", "ambiguous_vs_misleading", "primary"):
            {"t": -1.84, "p": 0.077},
        ("decision_time", "fast_vs_slow", "primary"):
            {"mean_a": 1166, "sem_a": 71, "mean_b": 1293, "sem_b": 68,
             "t": -6.08, "p": 1.71e-6},
        ("error_count", "fast_vs_slow", "primary"):
            {"mean_a": 45.1, "mean_b": 36.0, "t": 6.10, "p": 1.7e-6},
    }
    for key, expected in published.items():
        observed = results.loc[key]
        for field, value in expected.items():
            if field == "p":
                assert np.isclose(observed[field], value, rtol=0.10, atol=1e-16)
            elif field == "t":
                assert observed[field] == pytest.approx(value, abs=0.05)
            else:
                assert observed[field] == pytest.approx(value, abs=1.0)
