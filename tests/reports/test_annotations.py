import math

import pandas as pd

from meg_tokens.reports.annotations import (
    format_p,
    format_stat,
    significance_marker,
    stat_from_row,
)


def test_format_p_below_threshold_uses_less_than():
    assert format_p(3.6e-9) == "p < .001"


def test_format_p_above_threshold_uses_equals():
    assert format_p(0.017) == "p = .017"


def test_significance_marker_conventions():
    assert significance_marker(0.0001) == "***"
    assert significance_marker(0.005) == "**"
    assert significance_marker(0.03) == "*"
    assert significance_marker(0.083) == "n.s."
    assert significance_marker(float("nan")) == ""
    assert significance_marker(None) == ""


def test_stat_from_row_prefers_mean_over_mean_difference():
    row = pd.Series({
        "n_subjects": 32, "mean": -238.9, "mean_difference": -999.0,
        "sem": 29.4, "t": -8.12, "p": 3.6e-9, "df": 31.0, "cohens_dz": -1.44,
    })
    result = stat_from_row(row, label="test")
    assert result.mean == -238.9
    assert result.n_subjects == 32


def test_stat_from_row_falls_back_to_mean_difference():
    row = pd.Series({
        "n_subjects": 32, "mean_difference": -0.108,
        "sem": None, "t": -2.52, "p": 0.017, "df": 31.0, "cohens_dz": -0.45,
    })
    result = stat_from_row(row, label="test")
    assert result.mean == -0.108


def test_stat_from_row_maps_non_finite_to_none_not_zero():
    row = pd.Series({
        "n_subjects": 0, "mean": float("nan"), "sem": float("nan"),
        "t": float("nan"), "p": float("nan"), "df": float("nan"), "cohens_dz": float("nan"),
    })
    result = stat_from_row(row, label="empty")
    assert result.mean is None
    assert result.t is None
    assert result.p is None


def test_format_stat_never_emits_the_substring_nan():
    row = pd.Series({
        "n_subjects": 0, "mean": float("nan"), "sem": float("nan"),
        "t": float("nan"), "p": float("nan"), "df": float("nan"), "cohens_dz": float("nan"),
    })
    result = stat_from_row(row, label="empty")
    assert "nan" not in format_stat(result).lower()
