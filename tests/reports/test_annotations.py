import pandas as pd

from meg_tokens.reports.annotations import format_stat, stat_from_row


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
