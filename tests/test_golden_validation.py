import json

import numpy as np
import pandas as pd
import pytest

from meg_tokens.io import load_array, save_array, save_table
from meg_tokens.validation import run_golden_validation


def _write_config(path, comparisons):
    path.write_text(json.dumps({"comparisons": comparisons}, indent=2), encoding="utf-8")
    return path


def test_run_golden_validation_writes_passing_array_and_table_report(tmp_path):
    modern_array = tmp_path / "modern.npy"
    reference_array = tmp_path / "reference.npy"
    modern_table = tmp_path / "modern_behavior.tsv"
    reference_table = tmp_path / "reference_behavior.tsv"

    save_array(
        modern_array,
        np.arange(6, dtype=float).reshape(2, 3),
        dims=("label", "time"),
        coords={"label": ["A", "B"], "time_sec": [0.0, 0.1, 0.2]},
        metadata={"stage": "erp_parcellation"},
    )
    save_array(
        reference_array,
        np.arange(6, dtype=float).reshape(2, 3),
        dims=("label", "time"),
        coords={"label": ["A", "B"], "time_sec": [0.0, 0.1, 0.2]},
        metadata={"stage": "legacy_reference"},
    )
    save_table(
        modern_table,
        pd.DataFrame({"trial": [2, 1], "rt": [0.2, 0.1], "condition": ["Fast", "Fast"]}),
        metadata={"stage": "behavior"},
    )
    save_table(
        reference_table,
        pd.DataFrame({"trial": [1, 2], "rt": [0.1, 0.2], "condition": ["Fast", "Fast"]}),
        metadata={"stage": "legacy_reference"},
    )
    config = _write_config(
        tmp_path / "golden_config.json",
        [
            {
                "name": "erp_array",
                "kind": "array",
                "modern": modern_array.name,
                "reference": reference_array.name,
                "compare_sidecar_keys": ["dims", "coords"],
            },
            {
                "name": "behavior_table",
                "kind": "table",
                "modern": modern_table.name,
                "reference": reference_table.name,
                "sort_by": ["trial"],
                "columns": ["trial", "rt", "condition"],
            },
        ],
    )

    report_path = tmp_path / "validation.tsv"
    report = run_golden_validation(config, out_tsv=report_path)
    loaded_report = pd.read_csv(report_path, sep="\t")

    assert report["status"].tolist() == ["pass", "pass"]
    assert loaded_report["name"].tolist() == ["erp_array", "behavior_table"]
    assert load_array(modern_array, require_sidecar=True).metadata["dims"] == ["label", "time"]


def test_run_golden_validation_reports_array_mismatch(tmp_path):
    modern_array = tmp_path / "modern.npy"
    reference_array = tmp_path / "reference.npy"
    save_array(modern_array, np.array([1.0, 2.0]), dims=("time",))
    save_array(reference_array, np.array([1.0, 3.0]), dims=("time",))
    config = _write_config(
        tmp_path / "golden_config.json",
        [
            {
                "name": "erp_array",
                "kind": "array",
                "modern": modern_array.name,
                "reference": reference_array.name,
                "atol": 0.0,
                "rtol": 0.0,
            }
        ],
    )

    report = run_golden_validation(config)

    assert report.loc[0, "status"] == "fail"
    assert report.loc[0, "max_abs_diff"] == 1.0
    assert report.loc[0, "message"] == "array values differ"


def test_run_golden_validation_requires_real_reference_file(tmp_path):
    modern_array = tmp_path / "modern.npy"
    save_array(modern_array, np.array([1.0]), dims=("time",))
    config = _write_config(
        tmp_path / "golden_config.json",
        [
            {
                "name": "missing_reference",
                "kind": "array",
                "modern": modern_array.name,
                "reference": "absent.npy",
            }
        ],
    )

    with pytest.raises(FileNotFoundError, match="Required input file"):
        run_golden_validation(config)
