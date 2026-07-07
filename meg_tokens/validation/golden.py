"""Golden-reference validation for real MEG Tokens derivatives."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from meg_tokens.io import load_array, require_file, save_table


SUPPORTED_KINDS = {"array", "table"}


def _read_json(path: str | Path) -> dict[str, Any]:
    config_path = require_file(path, purpose="golden-validation config")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_path(path_value: str | Path, *, base_dir: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base_dir / path


def _status(passed: bool) -> str:
    return "pass" if passed else "fail"


def _shape_text(shape: Sequence[int]) -> str:
    return "x".join(str(int(value)) for value in shape)


def _sidecar_equal(modern_meta: Mapping[str, Any], reference_meta: Mapping[str, Any], keys: Sequence[str]) -> tuple[bool, str]:
    messages = []
    for key in keys:
        if modern_meta.get(key) != reference_meta.get(key):
            messages.append(f"sidecar {key} differs")
    return not messages, "; ".join(messages)


def _compare_arrays(
    name: str,
    modern_path: Path,
    reference_path: Path,
    *,
    atol: float,
    rtol: float,
    compare_sidecar_keys: Sequence[str],
) -> dict[str, Any]:
    modern = load_array(modern_path, require_sidecar=True)
    reference = load_array(reference_path, require_sidecar=False)

    row: dict[str, Any] = {
        "name": name,
        "kind": "array",
        "modern_path": str(modern_path),
        "reference_path": str(reference_path),
        "modern_shape": _shape_text(modern.data.shape),
        "reference_shape": _shape_text(reference.data.shape),
        "atol": float(atol),
        "rtol": float(rtol),
        "n_values": int(modern.data.size),
        "max_abs_diff": np.nan,
        "mean_abs_diff": np.nan,
        "nan_mismatch_count": np.nan,
        "message": "",
    }

    if modern.data.shape != reference.data.shape:
        row["status"] = "fail"
        row["message"] = "shape differs"
        return row

    modern_values = np.asarray(modern.data, dtype=float)
    reference_values = np.asarray(reference.data, dtype=float)
    abs_diff = np.abs(modern_values - reference_values)
    finite = np.isfinite(abs_diff)
    if np.any(finite):
        row["max_abs_diff"] = float(np.max(abs_diff[finite]))
        row["mean_abs_diff"] = float(np.mean(abs_diff[finite]))
    row["nan_mismatch_count"] = int(np.sum(np.isnan(modern_values) != np.isnan(reference_values)))

    values_pass = bool(np.allclose(modern_values, reference_values, atol=atol, rtol=rtol, equal_nan=True))
    sidecar_pass = True
    sidecar_message = ""
    if compare_sidecar_keys and reference.metadata:
        sidecar_pass, sidecar_message = _sidecar_equal(
            modern.metadata,
            reference.metadata,
            compare_sidecar_keys,
        )

    row["status"] = _status(values_pass and sidecar_pass)
    if not values_pass:
        row["message"] = "array values differ"
    if sidecar_message:
        row["message"] = "; ".join(part for part in [row["message"], sidecar_message] if part)
    return row


def _read_table(path: Path) -> pd.DataFrame:
    require_file(path, purpose="table derivative")
    sep = "\t" if path.suffix == ".tsv" else ","
    return pd.read_csv(path, sep=sep)


def _select_and_sort_table(
    table: pd.DataFrame,
    *,
    columns: Optional[Sequence[str]],
    sort_by: Optional[Sequence[str]],
    path: Path,
) -> pd.DataFrame:
    out = table.copy()
    if columns:
        missing = sorted(set(columns) - set(out.columns))
        if missing:
            raise ValueError(f"Table {path} is missing requested columns: {missing}")
        out = out[list(columns)]
    if sort_by:
        missing = sorted(set(sort_by) - set(out.columns))
        if missing:
            raise ValueError(f"Table {path} is missing sort columns: {missing}")
        out = out.sort_values(list(sort_by), kind="mergesort").reset_index(drop=True)
    return out


def _column_equal(left: pd.Series, right: pd.Series, *, atol: float, rtol: float) -> tuple[bool, float]:
    left_num = pd.to_numeric(left, errors="coerce")
    right_num = pd.to_numeric(right, errors="coerce")
    numeric = left_num.notna() | right_num.notna()
    if numeric.all():
        left_values = left_num.to_numpy(dtype=float)
        right_values = right_num.to_numpy(dtype=float)
        diff = np.abs(left_values - right_values)
        finite = np.isfinite(diff)
        max_diff = float(np.max(diff[finite])) if np.any(finite) else 0.0
        return bool(np.allclose(left_values, right_values, atol=atol, rtol=rtol, equal_nan=True)), max_diff

    left_text = left.astype("string").fillna("<NA>")
    right_text = right.astype("string").fillna("<NA>")
    return bool(left_text.equals(right_text)), 0.0


def _compare_tables(
    name: str,
    modern_path: Path,
    reference_path: Path,
    *,
    atol: float,
    rtol: float,
    columns: Optional[Sequence[str]],
    sort_by: Optional[Sequence[str]],
) -> dict[str, Any]:
    modern = _select_and_sort_table(_read_table(modern_path), columns=columns, sort_by=sort_by, path=modern_path)
    reference = _select_and_sort_table(
        _read_table(reference_path),
        columns=columns,
        sort_by=sort_by,
        path=reference_path,
    )

    row: dict[str, Any] = {
        "name": name,
        "kind": "table",
        "modern_path": str(modern_path),
        "reference_path": str(reference_path),
        "modern_shape": _shape_text(modern.shape),
        "reference_shape": _shape_text(reference.shape),
        "atol": float(atol),
        "rtol": float(rtol),
        "n_values": int(modern.shape[0] * modern.shape[1]),
        "max_abs_diff": np.nan,
        "mean_abs_diff": np.nan,
        "nan_mismatch_count": np.nan,
        "message": "",
    }

    if modern.shape != reference.shape:
        row["status"] = "fail"
        row["message"] = "shape differs"
        return row
    if list(modern.columns) != list(reference.columns):
        row["status"] = "fail"
        row["message"] = "columns differ"
        return row

    failed_columns = []
    max_diffs = []
    for column in modern.columns:
        passed, max_diff = _column_equal(modern[column], reference[column], atol=atol, rtol=rtol)
        max_diffs.append(max_diff)
        if not passed:
            failed_columns.append(column)

    row["max_abs_diff"] = float(max(max_diffs)) if max_diffs else 0.0
    row["mean_abs_diff"] = np.nan
    row["status"] = _status(not failed_columns)
    if failed_columns:
        row["message"] = "columns differ: " + ", ".join(failed_columns)
    return row


def compare_from_config(item: Mapping[str, Any], *, base_dir: Path) -> dict[str, Any]:
    """Compare one configured real-reference item."""
    kind = str(item.get("kind", "")).strip()
    if kind not in SUPPORTED_KINDS:
        raise ValueError(f"Comparison kind must be one of {sorted(SUPPORTED_KINDS)}, got {kind!r}")
    name = str(item.get("name") or Path(str(item.get("modern", ""))).stem)
    modern = _resolve_path(item["modern"], base_dir=base_dir)
    reference = _resolve_path(item["reference"], base_dir=base_dir)
    atol = float(item.get("atol", 1e-8))
    rtol = float(item.get("rtol", 1e-5))

    if kind == "array":
        return _compare_arrays(
            name,
            modern,
            reference,
            atol=atol,
            rtol=rtol,
            compare_sidecar_keys=tuple(item.get("compare_sidecar_keys", ("dims", "coords"))),
        )
    return _compare_tables(
        name,
        modern,
        reference,
        atol=atol,
        rtol=rtol,
        columns=item.get("columns"),
        sort_by=item.get("sort_by"),
    )


def run_golden_validation(
    config_path: str | Path,
    *,
    out_tsv: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Run configured golden-reference comparisons and optionally save a report."""
    config_file = require_file(config_path, purpose="golden-validation config")
    config = _read_json(config_file)
    comparisons = config.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError(f"Config {config_file} must contain a non-empty 'comparisons' list")

    rows = [compare_from_config(item, base_dir=config_file.parent) for item in comparisons]
    report = pd.DataFrame(rows)
    if out_tsv is not None:
        save_table(
            out_tsv,
            report,
            metadata={
                "stage": "golden_validation",
                "config": str(config_file),
                "n_comparisons": int(len(report)),
                "n_failed": int(np.sum(report["status"] != "pass")),
            },
        )
    return report
