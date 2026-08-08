"""Modern derivative path and array I/O contract.

The project uses MNE-native files for raw/epochs/source objects and explicit
array-plus-sidecar files for analysis tensors that are consumed by later stages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd


PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    """Create a directory if needed and return it as a ``Path``."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def require_file(path: PathLike, *, purpose: Optional[str] = None) -> Path:
    """Return an existing file path or raise a clear input error."""
    file_path = Path(path)
    if not file_path.is_file():
        detail = f" for {purpose}" if purpose else ""
        raise FileNotFoundError(f"Required input file{detail} does not exist: {file_path}")
    return file_path


def _clean_entity(value: Optional[Union[str, int]]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for prefix in ("sub-", "ses-", "task-", "run-", "space-", "desc-", "proc-"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    return "".join(ch for ch in text.replace(" ", "") if ch.isalnum() or ch in ("-", "+"))


def _clean_extension(extension: str) -> str:
    if extension == "":
        return ""
    return extension if extension.startswith(".") else f".{extension}"


def derivative_path(
    root: PathLike,
    *,
    subject: str,
    suffix: str,
    extension: str,
    datatype: str = "meg",
    session: Optional[str] = None,
    task: str = "tokens",
    run: Optional[Union[str, int]] = None,
    acquisition: Optional[str] = None,
    processing: Optional[str] = None,
    space: Optional[str] = None,
    description: Optional[str] = None,
) -> Path:
    """Build a BIDS-derivatives-style output path.

    Non-standard analysis dimensions such as condition, alignment, band, ROI,
    and parcellation belong in the JSON sidecar metadata. Use ``description``
    only as a compact filename discriminator when multiple outputs share the
    same BIDS entities and suffix.
    """
    subject_clean = _clean_entity(subject)
    if subject_clean is None:
        raise ValueError("subject is required")

    entities: list[tuple[str, Optional[str]]] = [
        ("sub", subject_clean),
        ("ses", _clean_entity(session)),
        ("task", _clean_entity(task)),
        ("acq", _clean_entity(acquisition)),
        ("run", _clean_entity(run)),
        ("proc", _clean_entity(processing)),
        ("space", _clean_entity(space)),
        ("desc", _clean_entity(description)),
    ]
    stem = "_".join(f"{key}-{value}" for key, value in entities if value is not None)
    filename = f"{stem}_{_clean_entity(suffix) or suffix}{_clean_extension(extension)}"

    parts = [Path(root), "derivatives", f"sub-{subject_clean}"]
    session_clean = _clean_entity(session)
    if session_clean is not None:
        parts.append(f"ses-{session_clean}")
    parts.append(datatype)
    return Path(*parts) / filename


def sidecar_path(array_path: PathLike) -> Path:
    """Return the JSON sidecar path for an array derivative."""
    path = Path(array_path)
    if path.suffix:
        return path.with_suffix(".json")
    return path.with_name(f"{path.name}.json")


@dataclass(frozen=True)
class ArrayWithMetadata:
    """Loaded array derivative and its sidecar metadata."""

    data: np.ndarray
    metadata: dict[str, Any]


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def save_array(
    path: PathLike,
    data: np.ndarray,
    *,
    dims: Sequence[str],
    metadata: Optional[Mapping[str, Any]] = None,
    coords: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Save an analysis tensor as ``.npy`` plus a JSON sidecar.

    The sidecar carries shape, dtype, dimensions, coordinates, and stage-specific
    metadata so later stages can validate that they are consuming the expected
    derivative.
    """
    array = np.asarray(data)
    if len(dims) != array.ndim:
        raise ValueError(f"dims has length {len(dims)} but data has {array.ndim} dimensions")

    out_path = Path(path)
    ensure_dir(out_path.parent)
    np.save(out_path, array)

    sidecar = {
        "format": "npy+json",
        "schema_version": 1,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "dims": list(dims),
        "coords": _json_ready(coords or {}),
        "metadata": _json_ready(metadata or {}),
    }
    with sidecar_path(out_path).open("w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2, sort_keys=True)
        f.write("\n")
    return out_path


def load_array(
    path: PathLike,
    *,
    expected_ndim: Optional[int] = None,
    require_sidecar: bool = False,
    allow_pickle: bool = False,
) -> ArrayWithMetadata:
    """Load an analysis tensor saved by ``save_array`` or a legacy ``.npy``."""
    array_path = require_file(path, purpose="array derivative")
    data = np.load(array_path, allow_pickle=allow_pickle)
    if expected_ndim is not None and data.ndim != expected_ndim:
        raise ValueError(f"Expected {array_path} to have {expected_ndim} dimensions, got {data.ndim}")

    meta_path = sidecar_path(array_path)
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    elif require_sidecar:
        raise FileNotFoundError(f"Required sidecar does not exist: {meta_path}")
    else:
        metadata = {}
    return ArrayWithMetadata(data=data, metadata=metadata)


def save_table(
    path: PathLike,
    table: pd.DataFrame,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    sep: Optional[str] = None,
) -> Path:
    """Save a tabular derivative with a JSON sidecar."""
    out_path = Path(path)
    ensure_dir(out_path.parent)
    if sep is None:
        sep = "\t" if out_path.suffix == ".tsv" else ","
    table.to_csv(out_path, index=False, sep=sep)
    sidecar = {
        "format": "tsv+json" if sep == "\t" else "csv+json",
        "schema_version": 1,
        "n_rows": int(len(table)),
        "columns": list(table.columns),
        "metadata": _json_ready(metadata or {}),
    }
    with sidecar_path(out_path).open("w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2, sort_keys=True)
        f.write("\n")
    return out_path


def load_table(
    path: PathLike,
    *,
    sep: Optional[str] = None,
    converters: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """Load a tabular derivative without applying domain-specific semantics."""
    table_path = require_file(path, purpose="tabular derivative")
    if sep is None:
        sep = "\t" if table_path.suffix == ".tsv" else ","
    return pd.read_csv(table_path, sep=sep, converters=converters)


def save_sidecar(path: PathLike, metadata: Mapping[str, Any]) -> Path:
    """Write a JSON sidecar next to any derivative file or file base."""
    out_path = sidecar_path(path)
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(_json_ready(metadata), f, indent=2, sort_keys=True)
        f.write("\n")
    return out_path
