"""Shared I/O helpers for MEG Tokens derivatives."""

from .contract import (
    ArrayWithMetadata,
    derivative_path,
    ensure_dir,
    load_array,
    save_table,
    save_sidecar,
    require_file,
    save_array,
    sidecar_path,
)

__all__ = [
    "ArrayWithMetadata",
    "derivative_path",
    "ensure_dir",
    "load_array",
    "save_table",
    "save_sidecar",
    "require_file",
    "save_array",
    "sidecar_path",
]
