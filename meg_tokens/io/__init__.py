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
from .layout import DerivativeLayout
from .labeled import load_dataarray, save_dataarray

__all__ = [
    "ArrayWithMetadata",
    "derivative_path",
    "DerivativeLayout",
    "ensure_dir",
    "load_array",
    "load_dataarray",
    "save_table",
    "save_sidecar",
    "require_file",
    "save_array",
    "save_dataarray",
    "sidecar_path",
]
