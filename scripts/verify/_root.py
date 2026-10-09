"""Data root for the doc-verification scripts.

Read from this checkout's ``tokens.toml`` through ``ProjectConfig``, the way
every command and the real-data tests find the data, so these run unchanged on
any machine where the data is configured.
"""

from __future__ import annotations

from pathlib import Path

from meg_tokens.core import ProjectConfig

CONFIG = Path(__file__).resolve().parents[2] / "tokens.toml"


def group_beh() -> Path:
    """The group behavioural derivative directory."""
    return ProjectConfig.from_toml(CONFIG).bids_root / "derivatives" / "sub-group" / "beh"
