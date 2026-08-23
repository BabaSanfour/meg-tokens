"""Data root for the doc-verification scripts.

Reads `data_root` from the project config so these run unchanged on the
laptop, the cluster and the desktop -- the three roots differ and hardcoding
one is why these lived in a scratch directory instead of the repo.
Override with MEG_TOKENS_DATA_ROOT.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


def group_beh() -> Path:
    """The group behavioural derivative directory."""
    override = os.environ.get("MEG_TOKENS_DATA_ROOT")
    if override:
        root = Path(override)
    else:
        config = Path(__file__).resolve().parents[2] / "tokens.toml"
        match = re.search(
            r'^data_root\s*=\s*"([^"]+)"', config.read_text(), flags=re.M
        )
        if match is None:
            raise SystemExit(f"no data_root in {config}")
        root = Path(match.group(1))
    return root / "BIDS/derivatives/sub-group/beh"
