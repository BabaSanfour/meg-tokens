"""Canonical identifiers for subjects and task runs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple


_SUBJECT_RE = re.compile(r"H0*([0-9]+)", re.IGNORECASE)
_RUN_RE = re.compile(r"([A-Za-z]+)?0*([0-9]+)")


def normalize_subject_id(subject_id: str) -> str:
    """Normalize a Tokens subject label to the canonical ``H01`` form."""
    match = _SUBJECT_RE.fullmatch(str(subject_id).strip())
    if not match:
        raise ValueError(f"Invalid subject ID: {subject_id}")
    return f"H{int(match.group(1)):02d}"


def parse_run_label(run_id: str) -> Tuple[str, Optional[str]]:
    """Return numeric run and optional condition from labels such as ``Slow1``."""
    text = str(run_id).strip()
    if text.lower().startswith("run-"):
        return text.split("-", 1)[1], None

    match = _RUN_RE.fullmatch(text)
    if match is None:
        cleaned = "".join(ch for ch in text if ch.isalnum() or ch in ("-", "+"))
        return cleaned, None
    condition = match.group(1)
    run = str(int(match.group(2)))
    return run, condition.capitalize() if condition else None


@dataclass(frozen=True)
class RunSpec:
    """Canonical identity of one Tokens recording or derivative run."""

    subject: str
    run: str
    condition: Optional[str] = None
    alignment: Optional[str] = None

    def __post_init__(self) -> None:
        subject = normalize_subject_id(self.subject)
        run, inferred_condition = parse_run_label(self.run)
        condition = self.condition or inferred_condition
        if condition:
            condition = "RT" if condition.upper() == "RT" else condition.capitalize()
        alignment = self.alignment.lower() if self.alignment else None
        if alignment not in {None, "go", "enter", "feedback"}:
            raise ValueError(f"Unknown alignment event: {self.alignment}")

        object.__setattr__(self, "subject", subject)
        object.__setattr__(self, "run", run)
        object.__setattr__(self, "condition", condition)
        object.__setattr__(self, "alignment", alignment)
