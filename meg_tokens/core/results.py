"""Common workflow return types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Tuple


@dataclass(frozen=True)
class WorkflowResult:
    """Declared inputs, outputs, and effective settings of one workflow run."""

    stage: str
    inputs: Tuple[Path, ...] = ()
    outputs: Tuple[Path, ...] = ()
    settings: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.stage.strip():
            raise ValueError("stage must not be empty")
        object.__setattr__(self, "inputs", tuple(Path(path) for path in self.inputs))
        object.__setattr__(self, "outputs", tuple(Path(path) for path in self.outputs))
