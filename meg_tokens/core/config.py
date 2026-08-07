"""Project-level paths and defaults loaded from a portable TOML file."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Optional

from .entities import normalize_subject_id

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


def _optional_path(value: object, *, base_dir: Path) -> Optional[Path]:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else base_dir / path


@dataclass(frozen=True)
class ProjectConfig:
    """Filesystem roots and stable naming defaults for a Tokens project."""

    data_root: Path
    subjects_dir: Optional[Path] = None
    trans_dir: Optional[Path] = None
    noise_dir: Optional[Path] = None
    task: str = "tokens"
    pipeline: str = "meg-tokens"
    behavior_ignore_files: tuple[str, ...] = ()
    subject_exclusions: tuple[str, ...] = ()
    # Classify random ('x') trials from the correct-target design profile.
    # Set False to use recorded 'e'/'a'/'m' labels only; see
    # docs/behavior_t0_1_nprob_trial_class.md section 3b.
    infer_random_classes: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", Path(self.data_root).expanduser())
        object.__setattr__(
            self, "behavior_ignore_files", tuple(self.behavior_ignore_files)
        )
        object.__setattr__(
            self,
            "subject_exclusions",
            tuple(normalize_subject_id(subject) for subject in self.subject_exclusions),
        )
        for field_name in ("subjects_dir", "trans_dir", "noise_dir"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, Path(value).expanduser())
        if not self.task.strip():
            raise ValueError("task must not be empty")
        if not self.pipeline.strip():
            raise ValueError("pipeline must not be empty")

    @property
    def raw_meg_root(self) -> Path:
        return self.data_root / "raw"

    @property
    def behavior_root(self) -> Path:
        return self.data_root / "tdms"

    @property
    def bids_root(self) -> Path:
        return self.data_root / "BIDS"

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        base_dir: str | Path = ".",
    ) -> "ProjectConfig":
        """Build configuration from a mapping and resolve relative paths."""
        section = values.get("project", values)
        if not isinstance(section, Mapping):
            raise ValueError("Project configuration must be a table")

        known = {item.name for item in fields(cls)}
        unknown = sorted(set(section) - known)
        if unknown:
            raise ValueError(f"Unknown project configuration fields: {unknown}")
        if "data_root" not in section:
            raise ValueError("Project configuration requires 'data_root'")

        root = Path(base_dir).expanduser()
        kwargs = dict(section)
        for name in ("data_root", "subjects_dir", "trans_dir", "noise_dir"):
            if name in kwargs:
                kwargs[name] = _optional_path(kwargs[name], base_dir=root)
        return cls(**kwargs)

    @classmethod
    def from_toml(cls, path: str | Path) -> "ProjectConfig":
        """Load a project configuration from TOML."""
        config_path = Path(path).expanduser()
        if not config_path.is_file():
            raise FileNotFoundError(f"Project configuration does not exist: {config_path}")
        with config_path.open("rb") as stream:
            values = tomllib.load(stream)
        return cls.from_mapping(values, base_dir=config_path.parent)
