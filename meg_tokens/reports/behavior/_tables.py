"""Lazy, cached access to Stage 2/2b group derivatives for the figure layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from meg_tokens.behavior.tables import read_trial_features
from meg_tokens.io import DerivativeLayout


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


@dataclass
class BehaviorTableSet:
    """Lazy, cached access to Stage 2/2b group derivatives for the figure layer."""

    layout: DerivativeLayout
    subjects: tuple[str, ...] | None = None
    _cache: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)
    _sources: dict[str, str] = field(default_factory=dict, repr=False)

    def _subject_filter(self, table: pd.DataFrame) -> pd.DataFrame:
        if self.subjects is None or "subject" not in table.columns:
            return table
        wanted = {subject for subject in self.subjects}
        return table.loc[table["subject"].astype(str).isin(wanted)].reset_index(drop=True)

    def _filtered_tsv(self, path: Path) -> pd.DataFrame:
        return self._subject_filter(_read_tsv(path))

    def _load(
        self,
        cache_key: str,
        source_key: str,
        path: Path,
        *,
        description: str,
        command: str,
        read: Callable[[Path], pd.DataFrame],
    ) -> pd.DataFrame:
        """Read one derivative at most once, recording the path it came from.

        A missing file raises `FileNotFoundError` naming the exact path and the
        command that writes it, matching the project's error convention. The
        report layer never falls back to a partial table: a figure silently
        drawn from one is worse than no figure.
        """
        if cache_key in self._cache:
            return self._cache[cache_key]
        if not path.is_file():
            raise FileNotFoundError(
                f"{description} required by the behavior report is missing: "
                f"{path}. Run 'meg-tokens {command}' first."
            )
        table = read(path)
        self._cache[cache_key] = table
        self._sources[source_key] = str(path)
        return table

    def analysis(self, name: str) -> pd.DataFrame:
        """Load `sub-group_task-tokens_desc-<name>_beh.tsv` via
        `DerivativeLayout.behavior_analysis(name)`, applying the subject filter
        when the table has a `subject` column. Cached per name.
        """
        return self._load(
            name,
            name,
            self.layout.behavior_analysis(name),
            description=f"The '{name}' derivative",
            command=(
                "behavior ssm-fit"
                if name.startswith("ssm")
                else "behavior characterization"
            ),
            read=self._filtered_tsv,
        )

    def trial_features(self) -> pd.DataFrame:
        """`behavior_trial_features()`, filtered to primary_analysis_eligible
        rows with finite dt_ms, condition lower-cased. Cached."""

        def read(path: Path) -> pd.DataFrame:
            table = read_trial_features(path)
            table = table.loc[table["primary_analysis_eligible"].astype(bool)].copy()
            table = table.loc[pd.to_numeric(table["dt_ms"], errors="coerce").notna()]
            table["condition"] = table["condition"].astype(str).str.lower()
            return self._subject_filter(table)

        return self._load(
            "__trialfeatures__",
            "trialfeatures",
            self.layout.behavior_trial_features(),
            description="The trial-feature table",
            command="behavior analyze",
            read=read,
        )

    def summary(self) -> pd.DataFrame:
        return self._load(
            "__summary__",
            "summary",
            self.layout.behavior_summary(),
            description="The subject-summary table",
            command="behavior analyze",
            read=self._filtered_tsv,
        )

    def group_statistics(self) -> pd.DataFrame:
        return self._load(
            "__groupstats__",
            "groupstats",
            self.layout.behavior_group_statistics(),
            description="The paired group-statistics table",
            command="behavior analyze",
            read=_read_tsv,
        )

    @property
    def sources(self) -> dict[str, str]:
        """Every path actually read, for the sidecar."""
        return dict(self._sources)
