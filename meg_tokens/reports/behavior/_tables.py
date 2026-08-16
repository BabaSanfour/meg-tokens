"""Lazy, cached access to Stage 2/2b group derivatives for the figure layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from meg_tokens.behavior.tables import read_trial_features
from meg_tokens.core import normalize_subject_id
from meg_tokens.io import DerivativeLayout


@dataclass
class BehaviorTableSet:
    """Lazy, cached access to Stage 2/2b group derivatives for the figure layer."""

    layout: DerivativeLayout
    subjects: tuple[str, ...] | None = None
    neural_metrics_path: str | None = None
    _cache: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)
    _sources: dict[str, str] = field(default_factory=dict, repr=False)

    def _subject_filter(self, table: pd.DataFrame) -> pd.DataFrame:
        if self.subjects is None or "subject" not in table.columns:
            return table
        wanted = {normalize_subject_id(subject) for subject in self.subjects}
        return table.loc[table["subject"].astype(str).isin(wanted)].reset_index(drop=True)

    def analysis(self, name: str) -> pd.DataFrame:
        """Load `sub-group_task-tokens_desc-<name>_beh.tsv` via
        `DerivativeLayout.behavior_analysis(name)`, applying the subject filter
        when the table has a `subject` column. Cached per name.

        Raises FileNotFoundError naming the exact path and the command that
        writes it ('meg-tokens behavior characterization', or
        'meg-tokens behavior ssm-fit' for any ssm* table), matching the
        project's error convention.
        """
        if name in self._cache:
            return self._cache[name]
        path = self.layout.behavior_analysis(name)
        if not path.is_file():
            command = "behavior ssm-fit" if name.startswith("ssm") else "behavior characterization"
            raise FileNotFoundError(
                f"The '{name}' derivative required by the behavior report is missing: "
                f"{path}. Run 'meg-tokens {command}' first."
            )
        table = pd.read_csv(path, sep="\t")
        table = self._subject_filter(table)
        self._cache[name] = table
        self._sources[name] = str(path)
        return table

    def trial_features(self) -> pd.DataFrame:
        """`behavior_trial_features()`, filtered to primary_analysis_eligible
        rows with finite dt_ms, condition lower-cased. Cached."""
        key = "__trialfeatures__"
        if key in self._cache:
            return self._cache[key]
        path = self.layout.behavior_trial_features()
        if not path.is_file():
            raise FileNotFoundError(
                "The trial-feature table required by the behavior report is missing: "
                f"{path}. Run 'meg-tokens behavior analyze' first."
            )
        table = read_trial_features(path)
        table = table.loc[table["primary_analysis_eligible"].astype(bool)].copy()
        table = table.loc[pd.to_numeric(table["dt_ms"], errors="coerce").notna()]
        table["condition"] = table["condition"].astype(str).str.lower()
        table = self._subject_filter(table)
        self._cache[key] = table
        self._sources["trialfeatures"] = str(path)
        return table

    def summary(self) -> pd.DataFrame:
        key = "__summary__"
        if key in self._cache:
            return self._cache[key]
        path = self.layout.behavior_summary()
        if not path.is_file():
            raise FileNotFoundError(
                f"The subject-summary table required by the behavior report is missing: "
                f"{path}. Run 'meg-tokens behavior analyze' first."
            )
        table = self._subject_filter(pd.read_csv(path, sep="\t"))
        self._cache[key] = table
        self._sources["summary"] = str(path)
        return table

    def group_statistics(self) -> pd.DataFrame:
        key = "__groupstats__"
        if key in self._cache:
            return self._cache[key]
        path = self.layout.behavior_group_statistics()
        if not path.is_file():
            raise FileNotFoundError(
                f"The paired group-statistics table required by the behavior report is "
                f"missing: {path}. Run 'meg-tokens behavior analyze' first."
            )
        table = pd.read_csv(path, sep="\t")
        self._cache[key] = table
        self._sources["groupstats"] = str(path)
        return table

    def neural_metrics(self) -> pd.DataFrame:
        """Optional subject-level MEG metrics table for F26, gated on
        `--neural-metrics`. Requires `subject` and `neural_peak_ms` columns."""
        key = "__neural__"
        if key in self._cache:
            return self._cache[key]
        if not self.neural_metrics_path:
            raise FileNotFoundError(
                "individualprofile-neural (F26) requires --neural-metrics "
                "<subject-level CSV/TSV with 'subject' and 'neural_peak_ms' columns>."
            )
        path = Path(self.neural_metrics_path)
        if not path.is_file():
            raise FileNotFoundError(f"--neural-metrics table does not exist: {path}")
        table = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
        required = {"subject", "neural_peak_ms"}
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"--neural-metrics table is missing columns: {sorted(missing)}")
        table = self._subject_filter(table)
        self._cache[key] = table
        self._sources["neural_metrics"] = str(path)
        return table

    @property
    def sources(self) -> dict[str, str]:
        """Every path actually read, for the sidecar."""
        return dict(self._sources)
