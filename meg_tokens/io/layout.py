"""Central construction of persisted derivative paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from meg_tokens.core import normalize_subject_id, parse_run_label

from .contract import derivative_path


def _condition(value: Optional[str]) -> Optional[str]:
    return value.lower() if value else None


def _description(*parts: Optional[str]) -> Optional[str]:
    values = [str(part) for part in parts if part]
    return "-".join(values) if values else None


def _frequency(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


@dataclass(frozen=True)
class DerivativeLayout:
    """Build exact MEG Tokens derivative paths from canonical entities."""

    root: Path
    task: str = "tokens"

    def __init__(
        self,
        root: str | Path,
        *,
        task: str = "tokens",
    ) -> None:
        object.__setattr__(self, "root", Path(root))
        object.__setattr__(self, "task", task)

    def path(
        self,
        *,
        subject: str,
        suffix: str,
        extension: str,
        datatype: str = "meg",
        run: Optional[str] = None,
        processing: Optional[str] = None,
        space: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Path:
        # The single place a subject label is canonicalized. Labels arrive
        # canonical -- the CLI normalizes on entry, the staged tree is
        # canonical by construction -- so this is the funnel, not a check
        # repeated by every caller.
        return derivative_path(
            self.root,
            subject=normalize_subject_id(subject) if subject.lower() != "group" else "group",
            suffix=suffix,
            extension=extension,
            datatype=datatype,
            task=self.task,
            run=run,
            processing=processing,
            space=space,
            description=description,
        )

    def behavior(self, *, subject: str, run: str, condition: str) -> Path:
        run_number, inferred = parse_run_label(run)
        return self.path(
            subject=subject,
            datatype="beh",
            run=run_number,
            description=_condition(condition or inferred),
            suffix="beh",
            extension=".tsv",
        )

    def find_behavior(
        self,
        *,
        subject: str,
        run: str,
        condition: Optional[str] = None,
    ) -> Path:
        """Find the unique behavior table for one run."""
        run_number, inferred = parse_run_label(run)
        condition = condition or inferred
        directory = self.root / "derivatives" / f"sub-{subject}" / "beh"
        if condition:
            candidates = [self.behavior(subject=subject, run=run_number, condition=condition)]
        else:
            candidates = sorted(
                directory.glob(
                    f"sub-{subject}_task-{self.task}_run-{run_number}_desc-*_beh.tsv"
                )
            )
        existing = [path for path in candidates if path.is_file()]
        if not existing:
            raise FileNotFoundError(
                f"No behavior TSV derivative found for subject={subject}, "
                f"run={run_number} under {self.root}"
            )
        if len(existing) > 1:
            raise ValueError(
                f"Multiple behavior TSV derivatives matched subject={subject}, "
                f"run={run_number}: {existing}"
            )
        return existing[0]

    def behavior_tables(
        self,
        *,
        subjects: Optional[Sequence[str]] = None,
        conditions: Optional[Sequence[str]] = None,
    ) -> list[Path]:
        """Find parsed behavior derivatives using canonical entities."""
        subject_filter = (
            {normalize_subject_id(subject) for subject in subjects}
            if subjects
            else None
        )
        condition_filter = (
            {condition.lower() for condition in conditions}
            if conditions
            else None
        )
        pattern = (
            f"derivatives/sub-*/beh/"
            f"sub-*_task-{self.task}_run-*_desc-*_beh.tsv"
        )
        selected = []
        for path in sorted(self.root.glob(pattern)):
            if not path.is_file():
                continue
            subject = path.name.split("_", 1)[0].replace("sub-", "")
            description = path.stem.split("_desc-", 1)[1].rsplit("_beh", 1)[0]
            if subject_filter is not None and subject not in subject_filter:
                continue
            if condition_filter is not None and description.lower() not in condition_filter:
                continue
            selected.append(path)
        if not selected:
            raise FileNotFoundError(
                f"No behavior derivatives matched subjects={subjects}, "
                f"conditions={conditions} under {self.root}"
            )
        return selected

    def behavior_summary(self) -> Path:
        """Return the group behavior summary derivative path."""
        return self.path(
            subject="group",
            datatype="beh",
            description="summary",
            suffix="beh",
            extension=".tsv",
        )

    def behavior_group_statistics(self) -> Path:
        """Return the paired group behavior statistics derivative path."""
        return self.path(
            subject="group",
            datatype="beh",
            description="groupstats",
            suffix="beh",
            extension=".tsv",
        )

    def behavior_trial_features(self) -> Path:
        """Return the trial-level behavior feature derivative path."""
        return self.path(
            subject="group",
            datatype="beh",
            description="trialfeatures",
            suffix="beh",
            extension=".tsv",
        )

    def behavior_analysis(self, name: str) -> Path:
        """Return a group behavior derivative path for one named analysis.

        Used by the Stage 2b behavioral characterization analyses, each of
        which writes its own subject-level table and, where it has one, a
        matching group-statistics table.
        """
        if not name or not name.isalnum():
            raise ValueError(
                "Behavior analysis names must be non-empty and alphanumeric "
                f"so that they stay valid BIDS description entities: {name!r}"
            )
        return self.path(
            subject="group",
            datatype="beh",
            description=name,
            suffix="beh",
            extension=".tsv",
        )

    def behavior_subject_analysis(self, subject: str, name: str) -> Path:
        """Return one subject's own table for a named behavior analysis.

        The counterpart of :meth:`behavior_analysis` for analyses expensive
        enough to be computed one subject at a time and pooled afterwards.
        """
        if not name or not name.isalnum():
            raise ValueError(
                "Behavior analysis names must be non-empty and alphanumeric "
                f"so that they stay valid BIDS description entities: {name!r}"
            )
        return self.path(
            subject=subject,
            datatype="beh",
            description=name,
            suffix="beh",
            extension=".tsv",
        )

    def preprocessed_raw(
        self,
        *,
        subject: str,
        run: str,
        condition: Optional[str] = None,
        processing: str = "clean",
    ) -> Path:
        run_number, inferred = parse_run_label(run)
        condition = condition or inferred
        return self.path(
            subject=subject,
            run=run_number,
            processing=processing,
            description=_condition(condition) or processing,
            suffix="raw",
            extension=".fif",
        )

    def epochs(
        self,
        *,
        subject: str,
        run: str,
        condition: Optional[str],
        alignment: str,
        stream: str,
    ) -> Path:
        """One run's epochs around one lock, from one preprocessing stream."""
        run_number, inferred = parse_run_label(run)
        return self.path(
            subject=subject,
            run=run_number,
            processing=stream,
            description=_description(_condition(condition or inferred), alignment),
            suffix="epo",
            extension=".fif",
        )

    def epoch_amplitudes(self, *, subject: str, stream: str) -> Path:
        """One subject's per-epoch peak-to-peak table, every run and lock of one stream."""
        return self.path(
            subject=subject,
            processing=stream,
            description="epochamplitudes",
            suffix="meg",
            extension=".tsv",
        )

    def epoch_thresholds(self, *, stream: str) -> Path:
        """The cohort's outlier threshold per lock and magnetometer, for one stream."""
        return self.path(
            subject="group",
            processing=stream,
            description="epochthresholds",
            suffix="meg",
            extension=".tsv",
        )

    def find_epochs(
        self,
        *,
        subject: str,
        run: str,
        condition: Optional[str],
        alignment: str,
        stream: str,
    ) -> Path:
        """The epoch derivative for one run, lock and stream, which must exist."""
        path = self.epochs(
            subject=subject, run=run, condition=condition, alignment=alignment, stream=stream,
        )
        if not path.is_file():
            raise FileNotFoundError(f"No epochs at {path}. Run `meg-tokens meg epoch` first.")
        return path

    def source(
        self,
        *,
        subject: str,
        suffix: str,
        extension: str,
        run: Optional[str] = None,
        condition: Optional[str] = None,
        description: Optional[str] = None,
        processing: Optional[str] = None,
        space: Optional[str] = None,
    ) -> Path:
        run_number = None
        inferred = None
        if run is not None:
            run_number, inferred = parse_run_label(run)
        return self.path(
            subject=subject,
            run=run_number,
            processing=processing,
            space=space,
            description=_description(_condition(condition or inferred), description),
            suffix=suffix,
            extension=extension,
        )

    def source_models(self, *, subject: str) -> dict[str, Path]:
        """One subject's noise covariance, BEM solution and source space."""
        return {
            "cov": self.path(subject=subject, description="noise", suffix="cov", extension=".fif"),
            "bem": self.path(subject=subject, suffix="bem", extension=".fif"),
            "src": self.path(subject=subject, space="subject", suffix="src", extension=".fif"),
        }

    def trans(self, *, subject: str, run: str, condition: str) -> Path:
        """One recording's MEG-MRI transform, as coregistration writes it."""
        return self.source(subject=subject, run=run, condition=condition, suffix="trans", extension=".fif")

    def forward(self, *, subject: str, run: str, condition: str) -> Path:
        """One recording's forward model; it does not depend on the stream."""
        return self.source(subject=subject, run=run, condition=condition, suffix="fwd", extension=".fif")

    def inverse(self, *, subject: str, run: str, condition: str, stream: str) -> Path:
        """One recording's inverse operator for one stream."""
        return self.source(
            subject=subject, run=run, condition=condition, processing=stream, suffix="inv", extension=".fif",
        )

    def band_power(self, *, subject: str, stream: str, lock: str, space: str) -> Path:
        """One subject's single-trial band power around one lock, every Fast and Slow run, one stream and space."""
        return self.path(
            subject=subject, processing=stream, space=space, description=lock, suffix="power", extension=".h5",
        )

    def band_power_trials(self, *, subject: str, stream: str, lock: str) -> Path:
        """The trials of a subject's band power, in its order: one row per trial, both spaces."""
        return self.path(
            subject=subject, processing=stream, description=lock, suffix="trials", extension=".tsv",
        )

    def beamformer(self, *, subject: str, run: str, condition: str, stream: str) -> Path:
        """One recording's LCMV filters for one stream, the check on deep-structure results."""
        return self.source(
            subject=subject, run=run, condition=condition, processing=stream, suffix="lcmv", extension=".h5",
        )

    def find_source_manifest(
        self,
        *,
        subject: str,
        run: str,
        condition: Optional[str],
        alignment: str,
        source_method: str,
    ) -> Path:
        """Find the unique source-estimate manifest for one run."""
        run_number, inferred = parse_run_label(run)
        condition = condition or inferred
        expected = self.source(
            subject=subject,
            run=run_number,
            condition=condition,
            description=f"{alignment}-{source_method}",
            suffix="stcmanifest",
            extension=".tsv",
        )
        if expected.is_file():
            return expected

        condition_part = f"{condition.lower()}-" if condition else ""
        pattern = (
            f"**/sub-{subject}_task-{self.task}_run-{run_number}_"
            f"desc-{condition_part}{alignment}-{source_method}_stcmanifest.tsv"
        )
        matches = sorted(path for path in self.root.glob(pattern) if path.is_file())
        if not matches:
            raise FileNotFoundError(
                "No source-estimate manifest found for "
                f"subject={subject}, run={run_number}, condition={condition}, "
                f"alignment={alignment}, method={source_method}. Expected: {expected}"
            )
        if len(matches) > 1:
            raise ValueError(f"Multiple source-estimate manifests matched: {matches}")
        return matches[0]

    def raw_staging_manifest(self) -> Path:
        """Path to the group-level Stage 0 raw-BIDSification manifest.

        One manifest covers every subject a ``meg-tokens meg stage-raw``
        run was asked to plan; re-running the plan overwrites it, matching
        the other group-level manifests in this layout.
        """
        return self.path(
            subject="group",
            datatype="meg",
            description="rawstaging",
            suffix="manifest",
            extension=".tsv",
        )

    def erp(
        self,
        *,
        subject: str,
        run: str,
        condition: Optional[str],
        alignment: str,
        source_method: str,
        parc: str,
        suffix: str,
        extension: str,
    ) -> Path:
        run_number, inferred = parse_run_label(run)
        return self.path(
            subject=subject,
            run=run_number,
            description=_description(_condition(condition or inferred), alignment, source_method, parc),
            suffix=suffix,
            extension=extension,
        )

    def find_erp(
        self,
        *,
        subject: str,
        condition: str,
        alignment: str,
        source_method: str,
        parc: str,
        runs: Optional[Sequence[str]] = None,
    ) -> list[Path]:
        """Find ERP arrays for one subject and condition."""
        if runs:
            candidates = [
                self.erp(
                    subject=subject,
                    run=run,
                    condition=condition,
                    alignment=alignment,
                    source_method=source_method,
                    parc=parc,
                    suffix="erp",
                    extension=".npy",
                )
                for run in runs
            ]
            existing = [path for path in candidates if path.is_file()]
        else:
            pattern = (
                f"**/sub-{subject}_task-{self.task}_run-*_"
                f"desc-{condition.lower()}-{alignment}-{source_method}-{parc}_erp.npy"
            )
            existing = sorted(path for path in self.root.glob(pattern) if path.is_file())
        if not existing:
            detail = f", runs={list(runs)}" if runs else ""
            raise FileNotFoundError(
                "No ERP derivatives found for "
                f"subject={subject}, condition={condition}, alignment={alignment}, "
                f"source_method={source_method}, parc={parc}{detail}"
            )
        return sorted(existing)

    def discover_erp_subjects(
        self,
        *,
        conditions: Sequence[str],
        alignment: str,
        source_method: str,
        parc: str,
    ) -> list[str]:
        """Find subjects with ERP arrays for every requested condition."""
        subject_sets = []
        for condition in conditions:
            pattern = (
                f"**/sub-*_task-{self.task}_run-*_"
                f"desc-{condition.lower()}-{alignment}-{source_method}-{parc}_erp.npy"
            )
            subject_sets.append(
                {
                    path.name.split("_", 1)[0].replace("sub-", "")
                    for path in self.root.glob(pattern)
                    if path.is_file()
                }
            )
        common = set.intersection(*subject_sets) if subject_sets else set()
        if not common:
            raise FileNotFoundError(
                f"No subjects have ERP derivatives for all requested conditions: {conditions}"
            )
        return sorted(common)

    def power(
        self,
        *,
        subject: str,
        run: str,
        condition: Optional[str],
        alignment: str,
        source_method: str,
        power_method: str,
        band: str,
    ) -> Path:
        run_number, inferred = parse_run_label(run)
        return self.path(
            subject=subject,
            run=run_number,
            description=_description(
                _condition(condition or inferred),
                alignment,
                source_method,
                power_method,
                band.replace("_", "-"),
            ),
            suffix="power",
            extension=".npy",
        )

    def psd(
        self,
        *,
        subject: str,
        run: str,
        condition: Optional[str],
        alignment: str,
        method: str,
        fmin: float,
        fmax: float,
        suffix: str,
        extension: str,
    ) -> Path:
        run_number, inferred = parse_run_label(run)
        return self.path(
            subject=subject,
            run=run_number,
            description=_description(
                _condition(condition or inferred),
                alignment,
                method,
                f"{_frequency(fmin)}to{_frequency(fmax)}hz",
            ),
            suffix=suffix,
            extension=extension,
        )

    def group_stats(
        self,
        *,
        conditions: Sequence[str],
        alignment: str,
        source_method: str,
        parc: str,
        suffix: str,
        extension: str,
    ) -> Path:
        return self.path(
            subject="group",
            description=_description(
                conditions[0].lower(),
                "vs",
                conditions[1].lower(),
                alignment,
                source_method,
                parc,
            ),
            suffix=suffix,
            extension=extension,
        )

    def lateralized_stats(
        self,
        *,
        condition: str,
        alignment: str,
        source_method: str,
        parc: str,
        suffix: str,
        extension: str,
    ) -> Path:
        return self.path(
            subject="group",
            description=_description(
                condition.lower(),
                "lateralized",
                alignment,
                source_method,
                parc,
            ),
            suffix=suffix,
            extension=extension,
        )

    def decoding(
        self,
        *,
        conditions: Sequence[str],
        feature_source: str,
        alignment: str,
        source_method: str,
        parc: Optional[str],
        band: Optional[str],
        suffix: str,
        extension: str,
        roi: Optional[str] = None,
        lateralized: bool = False,
    ) -> Path:
        condition_parts = [conditions[0].lower(), "vs", *[item.lower() for item in conditions[1:]]]
        return self.path(
            subject="group",
            description=_description(
                *condition_parts,
                feature_source,
                alignment,
                source_method,
                parc,
                band.replace("_", "-") if band else None,
                roi,
                "lateralized" if lateralized else None,
            ),
            suffix=suffix,
            extension=extension,
        )

    def pca(
        self,
        *,
        conditions: Sequence[str],
        feature_source: str,
        alignment: str,
        source_method: str,
        parc: Optional[str],
        band: Optional[str],
        suffix: str,
        extension: str,
        labels: Optional[Sequence[str]] = None,
        lateralized: bool = False,
    ) -> Path:
        contrast = "-".join(
            [conditions[0].lower(), "vs", *[condition.lower() for condition in conditions[1:]]]
        )
        selection = "lateralized" if lateralized else None
        if labels and not lateralized:
            selection = str(labels[0]) if len(labels) == 1 else f"{len(labels)}labels"
        return self.path(
            subject="group",
            description=_description(
                contrast,
                feature_source,
                alignment,
                source_method,
                parc,
                band.replace("_", "-") if band else None,
                selection,
                "pca",
            ),
            suffix=suffix,
            extension=extension,
        )

    def connectivity(
        self,
        *,
        subject: str,
        run: str,
        condition: str,
        alignment: str,
        source_method: str,
        parc: str,
        method: str,
        suffix: str = "connectivity",
        extension: str = ".npy",
    ) -> Path:
        run_number, _ = parse_run_label(run)
        return self.path(
            subject=subject,
            run=run_number,
            description=_description(condition.lower(), alignment, source_method, parc, method),
            suffix=suffix,
            extension=extension,
        )

    def hilbert_feature(
        self,
        *,
        subject: str,
        run: str,
        condition: str,
        alignment: str,
        source_method: str,
        parc: str,
        band: str,
        feature: str,
    ) -> Path:
        run_number, _ = parse_run_label(run)
        return self.path(
            subject=subject,
            run=run_number,
            description=_description(
                condition.lower(),
                alignment,
                source_method,
                parc,
                band.replace("_", "-"),
                feature,
            ),
            suffix="hilbertfeature",
            extension=".npy",
        )

    def pac(
        self,
        *,
        subject: str,
        run: str,
        condition: str,
        alignment: str,
        source_method: str,
        parc: str,
        phase_bands: Sequence[str],
        amplitude_bands: Sequence[str],
        method: str,
    ) -> Path:
        run_number, _ = parse_run_label(run)
        phase = "+".join(item.replace("_", "-") for item in phase_bands)
        amplitude = "+".join(item.replace("_", "-") for item in amplitude_bands)
        return self.path(
            subject=subject,
            run=run_number,
            description=_description(
                condition.lower(),
                alignment,
                source_method,
                parc,
                phase,
                "to",
                amplitude,
                method.replace("_", "-"),
            ),
            suffix="pac",
            extension=".npy",
        )
