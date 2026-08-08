"""Central construction of persisted derivative paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from meg_tokens.core import RunSpec, normalize_subject_id, parse_run_label

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
        subject = normalize_subject_id(subject)
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

    def raw_files(self, *, subject: str) -> list[Path]:
        """Find cleaned or filtered raw derivatives for one subject."""
        subject = normalize_subject_id(subject)
        patterns = [
            (
                self.root
                / "derivatives"
                / f"sub-{subject}"
                / "meg"
                / "*_raw.fif"
            ),
            self.root / f"sub-{subject}" / "meg" / "*_raw.fif",
            self.root / subject / "*_filt_raw.fif",
            self.root / subject / "*_clean_raw.fif",
        ]
        files = []
        for pattern in patterns:
            files.extend(pattern.parent.glob(pattern.name))
        return sorted({path for path in files if path.is_file()})

    @staticmethod
    def raw_run(path: str | Path, *, subject: Optional[str] = None) -> RunSpec:
        """Parse subject, run, and condition from a raw derivative filename."""
        import re

        name = Path(path).name
        bids = re.search(
            r"sub-(?P<subject>H[0-9]+)_.*_run-(?P<run>[A-Za-z0-9]+)"
            r"(?:_proc-[^_]+)?(?:_desc-(?P<description>[^_]+))?_raw\.fif$",
            name,
            re.IGNORECASE,
        )
        if bids:
            description = bids.group("description")
            condition = None
            if description:
                first = description.split("-", 1)[0]
                if first.lower() in {"fast", "slow", "rt"}:
                    condition = first
            return RunSpec(
                subject=bids.group("subject"),
                run=bids.group("run"),
                condition=condition,
            )

        legacy = re.search(
            r"(?P<subject>H[0-9]+)?_?(?P<condition>Slow|Fast|RT)?"
            r"(?P<run>[0-9]+)_?(?:filt|clean)?_?raw\.fif$",
            name,
            re.IGNORECASE,
        )
        if legacy:
            resolved_subject = legacy.group("subject") or subject
            if resolved_subject is None:
                raise ValueError(f"Could not infer subject from raw filename: {path}")
            return RunSpec(
                subject=resolved_subject,
                run=legacy.group("run"),
                condition=legacy.group("condition"),
            )
        raise ValueError(f"Could not infer run from raw filename: {path}")

    def epochs(
        self,
        *,
        subject: str,
        run: str,
        condition: Optional[str],
        alignment: str,
        suffix: str = "epo",
        extension: str = ".fif",
    ) -> Path:
        run_number, inferred = parse_run_label(run)
        return self.path(
            subject=subject,
            run=run_number,
            description=_description(_condition(condition or inferred), alignment),
            suffix=suffix,
            extension=extension,
        )

    def find_epochs(
        self,
        *,
        subject: str,
        run: str,
        condition: Optional[str],
        alignment: str,
    ) -> Path:
        """Find the unique epoch derivative for one run and alignment."""
        subject = normalize_subject_id(subject)
        run_number, inferred = parse_run_label(run)
        condition = condition or inferred
        expected = self.epochs(
            subject=subject,
            run=run_number,
            condition=condition,
            alignment=alignment,
        )
        candidates = [expected]
        if not expected.is_file():
            candidates.extend(
                self.root.glob(
                    f"**/sub-{subject}_task-{self.task}_run-{run_number}_"
                    f"desc-*{alignment}*_epo.fif"
                )
            )
        existing = sorted({path for path in candidates if path.is_file()})
        if not existing:
            raise FileNotFoundError(
                f"No epochs found for subject={subject}, run={run_number}, "
                f"condition={condition}, alignment={alignment} under {self.root}"
            )
        if len(existing) > 1:
            raise ValueError(
                f"Multiple epoch files matched subject={subject}, run={run_number}: {existing}"
            )
        return existing[0]

    def find_noise(self, *, subject: str) -> Path:
        """Find one empty-room recording for a subject."""
        subject = normalize_subject_id(subject)
        patterns = [
            self.root / subject / "*noise*.ds",
            self.root / subject / "*noise*.fif",
            self.root / f"sub-{subject}" / "meg" / "*noise*.fif",
        ]
        matches = []
        for pattern in patterns:
            matches.extend(path for path in pattern.parent.glob(pattern.name) if path.exists())
        existing = sorted(set(matches))
        if not existing:
            raise FileNotFoundError(
                f"No empty-room noise file found for {subject} under {self.root}"
            )
        if len(existing) > 1:
            raise ValueError(
                f"Multiple empty-room noise files found for {subject}: {existing}"
            )
        return existing[0]

    def find_trans(self, *, subject: str, run: Optional[str] = None) -> Path:
        """Find one MEG-MRI transform for a subject and optional run."""
        subject = normalize_subject_id(subject)
        if run is not None:
            run_number, _ = parse_run_label(run)
            run_matches = sorted(
                {
                    path
                    for path in self.root.glob(
                        f"**/sub-{subject}*run-{run_number}*trans.fif"
                    )
                    if path.is_file()
                }
            )
            if len(run_matches) == 1:
                return run_matches[0]
            if len(run_matches) > 1:
                raise ValueError(
                    f"Multiple run-specific trans files matched {subject}, "
                    f"run={run_number}: {run_matches}"
                )

        matches = {
            path
            for pattern in (
                f"**/sub-{subject}*trans.fif",
                f"**/{subject}*trans.fif",
            )
            for path in self.root.glob(pattern)
            if path.is_file()
        }
        existing = sorted(matches)
        if not existing:
            raise FileNotFoundError(
                f"No MEG-MRI trans file found for {subject} under {self.root}"
            )
        if len(existing) > 1:
            raise ValueError(f"Multiple trans files matched {subject}: {existing}")
        return existing[0]

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

    def source_models(
        self,
        *,
        subject: str,
        spacing: str,
        mixed: bool = False,
    ) -> dict[str, Path]:
        """Return subject-level covariance, BEM, and source-space paths."""
        source_description = f"{spacing}-mixed" if mixed else spacing
        return {
            "cov": self.source(
                subject=subject,
                suffix="cov",
                extension=".fif",
                description="noise",
            ),
            "bem": self.source(
                subject=subject,
                suffix="bem",
                extension=".fif",
                description="singlelayer",
            ),
            "src": self.source(
                subject=subject,
                suffix="src",
                extension=".fif",
                description=source_description,
                space="subject",
            ),
        }

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
        subject = normalize_subject_id(subject)
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
        subject = normalize_subject_id(subject)
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
