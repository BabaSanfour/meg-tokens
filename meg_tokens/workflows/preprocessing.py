"""MEG preprocessing and epoching workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import mne
import pandas as pd

from meg_tokens.behavior.tables import read_behavior_table, read_trial_features
from meg_tokens.core import (
    EpochingConfig,
    PreprocessingConfig,
    ProjectConfig,
    RunSpec,
    WorkflowResult,
)
from meg_tokens.io import DerivativeLayout, load_table, save_table
from meg_tokens.meg.epoching import (
    EPOCH_CONDITIONS,
    OUTLIER_SD,
    align_go,
    amplitudes,
    drop_outliers,
    epoch,
    outlier_thresholds,
    read_triggers,
    save_epochs,
    trial_metadata,
)
from meg_tokens.meg.meg_bids import (
    raw_meg_path,
    raw_meg_sessions,
    staged_subjects,
)
from meg_tokens.meg.preprocessing import (
    HIGH_PASS,
    LOW_PASS,
    STREAMS,
    artifact_events,
    build_trunk,
    read_staged_raw,
    run_stream,
    save_stream,
)


def preprocess_run(
    project: ProjectConfig,
    run: RunSpec,
    *,
    settings: PreprocessingConfig = PreprocessingConfig(),
) -> WorkflowResult:
    """Preprocess one staged CTF recording through every stream.

    The single-run primitive Snakemake parallelizes over. It resolves its own
    input from the run's identity rather than being handed a path, so exactly
    one place knows where Stage 0 puts a recording.
    """
    if not run.condition:
        raise ValueError(
            f"{run.subject} run {run.run} needs a condition: every subject has an "
            "acq-fast run 1 and an acq-slow run 1, so a run number alone names two "
            "different recordings. Pass a labelled run such as 'Fast1', or the "
            "condition explicitly."
        )
    input_path = raw_meg_path(
        project.bids_root, run.subject, run.condition, run.run, task=project.task
    )
    # A CTF session is a directory, so the file-shaped existence helpers reject
    # it out of hand.
    if not input_path.is_dir():
        raise FileNotFoundError(
            f"No staged raw MEG for {run.subject} {run.condition} run {run.run} at "
            f"{input_path}. Run `meg-tokens meg apply-raw-staging` first."
        )
    # The trunk is built once and shared: computing bad channels or segments
    # per stream would make the streams differ in two ways at once, and the
    # comparison between cleaners would not mean anything.
    raw, source_grade = read_staged_raw(input_path)
    trunk, trunk_record = build_trunk(raw, source_grade)
    events = artifact_events(trunk)

    streams = tuple(settings.streams or STREAMS)
    unknown = [name for name in streams if name not in STREAMS]
    if unknown:
        raise ValueError(f"unknown stream(s) {unknown}; expected from {list(STREAMS)}")

    outputs: list[Path] = []
    ranks: dict[str, int] = {}
    for stream in streams:
        cleaned, artifact, stream_record = run_stream(trunk, events, stream)
        paths = save_stream(
            cleaned, artifact, stream_record, trunk_record,
            DerivativeLayout(project.bids_root), run.subject, run.condition, run.run,
        )
        outputs.extend(paths.values())
        ranks[stream] = (
            trunk_record["n_meg_channels"] - stream_record["n_dimensions_removed"]
        )
        del cleaned, artifact

    return WorkflowResult(
        stage="meg_preprocessing",
        inputs=(input_path,),
        outputs=tuple(outputs),
        settings={
            "subject": run.subject,
            "run": run.run,
            "condition": run.condition,
            "streams": list(streams),
            "compensation_applied": trunk_record["compensation_applied"],
            "bad_channels": trunk_record["bad_channels"],
            "annotated_seconds": trunk_record["trunk_annotated_seconds"],
            "rank": ranks,
        },
    )


def preprocess_subjects(
    project: ProjectConfig,
    *,
    subjects: Optional[Sequence[str]] = None,
    settings: PreprocessingConfig = PreprocessingConfig(),
) -> WorkflowResult:
    """Preprocess every staged run of selected subjects, defaulting to all.

    Raises ``FileNotFoundError`` when a requested subject has no staged MEG:
    that is a missing prerequisite, not an empty result, and quietly producing
    nothing would look identical to success.
    """
    bids_root = project.bids_root
    if not bids_root.is_dir():
        raise FileNotFoundError(
            f"BIDS root does not exist: {bids_root}. Run `meg-tokens meg stage-raw` "
            "and `meg-tokens meg apply-raw-staging` before preprocessing."
        )

    selected = list(subjects or staged_subjects(bids_root))
    if not selected:
        raise FileNotFoundError(f"No staged subject MEG was found under {bids_root}")

    inputs: list[Path] = []
    outputs: list[Path] = []
    for subject in selected:
        sessions = raw_meg_sessions(bids_root, subject, task=project.task)
        if not sessions:
            raise FileNotFoundError(
                f"No staged raw MEG for {subject} under {bids_root}. "
                "Run `meg-tokens meg apply-raw-staging` first."
            )
        for run, _ in sessions:
            result = preprocess_run(project, run, settings=settings)
            inputs.extend(result.inputs)
            outputs.extend(result.outputs)

    return WorkflowResult(
        stage="meg_preprocessing",
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        settings={
            "subjects": selected,
            "n_runs": len(outputs),
            "high_pass": HIGH_PASS,
            "low_pass": LOW_PASS,
            "streams": list(settings.streams or STREAMS),
        },
    )


def _cut_epochs(project: ProjectConfig, subject: str, settings: EpochingConfig, inputs: list[Path]):
    """Every stream and lock of one subject's Fast and Slow runs, as the marks leave them.

    The go cues are paired with the behavioral log once per run; every stream
    and lock is cut from that one pairing, so the streams differ only in what
    the cleaners did to the data. Yields the run, stream, lock, epochs and the
    sidecar record.
    """
    layout = DerivativeLayout(project.bids_root, task=project.task)
    features = read_trial_features(layout.behavior_trial_features())
    sessions = raw_meg_sessions(project.bids_root, subject, task=project.task)
    if not sessions:
        raise FileNotFoundError(f"No staged raw MEG for {subject} under {project.bids_root}")
    for run, _ in sessions:
        if run.condition not in EPOCH_CONDITIONS:
            continue
        behavior_path = layout.find_behavior(subject=run.subject, run=run.run, condition=run.condition)
        triggers = read_triggers(project.bids_root, run.subject, run.condition, run.run, task=project.task)
        go, trials, alignment = align_go(
            triggers, read_behavior_table(behavior_path), run.subject, run.condition, run.run
        )
        this_run = features[
            (features["subject"] == run.subject)
            & (features["condition"].str.lower() == run.condition.lower())
            & (features["run"].astype(str) == str(run.run))
        ]
        metadata = trial_metadata(trials, this_run)
        inputs.append(behavior_path)
        for stream in settings.streams:
            raw_path = layout.preprocessed_raw(
                subject=run.subject, run=run.run, condition=run.condition, processing=stream
            )
            raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
            inputs.append(raw_path)
            for lock in settings.locks:
                record = {
                    "subject": run.subject, "condition": run.condition.lower(),
                    "run": str(run.run), "stream": stream, "n_trials": len(trials), **alignment,
                }
                yield run, stream, lock, epoch(raw, go, metadata, lock), record
            del raw


def epoch_amplitude_subjects(
    project: ProjectConfig,
    *,
    subjects: Sequence[str],
    settings: EpochingConfig = EpochingConfig(),
) -> WorkflowResult:
    """Each subject's per-epoch peak-to-peak on every magnetometer, one table per stream.

    The epochs are those the marks leave; the cohort's outlier thresholds are
    computed from these tables, before any epoch is dropped as an outlier.
    """
    layout = DerivativeLayout(project.bids_root, task=project.task)
    inputs: list[Path] = [layout.behavior_trial_features()]
    outputs: list[Path] = []
    for subject in subjects:
        tables: dict[str, list[pd.DataFrame]] = {stream: [] for stream in settings.streams}
        for run, stream, lock, epochs, _ in _cut_epochs(project, subject, settings, inputs):
            table = amplitudes(epochs)
            table.insert(0, "trial", epochs.metadata["nTrialIndex"].to_numpy())
            table.insert(0, "lock", lock)
            table.insert(0, "run", str(run.run))
            table.insert(0, "condition", run.condition.lower())
            tables[stream].append(table)
        for stream, parts in tables.items():
            outputs.append(save_table(
                layout.epoch_amplitudes(subject=subject, stream=stream),
                pd.concat(parts, ignore_index=True),
                metadata={"subject": subject, "stream": stream, "measure": "log10 peak-to-peak (T), analysis window"},
            ))
    return WorkflowResult(
        stage="meg_epoch_amplitudes",
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        settings={"subjects": list(subjects), "locks": list(settings.locks), "streams": list(settings.streams)},
    )


def epoch_thresholds(
    project: ProjectConfig,
    *,
    subjects: Optional[Sequence[str]] = None,
    settings: EpochingConfig = EpochingConfig(),
) -> WorkflowResult:
    """The cohort's outlier threshold per stream, lock and magnetometer.

    Pools every subject's amplitude table; a subject without one raises, since
    thresholds from part of the cohort would quietly differ from the rule.
    """
    layout = DerivativeLayout(project.bids_root, task=project.task)
    selected = list(subjects or staged_subjects(project.bids_root))
    inputs: list[Path] = []
    outputs: list[Path] = []
    for stream in settings.streams:
        paths = [layout.epoch_amplitudes(subject=subject, stream=stream) for subject in selected]
        pooled = pd.concat([load_table(path) for path in paths], ignore_index=True)
        inputs.extend(paths)
        sensors = [column for column in pooled.columns if column not in ("condition", "run", "lock", "trial")]
        rows = []
        for lock in settings.locks:
            this = pooled.loc[pooled["lock"] == lock, sensors]
            if this.empty:
                raise ValueError(f"no {lock} epochs in the {stream} amplitude tables")
            rows.append({"lock": lock, **outlier_thresholds(this).to_dict()})
        outputs.append(save_table(
            layout.epoch_thresholds(stream=stream),
            pd.DataFrame(rows),
            metadata={
                "stream": stream, "subjects": selected, "outlier_sd": OUTLIER_SD,
                "measure": "log10 peak-to-peak (T), analysis window",
                "n_epochs": {lock: int((pooled["lock"] == lock).sum()) for lock in settings.locks},
            },
        ))
    return WorkflowResult(
        stage="meg_epoch_thresholds",
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        settings={"subjects": selected, "locks": list(settings.locks), "streams": list(settings.streams)},
    )


def epoch_subjects(
    project: ProjectConfig,
    *,
    subjects: Sequence[str],
    settings: EpochingConfig = EpochingConfig(),
    output_root: Optional[str | Path] = None,
) -> WorkflowResult:
    """Epoch every staged Fast and Slow run of the selected subjects, per stream and lock.

    The marks drop what they touch; then the cohort's outlier thresholds
    (:func:`epoch_thresholds`) drop what stands out against the other people.
    """
    layout = DerivativeLayout(project.bids_root, task=project.task)
    destination = DerivativeLayout(output_root or project.bids_root, task=project.task)
    threshold_paths = {stream: layout.epoch_thresholds(stream=stream) for stream in settings.streams}
    thresholds = {
        stream: load_table(path).set_index("lock") for stream, path in threshold_paths.items()
    }
    inputs: list[Path] = [layout.behavior_trial_features(), *threshold_paths.values()]
    outputs: list[Path] = []
    for subject in subjects:
        for run, stream, lock, epochs, record in _cut_epochs(project, subject, settings, inputs):
            path = destination.epochs(
                subject=run.subject, run=run.run, condition=run.condition, alignment=lock, stream=stream,
            )
            outputs.append(save_epochs(drop_outliers(epochs, thresholds[stream].loc[lock]), path, record))

    return WorkflowResult(
        stage="meg_epoching",
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        settings={
            "subjects": list(subjects),
            "locks": list(settings.locks),
            "streams": list(settings.streams),
            "outlier_sd": OUTLIER_SD,
        },
    )
