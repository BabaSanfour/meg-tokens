"""MEG preprocessing and epoching workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import mne

from meg_tokens.behavior.tables import read_behavior_table
from meg_tokens.core import (
    EpochingConfig,
    PreprocessingConfig,
    ProjectConfig,
    RunSpec,
    WorkflowResult,
)
from meg_tokens.io import DerivativeLayout
from meg_tokens.meg.epoching import (
    build_epochs_with_metadata,
    exclude_unrecoverable_trials,
    get_event_id,
    mismatch_policy,
    needs_go_reconstruction,
    reconstruct_missing_go_events,
    save_epochs_and_events,
)
from meg_tokens.meg.meg_bids import (
    preprocessed_meg_runs,
    raw_meg_path,
    raw_meg_sessions,
    staged_subjects,
)
from meg_tokens.meg.preprocessing import (
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
    trunk, report = build_trunk(raw, source_grade)
    events = artifact_events(trunk)

    streams = tuple(settings.streams or STREAMS)
    unknown = [name for name in streams if name not in STREAMS]
    if unknown:
        raise ValueError(f"unknown stream(s) {unknown}; expected from {list(STREAMS)}")

    outputs: list[Path] = []
    ranks: dict[str, int] = {}
    for stream in streams:
        result, tail = run_stream(trunk, events, stream)
        paths = save_stream(
            result, report, tail, DerivativeLayout(project.bids_root),
            run.subject, run.condition, run.run,
        )
        outputs.extend(paths.values())
        ranks[stream] = result.rank(report.n_meg_channels)
        del result

    return WorkflowResult(
        stage="meg_preprocessing",
        inputs=(input_path,),
        outputs=tuple(outputs),
        settings={
            "subject": run.subject,
            "run": run.run,
            "condition": run.condition,
            "streams": list(streams),
            "compensation_applied": report.compensation_applied,
            "bad_channels": report.bad_channels,
            "annotated_seconds": report.annotated_seconds,
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
            "high_pass": settings.high_pass,
            "low_pass": settings.low_pass,
            "streams": list(settings.streams or STREAMS),
            "ica_exclude": settings.ica_exclude,
        },
    )


def epoch_subjects(
    project: ProjectConfig,
    *,
    subjects: Sequence[str],
    settings: EpochingConfig = EpochingConfig(),
    raw_root: Optional[str | Path] = None,
    behavior_root: Optional[str | Path] = None,
    output_root: Optional[str | Path] = None,
) -> WorkflowResult:
    """Build event-locked epochs for every staged run of selected subjects."""
    raw_layout = DerivativeLayout(
        raw_root or project.bids_root,
        task=project.task,
    )
    behavior_layout = DerivativeLayout(
        behavior_root or project.bids_root,
        task=project.task,
    )
    destination = Path(output_root or project.bids_root)
    inputs = []
    outputs = []

    for subject in subjects:
        runs = preprocessed_meg_runs(raw_layout, subject, stream=settings.stream)
        if not runs:
            raise FileNotFoundError(
                f"No preprocessed raw FIF files found for {subject} under "
                f"{raw_layout.root}. Run `meg-tokens meg preprocess` first."
            )
        for run, raw_path in runs:
            behavior_path = behavior_layout.find_behavior(
                subject=run.subject,
                run=run.run,
                condition=run.condition,
            )
            raw = mne.io.read_raw_fif(str(raw_path), preload=True)
            events = mne.find_events(raw)
            behavior = read_behavior_table(behavior_path)
            event_id = get_event_id(settings.alignment, run.subject)
            if settings.alignment.lower() == "go" and needs_go_reconstruction(
                run.subject, run.condition, run.run
            ):
                print(
                    f"{run.subject} {run.condition}{run.run}: reconstructing "
                    "missing go-cue event(s) from trial-start pulses + tGO "
                    "(confirmed trigger dropout -- see "
                    "docs/meg.md)"
                )
                events = reconstruct_missing_go_events(
                    events,
                    behavior,
                    raw.info['sfreq'],
                    start_code=get_event_id('start', run.subject)['Start'],
                    go_code=event_id['Go'],
                )
            before_exclusion = len(behavior)
            behavior = exclude_unrecoverable_trials(
                run.subject, run.condition, run.run, behavior
            )
            if len(behavior) != before_exclusion:
                print(
                    f"{run.subject} {run.condition}{run.run}: excluding "
                    f"{before_exclusion - len(behavior)} trial(s) with no "
                    "MEG event at all for this run (confirmed unrecoverable "
                    "-- see docs/meg.md)"
                )
            policy = mismatch_policy(run.subject, run.condition, run.run)
            if policy == "truncate":
                print(
                    f"{run.subject} {run.condition}{run.run}: applying known "
                    "trailing-trial truncation (aborted final trial, "
                    "confirmed via MEG timing cross-check -- see "
                    "docs/meg.md)"
                )
            epochs = build_epochs_with_metadata(
                raw=raw,
                events=events,
                event_ids=event_id,
                tmin=settings.tmin,
                tmax=settings.tmax,
                behavior_df=behavior,
                on_mismatch=policy,
            )
            saved = save_epochs_and_events(
                epochs,
                str(destination),
                run.subject,
                run.run,
                bids_format=True,
                condition=run.condition,
                align_to=settings.alignment,
            )
            inputs.extend((raw_path, behavior_path))
            outputs.extend(Path(path) for path in saved.values())

    return WorkflowResult(
        stage="meg_epoching",
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        settings={
            "subjects": list(subjects),
            "alignment": settings.alignment,
            "tmin": settings.tmin,
            "tmax": settings.tmax,
        },
    )
