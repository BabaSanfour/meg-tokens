"""MEG preprocessing and epoching workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import mne
import numpy as np

from meg_tokens.behavior.tables import read_behavior_table
from meg_tokens.core import (
    EpochingConfig,
    PreprocessingConfig,
    ProjectConfig,
    RunSpec,
    WorkflowResult,
    normalize_subject_id,
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
from meg_tokens.meg.preprocessing import (
    load_and_filter_raw,
    run_ica_rejection,
    save_clean_raw,
)


def preprocess_run(
    project: ProjectConfig,
    run: RunSpec,
    *,
    raw_path: str | Path,
    settings: PreprocessingConfig = PreprocessingConfig(),
) -> WorkflowResult:
    """Filter one CTF run, optionally apply ICA, and persist the raw derivative."""
    input_path = Path(raw_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Raw dataset path does not exist: {input_path}")
    notch_freqs = (
        np.asarray(settings.notch_freqs, dtype=float)
        if settings.notch_freqs is not None
        else None
    )
    raw = load_and_filter_raw(
        str(input_path),
        notch_freqs=notch_freqs,
        high_pass=settings.high_pass,
        low_pass=settings.low_pass,
        preload=True,
    )
    if settings.run_ica:
        ica = run_ica_rejection(raw)
        if settings.ica_exclude is not None:
            ica.exclude = [int(index) for index in settings.ica_exclude]
        ica.apply(raw)

    output_path = Path(
        save_clean_raw(
            raw,
            str(project.bids_root),
            subject_id=run.subject,
            run_id=run.run,
            condition=run.condition,
            processing="clean" if settings.run_ica else "filt",
        )
    )
    return WorkflowResult(
        stage="meg_preprocessing",
        inputs=(input_path,),
        outputs=(output_path,),
        settings={
            "subject": run.subject,
            "run": run.run,
            "condition": run.condition,
            "high_pass": settings.high_pass,
            "low_pass": settings.low_pass,
            "notch_freqs": settings.notch_freqs,
            "run_ica": settings.run_ica,
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

    for subject_value in subjects:
        subject = normalize_subject_id(subject_value)
        raw_paths = raw_layout.raw_files(subject=subject)
        if not raw_paths:
            raise FileNotFoundError(
                f"No cleaned or filtered raw FIF files found for {subject} under {raw_layout.root}"
            )
        for raw_path in raw_paths:
            run = raw_layout.raw_run(raw_path, subject=subject)
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
            "subjects": [normalize_subject_id(subject) for subject in subjects],
            "alignment": settings.alignment,
            "tmin": settings.tmin,
            "tmax": settings.tmax,
        },
    )
