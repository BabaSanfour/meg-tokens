"""Source reconstruction over the cleaned recordings: per-subject models, then
per-run forward and per-stream inverse operators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import mne

from meg_tokens.core import STREAMS, ProjectConfig, RunSpec, WorkflowResult
from meg_tokens.io import DerivativeLayout, require_file, save_sidecar
from meg_tokens.meg.epoching import EPOCH_CONDITIONS
from meg_tokens.meg.meg_bids import nearest_emptyroom, raw_meg_sessions
from meg_tokens.meg.preprocessing import stream_paths
from meg_tokens.meg.sources import (
    DEPTH,
    LOOSE,
    SPACING,
    cleaning_operator,
    forward,
    head_model,
    inverse,
    noise_covariance,
)


def _subjects_dir(project: ProjectConfig) -> Path:
    if project.subjects_dir is None:
        raise ValueError("Project configuration requires subjects_dir for source reconstruction")
    return Path(project.subjects_dir)


def source_models(
    project: ProjectConfig,
    *,
    subjects: Sequence[str],
    output_root: Optional[str | Path] = None,
) -> WorkflowResult:
    """Each subject's noise covariance, BEM solution and source space."""
    layout = DerivativeLayout(output_root or project.bids_root, task=project.task)
    subjects_dir = _subjects_dir(project)
    inputs: list[Path] = []
    outputs: list[Path] = []
    for subject in subjects:
        paths = layout.source_models(subject=subject)
        paths["cov"].parent.mkdir(parents=True, exist_ok=True)

        emptyroom, hours = nearest_emptyroom(project.bids_root, subject)
        cov, record = noise_covariance(emptyroom)
        cov.save(paths["cov"], overwrite=True, verbose=False)
        save_sidecar(paths["cov"], {**record, "hours_from_recording": round(hours, 1)})

        bem, src, bem_record = head_model(subject, subjects_dir)
        mne.write_bem_solution(paths["bem"], bem, overwrite=True, verbose=False)
        src.save(paths["src"], overwrite=True, verbose=False)
        save_sidecar(paths["bem"], bem_record)
        save_sidecar(paths["src"], {"spacing": SPACING, "n_sources": int(sum(s["nuse"] for s in src))})

        inputs.append(emptyroom)
        outputs.extend(paths.values())
    return WorkflowResult(
        stage="source_models", inputs=tuple(inputs), outputs=tuple(outputs),
        settings={"subjects": list(subjects)},
    )


def source_operators(
    project: ProjectConfig,
    run: RunSpec,
    *,
    streams: Sequence[str] = STREAMS,
    output_root: Optional[str | Path] = None,
) -> WorkflowResult:
    """One recording's forward model and, per stream, its inverse operator."""
    layout = DerivativeLayout(project.bids_root, task=project.task)
    destination = DerivativeLayout(output_root or project.bids_root, task=project.task)
    models = {
        kind: require_file(path, purpose=f"{run.subject} {kind}; run `meg source-models` first")
        for kind, path in destination.source_models(subject=run.subject).items()
    }
    trans = require_file(
        layout.trans(
            subject=run.subject, run=run.run, condition=run.condition,
        ),
        purpose=f"{run.subject} {run.condition} {run.run} transform; run `meg coreg` first",
    )
    recordings = {
        stream: {
            kind: require_file(path, purpose=f"{run.subject} {run.condition} {run.run} {stream} {kind}")
            for kind, path in stream_paths(layout, run.subject, run.condition, run.run, stream).items()
        }
        for stream in streams
    }
    entities = dict(subject=run.subject, run=run.run, condition=run.condition)

    # Every stream carries the run's own head position and channels; the
    # forward model is one per recording.
    fwd = forward(
        mne.io.read_info(recordings[streams[0]]["cleaned"], verbose=False), trans,
        mne.read_source_spaces(models["src"], verbose=False), mne.read_bem_solution(models["bem"], verbose=False),
    )
    fwd_path = destination.forward(**entities)
    fwd_path.parent.mkdir(parents=True, exist_ok=True)
    mne.write_forward_solution(fwd_path, fwd, overwrite=True, verbose=False)
    outputs = [fwd_path]

    cov = mne.read_cov(models["cov"], verbose=False)
    for stream, paths in recordings.items():
        raw = mne.io.read_raw_fif(paths["cleaned"], preload=True, verbose=False)
        artifact = mne.io.read_raw_fif(paths["artifact"], preload=True, verbose=False)
        rank = json.loads(paths["sidecar"].read_text(encoding="utf-8"))["rank"]
        cleaning, record = cleaning_operator(raw, artifact, rank)
        operator = inverse(raw.info, fwd, cov, cleaning, rank)
        inv_path = destination.inverse(**entities, stream=stream)
        mne.minimum_norm.write_inverse_operator(inv_path, operator, overwrite=True, verbose=False)
        save_sidecar(inv_path, {**record, "stream": stream, "loose": LOOSE, "depth": DEPTH})
        outputs.append(inv_path)

    return WorkflowResult(
        stage="source_operators",
        inputs=(*models.values(), trans, *(path for paths in recordings.values() for path in paths.values())),
        outputs=tuple(outputs),
        settings={"run": f"{run.subject} {run.condition} {run.run}", "streams": list(streams)},
    )


def source_operator_subjects(
    project: ProjectConfig,
    *,
    subjects: Sequence[str],
    streams: Sequence[str] = STREAMS,
    output_root: Optional[str | Path] = None,
) -> WorkflowResult:
    """Every Fast and Slow recording of the selected subjects."""
    results = [
        source_operators(project, run, streams=streams, output_root=output_root)
        for subject in subjects
        for run, _ in raw_meg_sessions(project.bids_root, subject, task=project.task)
        if run.condition in EPOCH_CONDITIONS
    ]
    return WorkflowResult(
        stage="source_operators",
        inputs=tuple(path for result in results for path in result.inputs),
        outputs=tuple(path for result in results for path in result.outputs),
        settings={"subjects": list(subjects), "streams": list(streams)},
    )
