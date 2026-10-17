"""Single-trial band power in source space for one subject: every Fast and Slow
run, per stream and lock, in the HCP-MMP1 regions and deep structures and, if
asked, on ``fsaverage``."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import h5py
import mne
import numpy as np
import pandas as pd

from meg_tokens.core import STREAMS, ProjectConfig, WorkflowResult
from meg_tokens.io import DerivativeLayout, require_file, save_sidecar
from meg_tokens.meg.band_power import (
    BANDS,
    INVALID_NEAR,
    LOCKS,
    PARCELLATION,
    TRANSITIONS,
    analytic,
    deep_labels,
    fsaverage_morph,
    half_length,
    hcp_labels,
    invalid_samples,
    trial_power,
    window,
)
from meg_tokens.meg.epoching import WINDOWS
from meg_tokens.meg.meg_bids import RAW_MEG_RUNS, epoched_meg_runs
from meg_tokens.meg.preprocessing import stream_paths
from meg_tokens.meg.sources import LAMBDA2, METHOD, VOLUME_LABELS, deep_structures


def _subjects_dir(project: ProjectConfig) -> Path:
    if project.subjects_dir is None:
        raise ValueError("Project configuration requires subjects_dir for band power")
    return Path(project.subjects_dir)


def _create(path: Path, n_trials: int, names: Sequence[str], times: np.ndarray, record: dict) -> h5py.File:
    """An HDF5 file holding trials x bands x sources x times of power, chunked one trial and band at a time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = h5py.File(path, "w")
    shape = (n_trials, len(BANDS), len(names), times.size)
    chunk_sources = min(len(names), 512)
    handle.create_dataset("power", shape=shape, dtype="float32", chunks=(1, 1, chunk_sources, times.size), fillvalue=np.nan)
    handle.create_dataset("times", data=times)
    handle.create_dataset("bands", data=np.array(list(BANDS), dtype="S"))
    handle.create_dataset("band_edges_hz", data=np.array(list(BANDS.values())))
    handle.create_dataset("sources", data=np.array([str(name) for name in names], dtype="S"))
    for key, value in record.items():
        handle.attrs[key] = value
    return handle


def band_power_subject(
    project: ProjectConfig,
    subject: str,
    *,
    streams: Sequence[str] = STREAMS,
    locks: Sequence[str] = LOCKS,
    fsaverage_root: Optional[str | Path] = None,
    output_root: Optional[str | Path] = None,
) -> WorkflowResult:
    """Band power of every kept trial of one subject, per stream and lock.

    Writes, per stream and lock, one file of trials x bands x (360 regions + 17
    deep structures) x times on the derivatives, one table of its trials, and,
    with ``fsaverage_root``, one file of trials x bands x 8,196 ``fsaverage``
    sources x times there. Power is linear and at the recording's rate; a
    sample whose filter reaches a flux jump or the recording's end is NaN.
    """
    layout = DerivativeLayout(project.bids_root, task=project.task)
    destination = DerivativeLayout(output_root or project.bids_root, task=project.task)
    fsaverage_layout = None if fsaverage_root is None else DerivativeLayout(fsaverage_root, task=project.task)
    subjects_dir = _subjects_dir(project)
    inputs, outputs = [], []

    for stream in streams:
        epoched = {lock: dict(epoched_meg_runs(layout, subject, alignment=lock, stream=stream)) for lock in locks}
        order = [(condition, str(number)) for condition, number in RAW_MEG_RUNS]
        runs = sorted({run for lock in locks for run in epoched[lock]}, key=lambda run: order.index((run.condition, run.run)))
        if not runs:
            continue
        epochs = {
            lock: {run: mne.read_epochs(path, preload=False, verbose=False) for run, path in epoched[lock].items()}
            for lock in locks
        }
        trials = {
            lock: pd.concat([
                ep.metadata.assign(subject=run.subject, condition=run.condition, run=run.run, epoch=ep.selection)
                for run, ep in epochs[lock].items()
            ], ignore_index=True)
            for lock in locks
        }
        offsets = {
            lock: dict(zip(epochs[lock], np.cumsum([0] + [len(ep) for ep in epochs[lock].values()])))
            for lock in locks
        }

        files, geometry = {}, None
        for run in runs:
            paths = stream_paths(layout, run.subject, run.condition, run.run, stream)
            raw_path = require_file(paths["cleaned"], purpose=f"{run.subject} {run.condition} {run.run} {stream} cleaned run")
            inv_path = require_file(
                layout.inverse(subject=run.subject, run=run.run, condition=run.condition, stream=stream),
                purpose=f"{run.subject} {run.condition} {run.run} {stream} inverse; run `meg source` first",
            )
            inputs.extend([raw_path, inv_path, *(epoched[lock][run] for lock in locks if run in epoched[lock])])
            raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
            inv = mne.minimum_norm.read_inverse_operator(inv_path, verbose=False)
            src = inv["src"]
            vertices = [hemi["vertno"] for hemi in src]

            if geometry is None:
                anatomy = src[0]["subject_his_id"]
                labels = hcp_labels(anatomy, subjects_dir)
                deep = deep_labels(anatomy, subjects_dir)
                morph = None if fsaverage_layout is None else fsaverage_morph(src, subjects_dir)
                counts = np.r_[
                    [len(label.restrict(src[:2]).vertices) for label in labels],
                    [where.size for where in deep_structures(src, subjects_dir, mri_head_t=inv["mri_head_t"]).values()],
                ]
                geometry = vertices
                sfreq = raw.info["sfreq"]
                source_names = [label.name for label in labels] + list(VOLUME_LABELS)
                record = {
                    "subject": run.subject, "stream": stream, "method": METHOD, "lambda2": LAMBDA2,
                    "parcellation": PARCELLATION, "sfreq": sfreq, "power": "linear |z|^2",
                }
                for lock in locks:
                    times = window(sfreq, lock) / sfreq
                    region_path = destination.band_power(subject=subject, stream=stream, lock=lock, space=PARCELLATION)
                    files[lock, "regions"] = _create(region_path, len(trials[lock]), source_names, times, {**record, "lock": lock})
                    files[lock, "regions"].create_dataset("n_sources", data=counts)
                    if morph is not None:
                        fsaverage_path = fsaverage_layout.band_power(subject=subject, stream=stream, lock=lock, space="fsaverage")
                        n_fsaverage = sum(len(v) for v in morph.vertices_to)
                        files[lock, "fsaverage"] = _create(
                            fsaverage_path, len(trials[lock]), [str(index) for index in range(n_fsaverage)],
                            times, {**record, "lock": lock},
                        )
            elif any(not np.array_equal(a, b) for a, b in zip(vertices, geometry)):
                raise RuntimeError(f"{run.subject} {run.condition} {run.run}: source space differs from the subject's first run")

            prepared = mne.minimum_norm.prepare_inverse_operator(inv, 1, LAMBDA2, METHOD, verbose=False)
            for band_index, band in enumerate(BANDS):
                signal = analytic(raw, band)
                invalid = invalid_samples(raw, half_length(sfreq, band))
                for lock in locks:
                    if run not in epochs[lock]:
                        continue
                    kept = epochs[lock][run]
                    trial_epochs = mne.Epochs(
                        signal, kept.events, kept.event_id, *WINDOWS[lock], baseline=None,
                        reject_by_annotation=False, preload=False, verbose=False,
                    )
                    bad = invalid[kept.events[:, 0, None] - raw.first_samp + window(sfreq, lock)]
                    start = offsets[lock][run]
                    for index, (power, average) in enumerate(trial_power(trial_epochs, prepared, labels, deep, morph)):
                        files[lock, "regions"]["power"][start + index, band_index] = np.where(bad[index], np.nan, power)
                        if average is not None:
                            files[lock, "fsaverage"]["power"][start + index, band_index] = np.where(bad[index], np.nan, average)

        for (lock, space), handle in files.items():
            path = Path(handle.filename)
            handle.close()
            outputs.append(path)
            save_sidecar(path, {
                "subject": subject, "stream": stream, "lock": lock, "space": space,
                "analysis_window_s": list(WINDOWS[lock]), "method": METHOD, "lambda2": LAMBDA2,
                "orientation": {"cortex": "normal", "deep": "sum of x, y, z power"},
                "bands_hz": BANDS, "transitions_hz": TRANSITIONS, "invalid_near": list(INVALID_NEAR),
                "n_trials": int(len(trials[lock])),
                "empty_sources": [] if space == "fsaverage" else [
                    name for name, count in zip(source_names, counts) if count == 0
                ],
            })
        for lock in locks:
            table_path = destination.band_power_trials(subject=subject, stream=stream, lock=lock)
            trials[lock].to_csv(table_path, sep="\t", index=False, na_rep="n/a")
            outputs.append(table_path)

    return WorkflowResult(
        stage="band_power",
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        settings={"subject": subject, "streams": list(streams), "locks": list(locks),
                  "fsaverage": fsaverage_root is not None},
    )
