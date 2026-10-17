"""Single-trial band power in source space, in two steps so the work spreads
over many jobs: each run, stream and band computed on its own (every lock),
then a subject's parts assembled per stream and lock."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import h5py
import mne
import numpy as np
import pandas as pd

from meg_tokens.core import ProjectConfig, RunSpec, WorkflowResult
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
from meg_tokens.meg.meg_bids import epoched_meg_runs
from meg_tokens.meg.preprocessing import stream_paths
from meg_tokens.meg.sources import LAMBDA2, METHOD, VOLUME_LABELS, deep_structures


def _subjects_dir(project: ProjectConfig) -> Path:
    if project.subjects_dir is None:
        raise ValueError("Project configuration requires subjects_dir for band power")
    return Path(project.subjects_dir)


def band_power_part(
    project: ProjectConfig,
    run: RunSpec,
    *,
    stream: str,
    band: str,
    locks: Sequence[str] = LOCKS,
    fsaverage_root: Optional[str | Path] = None,
    output_root: Optional[str | Path] = None,
    n_jobs: int = 1,
) -> WorkflowResult:
    """One run's band power in one band, for every lock with kept epochs.

    Writes one HDF5 file holding, per lock, trials x (360 regions + 17 deep
    structures) x times of linear power at the recording's rate, in the order
    of the lock's epoch file; with ``fsaverage_root``, a second file there with
    trials x 8,196 ``fsaverage`` sources x times. A sample whose filter reaches
    a flux jump or the recording's end is NaN.
    """
    layout = DerivativeLayout(project.bids_root, task=project.task)
    destination = DerivativeLayout(output_root or project.bids_root, task=project.task)
    subjects_dir = _subjects_dir(project)
    entities = dict(subject=run.subject, run=run.run, condition=run.condition, stream=stream)

    raw_path = require_file(
        stream_paths(layout, run.subject, run.condition, run.run, stream)["cleaned"],
        purpose=f"{run.subject} {run.condition} {run.run} {stream} cleaned run",
    )
    inv_path = require_file(
        layout.inverse(**entities),
        purpose=f"{run.subject} {run.condition} {run.run} {stream} inverse; run `meg source` first",
    )
    kept = {}
    for lock in locks:
        path = layout.epochs(subject=run.subject, run=run.run, condition=run.condition, alignment=lock, stream=stream)
        if path.is_file():
            epochs = mne.read_epochs(path, preload=False, verbose=False)
            if len(epochs):
                kept[lock] = epochs

    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
    inv = mne.minimum_norm.read_inverse_operator(inv_path, verbose=False)
    src = inv["src"]
    anatomy = src[0]["subject_his_id"]
    labels = hcp_labels(anatomy, subjects_dir)
    deep = deep_labels(anatomy, subjects_dir)
    morph = None if fsaverage_root is None else fsaverage_morph(src, subjects_dir)
    counts = np.r_[
        [len(label.restrict(src[:2]).vertices) for label in labels],
        [where.size for where in deep_structures(src, subjects_dir, mri_head_t=inv["mri_head_t"]).values()],
    ]
    sfreq = raw.info["sfreq"]
    signal = analytic(raw, band, n_jobs=n_jobs)
    invalid = invalid_samples(raw, half_length(sfreq, band))
    prepared = mne.minimum_norm.prepare_inverse_operator(inv, 1, LAMBDA2, METHOD, verbose=False)

    outputs = [destination.band_power_part(**entities, band=band, space=PARCELLATION)]
    outputs[0].parent.mkdir(parents=True, exist_ok=True)
    handles = {"regions": h5py.File(outputs[0], "w")}
    if morph is not None:
        outputs.append(DerivativeLayout(fsaverage_root, task=project.task).band_power_part(
            **entities, band=band, space="fsaverage"))
        outputs[-1].parent.mkdir(parents=True, exist_ok=True)
        handles["fsaverage"] = h5py.File(outputs[-1], "w")
    record = {
        "subject": run.subject, "run": run.run, "condition": run.condition, "stream": stream, "band": band,
        "method": METHOD, "lambda2": LAMBDA2, "parcellation": PARCELLATION, "sfreq": sfreq,
    }
    for handle in handles.values():
        handle.attrs.update(record)
    handles["regions"].create_dataset("sources", data=np.array([label.name for label in labels] + list(VOLUME_LABELS), dtype="S"))
    handles["regions"].create_dataset("n_sources", data=counts)
    widths = {"regions": counts.size}
    if morph is not None:
        widths["fsaverage"] = sum(len(vertices) for vertices in morph.vertices_to)

    for lock, epochs in kept.items():
        offsets = window(sfreq, lock)
        trials = mne.Epochs(
            signal, epochs.events, epochs.event_id, *WINDOWS[lock], baseline=None,
            reject_by_annotation=False, preload=False, verbose=False,
        )
        bad = invalid[epochs.events[:, 0, None] - raw.first_samp + offsets]
        power = {
            space: handles[space].create_dataset(
                f"{lock}/power", shape=(len(epochs), width, offsets.size), dtype="float32",
                chunks=(1, min(width, 512), offsets.size),
            )
            for space, width in widths.items()
        }
        for handle in handles.values():
            handle.create_dataset(f"{lock}/times", data=offsets / sfreq)
        for index, (regions, average) in enumerate(trial_power(trials, prepared, labels, deep, morph)):
            power["regions"][index] = np.where(bad[index], np.nan, regions)
            if average is not None:
                power["fsaverage"][index] = np.where(bad[index], np.nan, average)

    for handle in handles.values():
        handle.close()
    return WorkflowResult(
        stage="band_power_part",
        inputs=(raw_path, inv_path),
        outputs=tuple(outputs),
        settings={"run": f"{run.subject} {run.condition} {run.run}", "stream": stream, "band": band,
                  "locks": list(kept)},
    )


def band_power_merge(
    project: ProjectConfig,
    subject: str,
    *,
    stream: str,
    locks: Sequence[str] = LOCKS,
    output_root: Optional[str | Path] = None,
) -> WorkflowResult:
    """A subject's band power per lock, from its parts: every Fast and Slow run, every band.

    Writes, per lock, one HDF5 file of trials x bands x (360 regions + 17 deep
    structures) x times, its sidecar, and one table of its trials, row for row.
    """
    layout = DerivativeLayout(project.bids_root, task=project.task)
    destination = DerivativeLayout(output_root or project.bids_root, task=project.task)
    inputs, outputs = [], []
    for lock in locks:
        epochs = {
            run: mne.read_epochs(path, preload=False, verbose=False)
            for run, path in epoched_meg_runs(layout, subject, alignment=lock, stream=stream)
        }
        trials = pd.concat([
            ep.metadata.assign(subject=run.subject, condition=run.condition, run=run.run, epoch=ep.selection)
            for run, ep in epochs.items()
        ], ignore_index=True)
        parts = {
            (run, band): require_file(
                destination.band_power_part(subject=subject, run=run.run, condition=run.condition, stream=stream,
                                            band=band, space=PARCELLATION),
                purpose=f"{subject} {run.condition} {run.run} {stream} {band} band power; run `meg band-power` first",
            )
            for run in epochs for band in BANDS
        }
        inputs.extend(parts.values())
        with h5py.File(next(iter(parts.values())), "r") as first:
            sources, counts, times = first["sources"][:], first["n_sources"][:], first[f"{lock}/times"][:]
            sfreq = float(first.attrs["sfreq"])

        path = destination.band_power(subject=subject, stream=stream, lock=lock, space=PARCELLATION)
        with h5py.File(path, "w") as handle:
            power = handle.create_dataset(
                "power", shape=(len(trials), len(BANDS), counts.size, times.size), dtype="float32",
                chunks=(1, 1, counts.size, times.size),
            )
            start = 0
            for run, ep in epochs.items():
                for band_index, band in enumerate(BANDS):
                    with h5py.File(parts[run, band], "r") as part:
                        power[start:start + len(ep), band_index] = part[f"{lock}/power"][:]
                start += len(ep)
            handle.create_dataset("times", data=times)
            handle.create_dataset("bands", data=np.array(list(BANDS), dtype="S"))
            handle.create_dataset("band_edges_hz", data=np.array(list(BANDS.values())))
            handle.create_dataset("sources", data=sources)
            handle.create_dataset("n_sources", data=counts)
            handle.attrs.update({
                "subject": subject, "stream": stream, "lock": lock, "method": METHOD, "lambda2": LAMBDA2,
                "parcellation": PARCELLATION, "sfreq": sfreq, "power": "linear |z|^2",
            })
        save_sidecar(path, {
            "subject": subject, "stream": stream, "lock": lock, "space": PARCELLATION,
            "analysis_window_s": list(WINDOWS[lock]), "method": METHOD, "lambda2": LAMBDA2,
            "orientation": {"cortex": "normal", "deep": "sum of x, y, z power"},
            "bands_hz": BANDS, "transitions_hz": TRANSITIONS, "invalid_near": list(INVALID_NEAR),
            "n_trials": int(len(trials)),
            "empty_sources": [name.decode() for name, count in zip(sources, counts) if count == 0],
        })
        table = destination.band_power_trials(subject=subject, stream=stream, lock=lock)
        trials.to_csv(table, sep="\t", index=False, na_rep="n/a")
        outputs.extend([path, table])
    return WorkflowResult(
        stage="band_power_merge",
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        settings={"subject": subject, "stream": stream, "locks": list(locks)},
    )
