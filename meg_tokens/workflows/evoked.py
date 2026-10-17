"""Event-related fields for one subject and stream: per lock, the twelve
conditions' averages in sensors, in the HCP-MMP1 regions and deep structures,
and on fsaverage, every Fast and Slow run combined."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import h5py
import mne
import numpy as np

from meg_tokens.core import ProjectConfig, WorkflowResult
from meg_tokens.io import DerivativeLayout, require_file, save_sidecar
from meg_tokens.meg.band_power import LOCKS, PARCELLATION, deep_labels, fsaverage_morph, hcp_labels, morph_map
from meg_tokens.meg.epoching import WINDOWS
from meg_tokens.meg.evoked import CONDITIONS, condition_masks, source_time_courses
from meg_tokens.meg.meg_bids import epoched_meg_runs
from meg_tokens.meg.sources import LAMBDA2, METHOD, VOLUME_LABELS


def evoked_subject(
    project: ProjectConfig,
    subject: str,
    *,
    stream: str,
    locks: Sequence[str] = LOCKS,
    output_root: Optional[str | Path] = None,
) -> WorkflowResult:
    """The twelve conditions' averages of one subject and stream, per lock.

    Writes per lock: an HDF5 file of conditions x 360 regions x times and
    conditions x 17 deep structures x 3 components x times; an HDF5 file of
    conditions x 8,196 fsaverage sources x times; and an MNE ``-ave.fif`` of
    the sensor averages (signed, sLORETA at nave = 1, 0.5-150 Hz as cleaned, no
    baseline). A condition with no trial is NaN in source space and absent from
    the sensors.
    """
    layout = DerivativeLayout(project.bids_root, task=project.task)
    destination = DerivativeLayout(output_root or project.bids_root, task=project.task)
    inputs, outputs, labels, deep, morph = [], [], None, None, None
    for lock in locks:
        totals, counts, sensors, times = {}, dict.fromkeys(CONDITIONS, 0), {name: [] for name in CONDITIONS}, None
        for run, path in epoched_meg_runs(layout, subject, alignment=lock, stream=stream):
            inv_path = require_file(
                layout.inverse(subject=run.subject, run=run.run, condition=run.condition, stream=stream),
                purpose=f"{run.subject} {run.condition} {run.run} {stream} inverse; run `meg source` first",
            )
            inputs.extend([path, inv_path])
            epochs = mne.read_epochs(path, preload=True, verbose=False)
            inv = mne.minimum_norm.read_inverse_operator(inv_path, verbose=False)
            if labels is None:
                anatomy = inv["src"][0]["subject_his_id"]
                morph_map(anatomy, project.subjects_dir)
                labels = hcp_labels(anatomy, project.subjects_dir)
                deep = deep_labels(anatomy, project.subjects_dir)
                morph = fsaverage_morph(inv["src"], project.subjects_dir)
            prepared = mne.minimum_norm.prepare_inverse_operator(inv, 1, LAMBDA2, METHOD, verbose=False)
            times = epochs.times
            for name, mask in condition_masks(epochs.metadata).items():
                n = int(mask.sum())
                if not n:
                    continue
                average = epochs[mask].average()
                courses = source_time_courses(average, prepared, labels, deep, morph)
                totals[name] = [n * course + (totals[name][k] if name in totals else 0) for k, course in enumerate(courses)]
                counts[name] += n
                sensors[name].append(average)
        if times is None:
            continue

        shapes = [(len(labels), times.size), (len(VOLUME_LABELS), 3, times.size), (morph.morph_mat.shape[0], times.size)]
        regions, structures, fsaverage = (
            np.full((len(CONDITIONS), *shape), np.nan, dtype=np.float32) for shape in shapes
        )
        for index, name in enumerate(CONDITIONS):
            if counts[name]:
                regions[index], structures[index], fsaverage[index] = (total / counts[name] for total in totals[name])
        n_trials = np.array([counts[name] for name in CONDITIONS])
        attrs = {
            "subject": subject, "stream": stream, "lock": lock, "method": METHOD, "lambda2": LAMBDA2,
            "nave": 1, "filter": "0.5-150 Hz as cleaned", "baseline": "none",
        }
        path = destination.evoked(subject=subject, stream=stream, lock=lock, space=PARCELLATION)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            handle.create_dataset("erp", data=regions)
            handle.create_dataset("deep", data=structures)
            handle.create_dataset("n_trials", data=n_trials)
            handle.create_dataset("conditions", data=np.array(CONDITIONS, dtype="S"))
            handle.create_dataset("sources", data=np.array([label.name for label in labels], dtype="S"))
            handle.create_dataset("deep_sources", data=np.array(VOLUME_LABELS, dtype="S"))
            handle.create_dataset("deep_axes", data=np.array(["x", "y", "z"], dtype="S"))
            handle.create_dataset("times", data=times)
            handle.attrs.update({**attrs, "orientation": "cortex normal, mean_flip; deep x, y, z (head), mean",
                                 "parcellation": PARCELLATION})
        fsaverage_path = destination.evoked(subject=subject, stream=stream, lock=lock, space="fsaverage")
        with h5py.File(fsaverage_path, "w") as handle:
            handle.create_dataset("erp", data=fsaverage, chunks=(1, min(fsaverage.shape[1], 1024), times.size))
            handle.create_dataset("n_trials", data=n_trials)
            handle.create_dataset("conditions", data=np.array(CONDITIONS, dtype="S"))
            handle.create_dataset("vertices_lh", data=morph.vertices_to[0])
            handle.create_dataset("vertices_rh", data=morph.vertices_to[1])
            handle.create_dataset("times", data=times)
            handle.attrs.update({**attrs, "orientation": "cortex normal", "space": "fsaverage oct6"})
        save_sidecar(path, {
            "subject": subject, "stream": stream, "lock": lock, "analysis_window_s": list(WINDOWS[lock]),
            "method": METHOD, "lambda2": LAMBDA2, "nave": 1, "filter": "0.5-150 Hz as cleaned",
            "regions": "cortex normal component, mean_flip per HCP-MMP1 region",
            "deep": "mean of each of x, y, z (head coordinates) per aseg structure",
            "baseline": None, "n_trials": {name: counts[name] for name in CONDITIONS},
        })
        averages = [
            mne.combine_evoked(sensors[name], weights="nave") for name in CONDITIONS if sensors[name]
        ]
        for average, name in zip(averages, [name for name in CONDITIONS if sensors[name]]):
            average.comment = name
        sensor_path = destination.evoked_sensors(subject=subject, stream=stream, lock=lock)
        mne.write_evokeds(sensor_path, averages, overwrite=True, verbose=False)
        outputs.extend([path, fsaverage_path, sensor_path])
    return WorkflowResult(
        stage="evoked",
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        settings={"subject": subject, "stream": stream, "locks": list(locks)},
    )
