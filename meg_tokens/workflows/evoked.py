"""Event-related fields for one subject and stream: per lock, the twelve
conditions' averages in sensors and in the HCP-MMP1 regions, every Fast and
Slow run combined."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import h5py
import mne
import numpy as np

from meg_tokens.core import ProjectConfig, WorkflowResult
from meg_tokens.io import DerivativeLayout, require_file, save_sidecar
from meg_tokens.meg.band_power import LOCKS, PARCELLATION, hcp_labels, morph_map
from meg_tokens.meg.epoching import WINDOWS
from meg_tokens.meg.evoked import CONDITIONS, condition_masks, region_time_courses
from meg_tokens.meg.meg_bids import epoched_meg_runs
from meg_tokens.meg.sources import LAMBDA2, METHOD


def evoked_subject(
    project: ProjectConfig,
    subject: str,
    *,
    stream: str,
    locks: Sequence[str] = LOCKS,
    output_root: Optional[str | Path] = None,
) -> WorkflowResult:
    """The twelve conditions' averages of one subject and stream, per lock.

    Writes per lock an HDF5 file of conditions x 360 regions x times (signed,
    sLORETA at nave = 1, 0.5-150 Hz as cleaned, no baseline), the trials each
    averages, and an MNE ``-ave.fif`` of the sensor averages. A condition with
    no trial is NaN in the regions and absent from the sensors.
    """
    layout = DerivativeLayout(project.bids_root, task=project.task)
    destination = DerivativeLayout(output_root or project.bids_root, task=project.task)
    inputs, outputs, labels = [], [], None
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
            prepared = mne.minimum_norm.prepare_inverse_operator(inv, 1, LAMBDA2, METHOD, verbose=False)
            times = epochs.times
            for name, mask in condition_masks(epochs.metadata).items():
                n = int(mask.sum())
                if not n:
                    continue
                average = epochs[mask].average()
                totals[name] = totals.get(name, 0) + n * region_time_courses(average, prepared, labels)
                counts[name] += n
                sensors[name].append(average)
        if times is None:
            continue

        regions = np.full((len(CONDITIONS), len(labels), times.size), np.nan, dtype=np.float32)
        for index, name in enumerate(CONDITIONS):
            if counts[name]:
                regions[index] = totals[name] / counts[name]
        path = destination.evoked_regions(subject=subject, stream=stream, lock=lock)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            handle.create_dataset("erp", data=regions)
            handle.create_dataset("n_trials", data=np.array([counts[name] for name in CONDITIONS]))
            handle.create_dataset("conditions", data=np.array(CONDITIONS, dtype="S"))
            handle.create_dataset("sources", data=np.array([label.name for label in labels], dtype="S"))
            handle.create_dataset("times", data=times)
            handle.attrs.update({
                "subject": subject, "stream": stream, "lock": lock, "method": METHOD, "lambda2": LAMBDA2,
                "nave": 1, "orientation": "normal", "mode": "mean_flip", "parcellation": PARCELLATION,
            })
        save_sidecar(path, {
            "subject": subject, "stream": stream, "lock": lock, "analysis_window_s": list(WINDOWS[lock]),
            "method": METHOD, "lambda2": LAMBDA2, "nave": 1, "mode": "mean_flip", "filter": "0.5-150 Hz as cleaned",
            "baseline": None, "n_trials": {name: counts[name] for name in CONDITIONS},
        })
        averages = [
            mne.combine_evoked(sensors[name], weights="nave") for name in CONDITIONS if sensors[name]
        ]
        for average, name in zip(averages, [name for name in CONDITIONS if sensors[name]]):
            average.comment = name
        sensor_path = destination.evoked_sensors(subject=subject, stream=stream, lock=lock)
        mne.write_evokeds(sensor_path, averages, overwrite=True, verbose=False)
        outputs.extend([path, sensor_path])
    return WorkflowResult(
        stage="evoked",
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        settings={"subject": subject, "stream": stream, "locks": list(locks)},
    )
