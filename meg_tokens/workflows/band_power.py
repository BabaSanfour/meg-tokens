"""Single-trial band power in source space, in two steps so the work spreads
over many jobs: each run, stream and band computed on its own (every lock),
then a subject's parts assembled per stream and lock."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Iterator
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
    morph_map,
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
    morph_map(anatomy, subjects_dir)
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


QUANTILES = 5

# The trajectory primer's baseline: each trial's mean power over the 200 ms
# before the first token, taken from its go epoch whatever the lock.
BASELINE = (-0.2, 0.0)

# The three forms of power averaged per condition and quintile.
MEASURES = ("power", "log10", "db")

# Trials read at once by the passes over every trial of a file.
BLOCK = 32


def _groups(trials: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """The rows of each of the twelve conditions, and of Fast and Slow trials in
    each fifth of the subject's own decision times in that condition."""
    from meg_tokens.meg.evoked import condition_masks

    groups = {name: np.flatnonzero(mask) for name, mask in condition_masks(trials).items()}
    quantiles = {}
    for speed in ("Fast", "Slow"):
        rows = trials.index[(trials["condition"] == speed) & trials["dt_ms"].notna()]
        bins = pd.qcut(trials.loc[rows, "dt_ms"].rank(method="first"), QUANTILES, labels=False)
        for q in range(QUANTILES):
            quantiles[f"{speed.lower()}-q{q + 1}"] = rows[bins.to_numpy() == q].to_numpy()
    return groups, quantiles


def _trial_keys(trials: pd.DataFrame) -> pd.MultiIndex:
    """A trial's identity in every lock's table: its run and its index in the run's task log."""
    return pd.MultiIndex.from_frame(trials[["condition", "run", "nTrialIndex"]])


def _go_baselines(destination: DerivativeLayout, subject: str, stream: str) -> tuple[pd.MultiIndex, np.ndarray]:
    """Each go epoch's mean linear power over ``BASELINE``, trials x bands x sources, with its trials' keys."""
    path = require_file(
        destination.band_power(subject=subject, stream=stream, lock="go", space=PARCELLATION),
        purpose=f"{subject} {stream} go band power, the baseline of every lock; run `meg band-power-merge` first",
    )
    trials = pd.read_csv(destination.band_power_trials(subject=subject, stream=stream, lock="go"), sep="\t")
    with h5py.File(path, "r") as source:
        power, times = source["power"], source["times"][:]
        span = (times >= BASELINE[0]) & (times <= BASELINE[1])
        baseline = np.concatenate([
            np.nanmean(power[start:start + BLOCK][..., span], axis=-1)
            for start in range(0, power.shape[0], BLOCK)
        ])
    return _trial_keys(trials), baseline


def band_power_means(
    project: ProjectConfig,
    subject: str,
    *,
    stream: str,
    locks: Sequence[str] = LOCKS,
    output_root: Optional[str | Path] = None,
) -> WorkflowResult:
    """A subject's mean band power per condition and per decision-time quintile, per lock, three ways.

    Averages the assembled single-trial power (NaN ignored) over the trials of
    each of the twelve conditions, and over Fast and Slow trials in each fifth
    of the subject's own decision times in that condition (the quantile-matched
    bins of the time-scaling test), as linear power, as log10 power, and in dB
    against each trial's own baseline: its mean power over ``BASELINE`` in its
    go epoch. A trial without a kept go epoch has no baseline and is left out
    of the dB means. Writes one HDF5 file per lock: ``condition_<measure>`` (12
    x bands x sources x times) and ``quantile_<measure>`` (Fast q1-q5, Slow
    q1-q5) for each of ``MEASURES``, their trial counts (``*_n``, and ``*_n_db``
    for the trials with a baseline), ``log10_center`` (bands x sources: the mean
    log10 power over every trial and time of the lock, to centre by) and the
    quantiles' mean decision times.
    """
    from meg_tokens.meg.evoked import CONDITIONS

    destination = DerivativeLayout(output_root or project.bids_root, task=project.task)
    go_keys, go_baseline = _go_baselines(destination, subject, stream)
    inputs, outputs = [], []
    for lock in locks:
        power_path = require_file(
            destination.band_power(subject=subject, stream=stream, lock=lock, space=PARCELLATION),
            purpose=f"{subject} {stream} {lock} band power; run `meg band-power-merge` first",
        )
        trials = pd.read_csv(destination.band_power_trials(subject=subject, stream=stream, lock=lock), sep="\t")
        inputs.append(power_path)
        groups, quantiles = _groups(trials)
        position = go_keys.get_indexer(_trial_keys(trials))
        has_baseline = position >= 0
        baseline = np.where(has_baseline[:, None, None], go_baseline[position], np.nan)

        with h5py.File(power_path, "r") as source:
            power = source["power"]
            shape = power.shape[1:]

            def means(indices: np.ndarray) -> tuple[list[np.ndarray], int]:
                if not indices.size:
                    return [np.full(shape, np.nan, dtype=np.float32)] * len(MEASURES), 0
                rows = np.sort(indices)
                linear = power[rows]
                logged = np.log10(linear)
                # MNE's rescale takes the baseline from the array it rescales;
                # this one comes from another lock's epoch, so dB is written out.
                db = 10 * (logged - np.log10(baseline[rows])[..., None])
                return [np.nanmean(x, axis=0).astype(np.float32) for x in (linear, logged, db)], int(has_baseline[rows].sum())

            condition = [means(groups[name]) for name in CONDITIONS]
            quantile = [means(indices) for indices in quantiles.values()]
            total, count = np.zeros(shape[:-1]), np.zeros(shape[:-1])
            for start in range(0, power.shape[0], BLOCK):
                logged = np.log10(power[start:start + BLOCK])
                total += np.nansum(logged, axis=(0, -1))
                count += np.isfinite(logged).sum(axis=(0, -1))
            meta = {key: source[key][:] for key in ("times", "bands", "band_edges_hz", "sources", "n_sources")}

        path = destination.band_power_means(subject=subject, stream=stream, lock=lock)
        with h5py.File(path, "w") as handle:
            for prefix, results, sizes, names in (
                ("condition", condition, [groups[name].size for name in CONDITIONS], CONDITIONS),
                ("quantile", quantile, [indices.size for indices in quantiles.values()], list(quantiles)),
            ):
                for index, measure in enumerate(MEASURES):
                    handle.create_dataset(f"{prefix}_{measure}", data=np.stack([r[0][index] for r in results]))
                handle.create_dataset(f"{prefix}_n", data=np.array(sizes))
                handle.create_dataset(f"{prefix}_n_db", data=np.array([r[1] for r in results]))
                handle.create_dataset(f"{prefix}s", data=np.array(names, dtype="S"))
            handle.create_dataset("log10_center", data=total / count)
            handle.create_dataset("quantile_dt_ms", data=np.array([
                trials.loc[indices, "dt_ms"].mean() if indices.size else np.nan for indices in quantiles.values()
            ]))
            for key, value in meta.items():
                handle.create_dataset(key, data=value)
            handle.attrs.update({
                "subject": subject, "stream": stream, "lock": lock,
                "power": "linear |z|^2, mean of trials",
                "log10": "log10 of each trial's power, mean of trials",
                "db": "10 log10 of each trial's power over its mean power in the go epoch's baseline, mean of trials",
                "baseline_s": BASELINE,
            })
        outputs.append(path)
    return WorkflowResult(
        stage="band_power_means",
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        settings={"subject": subject, "stream": stream, "locks": list(locks)},
    )


# Trials of a fsaverage file read at once (each about 69 MB at the longer locks).
FSAVERAGE_BLOCK = 8


def band_power_means_fsaverage(
    project: ProjectConfig,
    subject: str,
    *,
    stream: str,
    band: str,
    fsaverage_root: str | Path,
    locks: Sequence[str] = LOCKS,
    output_root: Optional[str | Path] = None,
) -> WorkflowResult:
    """A subject's mean fsaverage band power per condition and decision-time quintile, one band, every lock.

    The groups, the three forms and the baseline are those of
    ``band_power_means``, on the 8,196 fsaverage sources. The fsaverage power
    stays one file per run and band (``band_power_part``), each holding a
    run's trials in the order of the subject's trial table, so every run's file
    is read once per lock and each trial added to the sums of its condition
    and its quintile. Writes one file per lock under ``fsaverage_root`` (or
    ``output_root``): ``condition_<measure>`` (12 x 8,196 x times) and
    ``quantile_<measure>`` (10 x ...) for each of ``MEASURES``, the counts
    (``*_n``, ``*_n_db``), ``log10_center`` (sources), ``quantile_dt_ms``,
    ``times`` and ``vertices_lh`` / ``vertices_rh``.
    """
    from meg_tokens.meg.band_power import fsaverage_vertices
    from meg_tokens.meg.evoked import CONDITIONS

    layout = DerivativeLayout(project.bids_root, task=project.task)
    source = DerivativeLayout(fsaverage_root, task=project.task)
    destination = DerivativeLayout(output_root or fsaverage_root, task=project.task)
    vertices = fsaverage_vertices(_subjects_dir(project))
    tables = {
        lock: pd.read_csv(layout.band_power_trials(subject=subject, stream=stream, lock=lock), sep="\t")
        for lock in dict.fromkeys(("go", *locks))
    }

    def runs(table: pd.DataFrame) -> Iterator[tuple[Path, np.ndarray]]:
        for (condition, run), rows in table.groupby(["condition", "run"], sort=False).indices.items():
            yield require_file(
                source.band_power_part(subject=subject, run=str(run), condition=condition, stream=stream,
                                       band=band, space="fsaverage"),
                purpose=f"{subject} {condition} {run} {stream} {band} fsaverage band power",
            ), np.sort(rows)

    def blocks(path: Path, lock: str, rows: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        with h5py.File(path, "r") as part:
            power = part[f"{lock}/power"]
            if power.shape[0] != rows.size:
                raise ValueError(f"{path.name} holds {power.shape[0]} {lock} trials, its trial table {rows.size}")
            for start in range(0, rows.size, FSAVERAGE_BLOCK):
                yield rows[start:start + FSAVERAGE_BLOCK], power[start:start + FSAVERAGE_BLOCK]

    go = tables["go"]
    baseline = np.full((len(go), sum(v.size for v in vertices)), np.nan, dtype=np.float32)
    inputs = []
    for path, rows in runs(go):
        inputs.append(path)
        with h5py.File(path, "r") as part:
            times = part["go/times"][:]
        span = (times >= BASELINE[0]) & (times <= BASELINE[1])
        for block_rows, block in blocks(path, "go", rows):
            baseline[block_rows] = np.nanmean(block[..., span], axis=-1)
    go_keys = _trial_keys(go)

    outputs = []
    for lock in locks:
        trials = tables[lock]
        groups, quantiles = _groups(trials)
        partitions = {"condition": list(groups.values()), "quantile": list(quantiles.values())}
        member = {}
        for prefix, members in partitions.items():
            member[prefix] = np.full(len(trials), -1)
            for index, rows in enumerate(members):
                member[prefix][rows] = index
        position = go_keys.get_indexer(_trial_keys(trials))
        own_baseline = np.where((position >= 0)[:, None], baseline[position], np.nan)
        with h5py.File(next(runs(trials))[0], "r") as part:
            times = part[f"{lock}/times"][:]
        shape = (baseline.shape[1], times.size)
        sums = {(p, m): np.zeros((len(members), *shape), np.float32) for p, members in partitions.items() for m in MEASURES}
        counts = {(p, k): np.zeros((len(members), *shape), np.int16) for p, members in partitions.items() for k in ("n", "n_db")}
        centre_sum, centre_n = np.zeros(shape[0]), np.zeros(shape[0])
        for path, rows in runs(trials):
            for block_rows, linear in blocks(path, lock, rows):
                logged = np.log10(linear)
                # MNE's rescale takes the baseline from the array it rescales;
                # this one comes from the go epoch, so dB is written out.
                db = 10 * (logged - np.log10(own_baseline[block_rows])[..., None])
                centre_sum += np.nansum(logged, axis=(0, 2))
                centre_n += np.isfinite(logged).sum(axis=(0, 2))
                for j, row in enumerate(block_rows):
                    for prefix in partitions:
                        g = member[prefix][row]
                        if g < 0:
                            continue
                        for measure, values in zip(MEASURES, (linear[j], logged[j], db[j])):
                            finite = np.isfinite(values)
                            sums[prefix, measure][g] += np.where(finite, values, 0)
                            if measure != "log10":
                                counts[prefix, "n" if measure == "power" else "n_db"][g] += finite

        path = destination.band_power_means_fsaverage(subject=subject, stream=stream, lock=lock, band=band)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            for prefix, members in partitions.items():
                for measure in MEASURES:
                    n = counts[prefix, "n_db" if measure == "db" else "n"]
                    mean = np.full(sums[prefix, measure].shape, np.nan, np.float32)
                    np.divide(sums[prefix, measure], n, out=mean, where=n > 0)
                    handle.create_dataset(f"{prefix}_{measure}", data=mean)
                handle.create_dataset(f"{prefix}_n", data=np.array([rows.size for rows in members]))
                handle.create_dataset(f"{prefix}_n_db", data=np.array([
                    int((position[rows] >= 0).sum()) for rows in members
                ]))
            handle.create_dataset("conditions", data=np.array(CONDITIONS, dtype="S"))
            handle.create_dataset("quantiles", data=np.array(list(quantiles), dtype="S"))
            handle.create_dataset("quantile_dt_ms", data=np.array([
                trials.loc[rows, "dt_ms"].mean() if rows.size else np.nan for rows in quantiles.values()
            ]))
            handle.create_dataset("log10_center", data=centre_sum / centre_n)
            handle.create_dataset("times", data=times)
            handle.create_dataset("vertices_lh", data=vertices[0])
            handle.create_dataset("vertices_rh", data=vertices[1])
            handle.attrs.update({
                "subject": subject, "stream": stream, "lock": lock, "band": band,
                "power": "linear |z|^2, mean of trials",
                "log10": "log10 of each trial's power, mean of trials",
                "db": "10 log10 of each trial's power over its mean power in the go epoch's baseline, mean of trials",
                "baseline_s": BASELINE,
            })
        outputs.append(path)
    return WorkflowResult(
        stage="band_power_means_fsaverage",
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        settings={"subject": subject, "stream": stream, "band": band, "locks": list(locks)},
    )


def deep_power_part(
    project: ProjectConfig,
    run: RunSpec,
    *,
    stream: str,
    band: str,
    output_root: str | Path,
    locks: Sequence[str] = LOCKS,
    n_jobs: int = 1,
) -> WorkflowResult:
    """One run's band power at every deep source, one band, every lock, as ``band_power_part`` computes it.

    The region file keeps one mean per deep structure; this keeps each of the
    volume's sources (all 17 structures, whatever share of their estimate is
    cortex: meg.md §4), its power the sum over x, y and z of |z|^2. Writes,
    under ``output_root``, ``<lock>/power`` (trials x sources x times, in the
    order of the lock's epoch file, NaN where the filter reaches a flux jump or
    the recording's end) and ``<lock>/times``, with each source's
    ``structure``, its volume vertex (``vertno``) and its position in the MRI
    (``positions_mri``, metres).
    """
    from meg_tokens.meg.band_power import deep_trial_power

    layout = DerivativeLayout(project.bids_root, task=project.task)
    destination = DerivativeLayout(output_root, task=project.task)
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
    volume = inv["src"][-1]
    structure = np.full(volume["nuse"], "", dtype=object)
    for name, where in deep_structures(inv["src"], subjects_dir, mri_head_t=inv["mri_head_t"]).items():
        structure[where] = name
    positions = mne.transforms.apply_trans(
        mne.transforms.invert_transform(inv["mri_head_t"]), volume["rr"][volume["vertno"]],
    )
    sfreq = raw.info["sfreq"]
    signal = analytic(raw, band, n_jobs=n_jobs)
    invalid = invalid_samples(raw, half_length(sfreq, band))
    prepared = mne.minimum_norm.prepare_inverse_operator(inv, 1, LAMBDA2, METHOD, verbose=False)

    path = destination.band_power_part(**entities, band=band, space="aseg")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs.update({
            "subject": run.subject, "run": run.run, "condition": run.condition, "stream": stream, "band": band,
            "method": METHOD, "lambda2": LAMBDA2, "sfreq": sfreq, "power": "sum over x, y, z of |z|^2",
        })
        handle.create_dataset("structure", data=structure.astype("S"))
        handle.create_dataset("vertno", data=volume["vertno"])
        handle.create_dataset("positions_mri", data=positions)
        for lock, epochs in kept.items():
            offsets = window(sfreq, lock)
            trials = mne.Epochs(
                signal, epochs.events, epochs.event_id, *WINDOWS[lock], baseline=None,
                reject_by_annotation=False, preload=False, verbose=False,
            )
            bad = invalid[epochs.events[:, 0, None] - raw.first_samp + offsets]
            power = handle.create_dataset(
                f"{lock}/power", shape=(len(epochs), volume["nuse"], offsets.size), dtype="float32",
                chunks=(1, volume["nuse"], offsets.size),
            )
            handle.create_dataset(f"{lock}/times", data=offsets / sfreq)
            for index, trial in enumerate(deep_trial_power(trials, prepared)):
                power[index] = np.where(bad[index], np.nan, trial)
    return WorkflowResult(
        stage="deep_power_part",
        inputs=(raw_path, inv_path),
        outputs=(path,),
        settings={"run": f"{run.subject} {run.condition} {run.run}", "stream": stream, "band": band,
                  "locks": list(kept)},
    )
