"""PCA trajectories of the cohort's condition means (meg.md §2, Analyses).

One stream and lock at a time: every subject's twelve condition means, of band
power (each band on its own) or of the signed event-related fields, are read,
put on one PCA, fitted on their grand average or on all of them pooled
(``analysis/trajectories.py``, through coco-pipe), and every subject's
condition means, and for band power its decision-time quintile means, are
projected on the components. The 17 deep structures stay out: through this
inverse they are mostly cortex (meg.md §4), and the preprint's PCA was of the
cortex alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import h5py
import numpy as np

from meg_tokens.analysis import trajectories as pca
from meg_tokens.core import ProjectConfig, TrajectoryConfig, WorkflowResult
from meg_tokens.io import DerivativeLayout, require_file, save_sidecar
from meg_tokens.meg.band_power import BANDS, PARCELLATION
from meg_tokens.meg.meg_bids import staged_subjects
from meg_tokens.meg.sources import VOLUME_LABELS


def _names(values: np.ndarray) -> list[str]:
    return [value.decode() for value in values]


def _require_every_condition(handle: h5py.File, counts: str, subject: str, what: str) -> None:
    empty = [name for name, n in zip(_names(handle["conditions"][:]), handle[counts][:]) if n == 0]
    if empty:
        raise ValueError(f"{subject} {what}: no trials in {empty}; the PCA needs every subject's twelve conditions")


def _power_means(
    layout: DerivativeLayout, subjects: Sequence[str], *, stream: str, lock: str, measure: str, band: int,
) -> tuple[np.ndarray, np.ndarray, dict, list[Path]]:
    """One band's condition and quintile means of the subjects' cortical regions,
    subject x (condition | quintile) x time x region, log10 centred."""
    paths = [
        require_file(layout.band_power_means(subject=subject, stream=stream, lock=lock),
                     purpose=f"{subject} {stream} {lock} band power means; run `meg band-power-means` first")
        for subject in subjects
    ]
    with h5py.File(paths[0], "r") as first:
        cortex = ~np.isin(_names(first["sources"][:]), VOLUME_LABELS)
        meta = {
            "conditions": _names(first["conditions"][:]), "quantiles": _names(first["quantiles"][:]),
            "sources": np.array(_names(first["sources"][:]))[cortex].tolist(), "times": first["times"][:],
        }
        shape = (len(meta["times"]), int(cortex.sum()))
    # Kept in the files' float32; the fit upcasts what needs it (analysis/trajectories.py).
    conditions = np.empty((len(subjects), len(meta["conditions"]), *shape), dtype=np.float32)
    quantiles = np.empty((len(subjects), len(meta["quantiles"]), *shape), dtype=np.float32)
    centres = np.zeros((len(subjects), shape[1]), dtype=np.float32)
    for index, (subject, path) in enumerate(zip(subjects, paths)):
        with h5py.File(path, "r") as handle:
            _require_every_condition(handle, "condition_n", subject, f"{stream} {lock}")
            conditions[index] = np.swapaxes(handle[f"condition_{measure}"][:, band][:, cortex], 1, 2)
            quantiles[index] = np.swapaxes(handle[f"quantile_{measure}"][:, band][:, cortex], 1, 2)
            if measure == "log10":
                centres[index] = handle["log10_center"][band][cortex]
    if measure == "log10":
        conditions, quantiles = pca.centre_log10(conditions, centres), pca.centre_log10(quantiles, centres)
    return conditions, quantiles, meta, paths


def _field_means(
    layout: DerivativeLayout, subjects: Sequence[str], *, stream: str, lock: str, space: str,
) -> tuple[np.ndarray, dict, list[Path]]:
    """The subjects' signed condition fields on the cortex, subject x condition x time x source."""
    paths = [
        require_file(layout.evoked(subject=subject, stream=stream, lock=lock, space=space),
                     purpose=f"{subject} {stream} {lock} {space} event-related fields; run `meg erp` first")
        for subject in subjects
    ]
    with h5py.File(paths[0], "r") as first:
        if space == PARCELLATION:
            sources = _names(first["sources"][:])
        else:
            sources = [f"lh-{v}" for v in first["vertices_lh"][:]] + [f"rh-{v}" for v in first["vertices_rh"][:]]
        meta = {"conditions": _names(first["conditions"][:]), "sources": sources, "times": first["times"][:]}
    fields = np.empty((len(subjects), len(meta["conditions"]), len(meta["times"]), len(sources)), dtype=np.float32)
    for index, (subject, path) in enumerate(zip(subjects, paths)):
        with h5py.File(path, "r") as handle:
            _require_every_condition(handle, "n_trials", subject, f"{stream} {lock} {space}")
            fields[index] = np.swapaxes(handle["erp"][:], 1, 2)
    return fields, meta, paths


def trajectories(
    project: ProjectConfig,
    *,
    stream: str,
    lock: str,
    settings: TrajectoryConfig,
    subjects: Optional[Sequence[str]] = None,
    output_root: Optional[str | Path] = None,
) -> WorkflowResult:
    """The cohort's PCA trajectories around one lock, one fit and measure, every band.

    Writes one HDF5 file with a leading band axis (``field`` alone for the
    event-related fields): ``components`` (component x source), ``mean`` (the
    fit's centre), ``explained_variance_ratio``, ``participation_ratio``,
    ``reconstruction_error`` (per subject), ``scores`` (subject x condition x
    time x component) and, for band power, ``quantile_scores`` (subject x
    quintile x time x component); with ``subjects``, ``conditions``,
    ``quantiles``, ``sources`` (those finite in every subject), ``times`` and,
    for the region fields, ``signs`` (subject x region, ``align_signs``).
    """
    layout = DerivativeLayout(project.bids_root, task=project.task)
    destination = DerivativeLayout(output_root or project.bids_root, task=project.task)
    selected = list(subjects or staged_subjects(project.bids_root))
    is_field = settings.measure == "erp"
    bands = ("field",) if is_field else tuple(BANDS)

    results, signs = [], None
    for band_index, _ in enumerate(bands):
        quantile_means = None
        if is_field:
            means, meta, paths = _field_means(layout, selected, stream=stream, lock=lock, space=settings.space)
        else:
            means, quantile_means, meta, paths = _power_means(
                layout, selected, stream=stream, lock=lock, measure=settings.measure, band=band_index,
            )
        keep = np.isfinite(means).all(axis=(0, 1, 2))
        if not keep.all():
            means = means[..., keep]
        if is_field and settings.space == PARCELLATION:
            signs = pca.align_signs(means)
            means *= signs[:, None, None, :]
        reducer = pca.fit(means, settings.fit, settings.n_components)
        result = {
            "components": reducer.get_components(),
            "mean": reducer.reducer.model.mean_,
            **pca.describe(reducer, means),
            "scores": pca.project(reducer, means, settings.fit),
        }
        if quantile_means is not None:
            result["quantile_scores"] = pca.project(reducer, quantile_means[..., keep], settings.fit)
        results.append(result)

    path = destination.trajectories(
        stream=stream, lock=lock, space=settings.space, fit=settings.fit, measure=settings.measure,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    projection = "uncentred" if settings.fit == "grandaverage" else "centred"
    with h5py.File(path, "w") as handle:
        for key in results[0]:
            handle.create_dataset(key, data=np.stack([result[key] for result in results]))
        handle.create_dataset("bands", data=np.array(bands, dtype="S"))
        handle.create_dataset("subjects", data=np.array(selected, dtype="S"))
        handle.create_dataset("conditions", data=np.array(meta["conditions"], dtype="S"))
        handle.create_dataset("sources", data=np.array(meta["sources"], dtype="S")[keep])
        handle.create_dataset("times", data=meta["times"])
        if not is_field:
            handle.create_dataset("quantiles", data=np.array(meta["quantiles"], dtype="S"))
        if signs is not None:
            handle.create_dataset("signs", data=signs)
        handle.attrs.update({
            "stream": stream, "lock": lock, "space": settings.space, "fit": settings.fit,
            "measure": settings.measure, "n_components": settings.n_components, "projection": projection,
        })
    save_sidecar(path, {
        "stream": stream, "lock": lock, "space": settings.space, "fit": settings.fit,
        "measure": settings.measure, "n_components": settings.n_components, "projection": projection,
        "fitted_on": "grand average of the subjects' condition means" if settings.fit == "grandaverage"
        else "every subject's condition means pooled at equal weight",
        "subjects": selected, "bands": list(bands), "n_sources": int(keep.sum()),
        "dropped_sources": np.array(meta["sources"])[~keep].tolist(),
        "deep_structures": "left out",
    })
    return WorkflowResult(
        stage="trajectories",
        inputs=tuple(paths),
        outputs=(path,),
        settings={"stream": stream, "lock": lock, "subjects": selected, **settings.__dict__},
    )
