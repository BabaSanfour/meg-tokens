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


def _in_window(times: np.ndarray, window: Optional[tuple[float, float]]) -> np.ndarray:
    """The samples inside ``window`` (seconds from the lock, both ends kept); all without one."""
    if window is None:
        return np.ones(times.size, bool)
    return (times >= window[0] - 1e-9) & (times <= window[1] + 1e-9)


def _require_every_condition(handle: h5py.File, counts: str, subject: str, what: str) -> None:
    empty = [name for name, n in zip(_names(handle["conditions"][:]), handle[counts][:]) if n == 0]
    if empty:
        raise ValueError(f"{subject} {what}: no trials in {empty}; the PCA needs every subject's twelve conditions")


class _PowerMeans:
    """One band's means of the subjects' cortical sources, read from the means files:
    the 360 HCP-MMP1 regions (deep structures left out) or the 8,196 fsaverage
    sources; within ``window``; log10 centred by each subject's ``log10_center``."""

    def __init__(
        self, layout: DerivativeLayout, fsaverage: Optional[DerivativeLayout], subjects: Sequence[str], *,
        stream: str, lock: str, measure: str, band: int, space: str, window: Optional[tuple[float, float]],
    ) -> None:
        self.measure, self.band, self.space = measure, band, space
        if space == PARCELLATION:
            self.paths = [
                require_file(layout.band_power_means(subject=subject, stream=stream, lock=lock),
                             purpose=f"{subject} {stream} {lock} band power means; run `meg band-power-means` first")
                for subject in subjects
            ]
        else:
            self.paths = [
                require_file(fsaverage.band_power_means_fsaverage(subject=subject, stream=stream, lock=lock,
                                                                  band=list(BANDS)[band]),
                             purpose=f"{subject} {stream} {lock} fsaverage band power means; "
                                     "run `meg band-power-means-fsaverage` first")
                for subject in subjects
            ]
        with h5py.File(self.paths[0], "r") as first:
            if space == PARCELLATION:
                names = _names(first["sources"][:])
                self.cortex = ~np.isin(names, VOLUME_LABELS)
                sources = np.array(names)[self.cortex].tolist()
            else:
                sources = [f"lh-{v}" for v in first["vertices_lh"][:]] + [f"rh-{v}" for v in first["vertices_rh"][:]]
                self.cortex = np.ones(len(sources), bool)
            self.crop = _in_window(first["times"][:], window)
            self.meta = {
                "conditions": _names(first["conditions"][:]), "quantiles": _names(first["quantiles"][:]),
                "sources": sources, "times": first["times"][:][self.crop],
            }
        self.subjects, self.what = list(subjects), f"{stream} {lock}"

    def _read(self, handle: h5py.File, group: str) -> np.ndarray:
        data = handle[f"{group}_{self.measure}"]
        data = data[:, self.band] if self.space == PARCELLATION else data[:]
        data = np.swapaxes(data[:, self.cortex][..., self.crop], 1, 2)
        if self.measure == "log10":
            centre = handle["log10_center"][self.band] if self.space == PARCELLATION else handle["log10_center"][:]
            data = pca.centre_log10(data[None], centre[None, self.cortex].astype(np.float32))[0]
        return data

    def conditions(self) -> np.ndarray:
        """Subject x condition x time x source, float32 as stored."""
        shape = (len(self.meta["conditions"]), int(self.crop.sum()), int(self.cortex.sum()))
        means = np.empty((len(self.paths), *shape), dtype=np.float32)
        for index, (subject, path) in enumerate(zip(self.subjects, self.paths)):
            with h5py.File(path, "r") as handle:
                _require_every_condition(handle, "condition_n", subject, self.what)
                means[index] = self._read(handle, "condition")
        return means

    def quantiles(self, index: int) -> np.ndarray:
        """One subject's quintile means, quintile x time x source."""
        with h5py.File(self.paths[index], "r") as handle:
            return self._read(handle, "quantile")


def _field_means(
    layout: DerivativeLayout, subjects: Sequence[str], *, stream: str, lock: str, space: str,
    window: Optional[tuple[float, float]],
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
        crop = _in_window(first["times"][:], window)
        meta = {"conditions": _names(first["conditions"][:]), "sources": sources, "times": first["times"][:][crop]}
    fields = np.empty((len(subjects), len(meta["conditions"]), int(crop.sum()), len(sources)), dtype=np.float32)
    for index, (subject, path) in enumerate(zip(subjects, paths)):
        with h5py.File(path, "r") as handle:
            _require_every_condition(handle, "n_trials", subject, f"{stream} {lock} {space}")
            fields[index] = np.swapaxes(handle["erp"][:][..., crop], 1, 2)
    return fields, meta, paths


def trajectories(
    project: ProjectConfig,
    *,
    stream: str,
    lock: str,
    settings: TrajectoryConfig,
    subjects: Optional[Sequence[str]] = None,
    fsaverage_root: Optional[str | Path] = None,
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
    Band power on fsaverage reads its means under ``fsaverage_root``. Each
    subject is projected on its own, so no more than one subject's means is
    ever copied to the components' precision.
    """
    layout = DerivativeLayout(project.bids_root, task=project.task)
    destination = DerivativeLayout(output_root or project.bids_root, task=project.task)
    fsaverage = None if fsaverage_root is None else DerivativeLayout(fsaverage_root, task=project.task)
    if settings.space == "fsaverage" and settings.measure != "erp" and fsaverage is None:
        raise ValueError("band power on fsaverage needs fsaverage_root, where its means are")
    selected = list(subjects or staged_subjects(project.bids_root))
    is_field = settings.measure == "erp"
    bands = ("field",) if is_field else tuple(BANDS)

    results, signs = [], None
    for band_index, _ in enumerate(bands):
        source = None
        if is_field:
            means, meta, paths = _field_means(layout, selected, stream=stream, lock=lock, space=settings.space,
                                              window=settings.window)
        else:
            source = _PowerMeans(layout, fsaverage, selected, stream=stream, lock=lock, measure=settings.measure,
                                 band=band_index, space=settings.space, window=settings.window)
            means, meta, paths = source.conditions(), source.meta, source.paths
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
            "scores": np.stack([pca.project(reducer, subject, settings.fit) for subject in means]),
        }
        if source is not None:
            result["quantile_scores"] = np.stack([
                pca.project(reducer, source.quantiles(index)[..., keep], settings.fit) for index in range(len(selected))
            ])
        results.append(result)
        del means

    path = destination.trajectories(
        stream=stream, lock=lock, space=settings.space, fit=settings.fit, measure=settings.measure,
        window=settings.window,
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
            "window_s": list(settings.window) if settings.window else [float(meta["times"][0]), float(meta["times"][-1])],
        })
    save_sidecar(path, {
        "stream": stream, "lock": lock, "space": settings.space, "fit": settings.fit,
        "measure": settings.measure, "n_components": settings.n_components, "projection": projection,
        "fitted_on": "grand average of the subjects' condition means" if settings.fit == "grandaverage"
        else "every subject's condition means pooled at equal weight",
        "window_s": list(settings.window) if settings.window else "the lock's whole analysis window",
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
