"""Sensor spectra per subject, and the cohort's band choice from their spectral peaks (meg.md §2, Bands)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import h5py
import mne
import numpy as np

from meg_tokens.analysis.bands import (
    FOOOF_SETTINGS,
    band_at,
    bands_above,
    line_sweep,
    peak_shares,
    periodic_power,
    resampled_bands,
)
from meg_tokens.core import BandChoiceConfig, ProjectConfig, SpectralConfig, WorkflowResult
from meg_tokens.features.time_frequency import compute_psd, fit_specparam
from meg_tokens.io import DerivativeLayout, require_file, save_sidecar
from meg_tokens.meg.band_power import BANDS
from meg_tokens.meg.meg_bids import epoched_meg_runs, staged_subjects

# The lines of the sweep, in multiples of the histogram's mean bin.
SWEEP_LINES = np.linspace(0.5, 2.0, 61)


def sensor_spectra(
    project: ProjectConfig,
    subject: str,
    *,
    stream: str,
    settings: SpectralConfig,
    output_root: Optional[str | Path] = None,
) -> WorkflowResult:
    """One subject's magnetometer spectra around one lock, trial-averaged over every Fast and Slow run.

    Welch's method (MNE, through ``compute_psd``) on every kept epoch's whole
    analysis window; one HDF5 file with ``psd`` (magnetometers x frequencies),
    ``freqs``, ``channels`` and ``n_epochs``.
    """
    layout = DerivativeLayout(project.bids_root, task=project.task)
    destination = DerivativeLayout(output_root or project.bids_root, task=project.task)
    total, count, freqs, channels, inputs = 0.0, 0, None, None, []
    for _, path in epoched_meg_runs(layout, subject, alignment=settings.lock, stream=stream):
        epochs = mne.read_epochs(path, preload=True, verbose=False)
        if not len(epochs):
            continue
        names = epochs.copy().pick("mag").ch_names
        if channels is not None and names != channels:
            raise ValueError(f"{path.name}: magnetometers differ from the subject's other runs")
        psd, freqs = compute_psd(epochs, fmin=settings.fmin, fmax=settings.fmax, method="welch",
                                 n_fft=settings.n_fft, n_overlap=settings.n_overlap, picks="mag")
        total, count, channels = total + psd.sum(axis=0), count + psd.shape[0], names
        inputs.append(path)
    if not count:
        raise FileNotFoundError(f"{subject} {stream}: no kept {settings.lock} epochs")

    path = destination.sensor_spectra(subject=subject, stream=stream, lock=settings.lock)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("psd", data=total / count)
        handle.create_dataset("freqs", data=freqs)
        handle.create_dataset("channels", data=np.array(channels, dtype="S"))
        handle.create_dataset("n_epochs", data=count)
    save_sidecar(path, {
        "stage": "sensor_spectra", "subject": subject, "stream": stream, "lock": settings.lock,
        "method": "welch", "fmin_hz": settings.fmin, "fmax_hz": settings.fmax, "n_fft": settings.n_fft,
        "n_overlap": settings.n_overlap, "resolution_hz": float(freqs[1] - freqs[0]), "n_epochs": count,
        "n_runs": len(inputs), "inputs": [str(p) for p in inputs],
    })
    return WorkflowResult(stage="sensor_spectra", inputs=tuple(inputs), outputs=(path,),
                          settings={"subject": subject, "stream": stream, **settings.__dict__})


def band_choice(
    project: ProjectConfig,
    *,
    stream: str,
    settings: BandChoiceConfig,
    subjects: Optional[Sequence[str]] = None,
    output_root: Optional[str | Path] = None,
) -> WorkflowResult:
    """The preprint's band choice (its Fig S1) on the cohort's sensor spectra, and the checks of it.

    One specparam model of every subject's magnetometer spectra (FOOOF's
    defaults, ``settings.aperiodic``), saved as specparam's own JSON (its
    report is drawn from it), and one HDF5 file of what the figures read:
    each subject's power above the aperiodic fit (``periodic``, subjects x
    ``freqs``) and its 95 % interval over resampled subjects; each subject's
    peaks per bin (``shares``, subjects x bins between ``edges``) and its
    interval; the bands above the mean bin (``bands``, NaN where none holds a
    preprint band's centre) and their edges in each resample
    (``bands_resampled``); which bins lie above each line of the sweep
    (``sweep``, ``lines`` x bins); and per spectrum the ``exponent``, ``r2``
    and ``subject``.
    """
    layout = DerivativeLayout(project.bids_root, task=project.task)
    destination = DerivativeLayout(output_root or project.bids_root, task=project.task)
    selected = list(subjects or staged_subjects(project.bids_root))
    spectra, owner, inputs, freqs = [], [], [], None
    for index, subject in enumerate(selected):
        path = require_file(layout.sensor_spectra(subject=subject, stream=stream, lock=settings.lock),
                            purpose=f"{subject} {stream} sensor spectra; run `meg sensor-spectra` first")
        with h5py.File(path, "r") as handle:
            spectra.append(handle["psd"][:])
            freqs = handle["freqs"][:]
        owner.append(np.full(spectra[-1].shape[0], index))
        inputs.append(path)
    spectra, owner = np.concatenate(spectra), np.concatenate(owner)

    model = fit_specparam(freqs, spectra, freq_range=(settings.fmin, settings.fmax),
                          aperiodic_mode=settings.aperiodic, **FOOOF_SETTINGS)
    peaks = model.results.get_params("peak")  # centre frequency, power, bandwidth, spectrum
    centre, peak_owner = peaks[:, 0], owner[peaks[:, 3].astype(int)]
    aperiodic = model.get_params("aperiodic")
    r2, error = model.results.get_metrics("gof", "rsquared"), model.results.get_metrics("error", "mae")
    n = len(selected)

    periodic = periodic_power(model.data.power_spectra, aperiodic, model.data.freqs, settings.aperiodic)
    periodic = np.stack([periodic[owner == s].mean(axis=0) for s in range(n)])
    edges = np.arange(settings.fmin, settings.fmax + settings.bin_hz / 2, settings.bin_hz)
    shares = peak_shares(centre, peak_owner, n, edges)
    preprint = {name: band for name, band in BANDS.items() if band[1] <= settings.fmax}
    resamples = np.random.default_rng(0).integers(0, n, (settings.n_resamples, n))
    resampled_shares, resampled = resampled_bands(shares, edges, preprint, resamples)
    found = bands_above(shares.mean(axis=0), edges, shares.mean())
    chosen = np.array([band_at(found, band) or (np.nan, np.nan) for band in preprint.values()])

    model_path = destination.band_choice(stream=stream, lock=settings.lock, aperiodic=settings.aperiodic,
                                         suffix="specparam", extension=".json")
    path = destination.band_choice(stream=stream, lock=settings.lock, aperiodic=settings.aperiodic,
                                   suffix="bands", extension=".h5")
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path.name, str(model_path.parent), save_results=True, save_settings=True)
    with h5py.File(path, "w") as handle:
        for name, data in {
            "freqs": model.data.freqs, "periodic": periodic,
            "periodic_interval": np.percentile(np.stack([periodic[p].mean(axis=0) for p in resamples]),
                                               [2.5, 97.5], axis=0),
            "edges": edges, "shares": shares, "shares_interval": np.percentile(resampled_shares, [2.5, 97.5], axis=0),
            "bands": chosen, "bands_resampled": np.stack([resampled[name] for name in preprint]),
            "preprint_bands": np.array(list(preprint.values())), "lines": SWEEP_LINES,
            "sweep": line_sweep(shares.mean(axis=0), SWEEP_LINES), "exponent": aperiodic[:, -1], "r2": r2,
            "error": error, "subject": owner,
        }.items():
            handle.create_dataset(name, data=data)
        handle.create_dataset("band_names", data=np.array(list(preprint), dtype="S"))
        handle.create_dataset("subjects", data=np.array(selected, dtype="S"))

    def interval(values: np.ndarray) -> list[float] | None:
        values = values[np.isfinite(values)]
        return np.percentile(values, [2.5, 97.5]).round(1).tolist() if values.size else None

    save_sidecar(path, {
        "stage": "band_choice", "stream": stream, **settings.__dict__, "specparam": {
            **{k: (str(v) if k == "max_n_peaks" else v) for k, v in FOOOF_SETTINGS.items()},
            "aperiodic_mode": settings.aperiodic},
        "subjects": selected, "n_spectra": int(spectra.shape[0]), "n_peaks": int(centre.size),
        "peaks_per_spectrum": round(centre.size / spectra.shape[0], 2),
        "exponent_range": [round(float(aperiodic[:, -1].min()), 3), round(float(aperiodic[:, -1].max()), 3)],
        "r2_range": [round(float(r2.min()), 3), round(float(r2.max()), 3)],
        "spectra_below_min_r2": int((r2 < settings.min_r2).sum()),
        "bands_above_mean_bin": [list(band) for band in found],
        "bands": {name: {"preprint": list(band), "here": None if np.isnan(edge[0]) else edge.tolist(),
                         "lower_edge_95": interval(resampled[name][:, 0]),
                         "upper_edge_95": interval(resampled[name][:, 1]),
                         "resamples_without": round(float(np.isnan(resampled[name][:, 0]).mean()), 3)}
                  for (name, band), edge in zip(preprint.items(), chosen)},
        "resamples_theta_alpha_one_band": round(float(np.mean(np.all(resampled["theta"] == resampled["alpha"], axis=1))), 3)
        if {"theta", "alpha"} <= set(resampled) else None,
        "model": str(model_path), "inputs": [str(p) for p in inputs],
    })
    return WorkflowResult(stage="band_choice", inputs=tuple(inputs), outputs=(model_path, path),
                          settings={"stream": stream, "subjects": selected, **settings.__dict__})


__all__ = ["SWEEP_LINES", "sensor_spectra", "band_choice"]
