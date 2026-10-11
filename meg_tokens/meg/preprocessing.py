"""Preprocessing: one shared trunk, then DSS and ICA, each persisted with the
artifact it removed. Specification and rationale: ``docs/meg.md``.

    read + compensate - ZapLine - bandpass - bad channels - bad segments (1)
        -> DSS | ICA -> ASR (muscle) - bad segments (2) - resample - persist
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
from mne.preprocessing import (
    ICA,
    annotate_amplitude,
    annotate_muscle_zscore,
    find_bad_channels_lof,
    find_ecg_events,
)
from scipy.signal import butter, find_peaks, lfilter

from meg_tokens.io import sidecar_path

LINE_FREQ = 60.0
LINE_HARMONICS = 2
HIGH_PASS = 0.5
LOW_PASS = 150.0
RESAMPLE_HZ = 600.0
COMPENSATION_GRADE = 3

VEOG_CHANNEL = "EEG057"
HEOG_CHANNEL = "EEG058"
ECG_CHANNEL = "EEG059"
REFERENCE_CHANNELS = (VEOG_CHANNEL, HEOG_CHANNEL, ECG_CHANNEL)

# 20x and 0.002x the cohort noise floor, 6.243e-14 T
# (``robust_scale`` in ``scripts/qc/artifact_detection.py``).
AMPLITUDE_PEAK_T = 1.2485926067065662e-12
AMPLITUDE_FLAT_T = 1.2485926067065662e-16
MUSCLE_Z = 7.0
MUSCLE_BAND = (110.0, 140.0)

# Ocular events: band-limited excursions past a multiple of the channel's MAD.
OCULAR_BAND = (0.5, 7.0)
OCULAR_SCALE_MULTIPLE = 3.0
OCULAR_MIN_SEPARATION = 0.4

# DSS: event-locked windows and the eigenvalue truncation -- the one selection
# setting that bounds how many dimensions a stage removes.
QRS_WINDOW = (-0.1, 0.2)
BLINK_WINDOW = (-0.2, 0.5)
SACCADE_WINDOW = (-0.1, 0.4)
DSS_N_COMPONENTS = 40
DSS_MIN_EVENTS = 10

# ASR, the muscle step both streams share. Its statistics -- calibration and
# the per-window decision to reconstruct -- see only the data above the
# high-pass, so what triggers it is muscle rather than the alpha and evoked
# power that dominate the broadband recording.
ASR_STATISTICS_HIGH_PASS = 80.0
ASR_STATISTICS_FILTER_ORDER = 4
ASR_CUTOFF = 20.0
ASR_WINDOW = 0.5
ASR_MAX_DIMS = 0.25

STREAMS = ("dss", "ica")

__all__ = [
    "STREAMS",
    "artifact_events",
    "build_trunk",
    "clean_dss",
    "clean_ica",
    "finish",
    "dss_subtract",
    "ocular_events",
    "read_staged_raw",
    "remove_muscle",
    "run_stream",
    "save_stream",
    "stream_paths",
]


def read_staged_raw(path: str | Path, *, preload: bool = True) -> tuple[mne.io.BaseRaw, int]:
    """Read a staged recording at the cohort's common compensation grade.

    Four of the 319 staged runs -- H11 Fast 3/4 and Slow 3/4 -- were acquired
    at grade 0 while the rest are grade 3. CTF keeps the grade in the .res4
    header and applies compensation on read, so it cannot be corrected when
    staging; it is corrected here, and the acquired grade is returned so the
    sidecar can record that it was.
    """
    raw = mne.io.read_raw_ctf(str(path), preload=preload, verbose=False)
    source_grade = int(raw.compensation_grade)
    if source_grade != COMPENSATION_GRADE:
        raw.load_data()
        raw.apply_gradient_compensation(COMPENSATION_GRADE)
    if raw.compensation_grade != COMPENSATION_GRADE:
        raise ValueError(
            f"could not reach grade {COMPENSATION_GRADE}, at {raw.compensation_grade}"
        )
    return raw, source_grade


def _ocular_excursions(
    raw: mne.io.BaseRaw, channel: str, scale_multiple: float
) -> tuple[np.ndarray, np.ndarray]:
    """Where a reference channel leaves its resting level, and which way."""
    band = (
        raw.copy()
        .pick([channel])
        .filter(*OCULAR_BAND, fir_design="firwin", verbose=False)
        .get_data()[0]
    )
    centred = band - np.median(band)
    scale = 1.4826 * float(np.median(np.abs(centred)))
    if not scale > 0:
        raise ValueError(f"{channel} has no variation to threshold against")
    peaks, _ = find_peaks(
        np.abs(centred),
        height=scale_multiple * scale,
        distance=max(int(round(OCULAR_MIN_SEPARATION * raw.info["sfreq"])), 1),
    )
    peaks = np.asarray(peaks, dtype=int)
    return peaks, centred[peaks]


def ocular_events(
    raw: mne.io.BaseRaw,
    channel: str,
    *,
    scale_multiple: float = OCULAR_SCALE_MULTIPLE,
) -> np.ndarray:
    """Blink or saccade excursions, data-relative.

    Thresholded on the channel's own median absolute deviation, which does not
    move when the recording contains a few large excursions. ``scale_multiple``
    trades event count against purity and is the axis a parameter search moves;
    the default is the value validated against the cohort blink rate.
    """
    return _ocular_excursions(raw, channel, scale_multiple)[0]


def artifact_events(raw: mne.io.BaseRaw) -> dict[str, np.ndarray]:
    """R peaks, blinks and saccades, all data-relative.

    Saccades are split by the sign of the horizontal EOG deflection: the two
    directions have opposite field patterns, and averaged together they cancel.

    A missing reference channel raises rather than yielding an empty list: a
    stream scored against no events would report a perfect, meaningless result.
    """
    missing = [name for name in REFERENCE_CHANNELS if name not in raw.ch_names]
    if missing:
        raise ValueError(f"reference channels absent from the recording: {missing}")
    ecg = find_ecg_events(raw, ch_name=ECG_CHANNEL, verbose=False)[0][:, 0]
    saccades, deflection = _ocular_excursions(raw, HEOG_CHANNEL, OCULAR_SCALE_MULTIPLE)
    return {
        "ecg": np.asarray(ecg, dtype=int) - raw.first_samp,
        "blink": ocular_events(raw, VEOG_CHANNEL),
        "saccade_positive": saccades[deflection > 0],
        "saccade_negative": saccades[deflection < 0],
    }


def spectral_outlier_channels(
    raw: mne.io.BaseRaw, picks: np.ndarray, *, z_threshold: float = 4.0
) -> tuple[list[str], dict[str, float]]:
    """Channels with excessive broadband power, the paper's own criterion.

    Robust z of per-channel 1-150 Hz power in dB across the array. Scores are
    returned for every channel so a threshold can be revisited from the cohort
    distribution rather than trusted in advance.
    """
    spectrum = raw.compute_psd(
        method="welch", fmin=1.0, fmax=LOW_PASS, picks=picks,
        n_fft=2048, n_per_seg=2048, verbose=False,
    )
    power = 10.0 * np.log10(spectrum.get_data().mean(axis=1))
    centre = np.median(power)
    scale = 1.4826 * np.median(np.abs(power - centre))
    scores = (power - centre) / scale if scale > 0 else np.zeros_like(power)
    names = [raw.ch_names[index] for index in picks]
    return (
        [name for name, z in zip(names, scores) if abs(z) > z_threshold],
        {name: float(z) for name, z in zip(names, scores)},
    )


def build_trunk(raw: mne.io.BaseRaw, source_grade: int) -> tuple[mne.io.BaseRaw, dict]:
    """ZapLine, bandpass, bad channels, bad segments -- once, shared by every stream.

    ZapLine runs on the unfiltered recording, while the harmonics still exist.
    Bad channels are recorded and scored but not dropped: the decision is left
    to a later stage that can see the cohort distribution. Returns the trunk and
    what it found, keyed as the sidecar records it.
    """
    from mne_denoise.zapline import ZapLine

    picks = mne.pick_types(raw.info, meg=True, ref_meg=False, eeg=False, exclude=[])
    raw = ZapLine(
        sfreq=raw.info["sfreq"], line_freq=LINE_FREQ, n_harmonics=LINE_HARMONICS
    ).fit_transform(raw)
    raw.filter(
        HIGH_PASS, LOW_PASS,
        picks=mne.pick_types(raw.info, meg=True, eeg=True, exclude=[]),
        fir_design="firwin", verbose=False,
    )

    amplitude, amplitude_bads = annotate_amplitude(
        raw, peak={"mag": AMPLITUDE_PEAK_T}, flat={"mag": AMPLITUDE_FLAT_T},
        bad_percent=5, min_duration=0.005, picks=picks, verbose=False,
    )
    muscle, _ = annotate_muscle_zscore(
        raw, threshold=MUSCLE_Z, ch_type="mag", min_length_good=0.1,
        filter_freq=MUSCLE_BAND, n_jobs=1, verbose=False,
    )
    annotations = amplitude + muscle
    raw.set_annotations(raw.annotations + annotations)

    spectral_bads, scores = spectral_outlier_channels(raw, picks)
    lof_bads = find_bad_channels_lof(raw, picks=picks, verbose=False)

    return raw, {
        "compensation_grade_source": source_grade,
        "compensation_applied": source_grade != COMPENSATION_GRADE,
        "source_sfreq": float(raw.info["sfreq"]),
        "duration_seconds": float(raw.times[-1]),
        "n_meg_channels": int(len(picks)),
        # The union: a channel any detector flagged is reported, not voted on.
        "bad_channels": sorted(set(amplitude_bads) | set(spectral_bads) | set(lof_bads)),
        "bad_channels_amplitude": sorted(amplitude_bads),
        "bad_channels_spectral": sorted(spectral_bads),
        "bad_channels_lof": sorted(lof_bads),
        "channel_scores": scores,
        "trunk_annotated_seconds": float(np.sum(annotations.duration)),
        "trunk_annotations": int(len(annotations)),
    }


def _artifact_of(before: mne.io.BaseRaw, cleaned: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """What a stream removed, as a recording in its own right.

    Defined as the residual rather than reconstructed per method, so the
    streams' artifact estimates are the same quantity and can be compared.
    """
    artifact = before.copy()
    picks = mne.pick_types(artifact.info, meg=True, ref_meg=False, eeg=False, exclude=[])
    artifact._data[picks] = before.get_data(picks=picks) - cleaned.get_data(picks=picks)
    return artifact


def clean_ica(raw: mne.io.BaseRaw, events: dict[str, np.ndarray]) -> tuple[mne.io.BaseRaw, int, dict]:
    """20-component ICA on the filtered copy, both EOG channels and the ECG.

    The paper describes two bipolar EOG channels but its code passed only the
    vertical one, leaving saccade artifact invisible to component selection.
    A missing reference raises: silently continuing with an empty exclusion
    list would produce a run that "succeeded" having removed nothing.
    """
    missing = [name for name in REFERENCE_CHANNELS if name not in raw.ch_names]
    if missing:
        raise ValueError(f"reference channels absent from the recording: {missing}")

    ica = ICA(n_components=20, random_state=0, max_iter="auto")
    ica.fit(
        raw,
        picks=mne.pick_types(raw.info, meg=True, ref_meg=False, eeg=False, exclude=[]),
        reject_by_annotation=True,
    )
    exclude: set[int] = set()
    for channel in (VEOG_CHANNEL, HEOG_CHANNEL):
        exclude.update(ica.find_bads_eog(raw, ch_name=channel)[0])
    exclude.update(ica.find_bads_ecg(raw, ch_name=ECG_CHANNEL)[0])
    # Plain ints, not the numpy scalars `find_bads_*` returns. The sidecar is
    # JSON and the audit record for what each stream removed; `int64` there
    # either raises or, with a stringifying fallback, records the component
    # indices as `["3", "7"]` and calls it done.
    ica.exclude = sorted(int(index) for index in exclude)

    return (
        ica.apply(raw.copy()),
        len(ica.exclude),
        {"n_components": 20, "excluded": list(ica.exclude)},
    )


def dss_subtract(
    raw: mne.io.BaseRaw,
    bias,
    *,
    n_components: int | None = DSS_N_COMPONENTS,
    **selection,
) -> tuple[mne.io.BaseRaw, int]:
    """Fit DSS on the magnetometers and subtract the selected components.

    ``selection`` passes further DSS selection settings through (the parameter
    search varies them); production passes none and takes the library's.
    """
    from mne_denoise.dss import DSS

    model = DSS(
        bias=bias, n_components=n_components, n_select="auto",
        component_action="subtract", **selection,
    )
    cleaned = raw.copy()
    picks = mne.pick_types(cleaned.info, meg=True, ref_meg=False, eeg=False, exclude=[])
    subset = cleaned.copy().pick([cleaned.ch_names[index] for index in picks])
    result = model.fit_transform(subset)
    data = result.get_data() if hasattr(result, "get_data") else np.asarray(result)
    if data.shape != (len(picks), cleaned.n_times):
        raise ValueError(f"DSS returned {data.shape}, expected {(len(picks), cleaned.n_times)}")
    cleaned._data[picks] = data
    # ``n_selected_``, not ``n_select_`` -- the latter silently returned the
    # getattr default, so every DSS stream reported removing 0 dimensions and
    # a rank of 270 regardless of what it actually subtracted.
    if model.n_selected_ is None:
        raise ValueError("DSS did not report a component count after fitting")
    n_removed = int(model.n_selected_)
    return cleaned, n_removed


def clean_dss(raw: mne.io.BaseRaw, events: dict[str, np.ndarray]) -> tuple[mne.io.BaseRaw, int, dict]:
    """Cardiac, blinks, then saccades in each direction -- each a DSS bias
    locked to one artifact's events.

    Too few heartbeats or blinks to average means the reference channel or its
    detector failed, and raises. Too few saccades in one direction is a short
    recording in which the subject rarely looked that way: the stage is left
    out and the sidecar shows the direction's events with no components.
    """
    from mne_denoise.dss.denoisers import CycleAverageBias

    sfreq = float(raw.info["sfreq"])
    current, removed, detail = raw, 0, {}

    for label, key, window, required in (
        ("qrs", "ecg", QRS_WINDOW, True),
        ("blink", "blink", BLINK_WINDOW, True),
        ("saccade_positive", "saccade_positive", SACCADE_WINDOW, False),
        ("saccade_negative", "saccade_negative", SACCADE_WINDOW, False),
    ):
        first = int(round(window[0] * sfreq))
        last = int(round(window[1] * sfreq))
        samples = np.asarray(events[key], dtype=int)
        # CycleAverageBias requires a complete window for every event.
        samples = samples[(samples + first >= 0) & (samples + last <= raw.n_times)]
        detail[f"{label}_events"] = int(samples.size)
        if samples.size < DSS_MIN_EVENTS:
            if required:
                raise ValueError(f"{key} has {samples.size} usable events, too few to average")
            detail[f"{label}_components"] = 0
            continue
        current, n = dss_subtract(
            current,
            CycleAverageBias(
                event_samples=samples, window=window,
                window_unit="seconds", sfreq=sfreq,
            ),
        )
        removed += n
        detail[f"{label}_components"] = n

    return current, removed, detail


CLEANERS = {"dss": clean_dss, "ica": clean_ica}


def remove_muscle(raw: mne.io.BaseRaw) -> tuple[mne.io.BaseRaw, dict]:
    """ASR on the magnetometers, in place -- the same step after either cleaner.

    Calibrated on the recording itself and run on the arrays, annotated spans
    included: the muscle segments are the ones it is here for. The statistics
    filter is set on the state by hand because the library offers no high-pass
    at this frequency; calibration sees the filtered data, reconstruction acts
    on the unfiltered recording.
    """
    from mne_denoise.asr import calibrate_asr, process_asr

    sfreq = float(raw.info["sfreq"])
    picks = mne.pick_types(raw.info, meg=True, ref_meg=False, eeg=False, exclude=[])
    data = raw.get_data(picks=picks)
    b, a = butter(
        ASR_STATISTICS_FILTER_ORDER, ASR_STATISTICS_HIGH_PASS, btype="highpass", fs=sfreq
    )
    state, _ = calibrate_asr(
        lfilter(b, a, data, axis=1), sfreq,
        cutoff=ASR_CUTOFF, window_length=ASR_WINDOW, filter_kind="none",
    )
    state.filter_b, state.filter_a = b, a
    state.filter_zi = np.zeros((len(picks), ASR_STATISTICS_FILTER_ORDER))
    cleaned, diagnostics = process_asr(
        data, sfreq, state, window_length=ASR_WINDOW, max_dims=ASR_MAX_DIMS,
    )
    raw._data[picks] = cleaned
    return raw, {
        "reconstructed_fraction": float(diagnostics["fraction_reconstructed_samples"]),
        "max_components": int(diagnostics["max_components_reconstructed"]),
    }


def finish(raw: mne.io.BaseRaw) -> tuple[mne.io.BaseRaw, dict]:
    """Bad segments pass 2, then resample -- last, after the low-pass.

    The second annotation pass is the paper's own stated purpose: residual
    artifact "not captured by any component". Only a post-clean pass can see it.
    The trunk's muscle annotations stay on the recording through ASR, so what
    it reconstructed can still be excluded downstream.
    Resampling before the low-pass aliases, and before ZapLine it folds the
    higher harmonics onto real signal.
    """
    residual, _ = annotate_muscle_zscore(
        raw, threshold=MUSCLE_Z, ch_type="mag", min_length_good=0.1,
        filter_freq=MUSCLE_BAND, n_jobs=1, verbose=False,
    )
    raw.set_annotations(raw.annotations + residual)
    raw.resample(RESAMPLE_HZ, verbose=False)
    return raw, {
        "residual_annotations": int(len(residual)),
        "residual_seconds": float(np.sum(residual.duration)),
        "sfreq": float(raw.info["sfreq"]),
    }


def run_stream(
    trunk: mne.io.BaseRaw, events: dict[str, np.ndarray], stream: str
) -> tuple[mne.io.BaseRaw, mne.io.BaseRaw, dict]:
    """One cleaner, then the shared muscle step and tail: the cleaned
    recording, what the stream removed, and the stream's sidecar fields."""
    if stream not in CLEANERS:
        raise ValueError(f"unknown stream {stream!r}; expected one of {sorted(CLEANERS)}")
    cleaned, n_removed, detail = CLEANERS[stream](trunk, events)
    cleaned, muscle = remove_muscle(cleaned)
    artifact = _artifact_of(trunk, cleaned)
    cleaned, tail = finish(cleaned)
    artifact.resample(RESAMPLE_HZ, verbose=False)
    return cleaned, artifact, {
        "stream": stream,
        "n_dimensions_removed": n_removed,
        **{f"muscle_{key}": value for key, value in muscle.items()},
        **{f"tail_{key}": value for key, value in tail.items()},
        **{f"detail_{key}": value for key, value in detail.items()},
    }


def stream_paths(
    layout, subject: str, condition: str, run: int | str, stream: str
) -> dict[str, Path]:
    """Where one stream's three outputs live, via the layout that owns the contract.

    The ``proc`` entity carries the stream. ``desc`` keeps the condition and
    grows the artifact tag onto it, per the hyphen-joined tag list in
    ``docs/data_contract.md``.
    """
    cleaned = layout.preprocessed_raw(
        subject=subject, run=str(run), condition=condition, processing=stream
    )
    artifact = layout.path(
        subject=subject,
        run=str(run),
        processing=stream,
        description=f"{condition.lower()}-artifact",
        suffix="raw",
        extension=".fif",
    )
    return {
        "cleaned": cleaned,
        "artifact": artifact,
        "sidecar": sidecar_path(cleaned),
    }


def save_stream(
    cleaned: mne.io.BaseRaw,
    artifact: mne.io.BaseRaw,
    stream_record: dict,
    trunk_record: dict,
    layout,
    subject: str,
    condition: str,
    run: int | str,
    *,
    overwrite: bool = True,
) -> dict[str, Path]:
    """Persist a stream's cleaned recording, its artifact, and the sidecar."""
    import json

    paths = stream_paths(layout, subject, condition, run, stream_record["stream"])
    paths["cleaned"].parent.mkdir(parents=True, exist_ok=True)
    cleaned.save(str(paths["cleaned"]), overwrite=overwrite, verbose=False)
    artifact.save(str(paths["artifact"]), overwrite=overwrite, verbose=False)

    sidecar = {
        "condition": condition.lower(),
        "run": str(run),
        **trunk_record,
        **stream_record,
        # The inverse operator needs the rank; both cleaners remove dimensions.
        # ASR does not count: it reconstructs within windows, not a fixed subspace.
        "rank": trunk_record["n_meg_channels"] - stream_record["n_dimensions_removed"],
    }
    with paths["sidecar"].open("w", encoding="utf-8") as handle:
        # No `default=` fallback. A value JSON cannot represent is a type that
        # should not have reached the sidecar, and stringifying it writes a
        # plausible wrong record instead of failing where the cause is.
        json.dump(sidecar, handle, indent=1, sort_keys=True)
    return paths
