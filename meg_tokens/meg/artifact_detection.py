"""Deriving artifact-detection thresholds, and testing that they work.

Two ideas, both needed by the bad-channel and bad-segment detection in preprocessing
(``docs/meg.md``).

**Thresholds as multiples of a measured noise floor.** :func:`robust_scale`
takes the median absolute deviation of *adjacent-sample* differences at the
native rate. Differencing high-passes the recording, so what comes back
reflects sample-to-sample noise rather than signal amplitude. Pooling that
across recordings gives one cohort scale, and a threshold expressed as a
multiple of it is transferable -- "20x the cohort noise floor" rather than a
constant in Tesla that means nothing on another dataset. The scale must be
frozen *before* any threshold is evaluated against it; tuning both together
can make any threshold look good.

**A falsification test for a detector.** :func:`enrichment_db` asks the only
question that matters about an annotator: does what it flagged carry more
in-band power than an equal amount of what it did not? The comparison spans
are duration-matched (:func:`matched_clean_spans`), so enrichment cannot be
manufactured by flagging more or less data.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import mne
import numpy as np
from mne.time_frequency import psd_array_welch

CHUNK_SECONDS = 30.0
MUSCLE_BAND = (110.0, 140.0)

__all__ = [
    "MUSCLE_BAND",
    "band_power",
    "enrichment_db",
    "matched_clean_spans",
    "robust_scale",
    "robust_scales",
]


def robust_scales(raw: mne.io.BaseRaw, picks: np.ndarray) -> np.ndarray:
    """Per-channel MAD of adjacent-sample differences, median over chunks.

    Chunked so a long recording is not held as a second full-size difference
    array, and taken at the native rate because the quantity of interest is
    sample-to-sample noise.
    """
    chunk = max(int(round(CHUNK_SECONDS * raw.info["sfreq"])), 1)
    scales = []
    for start in range(0, raw.n_times - 1, chunk):
        stop = min(start + chunk + 1, raw.n_times)
        data = raw.get_data(picks=picks, start=start, stop=stop)
        if data.shape[1] < 2:
            continue
        difference = np.diff(data, axis=1)
        center = np.median(difference, axis=1, keepdims=True)
        scales.append(1.4826 * np.median(np.abs(difference - center), axis=1))
    if not scales:
        raise ValueError("recording is too short to estimate a scale")
    return np.median(np.asarray(scales), axis=0)


def robust_scale(raw: mne.io.BaseRaw, picks: np.ndarray) -> float:
    """One scale for a recording: the median over channels, ignoring dead ones."""
    scales = robust_scales(raw, picks)
    alive = scales[scales > 0]
    if not alive.size:
        raise ValueError("every channel has zero scale")
    return float(np.median(alive))


def annotation_spans(
    annotations: mne.Annotations, sfreq: float, n_times: int
) -> list[tuple[int, int]]:
    """Annotation onsets and durations as sample spans, clipped to the data."""
    spans = []
    for onset, duration in zip(annotations.onset, annotations.duration):
        start = max(int(round(onset * sfreq)), 0)
        stop = min(start + max(int(round(duration * sfreq)), 1), n_times)
        if stop > start:
            spans.append((start, stop))
    return spans


def matched_clean_spans(
    bad: Sequence[tuple[int, int]], n_times: int
) -> list[tuple[int, int]]:
    """Earliest non-overlapping unflagged spans with the same durations.

    Duration matching is what makes an enrichment comparison fair: without it
    a detector could raise its score by flagging shorter or longer segments
    rather than better ones.
    """
    occupied = sorted(bad)
    result: list[tuple[int, int]] = []
    cursor = 0
    for start, stop in occupied:
        length = stop - start
        candidate = cursor
        while candidate + length <= n_times and any(
            candidate < b and candidate + length > a for a, b in occupied
        ):
            candidate = max(
                candidate + 1,
                max(b for a, b in occupied if candidate < b and candidate + length > a),
            )
        if candidate + length <= n_times:
            result.append((candidate, candidate + length))
            cursor = candidate + length
    return result


def band_power(
    raw: mne.io.BaseRaw,
    picks: np.ndarray,
    spans: Iterable[tuple[int, int]],
    band: tuple[float, float] = MUSCLE_BAND,
) -> float:
    """Median Welch power in ``band`` over the concatenated spans."""
    chunks = [
        data
        for start, stop in spans
        if (data := raw.get_data(picks=picks, start=start, stop=stop)).shape[1] >= 32
    ]
    if not chunks:
        return float("nan")
    data = np.concatenate(chunks, axis=1)
    n_per_seg = min(2048, data.shape[1])
    psd, _ = psd_array_welch(
        data,
        sfreq=float(raw.info["sfreq"]),
        fmin=band[0],
        fmax=band[1],
        n_fft=n_per_seg,
        n_per_seg=n_per_seg,
        n_jobs=1,
        verbose=False,
    )
    return float(np.nanmedian(np.maximum(psd, np.finfo(float).tiny)))


def enrichment_db(
    raw: mne.io.BaseRaw,
    picks: np.ndarray,
    annotations: mne.Annotations,
    band: tuple[float, float] = MUSCLE_BAND,
) -> float:
    """In-band power of the flagged spans over duration-matched clean spans, dB.

    Positive means the detector found something: the segments it flagged carry
    more power in ``band`` than an equal amount of data it left alone. This is
    a separate question from how *much* it flagged, and the two should not be
    traded against each other in one rule.
    """
    bad = annotation_spans(annotations, float(raw.info["sfreq"]), raw.n_times)
    if not bad:
        return float("nan")
    clean = matched_clean_spans(bad, raw.n_times)
    if not clean:
        return float("nan")
    tiny = np.finfo(float).tiny
    return float(
        10.0
        * np.log10(
            max(band_power(raw, picks, bad, band), tiny)
            / max(band_power(raw, picks, clean, band), tiny)
        )
    )
