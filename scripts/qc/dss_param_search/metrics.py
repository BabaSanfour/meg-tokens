"""One scoring battery, applied identically to every configuration.

Three quantities per configuration, and they answer different questions:

*   **removal** -- how much of the event-locked evoked field went. It is
    reported, never optimised: a larger subspace always removes more, signal
    included, so ranking on it rewards taking dimensions out.
*   **reference correlation** -- whether what went *is* the artifact. The
    dominant time course of the removed signal against the reference channel.
    The reference waveform plays no part in the fit; only its timing does, so
    this is independent of both the bias window and the removal figure.
*   **alpha cost** -- what it cost in a band the analysis actually reads.

The evoked-field scoring window is fixed for every configuration, deliberately
wider than the widest bias window. Scoring on the window a configuration was
fit with would let a narrow template win by being measured narrowly.
"""

from __future__ import annotations

import numpy as np

import mne

#: Fixed comparison window for the event-locked evoked field, in seconds. Wide
#: enough to contain every bias window in the grid. Not a physiological claim:
#: at 60 bpm it spans a neighbouring beat, but it spans the same neighbouring
#: beat for every configuration.
#:
#: It is the right window for *ranking* configurations and the wrong one for
#: quoting an absolute. It is far wider than the artifacts themselves, so the
#: denominator carries a lot of time with no artifact in it -- which a cleaner
#: does not remove and should not. Measured on H01's cardiac stage: 62.6% here
#: against 80.5% on a window matched to the artifact. Report both.
SCORING_WINDOW = (-0.35, 0.8)

#: Windows matched to each artifact, for absolute figures.
#:
#: Cardiac spans the QRS complex and the T wave, which runs to roughly 400 ms
#: after the R peak -- the same reasoning that says a 200 ms bias template
#: truncates it. A blink's deflection and recovery span about 100-400 ms, and a
#: saccade is faster but scored on the same window so the two are comparable.
NARROW_WINDOWS = {
    "ecg": (-0.05, 0.40),
    "blink": (-0.10, 0.40),
    "saccade": (-0.10, 0.40),
}

#: The band the alpha cost is read in. This is the FOOOF alpha definition used
#: throughout the project, not MNE's 8-12.
ALPHA_BAND = (9.0, 13.0)


def band_power_db(raw: mne.io.BaseRaw, picks: np.ndarray, band: tuple[float, float]) -> float:
    """Mean power across channels and frequencies in one band, in dB."""
    spectrum = raw.compute_psd(
        method="welch", fmin=band[0], fmax=band[1], picks=picks,
        n_fft=2048, n_per_seg=2048, verbose=False,
    )
    return float(10.0 * np.log10(spectrum.get_data().mean()))


def evoked_rms(
    raw: mne.io.BaseRaw,
    picks: np.ndarray,
    samples: np.ndarray,
    window: tuple[float, float] = SCORING_WINDOW,
) -> float:
    """RMS of the event-locked average field over the fixed scoring window."""
    sfreq = float(raw.info["sfreq"])
    first, last = int(round(window[0] * sfreq)), int(round(window[1] * sfreq))
    epochs = [
        raw.get_data(picks=picks, start=int(sample) + first, stop=int(sample) + last)
        for sample in samples
        if int(sample) + first >= 0 and int(sample) + last <= raw.n_times
    ]
    if not epochs:
        return float("nan")
    return float(np.sqrt(np.mean(np.mean(epochs, axis=0) ** 2)))


def dominant_time_course(removed: np.ndarray) -> np.ndarray:
    """The removed signal projected onto its own strongest spatial pattern.

    The top eigenvector of the channel covariance is the top left singular
    vector, reached without the economy SVD's ``(n_channels, n_times)`` factor
    -- which at 270 channels and 315 s at 1200 Hz is 800 MB we would allocate
    fifty times per recording to then discard.
    """
    centred = removed - removed.mean(axis=1, keepdims=True)
    covariance = centred @ centred.T / centred.shape[1]
    _, vectors = np.linalg.eigh(covariance)
    return vectors[:, -1] @ removed


def reference_correlation(removed: np.ndarray, reference: np.ndarray) -> float:
    """|r| between the removed signal's dominant time course and a reference channel.

    Absolute, because DSS fixes neither the sign of a component nor of a
    topography, and the sign carries no information about whether the right
    thing was removed.
    """
    course = dominant_time_course(removed)
    if not np.std(course) > 0 or not np.std(reference) > 0:
        return float("nan")
    return float(abs(np.corrcoef(course, reference)[0, 1]))


def reference_correlation_top(removed: np.ndarray, reference: np.ndarray, k: int = 3) -> float:
    """Best |r| against a reference among the removed subspace's top ``k`` patterns.

    The single-component form is right for a stage biased at one artifact. A
    whole-recording cleaner removes cardiac *and* ocular at once, so its
    dominant component tracks whichever is larger and the other scores near
    zero however well it was removed. Taking the best of the top few asks the
    fair question of both.
    """
    centred = removed - removed.mean(axis=1, keepdims=True)
    covariance = centred @ centred.T / centred.shape[1]
    _, vectors = np.linalg.eigh(covariance)
    if not np.std(reference) > 0:
        return float("nan")
    best = 0.0
    for index in range(1, min(k, vectors.shape[1]) + 1):
        course = vectors[:, -index] @ removed
        if np.std(course) > 0:
            best = max(best, abs(float(np.corrcoef(course, reference)[0, 1])))
    return best


def score_cleaning(
    before: mne.io.BaseRaw,
    after: mne.io.BaseRaw,
    picks: np.ndarray,
    events: dict[str, np.ndarray],
    references: dict[str, np.ndarray],
) -> dict[str, float]:
    """The full battery for one cleaner, on both scoring windows.

    Applied identically to the DSS chain, to ICA and to SSP. The whole point of
    a shared battery is that no arm is scored on terms of its own -- B5's
    verdict rested on absolute limits only the challenger had to clear.
    """
    removed = before.get_data(picks=picks) - after.get_data(picks=picks)
    scores: dict[str, float] = {}
    for key in ("ecg", "blink", "saccade"):
        samples = events[key]
        for label, window in (("wide", SCORING_WINDOW), ("narrow", NARROW_WINDOWS[key])):
            baseline = evoked_rms(before, picks, samples, window)
            cleaned = evoked_rms(after, picks, samples, window)
            scores[f"{key}_removed_{label}_percent"] = (
                100.0 * (1.0 - cleaned / baseline) if baseline == baseline and baseline > 0
                else float("nan")
            )
    for name, reference in references.items():
        scores[f"reference_r_{name}"] = reference_correlation_top(removed, reference)
    before_alpha = band_power_db(before, picks, ALPHA_BAND)
    before_muscle = band_power_db(before, picks, (110.0, 140.0))
    scores["alpha_delta_db"] = band_power_db(after, picks, ALPHA_BAND) - before_alpha
    scores["muscle_reduction_db"] = (
        before_muscle - band_power_db(after, picks, (110.0, 140.0))
    )
    del removed
    return scores


def artifact_presence(
    raw: mne.io.BaseRaw,
    events: dict[str, np.ndarray],
    picks: np.ndarray,
) -> dict[str, float]:
    """How much of each artifact this recording actually contains.

    Without this a stage that correctly removes nothing cannot be told from one
    that missed something. Two DSS stages selecting zero components on H11 and
    H01 look identical in the results and are opposite findings: the arena
    measured H11 at 0.00 dB muscle from an independent method, so zero there is
    right, and 0.91 dB on H01, so zero there is a miss.
    """
    duration = float(raw.times[-1])
    muscle_seconds = float(
        sum(
            annotation["duration"]
            for annotation in raw.annotations
            if str(annotation["description"]).lower().startswith("bad_muscle")
        )
    )
    return {
        "duration_seconds": duration,
        "ecg_per_minute": 60.0 * len(events["ecg"]) / duration,
        "blink_per_minute": 60.0 * len(events["blink"]) / duration,
        "saccade_per_minute": 60.0 * len(events["saccade"]) / duration,
        "muscle_annotated_seconds": muscle_seconds,
        "muscle_annotated_fraction": muscle_seconds / duration,
        "alpha_db": band_power_db(raw, picks, ALPHA_BAND),
        "muscle_band_db": band_power_db(raw, picks, (110.0, 140.0)),
    }
