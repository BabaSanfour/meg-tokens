import mne
import numpy as np

from meg_tokens.io import DerivativeLayout
from meg_tokens.meg.preprocessing import (
    REFERENCE_CHANNELS,
    STREAMS,
    artifact_events,
    stream_paths,
)


def test_every_stream_gets_a_distinct_path(tmp_path):
    """Each cleaner writes its own derivative for one recording."""
    layout = DerivativeLayout(tmp_path)
    paths = [stream_paths(layout, "H1", "Slow", 1, s)["cleaned"] for s in STREAMS]
    assert len(set(paths)) == len(STREAMS)


def _bumps(times, centres, width, amplitudes):
    return sum(
        a * np.exp(-0.5 * ((times - c) / width) ** 2) for c, a in zip(centres, amplitudes)
    )


def test_saccades_are_split_by_the_direction_of_the_deflection():
    """Opposite directions cancel in one average, so each gets its own events."""
    sfreq, seconds = 600.0, 60
    times = np.arange(int(sfreq * seconds)) / sfreq
    rng = np.random.default_rng(0)

    rightward = np.arange(2.0, seconds - 2, 2.6)
    leftward = rightward[:-5] + 1.3
    heog = _bumps(times, rightward, 0.08, np.ones(len(rightward))) - _bumps(
        times, leftward, 0.08, np.ones(len(leftward))
    )
    beats = np.arange(1.0, seconds - 1, 0.85)
    ecg = _bumps(times, beats, 0.012, np.ones(len(beats)))
    blinks = np.arange(3.0, seconds - 3, 3.7)
    veog = _bumps(times, blinks, 0.1, np.ones(len(blinks)))

    data = np.vstack([veog, heog, ecg]) + 1e-3 * rng.standard_normal((3, len(times)))
    raw = mne.io.RawArray(
        data, mne.create_info(list(REFERENCE_CHANNELS), sfreq, "eeg"), verbose=False
    )
    events = artifact_events(raw)

    def _found(samples, centres):
        return np.allclose(np.sort(samples) / sfreq, centres, atol=0.05)

    assert len(events["saccade_positive"]) == len(rightward)
    assert len(events["saccade_negative"]) == len(leftward)
    assert _found(events["saccade_positive"], rightward)
    assert _found(events["saccade_negative"], leftward)
