import mne
import numpy as np

from meg_tokens.io import DerivativeLayout
from meg_tokens.meg.preprocessing import (
    REFERENCE_CHANNELS,
    STREAMS,
    artifact_events,
    repair_flux_jumps,
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


def test_a_flux_jump_is_repaired_and_a_large_slow_swing_is_left_alone():
    """Only a step is a jump: the level is restored on every channel it moved."""
    sfreq, seconds = 1200.0, 20
    times = np.arange(int(sfreq * seconds)) / sfreq
    rng = np.random.default_rng(0)
    clean = 1e-13 * rng.standard_normal((3, len(times)))
    clean[2] += _bumps(times, [12.0], 0.3, [4e-11])

    after = times >= 8.0
    stepped = clean.copy()
    stepped[0, after] += 3.6e-10
    stepped[1, after] -= 8e-12
    raw = mne.io.RawArray(
        stepped.copy(), mne.create_info(["MLT11", "MLT12", "MRT41"], sfreq, "mag"),
        verbose=False,
    )
    annotations, jumps = repair_flux_jumps(raw)

    assert len(jumps) == 1
    assert abs(jumps[0]["time"] - 8.0) < 0.005
    assert set(jumps[0]["steps"]) == {"MLT11", "MLT12"}
    assert abs(jumps[0]["steps"]["MLT11"] - 3.6e-10) < 1e-12
    assert list(annotations.description) == ["BAD_jump"]
    assert annotations.onset[0] < 8.0 < annotations.onset[0] + annotations.duration[0]

    repaired = raw.get_data()
    outside = np.abs(times - 8.0) > 0.03
    assert np.abs(repaired[:2, outside] - clean[:2, outside]).max() < 1e-12
    assert np.abs(repaired[:2, ~outside]).max() < 1e-12
    assert np.array_equal(repaired[2], stepped[2])
