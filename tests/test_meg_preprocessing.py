import mne
import numpy as np

from meg_tokens.io import DerivativeLayout
import pytest

from meg_tokens.meg.preprocessing import (
    HEAD_COILS,
    REFERENCE_CHANNELS,
    STREAMS,
    artifact_events,
    remove_head_movement,
    remove_tonic_muscle_dss,
    remove_tonic_muscle_ica,
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


def _shaped_noise(rng, n_sources, n_times, sfreq, exponent):
    """Gaussian noise whose power goes as frequency to ``exponent``, 1-150 Hz."""
    freqs = np.fft.rfftfreq(n_times, 1 / sfreq)
    gain = np.zeros_like(freqs)
    band = (freqs >= 1) & (freqs <= 150)
    gain[band] = freqs[band] ** (exponent / 2)
    noise = np.fft.irfft(np.fft.rfft(rng.standard_normal((n_sources, n_times)), axis=1) * gain, n=n_times, axis=1)
    return noise / noise.std(axis=1, keepdims=True)


def _recording_with_tonic_muscle(with_muscle):
    """Sixty sensors of brain-like noise; muscle, rising with frequency, on six."""
    rng = np.random.default_rng(0)
    sfreq, n_times, n_channels = 600.0, 36000, 60
    times = np.arange(n_times) / sfreq
    # each source waxes and wanes at its own rate, which is what lets ICA converge
    waxing = 1 + 0.8 * np.sin(2 * np.pi * np.outer(rng.uniform(0.05, 0.5, n_channels), times))
    data = rng.standard_normal((n_channels, n_channels)) @ (
        _shaped_noise(rng, n_channels, n_times, sfreq, -1.0) * waxing
    )
    if with_muscle:
        envelope = 1 + 0.8 * np.sin(2 * np.pi * 0.2 * times)
        muscle = _shaped_noise(rng, 1, n_times, sfreq, 1.0)[0] * envelope
        data[:6] += 20 * np.outer(rng.uniform(0.5, 1.0, 6), muscle)
    info = mne.create_info([f"M{i:02d}" for i in range(n_channels)], sfreq, "mag")
    return mne.io.RawArray(data * 1e-13, info, verbose=False)


def _band_power(raw, low, high):
    return (raw.copy().filter(low, high, verbose=False).get_data() ** 2).mean(axis=1)


def test_each_tonic_muscle_stage_removes_muscle_and_nothing_else():
    for stage in (remove_tonic_muscle_dss, remove_tonic_muscle_ica):
        clean = _recording_with_tonic_muscle(with_muscle=False)
        untouched = clean.get_data().copy()
        _, n_removed, _ = stage(clean)
        assert n_removed == 0
        np.testing.assert_allclose(clean.get_data(), untouched, rtol=1e-9, atol=1e-22)

        raw = _recording_with_tonic_muscle(with_muscle=True)
        reference = _band_power(_recording_with_tonic_muscle(with_muscle=False), 65, 100)
        alpha_before = _band_power(raw, 8, 13)
        assert (_band_power(raw, 65, 100)[:6] > 5 * reference[:6]).all()
        _, n_removed, record = stage(raw)
        assert n_removed == record["components"] == 1
        gamma = _band_power(raw, 65, 100)
        # the six muscle sensors are back at the level they have without muscle
        np.testing.assert_allclose(gamma[:6], reference[:6], rtol=0.3)
        np.testing.assert_allclose(gamma[6:], reference[6:], rtol=0.05)
        np.testing.assert_allclose(_band_power(raw, 8, 13)[6:], alpha_before[6:], rtol=0.05)



def _recording_with_head_coils(moving):
    """Twenty sensors of noise; when the head moves, each also follows the coils."""
    rng = np.random.default_rng(0)
    sfreq, n_times = 600.0, 36000
    times = np.arange(n_times) / sfreq
    brain = 1e-13 * rng.standard_normal((20, n_times))
    coils = 0.07 + 1e-6 * rng.standard_normal((9, n_times))
    if moving:
        drift = np.cumsum(rng.standard_normal((3, n_times)), axis=1)
        drift = 1e-3 * (drift - drift.mean(axis=1, keepdims=True)) / drift.std(axis=1, keepdims=True)
        coils += np.repeat(drift, 3, axis=0) * (1 + 0.2 * np.sin(times))
    gain = 1e-9 * rng.standard_normal((20, 9))
    data = brain + (gain @ (coils - coils.mean(axis=1, keepdims=True)) if moving else 0)
    names = [f"M{i:02d}" for i in range(20)] + [f"{name}-2104" for name in HEAD_COILS]
    info = mne.create_info(names, sfreq, ["mag"] * 20 + ["misc"] * 9)
    return mne.io.RawArray(np.vstack([data, coils]), info, verbose=False), brain


def test_head_movement_is_regressed_out_and_a_still_head_is_left_alone():
    raw, brain = _recording_with_head_coils(moving=True)
    assert (raw.get_data(picks="mag") - brain).std() > 5 * brain.std()
    _, record = remove_head_movement(raw)
    assert (raw.get_data(picks="mag") - brain).std() < 0.05 * brain.std()
    assert record["variance_removed_median"] > 0.9

    still, brain = _recording_with_head_coils(moving=False)
    _, record = remove_head_movement(still)
    # nine regressors over 36,000 samples fit sqrt(9 / 36000) = 1.6% of the noise by chance
    assert (still.get_data(picks="mag") - brain).std() < 0.03 * brain.std()
    assert record["variance_removed_max"] < 0.01


def test_a_recording_without_the_head_coils_raises():
    raw, _ = _recording_with_head_coils(moving=False)
    raw.drop_channels([raw.ch_names[-1]])
    with pytest.raises(ValueError, match="HLC0033 absent"):
        remove_head_movement(raw)
