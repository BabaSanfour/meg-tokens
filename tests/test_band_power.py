import mne
import numpy as np

from meg_tokens.meg import band_power as bp
from meg_tokens.meg.epoching import WINDOWS


def test_the_bands_meet_at_their_6_db_edges_without_overlapping():
    # A zero-phase FIR is down 6 dB half a transition outside the band.
    edges = [
        (low - bp.TRANSITIONS[band][0] / 2, high + bp.TRANSITIONS[band][1] / 2)
        for band, (low, high) in bp.BANDS.items()
    ]
    for (_, upper), (lower, _) in zip(edges, edges[1:]):
        assert upper <= lower or (upper, lower) == (61.0, 59.0)  # gamma and high gamma share 60 Hz


def test_a_band_keeps_its_own_frequency_and_rejects_its_neighbours():
    sfreq, seconds = 600.0, 20
    t = np.arange(int(sfreq * seconds)) / sfreq
    info = mne.create_info(["M1", "M2", "R1"], sfreq, ["mag", "mag", "ref_meg"])
    data = np.vstack([np.sin(2 * np.pi * 11 * t), np.sin(2 * np.pi * 6.5 * t) + np.sin(2 * np.pi * 20 * t), t])
    signal = bp.analytic(mne.io.RawArray(data, info, verbose=False), "alpha")

    assert signal.ch_names == ["M1", "M2"]
    z = signal.get_data()
    middle = slice(len(t) // 4, 3 * len(t) // 4)
    assert np.iscomplexobj(z)
    assert np.allclose(np.abs(z[0, middle]), 1, atol=0.02)
    assert np.abs(z[1, middle]).max() < 0.02


def test_the_window_is_the_epochs_grid():
    for lock, (tmin, tmax) in WINDOWS.items():
        offsets = bp.window(600.0, lock)
        assert offsets[0] / 600 == tmin and offsets[-1] / 600 == tmax


def test_samples_near_a_flux_jump_or_the_ends_are_invalid():
    info = mne.create_info(["M1"], 600.0, ["mag"])
    raw = mne.io.RawArray(np.zeros((1, 6000)), info, verbose=False)
    raw.set_annotations(mne.Annotations([5.0], [0.1], ["BAD_jump"], orig_time=raw.annotations.orig_time))
    raw.annotations.append(2.0, 0.5, "BAD_muscle")
    invalid = bp.invalid_samples(raw, half=60)
    assert invalid[:60].all() and invalid[-60:].all() and not invalid[60]
    assert invalid[3000 - 60:3060 + 61].all() and not invalid[3000 - 61] and not invalid[3060 + 61]
    assert not invalid[1200:1500].any()  # muscle is left to the epoch rejection


def test_power_is_squared_magnitude_and_a_vector_sums_its_three_orientations():
    rng = np.random.default_rng(0)
    surface = (rng.standard_normal((4, 5)) + 1j * rng.standard_normal((4, 5)))
    stc = mne.SourceEstimate(surface, [np.arange(2), np.arange(2)], 0, 0.1)
    assert np.allclose(bp._power(stc).data, np.abs(surface) ** 2)

    volume = rng.standard_normal((3, 3, 5)) + 1j * rng.standard_normal((3, 3, 5))
    vector = mne.VolVectorSourceEstimate(volume, [np.arange(3)], 0, 0.1)
    power = bp._power(vector)
    assert isinstance(power, mne.VolSourceEstimate)
    assert np.allclose(power.data, (np.abs(volume) ** 2).sum(axis=1))
