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


def test_a_part_needs_its_morph_map_built_beforehand(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError, match="band-power-prepare"):
        bp.morph_map("H01", tmp_path)
    (tmp_path / "morph-maps").mkdir()
    (tmp_path / "morph-maps" / "fsaverage-H01-morph.fif").touch()
    assert bp.morph_map("H01", tmp_path).name == "fsaverage-H01-morph.fif"


def test_means_per_condition_and_decision_time_quintile(tmp_path):
    import h5py
    import pandas as pd

    from meg_tokens.core import ProjectConfig
    from meg_tokens.io import DerivativeLayout
    from meg_tokens.meg.evoked import CONDITIONS
    from meg_tokens.workflows.band_power import band_power_means

    project = ProjectConfig(data_root=tmp_path)
    layout = DerivativeLayout(project.bids_root, task=project.task)
    n = 20
    trials = pd.DataFrame({
        "condition": ["Fast"] * 10 + ["Slow"] * 10,
        "trial_class_name": ["easy", "ambiguous"] * 10,
        "choice_side": [1.0] * 20,
        "dt_ms": list(range(10)) + list(range(100, 110)),
    })
    power = np.arange(n, dtype=np.float32)[:, None, None, None] * np.ones((n, 2, 3, 4), np.float32)
    power[0, 0, 0, 0] = np.nan
    path = layout.band_power(subject="H01", stream="dss", lock="go", space=bp.PARCELLATION)
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as handle:
        handle["power"] = power
        for key in ("times", "bands", "band_edges_hz", "sources", "n_sources"):
            handle[key] = np.arange(3)
    trials.to_csv(layout.band_power_trials(subject="H01", stream="dss", lock="go"), sep="\t", index=False)

    band_power_means(project, "H01", stream="dss", locks=["go"])

    with h5py.File(layout.band_power_means(subject="H01", stream="dss", lock="go"), "r") as handle:
        condition = handle["condition_power"][:]
        n_condition = dict(zip([c.decode() for c in handle["conditions"]], handle["condition_n"][:]))
        quantile = handle["quantile_power"][:]
        dt = handle["quantile_dt_ms"][:]
    assert n_condition["fast-easy-right"] == 5 and n_condition["fast-easy-left"] == 0
    fast_easy = CONDITIONS.index("fast-easy-right")
    assert condition[fast_easy, 0, 0, 0] == np.mean([2, 4, 6, 8])  # trial 0's NaN ignored
    assert condition[fast_easy, 1, 0, 0] == np.mean([0, 2, 4, 6, 8])
    assert np.isnan(condition[CONDITIONS.index("fast-easy-left")]).all()
    assert quantile[0, 1, 0, 0] == 0.5 and quantile[9, 1, 0, 0] == 18.5  # fast q1, slow q5
    assert dt[0] == 0.5 and dt[9] == 108.5
