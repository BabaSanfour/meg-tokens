import json

import h5py
import mne
import numpy as np

from meg_tokens.analysis.bands import band_at, bands_above, line_sweep, peak_shares, resampled_bands
from meg_tokens.core import BandChoiceConfig, ProjectConfig, SpectralConfig
from meg_tokens.io import DerivativeLayout


def test_bands_are_the_runs_above_the_line_and_a_band_is_matched_by_its_centre():
    edges = np.arange(1.0, 11.0)
    values = np.array([0, 3, 3, 0, 1, 4, 4, 4, 0, 2.0])[:9]
    found = bands_above(values, edges, 2.0)
    assert found == [(2.0, 4.0), (6.0, 9.0)]
    assert band_at(found, (6.5, 8.5)) == (6.0, 9.0)
    assert band_at(found, (4.0, 5.0)) is None


def test_each_subject_weighs_the_same_and_the_sweep_marks_bins_above_each_line():
    edges = np.array([0.0, 1.0, 2.0])
    shares = peak_shares(np.array([0.5, 0.5, 0.5, 1.5, 1.5]), np.array([0, 0, 0, 0, 1]), 2, edges)
    np.testing.assert_allclose(shares, [[75.0, 25.0], [0.0, 100.0]])  # subject 0's four peaks count as much as one
    np.testing.assert_array_equal(line_sweep(np.array([1.5, 3.0]), np.array([0.5, 1.0, 2.0])),
                                  [[True, True], [False, True], [False, False]])


def test_resampling_subjects_gives_each_band_its_edges_or_none():
    edges = np.arange(0.0, 5.0)
    shares = np.array([[10.0, 0.0, 10.0, 0.0], [10.0, 0.0, 0.0, 10.0]])
    means, found = resampled_bands(shares, edges, {"low": (0.0, 1.0), "high": (2.0, 3.0)}, np.array([[0, 0], [1, 1]]))
    np.testing.assert_allclose(means, shares)
    np.testing.assert_allclose(found["low"], [[0.0, 1.0], [0.0, 1.0]])
    np.testing.assert_allclose(found["high"], [[2.0, 3.0], [np.nan, np.nan]])


def test_sensor_spectra_average_every_kept_epoch_of_the_subjects_runs(tmp_path):
    from meg_tokens.workflows.spectral import sensor_spectra

    project = ProjectConfig(data_root=tmp_path)
    layout = DerivativeLayout(project.bids_root, task=project.task)
    info = mne.create_info(["MLC11", "MLC12", "MRC11"], 600.0, "mag")
    rng = np.random.default_rng(0)
    times = np.arange(-900, 1201) / 600.0  # the press lock's -1.5 to +2.0 s
    counts = {"Fast": 3, "Slow": 5}
    for condition, n in counts.items():
        data = rng.normal(size=(n, 3, times.size)) * 1e-13 + 1e-12 * np.sin(2 * np.pi * 10 * times)
        path = layout.epochs(subject="H01", run="1", condition=condition, alignment="enter", stream="dss")
        path.parent.mkdir(parents=True, exist_ok=True)
        mne.EpochsArray(data, info, tmin=times[0], verbose=False).save(path, verbose=False)
        path.with_suffix(".json").write_text(json.dumps({"n_epochs": n}))

    result = sensor_spectra(project, "H01", stream="dss", settings=SpectralConfig())

    with h5py.File(result.outputs[0], "r") as handle:
        freqs, psd = handle["freqs"][:], handle["psd"][:]
        assert int(handle["n_epochs"][()]) == 8
        assert [c.decode() for c in handle["channels"]] == ["MLC11", "MLC12", "MRC11"]
    assert freqs[1] - freqs[0] == 0.5 and freqs[0] == 1.0 and freqs[-1] == 90.0
    assert freqs[np.argmax(psd[0])] == 10.0
    assert result.outputs[0].name == "sub-H01_task-tokens_proc-dss_desc-enter_psd.h5"


def test_band_choice_finds_the_cohorts_bands_and_draws_them(tmp_path):
    from meg_tokens.reports.bands import render_band_choice
    from meg_tokens.workflows.spectral import band_choice

    project = ProjectConfig(data_root=tmp_path)
    layout = DerivativeLayout(project.bids_root, task=project.task)
    freqs = np.arange(1.0, 90.01, 0.5)
    rng = np.random.default_rng(0)
    subjects = ["H01", "H02", "H03", "H04"]
    for subject in subjects:
        logged = np.array([1 - np.log10(freqs) + 0.5 * np.exp(-(freqs - 10 - rng.normal()) ** 2 / 3)
                           + 0.25 * np.exp(-(freqs - 6) ** 2 / 1) + 0.2 * np.exp(-(freqs - 20) ** 2 / 10)
                           + 0.02 * rng.normal(size=freqs.size) for _ in range(12)])
        path = layout.sensor_spectra(subject=subject, stream="dss", lock="enter")
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            handle["psd"], handle["freqs"] = 10 ** logged, freqs

    settings = BandChoiceConfig(n_resamples=50)
    result = band_choice(project, stream="dss", settings=settings, subjects=subjects)

    model, bands = result.outputs
    assert model.name == "sub-group_task-tokens_proc-dss_desc-enter_specparam.json" and model.is_file()
    with h5py.File(bands, "r") as handle:
        names = [n.decode() for n in handle["band_names"]]
        chosen = dict(zip(names, handle["bands"][:]))
        assert names == ["theta", "alpha", "beta"]
        assert handle["periodic"].shape == (4, handle["freqs"].shape[0])
        assert handle["shares"].shape == (4, 34) and handle["bands_resampled"].shape == (3, 50, 2)
        assert handle["r2"].shape == (48,)
    assert chosen["alpha"][0] <= 10 < chosen["alpha"][1] and chosen["theta"][0] <= 6 < chosen["theta"][1]
    sidecar = json.loads(bands.with_suffix(".json").read_text())
    assert sidecar["n_spectra"] == 48 and sidecar["bands"]["alpha"]["preprint"] == [9.0, 13.0]

    drawn = render_band_choice(project, streams=["dss"], settings=settings)
    assert sorted(p.name for p in drawn.outputs) == [
        "sub-group_task-tokens_proc-dss_desc-enter_bands.pdf",
        "sub-group_task-tokens_proc-dss_desc-enter_bands.png",
        "sub-group_task-tokens_proc-dss_desc-enter_specparam.pdf",
    ]
    assert all(p.is_file() and p.parent.name == "fig" for p in drawn.outputs)
