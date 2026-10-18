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


def test_means_per_condition_and_quintile_as_power_log10_and_db(tmp_path):
    import h5py
    import pandas as pd

    from meg_tokens.core import ProjectConfig
    from meg_tokens.io import DerivativeLayout
    from meg_tokens.workflows.band_power import band_power_means

    project = ProjectConfig(data_root=tmp_path)
    layout = DerivativeLayout(project.bids_root, task=project.task)
    n = 20
    go = pd.DataFrame({
        "condition": ["Fast"] * 10 + ["Slow"] * 10,
        "run": [1] * 20,
        "nTrialIndex": list(range(1, 11)) * 2,
        "trial_class_name": ["easy", "ambiguous"] * 10,
        "choice_side": [1.0] * 20,
        "dt_ms": list(range(10)) + list(range(100, 110)),
    })
    times = np.array([-0.4, -0.2, 0.0, 0.2])
    level = np.arange(1, n + 1, dtype=np.float32)
    shape = np.array([5.0, 1.0, 1.0, 10.0], np.float32)  # 1 over the baseline, -0.2 to 0 s
    power = level[:, None, None, None] * shape * np.ones((n, 2, 3, 4), np.float32)
    power[1, 0, 0, 3] = np.nan
    decision = go.assign(nTrialIndex=[99] + go["nTrialIndex"].tolist()[1:])  # trial 0 has no go epoch

    for lock, table in (("go", go), ("decision", decision)):
        path = layout.band_power(subject="H01", stream="dss", lock=lock, space=bp.PARCELLATION)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            handle["power"] = power
            handle["times"] = times
            for key in ("bands", "band_edges_hz", "sources", "n_sources"):
                handle[key] = np.arange(3)
        table.to_csv(layout.band_power_trials(subject="H01", stream="dss", lock=lock), sep="\t", index=False)

    band_power_means(project, "H01", stream="dss", locks=["go", "decision"])

    def read(lock):
        with h5py.File(layout.band_power_means(subject="H01", stream="dss", lock=lock), "r") as handle:
            return {key: handle[key][:] for key in handle}

    means = read("go")
    names = [c.decode() for c in means["conditions"]]
    easy, ambiguous, empty = (names.index(f"fast-{c}") for c in ("easy-right", "ambiguous-right", "easy-left"))
    assert means["condition_n"][easy] == 5 and means["condition_n"][empty] == 0
    assert means["condition_power"][easy, 0, 0, 0] == np.mean([1, 3, 5, 7, 9]) * 5
    assert means["condition_power"][ambiguous, 0, 0, 3] == np.mean([4, 6, 8, 10]) * 10  # trial 1's NaN ignored
    assert np.isclose(means["condition_log10"][easy, 0, 0, 1], np.mean(np.log10([1, 3, 5, 7, 9])))
    assert np.allclose(means["condition_db"][easy, 0, 0], 10 * np.log10(shape))
    assert np.isnan(means["condition_power"][empty]).all()
    assert np.allclose(means["log10_center"], np.nanmean(np.log10(power), axis=(0, 3)))
    assert means["quantile_power"][0, 1, 0, 0] == 1.5 * 5 and means["quantile_power"][9, 1, 0, 0] == 19.5 * 5
    assert means["quantile_dt_ms"][0] == 0.5 and means["quantile_dt_ms"][9] == 108.5

    means = read("decision")
    assert means["condition_n"][easy] == 5 and means["condition_n_db"][easy] == 4
    assert means["condition_power"][easy, 0, 0, 0] == np.mean([1, 3, 5, 7, 9]) * 5
    assert np.allclose(means["condition_db"][easy, 0, 0], 10 * np.log10(shape))
    assert means["quantile_n_db"][0] == 1


def test_fsaverage_means_equal_the_region_means_of_the_same_trials(tmp_path, monkeypatch):
    import h5py
    import pandas as pd

    import meg_tokens.meg.band_power as domain
    from meg_tokens.core import ProjectConfig
    from meg_tokens.io import DerivativeLayout
    from meg_tokens.workflows.band_power import band_power_means, band_power_means_fsaverage

    monkeypatch.setattr(domain, "fsaverage_vertices", lambda subjects_dir: [np.array([0, 1]), np.array([0])])
    project = ProjectConfig(data_root=tmp_path, subjects_dir=tmp_path)
    layout = DerivativeLayout(project.bids_root, task=project.task)
    fsaverage_root = tmp_path / "fsaverage"
    rng = np.random.default_rng(0)
    times = {"go": np.array([-0.4, -0.2, 0.0, 0.2]), "decision": np.linspace(-1.5, 2.0, 6)}
    go = pd.DataFrame({
        "condition": ["Fast"] * 10 + ["Slow"] * 10,
        "run": [1] * 20,
        "nTrialIndex": list(range(1, 11)) * 2,
        "trial_class_name": ["easy", "ambiguous"] * 10,
        "choice_side": [1.0] * 20,
        "dt_ms": rng.permutation(np.arange(20.0)),
    })
    tables = {"go": go, "decision": go.drop(index=3).reset_index(drop=True)}  # one trial lost at decision
    for lock, table in tables.items():
        power = rng.lognormal(size=(len(table), 1, 3, times[lock].size)).astype(np.float32)
        power[0, 0, 1, 2] = np.nan
        path = layout.band_power(subject="H01", stream="dss", lock=lock, space=bp.PARCELLATION)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            handle["power"] = power
            handle["times"] = times[lock]
            handle["bands"] = np.array([b"alpha"])
            for key in ("band_edges_hz", "sources", "n_sources"):
                handle[key] = np.arange(3)
        table.to_csv(layout.band_power_trials(subject="H01", stream="dss", lock=lock), sep="\t", index=False)
        for condition in ("Fast", "Slow"):
            rows = np.flatnonzero(table["condition"] == condition)
            part = DerivativeLayout(fsaverage_root, task=project.task).band_power_part(
                subject="H01", run="1", condition=condition, stream="dss", band="alpha", space="fsaverage")
            part.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(part, "a") as handle:
                handle[f"{lock}/power"] = power[rows, 0]
                handle[f"{lock}/times"] = times[lock]

    band_power_means(project, "H01", stream="dss", locks=["go", "decision"])
    band_power_means_fsaverage(project, "H01", stream="dss", band="alpha", locks=["go", "decision"],
                               fsaverage_root=fsaverage_root)

    for lock in ("go", "decision"):
        with h5py.File(layout.band_power_means(subject="H01", stream="dss", lock=lock), "r") as regions, \
                h5py.File(DerivativeLayout(fsaverage_root, task=project.task).band_power_means_space(
                    subject="H01", stream="dss", lock=lock, band="alpha", space="fsaverage"), "r") as vertices:
            for key in ("condition_power", "condition_log10", "condition_db",
                        "quantile_power", "quantile_log10", "quantile_db"):
                np.testing.assert_allclose(vertices[key][:], regions[key][:, 0], rtol=1e-5, equal_nan=True)
            for key in ("condition_n", "condition_n_db", "quantile_n", "quantile_n_db", "quantile_dt_ms"):
                np.testing.assert_array_equal(vertices[key][:], regions[key][:])
            np.testing.assert_allclose(vertices["log10_center"][:], regions["log10_center"][0], rtol=1e-6)
            assert vertices["vertices_lh"].size == 2


def test_deep_means_are_the_part_means_morphed_onto_fsaverages_grid(tmp_path, monkeypatch):
    import h5py
    import mne
    import pandas as pd

    import meg_tokens.meg.sources as sources
    import meg_tokens.workflows.band_power as workflow
    from meg_tokens.core import ProjectConfig
    from meg_tokens.io import DerivativeLayout

    class Doubling:  # stands in for the subject's morph: half the first source, onto three target sources
        subject_from = "H01"

        def apply(self, stc):
            return mne.VolSourceEstimate(np.repeat(0.5 * stc.data[:1], 3, axis=0), [np.arange(3)], 0, 1, "fsaverage")

    project = ProjectConfig(data_root=tmp_path, subjects_dir=tmp_path)
    layout = DerivativeLayout(project.bids_root, task=project.task)
    morph_path = layout.deep_morph(subject="H01")
    morph_path.parent.mkdir(parents=True, exist_ok=True)
    morph_path.touch()
    monkeypatch.setattr(mne, "read_source_morph", lambda path: Doubling())
    monkeypatch.setattr(sources, "deep_volume", lambda subject, subjects_dir: [{"nuse": 3, "vertno": np.arange(3)}])
    monkeypatch.setattr(workflow, "deep_structures", lambda src, subjects_dir: {"Left-Caudate": np.arange(3)})
    deep_root = tmp_path / "deep"
    trials = pd.DataFrame({
        "condition": ["Fast"] * 4 + ["Slow"] * 4, "run": [1] * 8, "nTrialIndex": list(range(1, 5)) * 2,
        "trial_class_name": ["easy"] * 8, "choice_side": [1.0] * 8, "dt_ms": np.arange(8.0),
    })
    trials.to_csv(layout.band_power_trials(subject="H01", stream="dss", lock="go"), sep="\t", index=False)
    power = np.random.default_rng(1).lognormal(size=(8, 2, 4)).astype(np.float32)
    for condition in ("Fast", "Slow"):
        part = DerivativeLayout(deep_root, task=project.task).band_power_part(
            subject="H01", run="1", condition=condition, stream="dss", band="alpha", space="aseg")
        part.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(part, "w") as handle:
            handle["go/power"] = power[trials["condition"] == condition]
            handle["go/times"] = np.array([-0.4, -0.2, 0.0, 0.2])
            handle["vertno"] = np.array([7, 9])

    workflow.deep_power_means(project, "H01", stream="dss", band="alpha", deep_root=deep_root, locks=["go"])

    path = DerivativeLayout(deep_root, task=project.task).band_power_means_space(
        subject="H01", stream="dss", lock="go", band="alpha", space="fsaverageaseg")
    with h5py.File(path, "r") as handle:
        fast = handle["conditions"][:].tolist().index(b"fast-easy-right")
        # divided by the morph of a constant (0.5), the first source's mean comes back whole
        np.testing.assert_allclose(handle["condition_power"][fast], np.repeat(power[:4, :1].mean(axis=0), 3, axis=0),
                                   rtol=1e-6)
        np.testing.assert_allclose(handle["coverage"][:], 0.5)
        assert [s.decode() for s in handle["structure"]] == ["Left-Caudate"] * 3


def test_task_events_place_every_go_token_press_and_feedback():
    import pandas as pd

    from meg_tokens.meg.epoching import TASK_EVENTS, task_events

    trials = pd.DataFrame({
        "tGO": [1000.0, 1000.0], "tTime": [[1200.0, 1400.0, 1500.0, 1520.0, 1700.0], [1200.0, 1400.0]],
        "nTokenDir": [[1, 2, 1, 2, 1], [2, 2]], "tEnterTarget": [1500.0, 0.0], "tTrialEnd": [1700.0, 0.0],
    })
    events, present, covariates, trial = task_events(np.array([10.0, 20.0]), trials, "Fast", sfreq=100.0, first_samp=50)
    by_name = {name: events[events[:, 2] == code, 0].tolist() for name, code in TASK_EVENTS.items()}
    assert by_name["go"] == [1050, 2050]
    assert by_name["token right before press"] == [1070]  # trial start 9 s: +1.2 s
    assert by_name["token left before press"] == [1090, 2070, 2090]  # trial 2 has no press: all before it
    # the token at the press's time gives way to the press, the one at the feedback's to the feedback
    assert by_name["press"] == [1100] and by_name["feedback"] == [1120]
    assert by_name["token after press fast"] == [1102]
    assert "token after press slow" not in present
    place = {name: dict(zip(events[:, 0].tolist(), values.tolist())) for name, values in covariates.items()}
    assert {k: v for k, v in place["place of token right before press"].items() if v} == {1070: 1.0}
    assert {k: v for k, v in place["place of token left before press"].items() if v} == {1090: 2.0, 2070: 1.0, 2090: 2.0}
    assert trial.tolist() == [0] * 6 + [1] * 3


def _events_at(onsets, covariates=None):
    """``task_events``' interface for regression tests: onsets in seconds per event type (codes 1, 2, ...)."""
    names = {code: name for code, name in enumerate(onsets, start=1)}
    times = np.concatenate(list(onsets.values()))
    codes = np.concatenate([np.full(len(v), code) for code, v in zip(names, onsets.values())])
    values = {name: np.asarray(v, float) for name, v in (covariates or {}).items()}
    order = np.argsort(times)

    def at(sfreq, first_samp):
        samples = first_samp + np.round(times[order] * sfreq).astype(int)
        keep = np.r_[True, np.diff(samples) > 0]  # one event per sample, as task_events keeps
        events = np.column_stack([samples[keep], np.zeros(keep.sum(), int), codes[order][keep]])
        return events, {name: code for code, name in names.items()}, {k: v[order][keep] for k, v in values.items()}, order[keep]

    return at


def _smooth_raw(data, sfreq=600.0):
    import mne

    return mne.io.RawArray(data, mne.create_info([f"MLC1{i}" for i in range(len(data))], sfreq, "mag"), verbose=False)


def _response(t, kind):
    """Band-limited responses on -0.1 to +0.3 s."""
    shape = np.sin(2 * np.pi * 5 * t) if kind == "a" else -np.exp(-((t - 0.1) / 0.04) ** 2)
    return shape * np.hanning(t.size) * 1e-12


def test_event_responses_are_estimated_and_taken_out():
    from meg_tokens.meg.band_power import RESPONSE_SFREQ, remove_event_responses

    rng = np.random.default_rng(0)
    sfreq, n = 600.0, 600 * 120
    lags = np.arange(-60, 181)  # -0.1 to +0.3 s
    onsets = {"a": np.arange(60, 114, 0.37), "b": np.arange(61.05, 114, 0.53)}  # on the 10 ms grid
    onsets["b"] = onsets["b"][~np.isin(np.round(onsets["b"] * 100), np.round(onsets["a"] * 100))]  # never together
    clean = rng.normal(size=(2, n)) * 1e-13
    data = clean.copy()
    for kind, times in onsets.items():
        for onset in np.round(times * sfreq).astype(int):
            data[:, onset + lags] += _response(lags / sfreq, kind)

    cleaned, fits = remove_event_responses(_smooth_raw(data), _events_at(onsets), window=(-0.1, 0.3))
    assert fits[0]["a"].info["sfreq"] == RESPONSE_SFREQ
    np.testing.assert_allclose(fits[0]["a"].data[0], _response(np.arange(-10, 31) / RESPONSE_SFREQ, "a"), atol=5e-14)
    assert np.std(cleaned.get_data() - clean) < 0.1 * np.std(clean)


def test_a_response_scaled_by_a_covariate_is_taken_out():
    from meg_tokens.meg.band_power import remove_event_responses

    rng = np.random.default_rng(1)
    sfreq, n = 600.0, 600 * 120
    lags = np.arange(-60, 181)
    times = np.arange(60, 114, 0.37)
    place = rng.integers(1, 10, times.size).astype(float)
    clean = rng.normal(size=(1, n)) * 1e-13
    data = clean.copy()
    for onset, value in zip(np.round(times * sfreq).astype(int), place):
        data[0, onset + lags] += _response(lags / sfreq, "a") + 0.2 * value * _response(lags / sfreq, "b")

    cleaned, fits = remove_event_responses(_smooth_raw(data), _events_at({"a": times}, {"place of a": place}),
                                           window=(-0.1, 0.3))
    np.testing.assert_allclose(fits[0]["place of a"].data[0], 0.2 * _response(np.arange(-10, 31) / 100.0, "b"), atol=5e-14)
    assert np.std(cleaned.get_data() - clean) < 0.1 * np.std(clean)


def test_fitted_on_its_own_events_the_regression_takes_out_noise_and_on_other_folds_it_adds_some():
    import mne

    from meg_tokens.meg.band_power import remove_event_responses

    rng = np.random.default_rng(2)
    noise = mne.filter.filter_data(rng.normal(size=(1, 600 * 60)), 600.0, None, 30.0, verbose=False) * 1e-13
    times = np.arange(1, 58, 0.2)  # no response at all; the fit's band holds all the noise

    in_sample, _ = remove_event_responses(_smooth_raw(noise), _events_at({"a": times}), window=(-0.5, 0.5))
    crossed, fits = remove_event_responses(_smooth_raw(noise), _events_at({"a": times}), n_folds=10, window=(-0.5, 0.5))
    assert len(fits) == 10
    assert np.var(in_sample.get_data()) < 0.99 * np.var(noise) < np.var(noise) < np.var(crossed.get_data())  # 101 lags


def test_the_induced_copy_has_its_own_files_beside_the_first(tmp_path):
    import h5py
    import pandas as pd

    from meg_tokens.core import ProjectConfig
    from meg_tokens.io import DerivativeLayout
    from meg_tokens.workflows.band_power import band_power_means

    project = ProjectConfig(data_root=tmp_path)
    layout = DerivativeLayout(project.bids_root, task=project.task)
    trials = pd.DataFrame({
        "condition": ["Fast"] * 4 + ["Slow"] * 4, "run": [1] * 8, "nTrialIndex": list(range(1, 5)) * 2,
        "trial_class_name": ["easy"] * 8, "choice_side": [1.0] * 8, "dt_ms": np.arange(8.0),
    })
    for variant, level in ((None, 1.0), ("induced", 2.0)):
        path = layout.band_power(subject="H01", stream="dss", lock="go", space=bp.PARCELLATION, variant=variant)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            handle["power"] = np.full((8, 1, 2, 3), level, np.float32)
            handle["times"] = np.array([-0.2, 0.0, 0.2])
            for key in ("bands", "band_edges_hz", "sources", "n_sources"):
                handle[key] = np.arange(2)
    trials.to_csv(layout.band_power_trials(subject="H01", stream="dss", lock="go"), sep="\t", index=False)

    band_power_means(project, "H01", stream="dss", locks=["go"], induced=True)

    induced = layout.band_power_means(subject="H01", stream="dss", lock="go", variant="induced")
    assert induced.name.endswith("desc-go-induced_powermeans.h5")
    assert not layout.band_power_means(subject="H01", stream="dss", lock="go").exists()
    with h5py.File(induced, "r") as handle:
        assert np.allclose(handle["condition_power"][:][handle["condition_n"][:] > 0], 2.0)
