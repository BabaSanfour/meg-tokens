import numpy as np
import mne
from meg_tokens.features.time_frequency import (
    compute_window_times,
    _slide_window,
    compute_hilbert_band_features,
    compute_band_power,
    compute_psd,
    fit_specparam,
    rescale_baseline,
    DEFAULT_BANDS
)

def test_slide_window():
    # Test simple 1D/2D slide window
    data = np.arange(10, dtype=float)  # 0, 1, ..., 9
    # Width 4, step 2.
    # Windows:
    # 0: [0, 1, 2, 3] -> mean = 1.5
    # 2: [2, 3, 4, 5] -> mean = 3.5
    # 4: [4, 5, 6, 7] -> mean = 5.5
    # 6: [6, 7, 8, 9] -> mean = 7.5
    res = _slide_window(data, width=4, step=2)
    assert np.allclose(res, [1.5, 3.5, 5.5, 7.5])
    
    # Test width larger than signal
    res_large = _slide_window(data, width=15, step=2)
    assert np.allclose(res_large, [4.5])

def test_compute_band_power_numpy():
    # Test hilbert on EpochsArray
    sfreq = 100.0
    n_channels = 2
    n_times = 200
    t = np.arange(n_times) / sfreq
    sine = np.sin(2 * np.pi * 10.0 * t)
    data = np.vstack([sine, sine])
    
    info = mne.create_info(n_channels, sfreq, ch_types='mag')
    epochs = mne.EpochsArray(data[np.newaxis, :, :], info)
    
    bands = {'alpha': (8.0, 12.0)}
    
    res = compute_band_power(
        epochs, sfreq=sfreq, freq_bands=bands,
        method='hilbert', width=50, step=10, return_mne=False
    )
    
    assert 'alpha' in res
    assert res['alpha'].shape == (2, 1, 16)


def test_compute_psd_uses_modern_epochs_api():
    sfreq = 100.0
    n_times = 300
    times = np.arange(n_times) / sfreq
    data = np.stack([
        np.sin(2 * np.pi * 10.0 * times),
        np.cos(2 * np.pi * 15.0 * times),
    ])[np.newaxis, :, :]
    info = mne.create_info(["MEG001", "MEG002"], sfreq, ch_types="mag")
    epochs = mne.EpochsArray(data, info, verbose=False)

    psds, freqs = compute_psd(epochs, fmin=2.0, fmax=40.0, method="welch", n_fft=128, n_overlap=0)

    assert psds.shape[0] == 1
    assert psds.shape[1] == 2
    assert psds.shape[2] == len(freqs)
    assert freqs.min() >= 2.0
    assert freqs.max() <= 40.0


def test_fit_specparam_group_model_returns_channel_parameters():
    freqs = np.linspace(2.0, 40.0, 120)
    spectra = np.vstack([
        1.0 / freqs + 0.4 * np.exp(-((freqs - 10.0) ** 2) / 4.0),
        1.0 / freqs + 0.3 * np.exp(-((freqs - 20.0) ** 2) / 6.0),
    ])

    model = fit_specparam(
        freqs,
        spectra,
        freq_range=(2.0, 40.0),
        max_n_peaks=3,
        min_peak_height=0.01,
        peak_threshold=1.0,
        n_jobs=1,
    )
    params = model.to_df()

    assert len(params) == 2
    assert {"offset", "exponent"}.issubset(params.columns)


def test_compute_hilbert_band_features_preserves_shape_and_feature_names():
    sfreq = 100.0
    n_times = 400
    t = np.arange(n_times) / sfreq
    data = np.stack([
        np.sin(2 * np.pi * 10.0 * t),
        np.cos(2 * np.pi * 10.0 * t),
    ]).reshape(1, 2, n_times)

    out = compute_hilbert_band_features(
        data,
        sfreq=sfreq,
        freq_bands={"alpha": (8.0, 12.0)},
        features=("amplitude", "power", "phase", "sigfilt"),
        n_jobs=1,
    )

    assert set(out) == {"alpha"}
    assert set(out["alpha"]) == {"amplitude", "power", "phase", "sigfilt"}
    assert out["alpha"]["amplitude"].shape == data.shape
    assert out["alpha"]["power"].shape == data.shape
    assert out["alpha"]["phase"].shape == data.shape
    assert out["alpha"]["sigfilt"].shape == data.shape
    assert np.all(out["alpha"]["amplitude"] >= 0.0)
    assert np.nanmax(np.abs(out["alpha"]["phase"])) <= np.pi

def test_compute_band_power_morlet_multitaper():
    sfreq = 100.0
    n_times = 200
    t = np.arange(n_times) / sfreq
    data = np.vstack([
        np.sin(2 * np.pi * 20.0 * t),
        np.cos(2 * np.pi * 20.0 * t),
    ])
    bands = {'beta': (15.0, 25.0)}
    
    res_morlet = compute_band_power(
        data, sfreq=sfreq, freq_bands=bands,
        method='morlet', width=20, step=5, return_mne=False,
        n_cycles=2.0
    )
    assert res_morlet['beta'].shape == (2, 37)
    
    res_mt = compute_band_power(
        data, sfreq=sfreq, freq_bands=bands,
        method='multitaper', width=20, step=5, return_mne=False,
        n_cycles=2.0
    )
    assert res_mt['beta'].shape == (2, 37)

def test_compute_band_power_mne_stc():
    sfreq = 100.0
    n_vertices = 4
    n_times = 200
    t = np.arange(n_times) / sfreq
    data_2d = np.vstack([
        np.sin(2 * np.pi * 6.0 * t),
        np.cos(2 * np.pi * 6.0 * t),
        np.sin(2 * np.pi * 6.0 * t + 0.5),
        np.cos(2 * np.pi * 6.0 * t + 0.5),
    ])
    
    vertices = [np.array([0, 1]), np.array([2, 3])]
    stc = mne.SourceEstimate(data_2d, vertices=vertices, tmin=0.0, tstep=1.0/sfreq)
    
    bands = {'theta': (4.0, 8.0)}
    res = compute_band_power(
        stc, sfreq=sfreq, freq_bands=bands,
        method='hilbert', width=20, step=5, return_mne=True
    )
    
    assert 'theta' in res
    assert isinstance(res['theta'], mne.SourceEstimate)
    assert res['theta'].data.shape == (n_vertices, 37)
    assert len(res['theta'].vertices[0]) == 2
    assert len(res['theta'].vertices[1]) == 2

def test_compute_band_power_mne_vector_stc():
    sfreq = 100.0
    n_vertices = 4
    n_times = 200
    t = np.arange(n_times) / sfreq
    base = np.vstack([
        np.sin(2 * np.pi * 6.0 * t),
        np.cos(2 * np.pi * 6.0 * t),
        np.sin(2 * np.pi * 6.0 * t + 0.5),
        np.cos(2 * np.pi * 6.0 * t + 0.5),
    ])
    data_3d = np.stack([base, base * 0.5, base * 0.25], axis=1)
    vertices = [np.array([0, 1]), np.array([2, 3])]
    v_stc = mne.VectorSourceEstimate(data_3d, vertices=vertices, tmin=0.0, tstep=1.0/sfreq)
    
    bands = {'theta': (4.0, 8.0)}
    res = compute_band_power(
        v_stc, sfreq=sfreq, freq_bands=bands,
        method='hilbert', width=20, step=5, return_mne=True
    )

    assert isinstance(res['theta'], mne.VectorSourceEstimate)
    assert res['theta'].data.shape == (n_vertices, 3, 37)

def test_compute_window_times():
    times = compute_window_times(tmin=-0.2, sfreq=100.0, n_times=100, width=20, step=10)
    assert np.allclose(times[:3], [-0.105, -0.005, 0.095])
    assert len(times) == 9

def test_rescale_baseline():
    times = np.linspace(0.0, 2.0, num=200)
    power_data = np.ones((2, 200)) * 8.0
    power_data[:, :100] = 4.0
    
    baseline = (0.0, 1.0)
    
    res_percent = rescale_baseline(power_data, times, baseline, method='percent')
    assert np.allclose(res_percent[:, :100], 0.0)
    assert np.allclose(res_percent[:, 100:], 1.0)
    
    res_ratio = rescale_baseline(power_data, times, baseline, method='ratio')
    assert np.allclose(res_ratio[:, :100], 1.0)
    assert np.allclose(res_ratio[:, 100:], 2.0)
    
    res_diff = rescale_baseline(power_data, times, baseline, method='difference')
    assert np.allclose(res_diff[:, :100], 0.0)
    assert np.allclose(res_diff[:, 100:], 4.0)
    
    res_log = rescale_baseline(power_data, times, baseline, method='logratio')
    assert np.allclose(res_log[:, :100], 0.0)
    assert np.allclose(res_log[:, 100:], 10 * np.log10(2.0))
