import pytest
import numpy as np
import mne
from meg_tokens.meg.time_frequency import (
    _slide_window,
    compute_band_power,
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

def test_compute_band_power_morlet_multitaper():
    sfreq = 100.0
    data = np.random.randn(2, 100)
    bands = {'beta': (15.0, 25.0)}
    
    res_morlet = compute_band_power(
        data, sfreq=sfreq, freq_bands=bands,
        method='morlet', width=20, step=5, return_mne=False
    )
    assert res_morlet['beta'].shape == (2, 17)
    
    res_mt = compute_band_power(
        data, sfreq=sfreq, freq_bands=bands,
        method='multitaper', width=20, step=5, return_mne=False
    )
    assert res_mt['beta'].shape == (2, 17)

def test_compute_band_power_mne_stc():
    sfreq = 100.0
    n_vertices = 4
    n_times = 100
    data_2d = np.random.randn(n_vertices, n_times)
    
    vertices = [np.array([0, 1]), np.array([2, 3])]
    stc = mne.SourceEstimate(data_2d, vertices=vertices, tmin=0.0, tstep=1.0/sfreq)
    
    bands = {'theta': (4.0, 8.0)}
    res = compute_band_power(
        stc, sfreq=sfreq, freq_bands=bands,
        method='hilbert', width=20, step=5, return_mne=True
    )
    
    assert 'theta' in res
    assert isinstance(res['theta'], mne.SourceEstimate)
    assert res['theta'].data.shape == (n_vertices, 17)
    assert len(res['theta'].vertices[0]) == 2
    assert len(res['theta'].vertices[1]) == 2

def test_compute_band_power_mne_vector_stc():
    sfreq = 100.0
    n_vertices = 4
    n_times = 100
    data_3d = np.random.randn(n_vertices, 3, n_times)
    vertices = [np.array([0, 1]), np.array([2, 3])]
    v_stc = mne.VectorSourceEstimate(data_3d, vertices=vertices, tmin=0.0, tstep=1.0/sfreq)
    
    bands = {'theta': (4.0, 8.0)}
    with pytest.raises(TypeError, match="VectorSourceEstimate"):
        compute_band_power(
            v_stc, sfreq=sfreq, freq_bands=bands,
            method='hilbert', width=20, step=5, return_mne=True
        )

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
