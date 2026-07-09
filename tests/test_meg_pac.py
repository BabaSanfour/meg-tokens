import numpy as np

from meg_tokens.features.pac import modulation_index, select_time_window


def test_modulation_index_detects_phase_locked_amplitude():
    phase_line = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    phase = np.tile(phase_line, (2, 2, 1))
    amplitude = np.ones_like(phase)
    amplitude[:, 0, :] = 1.5 + np.cos(phase[:, 0, :])

    mi = modulation_index(phase, amplitude, n_bins=18)

    assert mi.shape == (2,)
    assert mi[0] > mi[1]
    assert np.isclose(mi[1], 0.0)


def test_select_time_window_uses_last_axis():
    data = np.arange(2 * 3 * 10).reshape(2, 3, 10)
    times = np.arange(10, dtype=float) / 10.0

    selected = select_time_window(data, times, (0.2, 0.5))

    assert selected.shape == (2, 3, 4)
    assert selected[0, 0].tolist() == [2, 3, 4, 5]
