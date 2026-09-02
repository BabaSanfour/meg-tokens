import os
import json
import tempfile
import mne
import numpy as np
import pandas as pd
import pytest
from meg_tokens.io import DerivativeLayout
from meg_tokens.meg.epoching import realign_epochs
from meg_tokens.meg.preprocessing import STREAMS, stream_paths


def test_realign_epochs():
    info = mne.create_info(ch_names=['MEG1', 'MEG2'], sfreq=1000.0, ch_types=['mag', 'mag'])
    data = np.arange(3 * 2 * 5000, dtype=float).reshape(3, 2, 5000)
    
    metadata = pd.DataFrame({
        'tGO': [1000, 1000, 1000],
        'tEnterTarget': [1800, 2000, 2200], # Latencies of 800, 1000, 1200 ms
        'tTrialEnd': [3000, 3100, 3200]
    })
    
    epochs = mne.EpochsArray(data, info, tmin=-1.0, metadata=metadata)
    
    # 1. Test alignment to 'go' (tmin_new=-0.5, tmax_new=1.5, length=2.0s = 2000 samples)
    realigned_go = realign_epochs(epochs, align_to='go', tmin_new=-0.5, tmax_new=1.5)
    assert realigned_go.tmin == -0.5
    assert realigned_go.get_data().shape == (3, 2, 2000)
    
    # 2. Test alignment to 'enter' (offset = 0.8s, 1.0s, 1.2s from Go)
    # tmin_new = -0.2, tmax_new = 0.8, length = 1.0s = 1000 samples
    realigned_enter = realign_epochs(epochs, align_to='enter', tmin_new=-0.2, tmax_new=0.8)
    assert realigned_enter.tmin == -0.2
    assert realigned_enter.get_data().shape == (3, 2, 1000)
    
    # 3. Test alignment to 'DT' (offset = tEnterTarget - tGO - mean_rt)
    realigned_dt = realign_epochs(epochs, align_to='DT', tmin_new=-0.2, tmax_new=0.8, mean_rt=300.0)
    assert realigned_dt.tmin == -0.2
    assert realigned_dt.get_data().shape == (3, 2, 1000)


def test_stream_paths_name_the_cleaner_in_proc_and_keep_the_condition(tmp_path):
    """`proc` carries the stream; `desc` keeps the condition and grows the tag.

    The previous layout put `clean`/`filt` in `proc` to say whether ICA had
    run, which cannot name three cleaners for one recording.
    """
    layout = DerivativeLayout(tmp_path)
    paths = stream_paths(layout, "H1", "Slow", 1, "dss")

    assert paths["cleaned"].name == (
        "sub-H01_task-tokens_run-1_proc-dss_desc-slow_raw.fif"
    )
    assert paths["artifact"].name == (
        "sub-H01_task-tokens_run-1_proc-dss_desc-slow-artifact_raw.fif"
    )
    assert paths["sidecar"].name.endswith("_raw.json")


def test_every_stream_gets_a_distinct_path(tmp_path):
    """Three cleaners write three derivatives for one recording."""
    layout = DerivativeLayout(tmp_path)
    paths = [stream_paths(layout, "H1", "Slow", 1, s)["cleaned"] for s in STREAMS]
    assert len(set(paths)) == len(STREAMS)


def test_dss_selection_caps_are_inert_outside_adaptive_mode():
    """`max_prop_remove` is accepted, documented as a cap, and never applied.

    `DSS._process_segment` applies it, and that runs only when
    `adaptive=True`. The non-adaptive path -- the one `clean_dss` uses -- sets
    `n_selected_` straight from `auto_select()`.

    This is pinned rather than worked around because the failure mode of an
    upstream fix is silent: `DSS_MAX_PROP_REMOVE = 0.3` would begin capping at
    81 of 270 and every DSS derivative would change with nothing to say why.
    """
    import mne
    import numpy as np
    from mne_denoise.dss import DSS
    from mne_denoise.dss.denoisers import BandpassBias

    from meg_tokens.meg.preprocessing import DSS_MAX_PROP_REMOVE

    rng = np.random.default_rng(0)
    sfreq, n_channels, n_times = 600.0, 60, 24000
    times = np.arange(n_times) / sfreq
    data = rng.normal(size=(n_channels, n_times)) * 0.05
    for index in range(30):
        data += np.outer(rng.normal(size=n_channels), np.sin(2 * np.pi * (40 + index * 0.3) * times))
    raw = mne.io.RawArray(data, mne.create_info(n_channels, sfreq, "mag"), verbose=False)

    def n_selected(**kwargs):
        model = DSS(
            bias=BandpassBias(freq_band=(38.0, 50.0), sfreq=sfreq),
            n_components=None, n_select="auto", component_action="subtract", **kwargs
        )
        model.fit_transform(raw.copy())
        return model.n_selected_

    uncapped = n_selected()
    assert uncapped > int(n_channels * 0.05), "the fixture must exceed the cap to test it"
    assert n_selected(max_prop_remove=0.05) == uncapped
    assert n_selected(max_prop_remove=DSS_MAX_PROP_REMOVE) == uncapped


def test_a_streams_detail_survives_json_without_a_stringifying_fallback():
    """The sidecar is the audit record for what a stream removed.

    `find_bads_eog`/`find_bads_ecg` return numpy scalars, and `save_stream`
    used to carry a `default=str` fallback -- so an `int64` component index was
    written as `"3"` and nothing said so. The fallback is gone, which makes a
    stray numpy type raise here rather than corrupt a record.
    """
    import json

    import numpy as np

    from meg_tokens.meg.preprocessing import StreamResult

    result = StreamResult(
        stream="ica", cleaned=None, artifact=None, n_dimensions_removed=3,
        detail={"n_components": 20, "excluded": [int(i) for i in np.array([3, 7, 11])]},
    )
    dumped = json.dumps(result.detail, sort_keys=True)
    assert json.loads(dumped)["excluded"] == [3, 7, 11]
    assert all(isinstance(index, int) for index in json.loads(dumped)["excluded"])

    with pytest.raises(TypeError, match="not JSON serializable"):
        json.dumps({"excluded": list(np.array([3, 7]))})
