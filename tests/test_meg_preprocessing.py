import os
import json
import tempfile
import mne
import numpy as np
import pandas as pd
from meg_tokens.meg.preprocessing import realign_epochs, save_clean_raw


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


def test_save_clean_raw_derivative(tmp_path):
    info = mne.create_info(ch_names=['MEG1', 'MEG2'], sfreq=1000.0, ch_types=['mag', 'mag'])
    data = np.zeros((2, 1000))
    raw = mne.io.RawArray(data, info)

    path = save_clean_raw(raw, tmp_path, subject_id='H1', run_id='Slow1', processing='filt')

    assert path.endswith(
        "derivatives/sub-H01/meg/sub-H01_task-tokens_run-1_proc-filt_desc-slow_raw.fif"
    )
    assert os.path.exists(path)
    sidecar = path.replace(".fif", ".json")
    assert os.path.exists(sidecar)
    with open(sidecar, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["subject"] == "H01"
    assert meta["condition"] == "Slow"
    assert meta["run"] == "1"
