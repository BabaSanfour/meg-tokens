import os
import json
import tempfile
import pytest
import mne
import numpy as np
import pandas as pd
from meg_tokens.meg.preprocessing import convert_ctf_headshape_to_pos, realign_epochs, save_clean_raw


def test_convert_ctf_headshape_to_pos():
    eeg_content = (
        "NZ 1.0 2.0 3.0\n"
        "OG 4.0 5.0 6.0\n"
        "OD 7.0 8.0 9.0\n"
        "POINT1 10.0 11.0 12.0\n"
        "POINT2 13.0 14.0 15.0\n"
        "POINT3 16.0 17.0 18.0\n"
        "POINT4 19.0 20.0 21.0\n"
        "POINT5 22.0 23.0 24.0\n"
        "POINT6 25.0 26.0 27.0\n"
        "POINT7 28.0 29.0 30.0\n"
        "POINT8 31.0 32.0 33.0\n"
        "POINT9 34.0 35.0 36.0\n"
        "POINT10 37.0 38.0 39.0\n"
        "POINT11 40.0 41.0 42.0\n"
        "POINT12 43.0 44.0 45.0\n"
    )
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        eeg_path = os.path.join(tmp_dir, 'headshape.eeg')
        pos_path = os.path.join(tmp_dir, 'headshape.pos')
        
        with open(eeg_path, 'w') as f:
            f.write(eeg_content)
            
        convert_ctf_headshape_to_pos(eeg_path, pos_path)
        
        assert os.path.exists(pos_path)
        
        with open(pos_path, 'r') as f:
            lines = f.readlines()
            
        # Header count should be total_lines - 11 = 15 - 11 = 4
        assert lines[0].strip() == "4"
        
        # Christchurch fiducials should be skipped, coordinates should be indexed starting from index-5
        # The remaining lines: POINT1 to POINT12 are 12 lines.
        # Let's check format of one of the formatted lines:
        # POINT1 line is at index 3 in raw content. index - 5 = 3 - 5 = -2.
        assert lines[1].strip() == "-2\tPOINT1 10.0 11.0 12.0"

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
        "derivatives/meg-tokens/sub-H01/meg/sub-H01_task-tokens_run-1_proc-filt_desc-slow_raw.fif"
    )
    assert os.path.exists(path)
    sidecar = path.replace(".fif", ".json")
    assert os.path.exists(sidecar)
    with open(sidecar, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["subject"] == "H01"
    assert meta["condition"] == "Slow"
    assert meta["run"] == "1"
