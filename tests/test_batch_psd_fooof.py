import json

import mne
import numpy as np

from meg_tokens.io import derivative_path, load_array, sidecar_path
from meg_tokens.utils.batch_psd_fooof import (
    find_epoch_derivatives,
    process_epochs_psd,
    psd_derivative_path,
    run_psd_specparam,
)


def _write_epochs(root):
    sfreq = 100.0
    n_times = 300
    times = np.arange(n_times) / sfreq
    epoch = np.stack([
        np.sin(2 * np.pi * 10.0 * times),
        np.cos(2 * np.pi * 15.0 * times),
    ])
    data = np.stack([epoch, epoch * 0.5], axis=0)
    info = mne.create_info(["MEG001", "MEG002"], sfreq, ch_types="mag")
    events = np.array([[0, 0, 1], [n_times, 0, 1]])
    epochs = mne.EpochsArray(data, info, events=events, event_id={"Go": 1}, tmin=0.0, verbose=False)
    path = derivative_path(
        root,
        subject="H01",
        datatype="meg",
        task="tokens",
        run="1",
        description="fast-go",
        suffix="epo",
        extension=".fif",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs.save(path, overwrite=True, verbose=False)
    with sidecar_path(path).open("w", encoding="utf-8") as f:
        json.dump(
            {
                "format": "mne-epochs-fif",
                "stage": "epoching",
                "subject": "H01",
                "condition": "Fast",
                "run": "1",
                "alignment": "go",
                "n_epochs": 2,
                "event_id": {"Go": 1},
            },
            f,
        )
    return path


def test_find_epoch_derivatives_filters_by_metadata(tmp_path):
    path = _write_epochs(tmp_path)

    found = find_epoch_derivatives(tmp_path, "H1", condition="Fast", align_to="go")

    assert found == [path]


def test_process_epochs_psd_writes_sidecar_and_specparam_tables(tmp_path):
    epochs_path = _write_epochs(tmp_path)

    outputs = process_epochs_psd(
        epochs_path,
        tmp_path,
        fmin=2.0,
        fmax=40.0,
        method="welch",
        n_fft=128,
        n_overlap=0,
        n_jobs=1,
        fit_model=True,
        min_peak_height=0.01,
        peak_threshold=1.0,
    )

    psd_path = psd_derivative_path(
        tmp_path,
        subject="H01",
        run="1",
        condition="Fast",
        align_to="go",
        method="welch",
        fmin=2.0,
        fmax=40.0,
        suffix="psd",
        extension=".npy",
    )
    loaded = load_array(psd_path, require_sidecar=True)

    assert outputs["psd"] == psd_path
    assert loaded.data.shape[0] == 2
    assert loaded.metadata["dims"] == ["channel", "frequency"]
    assert loaded.metadata["coords"]["channel"] == ["MEG001", "MEG002"]
    assert loaded.metadata["metadata"]["input_epochs"] == str(epochs_path)
    assert outputs["specparam"].name.endswith("_specparam.tsv")
    assert outputs["specparam_peaks"].name.endswith("_specparampeaks.tsv")
    assert outputs["specparam"].with_suffix(".json").exists()


def test_run_psd_specparam_can_skip_model_fit(tmp_path):
    _write_epochs(tmp_path)

    outputs = run_psd_specparam(
        tmp_path,
        tmp_path,
        subjects=["H01"],
        condition="Fast",
        align_to="go",
        fmin=2.0,
        fmax=40.0,
        method="welch",
        n_fft=128,
        n_overlap=0,
        fit_model=False,
    )

    assert len(outputs["H01"]) == 1
    assert set(outputs["H01"][0]) == {"psd"}
