import numpy as np
import pandas as pd
import mne

from meg_tokens.io import load_array, save_table
from meg_tokens.meg.sources import source_derivative_path
from meg_tokens.utils.batch_time_frequency import (
    extract_power_from_manifest,
    find_stc_manifest,
    parse_frequency_bands,
)


def _write_stage3_manifest(root):
    manifest_path = source_derivative_path(
        root,
        "H1",
        suffix="stcmanifest",
        extension=".tsv",
        run_id="Slow1",
        description="go-dSPM",
    )
    vertices = [np.array([0]), np.array([1])]
    sfreq = 100.0
    times = np.arange(200) / sfreq
    rows = []

    for trial, scale in enumerate([1.0, 1.5], start=1):
        data = np.vstack([
            np.sin(2 * np.pi * 10.0 * times),
            np.cos(2 * np.pi * 10.0 * times),
        ]) * scale
        stc = mne.SourceEstimate(data, vertices=vertices, tmin=-0.2, tstep=1.0 / sfreq)
        base = source_derivative_path(
            root,
            "H1",
            suffix=f"trial{trial:04d}stc",
            extension="",
            run_id="Slow1",
            description="go-dSPM",
        )
        base.parent.mkdir(parents=True, exist_ok=True)
        stc.save(str(base), ftype="stc", overwrite=True)
        rows.append({
            "trial": trial,
            "stc_base": str(base),
            "subject": "H01",
            "run": "1",
            "condition": "Slow",
            "alignment": "go",
            "method": "dSPM",
        })

    save_table(
        manifest_path,
        pd.DataFrame(rows),
        metadata={"stage": "source_reconstruction", "kind": "source_estimate_manifest"},
    )
    return manifest_path


def test_parse_frequency_bands_accepts_known_and_custom_bands():
    bands = parse_frequency_bands(["alpha", "custom=32,44"])

    assert bands["alpha"] == (8, 15)
    assert bands["custom"] == (32.0, 44.0)


def test_find_stc_manifest_uses_stage3_derivative_path(tmp_path):
    manifest = _write_stage3_manifest(tmp_path)

    found = find_stc_manifest(tmp_path, "H01", "Slow1", None, "go", "dSPM")

    assert found == manifest


def test_extract_power_from_manifest_writes_bids_array_with_sidecar(tmp_path):
    manifest = _write_stage3_manifest(tmp_path)

    outputs = extract_power_from_manifest(
        manifest,
        tmp_path,
        freq_bands={"alpha": (8.0, 12.0)},
        method="hilbert",
        width=50,
        step=25,
        n_jobs=1,
    )

    output = outputs["alpha"]
    loaded = load_array(output, require_sidecar=True)

    assert output.name == "sub-H01_task-tokens_run-1_desc-slow-go-dSPM-hilbert-alpha_power.npy"
    assert loaded.data.shape == (2, 2, 7)
    assert loaded.metadata["dims"] == ["trial", "source", "time"]
    assert loaded.metadata["coords"]["trial"] == [1, 2]
    assert len(loaded.metadata["coords"]["time_sec"]) == 7
    meta = loaded.metadata["metadata"]
    assert meta["stage"] == "time_frequency_power"
    assert meta["input_manifest"] == str(manifest)
    assert meta["band"] == "alpha"
    assert meta["condition"] == "Slow"
