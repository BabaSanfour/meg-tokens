"""Topographic reports for persisted sensor-space decoding scores."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import mne
import numpy as np

from meg_tokens.io import derivative_path, load_array, require_file, save_sidecar


def plot_spatial_decoding_topomap(
    score_path: str | Path,
    info_fif: str | Path,
    output_root: str | Path,
) -> Path:
    """Render sensor decoding scores using the channel coordinate in the sidecar."""
    loaded = load_array(score_path, expected_ndim=1, require_sidecar=True)
    if tuple(loaded.metadata.get("dims", ())) != ("sensor",):
        raise ValueError("Spatial decoding scores must have a sensor dimension")
    channels = list(loaded.metadata.get("coords", {}).get("sensor", ()))
    scores = np.asarray(loaded.data, dtype=float)
    if len(channels) != len(scores):
        raise ValueError("Spatial decoding sidecar must provide one sensor coordinate per score")

    info = mne.io.read_info(
        require_file(info_fif, purpose="sensor topomap info")
    )
    missing = sorted(set(channels) - set(info["ch_names"]))
    if missing:
        raise ValueError(f"Sensor info is missing decoding channels: {missing}")
    picks = [info["ch_names"].index(channel) for channel in channels]
    plot_info = mne.pick_info(info, picks, copy=True)

    stage_metadata = loaded.metadata.get("metadata", {})
    conditions = stage_metadata.get("conditions", ())
    description = (
        "-".join(
            [
                str(conditions[0]).lower(),
                "vs",
                *[str(item).lower() for item in conditions[1:]],
                "sensor",
            ]
        )
        if conditions
        else "sensor"
    )
    figure_path = derivative_path(
        output_root,
        subject="group",
        datatype="fig",
        task="tokens",
        description=description,
        suffix="spatialdecoding",
        extension=".png",
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 5))
    mne.viz.plot_topomap(scores, plot_info, axes=axis, show=False)
    axis.set_title(
        f"Spatial decoding: {' vs '.join(conditions)}"
        if conditions
        else "Spatial decoding"
    )
    figure.savefig(figure_path, dpi=300)
    plt.close(figure)
    save_sidecar(
        figure_path,
        {
            "stage": "spatial_decoding_report",
            "kind": "sensor_topomap",
            "score_file": str(score_path),
            "input_info": str(info_fif),
            "channels": channels,
            "conditions": list(conditions),
        },
    )
    return figure_path
