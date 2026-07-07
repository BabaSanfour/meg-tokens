import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import mne
from meg_tokens.io import derivative_path, load_array, require_file, save_array, save_sidecar
from meg_tokens.meg.decoding import compute_spatial_decoding, compute_spatial_decoding_permutations

def load_spatial_decoding_inputs(data_dir: str, conditions: list[str]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    base = Path(data_dir)
    X = load_array(base / "X.npy", expected_ndim=2).data
    y = load_array(base / "y.npy", expected_ndim=1, allow_pickle=True).data
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y disagree on epochs: {X.shape[0]} != {y.shape[0]}")

    groups_path = base / "groups.npy"
    groups = load_array(groups_path, expected_ndim=1, allow_pickle=True).data if groups_path.exists() else None
    if groups is not None and groups.shape[0] != X.shape[0]:
        raise ValueError(f"X and groups disagree on epochs: {X.shape[0]} != {groups.shape[0]}")

    if y.dtype.kind in {"U", "S", "O"}:
        keep = np.isin(y, conditions)
        missing = sorted(set(conditions) - set(y[keep].astype(str)))
        if missing:
            raise ValueError(f"Requested conditions absent from y.npy: {missing}")
        X = X[keep]
        y = y[keep].astype(str)
        if groups is not None:
            groups = groups[keep]
        label_to_int = {label: idx for idx, label in enumerate(conditions)}
        y = np.array([label_to_int[label] for label in y], dtype=int)

    return X, y, groups

def run_batch_decoding_topoplot(
    data_dir: str,
    output_dir: str,
    conditions: list,
    info_fif: str,
    permutations: int = 0,
    n_jobs: int = 4
):
    print(f"=== Starting Spatial MVPA Decoding (Topoplot) ===")
    print(f"Conditions: {conditions}")

    info = mne.io.read_info(require_file(info_fif, purpose="sensor topomap info"))
    print(f"Searching for sensor data in {data_dir}...")
    X, y, groups = load_spatial_decoding_inputs(data_dir, list(conditions))
    
    print(f"Constructed Data Matrix X: {X.shape} (Epochs x Sensors)")
    if X.shape[1] != len(info["ch_names"]):
        raise ValueError(f"X has {X.shape[1]} sensors but info has {len(info['ch_names'])} channels")
    
    if permutations > 0:
        print(f"Running Spatial LDA Searchlight with {permutations} permutations...")
        scores, perm_scores, threshold = compute_spatial_decoding_permutations(
            X=X, y=y, groups=groups, balance=True, n_permutations=permutations, n_jobs=n_jobs
        )
        print(f"Calculated 95% permutation threshold at Accuracy: {threshold:.3f}")
    else:
        print("Running Spatial LDA Searchlight...")
        scores = compute_spatial_decoding(
            X=X, y=y, groups=groups, balance=True, n_jobs=n_jobs
        )
        threshold = None
    
    # Save scores
    desc = "-".join([conditions[0].lower(), "vs", *[condition.lower() for condition in conditions[1:]], "sensor"])
    out_file = derivative_path(
        output_dir,
        subject="group",
        datatype="meg",
        task="tokens",
        description=desc,
        suffix="spatialdecoding",
        extension=".npy",
    )
    save_array(
        out_file,
        scores,
        dims=("sensor",),
        coords={"channel": info["ch_names"]},
        metadata={
            "stage": "decoding",
            "kind": "sensor_spatial_decoding",
            "conditions": list(conditions),
            "classifier": "LinearDiscriminantAnalysis",
            "input_data_dir": str(data_dir),
            "input_info": str(info_fif),
        },
    )
    
    fig, ax = plt.subplots(figsize=(6, 5))
    mne.viz.plot_topomap(scores, info, axes=ax, show=False)
    ax.set_title(f"Spatial decoding: {' vs '.join(conditions)}")
    fig_file = derivative_path(
        output_dir,
        subject="group",
        datatype="fig",
        task="tokens",
        description=desc,
        suffix="spatialdecoding",
        extension=".png",
    )
    fig_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_file, dpi=300)
    plt.close(fig)
    save_sidecar(
        fig_file,
        {
            "stage": "decoding",
            "kind": "sensor_spatial_decoding_topomap",
            "conditions": list(conditions),
            "score_file": str(out_file),
            "input_info": str(info_fif),
        },
    )
    print(f"Saved topoplot to {fig_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Spatial MVPA Decoding and Topoplot")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing time-averaged sensor data")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="BIDS derivatives root for outputs")
    parser.add_argument("--conditions", type=str, nargs='+', default=['Fast', 'Slow'],
                        help="Conditions to decode")
    parser.add_argument("--info_fif", type=str, required=True,
                        help="Path to an MNE FIF file whose info contains the sensor layout")
    parser.add_argument("--permutations", type=int, default=0,
                        help="Number of permutations")
    parser.add_argument("--n_jobs", type=int, default=4)
    
    args = parser.parse_args()
    run_batch_decoding_topoplot(args.data_dir, args.out_dir, args.conditions, args.info_fif, args.permutations, args.n_jobs)
