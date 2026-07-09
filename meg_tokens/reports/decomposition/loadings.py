import json
from typing import Optional, Sequence

import numpy as np
import mne
from meg_tokens.reports.meg import plot_brain_tmap
from meg_tokens.io import ensure_dir, load_array, require_file


def _load_vertices(vertices_path: Optional[str], metadata: dict) -> list:
    if vertices_path is not None:
        vertices_file = require_file(vertices_path, purpose="source vertices")
        if vertices_file.suffix == ".json":
            with vertices_file.open("r", encoding="utf-8") as f:
                return [np.asarray(vertices) for vertices in json.load(f)]
        loaded = np.load(vertices_file, allow_pickle=False)
        if loaded.ndim == 1:
            return [loaded]
        if loaded.ndim == 2:
            return [np.asarray(row) for row in loaded]
        raise ValueError("vertices_path must be a JSON list of vertex arrays or a 1D/2D numeric .npy file")

    stage_meta = metadata.get("metadata", {})
    if isinstance(stage_meta, dict) and stage_meta.get("source_vertices") is not None:
        return [np.asarray(vertices) for vertices in stage_meta["source_vertices"]]
    raise ValueError("--vertices_path is required unless the loadings sidecar contains source_vertices")


def create_stc_from_loadings(
    pca_loadings: np.ndarray,
    *,
    vertices: list,
    is_volume: bool = False,
) -> mne.SourceEstimate:
    """Create an MNE source estimate from loadings and real vertex metadata."""
    if pca_loadings.ndim == 1:
        pca_loadings = pca_loadings[:, np.newaxis]
        
    tmin = 0.0
    tstep = 0.01
    
    if is_volume:
        return mne.VolSourceEstimate(data=pca_loadings, vertices=vertices, tmin=tmin, tstep=tstep, subject='fsaverage')
    return mne.SourceEstimate(data=pca_loadings, vertices=vertices, tmin=tmin, tstep=tstep, subject='fsaverage')

def run_batch_plot_pca_loadings(
    loadings_path: str,
    output_dir: str,
    subjects_dir: str = None,
    threshold_percentile: float = None,
    export_csv: bool = False,
    save_movie: bool = False,
    vertices_path: str = None,
    is_volume: bool = False
):
    print(f"=== Starting PCA Loadings 3D Brain Plotting ===")
    print(f"Loading PCA loadings from: {loadings_path}")
    
    output_path = ensure_dir(output_dir)
    loaded = load_array(loadings_path)
    pca_loadings = loaded.data
        
    print(f"Loaded PCA loadings shape: {pca_loadings.shape}")
    
    if pca_loadings.ndim > 1 and pca_loadings.shape[1] > 1:
        if save_movie:
            print("Time-resolved data detected. Will generate frame-by-frame views over time.")
        else:
            print("Averaging spatial data across time for static 3D visualization. Use --save_movie to keep time resolution.")
            pca_loadings = np.mean(pca_loadings, axis=1)
        
    if threshold_percentile is not None:
        abs_loadings = np.abs(pca_loadings)
        thresh_val = np.percentile(abs_loadings, threshold_percentile)
        print(f"Thresholding at the {threshold_percentile}th percentile (value: {thresh_val:.4f})")
        
        # Zero out values below threshold
        mask = abs_loadings >= thresh_val
        pca_loadings = np.where(mask, pca_loadings, 0.0)
        
        if export_csv:
            active_indices = np.where(mask)[0]
            active_values = pca_loadings[mask]
            
            import pandas as pd
            df = pd.DataFrame({
                'Vertex_Index': active_indices,
                'Loading_Value': active_values
            }).sort_values(by='Loading_Value', ascending=False)
            
            csv_path = output_path / "top_active_regions.csv"
            df.to_csv(csv_path, index=False)
            print(f"Exported top active vertices to {csv_path}")
            
    print("Constructing mne.SourceEstimate...")
    vertices = _load_vertices(vertices_path, loaded.metadata)
    stc = create_stc_from_loadings(pca_loadings, vertices=vertices, is_volume=is_volume)

    print("Rendering 3D Brain...")
    brain = plot_brain_tmap(
        stc=stc,
        subject='fsaverage',
        subjects_dir=subjects_dir,
        title='PCA Loadings'
    )

    save_path = output_path / "pca_loadings_3d_render.png"

    if save_movie:
        from meg_tokens.reports.meg import save_brain_movie_frames
        save_brain_movie_frames(
            brain=brain,
            times=stc.times,
            output_dir=output_dir,
            prefix="Fig_pca",
            views=['dorsal', 'lateral', 'medial']
        )
    else:
        print(f"Brain object created. Configure the MNE 3D backend to save a screenshot at: {save_path}")
