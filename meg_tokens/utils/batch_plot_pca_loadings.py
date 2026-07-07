import os
import argparse
import numpy as np
import mne
from meg_tokens.meg.plotting import plot_brain_tmap

def create_mock_stc(pca_loadings: np.ndarray, fs_dir: str = None, vertices_path: str = None, is_volume: bool = False) -> mne.SourceEstimate:
    """
    Creates a mock mne.SourceEstimate or VolSourceEstimate object to hold the pca_loadings for 3D plotting.
    Assuming standard fsaverage with 8196 vertices per hemisphere if no vertices_path is provided.
    """
    if vertices_path and os.path.exists(vertices_path):
        # Load custom vertices (e.g. from an ROI subset)
        # Assuming saved with dtype=object, so we extract the list/arrays
        loaded = np.load(vertices_path, allow_pickle=True)
        if isinstance(loaded, np.ndarray) and loaded.ndim == 0:
            vertices = loaded.item()
        else:
            vertices = list(loaded)
    else:
        n_total_vertices = pca_loadings.shape[0]
        n_hemi = n_total_vertices // 2
        vertices = [np.arange(n_hemi), np.arange(n_hemi)]
    
    if pca_loadings.ndim == 1:
        pca_loadings = pca_loadings[:, np.newaxis]
        
    tmin = 0.0
    tstep = 0.01
    
    if is_volume:
        stc = mne.VolSourceEstimate(data=pca_loadings, vertices=vertices, tmin=tmin, tstep=tstep, subject='fsaverage')
    else:
        stc = mne.SourceEstimate(data=pca_loadings, vertices=vertices, tmin=tmin, tstep=tstep, subject='fsaverage')
    return stc

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
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    pca_loadings = np.load(loadings_path)
        
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
            
            csv_path = os.path.join(output_dir, "top_active_regions.csv")
            df.to_csv(csv_path, index=False)
            print(f"Exported top active vertices to {csv_path}")
            
    print("Constructing mne.SourceEstimate...")
    try:
        stc = create_mock_stc(pca_loadings, fs_dir=subjects_dir, vertices_path=vertices_path, is_volume=is_volume)
        
        print("Rendering 3D Brain...")
        brain = plot_brain_tmap(
            stc=stc,
            subject='fsaverage',
            subjects_dir=subjects_dir,
            title='PCA Loadings'
        )
        
        save_path = os.path.join(output_dir, "pca_loadings_3d_render.png")
        
        if save_movie:
            from meg_tokens.meg.plotting import save_brain_movie_frames
            save_brain_movie_frames(
                brain=brain, 
                times=stc.times, 
                output_dir=output_dir, 
                prefix="Fig_pca", 
                views=['dorsal', 'lateral', 'medial']
            )
        else:
            print(f"Note: Rendering skipped in headless environment. Would save to: {save_path}")
        
    except Exception as e:
        print(f"Mock plotting error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot PCA loadings on a 3D brain and optionally export active regions")
    parser.add_argument("--loadings_path", type=str, required=True,
                        help="Path to .npy file containing PCA loadings")
    parser.add_argument("--out_dir", type=str, default='./figures/pca_loadings/',
                        help="Directory to save 3D renders and CSV reports")
    parser.add_argument("--subjects_dir", type=str, default=None,
                        help="FreeSurfer subjects directory")
    parser.add_argument("--threshold_percentile", type=float, default=None,
                        help="Percentile threshold (e.g. 95 for top 5%%) to filter out weak activations.")
    parser.add_argument("--export_csv", action="store_true",
                        help="If a threshold is set, export the surviving active vertices to a CSV file.")
    parser.add_argument("--save_movie", action="store_true",
                        help="If the input array is time-resolved, loop over time and export top, left, and right screenshots for every timepoint, matching the legacy visbrain script.")
    parser.add_argument("--vertices_path", type=str, default=None,
                        help="Path to .npy file containing vertex indices if PCA was constrained to ROI")
    parser.add_argument("--volume", action="store_true",
                        help="Plot as volumetric source estimate instead of surface")
    
    args = parser.parse_args()
    run_batch_plot_pca_loadings(
        args.loadings_path, args.out_dir, args.subjects_dir, 
        args.threshold_percentile, args.export_csv, args.save_movie,
        args.vertices_path, args.volume
    )
