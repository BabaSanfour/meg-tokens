# MEG/iEEG Tokens Task Analysis (DDM Project)

This repository contains the analysis scripts and a clean, refactored Python library for investigating decision-making dynamics using the **Tokens Task** paired with Magnetoencephalography (MEG) and Intracranial EEG (iEEG).

## Project Structure

*   **`meg_tokens/`**: Main Python package with refactored, clean production code.
    *   `behavior/`: Modules for parsing behavior logs, calculating reaction times, and plotting performance.
    *   `meg/`: Modules for neural data preprocessing, ICA, and source localization.
    *   `utils/`: Helpers for reading TDMS files and I/O.
*   **`tests/`**: Unit tests.
*   **`pyproject.toml`**: Metadata and dependency configuration for the python package.
*   **`archive/`**: Contains the raw, unorganized scripts copied from the external drives:
    *   `DDM_scripts/`: Python/Jupyter notebooks (`scripts_new/`) and Matlab scripts (`matlab_scripts/`) copied from the `DDM_scripts` partition.
    *   `DDM_analysis_scripts/`: Jupyter notebooks copied from the `DDM/scripts/` partition.

## Data Locations

*   **Raw MEG Brain Recordings**: Located on the external drive at:
    `[Hamza Drive] /media/karim/Hamza/DDM-tthiery/`
    Contains raw CTF MEG datasets (`.ds` folders), digitized head shapes, and fiducial photos.
*   **Behavioral Logs (TDMS)**: Located at:
    `[cc197cfe-12fc-4d55-b0a8-4f52a93ef003 Drive] /media/karim/cc197cfe-12fc-4d55-b0a8-4f52a93ef003/DDM/tdms/`
    Contains LabVIEW behavioral event logs for all 32 subjects (`H1` to `H32`).
*   **Behavioral Dataframes (CSVs)**: Located at:
    `[cc197cfe-12fc-4d55-b0a8-4f52a93ef003 Drive] /media/karim/cc197cfe-12fc-4d55-b0a8-4f52a93ef003/DDM/dataframes/`
    Contains extracted behavior variables (choice, correct, reaction time, probabilities) exported as CSV files.

---
*Note: This repository was refactored and organized on 2026-06-25.*
