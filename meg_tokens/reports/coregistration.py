"""Visual QC for MEG-MRI coregistration, rebuilt from persisted transforms.

Deliberately 2-D. The conventional check is a 3-D rendering spun by hand, which
does not survive being run thirty times without a display and leaves no record
of *how far* anything was off. Three orthogonal projections plus the distance
distribution answer the same question -- do the digitized points lie on this
scalp, and if not, where -- in a file that can be flipped through and archived.

Nothing here refits. Everything is read back from what ``meg coreg`` wrote, so
a figure can be regenerated, restyled, or reviewed long after the fit, and the
picture is guaranteed to show the transform actually in use rather than a fresh
one that happens to resemble it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np

from meg_tokens.core import ProjectConfig
from meg_tokens.io import DerivativeLayout, ensure_dir, sidecar_path
from meg_tokens.meg.coregistration import (
    head_to_mri,
    headshape_in_head_frame,
    read_ctf_headshape,
    scalp_distances,
    scalp_surface,
)
from meg_tokens.meg.meg_bids import headshape_path, raw_meg_sessions, staged_subjects
from meg_tokens.reports.style import apply_publication_style, figure_grid, panel_label

#: Projection planes, as (axis pair, title, x label, y label).
_VIEWS = (
    ((1, 2), "Sagittal", "P-A (mm)", "I-S (mm)"),
    ((0, 2), "Coronal", "L-R (mm)", "I-S (mm)"),
    ((0, 1), "Axial", "L-R (mm)", "P-A (mm)"),
)


def plot_coregistration_qc(
    points_mri_mm: np.ndarray,
    distances_mm: np.ndarray,
    scalp_mm: np.ndarray,
    *,
    title: str,
    path: Optional[str | Path] = None,
    max_colour_mm: float = 10.0,
) -> plt.Figure:
    """Render digitized points against the scalp they were fitted to."""
    if len(distances_mm) != len(points_mri_mm):
        raise ValueError(
            f"{len(distances_mm)} scored distances for {len(points_mri_mm)} "
            "digitized points -- these must correspond."
        )

    with apply_publication_style():
        fig, axes = figure_grid(1, 4, width=13.0, panel_height_in=3.2)
        for index, ((horizontal, vertical), name, xlabel, ylabel) in enumerate(_VIEWS):
            ax = axes[0, index]
            ax.scatter(
                scalp_mm[:, horizontal], scalp_mm[:, vertical],
                s=0.6, c="0.85", linewidths=0, rasterized=True, zorder=1,
            )
            scatter = ax.scatter(
                points_mri_mm[:, horizontal], points_mri_mm[:, vertical],
                s=1.6, c=distances_mm, cmap="viridis", vmin=0, vmax=max_colour_mm,
                linewidths=0, rasterized=True, zorder=2,
            )
            ax.set_aspect("equal")
            ax.set_title(name)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            panel_label(ax, "ABC"[index])

        ax = axes[0, 3]
        ax.hist(distances_mm, bins=40, color="0.35")
        ax.axvline(np.median(distances_mm), color="crimson", lw=1.2)
        ax.set_xlabel("digitizer-scalp (mm)")
        ax.set_ylabel("points")
        ax.set_title("Distance")
        panel_label(ax, "D")

        fig.colorbar(scatter, ax=axes[0, :3].tolist(), label="mm", fraction=0.02, pad=0.01)
        fig.suptitle(title, fontsize=8, y=1.04)
        if path is not None:
            ensure_dir(Path(path).parent)
            fig.savefig(path)
            plt.close(fig)
    return fig


def render_subject_coregistration(
    project: ProjectConfig,
    subject: str,
    *,
    output_root: Optional[str | Path] = None,
) -> Path:
    """Rebuild one subject's coregistration QC figure from its saved transform."""
    layout = DerivativeLayout(output_root or project.bids_root, task=project.task)

    probe = layout.path(subject=subject, suffix="trans", extension=".fif")
    transforms = sorted(probe.parent.glob(f"sub-{subject}_task-{project.task}_*_trans.fif"))
    if not transforms:
        raise FileNotFoundError(
            f"No MEG-MRI transform for {subject} under {probe.parent}. "
            "Run `meg-tokens meg coreg` first."
        )
    metadata = json.loads(sidecar_path(transforms[0]).read_text())

    # Caught here rather than left to the head-shape reader, whose "file does
    # not exist" would point at staging instead of at the fit.
    if not metadata.get("headshape"):
        raise ValueError(
            f"{subject} was coregistered from landmarks alone -- no head shape "
            "was staged, so there is nothing to plot against the scalp."
        )

    # Located through the same discovery the fit used, not by joining a path:
    # a CTF session is a directory, so the file-shaped existence helpers reject
    # it out of hand.
    session = metadata["reference_session"]
    sessions = {
        path.name: path
        for _, path in raw_meg_sessions(project.bids_root, subject, task=project.task)
    }
    if session not in sessions:
        raise FileNotFoundError(
            f"{subject}'s reference recording {session} is no longer staged; "
            "re-run `meg-tokens meg coreg`."
        )
    info = mne.io.read_raw_ctf(str(sessions[session]), preload=False, verbose=False).info
    headshape = read_ctf_headshape(headshape_path(project.bids_root, subject))

    # The reference session's own transform, not a neighbour's: the figure has
    # to show the fit itself, and every other session's transform is that fit
    # plus a landmark correction.
    trans = mne.read_trans(
        str(
            layout.find_trans(
                subject=subject,
                run=session.split("_run-")[1].split("_")[0],
                condition=session.split("_acq-")[1].split("_")[0],
            )
        )
    )

    head_points, _ = headshape_in_head_frame(headshape, info)
    fs_subject = metadata["fs_subject"]
    scalp = scalp_surface(fs_subject, project.subjects_dir)
    distances_mm = 1000 * scalp_distances(head_points, trans, scalp)
    points_mri_mm = mne.transforms.apply_trans(head_to_mri(trans), head_points) * 1000

    anatomy = "template" if metadata["template_anatomy"] else "own MRI"
    figure_path = layout.path(
        subject=subject,
        datatype="fig",
        description="coreg",
        suffix="qc",
        extension=".png",
    )
    plot_coregistration_qc(
        points_mri_mm,
        distances_mm,
        scalp * 1000,
        title=(
            f"{subject}   median {np.median(distances_mm):.2f} mm   "
            f"p95 {np.percentile(distances_mm, 95):.2f} mm   "
            f"landmark residual {metadata['fiducial_residual_max_mm']:.2f} mm   "
            f"{anatomy}, MRI landmarks {metadata['mri_fiducials']}"
        ),
        path=figure_path,
    )
    return figure_path


def render_coregistration_qc(
    project: ProjectConfig,
    *,
    subjects: Optional[Sequence[str]] = None,
    output_root: Optional[str | Path] = None,
) -> list[Path]:
    """Rebuild coregistration QC figures for every coregistered subject."""
    selected = subjects or staged_subjects(project.bids_root)
    return [
        render_subject_coregistration(project, subject, output_root=output_root)
        for subject in selected
    ]
