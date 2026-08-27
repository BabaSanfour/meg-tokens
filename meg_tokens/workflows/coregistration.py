"""MEG-MRI coregistration over the staged BIDS raw layer.

One ICP fit per subject, carried to every one of that subject's recordings.

Fitting per recording would be the obvious design and is the wrong one. ICP is
the expensive part and the head shape is digitized once, so ten fits would be
ten noisy answers to a question with one answer. What genuinely differs between
recordings is the head *frame* -- the coils are placed afresh each time -- and
that difference is closed-form, recoverable from the three landmarks both
recordings localized. So the fit runs once against the subject's most
representative session and each recording gets an exact correction.

Subjects with no FreeSurfer reconstruction borrow a template anatomy, scaled to
their own head shape (``CoregistrationConfig.scale_template``). The scaled
anatomy is materialized as a real FreeSurfer subject, and ``mne.scale_mri``
records its own provenance in ``MRI scaling parameters.cfg``, so nothing
downstream has to be told which subjects are templates -- but every QC row says
so anyway.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import mne
import pandas as pd

from meg_tokens.core import (
    CoregistrationConfig,
    ProjectConfig,
    WorkflowResult,
    normalize_subject_id,
)
from meg_tokens.io import (
    DerivativeLayout,
    ensure_dir,
    save_sidecar,
    save_table,
    sidecar_path,
)
from meg_tokens.meg.coregistration import (
    HeadShape,
    compose_trans,
    coregister,
    has_own_anatomy,
    read_ctf_headshape,
    reference_session,
    session_alignment,
)
from meg_tokens.meg.meg_bids import (
    headshape_path,
    raw_meg_sessions,
    staged_subjects,
)


def qc_table(layout: DerivativeLayout, *, task: str = "tokens") -> pd.DataFrame:
    """Every persisted transform's QC record, read back from its sidecar.

    Built from disk rather than from the rows this run happened to produce, so
    coregistering one subject cannot silently replace a cohort-wide summary with
    a one-row table. The sidecars are the record; this is a view of them.
    """
    rows = []
    for path in sorted(Path(layout.root).glob("derivatives/sub-*/meg/*_trans.fif")):
        metadata = json.loads(sidecar_path(path).read_text())
        if metadata.get("kind") != "meg_mri_transform":
            continue
        rows.append(
            {
                key: metadata[key]
                for key in (
                    "subject", "condition", "run", "fs_subject", "template_anatomy",
                    "mri_fiducials", "reference_session", "session_drift_max_mm",
                    "fiducial_residual_max_mm", "dig_mri_median_mm", "dig_mri_mean_mm",
                    "dig_mri_p95_mm", "dig_mri_max_mm", "points_used", "points_omitted",
                    "flagged",
                )
                if key in metadata
            }
        )
    return pd.DataFrame(rows)


def _scaled_anatomy(
    subject: str,
    info: mne.Info,
    headshape: Optional[HeadShape],
    *,
    subjects_dir: Path,
    settings: CoregistrationConfig,
) -> str:
    """Materialize a template anatomy scaled to one subject's head shape."""
    sized = coregister(
        info,
        settings.template_subject,
        subjects_dir,
        headshape=headshape,
        coarse_iterations=settings.coarse_iterations,
        fine_iterations=settings.fine_iterations,
        omit_distance=settings.omit_distance,
        scale_mode="3-axis",
    )
    mne.scale_mri(
        settings.template_subject,
        subject,
        sized.scale,
        subjects_dir=subjects_dir,
        overwrite=True,
        labels=True,
        annot=True,
        verbose=False,
    )

    # scale_mri faithfully rescales every source space it finds in the
    # template's bem/, and fsaverage ships several large prebuilt ones -- 1.2 of
    # the 1.5 GB a scaled anatomy would otherwise occupy. This project uses none
    # of them: Step D builds its own source space and stores it as a derivative.
    for source_space in (subjects_dir / subject / "bem").glob("*-src.fif"):
        source_space.unlink()
    return subject


def coregister_subjects(
    project: ProjectConfig,
    *,
    subjects: Optional[Sequence[str]] = None,
    settings: CoregistrationConfig = CoregistrationConfig(),
    output_root: Optional[str | Path] = None,
) -> WorkflowResult:
    """Fit and persist one MEG-MRI transform per staged recording.

    Raises ``FileNotFoundError`` when a requested subject has no staged MEG:
    that is a missing prerequisite, not an empty result, and quietly producing
    nothing would look identical to success.

    Writes transforms and a QC table, not figures. Rendering reads this stage's
    output back (``meg_tokens.reports.coregistration``), so a figure always
    shows the transform in use and can be redrawn without refitting.
    """
    bids_root = project.bids_root
    if not bids_root.is_dir():
        raise FileNotFoundError(
            f"BIDS root does not exist: {bids_root}. Run `meg-tokens meg stage-raw` "
            "and `meg-tokens meg apply-raw-staging` before coregistering."
        )
    subjects_dir = Path(project.subjects_dir)
    destination = Path(output_root or bids_root)
    layout = DerivativeLayout(destination, task=project.task)

    selected = [normalize_subject_id(s) for s in (subjects or staged_subjects(bids_root))]
    if not selected:
        raise FileNotFoundError(f"No staged subject MEG was found under {bids_root}")

    inputs: list[Path] = []
    outputs: list[Path] = []

    for subject in selected:
        sessions = raw_meg_sessions(bids_root, subject, task=project.task)
        if not sessions:
            raise FileNotFoundError(
                f"No staged raw MEG for {subject} under {bids_root}. "
                "Run `meg-tokens meg apply-raw-staging` first."
            )
        infos = [
            mne.io.read_raw_ctf(str(path), preload=False).info for _, path in sessions
        ]

        shape_path = headshape_path(bids_root, subject)
        headshape = read_ctf_headshape(shape_path) if shape_path.is_file() else None
        if headshape is not None:
            inputs.append(shape_path)

        reference = reference_session(infos)
        own_anatomy = has_own_anatomy(subjects_dir, subject)
        if own_anatomy:
            fs_subject = subject
        elif settings.scale_template:
            fs_subject = _scaled_anatomy(
                subject,
                infos[reference],
                headshape,
                subjects_dir=subjects_dir,
                settings=settings,
            )
        else:
            fs_subject = settings.template_subject

        fit = coregister(
            infos[reference],
            fs_subject,
            subjects_dir,
            headshape=headshape,
            coarse_iterations=settings.coarse_iterations,
            fine_iterations=settings.fine_iterations,
            omit_distance=settings.omit_distance,
        )
        quality = fit.quality
        flagged = (
            quality["dig_mri_median_mm"] > settings.max_median_distance_mm
            or quality["fiducial_residual_max_mm"] > settings.max_landmark_residual_mm
        )

        for (run, path), info in zip(sessions, infos):
            alignment, drift_mm = session_alignment(infos[reference], info)
            trans = compose_trans(fit.trans, alignment)
            trans_path = layout.source(
                subject=subject,
                run=run.run,
                condition=run.condition,
                suffix="trans",
                extension=".fif",
            )
            ensure_dir(trans_path.parent)
            mne.write_trans(str(trans_path), trans, overwrite=True)
            save_sidecar(
                trans_path,
                {
                    "schema_version": 1,
                    "format": "mne-trans-fif",
                    "stage": "coregistration",
                    "kind": "meg_mri_transform",
                    "subject": subject,
                    "run": run.run,
                    "condition": run.condition,
                    "fs_subject": fs_subject,
                    "template_anatomy": not own_anatomy,
                    "headshape": shape_path.name if headshape else None,
                    "mri_fiducials": fit.fiducial_source,
                    "reference_session": sessions[reference][1].name,
                    "session_drift_max_mm": round(float(drift_mm.max()), 4),
                    "headshape_points": fit.n_points,
                    "flagged": flagged,
                    **{key: round(value, 4) for key, value in quality.items()},
                },
            )
            inputs.append(path)
            outputs.append(trans_path)

    table = qc_table(layout, task=project.task)
    summary_path = layout.path(
        subject="group",
        description="coreg",
        suffix="qcsummary",
        extension=".tsv",
    )
    save_table(
        summary_path,
        table,
        metadata={
            "stage": "coregistration",
            "kind": "coregistration_qc",
            "subjects": selected,
            "n_transforms": len(table),
            "n_flagged": int(table["flagged"].sum()) if len(table) else 0,
        },
    )
    outputs.append(summary_path)

    return WorkflowResult(
        stage="coregistration",
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        settings={
            "subjects": selected,
            "n_transforms": len(table),
            "flagged_subjects": sorted(
                set(table.loc[table["flagged"], "subject"])
            ) if len(table) else [],
            "template_subjects": sorted(
                set(table.loc[table["template_anatomy"], "subject"])
            ) if len(table) else [],
            "scale_template": settings.scale_template,
        },
    )
