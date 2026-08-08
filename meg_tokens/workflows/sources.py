"""Source reconstruction workflow over staged MNE derivatives."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import mne
from mne.minimum_norm import read_inverse_operator

from meg_tokens.core import (
    ProjectConfig,
    SourceConfig,
    WorkflowResult,
    normalize_subject_id,
    parse_run_label,
)
from meg_tokens.io import DerivativeLayout, require_file
from meg_tokens.meg.sources import (
    apply_inverse_operator,
    build_inverse_operator,
    compute_forward_solution,
    compute_noise_covariance,
    save_bem_solution,
    save_forward_solution,
    save_inverse_operator,
    save_noise_covariance,
    save_source_estimates,
    save_source_space,
    setup_bem_solution,
    setup_mixed_source_space,
    source_space_description,
)


def reconstruct_sources(
    project: ProjectConfig,
    *,
    subjects: Sequence[str],
    settings: SourceConfig,
    epochs_root: Optional[str | Path] = None,
    output_root: Optional[str | Path] = None,
) -> WorkflowResult:
    """Run selected source stages for each subject and one declared run."""
    destination = Path(output_root or project.bids_root)
    output_layout = DerivativeLayout(
        destination,
        task=project.task,
    )
    epoch_layout = DerivativeLayout(
        epochs_root or project.bids_root,
        task=project.task,
    )
    inputs = []
    outputs = []

    for subject_value in subjects:
        subject = normalize_subject_id(subject_value)
        models = output_layout.source_models(
            subject=subject,
            spacing=settings.spacing,
            mixed=bool(settings.volume_labels),
        )
        cov = None
        bem = None
        src = None

        if "cov" in settings.stages:
            if project.noise_dir is None:
                raise ValueError("Project configuration requires noise_dir for the cov stage")
            noise_path = DerivativeLayout(project.noise_dir).find_noise(subject=subject)
            cov = compute_noise_covariance(str(noise_path))
            outputs.append(Path(save_noise_covariance(cov, str(destination), subject)))
            inputs.append(noise_path)
        elif "inv" in settings.stages:
            cov = mne.read_cov(
                str(require_file(models["cov"], purpose=f"{subject} noise covariance"))
            )
            inputs.append(models["cov"])

        if "bem" in settings.stages:
            if project.subjects_dir is None:
                raise ValueError("Project configuration requires subjects_dir for the bem stage")
            bem = setup_bem_solution(subject, subjects_dir=str(project.subjects_dir))
            outputs.append(Path(save_bem_solution(bem, str(destination), subject)))
        elif "fwd" in settings.stages:
            bem = mne.read_bem_solution(
                str(require_file(models["bem"], purpose=f"{subject} BEM"))
            )
            inputs.append(models["bem"])

        if "src" in settings.stages:
            if project.subjects_dir is None:
                raise ValueError("Project configuration requires subjects_dir for the src stage")
            src = setup_mixed_source_space(
                subject,
                subjects_dir=str(project.subjects_dir),
                spacing=settings.spacing,
                volume_labels=list(settings.volume_labels) if settings.volume_labels else None,
                volume_pos=settings.volume_pos,
                bem=bem,
            )
            description = source_space_description(
                settings.spacing,
                list(settings.volume_labels) if settings.volume_labels else None,
            )
            outputs.append(
                Path(
                    save_source_space(
                        src,
                        str(destination),
                        subject,
                        description,
                        volume_labels=list(settings.volume_labels) if settings.volume_labels else None,
                        volume_pos=settings.volume_pos if settings.volume_labels else None,
                    )
                )
            )
        elif "fwd" in settings.stages:
            src = mne.read_source_spaces(
                str(require_file(models["src"], purpose=f"{subject} source space"))
            )
            inputs.append(models["src"])

        if any(stage in settings.stages for stage in ("fwd", "inv", "apply")):
            epoch_path = epoch_layout.find_epochs(
                subject=subject,
                run=str(settings.run),
                condition=settings.condition,
                alignment=settings.alignment,
            )
            epochs = mne.read_epochs(str(epoch_path), preload=True)
            run, inferred_condition = parse_run_label(str(settings.run))
            condition = settings.condition or inferred_condition
            inputs.append(epoch_path)

            fwd = None
            inverse = None
            if "fwd" in settings.stages:
                if project.trans_dir is None:
                    raise ValueError("Project configuration requires trans_dir for the fwd stage")
                if src is None:
                    src = mne.read_source_spaces(
                        str(require_file(models["src"], purpose=f"{subject} source space"))
                    )
                if bem is None:
                    bem = mne.read_bem_solution(
                        str(require_file(models["bem"], purpose=f"{subject} BEM"))
                    )
                trans_path = DerivativeLayout(project.trans_dir).find_trans(
                    subject=subject,
                    run=run,
                )
                fwd = compute_forward_solution(epochs.info, str(trans_path), src, bem)
                outputs.append(
                    Path(
                        save_forward_solution(
                            fwd,
                            str(destination),
                            subject,
                            run,
                            condition,
                            settings.alignment,
                        )
                    )
                )
                inputs.append(trans_path)
            elif "inv" in settings.stages:
                fwd_path = output_layout.source(
                    subject=subject,
                    run=run,
                    condition=condition,
                    description=settings.alignment,
                    suffix="fwd",
                    extension=".fif",
                )
                fwd = mne.read_forward_solution(
                    str(require_file(fwd_path, purpose=f"{subject} forward solution"))
                )
                inputs.append(fwd_path)

            if "inv" in settings.stages:
                if cov is None:
                    cov = mne.read_cov(
                        str(require_file(models["cov"], purpose=f"{subject} noise covariance"))
                    )
                inverse = build_inverse_operator(epochs.info, fwd, cov)
                outputs.append(
                    Path(
                        save_inverse_operator(
                            inverse,
                            str(destination),
                            subject,
                            run,
                            condition,
                            settings.alignment,
                            settings.method,
                        )
                    )
                )
            elif "apply" in settings.stages:
                inverse_path = output_layout.source(
                    subject=subject,
                    run=run,
                    condition=condition,
                    description=f"{settings.alignment}-{settings.method}",
                    suffix="inv",
                    extension=".fif",
                )
                inverse = read_inverse_operator(
                    str(require_file(inverse_path, purpose=f"{subject} inverse operator"))
                )
                inputs.append(inverse_path)

            if "apply" in settings.stages:
                estimates = apply_inverse_operator(
                    epochs,
                    inverse,
                    method=settings.method,
                    snr=settings.snr,
                )
                outputs.append(
                    Path(
                        save_source_estimates(
                            estimates,
                            str(destination),
                            subject,
                            run,
                            condition,
                            settings.alignment,
                            settings.method,
                        )
                    )
                )

    return WorkflowResult(
        stage="source_reconstruction",
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        settings={
            "subjects": [normalize_subject_id(subject) for subject in subjects],
            "stages": settings.stages,
            "spacing": settings.spacing,
            "volume_labels": settings.volume_labels,
            "volume_pos": settings.volume_pos,
            "run": settings.run,
            "condition": settings.condition,
            "alignment": settings.alignment,
            "method": settings.method,
            "snr": settings.snr,
        },
    )
