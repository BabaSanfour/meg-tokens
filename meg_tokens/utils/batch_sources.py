"""
Pipeline execution script for Stage 3: source reconstruction.
"""

import os
import glob
import mne
from mne.minimum_norm import read_inverse_operator
from pathlib import Path
from typing import List, Optional

from meg_tokens.io import require_file
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
    source_derivative_path,
)
from meg_tokens.utils.batch_processor import normalize_subject_id
from meg_tokens.utils.epochs_builder import parse_run_label


DEFAULT_SOURCE_STAGES = ("cov", "bem", "src", "fwd", "inv", "apply")


def find_noise_file(raw_dir: str, subject: str) -> str:
    subject = normalize_subject_id(subject)
    patterns = [
        os.path.join(raw_dir, subject, "*noise*.ds"),
        os.path.join(raw_dir, subject, "*noise*.fif"),
        os.path.join(raw_dir, "sub-" + subject, "meg", "*noise*.fif"),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No empty-room noise file found for {subject} under {raw_dir}")
    if len(files) > 1:
        raise ValueError(f"Multiple empty-room noise files found for {subject}: {sorted(files)}")
    return files[0]


def find_epoch_file(epochs_dir: str, subject: str, run_id: Optional[str], condition: Optional[str], align_to: str) -> str:
    subject = normalize_subject_id(subject)
    root = Path(epochs_dir)
    run, inferred_condition = parse_run_label(run_id or "1")
    condition = condition or inferred_condition
    desc = f"{condition.lower()}-{align_to}" if condition else align_to
    candidates = [
        root / "derivatives" / "meg-tokens" / f"sub-{subject}" / "meg" / f"sub-{subject}_task-tokens_run-{run}_desc-{desc}_epo.fif",
    ]
    candidates.extend(root.glob(f"**/sub-{subject}_task-tokens_run-{run}_desc-*{align_to}*_epo.fif"))
    existing = sorted({path for path in candidates if path.is_file()})
    if not existing:
        raise FileNotFoundError(f"No Stage 2 epochs found for subject={subject}, run={run}, alignment={align_to} under {epochs_dir}")
    if len(existing) > 1:
        raise ValueError(f"Multiple epoch files matched subject={subject}, run={run}: {existing}")
    return str(existing[0])


def find_trans_file(trans_dir: str, subject: str, run_id: Optional[str] = None) -> str:
    subject = normalize_subject_id(subject)
    root = Path(trans_dir)
    patterns = [
        f"**/sub-{subject}*trans.fif",
        f"**/{subject}*trans.fif",
    ]
    if run_id is not None:
        run, _ = parse_run_label(run_id)
        patterns.insert(0, f"**/sub-{subject}*run-{run}*trans.fif")

    files = []
    for pattern in patterns:
        files.extend(root.glob(pattern))
    existing = sorted({path for path in files if path.is_file()})
    if not existing:
        raise FileNotFoundError(f"No MEG-MRI trans file found for {subject} under {trans_dir}")
    if len(existing) > 1:
        raise ValueError(f"Multiple trans files matched {subject}: {existing}")
    return str(existing[0])


def source_space_description(spacing: str, volume_labels: Optional[List[str]] = None) -> str:
    """Return the source-space description used in derivative filenames."""
    return f"{spacing}-mixed" if volume_labels else spacing


def model_paths(out_dir: str, subject: str, spacing: str, volume_labels: Optional[List[str]] = None) -> dict:
    src_desc = source_space_description(spacing, volume_labels)
    return {
        "cov": source_derivative_path(out_dir, subject, suffix="cov", extension=".fif", description="noise"),
        "bem": source_derivative_path(out_dir, subject, suffix="bem", extension=".fif", description="singlelayer"),
        "src": source_derivative_path(out_dir, subject, suffix="src", extension=".fif", description=src_desc, space="subject"),
    }

def run_sources_pipeline(
    subjects_list: list,
    raw_dir: str,
    subjects_dir: str,
    out_dir: str,
    spacing: str = 'oct6',
    epochs_dir: Optional[str] = None,
    trans_dir: Optional[str] = None,
    stages: Optional[List[str]] = None,
    run: Optional[str] = None,
    condition: Optional[str] = None,
    align_to: str = "go",
    method: str = "dSPM",
    snr: float = 1.0,
    volume_labels: Optional[List[str]] = None,
    volume_pos: float = 5.0,
):
    stages = stages or list(DEFAULT_SOURCE_STAGES)
    invalid = sorted(set(stages) - set(DEFAULT_SOURCE_STAGES))
    if invalid:
        raise ValueError(f"Unknown source stages: {invalid}")

    for subject in subjects_list:
        subject = normalize_subject_id(subject)
        print(f"=== Running Source Localization for {subject} ===")
        paths = model_paths(out_dir, subject, spacing, volume_labels)
        src_desc = source_space_description(spacing, volume_labels)

        cov = None
        bem = None
        src = None

        if "cov" in stages:
            noise_file = find_noise_file(raw_dir, subject)
            cov = compute_noise_covariance(noise_file)
            save_noise_covariance(cov, out_dir, subject)
        elif any(stage in stages for stage in ("inv",)):
            cov = mne.read_cov(str(require_file(paths["cov"], purpose=f"{subject} noise covariance")))

        if "bem" in stages:
            bem = setup_bem_solution(subject, subjects_dir=subjects_dir)
            save_bem_solution(bem, out_dir, subject)
        elif any(stage in stages for stage in ("fwd",)):
            bem = mne.read_bem_solution(str(require_file(paths["bem"], purpose=f"{subject} BEM")))

        if "src" in stages:
            src = setup_mixed_source_space(
                subject,
                subjects_dir=subjects_dir,
                spacing=spacing,
                volume_labels=volume_labels,
                volume_pos=volume_pos,
                bem=bem,
            )
            save_source_space(
                src,
                out_dir,
                subject,
                src_desc,
                volume_labels=volume_labels,
                volume_pos=volume_pos if volume_labels else None,
            )
        elif any(stage in stages for stage in ("fwd",)):
            src = mne.read_source_spaces(str(require_file(paths["src"], purpose=f"{subject} source space")))

        if any(stage in stages for stage in ("fwd", "inv", "apply")):
            if epochs_dir is None:
                raise ValueError("--epochs_dir is required for fwd/inv/apply stages")
            if trans_dir is None:
                raise ValueError("--trans_dir is required for fwd/inv/apply stages")
            epoch_file = find_epoch_file(epochs_dir, subject, run, condition, align_to)
            epochs = mne.read_epochs(epoch_file, preload=True)
            run_id = run or parse_run_label(Path(epoch_file).name.split("_run-", 1)[1].split("_", 1)[0])[0]
            _, inferred_condition = parse_run_label(run_id)
            condition_for_save = condition or inferred_condition

            fwd = None
            inv = None
            if "fwd" in stages:
                if src is None:
                    src = mne.read_source_spaces(str(require_file(paths["src"], purpose=f"{subject} source space")))
                if bem is None:
                    bem = mne.read_bem_solution(str(require_file(paths["bem"], purpose=f"{subject} BEM")))
                trans_file = find_trans_file(trans_dir, subject, run_id)
                fwd = compute_forward_solution(epochs.info, trans_file, src, bem)
                save_forward_solution(fwd, out_dir, subject, run_id, condition_for_save, align_to)
            elif any(stage in stages for stage in ("inv",)):
                fwd_path = source_derivative_path(out_dir, subject, suffix="fwd", extension=".fif", run_id=run_id, condition=condition_for_save, description=align_to)
                fwd = mne.read_forward_solution(str(require_file(fwd_path, purpose=f"{subject} forward solution")))

            if "inv" in stages:
                if cov is None:
                    cov = mne.read_cov(str(require_file(paths["cov"], purpose=f"{subject} noise covariance")))
                inv = build_inverse_operator(epochs.info, fwd, cov)
                save_inverse_operator(inv, out_dir, subject, run_id, condition_for_save, align_to, method)
            elif "apply" in stages:
                inv_path = source_derivative_path(out_dir, subject, suffix="inv", extension=".fif", run_id=run_id, condition=condition_for_save, description=f"{align_to}-{method}")
                inv = read_inverse_operator(str(require_file(inv_path, purpose=f"{subject} inverse operator")))

            if "apply" in stages:
                if inv is None:
                    raise RuntimeError("Inverse operator was not available for apply stage")
                stcs = apply_inverse_operator(epochs, inv, method=method, snr=snr)
                manifest = save_source_estimates(stcs, out_dir, subject, run_id, condition_for_save, align_to, method)
                print(f"Saved source-estimate manifest: {manifest}")

        print(f"Saved Source Models for {subject}.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run source localization pipeline.")
    parser.add_argument("--subjects", type=str, nargs='+', default=['H01'])
    parser.add_argument("--raw_dir", type=str, default='/media/external/DDM/MEG_data/')
    parser.add_argument("--epochs_dir", type=str, default=None,
                        help="Stage 2 derivatives root containing epochs; required for fwd/inv/apply.")
    parser.add_argument("--trans_dir", type=str, default=None,
                        help="Directory containing MEG-MRI trans FIF files; required for fwd/inv/apply.")
    parser.add_argument("--subjects_dir", type=str, default='/media/external/DDM/IRM/')
    parser.add_argument("--out_dir", type=str, default='/media/external/DDM/source_rec/')
    parser.add_argument("--spacing", type=str, default='oct6')
    parser.add_argument("--volume_labels", type=str, nargs="+", default=None,
                        help="Optional FreeSurfer aseg labels for mixed surface+volume source spaces.")
    parser.add_argument("--volume_pos", type=float, default=5.0,
                        help="Volume grid spacing in millimeters when --volume_labels is used.")
    parser.add_argument("--stages", type=str, nargs="+", default=list(DEFAULT_SOURCE_STAGES),
                        choices=list(DEFAULT_SOURCE_STAGES),
                        help="Source stages to run in order.")
    parser.add_argument("--run", type=str, default=None, help="Run label for fwd/inv/apply stages.")
    parser.add_argument("--condition", type=str, default=None, help="Condition label for fwd/inv/apply outputs.")
    parser.add_argument("--align_to", type=str, default="go", choices=["go", "enter", "feedback"])
    parser.add_argument("--method", type=str, default="dSPM")
    parser.add_argument("--snr", type=float, default=1.0)
    args = parser.parse_args()
    
    run_sources_pipeline(
        args.subjects,
        args.raw_dir,
        args.subjects_dir,
        args.out_dir,
        args.spacing,
        epochs_dir=args.epochs_dir,
        trans_dir=args.trans_dir,
        stages=args.stages,
        run=args.run,
        condition=args.condition,
        align_to=args.align_to,
        method=args.method,
        snr=args.snr,
        volume_labels=args.volume_labels,
        volume_pos=args.volume_pos,
    )
