"""
Stage 5 ERP slicing, source parcellation, and derivative export.

This stage consumes Stage 3 source-estimate manifests and Stage 1 behavior
tables. It writes trial-level parcellated source time courses as `.npy` arrays
with JSON sidecars plus an aligned trial metadata table.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import mne
import numpy as np
import pandas as pd

from meg_tokens.io import derivative_path, require_file, save_array, save_table
from meg_tokens.meg.erp import align_and_pad_epochs, parcellate_source_estimates, select_source_feature_data
from meg_tokens.meg.sources import source_derivative_path
from meg_tokens.utils.batch_processor import normalize_subject_id
from meg_tokens.utils.batch_time_frequency import find_stc_manifest
from meg_tokens.utils.epochs_builder import find_behavior_table, load_behavior_table, parse_run_label


FEATURE_SPACES = ("parcellated", "all_source", "volume")


def erp_derivative_path(
    output_root: str,
    *,
    subject: str,
    run: str,
    condition: Optional[str],
    align_to: str,
    source_method: str,
    parc: str,
    suffix: str,
    extension: str,
) -> Path:
    subject = normalize_subject_id(subject)
    run_number, inferred_condition = parse_run_label(run)
    condition = condition or inferred_condition
    desc_parts = []
    if condition:
        desc_parts.append(condition.lower())
    desc_parts.extend([align_to, source_method, parc])
    return derivative_path(
        output_root,
        subject=subject,
        datatype="meg",
        task="tokens",
        run=run_number,
        description="-".join(desc_parts),
        suffix=suffix,
        extension=extension,
    )


def find_source_space_file(source_dir: str, subject: str, spacing: str) -> Optional[Path]:
    """Find a Stage 3 source-space file if MNE label extraction is requested."""
    expected = source_derivative_path(
        source_dir,
        subject,
        suffix="src",
        extension=".fif",
        description=spacing,
        space="subject",
    )
    if expected.is_file():
        return expected
    subject = normalize_subject_id(subject)
    matches = sorted(Path(source_dir).glob(f"**/sub-{subject}_task-tokens_space-subject_desc-{spacing}_src.fif"))
    existing = [path for path in matches if path.is_file()]
    if not existing:
        return None
    if len(existing) > 1:
        raise ValueError(f"Multiple source-space files matched subject={subject}, spacing={spacing}: {existing}")
    return existing[0]


def _read_manifest(path: str | Path) -> pd.DataFrame:
    manifest_path = require_file(path, purpose="Stage 3 source-estimate manifest")
    manifest = pd.read_csv(manifest_path, sep="\t")
    required = {"trial", "stc_base", "subject", "run", "alignment", "method"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest {manifest_path} is missing required columns: {missing}")
    if manifest.empty:
        raise ValueError(f"Manifest {manifest_path} does not contain any trials")
    return manifest


def _source_files_for_base(base_path: str | Path) -> list[Path]:
    base = Path(base_path)
    candidates = [
        base,
        base.with_name(f"{base.name}-lh.stc"),
        base.with_name(f"{base.name}-rh.stc"),
        base.with_name(f"{base.name}-vl.stc"),
        base.with_name(f"{base.name}-vol.stc"),
        base.with_suffix(".stc"),
    ]
    return [path for path in candidates if path.is_file()]


def _read_source_estimate(base_path: str | Path):
    if not _source_files_for_base(base_path):
        raise FileNotFoundError(f"Source-estimate files do not exist for base path: {base_path}")
    return mne.read_source_estimate(str(base_path))


def _single_manifest_value(table: pd.DataFrame, key: str, default=None):
    if key not in table.columns:
        return default
    values = table[key].dropna().unique()
    if len(values) == 0:
        return default
    if len(values) > 1:
        raise ValueError(f"Manifest column '{key}' contains multiple values: {list(values)}")
    return values[0]


def _same_vertices(left, right) -> bool:
    if len(left) != len(right):
        return False
    return all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(left, right))


def _validate_source_estimate(stc, reference, row_index: int) -> None:
    if stc.__class__ is not reference.__class__:
        raise ValueError(
            f"Source estimate at manifest row {row_index} has type "
            f"{stc.__class__.__name__}; expected {reference.__class__.__name__}"
        )
    if stc.data.shape != reference.data.shape:
        raise ValueError(
            f"Source estimate at manifest row {row_index} has shape "
            f"{stc.data.shape}; expected {reference.data.shape}"
        )
    if not np.isclose(stc.tmin, reference.tmin) or not np.isclose(stc.tstep, reference.tstep):
        raise ValueError(f"Source estimate at manifest row {row_index} has inconsistent timing")
    if not _same_vertices(stc.vertices, reference.vertices):
        raise ValueError(f"Source estimate at manifest row {row_index} has inconsistent vertices")


def _make_source_estimate_like(stc, data: np.ndarray):
    return stc.__class__(
        data,
        vertices=stc.vertices,
        tmin=stc.tmin,
        tstep=stc.tstep,
        subject=stc.subject,
    )


def _output_dims(stacked: np.ndarray, feature_space: str) -> tuple[str, ...]:
    if feature_space == "parcellated":
        if stacked.ndim == 3:
            return ("trial", "label", "time")
        if stacked.ndim == 4:
            return ("trial", "component", "label", "time")
    else:
        if stacked.ndim == 3:
            return ("trial", "source", "time")
        if stacked.ndim == 4:
            return ("trial", "source", "orientation", "time")
    raise ValueError(f"Unexpected ERP shape for feature_space={feature_space}: {stacked.shape}")


def _output_coords(
    stacked: np.ndarray,
    trials: Sequence[int],
    feature_names: Sequence[str],
    times: np.ndarray,
    feature_space: str,
) -> dict:
    coords = {
        "trial": list(trials),
        "time_sec": times,
    }
    if feature_space == "parcellated":
        coords["label"] = list(feature_names)
        if stacked.ndim == 4:
            coords["component"] = ["x", "y", "z"] if stacked.shape[1] == 3 else list(range(stacked.shape[1]))
    else:
        coords["source"] = list(feature_names)
        if stacked.ndim == 4:
            coords["orientation"] = ["x", "y", "z"] if stacked.shape[2] == 3 else list(range(stacked.shape[2]))
    return coords


def _time_coordinates(reference, align_to: str, max_duration_samples: int) -> np.ndarray:
    if align_to == "go":
        return reference.tmin + np.arange(max_duration_samples, dtype=float) * reference.tstep
    return reference.tmin + np.arange(reference.data.shape[-1], dtype=float) * reference.tstep


def extract_parcellated_erp_from_manifest(
    manifest_path: str | Path,
    behavior_path: str | Path,
    subjects_dir: Optional[str],
    out_dir: str,
    *,
    parc: str = "HCPMMP1",
    feature_space: str = "parcellated",
    hemi: str = "both",
    label_subject: Optional[str] = None,
    source_space_path: Optional[str | Path] = None,
    label_mode: str = "mean",
    max_duration_samples: int = 400,
    cutoff_before_enter_ms: float = 300.0,
    min_rt_ms: float = 100.0,
) -> dict[str, Path]:
    """Slice, optionally parcellate, and save one run of trial source estimates."""
    if feature_space not in FEATURE_SPACES:
        raise ValueError(f"feature_space must be one of {FEATURE_SPACES}, got {feature_space!r}")
    if feature_space == "parcellated" and not subjects_dir:
        raise ValueError("subjects_dir is required for parcellated ERP exports")
    if feature_space != "parcellated" and source_space_path is not None:
        raise ValueError("source_space_path is only valid for parcellated ERP exports")

    manifest_path = Path(require_file(manifest_path, purpose="Stage 3 source-estimate manifest"))
    manifest = _read_manifest(manifest_path)
    behavior_path = Path(require_file(behavior_path, purpose="Stage 1 behavior TSV derivative"))
    behavior = load_behavior_table(str(behavior_path))

    if len(behavior) != len(manifest):
        raise ValueError(
            f"Trial count mismatch for ERP export: behavior rows={len(behavior)}, "
            f"source estimates={len(manifest)}"
        )

    stcs = []
    reference = None
    for row_index, row in manifest.reset_index(drop=True).iterrows():
        stc = _read_source_estimate(row["stc_base"])
        if reference is None:
            reference = stc
        else:
            _validate_source_estimate(stc, reference, row_index)
        stcs.append(stc)

    if reference is None:
        raise ValueError(f"Manifest {manifest_path} did not yield any source estimates")

    subject = str(_single_manifest_value(manifest, "subject", reference.subject or ""))
    run = str(_single_manifest_value(manifest, "run"))
    condition = _single_manifest_value(manifest, "condition")
    align_to = str(_single_manifest_value(manifest, "alignment"))
    source_method = str(_single_manifest_value(manifest, "method"))
    if align_to not in {"go", "enter", "feedback"}:
        raise ValueError(f"Unsupported ERP alignment: {align_to}")

    sfreq = 1.0 / float(reference.tstep)
    aligned = align_and_pad_epochs(
        stcs,
        behavior,
        align_to=align_to,
        tmin=reference.tmin,
        sfreq=sfreq,
        max_duration_samples=max_duration_samples,
        cutoff_before_enter_ms=cutoff_before_enter_ms,
        min_rt_ms=min_rt_ms,
    )

    source_space = None
    if source_space_path is not None:
        source_space = mne.read_source_spaces(str(require_file(source_space_path, purpose="Stage 3 source space")))

    if feature_space == "parcellated":
        label_subject = label_subject or reference.subject or normalize_subject_id(subject)
    else:
        label_subject = None

    feature_trials = []
    included_trial_numbers = []
    included_rows = []
    feature_names = None

    for trial_pos, aligned_data in enumerate(aligned):
        if aligned_data is None:
            continue
        aligned_stc = _make_source_estimate_like(stcs[trial_pos], np.asarray(aligned_data))
        if feature_space == "parcellated":
            names, feature_data = parcellate_source_estimates(
                aligned_stc,
                subjects_dir=str(subjects_dir),
                subject=label_subject,
                parc=parc,
                hemi=hemi,
                source_space=source_space,
                mode=label_mode,
            )
        else:
            names, feature_data = select_source_feature_data(aligned_stc, feature_space)

        if feature_names is None:
            feature_names = names
        elif list(feature_names) != list(names):
            raise ValueError("ERP feature coordinates changed across trials")

        feature_trials.append(np.asarray(feature_data))
        trial_number = int(manifest.iloc[trial_pos]["trial"])
        included_trial_numbers.append(trial_number)
        trial_row = behavior.iloc[trial_pos].to_dict()
        trial_row.update({
            "manifest_trial": trial_number,
            "stc_base": manifest.iloc[trial_pos]["stc_base"],
            "output_trial_index": len(included_trial_numbers),
        })
        included_rows.append(trial_row)

    if not feature_trials:
        raise ValueError("No trials survived ERP alignment and RT filtering")
    if feature_names is None:
        raise ValueError("ERP export did not produce any feature coordinates")

    stacked = np.stack(feature_trials, axis=0)
    times = _time_coordinates(reference, align_to, max_duration_samples)
    if stacked.shape[-1] != len(times):
        raise ValueError(f"Time coordinate length {len(times)} does not match data shape {stacked.shape}")

    condition_for_path = None if pd.isna(condition) else str(condition)
    path_feature_name = parc if feature_space == "parcellated" else feature_space.replace("_", "-")
    data_path = erp_derivative_path(
        out_dir,
        subject=subject,
        run=run,
        condition=condition_for_path,
        align_to=align_to,
        source_method=source_method,
        parc=path_feature_name,
        suffix="erp",
        extension=".npy",
    )
    trials_path = erp_derivative_path(
        out_dir,
        subject=subject,
        run=run,
        condition=condition_for_path,
        align_to=align_to,
        source_method=source_method,
        parc=path_feature_name,
        suffix="erptrials",
        extension=".tsv",
    )
    save_table(
        trials_path,
        pd.DataFrame(included_rows),
        metadata={
            "stage": "erp_parcellation",
            "kind": "included_trial_metadata",
            "subject": normalize_subject_id(subject),
            "run": parse_run_label(run)[0],
            "condition": condition_for_path,
            "alignment": align_to,
            "source_method": source_method,
            "feature_space": feature_space,
            "parcellation": path_feature_name,
            "atlas": parc if feature_space == "parcellated" else None,
            "hemi": hemi if feature_space == "parcellated" else None,
            "input_manifest": str(manifest_path),
            "input_behavior": str(behavior_path),
            "n_included_trials": len(included_rows),
        },
    )
    save_array(
        data_path,
        stacked,
        dims=_output_dims(stacked, feature_space),
        coords=_output_coords(stacked, included_trial_numbers, feature_names, times, feature_space),
        metadata={
            "stage": "erp_parcellation",
            "kind": f"{feature_space}_source_timeseries",
            "subject": normalize_subject_id(subject),
            "run": parse_run_label(run)[0],
            "condition": condition_for_path,
            "alignment": align_to,
            "source_method": source_method,
            "source_estimate_type": reference.__class__.__name__,
            "feature_space": feature_space,
            "parcellation": path_feature_name,
            "atlas": parc if feature_space == "parcellated" else None,
            "hemi": hemi if feature_space == "parcellated" else None,
            "label_subject": label_subject,
            "label_mode": (
                label_mode
                if feature_space == "parcellated" and source_space is not None
                else ("manual-mean" if feature_space == "parcellated" else None)
            ),
            "sfreq_hz": float(sfreq),
            "max_duration_samples": int(max_duration_samples) if align_to == "go" else None,
            "cutoff_before_enter_ms": float(cutoff_before_enter_ms) if align_to == "go" else None,
            "min_rt_ms": float(min_rt_ms),
            "input_manifest": str(manifest_path),
            "input_behavior": str(behavior_path),
            "trial_table": str(trials_path),
            "source_vertices": [np.asarray(vertices).tolist() for vertices in reference.vertices],
        },
    )
    print(f"Saved ERP array: {data_path}")
    print(f"Saved ERP trial metadata: {trials_path}")
    return {"data": data_path, "trials": trials_path}


def run_erp_parcellation_pipeline(
    subjects_list: Sequence[str],
    source_dir: str,
    behavior_dir: str,
    subjects_dir: Optional[str],
    out_dir: str,
    *,
    run: str,
    condition: Optional[str] = None,
    align_to: str = "go",
    source_method: str = "dSPM",
    parc: str = "HCPMMP1",
    feature_space: str = "parcellated",
    hemi: str = "both",
    label_subject: Optional[str] = None,
    spacing: Optional[str] = None,
    label_mode: str = "mean",
    max_duration_samples: int = 400,
    cutoff_before_enter_ms: float = 300.0,
    min_rt_ms: float = 100.0,
) -> dict[str, dict[str, Path]]:
    outputs = {}
    for subject in subjects_list:
        subject = normalize_subject_id(subject)
        manifest = find_stc_manifest(source_dir, subject, run, condition, align_to, source_method)
        behavior = find_behavior_table(behavior_dir, subject, run, condition)
        source_space = find_source_space_file(source_dir, subject, spacing) if spacing else None
        print(f"=== ERP parcellation for {subject}: {manifest} ===")
        outputs[subject] = extract_parcellated_erp_from_manifest(
            manifest,
            behavior,
            subjects_dir,
            out_dir,
            parc=parc,
            feature_space=feature_space,
            hemi=hemi,
            label_subject=label_subject,
            source_space_path=source_space,
            label_mode=label_mode,
            max_duration_samples=max_duration_samples,
            cutoff_before_enter_ms=cutoff_before_enter_ms,
            min_rt_ms=min_rt_ms,
        )
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ERP slicing, parcellation, and array export.")
    parser.add_argument("--manifest", type=str, nargs="+", default=None,
                        help="One or more Stage 3 stcmanifest TSV files.")
    parser.add_argument("--subjects", type=str, nargs="+", default=None,
                        help="Subject IDs to process when --manifest is omitted.")
    parser.add_argument("--source_dir", type=str, default=None,
                        help="BIDS derivatives root containing Stage 3 source manifests.")
    parser.add_argument("--behavior_dir", type=str, required=True,
                        help="BIDS derivatives root containing Stage 1 behavior TSV files.")
    parser.add_argument("--subjects_dir", type=str, default=None,
                        help="FreeSurfer subjects directory containing annotations. Required for parcellated exports.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="BIDS derivatives root for ERP/parcellation arrays.")
    parser.add_argument("--run", type=str, default=None,
                        help="Run label, for example Slow1 or 1. Required when --manifest is omitted.")
    parser.add_argument("--condition", type=str, default=None)
    parser.add_argument("--align_to", type=str, default="go", choices=["go", "enter", "feedback"])
    parser.add_argument("--source_method", type=str, default="dSPM")
    parser.add_argument("--parc", type=str, default="HCPMMP1")
    parser.add_argument("--feature_space", type=str, default="parcellated", choices=list(FEATURE_SPACES),
                        help="Export parcellated labels, all source vertices, or volume source vertices.")
    parser.add_argument("--hemi", type=str, default="both", choices=["left", "right", "both"])
    parser.add_argument("--label_subject", type=str, default=None,
                        help="FreeSurfer subject to read labels from. Defaults to the STC subject.")
    parser.add_argument("--source_space", type=str, default=None,
                        help="Optional Stage 3 source-space FIF for MNE label time-course extraction. Only valid with one manifest.")
    parser.add_argument("--spacing", type=str, default=None,
                        help="Find the Stage 3 source-space derivative for this spacing and use MNE label extraction.")
    parser.add_argument("--label_mode", type=str, default="mean",
                        help="MNE label time-course extraction mode when a source space is supplied.")
    parser.add_argument("--max_duration_samples", type=int, default=400)
    parser.add_argument("--cutoff_before_enter_ms", type=float, default=300.0)
    parser.add_argument("--min_rt_ms", type=float, default=100.0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.manifest:
        if args.source_space is not None and len(args.manifest) != 1:
            parser.error("--source_space can only be used with one --manifest")
        for manifest in args.manifest:
            manifest_table = _read_manifest(manifest)
            subject = str(_single_manifest_value(manifest_table, "subject"))
            run = str(_single_manifest_value(manifest_table, "run"))
            condition = _single_manifest_value(manifest_table, "condition")
            behavior = find_behavior_table(args.behavior_dir, subject, run, None if pd.isna(condition) else str(condition))
            extract_parcellated_erp_from_manifest(
                manifest,
                behavior,
                args.subjects_dir,
                args.out_dir,
                parc=args.parc,
                feature_space=args.feature_space,
                hemi=args.hemi,
                label_subject=args.label_subject,
                source_space_path=args.source_space,
                label_mode=args.label_mode,
                max_duration_samples=args.max_duration_samples,
                cutoff_before_enter_ms=args.cutoff_before_enter_ms,
                min_rt_ms=args.min_rt_ms,
            )
        return

    if not args.subjects:
        parser.error("--subjects is required when --manifest is omitted")
    if args.source_dir is None:
        parser.error("--source_dir is required when --manifest is omitted")
    if args.run is None:
        parser.error("--run is required when --manifest is omitted")

    run_erp_parcellation_pipeline(
        args.subjects,
        args.source_dir,
        args.behavior_dir,
        args.subjects_dir,
        args.out_dir,
        run=args.run,
        condition=args.condition,
        align_to=args.align_to,
        source_method=args.source_method,
        parc=args.parc,
        feature_space=args.feature_space,
        hemi=args.hemi,
        label_subject=args.label_subject,
        spacing=args.spacing,
        label_mode=args.label_mode,
        max_duration_samples=args.max_duration_samples,
        cutoff_before_enter_ms=args.cutoff_before_enter_ms,
        min_rt_ms=args.min_rt_ms,
    )


if __name__ == "__main__":
    main()
