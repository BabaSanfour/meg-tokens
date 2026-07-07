"""
Stage 6 group-level statistics over parcellated ERP derivatives.

This stage consumes Stage 5 `*_erp.npy` array derivatives, builds paired
subject-level condition contrasts, and writes statistics as `.npy` arrays with
JSON sidecars plus a table of significant time windows.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from meg_tokens.io import derivative_path, load_array, save_array, save_table
from meg_tokens.meg.stats import compute_permutation_t_test, get_significance_windows
from meg_tokens.utils.batch_erp_parcellation import erp_derivative_path
from meg_tokens.utils.batch_processor import normalize_subject_id
from meg_tokens.utils.epochs_builder import parse_run_label


def stats_derivative_path(
    output_root: str,
    *,
    conditions: Sequence[str],
    align_to: str,
    source_method: str,
    parc: str,
    suffix: str,
    extension: str,
) -> Path:
    """Build a group-level stats derivative path."""
    desc = "-".join([
        conditions[0].lower(),
        "vs",
        conditions[1].lower(),
        align_to,
        source_method,
        parc,
    ])
    return derivative_path(
        output_root,
        subject="group",
        datatype="meg",
        task="tokens",
        description=desc,
        suffix=suffix,
        extension=extension,
    )


def find_erp_arrays(
    erp_dir: str,
    subject: str,
    condition: str,
    *,
    align_to: str,
    source_method: str,
    parc: str,
    runs: Optional[Sequence[str]] = None,
) -> list[Path]:
    """Find Stage 5 ERP array derivatives for one subject and condition."""
    subject = normalize_subject_id(subject)
    if runs:
        candidates = [
            erp_derivative_path(
                erp_dir,
                subject=subject,
                run=parse_run_label(run)[0],
                condition=condition,
                align_to=align_to,
                source_method=source_method,
                parc=parc,
                suffix="erp",
                extension=".npy",
            )
            for run in runs
        ]
        existing = [path for path in candidates if path.is_file()]
    else:
        pattern = (
            f"**/sub-{subject}_task-tokens_run-*_"
            f"desc-{condition.lower()}-{align_to}-{source_method}-{parc}_erp.npy"
        )
        existing = sorted(path for path in Path(erp_dir).glob(pattern) if path.is_file())

    if not existing:
        detail = f", runs={list(runs)}" if runs else ""
        raise FileNotFoundError(
            "No Stage 5 ERP derivatives found for "
            f"subject={subject}, condition={condition}, alignment={align_to}, "
            f"source_method={source_method}, parc={parc}{detail}"
        )
    return sorted(existing)


def discover_subjects(
    erp_dir: str,
    conditions: Sequence[str],
    *,
    align_to: str,
    source_method: str,
    parc: str,
) -> list[str]:
    """Discover subjects that have at least one ERP derivative for both conditions."""
    subject_sets = []
    for condition in conditions:
        pattern = (
            f"**/sub-*_task-tokens_run-*_"
            f"desc-{condition.lower()}-{align_to}-{source_method}-{parc}_erp.npy"
        )
        subjects = {
            path.name.split("_", 1)[0].replace("sub-", "")
            for path in Path(erp_dir).glob(pattern)
            if path.is_file()
        }
        subject_sets.append(subjects)

    common = set.intersection(*subject_sets) if subject_sets else set()
    if not common:
        raise FileNotFoundError(
            "No subjects have ERP derivatives for both requested conditions: "
            f"{conditions}"
        )
    return sorted(common)


def _coords_close(left, right) -> bool:
    try:
        return np.allclose(np.asarray(left, dtype=float), np.asarray(right, dtype=float), equal_nan=True)
    except (TypeError, ValueError):
        return list(left) == list(right)


def _load_subject_condition_mean(paths: Sequence[Path]) -> tuple[np.ndarray, tuple[str, ...], dict[str, object], list[str]]:
    """Average trials, then runs, for one subject-condition cell."""
    run_means = []
    reference_dims = None
    reference_coords = None
    source_paths = []

    for path in paths:
        loaded = load_array(path, require_sidecar=True)
        data = loaded.data
        sidecar = loaded.metadata
        dims = tuple(sidecar.get("dims", []))
        if not dims or dims[0] != "trial":
            raise ValueError(f"Expected first dimension to be trial in {path}, got {dims}")
        if data.ndim not in (3, 4):
            raise ValueError(f"Expected ERP array to be 3D or 4D, got shape {data.shape} in {path}")

        feature_dims = dims[1:]
        coords = sidecar.get("coords", {})
        feature_coords = {key: value for key, value in coords.items() if key != "trial"}
        trial_mean = np.nanmean(data, axis=0)

        if reference_dims is None:
            reference_dims = feature_dims
            reference_coords = feature_coords
        else:
            if feature_dims != reference_dims:
                raise ValueError(f"ERP feature dims changed across runs: {feature_dims} != {reference_dims}")
            for key, value in reference_coords.items():
                if key in feature_coords and not _coords_close(value, feature_coords[key]):
                    raise ValueError(f"ERP coordinate '{key}' changed across runs")

        run_means.append(trial_mean)
        source_paths.append(str(path))

    if reference_dims is None or reference_coords is None:
        raise ValueError("No ERP paths were provided")
    return np.nanmean(np.stack(run_means, axis=0), axis=0), reference_dims, reference_coords, source_paths


def _load_condition_matrix(
    erp_dir: str,
    subjects: Sequence[str],
    condition: str,
    *,
    align_to: str,
    source_method: str,
    parc: str,
    runs: Optional[Sequence[str]],
) -> tuple[np.ndarray, tuple[str, ...], dict[str, object], dict[str, list[str]]]:
    subject_means = []
    reference_dims = None
    reference_coords = None
    inputs = {}

    for subject in subjects:
        paths = find_erp_arrays(
            erp_dir,
            subject,
            condition,
            align_to=align_to,
            source_method=source_method,
            parc=parc,
            runs=runs,
        )
        subject_mean, dims, coords, source_paths = _load_subject_condition_mean(paths)
        if reference_dims is None:
            reference_dims = dims
            reference_coords = coords
        else:
            if dims != reference_dims:
                raise ValueError(f"ERP feature dims changed across subjects: {dims} != {reference_dims}")
            for key, value in reference_coords.items():
                if key in coords and not _coords_close(value, coords[key]):
                    raise ValueError(f"ERP coordinate '{key}' changed across subjects")
        subject_means.append(subject_mean)
        inputs[normalize_subject_id(subject)] = source_paths

    if reference_dims is None or reference_coords is None:
        raise ValueError(f"No ERP data loaded for condition {condition}")
    return np.stack(subject_means, axis=0), reference_dims, reference_coords, inputs


def _window_table(
    p_values: np.ndarray,
    dims: Sequence[str],
    coords: Mapping[str, object],
    *,
    alpha: float,
) -> pd.DataFrame:
    time_values = coords.get("time_sec", list(range(p_values.shape[-1])))
    rows = []

    if tuple(dims) == ("label", "time"):
        labels = coords.get("label", list(range(p_values.shape[0])))
        for label_idx, label in enumerate(labels):
            for start, end in get_significance_windows(p_values[label_idx], alpha=alpha):
                rows.append({
                    "label": label,
                    "label_index": label_idx,
                    "start_index": int(start),
                    "end_index_exclusive": int(end),
                    "start_time_sec": time_values[start] if start < len(time_values) else np.nan,
                    "end_time_sec": time_values[end - 1] if end - 1 < len(time_values) else np.nan,
                    "alpha": alpha,
                })
    elif tuple(dims) == ("component", "label", "time"):
        components = coords.get("component", list(range(p_values.shape[0])))
        labels = coords.get("label", list(range(p_values.shape[1])))
        for comp_idx, component in enumerate(components):
            for label_idx, label in enumerate(labels):
                for start, end in get_significance_windows(p_values[comp_idx, label_idx], alpha=alpha):
                    rows.append({
                        "component": component,
                        "component_index": comp_idx,
                        "label": label,
                        "label_index": label_idx,
                        "start_index": int(start),
                        "end_index_exclusive": int(end),
                        "start_time_sec": time_values[start] if start < len(time_values) else np.nan,
                        "end_time_sec": time_values[end - 1] if end - 1 < len(time_values) else np.nan,
                        "alpha": alpha,
                    })
    else:
        return pd.DataFrame(columns=["feature_index", "start_index", "end_index_exclusive", "alpha"])

    if rows:
        return pd.DataFrame(rows)
    columns = ["label", "label_index", "start_index", "end_index_exclusive", "start_time_sec", "end_time_sec", "alpha"]
    if tuple(dims) == ("component", "label", "time"):
        columns = ["component", "component_index", *columns]
    return pd.DataFrame(columns=columns)


def run_group_statistics_contrast(
    erp_dir: str,
    out_dir: str,
    conditions: Sequence[str] = ("Fast", "Slow"),
    n_permutations: int = 1000,
    subjects_list: Optional[Sequence[str]] = None,
    *,
    align_to: str = "go",
    source_method: str = "dSPM",
    parc: str = "HCPMMP1",
    runs_by_condition: Optional[Mapping[str, Sequence[str]]] = None,
    alpha: float = 0.05,
    tail: int = 0,
    n_jobs: int = 1,
) -> dict[str, Path]:
    """Run a paired group-level permutation t-test for two ERP conditions."""
    if len(conditions) != 2:
        raise ValueError("Group statistics currently requires exactly two conditions")
    conditions = [str(condition) for condition in conditions]
    runs_by_condition = runs_by_condition or {}

    if subjects_list is None:
        subjects = discover_subjects(
            erp_dir,
            conditions,
            align_to=align_to,
            source_method=source_method,
            parc=parc,
        )
    else:
        subjects = [normalize_subject_id(subject) for subject in subjects_list]

    if len(subjects) < 2:
        raise ValueError(f"At least two subjects are required for group statistics, got {len(subjects)}")

    print(f"=== Group statistics: {conditions[0]} vs {conditions[1]} ===")
    print(f"Subjects: {', '.join(subjects)}")

    cond1, feature_dims, feature_coords, cond1_inputs = _load_condition_matrix(
        erp_dir,
        subjects,
        conditions[0],
        align_to=align_to,
        source_method=source_method,
        parc=parc,
        runs=runs_by_condition.get(conditions[0]),
    )
    cond2, dims2, coords2, cond2_inputs = _load_condition_matrix(
        erp_dir,
        subjects,
        conditions[1],
        align_to=align_to,
        source_method=source_method,
        parc=parc,
        runs=runs_by_condition.get(conditions[1]),
    )
    if dims2 != feature_dims:
        raise ValueError(f"Condition feature dims differ: {feature_dims} != {dims2}")
    for key, value in feature_coords.items():
        if key in coords2 and not _coords_close(value, coords2[key]):
            raise ValueError(f"Condition coordinate '{key}' differs")

    contrast = cond1 - cond2
    flat = contrast.reshape(contrast.shape[0], -1)
    valid_features = np.all(np.isfinite(flat), axis=0)
    if not np.any(valid_features):
        raise ValueError("No finite subject-level contrast features are available for statistics")
    t_obs, p_values, h0 = compute_permutation_t_test(
        flat[:, valid_features],
        n_permutations=n_permutations,
        tail=tail,
        n_jobs=n_jobs,
    )
    feature_shape = contrast.shape[1:]
    t_full = np.full(flat.shape[1], np.nan, dtype=float)
    p_full = np.full(flat.shape[1], np.nan, dtype=float)
    t_full[valid_features] = np.asarray(t_obs)
    p_full[valid_features] = np.asarray(p_values)
    t_obs = t_full.reshape(feature_shape)
    p_values = p_full.reshape(feature_shape)
    h0 = np.asarray(h0)

    metadata = {
        "stage": "group_statistics",
        "kind": "paired_condition_contrast",
        "conditions": conditions,
        "contrast": f"{conditions[0]}-{conditions[1]}",
        "subjects": subjects,
        "alignment": align_to,
        "source_method": source_method,
        "parcellation": parc,
        "n_subjects": len(subjects),
        "n_permutations": int(n_permutations),
        "valid_feature_count": int(np.sum(valid_features)),
        "total_feature_count": int(flat.shape[1]),
        "tail": int(tail),
        "alpha": float(alpha),
        "condition_inputs": {
            conditions[0]: cond1_inputs,
            conditions[1]: cond2_inputs,
        },
    }
    coords = dict(feature_coords)

    paths = {
        "tstat": stats_derivative_path(
            out_dir,
            conditions=conditions,
            align_to=align_to,
            source_method=source_method,
            parc=parc,
            suffix="tstat",
            extension=".npy",
        ),
        "pvalue": stats_derivative_path(
            out_dir,
            conditions=conditions,
            align_to=align_to,
            source_method=source_method,
            parc=parc,
            suffix="pvalue",
            extension=".npy",
        ),
        "contrast": stats_derivative_path(
            out_dir,
            conditions=conditions,
            align_to=align_to,
            source_method=source_method,
            parc=parc,
            suffix="contrast",
            extension=".npy",
        ),
        "h0": stats_derivative_path(
            out_dir,
            conditions=conditions,
            align_to=align_to,
            source_method=source_method,
            parc=parc,
            suffix="h0",
            extension=".npy",
        ),
        "windows": stats_derivative_path(
            out_dir,
            conditions=conditions,
            align_to=align_to,
            source_method=source_method,
            parc=parc,
            suffix="sigwindows",
            extension=".tsv",
        ),
    }

    save_array(paths["tstat"], t_obs, dims=feature_dims, coords=coords, metadata={**metadata, "statistic": "t"})
    save_array(paths["pvalue"], p_values, dims=feature_dims, coords=coords, metadata={**metadata, "statistic": "p"})
    save_array(
        paths["contrast"],
        contrast,
        dims=("subject", *feature_dims),
        coords={"subject": subjects, **coords},
        metadata={**metadata, "statistic": "subject_contrast"},
    )
    save_array(
        paths["h0"],
        h0,
        dims=("permutation",),
        metadata={**metadata, "statistic": "max_statistic_null"},
    )
    save_table(
        paths["windows"],
        _window_table(p_values, feature_dims, coords, alpha=alpha),
        metadata={**metadata, "kind": "significant_time_windows"},
    )
    print(f"Saved group statistics to {paths['tstat'].parent}")
    return paths


def _condition_runs_from_args(conditions: Sequence[str], runs_cond1, runs_cond2) -> dict[str, Sequence[str]]:
    out = {}
    if runs_cond1:
        out[str(conditions[0])] = list(runs_cond1)
    if runs_cond2:
        out[str(conditions[1])] = list(runs_cond2)
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run group-level permutation statistics over ERP derivatives.")
    parser.add_argument("--erp_dir", type=str, required=True,
                        help="BIDS derivatives root containing Stage 5 ERP arrays.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="BIDS derivatives root for group statistics arrays.")
    parser.add_argument("--conditions", type=str, nargs=2, default=["Fast", "Slow"],
                        help="Two condition names to contrast.")
    parser.add_argument("--subjects", type=str, nargs="+", default=None,
                        help="Specific subjects to include. If omitted, subjects are discovered from ERP derivatives.")
    parser.add_argument("--align_to", type=str, default="go", choices=["go", "enter", "feedback"])
    parser.add_argument("--source_method", type=str, default="dSPM")
    parser.add_argument("--parc", type=str, default="HCPMMP1")
    parser.add_argument("--runs_cond1", type=str, nargs="+", default=None,
                        help="Optional run labels for the first condition.")
    parser.add_argument("--runs_cond2", type=str, nargs="+", default=None,
                        help="Optional run labels for the second condition.")
    parser.add_argument("--perms", type=int, default=1000,
                        help="Number of permutations.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--tail", type=int, default=0, choices=[-1, 0, 1])
    parser.add_argument("--n_jobs", type=int, default=1)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_group_statistics_contrast(
        erp_dir=args.erp_dir,
        out_dir=args.out_dir,
        conditions=args.conditions,
        n_permutations=args.perms,
        subjects_list=args.subjects,
        align_to=args.align_to,
        source_method=args.source_method,
        parc=args.parc,
        runs_by_condition=_condition_runs_from_args(args.conditions, args.runs_cond1, args.runs_cond2),
        alpha=args.alpha,
        tail=args.tail,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
