"""Stage 11 Hilbert feature extraction for PAC/CFC-ready derivatives."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from meg_tokens.io import derivative_path, ensure_dir, save_array
from meg_tokens.meg.connectivity import infer_sfreq_from_times
from meg_tokens.meg.time_frequency import DEFAULT_BANDS, compute_hilbert_band_features
from meg_tokens.utils.batch_decoding import _load_feature_array, find_feature_arrays
from meg_tokens.utils.batch_processor import normalize_subject_id
from meg_tokens.utils.batch_time_frequency import parse_frequency_bands
from meg_tokens.utils.epochs_builder import parse_run_label


DEFAULT_FEATURES = ("amplitude", "power", "phase", "sigfilt")


def hilbert_feature_derivative_path(
    output_root: str,
    *,
    subject: str,
    run: str,
    condition: str,
    align_to: str,
    source_method: str,
    parc: str,
    band: str,
    feature: str,
) -> Path:
    """Build a Stage 11 Hilbert-feature derivative path."""
    return derivative_path(
        output_root,
        subject=normalize_subject_id(subject),
        datatype="meg",
        task="tokens",
        run=parse_run_label(run)[0],
        description="-".join([
            condition.lower(),
            align_to,
            source_method,
            parc,
            band.replace("_", "-"),
            feature,
        ]),
        suffix="hilbertfeature",
        extension=".npy",
    )


def _stage_metadata(metadata: Mapping[str, object]) -> Mapping[str, object]:
    stage_meta = metadata.get("metadata", {})
    return stage_meta if isinstance(stage_meta, Mapping) else {}


def _metadata_value(metadata: Mapping[str, object], key: str, fallback):
    value = _stage_metadata(metadata).get(key)
    return fallback if value is None else value


def _coord_values(metadata: Mapping[str, object], key: str, length: int) -> list[object]:
    coords = metadata.get("coords", {})
    if not isinstance(coords, Mapping):
        return list(range(length))
    values = coords.get(key)
    if values is None:
        if key == "trial":
            return list(range(1, length + 1))
        return list(range(length))
    return list(values)


def _canonical_features(features: Sequence[str]) -> tuple[str, ...]:
    canonical = tuple("sigfilt" if item == "filtered" else item for item in features)
    valid = set(DEFAULT_FEATURES)
    unknown = sorted(set(canonical) - valid)
    if unknown:
        raise ValueError(f"Unknown Hilbert feature(s): {unknown}. Valid features: {sorted(valid)}")
    return canonical


def run_batch_hilbert_features(
    feature_dir: str,
    output_dir: str,
    subjects: Sequence[str],
    conditions: Sequence[str],
    *,
    align_to: str = "go",
    source_method: str = "dSPM",
    parc: str = "HCPMMP1",
    labels: Optional[Sequence[str]] = None,
    runs_by_condition: Optional[Mapping[str, Sequence[str]]] = None,
    freq_bands: Optional[Mapping[str, tuple[float, float]]] = None,
    features: Sequence[str] = DEFAULT_FEATURES,
    sfreq: Optional[float] = None,
    n_jobs: int = 1,
) -> dict[str, list[Path]]:
    """Extract band-filtered signal, amplitude, power, and phase from ERP tensors."""
    ensure_dir(output_dir)
    requested = _canonical_features(features)
    bands = dict(freq_bands or DEFAULT_BANDS)
    runs_by_condition = runs_by_condition or {}
    outputs: dict[str, list[Path]] = {}

    for subject in subjects:
        subject = normalize_subject_id(subject)
        outputs[subject] = []
        for condition in conditions:
            paths = find_feature_arrays(
                feature_dir,
                subject,
                condition,
                feature_source="erp",
                align_to=align_to,
                source_method=source_method,
                parc=parc,
                runs=runs_by_condition.get(condition),
            )
            for path in paths:
                data, times, feature_names, metadata = _load_feature_array(path, labels=labels, lateralize=False)
                sfreq_hz = float(sfreq) if sfreq is not None else infer_sfreq_from_times(times)
                features_by_band = compute_hilbert_band_features(
                    data,
                    sfreq=sfreq_hz,
                    freq_bands=bands,
                    features=requested,
                    n_jobs=n_jobs,
                )

                run = str(_metadata_value(metadata, "run", "1"))
                condition_meta = str(_metadata_value(metadata, "condition", condition))
                trial_coords = _coord_values(metadata, "trial", data.shape[0])

                for band, band_features in features_by_band.items():
                    bounds = bands[band]
                    for feature, values in band_features.items():
                        out_path = hilbert_feature_derivative_path(
                            output_dir,
                            subject=subject,
                            run=run,
                            condition=condition_meta,
                            align_to=align_to,
                            source_method=source_method,
                            parc=parc,
                            band=band,
                            feature=feature,
                        )
                        saved = save_array(
                            out_path,
                            values,
                            dims=("trial", "feature", "time"),
                            coords={
                                "trial": trial_coords,
                                "feature": feature_names,
                                "time_sec": times,
                            },
                            metadata={
                                "stage": "hilbert_features",
                                "kind": "erp_hilbert_feature",
                                "subject": subject,
                                "run": parse_run_label(run)[0],
                                "condition": condition_meta,
                                "alignment": align_to,
                                "source_method": source_method,
                                "parcellation": parc,
                                "selected_labels": list(labels) if labels else None,
                                "band": band,
                                "fmin_hz": float(bounds[0]),
                                "fmax_hz": float(bounds[1]),
                                "feature": feature,
                                "sfreq_hz": sfreq_hz,
                                "input_feature": str(path),
                            },
                        )
                        outputs[subject].append(saved)
                        print(f"Saved {feature} {band} Hilbert feature: {saved}")

    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract Hilbert phase/amplitude features from Stage 5 ERP derivatives."
    )
    parser.add_argument("--feature_dir", type=str, required=True,
                        help="BIDS derivatives root containing Stage 5 ERP arrays.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="BIDS derivatives root for Hilbert-feature outputs.")
    parser.add_argument("--subjects", type=str, nargs="+", required=True,
                        help="Subject IDs to process.")
    parser.add_argument("--conditions", type=str, nargs="+", required=True,
                        help="Condition labels to process.")
    parser.add_argument("--align_to", type=str, default="go", choices=["go", "enter", "feedback"])
    parser.add_argument("--source_method", type=str, default="dSPM")
    parser.add_argument("--parc", type=str, default="HCPMMP1")
    parser.add_argument("--labels", type=str, nargs="+", default=None,
                        help="Optional labels/ROIs to select before feature extraction.")
    parser.add_argument("--bands", type=str, nargs="+", required=True,
                        help="Bands to compute, using known names or name=fmin,fmax.")
    parser.add_argument("--features", type=str, nargs="+", default=list(DEFAULT_FEATURES),
                        choices=["amplitude", "power", "phase", "sigfilt", "filtered"],
                        help="Hilbert outputs to save.")
    parser.add_argument("--sfreq", type=float, default=None,
                        help="Override sample rate in Hz. By default this is inferred from time_sec.")
    parser.add_argument("--n_jobs", type=int, default=1)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    bands = parse_frequency_bands(args.bands)
    run_batch_hilbert_features(
        feature_dir=args.feature_dir,
        output_dir=args.out_dir,
        subjects=args.subjects,
        conditions=args.conditions,
        align_to=args.align_to,
        source_method=args.source_method,
        parc=args.parc,
        labels=args.labels,
        freq_bands=bands,
        features=args.features,
        sfreq=args.sfreq,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
