"""Lateralized ROI decoding wrapper around the Stage 8 decoder."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from meg_tokens.utils.batch_decoding import run_batch_decoding


def run_batch_decoding_lateralized(
    feature_dir: str,
    out_dir: str,
    conditions: Sequence[str],
    *,
    subjects: Optional[Sequence[str]] = None,
    align_to: str = "go",
    source_method: str = "dSPM",
    parc: str = "HCPMMP1",
    class_column: Optional[str] = None,
    class_values: Optional[Sequence[str]] = None,
    input_conditions: Optional[Sequence[str]] = None,
    permutations: int = 0,
    n_jobs: int = 4,
):
    return run_batch_decoding(
        feature_dir=feature_dir,
        output_dir=out_dir,
        conditions=conditions,
        feature_source="erp",
        subjects=subjects,
        input_conditions=input_conditions,
        align_to=align_to,
        source_method=source_method,
        parc=parc,
        lateralize=True,
        class_column=class_column,
        class_values=class_values,
        permutations=permutations,
        n_jobs=n_jobs,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run lateralized ROI decoding from Stage 6 ERP derivatives.")
    parser.add_argument("--feature_dir", "--data_dir", dest="feature_dir", type=str, required=True,
                        help="BIDS derivatives root containing Stage 6 ERP arrays.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="BIDS derivatives root for decoding outputs.")
    parser.add_argument("--conditions", type=str, nargs="+", default=["Fast", "Slow"])
    parser.add_argument("--subjects", type=str, nargs="+", default=None)
    parser.add_argument("--align_to", "--alignment", dest="align_to", type=str, default="go",
                        choices=["go", "enter", "feedback"])
    parser.add_argument("--source_method", type=str, default="dSPM")
    parser.add_argument("--parc", "--parcellation", dest="parc", type=str, default="HCPMMP1")
    parser.add_argument("--class_column", type=str, default=None)
    parser.add_argument("--class_values", type=str, nargs="+", default=None)
    parser.add_argument("--input_conditions", type=str, nargs="+", default=None)
    parser.add_argument("--permutations", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=4)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_batch_decoding_lateralized(
        feature_dir=args.feature_dir,
        out_dir=args.out_dir,
        conditions=args.conditions,
        subjects=args.subjects,
        align_to=args.align_to,
        source_method=args.source_method,
        parc=args.parc,
        class_column=args.class_column,
        class_values=args.class_values,
        input_conditions=args.input_conditions,
        permutations=args.permutations,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
