"""Run golden-reference validation for real staged derivatives."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from meg_tokens.validation import run_golden_validation


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare modern staged derivatives against frozen real-reference outputs."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="JSON file listing array/table comparisons.")
    parser.add_argument("--out_tsv", type=str, required=True,
                        help="Validation report TSV path.")
    parser.add_argument("--allow_failures", action="store_true",
                        help="Write the report and return success even when comparisons fail.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    report = run_golden_validation(args.config, out_tsv=args.out_tsv)
    failed = report[report["status"] != "pass"]
    print(f"Golden validation report: {Path(args.out_tsv)}")
    print(f"Comparisons: {len(report)}; failed: {len(failed)}")
    if len(failed) and not args.allow_failures:
        names = ", ".join(failed["name"].astype(str).tolist())
        raise SystemExit(f"Golden validation failed for: {names}")


if __name__ == "__main__":
    main()
