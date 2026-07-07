"""Stage 2 preprocessing command for CTF raw data."""

import argparse

from meg_tokens.meg.preprocessing import load_and_filter_raw, run_ica_rejection, save_clean_raw


def run_preprocess_pipeline(
    raw_path: str,
    output_root: str,
    subject: str,
    run: str,
    condition: str = None,
    high_pass: float = 0.5,
    low_pass: float = 150.0,
    run_ica: bool = False,
    ica_exclude: list = None,
) -> str:
    """Filter one raw CTF run and save it as a Stage 2 raw derivative."""
    raw = load_and_filter_raw(
        raw_path,
        high_pass=high_pass,
        low_pass=low_pass,
        preload=True,
    )

    if run_ica:
        ica = run_ica_rejection(raw)
        if ica_exclude is not None:
            ica.exclude = [int(idx) for idx in ica_exclude]
        ica.apply(raw)

    return save_clean_raw(
        raw,
        output_root,
        subject_id=subject,
        run_id=run,
        condition=condition,
        processing="clean" if run_ica else "filt",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter one CTF raw run and save a Stage 2 raw derivative.")
    parser.add_argument("--raw_path", required=True, help="Path to the raw CTF .ds dataset.")
    parser.add_argument("--out_dir", required=True, help="BIDS derivatives root.")
    parser.add_argument("--subject", required=True, help="Subject ID, e.g. H01.")
    parser.add_argument("--run", required=True, help="Run label, e.g. 1 or Slow1.")
    parser.add_argument("--condition", default=None, help="Optional condition label, e.g. Slow.")
    parser.add_argument("--high_pass", type=float, default=0.5)
    parser.add_argument("--low_pass", type=float, default=150.0)
    parser.add_argument("--run_ica", action="store_true", help="Fit ICA and apply detected artifact components.")
    parser.add_argument("--ica_exclude", type=int, nargs="*", default=None,
                        help="Optional explicit ICA component indices to exclude after fitting.")
    args = parser.parse_args()

    out = run_preprocess_pipeline(
        raw_path=args.raw_path,
        output_root=args.out_dir,
        subject=args.subject,
        run=args.run,
        condition=args.condition,
        high_pass=args.high_pass,
        low_pass=args.low_pass,
        run_ica=args.run_ica,
        ica_exclude=args.ica_exclude,
    )
    print(f"Saved preprocessed raw derivative: {out}")


if __name__ == "__main__":
    main()
