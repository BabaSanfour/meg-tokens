"""Head motion and truncation QC from CTF acquisition metadata.

Reproduces the movement and truncation columns in
docs/meg_t0_6_subject_exclusion_qc.md ("MEG-signal evidence"):

  - per-subject head-motion z-scores over Slow/Fast runs (H10 z=+3.50)
  - the single truncated recording in the dataset
    (H05_DDM-tthiery_20180220_02.ds, 138s vs 315s expected)

Reads only the tiny .hist/.res4 metadata, never the .meg4 sample data, so a
full pass over ~530 sessions takes a couple of minutes and leaves the raw
data untouched.

    python scripts/qc/meg_session_qc.py <raw-root>
"""

import argparse
import re
from pathlib import Path

import numpy as np

SLOWFAST_TRIAL_DURATION = 315.0


def read_session_metadata(ds_dir):
    """Return acquisition metadata for one .ds directory, or None."""
    hist_files = list(ds_dir.glob("*.hist"))
    if not hist_files:
        return None
    text = hist_files[0].read_text(errors="replace")

    def field(pattern, cast=str):
        match = re.search(pattern, text)
        return cast(match.group(1).strip()) if match else None

    subject_match = re.match(r"^H(\d+)_", ds_dir.name)
    if subject_match is None:
        return None
    subject = f"H{int(subject_match.group(1)):02d}"

    return {
        "subject": subject,
        "ds": ds_dir.name,
        "trial_duration": field(r"Trial duration:\s*(\S+)", float),
        "max_motion_mm": field(r"Maximum head motion recorded is\s*([\d.]+)\(mm\)", float),
    }


def measured_duration(ds_dir):
    """Actual recorded duration in seconds, from the MNE header only."""
    import mne

    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_ctf(str(ds_dir), preload=False, verbose=False)
    return raw.n_times / raw.info["sfreq"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path, help="Directory holding the *.ds sessions")
    parser.add_argument(
        "--check-durations",
        action="store_true",
        help="Also open each header to detect truncated recordings (slower)",
    )
    args = parser.parse_args()

    sessions = [
        meta
        for ds_dir in sorted(args.data_root.glob("*.ds"))
        if (meta := read_session_metadata(ds_dir)) is not None
    ]
    if not sessions:
        raise SystemExit(f"No subject .ds sessions found under {args.data_root}")

    slowfast = [s for s in sessions if s["trial_duration"] == SLOWFAST_TRIAL_DURATION]

    # Head motion, Slow/Fast only: resting-state runs are excluded because only
    # Slow/Fast trials feed the behavioural analysis.
    by_subject = {}
    for session in slowfast:
        by_subject.setdefault(session["subject"], []).append(session["max_motion_mm"])
    per_subject_max = {
        subject: max(v for v in values if v is not None)
        for subject, values in by_subject.items()
    }
    values = np.array(list(per_subject_max.values()))
    mean, sd = values.mean(), values.std(ddof=1)

    print(f"Head motion over Slow/Fast runs, {len(per_subject_max)} subjects")
    print(f"  population mean={mean:.2f}mm sd={sd:.2f}mm\n")
    print(f"  {'subject':<9}{'max mm':>9}{'z':>8}")
    for subject, value in sorted(per_subject_max.items(), key=lambda kv: -kv[1]):
        print(f"  {subject:<9}{value:>9.2f}{(value - mean) / sd:>8.2f}")

    if args.check_durations:
        print("\nTruncated recordings (measured duration < 99% of configured):")
        found = False
        for session in sessions:
            expected = session["trial_duration"]
            if not expected:
                continue
            actual = measured_duration(args.data_root / session["ds"])
            if actual < 0.99 * expected:
                found = True
                print(
                    f"  {session['ds']}: {actual:.1f}s actual vs {expected:.0f}s expected"
                )
        if not found:
            print("  none")


if __name__ == "__main__":
    main()
