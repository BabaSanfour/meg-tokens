"""Validate the inter-trial-interval fingerprint matcher against real recordings.

The matching itself lives in ``meg_tokens.meg.raw_staging``
(``fingerprint_candidates`` / ``resolve_by_fingerprint``) and runs as part of
``meg-tokens meg stage-raw``. This script is the validation harness for it:
it scores every candidate for every run and prints the full ranking, so the
separation between the correct match and the runner-up is visible rather than
merely asserted.

Method: a behavioral run logs each trial's ``nInitialTime``; the raw recording
emits a trial-start pulse per trial. The two clocks are unrelated, but the gaps
between consecutive trials are the same physical intervals measured twice, so
the correct pairing reproduces them to well under a millisecond while any other
recording of the same protocol is off by hundreds. Candidates are pre-filtered
by nominal trial duration (315s Slow/Fast, 135s RT) so a run is only ever
scored against sessions recorded under its own protocol.

Validation run against the real dataset (2026-08-07): all 17
KNOWN_SESSION_OVERRIDES entries (8 for H01, 9 for H05) are reproduced from the
recordings alone, at 0.39-0.57ms mean error and 144-892x separation from the
next-best candidate wherever more than one candidate existed -- confirming
those two subjects' mappings independently of the legacy notebook they cite.

Usage:
    # Cross-check every hardcoded KNOWN_SESSION_OVERRIDES entry:
    python scripts/qc/meg_tdms_fingerprint_matching.py \\
        <raw-root> <tdms-root>

    # Score a subject's runs whether or not it has overrides:
    python scripts/qc/meg_tdms_fingerprint_matching.py \\
        <raw-root> <tdms-root> \\
        --subjects H02 H12 --show-all
"""

import argparse
from pathlib import Path

import mne

from meg_tokens.core import RawStagingConfig
from meg_tokens.meg.raw_staging import (
    KNOWN_SESSION_OVERRIDES,
    discover_raw_sessions,
    fingerprint_candidates,
    load_behavior_runs,
    read_start_pulse_times,
)

mne.set_log_level("ERROR")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("tdms_root", type=Path)
    parser.add_argument(
        "--subjects", nargs="+",
        help="Subjects to score. Defaults to every subject with a KNOWN_SESSION_OVERRIDES entry.",
    )
    parser.add_argument(
        "--show-all", action="store_true",
        help="Print every candidate's score, not just the winner and runner-up.",
    )
    args = parser.parse_args()

    settings = RawStagingConfig()
    subjects = args.subjects or sorted({subject for subject, _, _ in KNOWN_SESSION_OVERRIDES})
    disagreements = 0

    for subject in subjects:
        print(f"\n===== {subject} =====")
        sessions = discover_raw_sessions(args.raw_root, subject, settings=settings)
        runs = load_behavior_runs(args.tdms_root, subject)
        if not runs:
            print("(no TDMS runs found)")
            continue

        # One read per session, reused across every run scored against it.
        pulse_times = {
            session.path: read_start_pulse_times(session.path, subject)
            for session in sessions
            if session.duration_class != "OTHER"
        }

        for run in sorted(runs, key=lambda r: (r.condition, r.run)):
            candidates = [s for s in sessions if s.duration_class == run.duration_class]
            if not candidates:
                print(f"{run.condition}{run.run}: no {run.duration_class}-duration candidates")
                continue

            scores = fingerprint_candidates(run, candidates, pulse_times)
            best = scores[0]
            separation = (
                scores[1].mean_abs_error_ms / best.mean_abs_error_ms
                if len(scores) > 1 and best.mean_abs_error_ms > 0
                else float("inf")
            )
            accepted = (
                best.mean_abs_error_ms <= settings.fingerprint_max_error_ms
                and separation >= settings.fingerprint_min_separation
            )

            expected = KNOWN_SESSION_OVERRIDES.get((subject, run.condition, run.run))
            if expected is None:
                verdict = "accepted" if accepted else "REJECTED (below threshold)"
            elif best.session.index == expected:
                verdict = f"OK (agrees with override {expected:02d})"
            else:
                verdict = f"DISAGREES with override {expected:02d} !!"
                disagreements += 1

            sep_text = "n/a (1 candidate)" if separation == float("inf") else f"{separation:.1f}x"
            print(
                f"{run.condition}{run.run}: trials={run.trial_count} "
                f"winner={best.session.index:02d} err={best.mean_abs_error_ms:.3f}ms "
                f"offset={best.window_offset} sep={sep_text} [{verdict}]"
            )
            if args.show_all:
                for score in scores[1:]:
                    print(f"      {score.session.index:02d}: {score.mean_abs_error_ms:.3f}ms")

    print(f"\n{disagreements} disagreement(s) with KNOWN_SESSION_OVERRIDES.")
    return 1 if disagreements else 0


if __name__ == "__main__":
    raise SystemExit(main())
