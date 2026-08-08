"""Per-trial pulse audit: does every trial's MEG trigger structure match its log?

Validates, against the staged BIDS raw layer, the three structural claims the
epoching stage relies on but never re-checks at runtime:

1. **OUTCOME_NEVER_STARTED (7003) trials still emit a trial-start pulse.**
   ``meg_tokens.meg.raw_staging.BehaviorRun.trial_count`` compares a session's
   Start-pulse count against *every* logged row, 7003 included, and
   ``reconstruct_missing_go_events`` pairs behavior row *i* positionally with
   Start pulse *i* -- both are wrong if a 7003 trial produces no pulse.
2. **7003 trials never emit a go-cue.** ``synchronize_events_and_behavior``
   drops them from the alignment entirely (``started_trials``), which is only
   correct if they have no Go event to align to.
3. **Every started trial has exactly one go-cue, at its logged tGO latency.**
   Interior violations are the genuine trigger dropouts enumerated in
   ``KNOWN_GO_RECONSTRUCTION_RUNS``; this script reports any run that violates
   it *without* being on that list.

Trials are paired positionally (behavior row i <-> Start pulse i) and a trial's
Go window is bounded by the next Start pulse, so the check never assumes the
counts already agree. Reads only the staged ``*_events.tsv``/``*_beh.tsv``, so
a full pass is seconds rather than minutes and needs no access to the raw
acquisition tree.

    python scripts/qc/meg_trial_pulse_qc.py <bids-root>
    python scripts/qc/meg_trial_pulse_qc.py <bids-root> --subjects H02 --verbose
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from meg_tokens.behavior.schema import OUTCOME_NEVER_STARTED
from meg_tokens.meg.epoching import (
    KNOWN_GO_RECONSTRUCTION_RUNS,
    KNOWN_TRAILING_TRIAL_MISMATCHES,
    KNOWN_UNRECOVERABLE_TRIALS,
    exclude_unrecoverable_trials,
    get_event_id,
    mismatch_policy,
    needs_go_reconstruction,
    reconstruct_missing_go_events,
    synchronize_events_and_behavior,
)

_ENTITY_RE = re.compile(
    r"^sub-(?P<subject>[^_]+)_task-[^_]+_acq-(?P<acq>[^_]+)_run-(?P<run>[0-9]+)_events\.tsv$"
)
_ACQ_TO_CONDITION = {"slow": "Slow", "fast": "Fast", "rt": "RT"}

# The real start->go hardware lag is a few ms (see DEFAULT_GO_LAG_S); an
# off-by-one trial pairing shifts the residual by a whole inter-trial interval
# (hundreds of ms), so anything above this threshold is a pairing error, not jitter.
MAX_ALIGNMENT_RESIDUAL_MS = 50.0


def load_run(events_path, beh_path):
    """Start/Go onsets (s) and the run's behavior rows in logged trial order."""
    events = pd.read_csv(events_path, sep="\t", encoding="utf-8-sig")
    beh = pd.read_csv(beh_path, sep="\t", encoding="utf-8-sig")
    starts = np.sort(events.loc[events["trial_type"] == "Start", "onset"].to_numpy(float))
    gos = np.sort(events.loc[events["trial_type"] == "Go", "onset"].to_numpy(float))
    return starts, gos, beh.sort_values("nTrialIndex").reset_index(drop=True)


def load_events_array(events_path):
    """The run's events as MNE's (sample, 0, code) array, plus its sample rate."""
    events = pd.read_csv(events_path, sep="\t", encoding="utf-8-sig")
    array = np.column_stack([
        events["sample"].to_numpy(int),
        np.zeros(len(events), dtype=int),
        events["value"].to_numpy(int),
    ])
    array = array[array[:, 0].argsort()]
    nonzero = events.loc[events["onset"] > 0]
    sfreq = float((nonzero["sample"] / nonzero["onset"]).median())
    return array, sfreq


def simulate_alignment(subject, condition, run, events, sfreq, beh):
    """Replay the epoching stage's own alignment decisions for one run.

    Calls the real ``meg_tokens.meg.epoching`` helpers in the same order
    ``meg_tokens.workflows.preprocessing`` does, then independently re-derives
    each retained epoch's start->go latency and compares it against that
    trial's logged ``tGO``. A correct alignment leaves the same few-ms hardware
    lag on every epoch; an off-by-one pairing leaves residuals of hundreds of
    ms, so this catches a silently mis-paired truncation, not just a crash.
    """
    event_id = get_event_id("go", subject)
    if needs_go_reconstruction(subject, condition, run):
        events = reconstruct_missing_go_events(
            events, beh, sfreq,
            start_code=get_event_id("start", subject)["Start"],
            go_code=event_id["Go"],
        )
    trials = exclude_unrecoverable_trials(subject, condition, run, beh)
    policy = mismatch_policy(subject, condition, run)
    try:
        final_events, metadata = synchronize_events_and_behavior(
            events, event_id, trials, on_mismatch=policy
        )
    except ValueError as error:
        return {"ok": False, "policy": policy, "error": str(error)}

    starts = np.sort(events[events[:, 2] == get_event_id("start", subject)["Start"]][:, 0])
    go_samples = final_events[:, 0]
    preceding = np.searchsorted(starts, go_samples, side="right") - 1
    latency_ms = (go_samples - starts[preceding]) / sfreq * 1000.0
    residuals = latency_ms - metadata["tGO"].to_numpy(float)
    return {
        "ok": True, "policy": policy, "n_epochs": len(metadata),
        "max_residual_ms": float(np.abs(residuals).max()),
    }


def audit_run(starts, gos, beh):
    """Per-trial verdicts for one run.

    Each behavior row i is paired with Start pulse i and its Go window runs to
    the next Start pulse (or the end of the recording), so a trial is judged on
    its own interval rather than on whole-run counts.
    """
    rows = []
    for i, trial in beh.iterrows():
        never_started = int(trial["nOutcome"]) == OUTCOME_NEVER_STARTED
        if i >= len(starts):
            rows.append({
                "trial": int(trial["nTrialIndex"]), "never_started": never_started,
                "has_start": False, "n_go": 0, "go_residual_ms": np.nan,
            })
            continue
        window_end = starts[i + 1] if i + 1 < len(starts) else np.inf
        in_window = gos[(gos > starts[i]) & (gos < window_end)]
        residual = np.nan
        if len(in_window) == 1 and not never_started:
            residual = (in_window[0] - starts[i]) * 1000.0 - float(trial["tGO"])
        rows.append({
            "trial": int(trial["nTrialIndex"]), "never_started": never_started,
            "has_start": True, "n_go": len(in_window), "go_residual_ms": residual,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("bids_root", type=Path)
    parser.add_argument("--subjects", nargs="+", help="Default: every staged sub-H* subject.")
    parser.add_argument("--verbose", action="store_true", help="Print every run, not just problems.")
    args = parser.parse_args()

    subject_dirs = sorted(
        d for d in args.bids_root.glob("sub-H*")
        if d.is_dir() and (args.subjects is None or d.name[4:] in args.subjects)
    )
    if not subject_dirs:
        raise SystemExit(f"No staged subjects found under {args.bids_root}")

    audits, alignments, problems = [], [], []
    for subject_dir in subject_dirs:
        subject = subject_dir.name[4:]
        for events_path in sorted((subject_dir / "meg").glob("*_events.tsv")):
            entities = _ENTITY_RE.match(events_path.name)
            if entities is None:
                continue
            condition = _ACQ_TO_CONDITION[entities.group("acq")]
            run = entities.group("run")
            beh_path = subject_dir / "beh" / events_path.name.replace("_events.tsv", "_beh.tsv")
            if not beh_path.exists():
                problems.append(f"{subject} {condition}{run}: no staged behavior TSV")
                continue

            starts, gos, beh = load_run(events_path, beh_path)
            audit = audit_run(starts, gos, beh)
            key = (subject, condition, run)

            n_never = int(audit["never_started"].sum())
            never = audit[audit["never_started"]]
            # A trial with no Start pulse at all trivially has no go-cue; it is
            # reported as `no_start` and must not also count as a go dropout,
            # which is a distinct failure needing a distinct remedy.
            started = audit[~audit["never_started"] & audit["has_start"]]
            missing_start = int((~audit["has_start"]).sum())
            never_with_go = int((never["n_go"] > 0).sum())
            started_no_go = int((started["n_go"] == 0).sum())
            started_multi_go = int((started["n_go"] > 1).sum())
            residuals = started["go_residual_ms"].dropna()

            # Only a go dropout with a real go-cue *after* it is interior, and
            # so unfixable by truncation. A run whose recording stopped between
            # a trial's start pulse and its go cue leaves a no-go trial at the
            # tail, contiguous with the trials that got no pulse at all -- the
            # same trailing boundary KNOWN_TRAILING_TRIAL_MISMATCHES covers.
            with_go = audit.index[audit["n_go"] > 0]
            last_go_position = with_go.max() if len(with_go) else -1
            interior_no_go = int(
                ((started["n_go"] == 0) & (started.index < last_go_position)).sum()
            )

            if args.verbose or missing_start or never_with_go or started_no_go or started_multi_go:
                print(
                    f"{subject:>4} {condition}{run:<2} rows={len(beh):>3} starts={len(starts):>3} "
                    f"gos={len(gos):>3} never_started={n_never:>2} | "
                    f"no_start={missing_start} 7003_with_go={never_with_go} "
                    f"started_no_go={started_no_go} (interior={interior_no_go}) "
                    f"started_multi_go={started_multi_go} | "
                    f"tGO residual mean={residuals.mean():.2f}ms max={residuals.abs().max():.2f}ms"
                )

            # A trailing surplus/deficit of Start pulses is a documented,
            # separately-verified condition (KNOWN_TRAILING_TRIAL_MISMATCHES),
            # so only pulses missing *before* the last logged trial count here.
            if missing_start and key not in KNOWN_TRAILING_TRIAL_MISMATCHES:
                problems.append(
                    f"{subject} {condition}{run}: {missing_start} trial(s) with no Start pulse, "
                    "and this run is not a KNOWN_TRAILING_TRIAL_MISMATCHES case"
                )
            if never_with_go:
                problems.append(
                    f"{subject} {condition}{run}: {never_with_go} OUTCOME_NEVER_STARTED trial(s) "
                    "have a go-cue -- started_trials() would drop a real epoch"
                )
            if interior_no_go and key not in KNOWN_GO_RECONSTRUCTION_RUNS:
                excluded = len(KNOWN_UNRECOVERABLE_TRIALS.get(key, ()))
                if interior_no_go > excluded:
                    problems.append(
                        f"{subject} {condition}{run}: {interior_no_go} started trial(s) with an "
                        f"interior go-cue dropout ({excluded} covered by "
                        "KNOWN_UNRECOVERABLE_TRIALS), and this run is not in "
                        "KNOWN_GO_RECONSTRUCTION_RUNS"
                    )
            if (
                started_no_go
                and not interior_no_go
                and key not in KNOWN_TRAILING_TRIAL_MISMATCHES
                and key not in KNOWN_GO_RECONSTRUCTION_RUNS
            ):
                problems.append(
                    f"{subject} {condition}{run}: {started_no_go} trailing trial(s) with a Start "
                    "pulse but no go-cue, and this run is not a "
                    "KNOWN_TRAILING_TRIAL_MISMATCHES case"
                )
            if started_multi_go:
                problems.append(
                    f"{subject} {condition}{run}: {started_multi_go} trial(s) with >1 go-cue "
                    "between consecutive Start pulses"
                )
            audits.append(audit.assign(subject=subject, condition=condition, run=run))

            events_array, sfreq = load_events_array(events_path)
            outcome = simulate_alignment(subject, condition, run, events_array, sfreq, beh)
            if not outcome["ok"]:
                problems.append(
                    f"{subject} {condition}{run}: go-alignment fails under policy "
                    f"'{outcome['policy']}' -- {outcome['error']}"
                )
            elif outcome["max_residual_ms"] > MAX_ALIGNMENT_RESIDUAL_MS:
                problems.append(
                    f"{subject} {condition}{run}: go-alignment survives policy "
                    f"'{outcome['policy']}' but leaves a {outcome['max_residual_ms']:.0f}ms "
                    "start->go residual -- epochs are paired with the wrong trials"
                )
            alignments.append({
                "subject": subject, "condition": condition, "run": run,
                "policy": outcome["policy"], "ok": outcome["ok"],
                "n_epochs": outcome.get("n_epochs"),
                "max_residual_ms": outcome.get("max_residual_ms"),
            })

    combined = pd.concat(audits, ignore_index=True)
    never = combined[combined["never_started"]]
    started = combined[~combined["never_started"]]
    print(f"\n===== {len(subject_dirs)} subject(s), {len(audits)} run(s), {len(combined)} trials =====")
    print(f"OUTCOME_NEVER_STARTED trials:      {len(never)}")
    print(f"  with a Start pulse:              {int(never['has_start'].sum())}/{len(never)}")
    print(f"  with NO go-cue (as assumed):     {int((never['n_go'] == 0).sum())}/{len(never)}")
    print(f"Started trials:                    {len(started)}")
    print(f"  with exactly one go-cue:         {int((started['n_go'] == 1).sum())}/{len(started)}")
    residuals = started["go_residual_ms"].dropna()
    print(f"  go-cue latency vs logged tGO:    mean={residuals.mean():.2f}ms "
          f"sd={residuals.std():.2f}ms max|.|={residuals.abs().max():.2f}ms")

    alignment = pd.DataFrame(alignments)
    print(f"\nGo-alignment replay (the epoching stage's own decisions):")
    print(f"  runs that align successfully:    {int(alignment['ok'].sum())}/{len(alignment)}")
    print(f"  policy=truncate / error:         "
          f"{int((alignment['policy'] == 'truncate').sum())} / "
          f"{int((alignment['policy'] == 'error').sum())}")
    print(f"  epochs retained:                 {int(alignment['n_epochs'].sum())}")
    print(f"  worst start->go residual:        {alignment['max_residual_ms'].max():.2f}ms")

    if problems:
        print(f"\n!! {len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("\nNo undocumented violations.")


if __name__ == "__main__":
    main()
