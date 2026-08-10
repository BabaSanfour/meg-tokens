"""Trial-class reference frames: ours vs Thomas's, on the real TDMS data.

Reproduces every number in docs/behavior.md's Findings section
(trial-classification reference frame):

  1. group DT statistics under three classification choices, showing that the
     chosen-target frame flips Ambiguous-vs-Misleading to the published sign
  2. the decomposition of the chosen frame's "misleading" class
     (46.3% genuinely misleading stimuli, 53.7% subject errors on clear
     evidence) -- the confound that makes us keep the design frame
  3. the token counts proving zero inferred-misleading is a property of the
     stimulus set, not a bug: across the random pool the correct target never
     trails after three jumps

"Ours" applies the SP thresholds to a design-derived correct-target profile
for random trials only; "Thomas" applies them to the runtime nProb
(chosen-target) profile and overwrites every trial, per
archive/replicated/DDM_scripts/scripts_new/Modify_df_preproc.ipynb.

    python scripts/qc/trial_class_frames.py <tdms-root>
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import sem

from meg_tokens.behavior.tdms import (
    OUTCOME_NEVER_STARTED,
    add_run_metadata,
    parse_tdms_file,
    parse_tdms_filename,
)

PREPRINT_EXCLUSIONS = {"H06", "H07", "H10", "H20"}
CLASS_NAMES = {1: "Easy", 2: "Ambiguous", 3: "Misleading"}


def thomas_class(nprob):
    """Rule from Modify_df_preproc.ipynb, applied to the runtime nProb array.

    Indices 1/2/4/7 are the success probabilities after 2/3/5/8 token jumps.
    """
    if not isinstance(nprob, list) or len(nprob) < 8:
        return 0
    if nprob[1] > 0.6 and nprob[4] > 0.75 and nprob[7] > 0.75:
        return 1
    if nprob[1] == 0.5 and 0.38 < nprob[2] < 0.65 and 0.35 < nprob[4] < 0.65:
        return 2
    if nprob[2] < 0.4:
        return 3
    return 0


def load_subject(subject_dir):
    """Parsed runs for one subject, keyed by condition."""
    runs = {}
    for path in sorted(subject_dir.glob("*.tdms")):
        if path.name.startswith("temp_"):  # documented scratch recordings
            continue
        info = parse_tdms_filename(path.name)
        table = add_run_metadata(parse_tdms_file(str(path)), info, path.name)
        runs.setdefault(info.condition, []).append((int(info.run), table))
    return {
        condition: [table for _, table in sorted(entries)]
        for condition, entries in runs.items()
    }


def report(label, per_subject):
    print(f"\n=== {label} (N={len(per_subject[1])}) ===")
    for trial_class in (1, 2, 3):
        values = np.array(per_subject[trial_class])
        print(f"  {CLASS_NAMES[trial_class]:<11}{values.mean():>8.1f} +/- {sem(values):.2f}")
    for left, right, name in ((1, 2, "Easy vs Amb"), (1, 3, "Easy vs Mis"), (2, 3, "Amb vs Mis")):
        t, p = stats.ttest_rel(per_subject[left], per_subject[right])
        print(f"  {name:<13}t={t:>7.2f}  p={p:.4g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tdms_root", type=Path)
    args = parser.parse_args()

    variants = {
        "ours: design frame, 'x' only": {1: [], 2: [], 3: []},
        "recorded labels only": {1: [], 2: [], 3: []},
        "Thomas: chosen frame, all trials": {1: [], 2: [], 3: []},
    }
    genuine_misleading = subject_errors = 0
    token_split = {}

    for subject_dir in sorted(p for p in args.tdms_root.iterdir() if p.is_dir()):
        subject = subject_dir.name
        runs = load_subject(subject_dir)

        started = [
            table.loc[table["nOutcome"] != OUTCOME_NEVER_STARTED]
            for table in runs.get("RT", [])
        ]
        baseline = float(np.mean([t["rawRT"].dropna().mean() for t in started]))

        per_variant = {name: {1: [], 2: [], 3: []} for name in variants}
        for table in runs.get("Fast", []) + runs.get("Slow", []):
            valid = table.loc[
                (table["nOutcome"] != OUTCOME_NEVER_STARTED) & table["rawRT"].notna()
            ]
            for row in valid.itertuples():
                decision_time = float(row.rawRT) - baseline
                recorded = str(row.sTrialClassRaw).lower()
                profile = row.sp_design_correct

                if row.sTrialClass in (1, 2, 3):
                    per_variant["ours: design frame, 'x' only"][row.sTrialClass].append(decision_time)
                if recorded in ("e", "a", "m") and row.sTrialClass in (1, 2, 3):
                    per_variant["recorded labels only"][row.sTrialClass].append(decision_time)
                chosen = thomas_class(row.nProb)
                if chosen in (1, 2, 3):
                    per_variant["Thomas: chosen frame, all trials"][chosen].append(decision_time)

                # Why the chosen frame's misleading class is contaminated: when
                # the subject errs, SP_chosen = 1 - SP_correct, so a clear trial
                # answered wrongly looks identical to a misleading stimulus.
                if chosen == 3 and isinstance(profile, list):
                    if profile[2] < 0.4:
                        genuine_misleading += 1
                    else:
                        subject_errors += 1

                # Tokens toward the correct target in the first three jumps.
                if recorded in ("e", "a", "m", "x"):
                    first_three = str(row.sTokenDirs)[:3]
                    toward = sum(1 for d in first_three if str(d) == str(row.nCorrectChoice))
                    token_split.setdefault(recorded, Counter())[f"{toward}-{3 - toward}"] += 1

        if subject in PREPRINT_EXCLUSIONS:
            continue
        for name, classes in per_variant.items():
            if all(classes[c] for c in (1, 2, 3)):
                for c in (1, 2, 3):
                    variants[name][c].append(np.mean(classes[c]))

    for name, per_subject in variants.items():
        report(name, per_subject)
    print("\n=== PREPRINT ===")
    print("  Easy 1028 +/- 59, Ambiguous 1405 +/- 74, Misleading 1433 +/- 79")
    print("  Easy vs Amb t=-15.04 | Easy vs Mis t=-13.10 | Amb vs Mis t=-1.84 (p=0.077)")

    total = genuine_misleading + subject_errors
    print("\n=== chosen-frame 'misleading' composition ===")
    print(f"  genuinely misleading stimulus : {genuine_misleading:5d} ({100 * genuine_misleading / total:.1f}%)")
    print(f"  clear evidence, subject erred : {subject_errors:5d} ({100 * subject_errors / total:.1f}%)")

    print("\n=== tokens toward correct target, first 3 jumps ===")
    for label in sorted(token_split):
        counts = dict(sorted(token_split[label].items()))
        print(f"  '{label}' n={sum(counts.values()):5d}: {counts}")
    print("  ('x' never trails => no random trial can be misleading by the SP rule)")


if __name__ == "__main__":
    main()
