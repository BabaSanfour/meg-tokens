# Behavior (TDMS) Pipeline: QC Findings and Fixes

This report documents data-quality issues found in the Stage 1 behavior ingestion pipeline (`meg_tokens/behavior/tdms.py`) and the MEG/behavior alignment step (`meg_tokens/meg/epoching.py`), the fixes applied, and the scientific reasoning and verification behind each decision. 

All findings below were verified against the real local dataset (323 `.tdms` files, 32 subjects) and cross-checked against the raw MEG recordings (`DDM-tthiery` CTF dataset, 399 `.ds` files).

## Background: what TDMS ingestion does

Each `.tdms` file is one behavioral run. `meg_tokens/behavior/tdms.py` parses
these into per-trial tables containing trial identity, raw and numeric class,
classification provenance, choices, outcomes, runtime token logs, the complete
design SP profile, and short-log QC flags. These become the Stage 1
BIDS-derivative `.tsv` behavior files used by every downstream analysis.

---

## 1. Silent data loss in the original parser (fixed)

The original implementation used `dict.get(key, default)` everywhere a field was read out of a trial's raw event log, and silently dropped or skipped malformed records. Concretely, before this work:

- **Unrecognized/renamed `.tdms` files were silently skipped.** Running the parser over the real dataset surfaced 3 scratch files that were being dropped with no warning: `H03/temp_180214.tdms`, `H18/temp_181024.tdms`, `H23/temp_181121.tdms`.
- **Missing structural fields in a trial's event log defaulted to `0`** (`nTrialIndex`, `nInitialTime`, `nCorrectChoice`, `tGO`, `tTrialEnd`, `sTokenDirs`) instead of raising — a truncated/corrupted log line would produce a fully-formed, plausible-looking (but fabricated) trial.
- **`sTrialClass == 'r'`** (a real label used on every RT-condition trial) was being silently treated the same as unrecognized/corrupted input.
- **Trials with a missing `Events` property, or files with zero valid trialgroups, produced silently incomplete or empty tables** instead of an error.
- **`nOutcome` was parsed but discarded** — never included in the output columns at all (see §4).

### Fix

Every one of these paths now raises a specific `ValueError` naming the file and trial at fault, instead of defaulting or dropping data. `sTrialClass`'s recognized vocabulary was corrected to include `'r'`. `nOutcome` is now a required output column. The distinction that was preserved: `nChoiceMade == 0` and `tEnterTarget` missing are **not** errors — they are the real, valid "subject did not respond" trial outcome (confirmed against `meg_tokens/behavior/metrics.py`, which explicitly filters on `nChoiceMade == 0` as a legitimate skipped-trial code).

Verified against all 323 real files: zero new false-positive errors introduced, and the 3 previously-hidden scratch files now surface instead of disappearing.

---

## 2. Scratch/non-run files on disk (resolved via config)

The 3 files named above (`temp_*.tdms`) are not real experimental runs — inspection showed they are short recording fragments (9–12 trials, a few seconds to ~1 minute long) sitting immediately before or after a real run for that subject on the same day (e.g. `H03/temp_180214.tdms` sits between `H03RT1` and `H03Fast1`). They are almost certainly aborted/test recordings, not data to analyze.

### Fix

Added an explicit `behavior_ignore_files` list to the project config (`config/tokens.toml.template`), with a comment recording why each file is excluded. This is an explicit allowlist, not a silent skip: any `.tdms` file that doesn't match the canonical naming pattern **and** isn't in this list still raises an error during ingestion.

---

## 3. `nTrialIndex` not starting at 1 — root cause and Stage 1 fix

### The symptom

`validate_behavior_dataframe` originally required each run's `nTrialIndex` column to be exactly `[1, 2, ..., N]`. Running the parser over all 323 real files, **15 files across 13 subjects** failed this check (1 of the 15 is the `H03/temp_180214.tdms` scratch file already excluded above, leaving 14 real subject runs).

### Root cause

None of the 15 files show corruption, duplicate trials, or out-of-order trials. Every one is a **perfectly consecutive run of indices that simply doesn't start at 1** (e.g. indices `26..68` instead of `1..43`). The `nTrialIndex` counter in these recordings is a **session-scoped LabVIEW counter, not reset per `.tdms` file** — it keeps incrementing across a subject's whole recording session. A file whose first logged trial has index > 1 means earlier trials were logged under a different (often the immediately preceding) file for that subject, not that trials were lost from *this* file.

### Stage 1 fix

`validate_behavior_dataframe` was changed to require a **gap-free, duplicate-free, non-decreasing consecutive sequence starting at any index ≥ 1** — instead of requiring the first index to be exactly `1`. This still rejects genuine corruption (gaps, duplicates, reordering); it only stops enforcing a false assumption about where the counter starts.

**Effect**: all 320 real runs (323 files minus the 3 excluded scratch files) now
parse and validate successfully (previously 15 failed). Stage-1 derivatives
created under the earlier schema must still be re-ingested to receive the latest
classification and SPD-provenance fields.

Whether a run's trial *count* matches its MEG recording is a separate concern, enforced downstream at the epoching stage (`meg_tokens/meg/epoching.py::synchronize_events_and_behavior`) — the rest of this report is about that.

---

## 4. `nOutcome` — a second silently-discarded field, and what it revealed

While investigating the 14 offset runs above (`H08RT1` in particular), a second, independent, dataset-wide issue was found: `parse_single_trial` parsed the `nOutcome` field out of every trial's raw event log but never
included it in the output — the field existed transiently and was thrown away. `nOutcome` is LabVIEW's own trial-outcome code, and one specific value matters a great deal for MEG alignment:

- **`nOutcome == 7003`** marks a trial that **never actually started**: no go cue was ever issued (`tGO == 0`), no tokens were shown (`Tokens.Data` empty), and `nChoiceMade == 0`. Checked structurally on **all 229 occurrences across 54 of the 323 source files** — every single one matches this pattern with zero exceptions. After excluding the three documented scratch files, production ingestion retains **226 occurrences across 53 files**; the difference is exactly three rows in `H18/temp_181024.tdms`. These trials have no MEG go-cue event *by design*, not by data loss.
- Other non-zero codes (`7021` = wrong choice, made with a real go-cue; `7006`/`7011` = ordinary no-response timeouts with a real go-cue already issued) don't affect MEG alignment — the trial has a normal go-cue either way.

Before this was accounted for, any run containing an `nOutcome == 7003` trial would show a spurious MEG/behavior trial-count mismatch at the epoching stage and simply fail — for example `H02Fast4` (never flagged by the `nTrialIndex` check, so previously invisible): 69 TDMS trials, 7 of them `nOutcome == 7003`, and MEG has exactly 62 go events — 69 − 7, not 69.

### Fix

`meg_tokens/meg/epoching.py::synchronize_events_and_behavior` now excludes `nOutcome == 7003` rows from the trial count and alignment before comparing to MEG events (the constant is `OUTCOME_NEVER_STARTED` in `meg_tokens/behavior/tdms.py`). These rows stay in the Stage 1 behavior TSV for provenance and QC, but the behavioral summary excludes them from trial counts and performance analyses because the participant was never presented with a trial. The summary reports their number separately as `n_never_started_trials`.

### Behavioral before/after impact

A full production replay on the 320 retained runs (32 subjects) compared the previous behavioral-analysis logic with the new started-trial view. The source TSVs are unchanged; only the rows passed to summary metrics and counts differ. The accounting closes exactly:

| Condition | Before: logged rows counted as trials | After: actually started trials | Excluded `7003` rows | Change |
|---|---:|---:|---:|---:|
| RT | 2,565 | 2,527 | 38 | -1.48% |
| Fast | 9,070 | 8,990 | 80 | -0.88% |
| Slow | 7,455 | 7,347 | 108 | -1.45% |
| **Total** | **19,090** | **18,864** | **226** | **-1.18%** |

The largest subject/condition count corrections illustrate why the distinction matters: H11 RT changed from 83 to 69 trials (-14; -16.87%), H03 Fast from 321 to 303 (-18; -5.61%), and H11 Slow from 243 to 212 (-31; -12.76%). Previously, those denominators implied that the participant had been presented with trials that LabVIEW never actually started. The new `n_never_started_trials` column keeps the excluded count visible rather than silently discarding it.

Most numerical performance results did **not** change. Motor baseline, mean Fast/Slow decision time, percent correct, correct/error decision time, and all three trial-class means are identical for every subject before and after. Those functions already required `nChoiceMade > 0`, which incidentally excluded `7003` rows along with ordinary no-response trials.

Post-error sequencing is the one performance analysis that changes. Previously, a `7003` log row between two presented trials broke their adjacency because the row had `nChoiceMade == 0`; removing never-started rows first correctly makes the surrounding presented trials consecutive. Mean post-correct decision time changed for 16 subjects and mean post-error decision time for 5. The largest observed shifts were small: H11 post-correct changed from 1450.75 to 1441.45 ms (-9.30 ms), and H08 post-error changed from 1032.94 to 1042.79 ms (+9.85 ms).

### Combined scientific impact

Taken together, the changes preserve the behavioral estimates that were already scientifically correct while fixing the places where data meaning or trial identity was wrong:

- Started-trial denominators and post-error adjacency are corrected, as quantified above.
- No-choice trials use missing `rawRT` and `isCorrect` values instead of appearing downstream as zero-latency responses or incorrect choices.
- Legitimate RT trial-class label `r` is represented as unclassified, while unknown labels and missing structural fields now raise errors instead of silently becoming plausible values.
- Valid session-scoped trial sequences starting above 1 are retained; genuine gaps, duplicates, and reordering remain errors.
- MEG epochs are paired only with trials that actually received a go cue. Never-started trials, genuine trigger dropouts, verified trailing-boundary mismatches, and the one unrecoverable trial remain distinct cases rather than being handled by one permissive truncation rule.

Therefore, unchanged motor-baseline, decision-time, accuracy, and difficulty results are evidence that already-valid conclusions were preserved—not that the pipeline returned to its previous behavior. Trial denominators, sequential behavioral interpretation, failure visibility, and neural/behavioral pairing are materially different and now follow the observed experimental structure.

---

## 5. Full-dataset MEG cross-check — methodology and final classification

### Methodology: from nearest-timestamp to exhaustive content-based matching

Every TDMS run was matched to its `.ds` recording with an exhaustive, per-subject, content-based method applied across the full dataset:

1. **Inter-trial-interval fingerprint**: the sequence of gaps between consecutive MEG trial-start trigger pulses (code `262144`, or `524288` for subject H06 whose two trigger codes are swapped — confirmed by a full 31-subject scan, no other subject affected) is compared against the sequence of gaps between consecutive TDMS trial timestamps (`nInitialTime`, the session-scoped counter each trial's `Events` block records), against every `.ds` file belonging to that subject. The winning offset separates from the next-best by 100–1000x in mean error, landing at sub-millisecond precision — a wrong match never comes close.
2. **Trial-start → go-cue latency**: on the trials that align, the MEG-measured delay between a trial-start pulse and its go-cue pulse is compared against the TDMS-logged `tGO` for that trial. A true match agrees to ~1.6–4.3ms (a small, highly consistent hardware trigger lag) — confirmed with zero exceptions across every run checked this way.

This is authoritative because it checks each run's actual internal timing structure against every candidate file for that subject, rather than assuming recording order or trusting a possibly-drifted software clock timestamp.

### Final classification — all 320 real runs

| Category | Count | Resolution |
|---|---|---|
| Exact match (no discrepancy once `nOutcome == 7003` is excluded) | 184 | none needed |
| Single trailing boundary discrepancy (MEG has one or more extra/missing trial-start pulses only at the very end of the recording) | 114 | `KNOWN_TRAILING_TRIAL_MISMATCHES` / `mismatch_policy` → `on_mismatch="truncate"` |
| Interior go-cue dropout (a trial has a real trial-start pulse but its go-cue never fired/was recorded) | 21 | `KNOWN_GO_RECONSTRUCTION_RUNS` / `reconstruct_missing_go_events` |
| Trial with no MEG event at all (no trial-start, no go-cue), despite being a real completed trial | 1 (`H12Slow2` trial 2) | `KNOWN_UNRECOVERABLE_TRIALS` / `exclude_unrecoverable_trials` |
| Excluded scratch files | 3 | `behavior_ignore_files` (§2) |

**184 + 114 + 21 + 1 + 3 + 0 remaining = 323.** Every file in the dataset is now either fixed or has an explicit, documented reason it needs no fix.

---

## 6. The 114 trailing-boundary runs

All 114 show the same structural pattern: trial-start pulses align with TDMS's trial sequence for the recording's entire length except at the very end, where either MEG has one or more extra trial-start pulses with no behavioral log entry (a trial that was triggered but never got written —e.g. `H03RT1`, MEG +1) or TDMS logged trials MEG never captured a matching start pulse for (e.g. `H06Slow2`, MEG −12; the extreme case, `H02RT1`, MEG−39, i.e. its entire go channel — handled separately, see §7). Every one of the 114 was individually verified both ways described in §5, zero exceptions, max residual 4.25ms.

### Fix

`meg_tokens/meg/epoching.py` gained `KNOWN_TRAILING_TRIAL_MISMATCHES`, an explicit, documented set of `(subject, condition, run)` keys (114 entries), and `mismatch_policy(subject, condition, run)`, which resolves to `"truncate"` only for those runs and `"error"` for everything else. `meg_tokens/workflows/preprocessing.py` calls this instead of hardcoding
`on_mismatch="error"`, and prints a message identifying the run whenever truncation fires, so it's visible in pipeline output, not silent. This mirrors the existing `SUBJECT_EVENT_OVERRIDES` pattern already in the codebase (used for H06's swapped trigger codes): a narrow, explicit, documented exception list, not a blanket behavior change.

`on_mismatch="truncate"` drops the excess from whichever side is longer (`n_keep = min(n_events, n_trials)`), which is correct here specifically *because* every one of the 114 was confirmed to have its discrepancy only at the trailing boundary, never scattered through the run and never at the front — that positional claim is what makes blind truncation safe, not just the count matching.

**Scientific impact**: this recovers all valid, correctly paired epochs from 114 runs (about 36% of the production dataset) instead of discarding those runs because of an unmatched trailing boundary. It improves trial coverage and statistical power without shifting MEG–behavior correspondence: only the unmatched tail is removed, while unknown or interior mismatches remain hard errors rather than being silently truncated.

---

## 7. The 21 go-cue reconstruction runs

21 runs have one or more trials with a real trial-start pulse but no matching go-cue — a genuine trigger dropout, not fixable by truncation since the gap isn't confined to a boundary:

- **`H02RT1`** is the extreme case: all ~39 real trials are missing their go-cue (0 go events in the recording at all), though the trial-start pulses are intact and align perfectly.
- The other **20 runs** each have exactly one interior trial with a missing go-cue, everything else in the run intact.

### Fix

`reconstruct_missing_go_events` (`meg_tokens/meg/epoching.py`) pairs each trial positionally with its trial-start pulse (valid for these 21 runs, confirmed via §5's method — any extra/unlogged trial-start pulses only trail past the last real trial, so they fall outside the trial range and are never touched), and for any trial with no matching go-cue within its expected window, synthesizes one at `trial_start + tGO + lag`. The lag is calibrated **from that run's own trials that do have a real go-cue** (mean ~2–4ms, matching the hardware lag already established) — falling back to documented default (4ms) only for `H02RT1`, the one run with zero real go-cues left to calibrate from. `KNOWN_GO_RECONSTRUCTION_RUNS` lists the 21 confirmed runs; `needs_go_reconstruction(subject, condition, run)` gates it. Trials with `nOutcome == OUTCOME_NEVER_STARTED` are never reconstructed — they structurally have no go-cue by design (§4), and the function correctly distinguishes "missing because of a real dropout" from "missing because it never happened," even within the same run (`H07Slow2` and `H07Slow4` each have both patterns simultaneously).

`meg_tokens/workflows/preprocessing.py` calls this before the truncate/error decision, only when aligning to `"go"`, and prints a message identifying the run whenever it fires.

---

## 8. `H12Slow2` — the one run needing a bespoke, single-trial fix

`H12Slow2` is matched to `H12_..._05.ds` (193x separation, confirmed correct) but has irregularities at *both* ends, so it doesn't fit the trailing-only pattern the other 114 runs share:

- The recording started a few seconds **after** TDMS logging began that file, missing the first 3 trials' trial-start pulses entirely: trial 1 (`nOutcome == 7003` — no loss, it never really happened), trial 2 (a completely normal, real trial — genuinely lost, no anchor to recover it from) and trial 3 (`nOutcome == 7003` — no loss).
- The recording also has the same trailing artifact as the 114-run group: one extra trial-start pulse at the very end with no go-cue and no TDMS log entry — but this pulse is a trial-*start* event, never a candidate "Go" event, so it's already invisible to the go-alignment count and needs no separate handling.

The arithmetic closes out exactly: 65 total trials − 7 (`nOutcome == 7003`)− 1 (trial 2, unrecoverable) = 57, precisely matching MEG's 57 go events. Verified against real data with the actual production code: the 57 aligned trials show a 2.38ms mean / 4.83ms max residual against `tGO` — the same signature as every other confirmed-correct alignment in this report.

### Fix

Since this exact combination doesn't recur anywhere else in the dataset, it was implemented as a narrow, explicit, single-run exception rather than a new general mechanism — consistent with `SUBJECT_EVENT_OVERRIDES` / `KNOWN_TRAILING_TRIAL_MISMATCHES` / `KNOWN_GO_RECONSTRUCTION_RUNS`. `KNOWN_UNRECOVERABLE_TRIALS` (`meg_tokens/meg/epoching.py`) maps `('H12', 'Slow', '2')` → `{2}`; `exclude_unrecoverable_trials(subject, condition, run, behavior_df)` drops those trial(s) before the mismatch check runs. `meg_tokens/workflows/preprocessing.py` calls it right before `mismatch_policy`, and prints how many trials were excluded when it fires. No truncation is needed afterward — once trial 2 is excluded, the count already matches exactly.

---

## 9. Later behavioral-label and SPD hardening

The T0-1/T0-2 investigation added four protections to the same Stage-1
contract. Full scientific detail is in
`docs/t01_nprob_trial_class_investigation.md` and
`docs/behavior_labelling_master_record.md`.

1. **Scientific-notation parsing.** The original `nProb` expression misread
   exponent notation: 881 trials contained an out-of-range parsed value, 870
   changed in value, and 11 lost an entry and became desynchronized from
   `tTime`. Parsing now accepts signs and exponents, clamps floating-point zero
   within tolerance, validates `[0,1]`, and requires equal token-field lengths.
   The defect changed no historical class or basic behavioral result because it
   occurred outside the indices used by the old override.
2. **Label preservation.** The historical chosen-target `nProb` override changed
   1,119 / 5,224 recorded designed labels. The production parser now preserves
   raw `'e'`, `'a'`, and `'m'` labels and applies the deterministic complete
   correct-target design rule only to raw `'x'` trials. All 5,224 designed labels
   are preserved in real-data validation.
3. **Provenance.** Parsed output now stores `sTrialClassRaw`,
   `trial_class_source`, `trial_class_rule`, `sp_design_correct`,
   `token_log_rows`, and `token_log_short`. Logged `nProb` and `tTime` remain
   unchanged. A random trial with no valid correct target remains unclassified
   with no design profile.
4. **Short-log SPD safeguards.** There are 5,363 short logs and 10,961 complete
   logs among the 16,324 started-and-chosen Fast/Slow trials. Logged
   chosen-target SPD is reported for all logs and separately for the validated
   15-row-only sensitivity set. Design-derived runtime alignment and SPD require
   exactly 15 rows and raise an error for every short log.

The cause of the one-jump `nTokenDir`/`nProb` offset in the 14-row logs remains
unknown. This does not affect classification, which uses the complete
`sTokenDirs` design profile.

---

## Summary of code changes

| File | Change |
|---|---|
| `meg_tokens/behavior/tdms.py` | Strict structural and token-field parsing; gap-free consecutive `nTrialIndex` starting at ≥1; required `nOutcome`; `OUTCOME_NEVER_STARTED` validation; raw-label preservation; random-only deterministic classification; design profile, provenance, and short-log fields. |
| `meg_tokens/behavior/success_probability.py` | Equation 1, complete target-referenced profiles, deterministic SP(2/3/5/8/11) rule, SPD timing helper, and a hard guard against aligning design SP to a short runtime log. |
| `meg_tokens/meg/epoching.py` | `synchronize_events_and_behavior` excludes `OUTCOME_NEVER_STARTED` trials before counting/aligning. Added `KNOWN_TRAILING_TRIAL_MISMATCHES` (114 runs) + `mismatch_policy()`. Added `KNOWN_GO_RECONSTRUCTION_RUNS` (21 runs) + `needs_go_reconstruction()` + `reconstruct_missing_go_events()`. Added `KNOWN_UNRECOVERABLE_TRIALS` (1 run) + `exclude_unrecoverable_trials()`. Added a `'start'` alignment event (code `262144`, swapped for H06) to `DEFAULT_EVENT_IDS`/`SUBJECT_EVENT_OVERRIDES`. |
| `meg_tokens/meg/__init__.py` | Registered the new public functions. |
| `meg_tokens/workflows/behavior.py` | Behavioral metrics and started-trial counts exclude `OUTCOME_NEVER_STARTED`; reports `n_never_started_trials`; writes paired all-log and validated-15-row SPD summaries with provenance metadata. |
| `meg_tokens/validation/spd.py` | Reproduces the historical labels, validates logged/design profiles and SPD, and withholds analysis-ready design SPD for short logs. |
| `meg_tokens/workflows/preprocessing.py` | `epoch_subjects` now: reconstructs missing go-cues when needed, excludes known-unrecoverable trials, and applies `mismatch_policy` instead of hardcoding `on_mismatch="error"` — each step prints when it fires. |
| `tests/test_tdms_parser.py` | Covers strict parsing, session-scoped indices, label preservation, random deterministic classification, SP(11), unmatched trials, and identical classification across 14-/15-row logs. |
| `tests/test_success_probability.py`, `tests/test_spd_validation.py` | Cover Equation 1, deterministic boundaries, SPD timing, paired-log validation, and rejection of short-log design alignment. |
| `tests/test_epochs_builder.py` | Added tests for `mismatch_policy`, `needs_go_reconstruction`, `reconstruct_missing_go_events` (gap-filling, skipping never-started trials, default-lag fallback, error on insufficient start pulses), and `exclude_unrecoverable_trials`. |
| Several other test fixtures across the suite | Updated hand-built behavior DataFrames to the current schema, including `nOutcome` and the classification/SPD provenance fields. |
| `config/tokens.toml.template` | `behavior_ignore_files` excludes the 3 confirmed scratch files, each with a comment explaining why. |

## Verification

- 97 focused behavior/TDMS/SPD/downstream compatibility tests pass. The broader
  nonplot suite has 231 passes and the same four unrelated `specparam`/archive
  failures; two plotting modules require the missing `seaborn` installation to
  collect.
- The full dataset scan covers all 323 real `.tdms` paths: 320 retained
  runs parse and validate with zero errors, and the three scratch files remain
  explicitly excluded.
- Across all 16,324 started-and-chosen Fast/Slow trials, the new method preserves
  5,224 / 5,224 designed labels. Analysis-ready design SPD is withheld for
  5,363 / 5,363 short logs and available for 10,961 / 10,961 complete logs.
- **Full-dataset, real-code verification** (not throwaway investigation scripts — the actual `mismatch_policy`, `needs_go_reconstruction`, `reconstruct_missing_go_events`, `exclude_unrecoverable_trials`, and `synchronize_events_and_behavior` functions, run against every real `.tdms` file matched to its real `.ds` recording): **184 exact + 114 truncated + 21 reconstructed + 1 explicitly-excluded-trial = 320/320 correct, zero unexpected failures, zero unexpected passes, zero wrong alignments.**

## Remaining notes

- **Stage-1 regeneration:** existing derivatives must be re-ingested from TDMS
  before downstream workflows can rely on the new class provenance, design SP,
  short-log, and SPD-summary fields.
- **QC command:** the validator is committed but is not yet exposed through the
  planned reproducible `meg-tokens behavior qc` command.
- **Project-wide raw-file mapping**: the exhaustive matching method in §5 was applied to every run while resolving this report, so the `.tdms` → `.ds` mapping is now verified dataset-wide, not just for the originally-flagged 14 runs. Worth keeping in mind if raw data folders are ever reorganized: rely on this method, not recording order.

> **Deferred to the epoching phase:** the complete production execution from mounted raw CTF `.ds` recordings through cleaned FIF inputs and final epochs will be rerun during epoching. That verification must confirm the operational order—go-marker reconstruction, unrecoverable-trial exclusion, `7003` filtering during synchronization, and verified trailing truncation—and reconcile the resulting event, metadata, and epoch counts for every run. The behavioral-phase verification in this report covers TDMS ingestion, behavioral summaries, and event-level QC; it does not replace that final end-to-end epoch-derivative check.
