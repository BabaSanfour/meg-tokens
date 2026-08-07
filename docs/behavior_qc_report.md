# Behavior (TDMS) and MEG/Behavior Alignment QC

This report records the final quality controls for Stage 1 TDMS ingestion and
MEG/behavior alignment. Results were checked against 323 TDMS files from 32
subjects and 399 raw CTF recordings in the `DDM-tthiery` dataset.

## 1. Stage 1 ingestion

`meg_tokens/behavior/tdms.py` now applies strict structural validation before
writing one BIDS-derivative behavior table per run:

- Every `.tdms` filename must match `H<subject><condition><run>_<YYMMDD>.tdms`.
  Non-matching files raise an error unless explicitly listed in
  `behavior_ignore_files`.
- Duplicate `(subject, condition, run)` files raise an error rather than
  overwriting one another.
- Missing `Events` properties, empty trial groups, and missing structural
  fields raise a file/trial-specific error. Valid no-choice trials remain
  valid: `nChoiceMade == 0` and a missing `tEnterTarget` are expected outcomes.
- `sTrialClass == 'r'` is recognized as the RT-task label.
- `nOutcome` is retained and validated. `nTrialIndex` must be gap-free,
  duplicate-free, ordered, and consecutive, but may start above 1 because the
  LabVIEW counter is session-scoped rather than reset per file.

The corrected index rule resolves the 15 apparent failures found in the full
scan (14 genuine runs plus one scratch file): all 320 retained runs now parse
and validate; the three excluded files are documented scratch recordings.

### Excluded scratch files

These files are short aborted/test fragments, not experimental runs:

| Subject | File |
|---|---|
| H03 | `temp_180214.tdms` |
| H18 | `temp_181024.tdms` |
| H23 | `temp_181121.tdms` |

They are excluded only through the explicit `behavior_ignore_files` config
allowlist. Any other unexpected `.tdms` file remains an error.

### Never-started trials

`nOutcome == 7003` means that no go cue was issued, no tokens were shown, and
the subject made no choice (`tGO == 0`, empty `Tokens.Data`,
`nChoiceMade == 0`). All 229 occurrences across 54 source files matched this
structure; production retains 226 occurrences across 53 files after excluding
the scratch files. Codes `7021`, `7006`, and `7011` represent trials with a
real go cue and do not affect MEG alignment.

During synchronization, `OUTCOME_NEVER_STARTED` rows are retained in the Stage
1 tables but excluded from behavioral counts and MEG event matching. The
accounting is:

| Condition | Logged rows | Started trials | Excluded 7003 | Change |
|---|---:|---:|---:|---:|
| RT | 2,565 | 2,527 | 38 | -1.48% |
| Fast | 9,070 | 8,990 | 80 | -0.88% |
| Slow | 7,455 | 7,347 | 108 | -1.45% |
| **Total** | **19,090** | **18,864** | **226** | **-1.18%** |

The excluded count is reported as `n_never_started_trials`. Motor baseline,
Fast/Slow decision time, accuracy, correct/error decision time, and difficulty
means are unchanged because those analyses already required a choice. Removing
never-started rows does correct post-error adjacency: post-correct means
changed for 16 subjects and post-error means for 5. The largest shifts were
H11 post-correct, -9.30 ms, and H08 post-error, +9.85 ms.

## 2. Dataset-wide MEG matching

Each retained TDMS run was matched to a subject’s candidate `.ds` recordings
using two independent timing fingerprints:

1. Inter-trial intervals between MEG trial-start pulses (code `262144`, or
   `524288` for H06) were compared with intervals between TDMS `nInitialTime`
   values. The correct offset separated from the next-best match by 100–1000×
   in mean error and reached sub-millisecond precision. H06’s swapped
   start-trigger code was the only subject-specific trigger exception.
2. Trial-start-to-go-cue latency was checked against TDMS `tGO`; confirmed
   matches differed by approximately 1.6–4.3 ms with no exceptions.

This avoids relying on recording order or software timestamps alone.

## 3. Alignment results and explicit exceptions

| Category | Count | Production resolution |
|---|---:|---|
| Exact alignment | 184 | No exception |
| Trailing-boundary mismatch | 114 | `KNOWN_TRAILING_TRIAL_MISMATCHES` and `on_mismatch="truncate"` |
| Interior missing go cue | 21 | `KNOWN_GO_RECONSTRUCTION_RUNS` and go-cue reconstruction |
| Unrecoverable real trial | 1 (`H12Slow2`, trial 2) | `KNOWN_UNRECOVERABLE_TRIALS` |
| Scratch files | 3 | `behavior_ignore_files` |

Thus, `184 + 114 + 21 + 1 + 3 = 323`: every source file has either a verified
resolution or an explicit exclusion.

### Trailing-boundary mismatches

All 114 runs align throughout their recordings and differ only at the final
boundary. Examples include one extra MEG trial-start pulse (`H03RT1`), missing
TDMS-side events (`H06Slow2`, -12), and the all-go-channel case (`H02RT1`,
-39). Maximum confirmed timing residual was 4.25 ms.

`mismatch_policy` truncates only these documented runs and leaves unknown or
interior mismatches as hard errors. Truncation is safe because the discrepancy
was independently confirmed to occur only at the trailing boundary. This
recovers valid epochs from approximately 36% of the production runs without
changing MEG/behavior pairing.

### Missing go cues

Of the 21 reconstruction runs, `H02RT1` has approximately 39 real trials with
no recorded go cue; the other 20 each have one interior dropout. For each run,
`reconstruct_missing_go_events` pairs trials with trial-start pulses and
synthesizes the missing go cue at `trial_start + tGO + lag`. The lag is
calibrated from that run’s real go cues, with the documented 4 ms fallback only
for `H02RT1`. Never-started `7003` trials are never reconstructed, including
mixed-pattern runs such as `H07Slow2` and `H07Slow4`.

### `H12Slow2`

This run matches `H12_..._05.ds` with a 193× separation from the next candidate.
It has both a leading and trailing discrepancy and therefore is not part of the
trailing-only group. The recording missed the first three TDMS trial
starts: trials 1 and 3 were never-started rows, while trial 2 was a genuine
unrecoverable trial. The trailing extra start pulse has no go cue and is not a
candidate event.

The count closes as `65 total - 7 never-started - 1 unrecoverable = 57` MEG go
events. The matched events have a 2.38 ms mean and 4.83 ms maximum `tGO`
residual. `exclude_unrecoverable_trials` removes only
`('H12', 'Slow', '2') → {2}` before synchronization.

## 4. Code and verification

| Area | Final implementation |
|---|---|
| TDMS parser (`meg_tokens/behavior/tdms.py`) | Strict filename, event, field, outcome, index, and timing validation; explicit scratch-file allowlist. |
| MEG epoching (`meg_tokens/meg/epoching.py`, `meg_tokens/workflows/preprocessing.py`) | Never-started filtering, start-event handling, documented trailing policy, go-cue reconstruction, and the H12 unrecoverable-trial exception. |
| Behavior workflow | Started-trial metrics and separate `n_never_started_trials` reporting. |
| Configuration | `behavior_ignore_files` with reasons for all three scratch files. |
| Public exports | New epoching helpers are exposed through `meg_tokens/meg/__init__.py`. |
| Tests | Parser, index, outcome, synchronization, mismatch, reconstruction, and exception coverage. |

Verification results:

- 97 focused behavior/TDMS/downstream compatibility tests passed; the
  broader nonplot suite had 231 passes and four unrelated
  `specparam`/archive failures.
- All 320 retained TDMS runs parse and validate with zero unexpected failures.
- Full real-code MEG verification classified the retained runs as **184 exact,
  114 truncated, 21 reconstructed, and 1 explicitly excluded trial**: 320/320
  correct, with zero wrong alignments.
- `meg-tokens behavior qc` runs the committed source-TDMS validator and prints
  its grouped summary.

Existing Stage 1 derivatives must be regenerated with:

```bash
meg-tokens --config tokens.toml behavior ingest
```

The full raw-recording-to-epoch replay remains deferred to the epoching phase;
that replay must verify go-cue reconstruction, unrecoverable-trial exclusion,
`7003` filtering, trailing truncation, and final event/metadata/epoch counts.
