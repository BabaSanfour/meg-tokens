# MEG/Behavior Alignment QC (Stage 4: Epoching)

Event-level alignment QC for epoch extraction. Stage 0/1 issues (TDMS
ingestion, trial classification, MEG session matching) moved to
`docs/behavioral_pipeline.md`. Results checked against 323 TDMS files from
32 subjects and 399 raw CTF recordings in the `DDM-tthiery` dataset.

## 1. Alignment results and explicit exceptions

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

### H02 motor baseline: keeping both RT runs

Thomas's notebooks use only `RT2` for `H02` (`if s == 'H02': files_RT =
[s+'_RT2']`), consistent with `H02RT1`'s go-cue defect above. We keep both
runs for every subject: `H02RT1` mean 508.6 ms vs `RT2` 588.4 ms — an 80 ms
gap, unexceptional across the cohort (mean |gap| 30.4 ms, max 95.5 ms at H08,
where Thomas also kept both runs). Using both runs puts our baseline 39.4 ms
below Thomas's RT2-only baseline; this cancels exactly in every paired
within-subject contrast and shifts absolute group means by ~1.4 ms.

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

## 2. Code and verification

| Area | Final implementation |
|---|---|
| TDMS parser and schema (`meg_tokens/behavior/tdms.py`, `meg_tokens/behavior/schema.py`) | Strict filename, event, field, outcome, index, and timing validation; explicit scratch-file allowlist. |
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

Checked against Thomas's original TDMS parser
(`archive/replicated/DDM_analysis_scripts/Create_df.ipynb`) and behavior
notebooks (`archive/replicated/DDM_scripts/scripts_new/`), two mismatches
matter beyond the `sTrialClass` `'r'`-code encoding (`docs/behavioral_pipeline.md`)
and the `H02` baseline decision above: the
`sTrialClass` reference-frame difference in `Modify_df_preproc.ipynb` (root
cause of the reversed ambiguous/misleading contrast — see
`docs/behavioral_pipeline.md`, Findings), and our `key: value` line
parser replacing Thomas's fixed-offset string slicing, a deliberate
modernization (`docs/data_contract.md`) that is functionally equivalent on
every field both extract. `rawRT` and the post-error-slowing adjacency rule
match exactly.
