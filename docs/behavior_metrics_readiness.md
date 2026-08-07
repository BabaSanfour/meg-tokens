# Behavioral Metrics Readiness

Active status for behavioral analysis and comparison with the 2022 Thiery et
al. preprint. The dataset has 32 subjects, 256 Fast/Slow runs, and 16,324
started-and-chosen task trials.

## Status

T0-1 through T0-5 are complete:

- Stage 1 retains all source rows while `7003` never-started trials are excluded
  from analysis through the shared `started_trials` filter.
- Canonical `rawRT`, `isCorrect`, trial-class provenance, and logged SPD are used
  throughout. SPD is reported for all logs and as a 15-row sensitivity view.
- `behavior analyze` writes subject summaries, paired group statistics, and a
  19,090-row trial-feature table with a stable MEG join key.
- Group results report mean ± SEM, paired `t`, `p`, `df`, and Cohen's `dz` for
  class DT, Fast/Slow DT and errors, and SPD by class.
- DT analyses retain every finite task DT without trimming or winsorization.
  Five negative DTs are retained and flagged as anticipations; no-responses have
  no valid RT and are excluded.

T0-6 remains: populate the four subject exclusions after MEG quality review and
before claiming replication of the preprint's **N=28** results. The config hook
exists but is intentionally empty.

**The excluded four**: `H06, H07, H10, H20`. Legacy code (10 scripts, MEG and behavior)
consistently excludes this set. MEG-signal corroboration for
*why* each was excluded is still weak/mixed (only H10 shows a clear outlier
signal), but no longer affects identity.

## Preprint comparison

Current values use all 32 subjects:

| Result | Preprint N=28 | Current N=32 |
|---|---|---|
| Easy vs ambiguous DT | 1028 ± 59 vs 1405 ± 74; `t=-15.04` | 1033 ± 59 vs 1433 ± 70; `t=-16.80` |
| Easy vs misleading DT | 1028 ± 59 vs 1433 ± 79; `t=-13.10` | 1033 ± 59 vs 1357 ± 78; `t=-11.86` |
| Ambiguous vs misleading DT | `t=-1.84`, `p=0.077` | `t=5.43`, `p=6.29e-6` |
| Fast vs Slow DT | 1166 ± 71 vs 1293 ± 68; `t=-6.08` | 1186 ± 70 vs 1313 ± 66; `t=-6.19` |
| Fast vs Slow errors | 45.1 vs 36; `t=6.10` | 46.3 vs 36.9; `t=6.06` |
| SPD by class | Higher in easy | Paired all-log and 15-row tests emitted |

Fast/Slow effects are close to the preprint. The ambiguous/misleading effect is
reversed, and none of the 35,960 possible four-subject exclusions reverses the
current positive effect; unknown exclusion IDs alone cannot explain it. The
opt-in published-value regression test uses `MEG_TOKENS_REAL_CONFIG` and should
run once T0-6 is populated.

**The cause of the reversed contrast is identified**: a *reference-frame*
difference in trial classification, not the exclusion list. Thomas's
`Modify_df_preproc.ipynb` applies the preprint's SP thresholds to the runtime
`nProb` (**chosen-target**) profile, overwriting every trial's class; we apply
them to a design-derived **correct-target** profile, for random (`'x'`) trials
only. The chosen frame does flip the contrast to the published sign, but it is
confounded — most of its "misleading" class are trials where evidence clearly
favoured the correct target and the subject simply erred. **We keep the
design frame.** Comparison table, confound breakdown, and the
zero-inferred-misleading explanation: `docs/behavior_t0_1_nprob_trial_class.md`
§3b.

Also ruled out: trial filtering, and the `H02` `RT1`/`RT2` baseline (~1.4 ms
on group means, cancels in every paired contrast; detail in
`docs/behavior_qc_report.md` §3).

Related details are maintained in:

- `docs/behavior_analysis_roadmap.md` — extended analyses and their status
- `docs/behavior_roadmap_results.md` — measured results for those analyses
- `docs/behavior_qc_report.md` — ingestion and MEG/behavior alignment QC
- `docs/behavior_t0_1_nprob_trial_class.md` — class and SPD specification
- `docs/data_contract.md` — derivative schemas and join keys
- `docs/meg_t0_6_subject_exclusion_qc.md` — MEG-quality subject exclusion
  investigation.

Preprint: Thiery et al. (2022),
[bioRxiv 10.1101/2022.06.14.494674](https://doi.org/10.1101/2022.06.14.494674).
