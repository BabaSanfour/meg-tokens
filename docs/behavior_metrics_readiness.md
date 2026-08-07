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

## Next

1. Begin MEG analysis and populate T0-6 from confirmed movement/artifact/session
   exclusions.
2. Use `docs/behavior_analysis_roadmap.md` for additional behavioral analyses.

Related details are maintained in:

- `docs/behavior_qc_report.md` — ingestion and MEG/behavior alignment QC
- `docs/behavior_t0_1_nprob_trial_class.md` — class and SPD specification
- `docs/data_contract.md` — derivative schemas and join keys

Preprint: Thiery et al. (2022),
[bioRxiv 10.1101/2022.06.14.494674](https://doi.org/10.1101/2022.06.14.494674).
