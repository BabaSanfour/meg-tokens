# MEG Tokens Data Contract

Behavioral replication of the legacy scripts using modern MNE-Python,
BIDS/MNE-BIDS-style derivatives, and explicit analysis tensors.

## Rules

- Real project inputs only; nothing generates fake MEG, source or behavior data.
- Missing inputs raise an error naming the required file.
- Modern MNE/scientific-Python APIs over step-by-step ports, where they
  preserve the same scientific behavior.
- MEG in MNE-native formats under a BIDS derivatives root; behavior as `.tsv`
  with JSON sidecars. No `.mat`, `.h5`, or undocumented pickles.

## Behavioral module boundaries

One dependency direction, never feeding back:

```text
project I/O → behavior schema/tables → trial features → analyses → workflows
```

| Module | Owns |
| :--- | :--- |
| `schema.py` | defines and validates table contracts |
| `tdms.py` | interprets TDMS records only; emits `RAW_TRIAL_COLUMNS`, every field LabVIEW wrote and nothing inferred. Raises on a label that is neither a known code (`x`/`e`/`a`/`m`/`r`) nor an integer |
| `tables.py` | composes the project table loader with the schema |
| `classification.py` | adds trial classes, applied by the derivative stage that owns the choice (`ProjectConfig.infer_random_classes`) |
| workflows | locate inputs, call these layers, write outputs; never implement formulas |

Both inputs to class inference (`sTrialClassRaw`, `sp_design_correct`) survive
into the raw table, so a derivative is regenerable from the staged layer alone.

## Paths and filenames

BIDS-Derivatives entity ordering (`sub-ses-task-acq-run-proc-space-desc-suffix`,
see `meg_tokens/io/contract.py`), but `desc` is a hyphen-joined growing tag list
(`desc-slow_beh`) rather than a single value. Every value chained into `desc` is
recorded verbatim in the sidecar's `metadata`; the two must never disagree.
Changing this breaks every `DerivativeLayout` path builder.

Build paths with `meg_tokens.io.derivative_path`, never by hand:

```python
from meg_tokens.io import derivative_path

path = derivative_path(
    "/data/tokens-bids", subject="H01", datatype="beh",
    task="tokens", run="1", description="slow", suffix="beh", extension=".tsv",
)
# /data/tokens-bids/derivatives/sub-H01/beh/sub-H01_task-tokens_run-1_desc-slow_beh.tsv
```

`datatype` defaults to `"meg"`.

## Preprocessing derivatives

Two cleaned recordings per staged run, one per cleaner, on a shared trunk.
`proc` names the cleaner (`dss`, `ica`); `desc` keeps the condition and
grows the `artifact` tag onto it. The artifact file is the residual — what that
cleaner removed — so the two estimates are the same quantity.

```
derivatives/sub-H01/meg/sub-H01_task-tokens_run-1_proc-dss_desc-fast_raw.fif
derivatives/sub-H01/meg/sub-H01_task-tokens_run-1_proc-dss_desc-fast-artifact_raw.fif
derivatives/sub-H01/meg/sub-H01_task-tokens_run-1_proc-dss_desc-fast_raw.json
```

Sidecar carries `stream`, `rank` and `n_dimensions_removed` (both cleaners
reduce rank and the inverse operator needs it), `compensation_grade_source` and
`compensation_applied`, the bad channels per detector with every channel's
spectral outlier score (`channel_scores`), annotated seconds before and after
cleaning, the per-cleaner detail, and the share of each magnetometer's
variance the head-movement regression removed (`head_movement_variance_removed_median`,
`head_movement_variance_removed_max`).

Every path below preprocessing names its stream (`proc`): the cleaners are
alternatives, not degrees of one thing, so there is no default.

## Epoch derivatives

One file per staged Fast and Slow run, stream and lock (`go`, `enter`,
`feedback`, `decision`); `desc` is the condition and the lock. RT runs are not
epoched: they only measure the motor baseline that decision time subtracts.

```
derivatives/sub-H01/meg/sub-H01_task-tokens_run-1_proc-dss_desc-fast-go_epo.fif
derivatives/sub-H01/meg/sub-H01_task-tokens_run-1_proc-dss_desc-fast-go_epo.json
```

Go cues come from the raw BIDS `events.tsv`, paired with the run's started
trials. The recordings carry no press or feedback trigger, so the other locks
sit at a behavioral offset from the go cue (`rawRT`, `tTrialEnd - tGO`,
`dt_ms`); a trial without a response has no `enter` or `decision` epoch. Epochs keep the MEG and reference sensors, carry the trial features
plus `tGO`, `tEnterTarget`, `tTrialEnd`, `go_onset_s` and `lock_onset_s` as
metadata, and span the lock's analysis window. An epoch whose window touches a
`BAD` annotation or runs past the recording is dropped and its reason kept in
the drop log; anything filtered is filtered on the continuous recording, not
on the epochs. An epoch the marks keep is still dropped, as `OUTLIER`, when any
magnetometer's log peak-to-peak over its analysis window exceeds that sensor's
cohort threshold for the stream and lock. The sidecar records the window, `outlier_sd`, the counts, the go cues
reconstructed, trials excluded or truncated, and the dropped epochs by reason.
A file whose rejection kept no epoch (16 of 2,048, all H10's) is still written
with its sidecar (`n_epochs` 0) as the record of why; MNE reads it back without
its metadata columns, and `epoched_meg_runs` does not return it.

The thresholds come from two derivatives built before any epoch is written.
Per subject and stream, every epoch the marks keep, one row each (`condition`,
`run`, `lock`, `trial`) with one column per magnetometer:

```
derivatives/sub-H01/meg/sub-H01_task-tokens_proc-dss_desc-epochamplitudes_meg.tsv
```

Per stream, the cohort's threshold per lock (rows) and magnetometer (columns),
the median plus `outlier_sd` robust SDs over every subject's table; the sidecar
names the subjects and the epoch counts:

```
derivatives/sub-group/meg/sub-group_task-tokens_proc-dss_desc-epochthresholds_meg.tsv
```

## Source derivatives

Per subject, the noise covariance, BEM solution and source space (the cortex
and 17 deep structures, one volume); per staged
Fast and Slow run, the forward model; per run and stream, the inverse
operator and the LCMV filters that check a deep-structure result. No source
estimate is stored: analyses apply the operator to the epochs.

```
derivatives/sub-H01/meg/sub-H01_task-tokens_desc-noise_cov.fif
derivatives/sub-H01/meg/sub-H01_task-tokens_bem.fif
derivatives/sub-H01/meg/sub-H01_task-tokens_space-subject_src.fif
derivatives/sub-H01/meg/sub-H01_task-tokens_run-1_desc-fast_fwd.fif
derivatives/sub-H01/meg/sub-H01_task-tokens_run-1_proc-dss_desc-fast_inv.fif
derivatives/sub-H01/meg/sub-H01_task-tokens_run-1_proc-dss_desc-fast_lcmv.h5
```

The covariance sidecar names the empty room, its acquired compensation grade,
flux jumps, seconds used, estimator and `hours_from_recording`; the BEM sidecar
the conductivities and the outer-skull and scalp points moved to clear the
inner skull (`*_moved`, `*_moved_max_mm`); the source-space sidecar the cortical
spacing, the volume grid (`volume_spacing_mm`), `n_sources`,
`n_cortical_sources` and `n_deep_sources` per `aseg` structure; the inverse
sidecar the stream, `rank`, `operator_residual`, `operator_rank_gap`, `loose`
(cortex and volume), `depth` and `deep_leakage`: per deep structure, the shares
of its sLORETA estimate's cross-talk from itself, the cortex and the other
structures (`own`, `cortex`, `other_deep`), the shares of its sources whose
point-spread peaks inside it and in the cortex (`found`, `found_in_cortex`),
and the cortical sources whose point-spread peaks inside it (`ghosts`).

## Band-power derivatives

Computed per run, stream and band (every lock), then assembled per subject,
stream and lock (`go`, `decision`, `enter`): every kept trial of the eight
Fast and Slow runs, in run order, with one table of its trials.

```
derivatives/sub-H01/meg/sub-H01_task-tokens_run-1_proc-dss_space-HCPMMP1_desc-fast-alpha_power.h5   (part, removed once assembled)
derivatives/sub-H01/meg/sub-H01_task-tokens_proc-dss_space-HCPMMP1_desc-go_power.h5
derivatives/sub-H01/meg/sub-H01_task-tokens_proc-dss_desc-go_trials.tsv
<fsaverage_root>/derivatives/sub-H01/meg/sub-H01_task-tokens_run-1_proc-dss_space-fsaverage_desc-fast-alpha_power.h5
```

An assembled file holds `power`, trials x bands x sources x times, float32,
linear (|z|² of the sLORETA estimate at λ² = 1), at 600 Hz over the lock's
analysis window, chunked one trial and band at a time; `times` (s from the
lock), `bands` (`theta`, `alpha`, `beta`, `gamma`, `highgamma`),
`band_edges_hz`, `sources` and `n_sources`. The sources are the 360 HCP-MMP1
regions (left then right, as the annotation orders them) followed by the 17
deep structures in `VOLUME_LABELS` order; `n_sources` gives how many of the
subject's sources each averages. A sample whose filter reaches a flux jump or
the recording's end is NaN, as is a region with no source. Row `i` of `power`
is row `i` of the trials table: the epoch metadata plus `subject`,
`condition`, `run` and `epoch` (the epoch's index in its run's drop log). The
sidecar names the method, λ², orientations, bands, transition widths,
`invalid_near`, `n_trials` and `empty_sources`.

The `fsaverage` power, about 24 TB for the cohort, stays one file per run,
stream and band under its own BIDS-like root (`band_power.fsaverage_root`):
per lock, `<lock>/power`, trials x 8,196 `oct6` sources (left then right) x
times, and `<lock>/times`, in the order of the lock's epoch file.

## Stage 0: Raw BIDSification

Raw CTF media carries no condition/run label (`H02_DDM-tthiery_20180213_03.ds`
is a subject prefix, date and sequential index). Two entry points, and no
separate command per data type:

| Command | Does |
| :--- | :--- |
| `meg-tokens meg stage-raw` | matches each session to a behavioral run, writes the manifest |
| `meg-tokens meg apply-raw-staging` | copies what the manifest says, fresh or hand-edited |

```text
BIDS/sub-H02/meg/sub-H02_task-tokens_acq-slow_run-1_meg.ds
BIDS/sub-H02/beh/sub-H02_task-tokens_acq-slow_run-1_beh.tsv
BIDS/sub-H02/anat/sub-H02_T1w.nii.gz
BIDS/sub-emptyroom/ses-20180213/meg/sub-emptyroom_ses-20180213_task-noise_meg.ds
```

Entities: condition uses `acq` (`Slow2` -> `acq-slow_run-2`) because `desc` is
derivatives-only in BIDS; `task` stays singular (`tokens`); no `ses` entity,
one acquisition date per subject.

### Matching

Candidates are pre-filtered by duration class (315s Slow/Fast, 135s RT,
everything else excluded), then matched on inter-trial-interval fingerprint.
A fingerprint match is accepted only when **all three** hold:

- mean absolute error within `fingerprint_max_error_ms` (5ms)
- beats the runner-up by at least `fingerprint_min_separation` (20x)
- no other run claims the same session

Fallback order, then: `KNOWN_SESSION_OVERRIDES` (H01/H05) -> chronological 1:1
pairing when candidate and run counts are equal -> flagged `ambiguous` for
hand-resolution. Every accepted pair is cross-checked against the
trigger-pulse count.

### Known dataset gaps

Permanent, not resolvable by re-running Stage 0:

| Subject | Kind | Gap |
| :--- | :--- | :--- |
| H05 | run RT1 | no raw MEG counterpart |
| H05 | noise | not found, any date |
| H07, H10 | anat | no FreeSurfer reconstruction |
| H26/H27 | noise (`20181206`) | `_01.ds` `.hc` corrupt (`RuntimeError: HPI information not available`); `_02.ds` valid |
| H11 | runs Fast3/4, Slow3/4 | acquired at gradient compensation grade 0; the other 315 task runs are grade 3 |

`KNOWN_NOISE_OVERRIDES` (`meg_tokens/meg/raw_staging.py`) redirects H26/H27's
noise date to `_02.ds`, same pattern as `KNOWN_SESSION_OVERRIDES`.

H11's grade cannot be corrected when staging: CTF keeps it in the `.res4`
header and applies compensation on read, so an in-memory correction is not
carried into a copied `.ds`. `read_staged_raw` raises those four runs to grade
3; the sidecar records `compensation_grade_source` and `compensation_applied`.

### Manifest

`derivatives/sub-group/meg/sub-group_task-tokens_desc-rawstaging_manifest.tsv`,
via `DerivativeLayout.raw_staging_manifest()`. Every file that will be staged,
of every kind, plus every gap -- one review pass covers the whole raw layer.
Columns are `MatchResult`'s fields in declared order; the dataclass is the
single definition and the workflow derives the header from it.

| Column | Values / rule |
| :--- | :--- |
| `subject` | |
| `kind` | `run` \| `noise` \| `headshape` \| `anat` |
| `condition`, `run`, `date` | blank where the kind has none (`anat` carries no `date`) |
| `source_path` | edit this to resolve a `review` row |
| `match_method` | `fingerprint` \| `known_override` \| `fast_path` \| `ambiguous` \| `found` \| `not_found` |
| `meg_start_pulse_count`, `behavior_trial_count` | |
| `count_agreement` | `exact` \| `within_tolerance` \| `mismatch` \| `not_applicable` \| `not_checked`; `not_checked` for `ambiguous` rows, which have no candidate to count on |
| `action` | `stage` \| `review`; only `stage` rows are materialized |
| `note` | for `review` rows, every real candidate with its Start-pulse count |
| `fingerprint_error_ms`, `fingerprint_separation` | recorded for every scored row including `fast_path`, so a chronological match is auditable on the same evidence; blank where no score exists, `separation` is `inf` when a duration class held one candidate |

To resolve a `review` row: set `source_path` and `action = stage`, save, re-run
`apply-raw-staging`. It applies the manifest exactly as saved and never
recomputes the plan.

### What each layer stages

| Layer | Rule |
| :--- | :--- |
| MEG | each matched `.ds` via `mne_bids.write_raw_bids`, which also writes `channels.tsv`, `*_meg.json`, `coordsystem.json`, `dataset_description.json`, `participants.tsv`. `*_events.tsv` comes from the trigger channel (`mne.find_events`); CTF's `MarkerFile.mrk` annotations are dropped first (`raw.set_annotations(None)`) to avoid double-counting |
| Empty room | under `sub-emptyroom/ses-<date>/meg/`, cross-referenced via `AssociatedEmptyRoom`; `workflows/sources.py` takes the one nearest each subject's first run by `scans.tsv` acquisition time, which is that link wherever one exists |
| Behavior | `parse_tdms_file(path)` under raw-legal entities; never infers trial classes. Independent of and additive to `derivatives/sub-*/beh/*_beh.tsv`. Staged for every requested subject regardless of MEG match status |
| Head shape | verbatim copy of the native CTF Polhemus export to `BIDS/sub-*/meg/sub-*_headshape.eeg`; BIDS leaves this extension to the digitizer's format |
| Anat | `subjects_dir/<ID>/mri/rawavg.mgz` -> `BIDS/sub-<ID>/anat/sub-<ID>_T1w.nii.gz` via `mne_bids.write_anat`. Lossless: identical shape, bit-identical voxels and affine, `sform_code = 2`. Row is `found`/`not_found`, needs no matching. `subjects_dir` itself is untouched |

Originals on the media and in `tdms/` are never modified. Re-copying passes
`overwrite=True`: per-run digitization varies by sub-mm noise, which is
harmless because coregistration uses a separately-managed `-trans.fif`, never
`coordsystem.json`.

## Stage 1: Behavioral Log Parsing

Input: `BIDS/sub-*/beh/`, not `tdms/`. Requires Stage 0 first; a subject
with no staged behavior raises `FileNotFoundError` naming
`apply-raw-staging`.

Filename contract: `H<subject><Condition><run>_<YYMMDD>.tdms`, `<Condition>`
in `{Slow, Fast, RT}`. Name/duplicate-run guards run in **Stage 0**
(`workflows/raw_staging._matching_tdms_files`), not here.

Output, one table per run:

```text
derivatives/sub-H01/beh/sub-H01_task-tokens_run-1_desc-slow_beh.tsv
```

Columns: `subject`, `condition`, `run`, `source_file`, `nTrialIndex`,
`sTrialClass`, `sTrialClassRaw`, `trial_class_source`, `trial_class_rule`,
`sp_design_correct`, `nChoiceMade`, `nCorrectChoice`, `tGO`,
`tEnterCenter`, `tExitCenter`, `tEnterTarget`, `tTrialEnd`, `sTokenDirs`,
`nTokenNum`, `nTokenDir`, `tTime`, `nProb`, `token_log_rows`,
`token_log_short`, `nOutcome`, `rawRT`, `isCorrect`.

| Column(s) | Rule |
| :--- | :--- |
| `sTrialClass`, `trial_class_source`, `trial_class_rule` | added here (`behavior/classification.py`); rest transcribed |
| `sTrialClassRaw` | unmodified LabVIEW label |
| `sTrialClass` | recorded label; inferred from `sp_design_correct` only for raw `'x'` |
| `infer_random_classes=false` | `'x'` left unclassified, `trial_class_rule="inference_disabled"` (`docs/behavior.md`, Findings) |
| `sp_design_correct` | absent when LabVIEW records no correct target |
| `sTokenDirs` vs `nTokenDir` | designed sequence vs runtime-recorded |
| `sp_design_correct`, `nTokenNum`, `nTokenDir`, `tTime`, `nProb` | deserialized to numeric sequences on load; empty token logs -> empty sequences |
| `tEnterCenter`/`tExitCenter` | center-hold timestamps; `tEnterTarget - tExitCenter` is the only recorded movement duration, ~0 on most trials (`docs/meg.md`, Alignment QC) |
| `nOutcome == 7003` | never-started; requires `tGO==0`, `nChoiceMade==0` |

Validation: sequential `nTrialIndex`; `tGO <= tExitCenter <= tEnterTarget <=
tTrialEnd` on chosen trials; equal-length `nTokenNum`/`nTokenDir`/`tTime`/
`nProb`. Subject IDs normalized to `H01` style.

## Stage 2: Behavioral Metrics Extraction

Input: Stage 1's `derivatives/sub-*/beh/*_beh.tsv`. Eligibility is expressed as
boolean flags on every row, never by deleting it. Subject exclusions
(`ProjectConfig.subject_exclusions`) are applied before any table is built, so
an excluded subject never enters the motor baseline either.

```text
derivatives/sub-group/beh/sub-group_task-tokens_desc-trialfeatures_beh.tsv
derivatives/sub-group/beh/sub-group_task-tokens_desc-summary_beh.tsv
derivatives/sub-group/beh/sub-group_task-tokens_desc-groupstats_beh.tsv
```

### Trial features

`build_trial_features`: one row per staged trial across every subject,
including never-started and no-response trials. MEG join key is `subject`,
`condition`, `run`, one-based `run_trial_index`.

| Column | Rule |
| :--- | :--- |
| `motor_baseline_ms` | subject's arithmetic mean `rawRT` over RT-condition trials with a response; computed once, stamped on every row of that subject |
| `dt_ms` | `rawRT - motor_baseline_ms`, only for started, chosen Fast/Slow trials; `NaN` on RT-condition, never-started and no-response rows |
| `dt_anticipation` | flags negative `dt_ms`, which is retained rather than clipped |
| `logged_spd` | logged chosen-target success probability nearest the motor-corrected commitment time (`tEnterTarget - motor_baseline_ms`), from the parallel `nProb`/`tTime` sequences |
| `logged_spd_validated_15row` | the same value only when `token_log_rows == 15`; otherwise `NaN`, with no fallback |
| `evidence_at_decision` | `logged_spd - 0.5` |
| `sp_design_early`, `sum_log_lr_design_early` | *designed* evidence from `sTokenDirs` and `nCorrectChoice`, always evaluated after the third jump regardless of when the subject responded |
| `sp_design_*_at_decision`, `sum_log_lr_*_at_decision`, `token_lead_at_decision` | same, but using the designed tokens observed by commitment |

A correct-target value outside `{1, 2}`, or a design shorter than three
tokens, yields `NaN`/`pd.NA` rather than a guess.

| Eligibility flag | True when |
| :--- | :--- |
| `is_started` | outcome is not `OUTCOME_NEVER_STARTED` |
| `has_choice` | finite `rawRT` |
| `is_no_response` | started but no choice |
| `primary_analysis_eligible` | started, chosen, Fast/Slow condition -- excludes RT runs and never-started/no-response rows |
| `design_time_alignment_valid` | additionally requires the 15-row token log |

Lapses (started, chosen, not primary-eligible for another reason) are selected
explicitly, never by a flag's absence.

### Subject summary

`summarize_behavior`: one row per subject -- motor baseline; Fast/Slow/RT and
trial-class trial and decision-time counts; percent-correct (0--100) overall
and by condition; paired all-logged/validated-15-row SPD means, overall and by
trial class (`easy`, `ambiguous`, `misleading`, with unclassified trials
excluded from class breakdowns but counted in the overall figures).

Counts follow the eligibility rule named by the column, and two similarly named
counts are deliberately different denominators: `n_fast_trials` counts every
started Fast row including no-response trials; `n_fast_dt_trials` counts only
finite decision times among primary-eligible trials.

### Group statistics

`behavior_group_statistics`: one row per fixed, declared-in-advance contrast --
Fast-vs-Slow decision time, Fast-vs-Slow raw error count, the three pairwise
trial-class decision-time contrasts, and the three class contrasts for each SPD
view. Each row carries a paired t-test and Cohen's dz from
`behavior/math/inference.paired_subject_statistics`, excluding a subject from a
contrast when either value is non-finite. No multiplicity correction; error
counts are raw trial counts, not rates.

## Stage 2b: Behavioral Characterization Analyses

Input: Stage 2's trial-feature and subject-summary tables. The analyses are
listed in `docs/behavior.md` ("Analyses"), implemented under
`behavior/analyses/` and orchestrated by
`workflows/behavior_characterization.py`.

Each analysis writes its own derivative and, where it has one, a matching
`*stats` group-statistics derivative, so one failed model never truncates the
rest of the output:

```text
derivatives/sub-group/beh/sub-group_task-tokens_desc-conditionclass_beh.tsv
derivatives/sub-group/beh/sub-group_task-tokens_desc-conditionclassstats_beh.tsv
```

Every sidecar records the analysis name, the subjects included and excluded,
and both source table paths.

Two analyses join optional inputs:

| Analysis | Optional input | When absent |
| :--- | :--- | :--- |
| `speciescomparison` | `ssmcomparison` | omits `urgency_minus_integrator_bic`, `urgency_growth_rate_per_s`, `urgency_growth_rate_fast_minus_slow` -- each one one-sample test against zero, sharing that table's row shape |
| `individual_profile` (source of `individualprofile`, `individualcorrelations`) | `--neural-metrics`, a TSV/CSV with a `subject` column | those columns are absent, not backfilled |

### Sequential-sampling derivatives

Fitting is a separate step -- `behavior ssm-fit --subjects <ids> --n-jobs <n>`
-- writing per-subject tables at the subject-level path. `behavior
characterization` fits nothing: it concatenates whichever per-subject tables
exist, names any subject missing one in `sequential_sampling_subjects_missing`,
and omits the `ssm*` derivatives when none exists. `behavior subjects` prints
the subject list the two steps agree on. Both models need the optional `pyddm`
dependency (`pip install meg-tokens[modeling]`).

```text
derivatives/sub-H01/beh/sub-H01_task-tokens_desc-ssmcomparison_beh.tsv
```

| Derivative | One row per | Holds |
| :--- | :--- | :--- |
| `ssmcomparison` | subject, condition, model family | log-likelihood, AIC, BIC, criterion difference against the integrator fit, trial count, `n_token_sequences`, and `t_dur_s` -- the integration horizon, so a fit is reconstructible from its row via `_build_model(model, t_dur_s, {parameter: value, ...})` |
| `ssmcomparisonstats` | analysis | `ssm_model_comparison` and `ssm_urgency_condition_contrast`, distinguished by the `analysis` column |
| `ssmtimecourse` | subject, condition, model, trial class, time point (20 ms grid) | `criterion`, `mean_decision_variable`, `predicted_density_correct`/`_error`, `observed_density_correct`/`_error` |
| `ssmtrialpredictions` | trial, model, keyed by `trial_id` | `criterion_at_decision`, `decision_variable_at_decision`, `predicted_accuracy` |

`ssmtimecourse` and `ssmtrialpredictions` are written by `ssm-fit` from the
same fitted models, because refitting to draw a curve costs half an hour.

| Model | Parameters |
| :--- | :--- |
| `ddm`, the bounded integrator | `drift_scale`, `bound`, `nondecision_s` |
| `urgency`, the urgency-gating model | `drift_scale`, `urgency_scale`, `urgency_onset_s`, `nondecision_s` |

Every parameter column is present in the table and missing where a model does
not use it.

A subject-condition cell with fewer than 50 scorable trials is written with
`converged=false` and missing estimates. Anticipations (non-positive `dt_ms`)
are dropped rather than flagged -- the one place in the pipeline a trial is
deleted.
