# MEG Tokens Data Contract

This project targets behavioral replication of the legacy scripts using modern
MNE-Python, BIDS/MNE-BIDS-style derivatives, and explicit analysis tensors.

## Core Rules

- Analysis code must use real project inputs only; nothing generates fake
  MEG, source, or behavior data.
- Missing inputs raise a clear error naming the required file.
- Modern MNE/scientific-Python APIs are preferred over step-by-step ports
  when they preserve the same scientific behavior.

## Behavioral module boundaries

Behavioral code follows one dependency direction:

```text
project I/O → behavior schema/tables → trial features → analyses → workflows
```

`behavior/schema.py` defines and validates table contracts, `behavior/tdms.py`
interprets TDMS source records only, and `behavior/tables.py` composes the
generic project table loader with the schema. Later stages
(`behavior/features.py`, `behavior/math/`, `behavior/analyses/`) build on top
of this without feeding back into it. Workflows locate inputs, call these
layers, and write outputs; they never implement scientific formulas
themselves.

## Storage

- Raw and cleaned MEG data uses MNE-native formats under a BIDS derivatives
  root.
- Parsed behavior runs are `.tsv` tables under
  `derivatives/sub-<ID>/beh/` with JSON sidecars.
- No `.mat`, `.h5`, or undocumented pickle files.

## Filename (`desc`) Convention

Output filenames follow BIDS-Derivatives entity ordering
(`sub-ses-task-acq-run-proc-space-desc-suffix`, see
`meg_tokens/io/contract.py`), but `desc` is a hyphen-joined, growing tag list
(e.g. `desc-slow_beh`) rather than a single value. Every value chained into
`desc` is also recorded verbatim in the JSON sidecar's `metadata` -- the two
must never disagree. This is a deliberate, project-wide convention:
changing it is a breaking change across every `DerivativeLayout` path
builder, not a local fix.

## Stage 0: Raw BIDSification

Raw CTF media has no condition/run label in its session names
(`H02_DDM-tthiery_20180213_03.ds`) -- only a subject prefix, date, and
sequential index. `meg-tokens meg stage-raw` matches each session to a
behavioral run and writes a reviewable manifest; `meg-tokens meg
apply-raw-staging` reads that manifest (fresh or hand-edited) and copies
matched data into:

```text
BIDS/sub-H02/meg/sub-H02_task-tokens_acq-slow_run-1_meg.ds
BIDS/sub-H02/beh/sub-H02_task-tokens_acq-slow_run-1_beh.tsv
BIDS/sub-H02/anat/sub-H02_T1w.nii.gz
BIDS/sub-emptyroom/ses-20180213/meg/sub-emptyroom_ses-20180213_task-noise_meg.ds
```

`stage-raw` and `apply-raw-staging` are the only two entry points; there is
no separate command per data type.

Matching never guesses. Nominal trial duration pre-filters candidates into
protocol classes (315s Slow/Fast, 135s RT, everything else excluded), and
within a class a run is identified by its **inter-trial-interval
fingerprint**: the logged per-trial `nInitialTime` gaps and the real MEG
trial-start pulse gaps are the same physical intervals measured on two
unrelated clocks, so the correct pairing reproduces them to well under a
millisecond while any other candidate is off by hundreds. A match is
accepted only when its mean absolute error is within
`fingerprint_max_error_ms` (5ms) *and* beats the runner-up by at least
`fingerprint_min_separation` (20x) *and* no other run claims the same
session; on the real dataset correct matches score 0.39-0.57ms at
144-892x, so these thresholds sit in an empty gap rather than on a
boundary. Runs the fingerprint declines fall back to `KNOWN_SESSION_OVERRIDES`
(H01/H05, kept as a cross-check -- a fingerprint that contradicts one is
flagged for review, never staged), then to chronological 1:1 pairing when
the remaining candidate and run counts are equal. Anything still unresolved
is flagged `ambiguous` with each candidate's pulse count and fingerprint
error as evidence, for hand-resolution in the manifest. Every accepted pair
is additionally cross-checked against the real trigger-pulse count. Full
design and evidence: `docs/meg_t0_7_raw_bidsification_plan.md`.

Because `desc` is derivatives-only in real BIDS, condition uses `acq`
instead (`Slow2` -> `acq-slow_run-2`); `task` stays singular (`tokens`); no
`ses` entity (one acquisition date per subject).

**Manifest** (`derivatives/sub-group/meg/sub-group_task-tokens_desc-rawstaging_manifest.tsv`,
via `DerivativeLayout.raw_staging_manifest()`) is the complete Stage 0 plan
for a subject: every file that will be staged, of every kind, plus every
gap. One review pass covers the whole raw layer. The columns are
`MatchResult`'s fields in its declared order -- the dataclass is the single
definition, and the workflow derives the header from it: `subject, kind
(run|noise|headshape|anat), condition, run, date, source_path, match_method
(fingerprint|known_override|fast_path|ambiguous|found|not_found),
meg_start_pulse_count, behavior_trial_count, count_agreement
(exact|within_tolerance|mismatch|not_applicable|not_checked), action
(stage|review), note, fingerprint_error_ms, fingerprint_separation`.
`count_agreement` is `not_checked` for `ambiguous` rows (no candidate
session picked yet, so nothing to count pulses on). The two `fingerprint_*`
columns are recorded for every scored row, `fast_path` ones included, so a
chronologically-assigned match is auditable on the same evidence as a
fingerprinted one; they are blank where no score exists and
`fingerprint_separation` is `inf` when a duration class held a single
candidate. Only
`action == "stage"` rows are materialized. To resolve a
`review` row: its `note` lists every real candidate with its Start-pulse
count -- set `source_path` and `action = stage`, save, and re-run
`apply-raw-staging` (it applies the manifest exactly as saved, never
recomputing the plan).

**MEG raw layer:** each matched `.ds` session is copied via
`mne_bids.write_raw_bids`, which also generates `channels.tsv`,
`*_meg.json`, `coordsystem.json`, `dataset_description.json`, and
`participants.tsv`. `*_events.tsv` comes from the real trigger channel
(`mne.find_events`); CTF's `MarkerFile.mrk` annotations are dropped first
(`raw.set_annotations(None)`) to avoid double-counting. Empty-room noise is
copied under `sub-emptyroom/ses-<date>/meg/` and cross-referenced via
`AssociatedEmptyRoom`. Originals on the media and in `tdms/` are never
modified. Re-copying always passes `overwrite=True` (real per-run
digitization varies by sub-mm noise; harmless since coregistration uses a
separately-managed `-trans.fif`, never `coordsystem.json`).
`meg_tokens/workflows/sources.py`'s empty-room lookup (`project.noise_dir`)
is unchanged by Stage 0 -- the `sub-emptyroom` tree doesn't feed it yet.

**Behavior raw layer:** `BIDS/sub-<ID>/beh/*_beh.tsv` is a minimally-parsed
TDMS export (`parse_tdms_file(path, infer_random_classes=False)`) under
raw-legal BIDS entities, independent of and additive to
`derivatives/sub-*/beh/*_beh.tsv` (the `behavior ingest` output,
which already makes real analysis choices like trial-class inference, so
stays a derivative). Staged for every requested subject's TDMS files
regardless of MEG match status.

**Anatomical (MRI) raw layer:** an `anat` manifest row per subject, copying
`subjects_dir/<ID>/mri/rawavg.mgz` -- native per-subject voxel
geometry/intensity, unlike `mri/T1.mgz`, which every subject shares as a
uniformly resampled 256x256x256 8-bit conformed volume -- into
`BIDS/sub-<ID>/anat/sub-<ID>_T1w.nii.gz` via `mne_bids.write_anat`. The
`.mgz`→`.nii.gz` conversion is lossless: identical shape, bit-identical
voxel data and affine, `sform_code = 2` (SCANNER_ANAT); only the dtype's
byte order is normalised.

Anat needs no matching -- there is at most one reconstruction per subject,
so discovery *is* the decision, and the row is `found`/`not_found` exactly
like `noise` and `headshape`. It is *not* a separate command: a subject
without a reconstruction (H07/H10) surfaces as a `not_found` / `review` row
in the same list as every other gap, rather than in a second command's
separate output. The `anat` row carries no `date`, since the MRI is a
separate acquisition from the MEG session. `subjects_dir` itself is
untouched and still feeds the BEM/source-space stages directly, which use
the conformed volumes.

## Stage 1: Behavioral Log Parsing

Every `*.tdms` file under a subject's `behavior_root` must match
`H<subject><Condition><run>_<YYMMDD>.tdms` (e.g. `H01Slow1_180131.tdms`),
where `<Condition>` is `Slow`, `Fast`, or `RT`. A non-matching filename
raises `ValueError` unless listed in `behavior_ignore_files` (add only
after confirming by hand it's not a real run, with the reason recorded in
the config comment). Two files resolving to the same `(subject, condition,
run)` also raise `ValueError` rather than one silently overwriting the
other.

Stage 1 writes one table per TDMS run:

```text
derivatives/sub-H01/beh/sub-H01_task-tokens_run-1_desc-slow_beh.tsv
```

Required trial columns: `subject`, `condition`, `run`, `source_file`,
`nTrialIndex`, `sTrialClass`, `sTrialClassRaw`, `trial_class_source`,
`trial_class_rule`, `sp_design_correct`, `nChoiceMade`, `nCorrectChoice`,
`tGO`, `tEnterCenter`, `tExitCenter`, `tEnterTarget`, `tTrialEnd`,
`sTokenDirs`, `nTokenNum`, `nTokenDir`, `tTime`, `nProb`, `token_log_rows`,
`token_log_short`, `nOutcome`, `rawRT`, `isCorrect`.

Subject labels are normalized to `H01` style. The parser validates
sequential trial indices, event ordering, and equal lengths across the
per-token `nTokenNum`/`nTokenDir`/`tTime`/`nProb` arrays before writing.
`sTokenDirs` is the trial-level designed sequence; `nTokenDir` independently
holds the runtime-recorded directions. `sTrialClassRaw` preserves the
LabVIEW label; `sTrialClass` keeps recorded designed labels and is inferred
only for raw `'x'` trials from `sp_design_correct` (provenance in
`trial_class_source`/`trial_class_rule`). Inference is on by default;
`infer_random_classes = false` leaves `'x'` trials unclassified
(`trial_class_rule = "inference_disabled"`; see
`docs/behavior_t0_1_nprob_trial_class.md` § 3b for why this is a real
analysis choice). `sp_design_correct` is unavailable when LabVIEW records
no correct target. The Stage 1 table loader deserializes
`sp_design_correct`, `nTokenNum`, `nTokenDir`, `tTime`, `nProb` into numeric
sequences before validation/analysis -- downstream code never parses
sequence-valued cells itself; empty runtime token logs are empty sequences.
`tEnterCenter`/`tExitCenter` are the center-hold timestamps, retained
because `tEnterTarget - tExitCenter` is the only recorded movement
duration; LabVIEW writes both from the same event, so this duration is zero
on essentially every trial (`docs/behavior_qc_report.md` § 1). Event
ordering is validated as `tGO <= tExitCenter <= tEnterTarget <= tTrialEnd`
on chosen trials. Rows with `nOutcome == 7003` are retained for provenance
but must have `tGO == 0` and `nChoiceMade == 0` (never started). Derivatives
written before these fields existed must be re-ingested.

## Path Convention

Use `meg_tokens.io.derivative_path` for BIDS-derivatives-style paths:

```python
from meg_tokens.io import derivative_path

path = derivative_path(
    "/data/tokens-bids", subject="H01", datatype="beh",
    task="tokens", run="1", description="slow", suffix="beh", extension=".tsv",
)
```

This yields
`/data/tokens-bids/derivatives/sub-H01/beh/sub-H01_task-tokens_run-1_desc-slow_beh.tsv`
(`datatype` defaults to `"meg"` when omitted).
