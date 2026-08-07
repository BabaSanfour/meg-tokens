# T0-7: Raw BIDSification Plan (media -> BIDS/sub-*/meg + BIDS/sub-*/beh)

**Status:** Implemented (`meg_tokens/meg/raw_staging.py`,
`meg_tokens/meg/bids_raw.py`, `meg_tokens/behavior/tdms_bids.py`,
`meg_tokens/workflows/raw_staging.py`, `meg-tokens meg stage-raw` /
`meg-tokens meg apply-raw-staging`), verified end-to-end against the real
mounted dataset. Deviations from this plan as originally written, all by
explicit direction during or after implementation:

- No `raw_media_root` config field was added. `meg-tokens meg stage-raw`
  defaults to the existing `ProjectConfig.raw_meg_root` (`data_root/raw`)
  as the media root, overridable per-call with `--media-root`.
- **Real copies, not symlinks.** An earlier revision of this plan symlinked
  the `.ds` directory (since `mne-bids`'s `write_raw_bids(...,
  symlink=True)` only supports FIFF, always physically copying CTF data
  otherwise) to avoid duplicating data off the media. That was reversed:
  `write_meg_bids`/`write_emptyroom_bids` now call
  `mne_bids.write_raw_bids(..., symlink=False)` directly for a real copy
  (originals on the media and in `tdms/` untouched), which also means the
  hand-rolled BIDS-MEG sidecar writer this plan originally called for was
  deleted -- `write_raw_bids` generates `channels.tsv`/`*_meg.json`/
  `coordsystem.json`/`dataset_description.json`/`participants.tsv` itself.
  Real ~500MB-per-session copies mean the full 32-subject dataset needs
  roughly 160GB of free disk space.
- **No scoring-based disambiguation.** The original design used a small DP
  to auto-pick the best-fitting candidate session by trigger-pulse count
  whenever a duration class's candidate count didn't equal its run count
  (2 of 32 real subjects). That is a guess, even gated behind manual
  review before staging, so it was removed: `match_subject` now reports
  every run in an ambiguous class as `match_method="ambiguous"` with every
  real candidate's trigger-pulse count listed as evidence, and never
  assigns a `media_path` automatically. See "Open questions" for the two
  real subjects this affects and the legacy evidence found for one of
  them.
- **Stage 0 no longer depends on Stage 1.** `load_behavior_runs` originally
  read `derivatives/meg-tokens/sub-*/beh/*_beh.tsv` (`behavior ingest`'s
  output). It now parses TDMS directly
  (`meg_tokens.behavior.tdms.parse_tdms_file`, the same low-level parser
  `write_beh_bids` uses), so Stage 0 has no dependency on any other stage
  having run first, consistent with being "Stage 0."

## Context

`meg_tokens` currently has no code path that turns the raw CTF acquisition
data on external media into a real raw layer. Every downstream command
(`meg preprocess --raw-path ...`) takes an explicit, already-known raw file
path -- something a human has always resolved by hand until now (evidenced
by the legacy notebook's per-subject header comments like `05.ds = Slow1
117`). And `BIDS/` today only ever means `BIDS/derivatives/meg-tokens/...`;
there is no `BIDS/sub-*/meg/` or `BIDS/sub-*/beh/` raw layer at all, even
though `docs/data_contract.md`/`docs/refactor/architecture.md` describe this
as a BIDS/MNE-BIDS-style project and explicitly permit delegating to
`mne_bids.BIDSPath`.

This plan adds that missing raw layer, built as real BIDS raw data (not just
BIDS-*flavored* naming, which is what `derivatives/` already does):

- `BIDS/sub-<ID>/meg/...` -- the raw MEG, via `mne_bids.write_raw_bids`
  (real copies -- see "Status" above for why this plan's original
  no-copy/symlink intent was reversed).
- `BIDS/sub-<ID>/beh/...` -- the raw behavioral log, minimally parsed from
  TDMS (no trial-class inference, no analysis-derived columns), separate
  from and upstream of the existing `derivatives/meg-tokens/sub-*/beh/`
  output, which stays exactly as it is today (it already makes real
  analysis choices -- e.g. `infer_random_classes` -- so it is correctly a
  derivative, not raw).

`data_root/raw` (a plain sibling folder) is **not** part of this plan -- the
raw layer is `BIDS/sub-*/meg/`, nothing else.

Matching a numbered raw session (`H02_DDM-tthiery_20180213_03.ds`) to a
condition/run (`Slow1`) is the data-critical core of this: a wrong match
silently corrupts a subject's whole neural/behavioral correspondence. The
design commits to matching with a verifiable signal wherever a fast,
unambiguous path isn't available, always recording a reviewable manifest,
and never forcing a match through.

## Evidence gathered against the real, currently-mounted dataset

- `data_root = /home/karim/Data/meg-tokens`. `tdms/` and
  `BIDS/derivatives/meg-tokens/.../beh/*.tsv` are already fully populated
  for all 32 subjects (behavior already ingested for real). `BIDS/sub-*/`
  does not exist yet.
- Media: `/media/karim/Hamza/DDM-tthiery/<Subject|Pilot01>_DDM-tthiery_<YYYYMMDD>_<NN>.ds`,
  `NOISE_noise_<YYYYMMDD>_01.ds`, `<Subject>_DDM-tthiery_<YYYYMMDD>_HEADSHAPE.eeg`
  (+ fiducial `.jpg`s, ignored). `H01` is stored as `Pilot01`. Each subject
  has exactly one acquisition date (spot-checked H02/H10/H12/H20/H32).
- Each `.ds`'s `.hist` file has a `Trial duration: <seconds>` line -- the
  **nominal/configured** duration, unaffected by mid-recording truncation
  (confirmed against the one known truncated file,
  `H05_DDM-tthiery_20180220_02.ds`, 138s actual vs 315s nominal per
  `docs/meg_t0_6_subject_exclusion_qc.md`) -- plus a `DATE:` collection
  timestamp and `Sample rate: 1200`.
- Scanning all 412 real subject sessions' nominal durations gives exactly
  three clusters, confirming `scripts/qc/meg_session_qc.py`'s existing
  `SLOWFAST_TRIAL_DURATION = 315.0`:
  - `315` (259 sessions) = Slow/Fast candidates (32 x 8 = 256 expected;
    **H01 has 10, H05 has 9** -- the only two subjects with extra
    candidates).
  - `135` (63 sessions) = RT candidates (32 x 2 = 64 expected; **H05 has
    only 1** -- the only subject short one).
  - `360`/`420` (90 sessions) = other protocols, never part of the 10
    canonical runs, always excluded from matching.
  - 30 of 32 subjects have exactly the expected 8-SF/2-RT candidate counts
    with zero ambiguity.
- The already-ingested behavior TSVs carry `nInitialTime` per trial. Its
  per-run minimum gives a clean, gapless chronological order across a
  subject's *entire* session (verified on H02: RT1, Slow1, Fast1, Slow2,
  Fast2, Slow3, Fast3, Slow4, Fast4, RT2 -- strict alternation, RT
  bookending the block, no clock resets).
- Reading a raw session's Start-pulse count (`mne.find_events`, event code
  `262144` per `meg_tokens.meg.epoching.DEFAULT_EVENT_IDS`, respecting
  `SUBJECT_EVENT_OVERRIDES`) takes ~2s per session and, spot-checked on
  `H02_..._03.ds` vs the TDMS `Slow1` run, gives an exact match (56 Start
  pulses vs 56 started trials from `meg_tokens.behavior.tdms.started_trials`).
  This is the cross-validation signal for the two ambiguous subjects, and
  for every match's audit record. ~320 sessions total is ~10 minutes of I/O.
- `mne_bids` is **not currently installed or a declared dependency**
  (`pyproject.toml` lists `mne>=1.0.0` but no `mne-bids`). It needs to be
  added. Modern `mne-bids` (>=0.12) supports `write_raw_bids(...,
  symlink=True)`, which is exactly the no-copy behavior we want; the
  implementation phase must confirm this against whatever version resolves,
  with a documented fallback (see "Risks" below) if `.ds` directories don't
  symlink cleanly in practice.

## Design

### 1. Matching algorithm (`meg_tokens/meg/raw_staging.py`, new)

Unchanged by the BIDS-location decision -- this module only decides *which*
media session is *which* run; it doesn't know where things get written.

Per subject:

1. Parse the media root for that subject's numbered `.ds` sessions
   (`H01`<->`Pilot01`), reading `.hist` for nominal duration + start
   timestamp. Sort chronologically (`.hist` `DATE:` timestamp; the numeric
   filename suffix is the same order and used only as a tie-break check).
2. Load that subject's already-ingested behavior runs (via
   `DerivativeLayout.find_behavior`-discoverable Stage-1 TSVs, or TDMS
   filenames directly via `meg_tokens.behavior.tdms.FILENAME_RE`), each
   tagged with condition/run/`nInitialTime` range/started-trial count
   (`started_trials`). Sort by `nInitialTime.min()`.
3. Bucket raw sessions into `SLOWFAST` (~315s), `RT` (~135s), or `OTHER`
   (excluded, listed for information only). Bucket behavior runs the same
   way by condition.
4. **Fast path** (per class, per subject): candidate count == expected-run
   count -> pair 1:1 in chronological order directly. Covers 30/32
   subjects.
5. **Disambiguation path** (only H01's SLOWFAST class and H05's RT class
   today, but handled generally): open every candidate in that class, count
   Start pulses, and choose the order-preserving assignment of candidates
   to runs that best matches each run's started-trial count (small DP,
   <=10 candidates x <=8 runs). A run left without an adequate candidate is
   reported `unmatched`, never guessed.
6. For every accepted match, record its real Start-pulse count next to the
   behavior run's started-trial count -- the manifest is auditable for
   every subject, not just the disambiguated ones.
7. Noise: look for `NOISE_noise_<date>_01.ds` (exact legacy pattern) on the
   subject's date. Present or absent, recorded as its own manifest row;
   never guesses among the dataset's other `NOISE*` variants.
8. Headshape: look for `<prefix>_DDM-tthiery_<date>_HEADSHAPE.eeg`.

Reuses `meg_tokens.core.normalize_subject_id`,
`meg_tokens.meg.epoching.get_event_id`,
`meg_tokens.behavior.tdms.started_trials`/`FILENAME_RE`, and the
`.hist`-parsing approach already proven in `scripts/qc/meg_session_qc.py`
(lifted into the package so QC scripts and the pipeline share one
implementation).

### 2. BIDS entities

Real BIDS raw filenames cannot use `desc-` (derivatives-only per the BIDS
spec), so condition needs a different raw-legal entity. Chosen mapping,
stated here for confirmation since it is hard to rename later without
breaking `mne_bids.BIDSPath` lookups everywhere downstream:

- `task-tokens` stays singular (it is one task with a speed-instruction
  manipulation, not three different tasks).
- Condition -> `acq-<slow|fast|rt>`.
- Run-within-condition -> `run-<N>` (same `N` already used everywhere in
  `derivatives/`, e.g. `Slow2` -> `acq-slow_run-2`; this keeps the
  condition/run identity model identical to what every other stage already
  uses, just re-expressed as raw-legal entities).
- No `ses-` (single session per subject in this dataset).

Concretely: `BIDS/sub-H02/meg/sub-H02_task-tokens_acq-slow_run-2_meg.ds`,
`BIDS/sub-H02/beh/sub-H02_task-tokens_acq-slow_run-2_beh.tsv`.

### 3. MEG raw layer (`meg_tokens/meg/bids_raw.py`)

`write_meg_bids(raw_path, *, subject, condition, run, bids_root, task,
empty_room_bids_path=None, overwrite=False) -> BIDSPath`:

- Reads the matched session with `mne.io.read_raw_ctf` and its real trigger
  events with `mne.find_events`.
- Drops `raw.annotations` (`raw.set_annotations(None)`) before writing:
  `mne.io.read_raw_ctf` auto-populates annotations from CTF's own
  `MarkerFile.mrk` (a Start/Go-only subset of the full trigger record), and
  `write_raw_bids` writes the *union* of `events` and `raw.annotations` --
  without dropping the annotations first, every real trigger pulse got
  written twice.
- Calls `mne_bids.write_raw_bids(raw, bids_path=BIDSPath(subject=...,
  task=task, acquisition=condition.lower(), run=str(run), datatype="meg",
  root=bids_root), events=..., event_id=..., empty_room=...,
  symlink=False, overwrite=...)` -- a real copy. `write_raw_bids` generates
  the standard sidecars (`channels.tsv`, `*_meg.json`, `coordsystem.json`,
  `dataset_description.json`, `participants.tsv`) and `*_events.tsv`
  itself; nothing is hand-rolled.
- `overwrite=True` at the call site in `apply_raw_staging` (not the
  function's own default of `False`): BIDS's `coordsystem.json` is
  subject+acquisition-scoped, not per-run, but real CTF head digitization
  varies by a sub-millimeter, run-to-run fitting noise (confirmed against
  real data), so `write_raw_bids` refuses on the second run of the same
  `acq` unless told to overwrite. Harmless -- this project's own
  coregistration uses a separately-managed `-trans.fif`, never
  `coordsystem.json`.
- Empty-room noise is written the same way to
  `BIDS/sub-emptyroom/ses-<date>/meg/sub-emptyroom_ses-<date>_task-noise_meg.ds`
  (the standard BIDS empty-room convention -- one real recording per
  acquisition date, which in this dataset is effectively one per subject)
  and cross-referenced via `write_raw_bids`'s `empty_room=` argument on
  each subject's own MEG writes.
- Headshape: converted (a real format conversion, not a raw copy) via the
  existing but previously-unwired
  `meg_tokens.meg.preprocessing.convert_ctf_headshape_to_pos`, written
  under `BIDS/sub-<ID>/meg/sub-<ID>_headshape.pos` (BIDS has no standard
  slot for a bare digitization file outside a raw's own metadata; keeping
  it named/located predictably next to that subject's raw MEG is enough
  for the coregistration step to find it later).

**Scope boundary, stated explicitly:** `meg_tokens/workflows/sources.py`'s
existing empty-room lookup (`project.noise_dir` +
`DerivativeLayout.find_noise`) is **not** changed by this plan -- that
would touch the already-tested source-reconstruction path. The BIDS
`sub-emptyroom` tree is written correctly regardless; pointing
`compute_noise_covariance` at it instead of `noise_dir` is a reasonable
near-term follow-up, not part of this change.

### 4. Behavior raw layer (`meg_tokens/behavior/tdms_bids.py`, new, additive-only)

`write_beh_bids(tdms_path, *, bids_root) -> Path`:

- Reuses the existing, already-tested
  `meg_tokens.behavior.tdms.parse_tdms_file(path, infer_random_classes=False)`
  (the one existing switch that already distinguishes "what LabVIEW
  logged" from "what we inferred") and `parse_tdms_filename`
  (subject/condition/run/date, already exists) to build entities.
- Writes the resulting per-trial table via the existing
  `meg_tokens.io.save_table` to
  `BIDS/sub-<ID>/beh/sub-<ID>_task-tokens_acq-<condition>_run-<N>_beh.tsv`
  with a JSON sidecar (`"stage": "raw_bids"`).
- **Does not touch** `meg_tokens/workflows/behavior.py`'s
  `ingest_behavior`/`analyze_behavior` or their
  `derivatives/meg-tokens/sub-*/beh/*_beh.tsv` output -- that pipeline is
  already validated (`docs/behavior_qc_report.md`,
  `docs/behavior_t0_1_nprob_trial_class.md`, extensive tests) and keeps
  reading straight from `tdms/` exactly as it does today. This is a
  parallel, additive raw-BIDS export, not a refactor of the existing
  ingester. Unifying them (derivatives ingestion reading from this
  raw-BIDS table instead of re-parsing TDMS) is a reasonable later
  cleanup, not part of this change.

### 5. Manifest

One TSV per staging run via a new `DerivativeLayout.raw_staging_manifest()`
path helper (`sub-group/meg/..._desc-rawstaging_manifest.tsv` under
`derivatives/meg-tokens`, following the existing group-manifest convention,
e.g. `find_source_manifest`), with a JSON sidecar recording the media root
and tolerances used. Columns: `subject, kind (run|noise|headshape),
condition, run, media_path, match_method (fast_path|disambiguated|unmatched),
meg_start_pulse_count, behavior_started_trial_count, count_agreement
(exact|within_tolerance|mismatch|not_applicable), bids_path, action
(stage|review)`. Rows needing a human look (`unmatched`, `mismatch`, no
noise/headshape found) get `action=review` and are never staged
automatically. A human can hand-edit `action` after checking a flagged row,
then re-apply from that file.

### 6. Config and CLI

**As implemented** (see "Status" note at the top for why this differs from
the original plan): no `raw_media_root` field was added and
`ProjectConfig.raw_meg_root` was kept, not removed -- `stage-raw` defaults
its media root to `project.raw_meg_root`, overridable with `--media-root`.

- `meg_tokens/core/settings.py`: `RawStagingConfig` (duration tolerance,
  RT/SLOWFAST nominal durations defaulting to 135.0/315.0, count-agreement
  tolerance), following the existing frozen-dataclass + `__post_init__`
  validation pattern every other `*Config` uses.
- `meg_tokens/workflows/raw_staging.py`: `plan_raw_staging(project, *,
  subjects, media_root=None, settings=RawStagingConfig()) ->
  WorkflowResult` (always writes the manifest + sidecar, otherwise
  read-only) and `apply_raw_staging(project, *, manifest_path=None,
  subjects=None) -> WorkflowResult` (reads a manifest -- freshly produced
  or hand-edited -- and calls `write_meg_bids`/`write_emptyroom_bids`/
  `write_beh_bids` for `action=stage` rows).
- `meg_tokens/cli/main.py`: `meg-tokens meg stage-raw` (`--subjects`,
  `--media-root`) runs `plan_raw_staging` and prints a per-subject
  action summary plus flagged rows, mirroring `behavior qc`'s
  print-and-return-0 style. `meg-tokens meg apply-raw-staging`
  (`--subjects`, `--manifest`) is a **separate** subcommand rather than a
  `--apply` flag on `stage-raw`, so applying a hand-edited manifest never
  triggers an unwanted re-plan that would overwrite those edits.
- `meg_tokens/meg/__init__.py` and `meg_tokens/behavior/__init__.py`:
  lazy-export the new public functions, matching the existing `_EXPORTS`
  dict pattern.
- `pyproject.toml`: added `mne-bids` to `dependencies` (used only for
  `BIDSPath`, `make_dataset_description`, and the pure per-channel
  `coil_type` classifier -- not for `write_raw_bids`'s data path).

### 7. Docs

- `docs/data_contract.md`: new "Stage 0: Raw BIDSification" section
  (manifest schema, BIDS entity mapping, materialization contract).
- `README.md`: new stage-0 row in the pipeline table; update "Data
  Locations" to describe `BIDS/sub-*/meg/` + `BIDS/sub-*/beh/` as the raw
  layer (replacing the current "raw/" folder description) and
  `derivatives/meg-tokens/...` as unchanged.
- `config/tokens.toml.template`: document that `stage-raw` reads media from
  `raw_meg_root` (`data_root/raw`) by default and clarify `raw/` is not a
  project-managed folder.

## Risks / things to confirm during implementation

- **Resolved during implementation, then reversed:** `mne-bids` 0.17.0's
  `write_raw_bids(..., symlink=True)` raises `NotImplementedError` for any
  non-FIFF format -- for CTF `.ds` it always physically copies the whole
  session directory (`copyfile_ctf`, confirmed ~500MB for one real
  session, ~160GB across ~320 sessions). An earlier revision worked around
  this by hand-symlinking the `.ds` directory and hand-writing the BIDS-MEG
  sidecars, to avoid copying. Per later, explicit direction, that was
  reversed: `write_meg_bids` now calls `write_raw_bids(...,
  symlink=False)` directly for a real copy, and the hand-rolled sidecar
  writer was deleted. Real copies mean ~160GB of free disk space is needed
  to stage the full 32-subject dataset.
- **`participants.tsv`/`dataset_description.json` churn.** `mne_bids`
  creates/updates these on first write; multiple subjects writing
  concurrently isn't a concern here (staging is sequential per subject).
  Repeated `apply-raw-staging` runs pass `overwrite=True` (see section 3)
  so re-copying an already-staged run is safe, at the cost of always
  re-copying the full session rather than detecting "already correct and
  unchanged."

## Files touched

- New: `meg_tokens/meg/raw_staging.py`, `meg_tokens/meg/bids_raw.py`,
  `meg_tokens/behavior/tdms_bids.py`, `meg_tokens/workflows/raw_staging.py`,
  `tests/test_meg_raw_staging.py`, `tests/test_bids_raw.py`,
  `tests/test_tdms_bids.py`, `tests/test_raw_staging_workflow.py`
- Edit: `meg_tokens/core/config.py`, `meg_tokens/core/settings.py`,
  `meg_tokens/io/layout.py` (add `raw_staging_manifest()`),
  `meg_tokens/meg/__init__.py`, `meg_tokens/behavior/__init__.py`,
  `meg_tokens/cli/main.py`, `pyproject.toml`, `config/tokens.toml.template`,
  `docs/data_contract.md`, `README.md`
- Removed: `ProjectConfig.raw_meg_root` (and its one test assertion in
  `tests/test_core.py`)

## Testing

Unit tests build synthetic `.ds`-like directories (a `.hist` file with
controlled `Trial duration:`/`DATE:` text -- no real MEG binary data needed
for bucketing/ordering logic) and synthetic behavior-run tables to
exercise: fast-path matching, the H01-shape case (extra same-class
candidate), the H05-shape case (missing candidate, must report `unmatched`
not guess), noise/headshape discovery, and manifest read/apply
round-tripping with a hand-edited `action` column -- all without touching
the real 500GB media, so CI does not depend on the drive being mounted.
`write_meg_bids` is tested against a tiny synthetic `mne.io.RawArray` with a
synthetic STIM channel (consistent with how `tests/test_meg_preprocessing.py`
already builds synthetic MNE objects), asserting the resulting `BIDSPath`
round-trips through `mne_bids.read_raw_bids`. `write_beh_bids` is tested
against a small real-shaped synthetic `.tdms`-parsed DataFrame fixture
(already available via existing TDMS test fixtures in
`tests/test_tdms_parser.py`).

Real-data verification (manual, run once the code exists, since it needs
the mounted drive):

```bash
meg-tokens --config <your tokens.toml> meg stage-raw --subjects H01 H02 H05 H10
```

Confirm the printed summary and manifest show H02/H10 fast-pathed, H01/H05
flagged `ambiguous` (see "H01/H05 resolution" below for how to fill those
in by hand), then run
`meg-tokens meg apply-raw-staging --subjects H01 H02 H05 H10` and confirm
`BIDS/sub-H02/meg/sub-H02_task-tokens_acq-slow_run-2_meg.ds` resolves via
`mne_bids.read_raw_bids`, `BIDS/sub-H02/beh/..._beh.tsv` has the expected
trial rows, and `meg-tokens meg preprocess --raw-path
.../BIDS/sub-H02/meg/..._meg.ds ...` runs unchanged.

Plus the standard phase-gate: `python -m pytest -q`,
`python -m compileall -q meg_tokens`.

## H01/H05 resolution: legacy evidence

The original researcher's own conversion notebook,
[`55_CTF_to_FIF.ipynb`](../archive/replicated/DDM_scripts/scripts_new/55_CTF_to_FIF.ipynb)
(cell 6), gives an explicit, complete, session-by-session mapping for
`Pilot01` (`H01`):

```text
01.ds=RT1  05.ds=Slow1  06.ds=Fast1  07.ds=Slow2  08.ds=Fast2
09.ds=Slow3  10.ds=Fast3  11.ds=Slow4  12.ds=Fast4  13.ds=RT2
```

This resolves H01's `Slow3` row exactly (`09.ds`) and confirms sessions
`02`/`03`/`04` are the extra SLOWFAST-duration candidates our own matching
correctly refuses to guess among (Thomas's own pipeline skips them too --
they're absent from his `filenames` list entirely). Cell 1's per-file
trial counts (e.g. `09.ds = Slow3 116`) also line up with the real
Start-pulse counts our matcher reports for that session. No equivalent
per-session listing exists in the archive for `H05` or any other subject
(cells 2-4 cover the general subject group by date only, not by session
number) -- searched `grep -rn "H05.*\.ds"` and every `*.ds =` comment across
`archive/replicated/`.

H05's case is structurally different from H01's, and not resolvable the
same way: the real media only has **one** 135s-duration (RT) session for
H05 (`{'135': 1, '315': 9, '360': 5}`), not two. One of RT1/RT2 has no
raw MEG counterpart on the media at all -- a genuine recording gap, not an
ambiguity our matcher failed to resolve. The one real RT session's
position (before or after the Slow/Fast block, matching the
RT-brackets-the-block pattern confirmed for every subject checked so far)
would say which of RT1/RT2 it is, but that's inferred from *other*
subjects' patterns, not evidence specific to H05 -- deliberately not
applied automatically here.

Whether to hard-code the H01 mapping (matching how
`meg_tokens/meg/epoching.py`'s `KNOWN_TRAILING_TRIAL_MISMATCHES`/
`SUBJECT_EVENT_OVERRIDES` already document verified, subject-specific
dataset exceptions directly in code) or leave it to manual manifest editing
is an open call -- see "Open questions" below.

## Open questions -- resolved during implementation

1. The `acq-<condition>_run-<N>` entity mapping (Section 2) was used as
   planned.
2. `mne-bids` was added as a hard dependency -- it's used only for
   `BIDSPath`/`make_dataset_description`/the pure `coil_type` classifier,
   all lightweight and dependency-light beyond `mne` itself, which is
   already a hard dependency.
3. Confirm the empty-room-in-`sub-emptyroom`/`compute_noise_covariance`
   scope boundary (Section 3) is acceptable for now, given it leaves two
   parallel noise conventions in place (`sub-emptyroom` for BIDS
   correctness, `noise_dir`/`find_noise` still driving the actual source
   pipeline) until a follow-up unifies them.
