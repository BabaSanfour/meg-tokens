# Refactoring preprocessing to the three-stream pipeline

Work plan for bringing `meg_tokens/meg/preprocessing.py` and
`meg_tokens/workflows/preprocessing.py` up to the specification in
`docs/meg_pipeline_plan.md`, Preprocessing. Delete this file when the work lands.

## What is wrong today

`meg_tokens/meg/preprocessing.py` predates every decision in the preprocessing specification:

| line | current behaviour | required |
|---|---|---|
| `load_and_filter_raw:44` | notch at `arange(60, sfreq/2, 60)` — nine harmonics, seven of them above the low-pass | ZapLine, fixed 60 Hz, 2 harmonics, **before** the filter |
| `load_and_filter_raw:35` | no compensation check | raise to grade 3 when the acquisition is lower |
| — | no bad-channel detection | spectral outlier + `find_bad_channels_lof` |
| — | no bad-segment annotation | `annotate_amplitude` + `annotate_muscle_zscore`, two passes |
| `run_ica_rejection:97` | `eog_ch: str = 'EEG057'` — one EOG channel | `EEG057` **and** `EEG058` |
| `run_ica_rejection:124,132` | `except Exception` prints and continues with an empty exclusion list | must raise; a misnamed channel would otherwise "succeed" having removed nothing |
| — | no resample | 600 Hz, last |
| — | one cleaner | three streams |
| — | cleaned data only | cleaned **and** artifact estimate per stream |

## The shape to build

```
read + compensate ─ ZapLine ─ bandpass ─ bad channels ─ bad segments (1)
                                                              │
                          ┌───────────────┬───────────────────┤
                        DSS             ICA                 SSP
                          └───────────────┴───────────────────┘
                                                              │
                              bad segments (2) ─ resample ─ persist
```

The trunk is computed **once** and shared. Computing bad channels or segments
per stream would make the streams differ in two ways at once.

## Work items

**0. `meg/artifact_detection.py` — already in place.**

Threshold derivation and detector validation, extracted from the benchmark
scripts that produced B4's constants:

- `robust_scale` / `robust_scales` — MAD of adjacent-sample differences. B4's
  frozen thresholds are multiples of this, so re-deriving or validating them
  needs no new code.
- `enrichment_db` + `matched_clean_spans` — does what a detector flagged carry
  more in-band power than a duration-matched amount of what it did not? This
  is how the open muscle-z question gets closed, and it applies unchanged to
  the new spectral bad-channel rule, which currently has no validation at all.

**1. `meg/preprocessing.py` — replace, do not extend.**

- `read_staged_raw(path)` — read, raise compensation to 3, return with the
  source grade recorded.
- `build_trunk(raw)` → `(raw, TrunkReport)`. ZapLine, bandpass, bad channels,
  bad segments pass 1. `TrunkReport` carries the channel scores, annotations
  and the source compensation grade for the sidecar.
- `ocular_events(raw, channel)` — 0.5–7 Hz excursions at 3.0× MAD, 0.4 s
  separation. Not `find_eog_events`; see the preprocessing specification.
- One callable per stream, all with the same signature
  `(raw, context) -> (cleaned, artifact, StreamReport)`:
  `clean_dss`, `clean_ica`, `clean_ssp`.
- `finish(cleaned)` — bad segments pass 2, resample to 600 Hz.
- Delete `load_and_filter_raw` and `run_ica_rejection` once nothing imports
  them.

**2. Naming — the `proc` entity carries the stream.**

Today `proc-clean` / `proc-filt` encodes whether ICA ran. That collapses to a
single output and cannot express three. Replace with:

```
sub-<ID>_task-tokens_acq-<cond>_run-<n>_proc-<dss|ica|ssp>_raw.fif
sub-<ID>_task-tokens_acq-<cond>_run-<n>_proc-<dss|ica|ssp>_desc-artifact_raw.fif
sub-<ID>_task-tokens_acq-<cond>_run-<n>_proc-<dss|ica|ssp>_raw.json
```

Three cleaned recordings per run, each with its artifact estimate and sidecar.
`docs/data_contract.md` needs the new entity values and a row per file.

**3. Discovery — `meg_bids.preprocessed_meg_runs`.**

It already takes `processing`; the values become `dss` / `ica` / `ssp`. Callers
that assume one preprocessed run per recording must be found and made explicit
about which stream they want — `epoch_subjects` in
`workflows/preprocessing.py:194` is the first.

**4. Workflow — `workflows/preprocessing.py`.**

`preprocess_run` builds the trunk once, then loops the streams. `settings.run_ica`
becomes `settings.streams: tuple[str, ...]` defaulting to all three. The
`processing="clean" if settings.run_ica else "filt"` branch at line 97 goes.

**5. Snakefile.** `PROCESSING = "clean" if run_ica else "filt"` and
`CLEAN_PATTERN` become stream-parameterised; the DAG fans out three ways per
run and every downstream rule needs to say which stream it consumes.

**6. Sidecar.** Per run per stream: components or stages removed, bad channels
with scores, annotated segments and seconds retained, source compensation grade
and whether it was raised, and the **rank** after cleaning — the inverse operator
needs it and nothing currently records it.

**7. Dependencies.** `mne-denoise` (ZapLine, DSS) is not in `pyproject.toml`
and is not installed on the cluster's shared venv. Pin the commit.

## Order

1. `read_staged_raw` + `build_trunk` + `ocular_events`, with tests
2. `clean_ica` first — it is the closest to the existing code and the baseline
3. `clean_dss`, `clean_ssp`
4. `finish`, sidecar, persistence
5. Naming, discovery, workflow, Snakefile — the fan-out, in one change
6. `data_contract.md`

Steps 1–4 are testable on one run without touching the workflow. Step 5 is
where every downstream consumer breaks at once, so it should be a single
reviewable change rather than incremental.

## Open

- **Three files or six.** The spec persists a cleaned recording *and* an
  artifact estimate per stream. If only the cleaned recordings are wanted, drop
  the `desc-artifact` files — but the artifact estimates are what make the
  streams comparable on what each *called* artifact.
- **Which stream is canonical** for Steps C and D, or whether all three carry
  through. Nothing downstream can be written until this is settled.
- **Muscle z-threshold.** `annotate_muscle_zscore` at z=6/7/8 annotates
  2.15–2.63% of a recording at the median and scores 8.1–8.6 dB enrichment —
  by the test in `artifact_detection`, all three *find muscle*. They were
  previously rejected on a 1% coverage budget, which is a different question
  from whether the detector works and should not be traded against it. The
  paper excludes jaw-clenching segments deliberately, so ~2.4% may be the
  correct amount. Re-run the grid with a coverage limit derived from that
  intent rather than a round number.
- **SSP `n_mag`.** MNE's default is 2 per artifact — 4 dimensions of 270. Not a
  measured choice.
- **preprocessing / autoreject boundary.** Continuous detection here and per-epoch
  rejection at epoching overlap. Decide which layer owns what before either is
  implemented.
