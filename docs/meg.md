# MEG

MEG-side QC: event-level alignment between TDMS logs and MEG trigger pulses
for epoch extraction, and the investigation behind the four-subject
exclusion list used to compare against the published preprint. Behavior-side
pipeline and issues: `docs/behavior.md`. Derivative schema:
`docs/data_contract.md`.

## Alignment QC (Stage 4: Epoching)

Results checked against 323 TDMS files from 32 subjects and 399 raw CTF
recordings in the `DDM-tthiery` dataset.

| Category | Count | Resolution |
| :--- | ---: | :--- |
| Exact alignment | 184 | No exception |
| Trailing-boundary mismatch | 114 | `KNOWN_TRAILING_TRIAL_MISMATCHES`, `on_mismatch="truncate"` |
| Interior missing go cue | 21 | `KNOWN_GO_RECONSTRUCTION_RUNS`, go-cue reconstruction |
| Unrecoverable real trial | 1 (`H12Slow2`, trial 2) | `KNOWN_UNRECOVERABLE_TRIALS` |
| Scratch files | 3 | `behavior_ignore_files` |

`184 + 114 + 21 + 1 + 3 = 323`: every source file has either a verified
resolution or an explicit exclusion. Reproduce with `meg-tokens behavior qc`.

### Trailing-boundary mismatches

All 114 runs align throughout their recordings and differ only at the final
boundary. Examples include one extra MEG trial-start pulse (`H03RT1`),
missing TDMS-side events (`H06Slow2`, -12), and the all-go-channel case
(`H02RT1`, -39). Maximum confirmed timing residual was 4.25 ms.
`mismatch_policy` truncates only these documented runs and leaves unknown or
interior mismatches as hard errors; this recovers valid epochs from
approximately 36% of the production runs without changing MEG/behavior
pairing.

### Missing go cues

Of the 21 reconstruction runs, `H02RT1` has approximately 39 real trials
with no recorded go cue; the other 20 each have one interior dropout. For
each run, `reconstruct_missing_go_events` pairs trials with trial-start
pulses and synthesizes the missing go cue at `trial_start + tGO + lag`, the
lag calibrated from that run's real go cues (4 ms fallback only for
`H02RT1`). Never-started `7003` trials are never reconstructed, including
mixed-pattern runs such as `H07Slow2` and `H07Slow4`.

### H02 motor baseline

Thomas's notebooks use only `RT2` for `H02`, consistent with `H02RT1`'s
go-cue defect above. We keep both runs for every subject: `H02RT1` mean
508.6 ms vs `RT2` 588.4 ms — an 80 ms gap, unexceptional across the cohort
(mean |gap| 30.4 ms, max 95.5 ms at H08, where Thomas also kept both runs).
Using both runs puts our baseline 39.4 ms below Thomas's RT2-only baseline;
this cancels exactly in every paired within-subject contrast and shifts
absolute group means by ~1.4 ms.

### `H12Slow2`

Matches `H12_..._05.ds` at 193× separation from the next candidate. The
recording missed the first three TDMS trial starts: trials 1 and 3 were
never-started rows, trial 2 was a genuine unrecoverable trial, and the
trailing extra start pulse has no go cue. `exclude_unrecoverable_trials`
removes only `('H12', 'Slow', '2') → {2}` before synchronization; the count
closes as `65 total - 7 never-started - 1 unrecoverable = 57` MEG go
events, matched at 2.38 ms mean / 4.83 ms maximum `tGO` residual.

### Implementation and legacy comparison

TDMS parser/schema validation: `meg_tokens/behavior/tdms.py`,
`meg_tokens/behavior/schema.py`. Epoching (never-started filtering,
start-event handling, trailing policy, go-cue reconstruction, the H12
exception): `meg_tokens/meg/epoching.py`,
`meg_tokens/workflows/preprocessing.py`. Scratch-file allowlist:
`behavior_ignore_files`.

Checked against Thomas's original TDMS parser
(`archive/replicated/DDM_analysis_scripts/Create_df.ipynb`) and behavior
notebooks (`archive/replicated/DDM_scripts/scripts_new/`): two mismatches
matter beyond the `H02` baseline decision above — the `sTrialClass`
reference-frame difference in `Modify_df_preproc.ipynb` (root cause of the
reversed ambiguous/misleading contrast; see `docs/behavior.md`, Findings),
and our `key: value` line parser replacing Thomas's fixed-offset string
slicing, a deliberate modernization (`docs/data_contract.md`) that is
functionally equivalent on every field both extract. `rawRT` and the
post-error-slowing adjacency rule match exactly.

## Subject Exclusion

**Excluded four: `H06, H07, H10, H20`** — confirmed by direct reproduction
of the published statistics, not inference. Not yet written to
`subject_exclusions` in any project TOML.

### Preprint criteria (primary source)

bioRxiv JATS full text,
[10.1101/2022.06.14.494674](https://doi.org/10.1101/2022.06.14.494674)
(Thiery, Rainville, Cisek, Jerbi), Methods → Subjects:

> "Thirty-two subjects... **Two were excluded before the start of analysis
> because of large head movement (extent of displacement during one session
> > 20 mm)**, and **one because of myographic artifacts during MEG
> scanning**. **Data from another subject was unusable due to an
> interruption of the MEG system during the experiment**, leaving 28
> subjects."

Three criteria, four subjects: 2× head movement (>20 mm/session), 1×
myographic (EMG) artifact, 1× MEG system interruption.

### Dataset

- Raw MEG (CTF `.ds`, read-only throughout): `/media/karim/Hamza/DDM-tthiery`,
  32/32 subjects, 8 Slow/Fast (315 s) + 2 RT (135 s) recordings each.
- TDMS: `/media/karim/Hamza/meg-tokens/tdms` — 32/32 subjects, 10 runs each.

### Identification

Ten of Thomas's scripts (six MEG under
`archive/replicated/DDM_scripts/scripts_new/` — `04_compute_sources*.py`,
`05_compute_resample_raw.py`, `05_compute_power_new_baseline.py`,
`08_Decoding_SRC_POWER_Trial_types_time_Trial_Types.py` — plus four behavior
notebooks) hardcode the same 28-subject list. By elimination: **H06, H07,
H10, H20**.

The *already-saved output cells* of his behavior notebooks, on exactly that
list, reproduce the published numbers:

| Contrast (source notebook) | Notebook output | Preprint | |
|---|---|---|---|
| Easy DT (`00_44_Behavior_Trial_Types`) | 1028.73 ± 59.44 | 1028 ± 59 | exact |
| Ambiguous DT | 1404.92 ± 74.30 | 1405 ± 74 | exact |
| Misleading DT | 1432.53 ± 78.86 | 1433 ± 79 | exact |
| Easy vs Ambiguous | t=-15.05, p=1.19e-14 | t=-15.04 | exact |
| Easy vs Misleading | t=-13.10, p=3.25e-13 | t=-13.10 | exact |
| Ambiguous vs Misleading | t=-1.84, p=0.0767 | t=-1.84, p=0.077 | exact |
| Fast vs Slow DT (`00_44_Behavior_Slow_Fast`) | t=-6.08 | t=-6.08 | exact |
| Fast/Slow errors | 45.11 vs 36.0; t=6.08 | 45.1 vs 36; t=6.10 | within tolerance |

No other subset was tried or needed — this is his unmodified code producing
his reported numbers, not curve-fitting.

Subject counts across notebook versions (ordered by which later-collected
subjects are present) show the exclusions accumulated over time:

| n | Script(s) | H06 | H07 | H10 | H20 |
|---|---|:---:|:---:|:---:|:---:|
| 23 | `Behavior_purcent_correct_5v5` | in | in | in | in |
| 22 | `Behavior_Fast_Slow_DT_5v5`, `44_Behavior_Ratio_DT_PurcentCorrect` (early) | in | in | in | **out** |
| 26 | `44_Behavior_DT_error_correct` | in | **out** | **out** | out |
| **28** | `00_44_Behavior_Slow_Fast`, `00_44_Behavior_Trial_Types`, `44_Behavior_Fast_Slow_DT` (final), `44_Behavior_post_error_slowing`, + 6 MEG scripts | **out** | out | out | out |

`H20` first, then `H07`/`H10`, then `H06` — progressive identification,
supporting `H06`'s exclusion as deliberate despite its weak signal below.

### MEG-signal evidence

Read-only checks on raw `.ds`, restricted to Slow/Fast recordings unless
noted (only Slow/Fast feeds the behavioral analysis). Reproduce with
`scripts/qc/meg_session_qc.py` (movement, truncation) and
`scripts/qc/meg_emg_qc.py` (EMG, ICA).

| Subject | Movement z | EMG broadband z | ICA muscle rank (/32) | Truncation | Cleaning rescues? |
|---|---:|---:|---:|---|---|
| H06 | +0.50 | -0.63 | 31 (cleanest) | none | n/a — nothing to clean |
| H07 | +0.26 | -0.88 | 26 | none | n/a; **15% epoch loss** on amplitude threshold |
| H10 | **+3.50** | +0.22 | 21 | none | inconclusive |
| H20 | -0.09 | -0.05 | 12 | none | inconclusive |
| H04 *(not excluded)* | n/a | **+3.01** | **1 (worst)** | none | **yes — normalized, 0% loss** |

**Movement** (CTF `.hist` head-localization QC): population mean 5.65 mm, sd
3.19 across 32 subjects × 8 sessions. H10 is a clear outlier (16.8 mm max)
but no subject crosses the literal 20 mm threshold within Slow/Fast — H10's
29.78 mm is a resting-state recording, the only one of 412 sessions above
20 mm.

**Truncation / interruption**: exact duration (MNE header vs. configured
trial length) and flat/dead-segment scans across every real recording found
**one** truncated file in the whole dataset — `H05_DDM-tthiery_20180220_02.ds`
(138 s vs. 315 s), and H05 is *not* excluded. All 54 H06/H07/H10/H20
recordings match exactly, no flat segments, TDMS sizes unremarkable. **No
footprint for this criterion on any candidate.**

**EMG**, two methods over the full 32-subject population — broadband power
ratio (40–100 Hz / 1–20 Hz; population mean 0.0551, sd 0.0294) and ICA
muscle-component scoring (spectral slope × frontal/temporal topography).
Both agree that **H04 is the standout outlier** (0.1437, z=+3.01; 14/20
muscle components; the only positive spectral slope in the cohort; confirmed
across all 15 of its recordings individually). H06/H07/H10 show no EMG
signal; H20 is only mildly elevated against the full population.

**Does cleaning rescue them?** Removing ICA-flagged muscle components and
simulating peak-to-peak epoch rejection (4000 fT): H04's contamination is
cleanly separable and fully removable (70% of IC variance, but 0% epoch
loss) — consistent with it correctly *not* being excluded. For the other
elevated subjects the automated flagging did not reliably reduce the
signature, so it cannot settle them either way. Under ordinary epoch
rejection nearly every subject loses ~0% of data; H07's 15% is the sole
exception and is a different signature (large-amplitude transients) from
anything the spectral/ICA checks probe.

### Known limitations

- The ICA muscle-flagging heuristic is not a reliable cleaning ground truth
  — removing flagged components made the broadband metric *worse* for
  several subjects.
- No operator notes exist in acquisition metadata (all 527 `.ds` folders
  grepped for abort/interrupt/EMG/artifact/comment), so the original
  visual-inspection judgment calls can't be reconstructed.
- H07's amplitude-based epoch-rejection signal is unexplored (no
  per-channel breakdown).
- EMG/ICA comparisons use 60–120 s windows, not full recordings; unlikely
  to change H04's or H10's standing but could shift marginal rankings. The
  ICA component counts are sensitive to that window choice (H07 scores 2 or
  4 depending on it), so treat individual counts as indicative and the
  H04-vs-rest gap as the robust result.
