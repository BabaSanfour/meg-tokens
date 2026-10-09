# MEG

**Living document.** This is the pipeline as decided: what each stage does and
why it deviates from the paper where it does. It is not a record of how the
decisions were reached.

The evidence behind two of those decisions follows the plan: event alignment
between the behavioral logs and the MEG triggers (§6), and why all 32 subjects
are kept where the preprint used 28 (§7). Companions: `docs/behavior.md`
(behavioral pipeline and the trial-classification finding),
`docs/data_contract.md` (derivative schema and naming).

---

## 0. Where the repo stands

**Stage 0 — complete.** 32 subjects staged with `meg/`, `anat/`, `beh/`; 319
staged task runs (4 Fast + 4 Slow + 2 RT each, H05 RT run 1 absent) as CTF
`.ds`; per-run `events.tsv` / `channels.tsv` / `meg.json`; 26 empty-room
sessions; 32/32 headshapes as native CTF `.eeg`. All 323 TDMS runs resolved.

**Coregistration — complete.** 319 MEG–MRI transforms and 32 QC figures. Per-subject
median digitizer-to-scalp distance 2.15–4.77 mm (mean 2.80, sd 0.61), point
retention 77.5%. H07 and H10 are the two worst fits and the two template
anatomies; that flag follows them into every source-level result. Committed as
`4a91e44` and `e11e727`.

**Preprocessing — implemented.** `meg_tokens/meg/preprocessing.py` follows
the specification below. It has not yet run on real data as a pipeline stage;
its DSS parameter search and three-cleaner comparison
(`scripts/dss_param_search_array.sh`) were readied for the 319-recording array
on `fir`.

**Epoching and source reconstruction — not started.** Zero preprocessed raws, zero epochs, zero
source estimates, zero features.

**Downstream helpers with no callers.** `psd_description`
(`workflows/spectral.py`), `run_batch_dpca` (`workflows/decomposition.py`),
`get_peak_latency` / `get_significance_onset` / `compute_motor_lateralization`
(`analysis/statistics.py`), `extract_roi_time_courses`
(`features/connectivity.py`), `plot_tfr_spectrogram` / `plot_sensor_topomap`
(`reports/meg.py`), `_parse_baseline` (`workflows/power.py`). Each stage either
uses or deletes them when it is built; nothing can verify them before that.

---

## 1. Paper vs. repo

Thiery, Rainville, Cisek, Jerbi (2022),
[10.1101/2022.06.14.494674](https://doi.org/10.1101/2022.06.14.494674).

| Step | Paper | Here |
|---|---|---|
| Sampling rate | states 2400 Hz | data is 1200 Hz; 2400 Hz is the nested `hz.ds` head localizer |
| Gradient compensation | not stated | grade 3, raised at read time where the acquisition lacked it |
| Line noise | notch 60/120 Hz | ZapLine, fixed frequency |
| Filter | 0.5–150 Hz | same |
| Resample | 600 Hz, before ICA | 600 Hz, **last** |
| Bad channels | visual, spectral | automated: broadband spectral outlier + LOF |
| Bad segments | visual | automated: amplitude + muscle z-score, two passes |
| Artifact removal | 20-component ICA, projector `I − ΦΦ⁺` on unfiltered raw | three parallel streams: DSS chain, ICA, SSP |
| EOG/ECG | 2 bipolar EOG + 1 ECG | `EEG057` vEOG, `EEG058` hEOG, `EEG059` ECG |
| Epoch locks | token jump, button press, feedback, decision time | `go`, `enter`, `feedback`, `decision` |
| Epoch windows | −500/+1000 token; −1500/+2000 press/fb/DT | per-lock, matching |
| Noise covariance | baseline [−200, 0] ms, shrunk | empty-room, shrunk |
| BEM | 3-layer, [0.3, 0.006, 0.3] | same |
| Source space | ~8196 vertices | `oct6` = 8196 |
| Inverse | dSPM, λ² = 1, single-trial | same |
| Bands | FOOOF: θ5–8, α9–13, β16–24, γ30–60, hγ60–90 | same |
| Power | FIR1 order 3 + Hilbert, 500 ms / 50 ms step | Hilbert, same windowing |
| PCA | source power, 12 classes, top 20 PCs | same, on the corrected class frame |
| Decoding | LDA, LOSO, 1000 permutations | same |

---

## 2. Decisions

**Replicate or improve → improve, and document every deviation.** Every
departure has a row in the deviations table.

**Subject set → all 32, with 30 on their own anatomy.** H07 and H10 have no T1
and run on `fsaverage` scaled to their own head shape, flagged in every QC row
and downstream table. The preprint used 28.

**Trial classes → the corrected correct-target design profile**, not the
paper's chosen-target `nProb`, which is response-confounded
(`docs/behavior.md`). Any class-keyed MEG analysis answers a different question
than the paper's.

**Storage → not a constraint.** Single-trial source estimates at `oct6` are
1.63 TB at 1200 Hz against 300 TB available.

---

## 3. Risks

**1. Two subjects can never be source-localized from their own anatomy.** H07
and H10; treat their source results as lower-confidence throughout.

**2. The paper's MEG classes inherit the behavioral confound.** The 12 PCA
classes rest on a trial classification the behavioral work showed is
response-confounded. Epoch metadata needs the corrected labels from the start.

**3. H05 has no empty room.** Recorded 2018-02-20; no empty-room session was
acquired that day. Open: nearest-day noise, baseline covariance for that
subject alone, or exclusion from source analysis.

**4. Four runs were acquired without gradient compensation.** H11 Fast 3/4 and
Slow 3/4 are grade 0; the other 315 are grade 3. Corrected at read time (see Preprocessing),
because CTF stores compensation in the `.res4` header and applies it on read —
it cannot be baked into a staged `.ds` without abandoning the CTF-native
format.

---

## 4. Steps

### Coregistration · `done`

One ICP fit per subject, composed with a closed-form per-session landmark
correction. The head frame and the digitizer frame disagree by 2–8 mm and a
subject's head frame drifts between its own sessions, which is why the fit is
per-subject and the correction per-session.

### Preprocessing · `implemented, not yet run as a pipeline stage`

Three cleaners run on one shared trunk, so the streams differ **only** in the
cleaner. Each stream persists its own cleaned recording and its own artifact
estimate: three MEG files per run per stream output.

**Shared trunk.**

| # | step | parameters |
|---|---|---|
| 0 | read staged CTF; raise compensation to grade 3 if lower | 1200 Hz |
| 1 | ZapLine on the **unfiltered** raw | `line_freq=60`, `n_harmonics=2`, fixed |
| 2 | bandpass | 0.5–150 Hz, `firwin` |
| 3 | bad channels — record and score, do not auto-drop | broadband spectral outlier over 1–150 Hz; `find_bad_channels_lof` |
| 4 | bad segments, pass 1 — excluded from cleaner *fitting* | `annotate_amplitude` peak `1.2485926067065662e-12 T`, flat `1.2485926067065662e-16 T`; `annotate_muscle_zscore` at 110–140 Hz |

ZapLine precedes the filter so the harmonics still exist. The trunk must stay
shared: computing bad channels or segments per stream would make the streams
differ in two ways at once and the comparison uninterpretable.

**Streams.**

| stream | method |
|---|---|
| **DSS** | `dss_cycle_qrs` → `dss_cycle_blink` → `dss_bandpass_muscle`. DSS with `CycleAverageBias` locked to R peaks (−0.1/+0.2 s) then blinks (−0.2/+0.5 s), then `BandpassBias(110, 140)`. `n_components=40`, `n_select="auto"`, `component_action="subtract"`, `max_prop_remove=0.3` |
| **ICA** | 20 components on the filtered copy; EOG from `EEG057` **and** `EEG058`, ECG from `EEG059`; `ica.apply` to the filtered copy. Missing reference channels raise |
| **SSP** | `compute_proj_ecg` on `EEG059`; `compute_proj_eog` run per EOG channel and concatenated; `n_mag=2` per artifact — 4 dimensions of 270 |

**Shared tail, per stream.**

| # | step |
|---|---|
| 5 | bad segments, pass 2 — residual muscle and saccade the cleaner missed |
| 6 | resample to 600 Hz — **last**, after the low-pass |
| 7 | persist: cleaned raw, artifact estimate, sidecar |

Resampling earlier aliases; resampling before ZapLine folds the higher
harmonics onto real signal.

**Sidecar, per run per stream:** components or stages removed, bad channels
with their scores, annotated segments and seconds retained, source
compensation grade and whether it was raised, and the **rank** after cleaning —
all three cleaners remove spatial dimensions and the inverse operator must be told.

**Artifact events.** Blink and saccade events come from excursions in the
0.5–7 Hz band of `EEG057` / `EEG058`, thresholded at 3.0× the channel's median
absolute deviation with 0.4 s minimum separation. `mne.preprocessing.find_eog_events`
is not usable: its threshold is set from the segment's extremes, so it reports
59 blinks/min over 60 s of H01 and 1.5/min over the same recording's full
315 s, against a cohort median of 16.9/min. R peaks come from
`find_ecg_events` on `EEG059`, which is stable to within 1.1% across lengths.

**Open.**

- **Which stream is canonical** for epoching and source reconstruction, or
  whether all three carry through. Nothing downstream can be written until this
  is settled; the three-cleaner comparison (`--mode streams`) is what should
  settle it.
- **Muscle z-threshold.** `annotate_muscle_zscore` at z=6/7/8 annotates
  2.15–2.63% of a recording at the median and scores 8.1–8.6 dB enrichment —
  all three *find muscle*. They were previously rejected on a 1% coverage
  budget, which is a different question from whether the detector works. The
  paper excludes jaw-clenching segments deliberately, so ~2.4% may be the
  correct amount; derive the coverage limit from that intent, not a round number.
- **SSP `n_mag`.** MNE's default is 2 per artifact — 4 dimensions of 270. Not a
  measured choice.
- **Preprocessing / autoreject boundary** — see Epoching.

### Epoching · `todo`

Four locks: `go`, `enter`, `feedback`, and a `decision` alignment (button press
minus the hand-specific motor baseline). Per-lock windows matching the paper.
Attach behavioral metadata: corrected-frame trial class, condition, choice,
correctness, `dt_ms`, SPD. `autoreject` handles per-epoch bad-channel
interpolation and epoch rejection — it is not yet a dependency and must be
added. The boundary between preprocessing's continuous detection and autoreject's
per-epoch pass must be settled before either is implemented.

### Source reconstruction · `todo`

`cov` / `bem` / `src` per subject, `fwd` / `inv` per run, single-trial dSPM,
morph to `fsaverage`. Noise covariance is BIDS-native, resolved through each
subject's own `AssociatedEmptyRoom`; 31 of 32 subjects resolve (see the risks section).
3-layer BEM is feasible with no new anatomical processing — every reconstructed
subject already carries `bem/` with the three surfaces.

---

## 5. Deviations from the paper

Every entry must be defensible in a methods section.

| Deviation | Paper | Here | Why |
|---|---|---|---|
| Line noise | notch 60/120 Hz | ZapLine, fixed frequency | Measured on four subjects: ZapLine leaves the 60 Hz peak 1.1 dB above the local floor with the ±0.3 Hz shoulders at −0.08 dB; MNE's default notch overshoots to 5.8 dB *below* the floor and takes 3–4 dB out of those shoulders. A notch removes real signal beside the line. |
| Gradient compensation | not stated | grade 3, raised at read time where absent | H11 Fast 3/4 and Slow 3/4 were acquired at grade 0. Applying it moves them from 0.41 dB to 0.10 dB median distance from the same subject's grade-3 runs across 1–150 Hz. |
| Resample position | before ICA | last | Resampling before the low-pass aliases; before ZapLine it folds harmonics onto signal. |
| Bad channels | visual inspection of power spectra | automated spectral outlier + LOF | 526 of 527 raw `.ds` carry an empty `BadChannels` file. The original selection left no record and is not reproducible. |
| Bad segments | visual inspection | automated amplitude + muscle z-score, two passes | Same: 328 empty `bad.segments` files. The second pass implements the paper's own stated target — residual artifact "not captured by any component". |
| Artifact removal | one ICA pass; projector `I − ΦΦ⁺` on unfiltered raw | three parallel streams, all persisted | The explicit projector gained only 1–3.5 percentage points over `ica.apply` on the same components while adding >0.25 dB distortion on three of four subjects. Running three cleaners and keeping every artifact estimate makes the choice measurable rather than assumed. |
| EOG channels | 2 bipolar EOG | both `EEG057` and `EEG058` | The paper describes two EOG channels; its code used only one. `EEG058` is a real horizontal EOG, so saccade artifact was invisible to component selection. |
| Coregistration | manual fiducials + ICP, visually verified | landmark fit + two-pass ICP, one fit per subject composed to each session | Reproducible rather than hand-placed, and it corrects between-session head-frame drift a single manual fit would absorb as error. |
| MRI landmarks | picked by hand | by hand for 24 of 30; estimated from `fsaverage` for H22, H27, H29, H30, H31, H32 | No picked set was ever saved for those six. Recorded per subject rather than left silent. |
| Subject set | 28 | 32 coregistered, 30 on their own anatomy | H06/H20 exclusions could not be reproduced from any MEG-signal criterion. H07/H10 run on scaled `fsaverage`, flagged everywhere. |
| Noise covariance | baseline [−200, 0] ms | empty-room, shrunk | An empty-room estimate measures sensor noise without task-locked signal leaking into the estimator. |
| Trial classes | chosen-target `nProb` | corrected correct-target design profile | The paper's frame is response-confounded; see `docs/behavior.md`. |

---

## 6. Alignment QC (Stage 4: Epoching)

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

Implemented in `meg_tokens/meg/epoching.py` (never-started filtering,
trailing policy, go-cue reconstruction, the H12 exception); the scratch-file
allowlist is `behavior_ignore_files`.

---

## 7. Subject Exclusion

The preprint excluded four subjects; reproducing its published statistics
identifies them as **`H06, H07, H10, H20`**. This project keeps all 32
(§2): no MEG-signal criterion reproduces the `H06` and `H20` exclusions, and
`H07`/`H10` lack a T1, which is handled by scaled `fsaverage` and a flag rather
than by dropping them. `subject_exclusions` stays empty. The evidence:

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

Earlier notebook versions carry 22-26 subjects, dropping `H20` first, then
`H07`/`H10`, then `H06`, so the four were removed one at a time and `H06`'s
exclusion was deliberate despite its weak signal below.

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
