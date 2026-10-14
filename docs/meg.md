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
the specification below. It has not yet run on real data as a pipeline stage.

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
| Artifact removal | 20-component ICA, projector `I − ΦΦ⁺` on unfiltered raw | two parallel streams, DSS chain and ICA, each followed by the same ASR muscle step and a tonic muscle stage of its own method |
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

### Preprocessing · `run on the cohort`

Two cleaners run on one shared trunk. Each is followed by the same ASR step
for muscle bursts and then by a tonic muscle stage of its own method, so the
DSS stream contains no ICA. The streams differ in the cleaner and in the
tonic muscle stage. Both streams are run on
every recording and both carry through; they are compared on results, not
chosen between in advance. Each stream persists its own cleaned recording and
its own artifact estimate: three MEG files per run per stream output.

**Shared trunk.**

| # | step | parameters |
|---|---|---|
| 0 | read staged CTF; raise compensation to grade 3 if lower | 1200 Hz |
| 1 | flux jumps — restore the level on every channel the jump moved, bridge the edge, annotate `BAD_jump` | step over 100 pT between 10 ms means; moved channels over 3 pT; 25 ms bridged either side; ±0.5 s annotated |
| 2 | ZapLine on the **unfiltered** raw | `line_freq=60`, `n_harmonics=2`, fixed |
| 3 | bandpass | 0.5–150 Hz, `firwin` |
| 4 | bad channels — record and score, do not auto-drop | broadband spectral outlier over 1–150 Hz; `find_bad_channels_lof` |
| 5 | bad segments, pass 1 — excluded from cleaner *fitting* | `annotate_amplitude` peak `1.2485926067065662e-12 T`, flat `1.2485926067065662e-16 T`; `annotate_muscle_zscore` at 110–140 Hz |

Flux jumps come first because a jump is a step only until something filters
it: the 0.5 Hz high-pass turns a 360 pT step into about six seconds of ringing
that no cleaner removes. Twelve of the 319 recordings hold one jump each, on
1 to 22 channels at once; each is one flux quantum (345–375 pT) or two. No
other step in the cohort's raw data exceeds 19 pT, so the level sits in an
empty decade. ZapLine precedes the filter so the harmonics still exist. The trunk must stay
shared: computing bad channels or segments per stream would make the streams
differ in two ways at once and the comparison uninterpretable.

**Streams.**

| stream | method |
|---|---|
| **DSS** | `dss_cycle_qrs` → `dss_cycle_blink` → `dss_cycle_saccade_positive` → `dss_cycle_saccade_negative`. DSS with `CycleAverageBias` locked to R peaks (−0.1/+0.2 s), then blinks (−0.2/+0.5 s), then saccades of each direction in turn (−0.1/+0.4 s). A saccade direction with fewer than 10 usable events gets no stage. `n_components=40`, `n_select="auto"`, `component_action="subtract"`; every other selection setting at the library default |
| **ICA** | 20 components on the filtered copy; EOG from `EEG057` **and** `EEG058`, ECG from `EEG059`; `ica.apply` to the filtered copy. Missing reference channels raise |

**Shared tail, per stream.**

| # | step |
|---|---|
| 6 | muscle — ASR on the magnetometers: cutoff 20, 0.5 s windows, `max_dims=0.25`, statistics high-passed at 80 Hz (4th-order Butterworth), calibrated on the recording itself |
| 7 | tonic muscle, by the stream's own method. DSS stream: DSS biased to 30–140 Hz power on the leading 80 principal components, subtracting the leading run of components whose spectral slope is above 0. ICA stream: a second ICA, 40 components, seed 0, removing every component whose spectral slope is above −0.5 |
| 8 | bad segments, pass 2 — residual muscle the stream missed |
| 9 | resample to 600 Hz — **last**, after the low-pass |
| 10 | persist: cleaned raw, artifact estimate, sidecar |

ASR's statistics — calibration and the per-window decision to reconstruct —
see only the data above 80 Hz, so muscle triggers it and alpha and evoked
power do not; the reconstruction itself acts on the unfiltered recording. It
runs on the arrays, annotated spans included: the library's MNE interface
pastes annotated samples back uncleaned, and those are the muscle segments.
The trunk's `BAD_muscle` annotations stay on the recording after ASR, so the
spans it reconstructed can still be excluded downstream. The artifact estimate
is the trunk minus the stream's output, so it holds what the cleaner, ASR and
the tonic muscle stage removed together.

The tonic muscle stage is there for what ASR cannot see: muscle held through
the whole recording is ASR's own baseline. A component counts as muscle when
its spectrum does not fall with frequency. The slope is that of log power on
log frequency, fitted over 7–55 and 65–70 Hz on 2 s Welch windows, outside
annotated spans (Dharmaprani 2016 for ICA, Janani 2018 for BSS). Both stages
subtract a fixed set of components per recording, which lowers the rank; the
sidecar's `rank` counts them.

Tested on the finished recordings of all 319 runs, both streams, before
adoption. People without tonic muscle (23): either stage changes alpha, beta
and gamma power on the median sensor by under 0.1 dB, leaves the evoked
response at a correlation of 0.999 or more, and keeps 96% (DSS) and 91% (ICA)
of a simulated source at the worst 5% of positions; 21% (DSS) and 38% (ICA)
of their recordings lose no component. People with it (9): 65–100 Hz power
on the occipital rim falls by 5.8 dB (DSS) and 6.8 dB (ICA), from 7.5 dB above
the median sensor to 1.5 and 0.4 dB, against 0.4 dB in the 23. Their
high-gamma task response map matches the 23's at 0.21 before, 0.29 after DSS
and 0.41 after ICA; beta and alpha responses are unchanged. Neither stage
changes power differently in Fast and Slow.

What the stages cost in the nine: alpha power on the rim sensors falls by 1.2
dB (DSS) and 1.8 dB (ICA) while the alpha peak rises above its floor (5.7 dB
to 6.9 and 7.5), and the 20 strongest alpha sensors lose under 0.1 dB, so the
loss is floor and not peak. The evoked response on the rim sensors shrinks by
1.2–1.3 dB (DSS) and 1.5–2.1 dB (ICA), waveform correlation 0.87 and 0.80;
whether that is noise leaving the average or signal is not known. A simulated
source low at the back of the head keeps 48% (DSS) and 21% (ICA) at the worst
5% of positions.

The two rules are not equally firm. DSS components sit close to its threshold:
at −0.15 it removes 17 components from a recording without muscle, at 0 it
removes 2.6. ICA components separate (median slope −0.25 removed, −1.32 kept),
but the fit depends on its seed: across three seeds the same recording gives
the same component count two times in three, the rim result in the nine
differs by 0.4 dB at the median and 1.4 dB in one recording in ten, and 5.5%
of fits stop without converging. The sidecar records
`tonic_muscle_converged`.

Resampling earlier aliases; resampling before ZapLine folds the higher
harmonics onto real signal.

**Sidecar, per run per stream:** components or stages removed, the fraction
of samples ASR reconstructed, flux jumps repaired, bad channels with their
scores, annotated segments and seconds retained, source compensation grade
and whether it was raised, and the **rank** after cleaning — both cleaners remove spatial
dimensions and the inverse operator must be told. ASR does not enter the rank:
it reconstructs within windows rather than removing a fixed subspace.

**Artifact events.** Blink and saccade events come from excursions in the
0.5–7 Hz band of `EEG057` / `EEG058`, thresholded at 3.0× the channel's median
absolute deviation with 0.4 s minimum separation. Saccades are split by the
sign of the `EEG058` deflection: the two directions have opposite field
patterns and cancel in a common average. `mne.preprocessing.find_eog_events`
is not usable: its threshold is set from the segment's extremes, so it reports
59 blinks/min over 60 s of H01 and 1.5/min over the same recording's full
315 s, against a cohort median of 16.9/min. R peaks come from
`find_ecg_events` on `EEG059`, which is stable to within 1.1% across lengths.

**Cohort check.** Both streams were run and measured on all 319 recordings.
Band-power and event-locked figures are from the run before the flux-jump
repair; the 307 recordings without a jump came out identical in every sidecar
field after it.

- *Rank.* DSS removes 20 dimensions at the median (rank about 250), ICA 3
  components (rank about 267).
- *Cardiac and ocular.* After DSS the event-locked averages for heartbeat,
  blinks and saccades sit at the level of averages locked to random events.
  ICA leaves the cardiac average 6.3 dB above that level at the median, more
  than 6 dB in 174 recordings, and selects no cardiac component in four of
  H16's. 28 saccade stages removed nothing, 20 of them for lack of events.
- *Spectrum, in unannotated time away from blinks and saccades.* DSS takes
  1.3 dB off alpha and adds 0.9 dB above 80 Hz (positive in 308 recordings);
  the gain is there before ASR. ICA takes 0.35 dB off alpha and 0.3 dB above
  80 Hz. Neither change differs between Fast and Slow within subjects (DSS
  above 80 Hz: -0.06 dB, 95% interval about +/-0.20), so neither biases the
  contrast, but the DSS change varies more between recordings (spread of
  Fast minus Slow 0.56 dB against 0.15 for ICA above 80 Hz, 0.36 against 0.07
  in alpha): DSS-stream power is the noisier of the two.
- *Muscle segments.* ASR lowers 110-140 Hz power inside annotated muscle by
  2.8 dB after DSS and 4.5 dB after ICA.
- *Flux jumps.* Twelve recordings, one jump each: H04 Fast 3, H06 Fast 4, H10
  Slow 2, H13 Fast 4, H19 Fast 3, H21 Slow 3, H22 RT 2, H23 Fast 2, H24 RT 2,
  H25 Slow 3, H27 Slow 4, H30 Fast 2. With the repair the jumped channels peak
  at 1-4 pT after either cleaner, from as much as 190 pT without it.
- *Bad channels.* The detectors flagged 270 channels in 72 recordings and none
  is a faulty sensor, so nothing is interpolated or dropped. 90 flags were the
  flux jumps. 119 were ocular field on the anterior temporal sensors
  (`MRT41`, `MLT31`, `MRT32` and neighbours), which both cleaners remove. 34
  were neck muscle. The rest were large slow deflections shared with the
  neighbouring sensors. Flagged channels correlate with their neighbours like
  any other (0.93 against 0.94), no channel is flat, and only 16 of 142
  subject-channel pairs recur in half the subject's runs. Fitting DSS on the
  unflagged channels only changes nothing on them. A power measure pooled
  over channels is dominated by any channel carrying a jump or a burst;
  measure per channel.
- *Neck muscle.* A plateau from about 20 to 100 Hz on the occipital rim
  (`MZO02`, `MZO03`, `MLO41`, `MLO51`, `MRO51` and neighbours), fading with
  distance and present through the whole recording. The same sensors sit
  within 2 dB of the array in the same-day empty-room recording of all 31
  subjects that have one, so it is the subject. Rim over the subject's median
  sensor, 40-100 Hz: H02 15.5 dB, H04 13.0, H16 11.0, H18 8.8, H21 8.5, H24
  8.0, H29 7.3, H03 7.0, H01 6.5. ASR is calibrated on the recording itself,
  so tonic muscle counts as its baseline and is barely reduced; the tonic
  muscle stages are there for it. H31, H27, H20, H23, H26 and H07 carry a
  smaller plateau of the same shape.
- *H10.* From its fourth recording on, bursts of 20-50 pT at 0.5-3 Hz on
  `MRT41`, `MRT51`, `MRT52`, `MRT42`. Elsewhere in the cohort the slow field
  at the helmet edge follows the EOG channels (64% explained at the median,
  7% by the head-position coils); in H10 the coils explain about 55% and the
  EOG about 5%. Something magnetic near the right temple moves with the head.
  Neither cleaner removes it. On the raw data about 14% of 3 s windows exceed
  4 pT peak to peak, 33-41% in Fast 3, Fast 4 and Slow 4.
- *Unexplained.* `MZF02` correlates with its neighbours at about 0.65 in 13
  subjects, with normal power.

**Open.**

- **Muscle z-threshold.** `annotate_muscle_zscore` at z=6/7/8 annotates
  2.15–2.63% of a recording at the median and scores 8.1–8.6 dB enrichment —
  all three *find muscle*. They were previously rejected on a 1% coverage
  budget, which is a different question from whether the detector works. The
  paper excludes jaw-clenching segments deliberately, so ~2.4% may be the
  correct amount; derive the coverage limit from that intent, not a round number.
- **Preprocessing / autoreject boundary** — see Epoching.
- **Evoked response on the occipital rim.** The tonic muscle stages shrink it
  in the nine people with tonic muscle (see Streams); noise or signal is
  undecided.
- **H10.** No subject-specific cleaning so far: count its surviving epochs
  after rejection and decide on exclusion from that.

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
| Artifact removal | one ICA pass; projector `I − ΦΦ⁺` on unfiltered raw | two parallel streams, both persisted | The explicit projector gained only 1–3.5 percentage points over `ica.apply` on the same components while adding >0.25 dB distortion on three of four subjects. Running two cleaners and keeping every artifact estimate makes the choice measurable rather than assumed. |
| Saccades | not addressed separately | DSS stage per saccade direction, events from `EEG058` | Removal at saccade onset rose from 20% to 35% across all 319 recordings, and from 35/23% to 58/48% per direction; better in 32 of 32 subjects. A direction with fewer than 10 saccades — 20 of 638, in 16 recordings, 14 of them RT runs — gets no stage: built from that few, a stage gained a median 4.9 points at onsets for 0.21 dB of alpha, about a quarter of the return at 10–14 events (17 points for 0.33 dB across both directions). |
| Muscle | excluded by visual inspection | ASR, statistics high-passed at 80 Hz, cutoff 20, after either cleaner; annotations kept | On all 319 recordings it removes 2.1 dB (after DSS) and 4.3 dB (after ICA) of 110–140 Hz power inside the annotated muscle segments, with no change in high-gamma, gamma or alpha power in clean time and evoked responses preserved in 30 of 32 subjects under both cleaners. A DSS muscle stage removed 0 components on the median recording; ICA muscle components removed nothing in muscle segments and stripped gamma elsewhere; cutoff 10 costs high gamma after ICA. |
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
