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

**Preprocessing — complete.** Both streams on all 319 recordings, through
the workflow (§4, Preprocessing).

**Epoching — implemented and checked on the cohort.** All 256 Fast and Slow
recordings, both streams, four locks (§4, Epoching). The epochs are not kept
yet.

**Source reconstruction — not started.** Zero source estimates, zero features.

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
for muscle bursts, a tonic muscle stage of its own method, so the DSS stream
contains no ICA, and the same head-movement regression. The streams differ in the cleaner and in the
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
| 5 | bad segments, pass 1 — excluded from cleaner *fitting* | `annotate_amplitude` peak `1.2485926067065662e-12 T` (a step between consecutive samples, not a peak-to-peak), flat `1.2485926067065662e-16 T`; `annotate_muscle_zscore` at 110–140 Hz |

Flux jumps come first because a jump is a step only until something filters
it: the 0.5 Hz high-pass turns a 360 pT step into about six seconds of ringing
that no cleaner removes. Twelve of the 319 recordings hold one jump each, on
1 to 22 channels at once; each is one flux quantum (345–375 pT) or two. No
other step in the cohort's raw data exceeds 19 pT, so the level sits in an
empty decade. ZapLine precedes the filter so the harmonics still exist. The trunk must stay
shared: computing bad channels or segments per stream would make the streams
differ in two ways at once and the comparison uninterpretable.

The muscle pass uses z = 7. At z = 6, 7 and 8 it annotates 2.15–2.63% of a
recording at the median, and what it annotates is 8.1–8.6 dB richer in
110–140 Hz power than the rest, so all three find muscle. The paper excluded
jaw-clenching segments by eye; about 2.4% of a recording is taken as that
amount.

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
| 8 | head movement — the nine head-coil channels (`HLC0011`–`HLC0033`: x, y, z of the nasion and ear coils) regressed out of the magnetometers, MNE `EOGRegression` |
| 9 | bad segments, pass 2 — residual muscle the stream missed |
| 10 | resample to 600 Hz — **last**, after the low-pass |
| 11 | persist: cleaned raw, artifact estimate, sidecar |

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

In the cohort run the stages do what the test predicted. The DSS stage
removes 2 components at the median from a recording of the 23 and 7 from one
of the nine, the ICA stage 1 and 5; 19% (DSS) and 38% (ICA) of the 23's
recordings lose none, and none of the nine's. The component count equals the
test's in 89% (DSS) and 82% (ICA) of recordings. Rim over the median sensor
at 65–100 Hz after cleaning: DSS stream 1.5 dB in the nine against −0.5 dB in
the 23, ICA stream 2.1 dB against 0.4 dB. The ICA stream starts higher (8.7
dB in the nine against 7.1 dB after the DSS cleaner), so its stage removes
more and still ends further from the 23. H02 stays highest in both streams
(4.7 and 4.1 dB), with H18 in the DSS stream (4.5 dB). The count does not
differ between Fast and Slow (3.9 and 3.8 DSS, 2.4 and 2.4 ICA). 12 of the
319 ICA fits did not converge, 3 each in H08 and H29.

What is left of the tonic muscle in the epochs was checked on all 32 people,
both streams, every kept epoch of the four locks. The rim here is the two
lowest occipital rows and the two lower midline sensors (`MLO41`–`44`,
`MRO41`–`44`, `MLO51`–`53`, `MRO51`–`53`, `MZO02`, `MZO03`), its power taken
over the median sensor's. Fast minus Slow, 65–100 Hz: the nine +0.17 dB and the
23 +0.07 dB in the DSS stream (permutation p 0.61), −0.05 and −0.01 dB in the
ICA stream (p 0.85); the same on every lock and at 30–55 Hz. Within people,
Fast and Slow runs differ in one person per stream, as chance gives: H16 has
more rim power in Slow (ICA), H31 in Fast (DSS). The DSS figure for the nine
comes from one run, H24 Fast 4, at 7.1 dB against 2.0–2.9 in that person's
other runs; it holds the most tonic muscle of H24's runs (7 components
removed, none reconstructed by ASR), and the ICA stream brings it to 2.7 dB.
Locked to the go cue, the press, the feedback or the decision, rim power moves
by under 0.1 dB in either group; the nine rise 0.05 dB more than the 23 around
the decision (−0.25 to +0.5 s; p 0.003 DSS, 0.04 ICA). The residual muscle is
therefore not time-locked to the trial and does not differ between Fast and
Slow in the group; a single person's result can still carry it.

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
  4 pT peak to peak, 33-41% in Fast 3, Fast 4 and Slow 4. H10 is kept, with
  no subject-specific cleaning. Under the epoch rule it keeps 419 of 480
  go-locked Fast and Slow epochs (DSS) and 429 (ICA), losing most in Fast 4
  (39 of 64 kept, DSS) and Slow 4 (42 of 57). The bursts carry no mark and stay in the data.
  They are not only slow: in the kept epochs the right temporal sensors
  (`MRT51` worst) also carry excess power above 30 Hz, strongest at 65–150 Hz
  (see Epoching, unmarked artifacts).
- *Unexplained.* `MZF02` correlates with its neighbours at about 0.65 in 13
  subjects, with normal power.

**Open.**

- **Evoked response on the occipital rim.** The tonic muscle stages shrink it
  in the nine people with tonic muscle (see Streams). Rebuilt before and after
  the stage on all 72 Fast and Slow runs of the nine (the rebuild reproduces
  the stored recordings to 4e-8 and the same component counts), go-locked,
  40 Hz low-pass, rim over 0–0.5 s, with the noise measured by the ±average
  (half the trials sign-flipped). DSS stream: what the stage removes is noise.
  The evoked power falls 0.85 dB and its noise 2.2 dB (medians); summed over
  the nine the noise is all of the evoked power removed, and the signal moves
  by +2% of the evoked power at the median (it falls in 4 of 9, Wilcoxon p
  0.65). Signal over noise improves in eight of the nine. ICA stream: the
  evoked power falls 1.4 dB and its noise 2.5 dB; noise is 57% of what is
  removed and the signal falls in 7 of 9, by 10% of the evoked power at the
  median (p 0.13), most in H18 (31%), H21 (25%) and H29 (17%). The ICA tonic
  stage likely takes some evoked signal from the rim in these people; the DSS
  stage does not. Rim waveforms differ too much between people (the 23 match
  each other at 0.24) for the shape to tell.

### Epoching · `checked on the cohort`

Fast and Slow runs, both streams, four locks: `go` (−0.5 to +1.0 s), `enter`
(button press), `feedback` and `decision` (press minus the hand-specific motor
baseline), each −1.5 to +2.0 s, as in the paper. RT runs are not epoched; they
only measure the motor baseline. The recordings carry Start and Go triggers
only, so press, feedback and decision sit at their behavioral offset from the
go cue. Metadata: the trial features (corrected-frame trial class, condition,
choice, correctness, `dt_ms`, SPD) and the trial timings.

**Padding.** Each epoch is cut with 1 s of margin on both sides of its
analysis window. Band power is computed in source space after the inverse —
band-pass, then Hilbert envelope — and the margins are trimmed after the
envelope, so filter edges fall outside the analysed time. Near the start or
end of a recording the margin is as long as the recording allows: the samples
beyond it are zero, each epoch's metadata gives its recorded margin
(`pad_before_s`, `pad_after_s`), and only the recorded span is filtered.

**Epoch rejection.** An epoch is dropped when its analysis window (not its
margins) touches any bad-segment annotation: `BAD_jump`, `BAD_muscle`,
`BAD_peak` or `BAD_flat`, or runs past the recording. An epoch the marks
keep is then dropped as an outlier when the log peak-to-peak of any
magnetometer over its analysis window exceeds that sensor's cohort median by
5 robust SDs (1.4826 × the median absolute deviation, per stream and lock,
over every subject's epochs). What the marks leave is abnormal against the
other people, not within its own recording (see the unmarked artifacts below),
so only a cohort reference finds it; the thresholds are computed from every
subject before any epoch is written.
No sensor is repaired; each recording's bad sensors are left out of its
forward and inverse model. For decoding, Fast and Slow trial counts are
equalised within each subject.

On all 256 Fast and Slow recordings (DSS stream; the ICA stream is within
0.1 point on every lock):

| Lock | Events | Kept | Lost | Window touches a mark | Window runs past the recording | Kept with a short margin |
|---|---|---|---|---|---|---|
| go | 16,290 | 15,257 | 6.3% | 979 | 54 | 32 |
| enter | 16,280 | 14,632 | 10.1% | 1,480 | 168 | 44 |
| feedback | 16,290 | 14,572 | 10.5% | 1,529 | 189 | 45 |
| decision | 16,280 | 14,676 | 9.9% | 1,452 | 152 | 44 |

Fast loses 6.5% of go-locked epochs and 10.2–10.6% on the longer locks, Slow
6.1% and 9.5–10.5%. Per subject the median loss is 5.1% (go) and 9.4–9.5%
(longer locks); H21 and H20 lose 27–29% on the longer locks, H12 and H04
17–20%. The windows that run past the recording include the 20 reconstructed
go cues, which by construction fall after their recording stopped. 1.4–2.0% of
trials are kept in one stream only, from each stream's own second-pass muscle
marks.

Checked on every one of the 2,048 files: the kept epochs equal an independent
recomputation of the rule in MNE's sample arithmetic; each lock sits at its
behavioral offset; go cues fall 2.2 ms after trial start plus `tGO` at the
median (8.7 ms at worst), so no trial is mispaired; the trigger channel and
`events.tsv` agree within half a sample; the recorded part of each epoch equals
the cleaned recording sample for sample and the rest is zero; 299 channels at
compensation grade 3, 600 Hz.

Marks fall evenly on Fast and Slow across the cohort (6.2% against 5.8% of
go-locked events; Fast higher in 17 of 32 subjects) but not within every
subject: H13 has 12 points more in Slow, H20 11 points more in Fast. After all
cleaning, an epoch touching a muscle mark still carries 1.6 dB more 60–90 Hz
power than clean epochs on its top 2% of sensors (0.5 dB on the median
sensor), from 0.7 dB when under 0.25 s of it lies in the mark to 2.8 dB when
over 1 s does.

**Unmarked artifacts in the kept epochs.** Every kept epoch of the cohort,
both streams, four locks, analysis window, 270 magnetometers: an epoch counts
as an outlier when any sensor's log peak-to-peak lies more than 5 robust SDs
above that sensor's cohort median (rule fixed beforehand). 1.6–2.0% of kept
epochs are outliers, and at the median subject none.

| Subject | Outlier epochs (DSS / ICA) | Where | What |
|---|---|---|---|
| H10 | 44–46% / 48–51% | right temporal, `MRT51` | broadband, strongest at 65–150 Hz (robust z +9.7 DSS, +6.5 ICA) and below 4 Hz (+5); its other epochs are raised on the same sensor too |
| H18 | 8–10% / 5–13% | occipital rim, `MRO51`, `MLO52` | below 4 Hz (+4.5); in DSS also 30–150 Hz; most in Fast 1 |
| H02 | 7.5% go, under 1% other locks / 4% go | right temporal `MRT54`, occipital | 4–30 Hz (+4.8); go-locked, most in Slow 3 |
| H31 | 0–0.2% / 1.6–2.2% | right temporal, `MRT42` | below 30 Hz; ICA stream only |
| H16 | 0–0.2% / 0–0.2% | — | none |

H16 carries nothing the marks miss; H31 only a little, and only in the ICA
stream. H18's slow occipital excess and H10's right temporal one escape the
marks in a share of epochs that matters, H10's in about half of them and into
high gamma.

Sensors and causes, from the outlier epochs and from 1 s windows of the
continuous recordings outside marks, each person against the other 31 on the
same sensor:

- *H10:* nine sensors carry most of it (DSS; six in ICA), right temporal
  (`MRT41`–`43`, `MRT51`–`54`) and left temporal (`MLT51`, `MLT52`); five
  sensors per outlier epoch at the median, up to 83. Its slow amplitude rises
  with 65–150 Hz power (Spearman 0.77; the others 0.02–0.61) and with head-coil
  movement (0.59; the others at most 0.23), and in its largest bursts the slow
  amplitude doubles on 133 sensors and the high-band power on 80. Movement
  episodes with temporal muscle (jaw or head), not a steady source.
- *H18:* one to three occipital rim sensors (`MLO51`, `MRO51`, `MZO03`;
  `MLO52`). Not muscle (no link to high-band power). The head-coil positions
  explain 35% of the slow signal there (the others at most 6%), and H18 moves
  the most of the cohort: the back of the head moving against the nearest
  sensors.
- *H02:* four right temporal and occipital sensors, small, mostly Slow 3.
- *H31, ICA stream only:* eight right temporal sensors; a slow field that
  doubles on 211 sensors in its bursts, without muscle or movement. DSS
  removes it, so it is ocular or cardiac field the ICA stream leaves.

Why the marks miss them: `BAD_peak` (`annotate_amplitude`) marks spans where
consecutive samples differ by at least the threshold, a step detector, not a
peak-to-peak one; a 50 pT wave at 3 Hz moves under 1 pT per sample at
1200 Hz, so slow excursions are never marked and `BAD_peak` drops no epoch in
the cohort. `BAD_muscle` averages its z-score over all 270 sensors, so muscle
on a few temporal sensors is diluted.

**Head-movement regression, two forms, on all 32 people** (go and press locks,
both streams, against the epochs as cut). *Continuous:* MNE's
`EOGRegression` with the nine head-coil channels as regressors, fitted and
applied on each cleaned recording. *Trial-wise* (Stolk et al. 2013, FieldTrip
`ft_regressconfound`): each epoch's mean coil position regressed out across the
subject's trials at every sensor and time point.

| | Continuous | Trial-wise |
|---|---|---|
| H18 outlier epochs | 5–11% → 1.1–1.7% | 5–11% → 6–10% |
| H18 slow power on its rim sensors | −1.4 to −1.7 dB | −0.1 dB |
| H10 outlier epochs | 44–51% → 43–47% | 47–55% |
| H02, H31 | unchanged | unchanged |
| Others: band power above 4 Hz | unchanged (worst sensor 0.2 dB) | −0.08 dB in every band |
| Others: Fast − Slow maps | r ≥ 0.999 | r 0.95–1.00, worst sensor 1.3–1.5 dB below 8 Hz |
| Others: evoked shape | r ≥ 0.997 | identical |
| Others: evoked signal over noise | +0.06 to −0.06 dB, half improve | +0.07 to +0.18 dB, 21–25 of 28 improve |

The continuous form removes H18's artifact and leaves everyone else as they
were; it is the head-movement step of preprocessing (step 8). The trial-wise form fixes nothing here: its uniform −0.08 dB is the
variance any nine regressors absorb from about 470 trials (10 log10(1 − 9/470)
= −0.08 dB), which also accounts for most of its evoked gain, and it shifts the
Fast − Slow maps below 8 Hz because head position differs between Fast and
Slow runs (1.1 mm at the median, 6.2 mm in H30). Neither form reaches H10.

**Marks that look only at the recording itself do not find what is left.**
On 1 s windows of all 319 recordings, both streams, after the continuous
head-movement regression: an artifact window (any sensor's log peak-to-peak
above that sensor's cohort median by 5 robust SD) makes up 0.09–0.14% of the
28 people's unmarked time and 40% of H10's. MNE's muscle score per sensor
region (11 regions, threshold 7) marks 0.07–0.09% of the 28's time against
0.05–0.08% for the whole array, and catches under 1% of the artifact windows,
H10's included: the score z-scores each sensor against its own recording, and
H10's temporal muscle is present for so much of it that it sets the baseline.
A peak-to-peak rule against the recording's own median catches at most half of
the artifact windows (k = 6 times the median over all sensors, nearly all of
them in H10), against a sensor's own median under a tenth. What remains after
the regression is abnormal against the cohort, not within its recording,
hence the outlier rule of the epoch rejection.

**Why no sensor is repaired.** Leaving ten sensors out at a time and
rebuilding them with MNE's field interpolation from the rest, on all 638
cleaned files (median sensor per subject, then cohort median, DSS stream):

| Band | Correlation with the real sensor | Variance not recovered |
|---|---|---|
| below 30 Hz | 0.98 | 4% |
| alpha | 0.985 | 3% |
| beta | 0.96 | 8% |
| 30–55 Hz | 0.88 | 23% |
| 65–100 Hz | 0.78 | 42% (ICA stream 60%) |

Below 30 Hz the worst subject is at 0.97, sensors on the helmet edge at
0.90–0.92, and 0.8% of sensor-recordings fall under 0.8; `MZF02` is the worst
sensor (0.78). A rebuilt sensor is trustworthy below 30 Hz and not in the
gamma and high-gamma bands the paper's speed-accuracy results come from.
The interpolation needs an explicit origin, `(0, 0, 0.04)` m in head
coordinates, since the files carry no digitised head points. It uses the 270
sensors alone and ignores the compensation grade; accounting for the
reference sensors recovers under one percentage point more.

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
| Head movement | not stated | head-coil positions (`HLC` channels) regressed out of each cleaned recording | On all 32 subjects it brought H18's movement-locked occipital outliers from 5–11% to 1–2% of epochs and left the others unchanged above 4 Hz (evoked shape r ≥ 0.997). The trial-wise alternative (Stolk et al. 2013) fixed nothing here and shifted the Fast − Slow maps below 8 Hz. |
| Epoch rejection | not stated | marks in the analysis window, then a cohort outlier rule on peak-to-peak; no sensor repaired | Sensor repair (autoreject) loses 42–60% of 65–100 Hz variance. What the marks leave is abnormal only against the cohort (H10: 40% of its time), which no rule within one recording finds. |
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
| Missing go cue | 21 | 20 Fast/Slow runs: `KNOWN_GO_RECONSTRUCTION_RUNS`, go-cue reconstruction; `H02RT1`: RT, not epoched |
| Unrecoverable real trial | 1 (`H12Slow2`, trial 2) | `KNOWN_UNRECOVERABLE_TRIALS` |
| Scratch files | 3 | `behavior_ignore_files` |

`184 + 114 + 21 + 1 + 3 = 323`: every source file has either a verified
resolution or an explicit exclusion. Reproduce with `meg-tokens behavior qc`.

### Trailing-boundary mismatches

All 114 runs align throughout their recordings and differ only at the final
boundary. Examples include one extra MEG trial-start pulse (`H03RT1`),
missing TDMS-side events (`H06Slow2`, -12), and the all-go-channel case
(`H02RT1`, -39). Maximum confirmed timing residual was 4.25 ms.
`align_go` truncates only these documented runs and leaves unknown or
interior mismatches as hard errors; this recovers valid epochs from
approximately 36% of the production runs without changing MEG/behavior
pairing.

### Missing go cues

Twenty Fast and Slow runs each lose one go cue, their final trial's: the
recording stopped between its start pulse and its go cue. For each,
`reconstruct_missing_go` pairs trials with trial-start pulses and places the
missing go cue at `trial_start + tGO + lag`, the lag calibrated from that
run's real go cues. Never-started `7003` trials are never reconstructed,
including mixed-pattern runs such as `H07Slow2` and `H07Slow4`. `H02RT1`, with
about 39 trials and no recorded go cue at all, is an RT run and is not epoched.

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

Implemented in `meg_tokens/meg/epoching.py` (`align_go`: never-started
filtering, trailing policy, go-cue reconstruction, the H12 exception); the scratch-file
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
