# MEG pipeline — plan

**Living document.** This is the pipeline as decided: what each stage does and
why it deviates from the paper where it does. It is not a record of how the
decisions were reached.

Companions: `docs/meg.md` (alignment QC, subject exclusion),
`docs/behavior.md` (behavioral pipeline and the trial-classification finding),
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
the specification below. It has not yet run on real data.

**Epoching and source reconstruction — not started.** Zero preprocessed raws, zero epochs, zero
source estimates, zero features.

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

### Preprocessing · `implemented, not yet run on real data`

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
