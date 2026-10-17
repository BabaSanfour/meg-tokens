# Analysis stage: audit and plan

2026-10-16. Findings of eight read-only reviews done before building the
analysis stage after source reconstruction: two audits of the never-run code,
the installed MNE, the preprint's pipeline, coco-pipe, the CoCo Lab trajectory
primer, best practice for source band power, and methods for the planned
time-scaling, urgency and deep-structure tests. Decisions taken from it move to
`docs/meg.md`; this file records the evidence.

Decided by the user: every analysis after source reconstruction goes through
coco-pipe (latest `main`, `7f17473`); meg-tokens keeps only the glue coco-pipe
lacks. The old analysis code is a source of ideas, refactored later, not fixed.
PCA is run both ways, the preprint's and the primer's. Connectivity and ERP
stay in the plan.

## 1. The never-run code

Nothing after `source` can run: every module reads the deleted per-trial
manifest or old file names with no stream (`proc-`) and no space, so DSS and
ICA outputs and the three spaces would overwrite each other
(`io/layout.py:407-746`). No Snakefile rule exists after `source`; no test
covers these modules. `meg/__init__.py:21-27` lazily exports
`apply_inverse_operator`, `morph_source_estimates` and others that
`meg/sources.py` no longer defines (AttributeError on access).

Scientific errors in it, to avoid in the new code:

| Where | Error |
|---|---|
| `features/time_frequency.py:106-113, 170-177`; `workflows/power.py:229` | filters across the zero padding; margins never trimmed |
| `features/erp.py:70,75` → `workflows/hilbert.py:121` | NaN padding through an FFT Hilbert makes every truncated trial all-NaN |
| `features/erp.py:96,177`; CLI `--label-mode mean` | region time course is a plain mean, so opposite-facing sources cancel; empty labels zero-filled; HCP-MMP1 `???` kept |
| `features/time_frequency.py:238,252-257`; `workflows/power.py:179` | free-orientation power never summed over x/y/z |
| `features/time_frequency.py:8-15` | wrong bands (δ2-4, α8-15, β15-30) and default transitions: γ and high γ overlap at 52-68 Hz |
| `features/time_frequency.py:381-410` | baseline outside the epoch silently becomes the whole epoch; zero baseline replaced by 1e-10 |
| `analysis/decoding.py:133,207` | labels permuted across subjects, unseeded; Fast/Slow is a block label |
| `analysis/decoding.py:21-36, 88-90` | classes balanced globally, not within subject; CV falls back to folds that mix subjects |
| `workflows/statistics.py:469` | "lateralised" statistics ignore the response hand |
| `decomposition.py:191,218`; `statistics.py:114,131` | `nanmean` over NaN padding averages late bins over slow trials only |
| `features/erp.py:43-46`; `workflows/erp.py:225,303` | press from `tEnterTarget − tGO` (epochs use `rawRT`); behaviour rows matched by position |
| CLI defaults `--source-method dSPM` | sLORETA was decided |

Worth keeping as ideas: band-pass then |Hilbert|²; a window helper rewritten in
seconds; `features/dataset.py` loaders and `lateralize_labels`;
`fit_condition_pca`; LOSO LDA with `SlidingEstimator`; max-t sign-flip
permutation; `compute_motor_lateralization` with side 1 = right hand.

## 2. MNE 1.13.2 on our mixed operators (read in the installed source, checked synthetically)

- `apply_inverse_epochs(..., pick_ori="normal")` raises on a mixed source space
  (`utils/check.py:1043`). `pick_ori=None` returns a rectified magnitude
  (nonlinear, never to be filtered); `"vector"` is linear.
- The kernel can be built directly: `prepare_inverse_operator(inv, 1, 1.0,
  "sLORETA")` then `_assemble_kernel(inv_p, None, "sLORETA", None)` gives
  `K` (3·n_src × n_ch, whitener and regularisation included) whose sLORETA
  factor is one scalar per location, so rows stand alone. Cortex normal rows:
  `K[2:3*ncort:3]` (rows are t1, t2, normal); volume rows are x, y, z. Checked
  equal to MNE's own `project('normal')`, and linear (filtering before or after
  gives the same result).
- `extract_label_time_course` on a mixed estimate returns one plain-mean row per
  volume source space: all 17 deep structures collapse into one row. Volume
  labels work only on `src[2:]` with `mri_resolution=False`.
- `compute_source_morph` fails on the mixed space (no interpolator; dipy
  absent); the cortex alone works: `compute_source_morph(inv['src'][:2], ...,
  src_to=<fsaverage oct6>)` gives a sparse linear `morph_mat`.
- `morph_labels` writes morph maps into `subjects_dir` (needs write access on
  `fir`).
- `source_band_induced_power` is Morlet, trial-averaged, and silently takes the
  z component of volume sources: unusable here.
- Filters at 600 Hz with MNE defaults: lengths 1.65 / 1.47 / 0.83 / 0.44 /
  0.22 s for θ/α/β/γ/high γ; edge transient about half the length. The default
  transitions make the bands overlap (effective −6 dB bands θ 4-9, α 7.9-14.6,
  β 14-27, γ 26-67.5, high γ 52.5-101 Hz). `Epochs.filter` pads with `edge`,
  `filter_data` with `reflect_limited`. `apply_hilbert` `n_fft='auto'` is a
  5-smooth length, zero-padded.
- Before sharing labels or a morph across a subject's runs, assert identical
  `vertno` across its operators.

## 3. Best practice for source band power

| Choice | Recommendation | Source |
|---|---|---|
| order | band-pass and Hilbert in sensor space, project real and imaginary parts through the kernel; exact by linearity, cheaper | MNE `apply_hilbert` docs; Larson (forum 822, issue 6290); Hipp 2012; Brookes 2011 |
| filtering | on the continuous run before epoching, explicit transitions; else filter only recorded samples and invalidate windows within half a filter length of a recording edge | MNE filtering tutorial; Brainstorm |
| power | \|z\|²; free orientation: sum of \|z\|² over x/y/z (rotation-invariant; summing amplitudes is not) | Brainstorm; Larson (issue 6109) |
| region power | mean of per-vertex power; envelope of the sign-flipped mean only as a sensitivity check (it depends on phase coherence within the region); no head-to-head benchmark found | Brainstorm ("always prefer"); MNE label power |
| inverse | within-subject log-power contrasts are the same for MNE, dSPM and sLORETA with one operator, λ² and `nave`; noise normalisation does not change cross-talk | Hauk 2011; Brainstorm |
| group space | morph power (sparse matrix, after windowing), never signed time courses | MNE `SourceMorph`; McCloy and Hämäläinen (forum 786) |
| normalisation | log10 power centred per subject × region × band × stream; a pre-trial baseline erases block (SAT) effects present before the trial, and single-trial baseline division is biased | Hipp 2012; Grandchamp & Delorme 2011; Gyurkovics 2021 |
| 360 regions | exceed the data rank; check vertices per region | Tait 2021 |

## 4. The preprint's MEG pipeline as its code ran it

From `archive/replicated/DDM_scripts/` (`scripts_new/`, `matlab_scripts/`) and
the NeuroMath toolbox in `archive/DDM_scripts/matlab_scripts/`:

- Cortex: single-trial dSPM, **λ² = 1/9** (snr 3; the paper says 1), normal
  orientation, empty-room covariance, single-layer BEM, morphed to fsaverage
  2×4098. A second mixed run (deep structures) used λ² = 1, 200 Hz and no
  `pick_ori`, i.e. non-negative magnitudes that were then Hilbert-transformed.
- Power: brainpipe `fir1` of order 3 cycles of the low edge (the paper's
  "FIR1 order 3"), `filtfilt`, |Hilbert|², 500 ms windows at 50 ms steps.
  **No baseline, log or z-score** before PCA; baselines only in figures.
- PCA: trial → block → subject nanmeans per 12 conditions (correct and error
  pooled), then the **grand average over 28 subjects**, a (12·T) × 8196 matrix
  (matching Fig. S2, not the Methods' 20496 × 8196); MATLAB `pca`, centred, not
  scaled, 20 PCs; per-subject **uncentred** projection, averaged over subjects.
- LDA: one PC at one time bin as the single feature, subject × condition means
  as samples, leave-one-subject-out, `permutation_test_score` with 95
  permutations (paper: 1000), max over a figure-dependent window.
- Defects: success-probability correlations with a fitted lag per band, sign
  flips and swapped misleading pairs, 23-25 of 28 subjects, conditions counted
  twice, p-values averaged; "beta PC2" is PC4 in the code; the top-5% loading
  threshold keeps 10-13%; `DEEP_enter.m` indexes cortical sources (missing
  +8196 offset); go-lock deep labels shifted by 250 ms; a mean-imputer fitted on
  the test subject; go-lock bins kept up to press +200 ms (survivor bias).

## 5. The CoCo Lab primer (arXiv 2609.32315; github.com/thecocolab/pca-neural-trajectories-eeg-meg)

PCA through coco-pipe `DimReduction` (centring only). One PCA per band; fit in
a shared space on subject × condition means (equal weights), single-trial fit
as a check; band power as per-trial dB against a pre-stimulus baseline, 40 ms
Gaussian smoothing (its notebooks warn the smoothing creates part of the
trajectories' smoothness); fit 10-20 PCs, report scree, participation ratio,
reconstruction error, loadings (mandatory), robustness to baseline and K; the
participant is the unit: SEM over participants, labels permuted within
participant, max-statistic over bands × contrasts; PCA and Procrustes alignment
fitted on training folds only for decoding. Time scaling is beyond the primer.

## 6. coco-pipe (`main` `7f17473`, v0.0.3 + 8)

| Step | coco-pipe | Glue in meg-tokens |
|---|---|---|
| source band power | none | all of it |
| normalisation | `DataContainer.aggregate`; decoding `use_scaler="subject"` | log10 and centring |
| PCA trajectories | `DimReduction`; `evaluation/geometry.py` speed, path length, separation, curvature; `TrajectoryResult`; `permutation_null_separation_auc(within_subject=True)` | the fit matrix, held-out loop, NaN-free time grid |
| decoding | `Experiment` (sliding/generalizing LDA, LOSO, max-stat permutations refitting the pipeline) | balancing within subject |
| time scaling | none (`TemporalProcrustesAlignment` aligns participants in space) | quantile bins, nearest-point timing |
| statistics, reports | BH, max-stat, HTML reports, temporal plots | cluster tests (mne.stats), cortical plots |

Gaps for coco-pipe itself: decoding permutations refuse labels that vary within
a unit (`decoding/stats.py:_run_permutation_loop`), so choice and Fast/Slow can
only be shuffled globally; no cluster permutation; no NaN windows; no time
warping; no cortical plots. Not installed in the meg-tokens environment (needs
pyriemann, hydra-core, plotly, markdown, openpyxl, filelock; matplotlib < 3.13).
Container: one xarray `(obs, band, region, time)` per subject × stream × lock,
trial table as `obs` coordinates.

## 7. Planned tests (literature)

Time scaling ("faster clock"):
- Quantile-matched bins: 5 decision-time quantiles per subject and condition,
  token-locked, cut before the bin's earliest commitment; a faster clock
  predicts Slow(t) ≈ Fast(t/1.12) in every bin.
- Relative speed by nearest-point time (KiNeT; Remington 2018), not raw ‖Δx‖/Δt.
- Scale vs shift fitted per subject, cross-validated; permutation of Fast/Slow
  at the run level within subject.
- Same endpoint at commitment; token-locked (5 Hz) dimensions as a negative
  control.
- Matching on absolute decision time conditions on a collider (tails, motor
  noise, lapses) and is dropped.
- Windows of 500 ms are wider than the expected 120-180 ms gap: ≤ 200 ms or
  causal smoothing for these tests, validated on simulated data.
- Risk: in monkeys the tokens-task block effect is a baseline offset, not a
  faster rate (Thura & Cisek 2016; Thura et al. 2022).

Urgency in motor beta: response-hand-referenced beta in motor parcels, urgency
U = (contra + ipsi)/2, evidence L = ipsi − contra, one shared baseline;
U(t) = b + u·t per condition (rate vs offset); evidence weight growing with
elapsed time (Steinemann 2018; Murphy 2016; Hanks 2014).

Deep structures: basal ganglia, thalamus and amygdala are 77-93% cortex through
sLORETA here, consistent with Attal & Schwartz 2013, Krishnaswamy 2017 and
Hauk 2022: no claim there. Cerebellum (about 50% own) only with a forward
simulation of the cortical effect through each subject's leadfield and the same
pipeline, a laterality test, and a phasic-vs-tonic muscle check (Andersen 2020).
The across-subject correlation (n = 32: 80% power only for r ≥ 0.47) is
supporting evidence only.
