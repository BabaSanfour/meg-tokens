# PCA trajectories: the preprint replicated, audited, and redone our way

The programme for the PCA stage, and where it stands. Three tracks, every
result in both streams (DSS and ICA), all 32 subjects:

- **A. Replicate.** Every PCA result of Thiery et al. (2022) reproduced with
  the preprint's method (its figures and numbers), keeping only the choices
  this project has decided (§1). No decoding or other machine learning in
  this stage: where the preprint tested a component with LDA, the component
  is tested with statistics on its scores instead.
- **B. Audit.** For each step of the preprint's method, whether it is sound;
  where it is not, the result redone with the step corrected, to see whether
  the claim survives.
- **C. Our method.** The PCA planned for this project (`meg.md` §2, Analyses):
  the pooled fit on log10 centred and dB power, the event-related fields,
  trajectory geometry and statistics with the participant as the unit. Which
  of A's results hold there and which do not.

Decided elsewhere and kept in every track: `meg.md` §2 (sLORETA, both streams,
the corrected trial classes, 600 Hz). The preprint's claims, with page and
figure, were inventoried from its PDF (`2022.06.14.494674v1.full.pdf`).

## 1. How the preprint's method is matched

| Preprint | Here | Why |
|---|---|---|
| dSPM, λ² 1/9 (its code), 8,196 vertices | sLORETA, λ² = 1; 360 HCP-MMP1 regions and the 8,196 fsaverage sources | `meg.md` §2; a finding should not depend on the inverse |
| 28 subjects | 32 | `meg.md` §7 |
| ICA cleaning | DSS and ICA streams | `meg.md` §2 |
| chosen-target classes (`nProb`) | correct-target classes | response-confounded (`behavior.md`); theirs run only where B needs it |
| 500 ms windows, 50 ms steps | every sample at 600 Hz; 100 ms smoothing for figures only | decided; it lowers gamma's explained variance (§3) |
| PCA of the grand average, press lock, −1.25 to +1.2 s, 20 PCs, projected uncentred | the same: `analyze trajectories --fit grandaverage --measure power --window -1.25 1.2` | — |
| commitment = press − 448 ms | press − each subject's motor latency (449 ms on average) | — |

## 2. Status, claim by claim (track A)

| # | Claim (preprint) | Status | Result, DSS [ICA] |
|---|---|---|---|
| 0 | Bands from FOOOF on sensor spectra, press lock, 1-35 Hz (Fig S1, p6, p25) | done | replicated (magnetometer spectra, 32 x 270 per stream, Welch 0.5 Hz, specparam without a knee, FOOOF's defaults): exponent 0.94 [1.03] vs 1.00, R2 0.987 [0.987] vs 0.975, 3.5 [3.4] peaks per spectrum vs 3.3; peaks at 5-6, 9-12 and 16-23 Hz, and the small 29-30 Hz bump of its histogram. Above the mean bin the bands are 5-7, 9-14, 16-23 Hz (vs 5-8, 9-13, 16-24), but the edges move with that undefined threshold: 0.75 times it merges all into 5-23 Hz, 1.25 times gives 5-7, 9-13, 19-23 |
| 1 | Variance per PC (Fig 2, p25) | done, regions | alpha 54/21/11 % [61/18/10] vs 55/22/8; beta 78/14/5 [79/13/4] vs 70/14/5; theta 30/25/20 vs 48/22/9 (a press transient takes PC1-2); gamma 53/9/4 and high gamma 14/8/6 vs 44/17/11 and 43/17/13 (600 Hz, no windows) |
| 2 | Alpha PC1 peaks before the movement; PC2 falls to the press (p11) | done | PC1 peak −473 ms [−465], at commitment; PC2 minimum +152 ms [+180] |
| 3 | Alpha bends at commitment, beta only at the press (Figs 4-5) | partly | beta's turn at the press: breakpoint −29 ms [−32], clear in PC1 x PC2; alpha bends before beta (breakpoint, 22-30 of 32 subjects), not reliably at commitment: −410 ms by peak curvature, −276 ms by breakpoint, later with a wider search window |
| 4 | Beta PC1 condition-independent suppression and rebound (~+600 ms); PC2 choice, crossing after the press (Fig 5) | done | PC1 minimum +132 ms, rebound peak +740 ms [+747]; PC1 varies 25 times more over time than across conditions. PC2 = right minus left sensorimotor beta (homologues r −0.98): right − left choice dz +0.76 before commitment, +0.98 commitment to press, −0.99 late [+0.71, +0.96, −0.98] |
| 5 | Choice separates on alpha PC3 from −350 ms, beta PC2 from −500 ms (its LDA) | done, without the LDA | right − left on each PC's scores in 50 ms bins, cluster permutation over time with sign flips per subject (MNE, 5,000), press lock. Replicated: beta PC2 −874 to +276 and +426 to +1,176 ms, both streams (preprint −500 to −50, +250 to +1,100); alpha PC3 from the window's start (−1,226) to +676 (preprint −350 to +800); theta PC3 −174 to +126 [−124 to +76] (preprint −400 to +600, narrower here); beta PC4 +476 to +726 [+526 to +726] (preprint +400 to +900). Not replicated: alpha PC4, −274 to −24 here against +550 to +1,100 |
| 6 | PC time courses track success probability (Fig S8) | done | with a fixed-target (right) success probability: beta PC2 R +0.62 [+0.64], alpha PC3 +0.68 [+0.50], theta PC3 +0.05 [+0.05] vs +0.83, +0.78, −0.01. With the chosen target's: condition-independent PCs score highest (beta PC1 −0.89) |
| 7 | Speed-accuracy (Fast/Slow) components: theta PC1, beta PC3, gamma PC2, gamma before the trial (its LDA) | done, without the LDA | the same test on Fast − Slow. After the press only: beta PC3 +276 to +576 and +726 to +1,176 ms, both streams (preprint +800 to +1,050); beta PC4 +676 to +1,176 (preprint +900 to +1,100), both in the window the post-press token schedule could set apart. Not replicated: theta PC1 (+250 to +1,200), theta PC4 (+450 to +750), gamma PC2 (−1,000 to −400, +700 to +1,100), none here; on the first-token lock, projected on the press components, no Fast/Slow cluster before the first token in any band, so gamma "before the trial" is absent. Alpha PC4 differs in DSS before the press (−1,226 to −674, −474 to +776), in ICA only after it. Clusters are corrected over time, not over the 20 components |
| 8 | Leading loadings, top 5 % by HCP-MMP1 area (Figs 2-5) | done, regions | alpha PC1 posterior cingulate 50 %, visual 39-44 % (vs PCC 63 %, V1 14 %); beta PC1 somatomotor 50-56 %, premotor 33 % (vs somatomotor 42 %, PCC 13 %); theta PC3 premotor 33-39 %, somatomotor 28 % (vs motor 35 %, premotor 16 %); theta PC1 and beta PC4 do not match (their "subcortical" shares) |
| 9 | Speed-accuracy components "mostly subcortical" | done | not replicated: with the 17 deep structures fitted beside the 360 regions, no deep structure is among any component's top 5 % (0 of 19, every band, PC 1-4, both streams), and the deep structures carry 0.2-5.4 % of a component's weight, about their 4.5 % share of the sources. Fast/Slow: none before the press except alpha PC4 (dz +0.47, DSS only); after it, beta PC3 (+0.6 to +1.2 s, dz −0.90 [−0.88]) and PC4 (−0.66 [−0.55]), where the preprint put beta PC3 (+800 to +1,050 ms), but in the window the post-press token schedule could set apart (`analysis_hypotheses.md` §2.4); gamma PC2 shows none (dz −0.2 [+0.2], the preprint's "before the trial" effect needs the go lock) |
| 10 | Deep-structure PCA, first-token lock, Fast/Slow on PCs 1-4 (Fig 6D) | running | per-voxel deep power, morphs onto fsaverage's grid done; deep means and the six structures' PCAs queued |
| 11 | Claims 1-4, 8 on the 8,196 vertices | done | variance alpha 42/29/9 % [51/24/8], beta 75/17/3 [76/16/3], theta 37/27/13 [35/26/14], gamma 48/11/3 [55/12/4], high gamma 8/5/3 [21/6/4]; alpha PC1 peak −327 ms [−312], beta PC1 minimum +130 ms, rebound +732 ms [+738]; loadings alpha PC1 posterior cingulate 39 %, early visual and V1 24-43 %; beta PC1 somatomotor 65 %, premotor 25-27 %; no component's top 5 % reaches the medial wall, the preprint's "subcortical" label |

## 3. The preprint's method under audit (track B)

Each step, what is wrong with it, and how its effect on the claim is measured.

| Step | Issue | Test | Status |
|---|---|---|---|
| Band edges (claim 0) | set where a histogram of FOOOF peak frequencies crosses an undefined line; gamma (30-60) and high gamma (60-90) not derived from the fit, which stops at 35 Hz | done: the edges move with the line (§2, claim 0); fitted to 90 Hz with a knee, peaks above 30 Hz are few between 35 and 55 Hz and pile up from 55 Hz to the 90 Hz edge (DSS 4.9, ICA 4.4 per spectrum), muscle's broadband plateau and the fit's edge rather than gamma oscillations: the gamma bands rest on no spectral peak | done |
| Token flicker in the bands | 5 Hz tokens could put responses at 5, 10, 15 and 20 Hz | from the continuous runs, 1 s before the first token against 1 s of the 5 Hz stream before the press (task-log token times, multitaper, ~14,900 and ~5,300 windows per stream): power at 5 Hz rises 8 % above its neighbours with the tokens (t(31) 6.6 [7.4], 29 [28] of 32 subjects) and at 10 Hz 13-15 % (p 0.01, 20 of 32); none at 15 or 20 Hz. FOOOF finds a 5 Hz peak in 12 % of spectra with the tokens against 6 % without. Part of the theta band's peaks is the token rhythm; beta's are not | done |
| Evoked responses in band power | the tokens, the press and the feedback each evoke a response that band power carries along with induced changes | a second copy of band power with every task event's response regressed out of the run (MNE `linear_regression_raw`), regions and fsaverage; every claim then read on both copies | building |
| Post-press Fast/Slow (claims 7, 9) | the post-press token stream differs: every 20 ms in Fast, the trial over 143 ms after the press; every 170 ms in Slow, over 1,024 ms after it (task log, all trials) | the replicated beta PC3 and PC4 Fast/Slow effects lie after the press, where the screen and the feedback time differ between speeds | done: confounded |
| Trial classes | chosen-target classes depend on the response they then explain (`behavior.md`) | claims 1-4, 6 in their classes against ours: one means pass grouped by chosen-target `nProb` | to do |
| Success-probability correlation (claim 6) | the chosen target's success probability rises with time in every profile, so any drifting component correlates; 7 points of two smooth curves | done: the fixed-target frame separates choice from drift (§2); next, trial-by-trial regression with elapsed time held, the misleading-trial reversal, shuffled-sequence nulls | partly |
| LDA permutations and imputation (claims 5, 7) | labels permuted across subjects (95 in the code, 1,000 stated), though Fast/Slow is a block label; a mean imputer fitted on the test subject | not run: no decoding in this stage. The statistics of claims 5 and 7 permute within subject, and by run for Fast/Slow | — |
| Go-lock bins | kept up to press + 200 ms, so late bins hold slow trials only | first-token-locked results with and without the cut-off | to do |
| "Subcortical" loadings (claim 9) | its source space had no deep sources; the only non-cortical label was the medial-wall cut | the fit with the deep structures beside the regions; leakage (`meg.md` §3, risk 5) | running |
| Deep power (claim 10) | band-passed and Hilbert-transformed from non-negative magnitudes; the press-locked deep PCA indexed cortical sources | redone from oriented sources (`deep-power`), deep sources indexed as such; leakage, LCMV, the 23 without neck muscle | running |
| Top-5 % loadings | its threshold keeps 10-13 % (code) | loadings at 3, 5 and 10 % | to do |
| Component labels | "beta PC2" is PC4 in its code; beta axes PC1/2/4 in the text, PC1/3/4 in Fig S9 | read against our components by loading and time course | partly |
| Alpha PC1 as the response to the first tokens (Reviewer 2, 2022) | if PC1 were evoked by the first tokens it would sit at a fixed time after them, so from the press it would move back one-for-one with decision time | per subject, PC1 timed in each of the ten decision-time fifths (peak, and the half-way fall from its plateau to its trough), regressed on the fifth's decision time: slope −0.19 [−0.33] by the peak, −0.22 [−0.24] by the fall, far from the −1 of a token response (p ~1e-13) and close to, though not at, the 0 of a press-locked one (p 1e-3 to 1e-4); over the three classes the slope is +0.04 [−0.20], not different from 0. Mostly locked to the press, not to the tokens | done |
| Bends at commitment (claim 3) | the commitment marker is the press shifted by a constant; one bend per trajectory | multi-segment fits over the whole window (number of bends by BIC), curvature at 50/100/200 ms smoothing | to do |

## 4. Our method (track C)

The plan of `meg.md` §2, Analyses, run on the cohort (42 fits, both streams,
three locks, every band; commits `48c3e8e`, `ad7eaa4`, `b3e4981`):

| Step | Status |
|---|---|
| Pooled fit, log10 centred (main) and dB; regions | computed, not yet read |
| Grand-average and pooled fits of the signed fields, regions (sign-aligned) and fsaverage | computed, not yet read |
| Scree, participation ratio, reconstruction error, loadings | computed |
| Geometry: speed, path length, curvature, separation (coco-pipe `TrajectoryResult`) | to do |
| Statistics: participant as unit, labels permuted within participant (strata subject x class x side for Fast/Slow), family max-statistic over bands x contrasts, standardised (coco-pipe) | to do |
| Each of track A's results checked in these fits | to do |

## 5. Where things are

- Code: `analysis/trajectories.py` (fits, projection, sign alignment),
  `workflows/trajectories.py`, `workflows/band_power.py` (means, fsaverage
  and deep means, deep power, deep morphs), `analyze trajectories`, the
  `meg band-power-means-fsaverage`, `deep-power`, `deep-morph` and
  `deep-power-means` commands, their arrays in `scripts/`.
- Derivatives: trajectory files `sub-group_..._trajectories.h5` (`data_contract.md`);
  fsaverage means and deep power under `/project/rrg-kjerbi/hamza97/meg-tokens/BIDS`.
- The scripts behind §2's claims 1-4, 6 and 8 (figures, bends, loadings,
  success-probability correlations) are exploratory and not in the package
  yet; those that stay move into `analysis/` and `reports/`.

## 6. Order

1. Read 10 when its jobs end; claims 5 and 7 as statistics on the PC scores.
2. Track B, step by step, beginning with the trial classes, the LDA
   permutations and the success-probability tests.
3. Track C: read the pooled and field fits; geometry and statistics; set
   each of A's results against them.
4. Move the scripts that stay into the package; results into `meg.md`.
