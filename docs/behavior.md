# Behavior

| Stage | Command | Purpose | Input → Output |
| :--- | :--- | :--- | :--- |
| 0: Raw BIDSification | `meg stage-raw`, `meg apply-raw-staging` | Match raw CTF/TDMS media to behavioral runs by inter-trial-interval fingerprint and copy it into a BIDS layout | Raw CTF `.ds` + `.tdms` → `BIDS/sub-*/{meg,anat}` and a minimally-parsed `BIDS/sub-*/beh/*_beh.tsv` |
| 1: Behavioral Log Parsing | `behavior ingest` | Turn the raw per-run TDMS transcription into an analysis-ready table, inferring trial class for random (`x`) trials | `BIDS/sub-*/beh/*_beh.tsv` → one derivative table per run |
| 2: Behavioral Metrics Extraction | `behavior analyze` | Derive the canonical trial-feature and subject-summary tables every downstream analysis reads | Stage 1 tables → trial-feature, summary, and group-stats tables |
| 2b: Behavioral Characterization | `behavior characterization`, `behavior ssm-fit` | Run the fixed analysis battery: distributions, design effects, evidence/criterion, sequential-sampling model fits, individual differences, cross-species comparison | Stage 2 tables → the analysis battery |

Full schema and validation for every stage:
`docs/data_contract.md`. Model specification for Stage 2b's sequential-sampling
fit: the module docstring of
`meg_tokens/behavior/analyses/sequential_sampling.py`.

## Analyses

Group inference is two-stage throughout: one fit per subject, then a test
across subjects on the fitted values — the random-effects equivalent of a
mixed model with by-subject random slopes, without a modelling dependency.
The sequential-sampling fits are the exception: maximum-likelihood point
fits, pooled under a population model rather than a paired test.

Each analysis writes its own derivative
(`sub-group_task-tokens_desc-<name>_beh.tsv`) and, where it has one, a
matching `<name>stats` group-statistics derivative, so one failed model
never truncates the rest of the output.

### `behavior/analyses/distributions.py`

- `spdcumulative` — SPD cumulative distributions, all-trial and 15-row
  sensitivity views, reported pooled and as the mean of per-subject curves.
- `dtdistribution` — subject-level DT quantiles, skewness, kurtosis, and
  ex-Gaussian (`mu`, `sigma`, `tau`) fits per condition, class, and
  condition × class cell; class and Fast/Slow contrasts on each statistic.

### `behavior/analyses/design_effects.py`

- `conditionclass` — DT and accuracy for all six condition × class cells,
  with a fully within-subject 2 × 3 ANOVA (main effects and interaction,
  each against its own `effect × subject` error term).
- `choiceside` — left/right choice proportions, DT asymmetry, and accuracy
  asymmetry, overall and per condition.
- `timeontask`, `conditionorder` — block-order effects from the session
  clock (`nInitialTime`), the only field that recovers it, since Fast and
  Slow blocks interleave and `nTrialIndex` restarts at 1 in each run.
- `lapses`, `extremedt`, `extremedttrials` — no-response outcomes
  summarized by LabVIEW code; extreme DTs flagged at a robust
  median-absolute-deviation cutoff and listed by trial, never removed.

### `behavior/analyses/evidence.py`

Evidence is `SumLogLR`, computed in `behavior/math/evidence.py` and
`behavior/features.py` and written into the trial-feature table: the exact
posterior from Equation 1, so with equal priors the cumulative
log-likelihood ratio is its log posterior odds. Certainty (SP exactly 0 or
1) is reported at ±log 255 rather than propagated as an infinity that would
drop the most decisive trials from every regression.

- `criteriondecline` — evidence at decision against the number of tokens
  observed, fitted on both the probability and the log-odds scale.
- `reversecorrelation` — per-subject logistic weights for each of the
  first eight jumps, tokens after commitment coded as unseen, plus a
  model-free kernel and a Fast/Slow comparison per jump.
- `conditionalaccuracy` — accuracy across within-subject DT quantiles per
  condition, with a test of the slope across bins.
- `continuousevidence` — DT on evidence strength and accuracy on signed
  evidence, over every task trial including unclassified random ones.
- `urgency` — per-subject intercept and slope of evidence at decision
  against decision time, by condition, on both evidence scales.

### `behavior/analyses/sequential.py`

- `posterror` — robust post-error slowing, `DT(error+1) − DT(error−1)`,
  with the classical contrast reported beside it; adjacency is broken by
  any gap in `run_trial_index`.
- `choicehistory` — win-stay/lose-shift, lag-1 side autocorrelation, and
  DT as a function of the previous trial's outcome and class.

### `behavior/analyses/sequential_sampling.py`

- `ssmcomparison` — the urgency-gating vs. bounded-integrator fits (model
  specification: the module's own docstring). Both are reported with
  log-likelihood, AIC, BIC, and the criterion difference against the
  integrator fit; the group table counts the subjects each model wins,
  tests that difference across subjects, and carries the paired
  Fast-minus-Slow contrast on every fitted urgency parameter.
- `ssmpopulation` — empirical-Bayes population model fitted over the
  subject-level estimates and their observed-information standard errors:
  a population mean, a between-subject standard deviation, and a
  population-informed estimate per subject, shrunk in proportion to that
  subject's own uncertainty.

### `behavior/analyses/individual.py`

- `individualprofile`, `individualcorrelations` — subject-level SAT
  adjustment, urgency slope, criterion slope, evidence sensitivity,
  accuracy, and lapse rate, correlated pairwise; `--neural-metrics` joins
  any subject-level MEG table into the same matrix.
- `speciescomparison` — DT by class, SP at decision by class, criterion
  decline, and the urgency-gating results (the model's advantage over
  integration, its fitted urgency growth, and that growth's
  Fast-minus-Slow contrast). Values from the papers are not reproduced in
  code; see Findings, below, and `docs/behavior_roadmap_results.md` for
  measured values.

## Known Issues (Stage 0–1)

Data-quality issues inherited from the original recording and parsing code,
how each was handled, and its impact.

### Stage 0: Raw BIDSification

| Issue | Handling | Impact |
| :--- | :--- | :--- |
| 3 non-standard `.tdms` filenames (scratch fragments: H03 `temp_180214.tdms`, H18 `temp_181024.tdms`, H23 `temp_181121.tdms`) | Explicit `behavior_ignore_files` allowlist; any other non-matching file still errors | 320/323 retained runs |
| H01, H05 raw-session duration counts don't match 8-Slow/Fast + 2-RT | `KNOWN_SESSION_OVERRIDES`, cross-referenced against the acquisition notebook (H01 labeled `Pilot01` there); fingerprint independently reproduces all 17 entries | H05 RT1 confirmed unmatched — real gap, not ambiguity |
| H01 headshape not found on any known drive | Left `not_found` / review | No headshape derivative for H01 |
| H05 RT1 has no raw MEG counterpart | Left `ambiguous` / review, 0 candidates | RT1 excluded from H05's MEG-joined analyses |
| H05 has no empty-room recording, any date | Left `not_found` / review | No empty-room for H05 |
| H07, H10 have no FreeSurfer reconstruction | Left `not_found` / review | No anat/BEM for H07, H10 |
| H26/H27 shared empty-room `_01.ds` `.hc` corrupt (`RuntimeError: HPI information not available`) | `KNOWN_NOISE_OVERRIDES` redirects to the valid retake `_02.ds` | H26/H27 empty-room resolved |
| H01 Slow3, H06 Slow2, H10 Fast2: Start-pulse count mismatch at the trailing boundary | `KNOWN_TRAILING_TRIAL_MISMATCHES`, staged with a note instead of sent to review | 3 runs recovered |
| H06 Start/Go trigger codes swapped vs. every other subject | `SUBJECT_EVENT_OVERRIDES` (event code `524288` is Start for H06, `262144` elsewhere) | H06 matches and epochs correctly |

### Stage 1: Behavioral Log Parsing

| Issue | Handling | Impact |
| :--- | :--- | :--- |
| 226 trials never started (`nOutcome==7003`, no go cue) | Retained in Stage 1; excluded from behavioral counts and MEG event matching | Post-error adjacency corrected: post-correct means changed for 16 subjects, post-error for 5 (largest: H11 post-correct −9.30 ms, H08 post-error +9.85 ms) |
| Movement time not recorded — `tEnterCenter`/`tExitCenter` written from the same LabVIEW event | Both fields retained; no movement measure computed | Response-vigor analysis omitted, not implemented against a null field |
| `nTrialIndex` doesn't order blocks within a session (session-scoped counter, resets inconsistently) | Block/session-drift analyses use `nInitialTime` (session clock) instead | Correct block order; confirms Fast/Slow blocks interleave |
| LabVIEW scientific-notation probabilities near 0/1 misparsed by the original parser | Parser hardened: signed/exponent notation, float-zero tolerance, `[0,1]` range check, equal-length array check | 881 out-of-range values found, 870 corrected, 11 dropped/misaligned; no historical label, DT result, or trial count changed |
| 5,363/16,324 trials have 14-row (not 15-row) token logs — cause unknown | Included for classification/logged-SPD; excluded from design-time-resolved SP; 15-row sensitivity set reported alongside the all-log figure | Defines which trials feed which analyses |
| Classifying by chosen-target `nProb` (original/preprint method) is response-confounded | Classify by the correct-target profile from `sTokenDirs` instead; recorded labels (`e`/`a`/`m`) preserved verbatim, deterministic rule applied only to random (`x`) trials | See Findings, below |

## Findings

### Trial-classification reference frame

**Method.** Two candidate frames for classifying random (`x`) trials: the
recorded runtime `nProb`, referenced to the *chosen* target (the original
notebook's method), vs. a profile derived from `sTokenDirs`, referenced to
the *correct* target. `SP_chosen = 1 - SP_correct` whenever the subject
erred, so the chosen frame's class assignment depends on the response it is
later used to explain — a circularity the design frame avoids.

Classification rule (reproduces all 5,224 recorded designed labels):

```python
if SP2 > 0.60 and SP5 > 0.75 and SP8 > 0.75:
    trial_class = EASY
elif SP2 == 0.50 and 0.40 < SP3 < 0.65 and 0.35 < SP5 < 0.65:
    trial_class = AMBIGUOUS
elif SP2 == 0.50 and 0.38 < SP3 < 0.40 and 0.35 < SP5 < 0.65 and SP11 == 1.0:
    trial_class = AMBIGUOUS
elif SP3 < 0.40:
    trial_class = MISLEADING
else:
    trial_class = UNCLASSIFIED
```

**Result.** Real dataset counts (32 subjects, 16,324 started-and-chosen
trials):

| Class | Implemented | Preprint |
| :--- | ---: | ---: |
| Easy | 4,098 (25.1%) | 26% |
| Ambiguous | 3,483 (21.3%) | 22% |
| Misleading | 1,718 (10.5%) | 11% |
| Unclassified | 7,025 (43.0%) | ~41% |

Random (`x`) trials, inferred: 2,244 easy, 1,831 ambiguous, **0 misleading**.

Mean decision time by class and frame (28-subject list, matching the
preprint's exclusions):

| Classification | Easy | Ambiguous | Misleading | Amb − Mis |
| :--- | ---: | ---: | ---: | ---: |
| Design frame, `'x'` only (implemented) | 1023.8 | 1415.2 | 1336.2 | +5.09 |
| Recorded labels only | 971.8 | 1400.4 | 1336.2 | +3.31 |
| Chosen frame, all trials (legacy) | 1037.1 | 1399.4 | 1563.6 | −5.18 |
| Preprint | 1028±59 | 1405±74 | 1433±79 | −1.84 |

**Interpretation.** The correct-target design frame is the valid method.
Chosen-frame "misleading" is 53.7% genuine errors on easy/ambiguous stimuli,
not misleading stimuli — the confound inflates its mean DT and produces the
preprint's ambiguous-vs-misleading sign as an artifact, not a real effect.
No random trial can be classified misleading under the design frame: after
3 jumps the correct target never trails in a random sequence — a property
of the stimulus set (`e` splits 3-0 always, `a` splits 2-1/1-2, `m` splits
1-2 [n=1,706] or 0-3 [n=12], `x` only ever 2-1 or 3-0), not an error.
`infer_random_classes=false` (recorded labels only, symmetric with the
preprint's treatment of the random 60% as a non-category) gives the closest
Easy-vs-Ambiguous match to the preprint (t=-15.12 vs -15.04); under it
misleading is slightly *faster* than ambiguous — coherent, since misleading
evidence invites early commitment while ambiguous evidence delays it.

**Supporting evidence.**

- Chosen-target `nProb` reconciles 100% against Equation 1
  (`p(R|N_R,N_L,N_C)`, binomial random-walk over the `N_C` centre tokens)
  for both 15-row (n=10,961) and 14-row (n=5,363) logs; frames mirror via
  `sp_correct = sp_chosen if nChoiceMade==nCorrectChoice else 1-sp_chosen`.
  Classifying by the chosen frame recovers only 41.3% of designed-misleading
  trials (25.8% of its assigned "misleading" trials are correct).
- Classification-rule signature `SP(2)=0.5, SP(3)=0.3872, SP(5)=0.623,
  SP(8)=0.7734` occurs in 641 designed-ambiguous and 381 designed-misleading
  trials; `SP(11)` separates them (`1.0` ambiguous vs. `0.6875`/`0.9375`
  misleading).
- 14-row logs (one token short, cause unknown): 0% full design-profile
  match either shifted or unshifted (vs. 100% for 15-row logs); of their
  logged-SPD alignment, 48.7% match only a shifted read, 49.0% match
  neither, and the 2.3% matching both are trivial (pre-first-token or
  post-final-token values) — none has a complete `sTokenDirs` match.
  Confirms the exclusion-from-design-time-alignment policy above.
- Parser defect (scientific-notation probabilities near 0/1) traced to the
  first parser commit (`13e07c6`), not the refactor; fix verified across
  all 320 retained runs.

## Preprint Replication

Current values use all 32 subjects, not the preprint's N=28 (subject
exclusion: `docs/meg.md`):

| Result | Preprint N=28 | Current N=32 |
| :--- | ---: | ---: |
| Easy vs ambiguous DT | 1028 ± 59 vs 1405 ± 74; `t=-15.04` | 1033 ± 59 vs 1433 ± 70; `t=-16.80` |
| Easy vs misleading DT | 1028 ± 59 vs 1433 ± 79; `t=-13.10` | 1033 ± 59 vs 1357 ± 78; `t=-11.86` |
| Ambiguous vs misleading DT | `t=-1.84`, `p=0.077` | `t=5.43`, `p=6.29e-6` |
| Fast vs Slow DT | 1166 ± 71 vs 1293 ± 68; `t=-6.08` | 1186 ± 70 vs 1313 ± 66; `t=-6.19` |
| Fast vs Slow errors | 45.1 vs 36; `t=6.10` | 46.3 vs 36.9; `t=6.06` |
| SPD by class | Higher in easy | Paired all-log and 15-row tests emitted |

Fast/Slow effects are close to the preprint. The ambiguous/misleading effect
is reversed; the cause is the trial-classification reference frame (see
Findings, above), not subject exclusion — no four-subject exclusion
reverses the current effect. The `H02` `RT1`/`RT2` motor-baseline choice was
also ruled out (~1.4 ms on group means, cancels in every paired contrast;
`docs/meg.md`, "H02 motor baseline").

## References

- Thiery T., Rainville P., Cisek P., Jerbi K. (2022). *Distinct trajectories in
  low-dimensional neural oscillation state space track dynamic decision-making
  in humans.* bioRxiv
  [10.1101/2022.06.14.494674](https://doi.org/10.1101/2022.06.14.494674).
- Cisek P., Puskas G.A., El-Murr S. (2009). *Decisions in changing conditions:
  the urgency-gating model.* J Neurosci 29(37):11560–11571.
- Thura D., Beauregard-Racine J., Fradet C.-W., Cisek P. (2012). *Decision
  making by urgency gating: theory and experimental support.* J Neurophysiol
  108:2912–2930.
- Thura D., Cisek P. (2014). *Deliberation and commitment in the premotor and
  primary motor cortex during dynamic decision making.* Neuron 81:1401–1416.
- Thura D., Cos I., Trung J., Cisek P. (2014). *Context-dependent urgency
  influences speed–accuracy trade-offs in decision-making and movement
  execution.* J Neurosci 34(49):16442–16454.
- Thura D., Cisek P. (2017). *The basal ganglia do not select reach targets but
  control the urgency of commitment.* Neuron 95:1160–1170.
- Carland M.A., Thura D., Cisek P. (2015). *The urgency-gating model can
  explain the effects of early evidence.* Psychon Bull Rev 22:1830–1838
  [10.3758/s13423-015-0851-2](https://doi.org/10.3758/s13423-015-0851-2).
- Carland M.A., Thura D., Cisek P. (2019). *The urge to decide and act:
  implications for brain function and dysfunction.* The Neuroscientist.
- HSSM — Hierarchical Sequential Sampling Modeling:
  [lnccbrown.github.io/HSSM](https://lnccbrown.github.io/HSSM/).
- PyDDM — generalized drift-diffusion modelling:
  [pyddm.readthedocs.io](https://pyddm.readthedocs.io/).
