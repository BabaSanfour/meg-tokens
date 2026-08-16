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
- `dtdistribution` — subject-level DT quantiles, skewness, and kurtosis per
  condition, class, and condition × class cell; class and Fast/Slow
  contrasts on each statistic.

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

## Figures

`meg-tokens report behavior --figures <keys or groups>` renders the figure
battery; `--list-figures` prints the full registry. Full per-figure
specification (source columns, chart-type justification, statistical
annotation): `docs/behavior_reporting_plan.md`. All 6 phases complete (26
figures); F26 additionally requires `--neural-metrics`.

| Key | Derivative(s) | Shows |
| :--- | :--- | :--- |
| `ssmcomparison-deltabic` | `ssmcomparison`, `ssmcomparisonstats` | H1: per-subject ΔBIC, urgency vs. bounded integration |
| `ssmcomparison-urgencyscale` | `ssmcomparison`, `ssmcomparisonstats` | H2: `urgency_scale`, Fast vs. Slow |
| `ssmcomparison-urgencyparams` | `ssmcomparison`, `ssmcomparisonstats` | All four urgency-model parameters, Fast vs. Slow |
| `ssmtimecourse-fit` | `ssmtimecourse` | Criterion time course, fit quality, decision-variable trajectory |
| `ssmpopulation-shrinkage` | `ssmpopulation`, `ssmpopulationstats` | Empirical-Bayes population model, per parameter |
| `dtdistribution-condition` | `dtdistribution`, `dtdistributionstats`, `trialfeatures`, `groupstats` | Decision time, Fast vs. Slow (`dt_ms`, not `rawRT`) |
| `dtdistribution-class` | `dtdistribution`, `dtdistributionstats` | Decision time by trial class |
| `spdcumulative-class` | `spdcumulative`, `groupstats` | SPD at decision, cumulative by class |
| `conditionclass-anova` | `conditionclass`, `conditionclassstats` | Condition x class (no interaction on either measure) |
| `choiceside-asymmetry` | `choiceside`, `choicesidestats` | Left/right choice, DT, and accuracy asymmetry |
| `timeontask-drift` | `timeontask`, `timeontaskstats`, `trialfeatures` | Session and within-block drift |
| `conditionorder-balance` | `conditionorder`, `conditionorderstats` | First-block counterbalancing (between-subject) |
| `lapses-quality` | `lapses`, `extremedt`, `extremedttrials` | Lapses and extreme decision times (QC) |
| `summary-cohort` | `summary` | Dataset overview (counts, motor baseline, accuracy) |
| `criteriondecline-tokens` | `criteriondecline`, `criteriondeclinestats`, `trialfeatures` | Evidence at decision vs. tokens observed |
| `urgency-decisiontime` | `urgency`, `urgencystats`, `trialfeatures` | Evidence at decision vs. decision time |
| `reversecorrelation-kernel` | `reversecorrelation`, `reversecorrelationstats` | Psychophysical kernel by token jump |
| `conditionalaccuracy-caf` | `conditionalaccuracy`, `conditionalaccuracystats` | Conditional accuracy function |
| `continuousevidence-effects` | `continuousevidence`, `continuousevidencestats`, `trialfeatures` | Continuous early-evidence effects (incl. unclassified trials) |
| `posterror-slowing` | `posterror`, `posterrorstats` | Robust post-error slowing |
| `choicehistory-effects` | `choicehistory`, `choicehistorystats` | Win-stay/lose-shift, side autocorrelation, DT by history |
| `individualcorrelations-matrix` | `individualcorrelations` | Pairwise correlation matrix, individual differences |
| `individualprofile-scatter` | `individualprofile`, `individualcorrelations` | Scatter grid for the correlations that matter |
| `speciescomparison-forest` | `speciescomparison` | Cross-species comparison statistics, unit-faceted |
| `individualprofile-neural` | `individualprofile`, `--neural-metrics` | Behaviour vs. neural metric (gated, optional) |

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

### Fast vs. Slow stretches decision time proportionally, rather than adding a fixed delay

**Result.** Decision time is 11% longer under Slow than Fast (1186 ± 70 ms
vs. 1313 ± 66 ms, t(31) = −6.19, p = 7.1e-7, dz = −1.10, n = 32;
`dtdistribution`, `groupstats`). Every part of the distribution moves, but
not by a common number of milliseconds — by a common *factor*
(`dtdistributionstats`):

| Quantile | Fast | Slow | Slow − Fast | Slow ÷ Fast |
| :--- | ---: | ---: | ---: | ---: |
| q10 | 677 ms | 756 ms | +79 ms | 1.124 |
| q25 | 867 ms | 966 ms | +99 ms | 1.131 |
| q50 | 1126 ms | 1255 ms | +129 ms | 1.140 |
| q75 | 1442 ms | 1601 ms | +159 ms | 1.127 |
| q90 | 1768 ms | 1916 ms | +148 ms | 1.091 |

The millisecond differences nearly double from the fast end to the slow end,
which rules out a fixed delay: Δq90 vs. Δq10, t(31) = 2.44, p = .021. The
ratios do not differ (log-ratio q90 vs. q10, t(31) = −1.42, p = .167).

Spread discriminates the two models directly, because a fixed delay moves
the mean without touching the SD while a stretch scales both. Giving each
model its best per-subject fit (that subject's own delay `mean_slow −
mean_fast`, or their own factor `mean_slow ÷ mean_fast`) and asking what it
predicts for their Slow trials:

| | Observed Slow | Fixed delay predicts | Stretch predicts |
| :--- | ---: | ---: | ---: |
| SD | 487 ms | 447 ms (t(31) = 2.69, p = .011) | 500 ms (t(31) = −1.01, p = .32) |
| CV | 0.394 | 0.355 (t(31) = 2.53, p = .017) | 0.401 (t(31) = −0.51, p = .62) |

The observed spread is incompatible with a fixed delay and compatible with a
proportional stretch. **Skewness (0.89 vs. 1.05, p = .53) is reported for
completeness but decides nothing here**: skewness is invariant under *any*
positive linear transform, so both models predict no change in it. What its
null result does rule out is a non-linear change — Slow occasionally
derailing a trial into a long deliberation would have inflated it.

Per-subject two-sample KS tests on the full trial-level distributions
(`trialfeatures`) localize it. Fast and Slow differ outright in 18 of 32
subjects; removing a per-condition additive offset leaves 6 still differing,
while removing a per-condition multiplicative factor leaves 2 — chance at
α = .05 is 1.6:

| Normalization applied before the test | Subjects with Fast ≠ Slow |
| :--- | ---: |
| none (raw decision time) | 18 / 32 |
| each condition's median **subtracted** | 6 / 32 |
| each condition's median **divided out** | 2 / 32 |

Fitting each subject's five Slow quantiles as a multiple of their Fast
quantiles gives a stretch factor of 1.117, 95% CI [1.073, 1.161]
(t(31) = 5.22 against 1, p = 1.1e-5; 27 of 32 subjects above 1).

**Interpretation.** q10/q50/q90 are each subject's trials sorted
fastest-to-slowest, read off at the 10th/50th/90th percentile — the fast
end, the typical trial, and the slow end. Two things could make all of them
later under Slow. Something could be *added* to every trial, moving each
percentile by the same number of milliseconds; or the whole process could
*run slower*, multiplying each percentile by the same factor and therefore
moving late trials more than early ones in absolute terms. The data are
inconsistent with the first and consistent with the second: Slow does not
delay decisions by 127 ms, it makes deliberation take about 12% longer.

That distinction carries the mechanism. A constant added to every trial is
what a change in non-decision time looks like — and non-decision time has
already been removed here by subtracting each subject's motor baseline, so
there is nothing left for it to describe. A constant *factor* is a
rescaling of the deliberation clock itself, which is exactly what changing
the slope of a rising urgency signal does: a signal that climbs more slowly
crosses a given level proportionally later, stretching the whole
distribution instead of translating it. The unchanged coefficient of
variation is the signature of that — pure time-rescaling preserves relative
spread.

This converges with the sequential-sampling fit, which reaches the same
conclusion by a different route: urgency gating is favored over bounded
integration (ΔBIC = −238.9, t(31) = −8.12, p = 3.6e-9, 30/32 subjects
individually favoring it), and its growth rate differs Fast vs. Slow
(`urgency_scale` Δ = −0.108, t(31) = −2.52, p = .017; `ssmcomparison`,
`ssmcomparisonstats`). Distribution geometry here and explicit model
comparison there land on the same mechanism, and the proportional form is
the more specific of the two predictions — bounded integration with a moved
threshold does not produce it.

**Scope.** "Proportional" describes the group-level pattern. Within a single
subject the two models are not separable at this trial count: fitting each
subject's five quantiles, the scale model's residual (55.6 ms) beats the
shift model's (63.0 ms) in only 18 of 32 subjects, t(31) = −0.89, p = .38.
The claim rests on the aggregate tests above, not on per-subject model
selection. Relatedly, a ratio depends on where zero sits; the origin used
here is decision onset with non-decision time subtracted out, which is the
theoretically meaningful zero for a deliberation-rate claim, not an
arbitrary one.

Figure: `dtdistribution-condition` (F04).

### Decision time by trial class: difficulty changes the shape of the distribution, not just its location

**Result.** Easy trials are faster than both harder classes at every
quantile, and the gap grows toward the tail (`dtdistribution`,
`dtdistributionstats`, `stratum_type == "class"`):

| Contrast | q10 | q50 | q90 |
| :--- | ---: | ---: | ---: |
| Easy − Ambiguous | −315 ms, t(31)=−9.11, p=2.9e-10 | −419 ms, t(31)=−13.29, p=2.4e-14 | −462 ms, t(31)=−12.88, p=5.6e-14 |
| Easy − Misleading | −207 ms, t(31)=−6.27, p=5.7e-7 | −355 ms, t(31)=−11.32, p=1.5e-12 | −448 ms, t(31)=−10.42, p=1.2e-11 |

Ambiguous vs. misleading is different in kind, not just size — misleading is
faster than ambiguous through the fast and typical range, but the two
classes converge in the tail:

| Quantile | Δ (ambiguous − misleading) | t(31) | p |
| :--- | ---: | ---: | ---: |
| q10 | +107 ms | 5.39 | 7.1e-6 |
| q50 | +64 ms | 4.37 | 1.3e-4 |
| q90 | +13 ms | 0.42 | .68 (n.s.) |

Applying the F04 geometry tests to each contrast (is the change a fixed
delay, a proportional stretch, or neither?) separates them cleanly. The
per-subject KS ladder — subjects whose two distributions still differ after
removing a per-class additive offset, or a multiplicative factor — is the
decisive column:

| Contrast | Fixed delay? | Stretch? | KS raw → −median → ÷median |
| :--- | :--- | :--- | :--- |
| Easy vs. Ambiguous | no (p = .003) | no (p = .003) | 32/32 → 11/32 → 9/32 |
| Easy vs. Misleading | no (p < 1e-4) | **yes** (p = .63) | 29/32 → 9/32 → **0/32** |
| Ambiguous vs. Misleading | no (p = .017) | no (p < 1e-4) | 5/32 → 0/32 → 0/32 |

Easy vs. misleading is a clean ~1.30× stretch — dividing out a
per-class factor leaves the two distributions indistinguishable in every
subject. Easy vs. ambiguous is neither: the differences grow toward the tail
(315 → 462 ms) while the ratios *shrink* (1.49 → 1.33), and both
normalizations leave 9–11 subjects still differing, well above the 1.6
expected by chance. That residual is a genuine change in distribution shape,
not a change in location or scale.

Ambiguous vs. misleading is the mirror-image case and needs care: only 5 of
32 subjects show any distributional difference at all, and either
normalization removes it entirely. The group-level quantile contrasts above
are real, but they aggregate a small effect that is consistent in direction
rather than a large one — the individual-subject evidence for this contrast
is weak.

**Interpretation.** Easy trials are defined by strong, consistent evidence
from early in the sequence (`SP2 > 0.60`, rising to `SP8 > 0.75` — see
"Trial-classification reference frame," above); the decision variable has
almost nowhere to go but toward the correct target quickly, so every trial,
fast responders and slow responders alike, resolves sooner.

The geometry says this is a *different kind* of effect from Fast vs. Slow,
not the same one with another cause — and the contrast is informative.
Fast/Slow rescales the deliberation clock and leaves the distribution's
shape alone, which is what changing the urgency slope predicts. Difficulty
does not behave that way. Easy vs. ambiguous survives neither normalization,
meaning ambiguous trials are not simply "easy trials run slower" — the shape
of the waiting-time distribution itself changes when the evidence stops
being informative. That fits the mechanism: on an ambiguous trial the
decision variable loiters near chance for a variable, evidence-dependent
stretch of time before anything resolves it, and that loitering has no
counterpart on an easy trial to be stretched *from*. Easy vs. misleading, by
contrast, *is* a clean stretch — misleading trials do have a decisive
evidence trajectory, just a delayed and reversed one, so the same process
runs on a slower clock. Manipulating urgency changes the rate; manipulating
evidence quality changes the rate for some contrasts and the shape for
others.

Ambiguous trials hover near chance (`SP ≈ 0.50`) for an extended stretch —
genuinely uninformative, so there is nothing to act on and a subject has to
wait. Misleading trials start with evidence pointing the *wrong* way
(`SP3 < 0.40`) before resolving toward the correct target. A trial with
strong-but-wrong-then-strong-and-right evidence plausibly gives the
accumulator something concrete to lock onto once it flips, resolving faster
on typical trials than a trial that never has strong evidence at all. But on
trials where neither class has resolved by the time urgency is forcing a
decision anyway (the slowest 10%), the specific evidence trajectory stops
mattering — both classes hit the same pressure, and the gap closes. That
last part is an interpretation consistent with the pattern and with the
urgency-gating result above, not itself a statistical claim this analysis
makes.

**The reverse, included.** This flips the published result. The preprint,
using a response-confounded classification (see "Trial-classification
reference frame," above), found ambiguous *faster* than misleading, and only
marginally (t = −1.84, p = .077). Under the corrected design-frame
classification used here, the sign reverses and the effect is solid:
misleading is reliably faster than ambiguous (t = 5.43, p = 6.3e-6 on the
full-sample mean; see "Preprint Replication," below). The mechanism for the
reversal — chosen-frame "misleading" is 53.7% genuine errors on easy/ambiguous
stimuli, not misleading stimuli, which inflates its mean DT under the old
classification — is diagnosed in "Trial-classification reference frame."

Figure: `dtdistribution-class` (F05).

### Success probability at decision: easy resolves at higher confidence; ambiguous and misleading resolve at the same confidence despite different timing

**Result.** SPD (success probability at decision) is the design's own
success-probability trajectory read off at whichever token count the subject
actually stopped on — a measure of how much evidence was in hand at
commitment, not how long commitment took. Mean SPD per class
(`validated_15row`; `summary`, `groupstats[analysis=spd]`):

| Class | Mean SPD | SEM |
| :--- | ---: | ---: |
| Easy | 0.781 | 0.010 |
| Ambiguous | 0.651 | 0.013 |
| Misleading | 0.648 | 0.006 |

| Contrast | Δ | t(31) | p | dz |
| :--- | ---: | ---: | ---: | ---: |
| Easy − Ambiguous | 0.129 | 17.03 | 2.7e-17 | 3.01 |
| Easy − Misleading | 0.133 | 17.51 | 1.3e-17 | 3.09 |
| Ambiguous − Misleading | 0.0035 | 0.37 | .713 | 0.07 |

Easy vs. either harder class is about as large an effect as behavioral data
gets (dz ≈ 3, near-zero group overlap).

Ambiguous vs. misleading returns p = .713, and a non-significant test is not
by itself evidence of equivalence — so this one is stated as a bound rather
than as an absence. The 90% CI on the paired difference is
[−0.012, +0.019] SPD, which passes a two-one-sided-tests equivalence check
at dz = ±0.4 (and at ±0.5) but fails at ±0.3. Read directly: **the data rule
out any difference larger than about 0.019 SPD (dz ≈ 0.36); a smaller one
remains possible and this sample cannot exclude it.**

**Interpretation.** This is a genuine dissociation from the decision-time
result above. In time, ambiguous and misleading differ (misleading faster
through the fast/typical range). In confidence-at-commitment, any difference
is at most a third the size of a conventional "small" effect — subjects
commit to both hard classes at an average SPD of ~0.65 either way, despite
reaching that point at different speeds. Time-to-decide and
quality-of-decision are not the same knob here: whatever produces the timing
difference between ambiguous and misleading does not produce a confidence
difference of any consequential size.

The other piece: a single fixed confidence threshold applied on every trial
would put SPD-at-decision in roughly the same place regardless of class —
the threshold doesn't know how the evidence got there. Instead, easy trials
resolve at a substantially higher confidence (0.78) than either hard class
(0.65). That fits urgency gating rather than a fixed-evidence threshold:
easy trials reach high confidence fast, comfortably clearing a high bar
before urgency would ever force the issue; hard trials take longer to reach
any given confidence level, so a rising urgency signal cuts them off lower,
regardless of whether stronger evidence might eventually have arrived. A
third independent line pointing at the same mechanism as H1 (model
comparison, above) and the Fast/Slow proportional stretch (above) — this
time from confidence-at-commitment rather than timing.

Figure: `spdcumulative-class` (F06).

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
