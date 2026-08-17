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
annotation): `docs/behavior_reporting_plan.md`. The registry contains 24
figures: F07 was removed and F12 was folded into F13; F26 additionally
requires `--neural-metrics`.

| Key | Derivative(s) | Shows |
| :--- | :--- | :--- |
| `ssmcomparison-deltabic` | `ssmcomparison`, `ssmcomparisonstats` | H1: per-subject ΔBIC, urgency vs. bounded integration |
| `ssmcomparison-urgencyscale` | `ssmcomparison`, `ssmcomparisonstats` | H2: `urgency_scale`, Fast vs. Slow |
| `ssmcomparison-urgencyparams` | `ssmcomparison`, `ssmcomparisonstats` | All four urgency-model parameters, Fast vs. Slow |
| `ssmtimecourse-fit` | `ssmtimecourse` | Criterion time course, fit quality, decision-variable trajectory |
| `ssmpopulation-shrinkage` | `ssmpopulation`, `ssmpopulationstats` | Empirical-Bayes population model, per parameter |
| `dtdistribution-condition` | `dtdistribution`, `dtdistributionstats` | Decision time, Fast vs. Slow (`dt_ms`, not `rawRT`) |
| `dtdistribution-class` | `dtdistribution`, `dtdistributionstats` | Decision time by trial class |
| `spdcumulative-class` | `spdcumulative` | SPD at decision, cumulative by class |
| `conditionclass-anova` | `conditionclass`, `conditionclassstats` | Decision time and accuracy by condition x difficulty |
| `choiceside-asymmetry` | `choiceside`, `choicesidestats` | Left − right asymmetry in choice, decision time, accuracy |
| `timeontask-drift` | `timeontaskstats` | Session and within-block drift |
| `conditionorder-balance` | `conditionorder`, `conditionorderstats` | First-block counterbalancing (between-subject) |
| `summary-cohort` | `summary`, `lapses`, `extremedt` | Cohort composition and data quality |
| `criteriondecline-tokens` | `criteriondecline`, `criteriondeclinestats` | Evidence at decision vs. tokens observed |
| `urgency-decisiontime` | `urgency`, `urgencystats` | Evidence at decision vs. decision time |
| `reversecorrelation-kernel` | `reversecorrelation`, `reversecorrelationstats` | Psychophysical kernel by token jump |
| `conditionalaccuracy-caf` | `conditionalaccuracy`, `conditionalaccuracystats` | Conditional accuracy function |
| `continuousevidence-effects` | `continuousevidence`, `continuousevidencestats` | Continuous early-evidence effects (incl. unclassified trials) |
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

The findings are ordered as a scientific argument rather than by figure
number: cohort composition and data quality first; checks on block order,
response side, and time on task next; then the classification reference frame
used by the class-based results; finally the condition, difficulty,
distribution, and confidence results. Figure numbers retain build order, so
their sequence below is intentionally non-numeric.

### Cohort and trial composition

**Result.**

- 32 subjects, **16,324 started-and-chosen task trials** (`summary`), 460–555
  per subject (median 516). Every subject contributes to every condition and
  every difficulty class; no cell is empty.
- Fast blocks hold more trials than Slow (240–323 vs. 211–241 per subject)
  because each Fast trial is shorter and block duration is fixed. **This is a
  design property, not an imbalance to correct** — but it makes any
  between-condition *count* confounded, which is why the preprint's
  error-count comparison is not reproduced (see "Deliberate deviations",
  below).
- **Decision time is available for 99.92% of trials** (16,324 of 16,337; at
  most 3 trials lost per subject), so no analysis in this document is
  meaningfully restricted by DT availability.
- Difficulty composition: easy 4,098 (25.1%), ambiguous 3,483 (21.3%),
  misleading 1,718 (10.5%), **unclassified 7,025 (43.0%)**, 198–237 per
  subject. The unclassified share is large and is drawn explicitly in the
  figure; class-keyed analyses run on the 57% that carry a class.
- Accuracy 78.5–90.1% (mean 83.7); motor baseline 367–575 ms (mean 449).
  Motor baseline does not track accuracy across subjects (r = +0.24,
  p = .19), so subtracting it does not import a performance confound into
  decision time.
- 226 trials were never started (`nOutcome == 7003`, no go cue), up to 54 in
  one subject; they are excluded from every count above.

**Interpretation.**

- The cohort is balanced enough that no analysis here needs weighting or
  exclusion: `subject_exclusions = []` throughout (`docs/meg.md`, Subject
  Exclusion).
- Two composition facts do real work later, and both are properties of the
  design rather than defects: the Fast/Slow **trial-count** asymmetry (which
  invalidates count-based condition comparisons) and the 43% **unclassified**
  share (which is why class-keyed panels carry a smaller *n* than
  condition-keyed ones).


**Data quality.**

- **Lapses are negligible.** 13 of 16,337 started trials (0.08%) ended
  without a response — 6 `7006` (reaction time too long), 7 `7011`
  (delay-1 error). 22 of 32 subjects have zero; the maximum is 3 (`lapses`).
- **Extreme decision times are rare and one-sided.** 59 of 16,324 (0.36%)
  exceed 5 robust MAD from the subject's own median (`extremedt`). **Every
  one is slow-side** (z from 5.0 to 24.4); there is not a single fast-side
  extreme. They split 32 Slow / 27 Fast, so they are not a property of one
  condition.
- They are concentrated: 17 of 32 subjects have none, and **H20 alone
  contributes 27 of the 59** (46%), H21 nine, H26 five.
- **Four anticipations** (dt < 0) exist across 4 subjects; none reaches the
  5 MAD threshold. **Nothing is removed** — extremes are flagged and listed
  per trial (`extremedttrials`), never dropped.
- The one-sidedness is expected rather than suspicious: decision time is
  bounded below by the motor baseline and above by the 15-jump deadline, so
  the distribution has a long right tail and no room for a symmetric one.
- **H20 is worth naming but not excluding.** Contributing 46% of the extremes
  is an outlier in *variability*, not in central tendency, and the MAD
  criterion is per subject, so the flag already accounts for that subject's
  own scale. Any analysis sensitive to tail behaviour should check its
  influence explicitly.

Figure: `summary-cohort` (F13) — panel E carries the quality census; it was a
separate figure (F12) until it was folded in, since it asked the same
question of the same subject rows.

### Block order does not explain the Slow − Fast effect

**Result.**

- Subjects were counterbalanced on which condition they met first: 15
  Fast-first, 17 Slow-first (`conditionorder`, assigned from
  `initial_time_ms`, not run numbers).
- **The Slow − Fast effect holds independently in both order groups**, which
  is what counterbalancing has to establish:

| First block | n | Slow − Fast | 95% CI | t | p | positive |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: |
| Fast-first | 15 | +104.1 ms | [24, 185] | 2.77 | .015 | 12/15 |
| Slow-first | 17 | +147.3 ms | [105, 189] | 7.44 | 1.4e-6 | 16/17 |

- The two groups do not differ reliably (Welch t(21.4) = −1.02, p = .32), and
  neither does baseline decision time in either condition (Fast p = .63,
  Slow p = .84; `conditionorderstats`).
- **That null is not evidence of balance, and is reported as a bound.** The
  90% CI on the between-group difference is [−116, +30] ms, so the test
  excludes only order effects larger than **~116 ms — about the size of the
  Slow − Fast effect itself (127 ms)**. It fails a TOST at d = ±0.8 and
  passes only at d = ±1.0. At n = 15 vs 17 this comparison cannot demonstrate
  balance and should not be quoted as if it had.

**Interpretation.**

- The design claim that survives is the within-group one: whichever condition
  a subject started on, they showed the effect. That is sufficient for F04 —
  the proportional stretch is not an artefact of practice or order.
- The between-group comparison is uninformative in both directions. It gives
  no reason to suspect an order effect and no power to rule a moderate one
  out; the honest statement is that this dataset cannot resolve it. A
  numerically larger effect in the Slow-first group (147 vs 104 ms) is well
  inside that uncertainty.

Figure: `conditionorder-balance` (F11).

### Response side: no bias in choice, accuracy, or decision time

**Result.**

- Left and right are balanced on all three measures
  (`choiceside`, `choicesidestats`, paired per subject, n = 32):

| Measure (left − right) | Δ | t(31) | p | 90% CI |
| :--- | ---: | ---: | ---: | :--- |
| choice proportion | +0.016 | 1.14 | .263 | [−0.008, +0.040] |
| accuracy | +0.002 | 0.16 | .877 | [−0.017, +0.020] |
| decision time | −9.8 ms | −0.94 | .354 | [−27.5, +7.9] |
| decision time, Fast | −0.3 ms | — | .979 | — |
| decision time, Slow | −18.9 ms | — | .250 | — |

- All three are bounded, not merely non-significant: the choice bias is
  within ±0.04 (≈4 percentage points), the accuracy difference within ±0.02,
  and the decision-time difference within ±28 ms. None passes a TOST at
  dz = ±0.5, so a small asymmetry is not excluded — but nothing approaching
  a side preference survives.
- One subject sits outside the plotted decision-time range: H21 at +239 ms
  under Slow, running opposite to the group (left *slower*), which pulls the
  group mean toward zero.

**Interpretation.**

- Subjects did not prefer a side, were not more accurate on one, and are not
  reliably faster with one hand once the motor baseline accounts for the
  hand. The two-alternative design is balanced in every way that would bias
  a choice analysis.
- **This is the result of a correction, not the raw data.** Under a single
  pooled motor baseline the same contrast gave −22.8 ms, p = .023 — an
  apparent left-hand speed advantage. The RT runs, which contain no
  deliberation, show a 13.9 ms hand difference of their own, so that
  "advantage" was motor latency the pooled baseline could not remove.
  Subtracting each hand's own baseline reduces it to −9.8 ms (p = .354).
  See "Deliberate deviations from the preprint's analysis," below.
- A residual ~10 ms remains, and the bound cannot exclude it. It is small
  relative to the Fast/Slow effect (127 ms) and common to both conditions,
  so it cannot generate that effect — but lateralised motor preparation is
  measured directly in the MEG analyses, and a ~10 ms per-hand offset is
  worth carrying there.

Figure: `choiceside-asymmetry` (F09).

### Time on task: faster across the session, slower within each block

**Result.**

- Fitted per subject with session-block order, within-block position and a
  Fast/Slow indicator in one model (`timeontask`, `timeontaskstats`, n = 32):

| Term | Slope | t(31) | p | dz |
| :--- | ---: | ---: | ---: | ---: |
| per session block | **−35.8 ms** | −4.14 | 2.5e-4 | −0.73 |
| per within-block trial | **+3.06 ms** | 7.95 | 5.7e-9 | 1.41 |

- Roughly −272 ms across eight task blocks against +180 ms across a 60-trial
  block: practice between blocks, fatigue or disengagement within them.
- **The session decline is a learning curve, not a steady drift.** Blocks 1→3
  account for −198 ms (t(31) = 4.61, p = 6.6e-5); blocks 3→8 for only −73 ms
  (p = .27). The linear coefficient understates the early change and
  overstates the late one. Both conditions show the same shape (panel A).
- **The decline is decisional, not motor.** Motor baseline is a single
  per-subject constant, so in principle motor learning could masquerade as
  decision-time drift — but the two RT runs sit at different points in the
  session and show no change: 451.9 → 446.5 ms, Δ = −5.4 ms, 95% CI
  [−19.7, +8.9], p = .45. Motor effects larger than ~20 ms are excluded,
  against a session drift of 199 ms.
- **Neither slope differs Fast vs. Slow, and here the null is informative.**
  Session drift Δ = −10.4 ms/block, 90% CI [−25.7, +4.9] — the difference is
  bounded well below the drift itself (35.8 ms). Within-block drift
  Δ = +0.25 ms/trial, 90% CI [−1.16, +1.66], against a slope of 3.06. Time on
  task acts on both conditions alike, so it cannot generate the Fast/Slow
  contrast in F04.
- **The confound is confined to within-block position.** Trial class is
  balanced across *session blocks* — 25.1% easy / 21.4% ambiguous / 10.5%
  misleading in every one of the eight — and the session slope is unchanged
  by class controls (−35.90 → −35.82 ms/block). Panels A and C are clean.
- **13% of the within-block slope is trial-class scheduling, not fatigue.**
  Trial class is *not* uniformly distributed across within-block positions —
  the first decile is 29% easy with almost no ambiguous trials, the second is
  25% ambiguous with almost no easy — and easy trials are ~450 ms faster.
  Panel B draws the class-adjusted profile alongside the observed one, so the
  compositional contribution is the gap between them: it shrinks the
  first-to-last rise from +254 ms to +150 ms and the total excursion from
  296 ms to 190 ms.
  Re-fitting with trial-class controls: slope drops from **+3.08 to +2.68
  ms/trial** (95% CI [1.85, 3.52], t(31) = 6.56, p = 2.5e-7; reduction
  reliable at t(31) = 6.13, p = 8.5e-7). The effect is genuine but smaller
  than the uncontrolled coefficient implies.
- **The within-block profile is not linear.** Deviations from a per-subject
  linear fit are reliable across deciles (omnibus F = 18.0, p = 4.5e-24; 6 of
  10 deciles differ from the linear prediction at Bonferroni α = .005), with
  a +123 ms excursion at decile 6. The linear coefficient is a summary, not a
  description of the shape.

**Interpretation.**

- For the story, the load-bearing point is the second one: time on task is
  common to Fast and Slow, so no result in F04–F06 can be attributed to
  practice or fatigue. The bounds make that checkable rather than assumed.
- The class-scheduling confound is worth carrying forward as a caution about
  *any* within-block positional analysis on this dataset — position and
  difficulty are entangled by design, so a raw positional effect always
  mixes the two.
- The non-linear profile is not interpreted here. Its shape does not
  replicate across conditions (Fast and Slow peak at different deciles), so
  it warrants a dedicated analysis before any account is offered.

Figure: `timeontask-drift` (F10) — two panels: session drift by condition,
and the within-block profile observed against class-adjusted. The
per-subject coefficient strips were folded into the panel annotations.

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

### Urgency and evidence quality are separable: the same stretch factor at every difficulty

**Result.**

- Group mean decision time (ms) and accuracy per cell (`conditionclass`,
  n = 32):

| Condition | Easy | Ambiguous | Misleading |
| :--- | ---: | ---: | ---: |
| Fast | 981 / 0.992 | 1367 / 0.767 | 1305 / 0.390 |
| Slow | 1094 / 0.983 | 1518 / 0.779 | 1427 / 0.359 |

- Both factors act strongly and independently on decision time
  (`conditionclassstats`; 2 × 3 repeated-measures ANOVA, n = 32):
  condition ηp² = .51, difficulty ηp² = .90, both p < .001.
- Decision time is **non-monotonic in difficulty**: ambiguous is the slowest
  class, slower than misleading by 62 ms (Fast) and 91 ms (Slow), even though
  misleading is by far the least accurate. Time-to-decide and
  probability-correct come apart — pursued in F05.
- The interaction depends entirely on the scale it is tested on, and only one
  scale asks the right question:

| Interaction tested on | F(2, 62) | p | ηp² |
| :--- | ---: | ---: | ---: |
| decision time in ms | 1.20 | .307 | **.037** |
| **log decision time** | 0.06 | .938 | **.0021** |

- Condition shifts decision time by a *factor*, not by a constant (F04), so
  the millisecond interaction tests a model already known to be wrong. On the
  log scale — where the interaction asks whether Fast→Slow scales every
  difficulty by the same amount — the residual effect is 18× smaller and
  essentially zero.
- Stated positively, the per-class stretch factors are near-identical while
  the millisecond effects are not:

| Class | Slow − Fast | Slow ÷ Fast |
| :--- | ---: | ---: |
| easy | +113 ms | 1.126 |
| ambiguous | +151 ms | 1.121 |
| misleading | +122 ms | 1.117 |

- Bound on the residual additive interaction: easy-vs-misleading
  difference-of-differences = −9.3 ms, 90% CI [−54.1, +35.5] ms.
- **Accuracy shows no condition effect** (ηp² = .04, p = .26), and no
  interaction (ηp² = .06, p = .15). Difficulty dominates it (ηp² = .95,
  p < .001), as designed: easy ≈ 0.99, ambiguous ≈ 0.77, misleading ≈ 0.37.
- Misleading accuracy is **significantly below chance** (0.375,
  t(31) = −5.83, p = 2e-6), which is what the class definition demands rather
  than a performance failure: early evidence points at the wrong target
  (`SP3 < 0.40`), and behaviour follows it. This is a direct check that the
  design-frame classification is selecting the trials it claims to.

**Interpretation.**

- The urgency manipulation rescales the deliberation clock **by the same
  factor regardless of evidence quality** — 1.117 to 1.126 across three
  classes that differ by 400 ms in mean decision time. Urgency and evidence
  quality are separable knobs, which is what licenses treating F04 (rate) and
  F05 (shape) as distinct effects rather than one effect seen twice.
- This is a positive claim, not a failure to detect an interaction. The
  earlier framing ("no interaction on either measure") asserted absence from
  p > .05; the log-scale result instead shows the multiplicative description
  leaves essentially nothing unexplained.
- Subjects took ~11% longer under Slow and were **no more accurate for it**.
  Read cautiously: accuracy here is largely fixed by the stimulus (easy is at
  ceiling, misleading below chance by construction), so there may have been
  little accuracy available to buy. The defensible statement is that the
  manipulation moved time without moving accuracy — see "Deliberate
  deviations from the preprint's analysis," below, for why the preprint's
  error-count comparison suggests otherwise.

Figure: `conditionclass-anova` (F08).

### Fast vs. Slow stretches decision time proportionally, rather than adding a fixed delay

**Result.**

- Slow decisions take ~11% longer than Fast: **1186 ± 70 ms (Fast) vs.
  1314 ± 66 ms (Slow)**, t(31) = −6.19, p = 7.1e-7, dz = −1.09, n = 32
  (`dtdistribution`, `groupstats`).
- Every quantile moves, but by a common *factor*, not a common number of
  milliseconds (`dtdistributionstats`):

| Quantile | Fast | Slow | Slow − Fast | Slow ÷ Fast |
| :--- | ---: | ---: | ---: | ---: |
| q10 | 678 ms | 757 ms | +80 ms | 1.127 |
| q25 | 867 ms | 966 ms | +99 ms | 1.131 |
| q50 | 1128 ms | 1255 ms | +127 ms | 1.139 |
| q75 | 1441 ms | 1601 ms | +160 ms | 1.128 |
| q90 | 1767 ms | 1915 ms | +148 ms | 1.091 |

Both right-hand columns aggregate *per subject*, not across group means:
Δ is the mean of each subject's difference, and the ratio is the geometric
mean of each subject's ratio (the estimator matching the log-ratio test
below). Dividing the two group-mean columns instead gives ~1.11 and is not
the quantity tested.

- **Not a fixed delay.** The millisecond gap nearly doubles from the fast end
  to the slow end: Δq90 vs. Δq10, t(31) = 2.42, p = .021.
- **Consistent with a stretch.** The ratios do not differ across quantiles:
  log-ratio q90 vs. q10, t(31) = −1.60, p = .119.
- **Spread agrees.** A delay moves the mean without touching the SD; a
  stretch scales both. Giving each model its best per-subject fit (that
  subject's own delay `mean_slow − mean_fast`, or their own factor
  `mean_slow ÷ mean_fast`) and asking what it predicts for their Slow trials:

| | Observed Slow | Fixed delay predicts | Stretch predicts |
| :--- | ---: | ---: | ---: |
| SD | 487 ms | 447 ms (t(31) = 2.68, p = .012) | 500 ms (t(31) = −1.01, p = .32) |
| CV | 0.394 | 0.355 (t(31) = 2.52, p = .017) | 0.401 (t(31) = −0.50, p = .62) |

- The observed spread is therefore incompatible with a fixed delay and
  compatible with a proportional stretch.
- **Skewness decides nothing here** (0.89 vs. 1.05, p = .52). It is invariant
  under *any* positive linear transform, so both models predict no change —
  reported for completeness only. Its null *does* rule out a non-linear
  change: Slow occasionally derailing a trial into a long deliberation would
  have inflated it.
- **The per-subject KS ladder is decisive.** Two-sample KS on the full
  trial-level distributions (`trialfeatures`), chance at α = .05 being 1.6:

| Normalization applied before the test | Subjects with Fast ≠ Slow |
| :--- | ---: |
| none (raw decision time) | 19 / 32 |
| each condition's median **subtracted** | 6 / 32 |
| each condition's median **divided out** | 2 / 32 |

- Fitting each subject's five Slow quantiles as a multiple of their Fast
  quantiles gives a stretch factor of **1.117, 95% CI [1.071, 1.163]**
  (t(31) = 5.21 against 1, p = 1.2e-5; 27 of 32 subjects above 1).

**Interpretation.**

- q10/q50/q90 are each subject's trials sorted fastest-to-slowest, read at
  the 10th/50th/90th percentile — the fast end, the typical trial, the slow
  end. Two things could make all three later under Slow: something *added* to
  every trial (same milliseconds at each percentile), or the whole process
  *running slower* (same factor at each percentile, so late trials move more
  in absolute terms).
- The data reject the first and fit the second. **Slow does not delay
  decisions by 127 ms; it makes deliberation take ~11% longer.**
- **Why that matters.** A constant added to every trial is what a change in
  *non-decision time* looks like — and non-decision time is already removed
  here by subtracting each subject's motor baseline, so an additive account
  has nothing left to describe.
- A constant *factor* rescales the deliberation clock itself, which is
  exactly what changing the slope of a rising urgency signal does: a signal
  climbing more slowly crosses a given level proportionally later, stretching
  the distribution instead of translating it. The unchanged CV is that
  signature — pure time-rescaling preserves relative spread.
- **Converges with the model fit** by an independent route: urgency gating is
  favored over bounded integration (ΔBIC = −238.9, t(31) = −8.12, p = 3.6e-9,
  30/32 subjects individually favoring it), and its growth rate differs Fast
  vs. Slow (`urgency_scale` Δ = −0.108, t(31) = −2.52, p = .017;
  `ssmcomparison`, `ssmcomparisonstats`). The proportional form is the more
  specific of the two predictions — bounded integration with a moved
  threshold does not produce it.

**Scope.**

- "Proportional" describes the *group-level* pattern. Within a single subject
  the two models are not separable at this trial count: fitting each
  subject's five quantiles, the scale model's residual (54.9 ms) beats the
  shift model's (62.3 ms) in only 18 of 32 subjects, t(31) = −0.92, p = .37.
  The claim rests on the aggregate tests, not on per-subject model selection.
- A ratio depends on where zero sits. The origin here is decision onset with
  non-decision time subtracted out — the theoretically meaningful zero for a
  deliberation-rate claim, not an arbitrary one.

Figure: `dtdistribution-condition` (F04).

### Decision time by trial class: difficulty changes the shape of the distribution, not just its location

**Result.**

- Easy trials are faster than both harder classes at every quantile, and the
  gap grows toward the tail (`dtdistribution`, `dtdistributionstats`,
  `stratum_type == "class"`):

| Contrast | q10 | q50 | q90 |
| :--- | ---: | ---: | ---: |
| Easy − Ambiguous | −316 ms, t(31)=−9.15, p=2.6e-10 | −420 ms, t(31)=−13.40, p=1.9e-14 | −459 ms, t(31)=−13.07, p=3.8e-14 |
| Easy − Misleading | −209 ms, t(31)=−6.32, p=4.9e-7 | −355 ms, t(31)=−11.30, p=1.6e-12 | −449 ms, t(31)=−10.55, p=8.8e-12 |

- Ambiguous vs. misleading differs in kind, not just size — misleading is
  faster through the fast and typical range, but the two converge in the
  tail:

| Quantile | Δ (ambiguous − misleading) | t(31) | p |
| :--- | ---: | ---: | ---: |
| q10 | +107 ms | 5.65 | 3.3e-6 |
| q50 | +65 ms | 4.17 | 2.3e-4 |
| q90 | +10 ms | 0.31 | .76 (n.s.) |

- Applying the F04 geometry tests to each contrast (fixed delay, proportional
  stretch, or neither?) separates them cleanly. The per-subject KS ladder —
  subjects whose two distributions still differ after removing a per-class
  additive offset, or a multiplicative factor — is the decisive column:

| Contrast | Fixed delay? | Stretch? | KS raw → −median → ÷median |
| :--- | :--- | :--- | :--- |
| Easy vs. Ambiguous | no (p = .004) | no (p = .003) | 32/32 → 10/32 → 6/32 |
| Easy vs. Misleading | no (p < 1e-4) | **yes** (p = .67) | 29/32 → 10/32 → **0/32** |
| Ambiguous vs. Misleading | no (p = .014) | no (p < 1e-4) | 5/32 → 0/32 → 0/32 |

- **Easy vs. misleading is a clean ~1.29× stretch.** Dividing out a per-class
  factor leaves the two distributions indistinguishable in every subject
  (0/32).
- **Easy vs. ambiguous is neither.** Differences grow toward the tail
  (316 → 459 ms) while the ratios *shrink* (1.49 → 1.33), and both
  normalizations leave 6–10 subjects still differing, well above the 1.6
  expected by chance. That residual is a genuine change in distribution
  *shape*, not in location or scale.
- **Ambiguous vs. misleading is group-real but individually weak.** Only 5 of
  32 subjects show any distributional difference, and either normalization
  removes it entirely. The group contrasts aggregate a small effect that is
  consistent in *direction*, not a large one.

**Interpretation.**

- Easy trials carry strong, consistent evidence from early in the sequence
  (`SP2 > 0.60`, rising to `SP8 > 0.75` — see "Trial-classification reference
  frame," above), so the decision variable has almost nowhere to go but
  toward the correct target quickly.
- **Difficulty is a different kind of effect from Fast vs. Slow**, not the
  same one with another cause. Fast/Slow rescales the deliberation clock and
  leaves shape alone — the urgency-slope prediction. Difficulty does not
  behave that way.
- **Ambiguous trials are not "easy trials run slower."** They hover near
  chance (`SP ≈ 0.50`) for a variable, evidence-dependent stretch before
  anything resolves them, and that loitering has no counterpart on an easy
  trial to be stretched *from* — so the shape of the waiting-time
  distribution itself changes when evidence stops being informative.
- **Misleading trials do have a decisive trajectory**, just a delayed and
  reversed one (`SP3 < 0.40` before resolving correctly). The accumulator has
  something concrete to lock onto once it flips, so the same process runs on
  a slower clock — hence the clean stretch.
- Net: manipulating urgency changes the *rate*; manipulating evidence quality
  changes the rate for some contrasts and the *shape* for others.
- **Tail convergence** (ambiguous ≈ misleading at q90) fits urgency forcing a
  decision once neither class has resolved, at which point the specific
  trajectory stops mattering. This is an interpretation consistent with the
  pattern and with the urgency-gating result — not a statistical claim this
  analysis makes.

**The reverse, included.**

- This flips the published result. The preprint, using a response-confounded
  classification (see "Trial-classification reference frame," above), found
  ambiguous *faster* than misleading, and only marginally (t = −1.84,
  p = .077).
- Under the corrected design-frame classification the sign reverses and the
  effect is solid: **misleading is reliably faster than ambiguous** (t = 5.43,
  p = 6.3e-6 on the full-sample mean; see "Preprint Replication," below).
- Mechanism: chosen-frame "misleading" is 53.7% genuine errors on
  easy/ambiguous stimuli, not misleading stimuli, which inflates its mean DT
  under the old classification — diagnosed in "Trial-classification reference
  frame."

Figure: `dtdistribution-class` (F05).

### Success probability at decision: easy resolves at higher confidence; ambiguous and misleading resolve at the same confidence despite different timing

**Result.**

- SPD (success probability at decision) is the design's own
  success-probability trajectory read off at whichever token count the
  subject actually stopped on — how much evidence was in hand at commitment,
  not how long commitment took.
- Mean SPD per class (`validated_15row`; `summary`,
  `groupstats[analysis=spd]`):

| Class | Mean SPD | SEM |
| :--- | ---: | ---: |
| Easy | 0.781 | 0.010 |
| Ambiguous | 0.651 | 0.013 |
| Misleading | 0.647 | 0.006 |

| Contrast | Δ | t(31) | p | dz |
| :--- | ---: | ---: | ---: | ---: |
| Easy − Ambiguous | 0.130 | 17.13 | 2.3e-17 | 3.03 |
| Easy − Misleading | 0.134 | 16.86 | 3.7e-17 | 2.98 |
| Ambiguous − Misleading | 0.0039 | 0.41 | .684 | 0.07 |

- Easy vs. either harder class is about as large an effect as behavioral data
  gets (dz ≈ 3, near-zero group overlap).
- **Ambiguous vs. misleading is stated as a bound, not an absence** —
  p = .684 is not by itself evidence of equivalence. The 90% CI on the paired
  difference is [−0.012, +0.020] SPD, which passes a two-one-sided-tests
  equivalence check at dz = ±0.4 (and ±0.5) but fails at ±0.3.
- Read directly: **the data rule out any difference larger than about
  0.020 SPD (dz ≈ 0.37); a smaller one remains possible and this sample
  cannot exclude it.**

**Interpretation.**

- **A genuine dissociation from the decision-time result above.** In *time*,
  ambiguous and misleading differ (misleading faster through the fast/typical
  range). In *confidence-at-commitment*, any difference is at most a third
  the size of a conventional "small" effect — subjects commit to both hard
  classes at SPD ≈ 0.65 either way, despite reaching that point at different
  speeds.
- Time-to-decide and quality-of-decision are not the same knob here: whatever
  produces the ambiguous-vs-misleading timing difference does not produce a
  confidence difference of any consequential size.
- **Not a fixed evidence threshold.** A single fixed confidence criterion
  would put SPD-at-decision in roughly the same place for every class — the
  threshold doesn't know how the evidence got there. Instead easy resolves at
  0.78 against 0.65 for both hard classes.
- **That fits urgency gating.** Easy trials reach high confidence fast,
  clearing a high bar before urgency would ever force the issue; hard trials
  take longer to reach any given confidence level, so a rising urgency signal
  cuts them off lower — regardless of whether stronger evidence might
  eventually have arrived.
- A third independent line pointing at the same mechanism as H1 (model
  comparison) and the Fast/Slow proportional stretch — this time from
  confidence-at-commitment rather than timing.

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
| Fast vs Slow accuracy † | *(error counts: 45.1 vs 36; `t=6.10`)* | 0.836 vs 0.839; `t=-0.61`, `p=.55` |
| SPD by class | Higher in easy | Paired all-log and 15-row tests emitted |

Fast/Slow effects are close to the preprint. The ambiguous/misleading effect
is reversed; the cause is the trial-classification reference frame (see
Findings, above), not subject exclusion — no four-subject exclusion
reverses the current effect. The `H02` `RT1`/`RT2` motor-baseline choice was
also ruled out (~1.4 ms on group means, cancels in every paired contrast;
`docs/meg.md`, "H02 motor baseline").

### Deliberate deviations from the preprint's analysis

Rows marked † do not reproduce the preprint's statistic, because the
statistic itself is confounded. The deviation is listed here rather than
silently applied.

- **† Fast vs Slow errors → accuracy.** The preprint compares error *counts*
  between conditions. Fast blocks contain more trials than Slow blocks (281
  vs 229 per subject here) because each Fast trial is shorter and the block
  duration is fixed, so the count comparison is confounded by trial number:
  more trials produce more errors at equal skill. Reproducing it exactly
  gives 46.3 vs 36.9, `t=6.06`, `p=1.0e-6` — a clean replication of a number
  that does not mean what it appears to. The error *rate* shows no condition
  effect (0.836 vs 0.839, `t=-0.61`, `p=.55` over all task trials; 0.716 vs
  0.707, `p=.26` restricted to classified trials), and the same null appears
  in the factorial test (`conditionclassstats`, accuracy × condition,
  ηp² = .04, `p=.26`; F08). **The speed manipulation changed decision time
  without changing accuracy.**
- **The motor baseline is estimated per response side.** The preprint
  subtracts one pooled RT latency per subject. The RT runs contain no
  deliberation, yet they show a reliable hand difference — left 442.1 ms vs.
  right 456.0 ms, Δ = 13.9 ms, t(31) = −3.29, p = .003 — so a pooled baseline
  leaves a purely motor asymmetry inside `dt_ms`. Each trial is now corrected
  with the baseline for the hand that answered it (a hand with no usable RT
  response falls back to the pooled value). Effect: the left-minus-right
  asymmetry in decision time drops from −22.8 ms (p = .023) to −9.8 ms
  (p = .354), while **every conclusion elsewhere is unchanged** — verified
  by regenerating the full battery: Slow − Fast +127.4 → +127.1 ms with
  identical t and p; class contrasts within 3 ms; per-class stretch factors
  within 0.001; time-on-task slopes within 0.1; and F06's equivalence bound
  on ambiguous-vs-misleading SPD moves from 0.019 to 0.020 SPD, still
  passing TOST at dz = ±0.4 and failing at ±0.3. The commitment time that
  `logged_spd` is read at shifts by 10.4 ms on average, which is why SPD
  had to be re-derived rather than assumed safe.
- **Trial classification uses the correct-target design frame**, not the
  recorded chosen-target `nProb`. This is the deviation that reverses the
  ambiguous-vs-misleading result; mechanism and evidence in
  "Trial-classification reference frame," above.

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
