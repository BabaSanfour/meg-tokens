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
- `dtgeometry`, `dtgeometrystats` — for each decision-time contrast (Fast vs.
  Slow and the three difficulty pairs), whether the two distributions differ
  by a fixed delay, a proportional stretch, or a change of shape: quantile
  differences and log ratios, SD/CV against what each model predicts, a
  per-subject stretch factor, and the KS ladder before and after removing a
  median offset and a median factor.

### `behavior/analyses/design_effects.py`

- `conditionclass` — DT and accuracy for all six condition × class cells,
  with a fully within-subject 2 × 3 ANOVA (main effects and interaction,
  each against its own `effect × subject` error term).
- `conditionclasscontrasts` — the planned contrasts that ANOVA cannot state:
  per-class Fast→Slow stretch factors (geometric mean, bounded on log
  ratios), their additive counterparts, the easy-vs-misleading bound on the
  leftover additive interaction, and each class's accuracy against chance.
- `choiceside` — left/right choice proportions, DT asymmetry, and accuracy
  asymmetry, overall and per condition.
- `timeontask`, `conditionorder` — block-order effects from the session
  clock (`nInitialTime`), the only field that recovers it, since Fast and
  Slow blocks interleave and `nTrialIndex` restarts at 1 in each run.
- `withinblockdecile` — the same within-block position resolved into deciles
  cut *within each block*, observed and trial-class-adjusted, per condition;
  `withinblockdecilestats` tests the shape the single linear coefficient in
  `timeontask` cannot describe.
- `lapses`, `extremedt`, `extremedttrials` — no-response outcomes
  summarized by LabVIEW code; extreme DTs flagged at a robust
  median-absolute-deviation cutoff and listed by trial, never removed.

### `behavior/analyses/evidence.py`

- `criterionscale`, `criterionscaleprofile`, `criterionscalestats` — the
  evidence criterion against decision time on both available scorings
  (first-order SumLogLR and the exact posterior in log odds), fitted on the
  one trial set where both are defined. Reports each scale's slope, the
  bounded Fast/Slow contrast, a test of departure from linearity, and the
  paired between-scale contrast that is the figure's result.

Two evidence scales are kept distinct. `sum_log_lr_at_decision` is the exact
posterior log odds from Equation 1; certainty is reported at ±log 255 rather
than propagated as infinity. The criterion analysis instead follows Cisek et
al. (2009): first-order chosen-target `SumLogLR`, a fixed log-likelihood
increment per token and therefore proportional to the selected-minus-
unselected token count. It is derived from `token_lead_at_decision` and
`isCorrect` in `behavior/analyses/evidence.py`; it is not silently equated
with the horizon-dependent exact posterior.

- `criteriondecline` — first-order chosen-target SumLogLR at commitment
  against continuous decision time, restricted to complete, alignable 15-row
  logs and excluding pre-first-token anticipations.
- `reversecorrelation` — per-subject logistic weights for each of the
  first eight jumps, tokens after commitment coded as unseen, plus a
  model-free kernel and a Fast/Slow comparison per jump.
- `conditionalaccuracy`, `conditionalaccuracystats` — accuracy across
  within-subject DT quantiles per condition, resolved by trial class and
  reported observed and class-adjusted. Bins are cut on all of a subject's
  trials, so the class rows partition the pooled row and every class is read
  at the same decision times. Registers each stratum's accuracy against
  chance, its slope in two units (per quintile and per second), the
  class-adjusted slope, each class's share of each bin, and the bounded
  Fast/Slow contrast on every slope.
- `continuousevidence`, `continuousevidenceprofile`, `continuousevidenceorder`,
  `continuousevidencestats`
  — early evidence (the design's success probability after 3 of 15 tokens) and
  what it predicts, over every task trial including unclassified random ones.
  The per-subject fits regress DT on evidence strength and accuracy on signed
  evidence, and each carries `n_predictor_levels`, because three tokens admit
  only four splits: those coefficients are contrasts between two and four
  levels, not tested linear trends. The profile resolves every level by trial
  class, and the statistics register the contrast between classes at a level,
  the contrast between levels within a class, and the bounded Fast/Slow
  contrast on each coefficient. The order profile resolves the same window by
  the *sequence* of the first three tokens, because a lead is reachable several
  ways and classes at one level can hold disjoint orders; its statistics give
  each class's share of each order and the effect of the first jump with class
  and level held fixed.
- `urgency` — per-subject intercept and slope of evidence at decision
  against decision time, by condition, on both evidence scales.

### `behavior/analyses/sequential.py`

- `posterror` — the local pair `DT(error+1) − DT(error−1)`, decomposed
  against each subject's own mean into post-error slowing and pre-error
  speeding, with the classical post-error−post-correct contrast beside it.
  Adjacency is broken by any gap in `run_trial_index`.
- `choicehistory` — win-stay/lose-shift and lag-1 side autocorrelation.

### `behavior/analyses/sequential_sampling.py`

- `ssmcomparison` — sequential-sampling model fits, one row per subject,
  condition, and model. `behavior ssm-fit` runs the complete Thura et
  al. (2012) reproduction, doi:10.1152/jn.01071.2011: urgency gating, a
  bounded integrator, a collapsing-bound integrator, and an
  additive-urgency sensitivity model, fit to the same trials under one
  observation model, followed by the compute-node-only aggregation and
  evaluation stages (order: `README.md`, "Stage ordering for the
  sequential-sampling models"). Every candidate is reported with
  log-likelihood, AIC, BIC, and the criterion difference against the
  integrator fit; the group table counts the subjects each model wins, tests
  that difference across subjects, and carries the paired Fast-minus-Slow
  contrast on every fitted urgency parameter. The four candidates, why each is
  included, and the shared conventions: "Model specification," below.

### `behavior/analyses/individual.py`

- `individualprofile`, `individualcorrelations` — subject-level SAT
  adjustment, urgency slope, criterion slope, evidence sensitivity,
  accuracy, and lapse rate, correlated pairwise.
  `behavior characterization --neural-metrics` will additionally join a
  subject-level MEG table into `individualprofile`; nothing reads those
  columns yet, and no figure draws them.
- `speciescomparison` — DT by class, SP at decision by class, criterion
  decline, and the urgency-gating results (the model's advantage over
  integration, its fitted urgency growth, and that growth's
  Fast-minus-Slow contrast). Values from the papers are not reproduced in
  code; see Findings, below, for
  measured values.

## Figures

`meg-tokens report behavior --figures <keys or groups>` renders the figure
battery; `--list-figures` prints the full registry. The rules every figure
obeys, the output/sidecar contract, and the result families that get no figure
are in "Figure conventions and output contract" below. The registry contains 18
figures, numbered in scientific-narrative order (foundation, commitment
policy, mechanistic adjudication, sequential adaptation, individual
differences) rather than implementation order, and runs contiguously from
F01 to F18. F-numbers are not durable identifiers; cite the figure key when a
reference has to survive.

| F# | Key | Derivative(s) | Shows |
| :--- | :--- | :--- | :--- |
| F01 | `summary-cohort` | `summary`, `lapses`, `extremedt` | Cohort composition and data quality |
| F02 | `conditionorder-balance` | `conditionorder`, `conditionorderstats` | First-block counterbalancing (between-subject) |
| F03 | `choiceside-asymmetry` | `choiceside`, `choicesidestats` | Left − right asymmetry in choice, decision time, accuracy |
| F04 | `timeontask-drift` | `timeontaskstats`, `withinblockdecile` | Session and within-block drift |
| F05 | `conditionclass-anova` | `conditionclass`, `conditionclassstats`, `conditionclasscontrasts` | Decision time and accuracy by condition x difficulty |
| F06 | `dtdistribution-condition` | `dtdistribution`, `dtdistributionstats`, `dtgeometry`, `groupstats` | Decision time, Fast vs. Slow (`dt_ms`, not `rawRT`) |
| F07 | `dtdistribution-class` | `dtdistribution`, `dtdistributionstats` | Decision time by trial class |
| F08 | `spdcumulative-class` | `spdcumulative`, `groupstats` | SPD at decision, cumulative by class |
| F09 | `criterionscale-comparison` | `criterionscale`, `criterionscaleprofile`, `criterionscalestats` | Evidence criterion vs. decision time, on the first-order and exact-posterior scales |
| F10 | `conditionalaccuracy-caf` | `conditionalaccuracy`, `conditionalaccuracystats` | Conditional accuracy by trial class, observed vs. class-adjusted |
| F11 | `continuousevidence-effects` | `continuousevidenceprofile`, `continuousevidencestats` | Early evidence by trial class (incl. unclassified trials) |
| F12 | `reversecorrelation-kernel` | `reversecorrelation`, `reversecorrelationstats` | Psychophysical kernel by token jump |
| F13 | `ssmcomparison-model` | `ssmcomparison`, `ssmcomparisonstats`, `ssmtimecourse`, `ssmdistributioncheckstats`, `ssmheldout`, `ssmheldoutstats`, `ssmheldoutpairwise` | H1: the four models' criterion, decision variable and absolute fit (A-C), then the adjudication in sample and out (D-F) |
| F14 | `ssmcomparison-urgency` | `ssmcomparison`, `ssmcomparisonstats`, `ssmboundaryaudit`, `ssmrobustnessstats` | H2: all four urgency parameters Fast vs. Slow, and the contrast on the one that moves against its six refits |
| F15 | `ssmparameterrecovery-identifiability` | `ssmparameterrecovery`, `ssmparameterrecoverystats`, `ssmmodelrecoverystats` | Parameter and model recovery in the fitted region: the contested pair against truth, every model × parameter correlation, the selection confusion matrix, and recovery against solver resolution |
| F16 | `posterror-slowing` | `posterror`, `posterrorstats` | Post-error slowing and pre-error speeding, and the two conventional measures that combine them |
| F17 | `choicehistory-effects` | `choicehistory`, `choicehistorystats` | Win-stay/lose-shift and side autocorrelation |
| F18 | `individualcorrelations-structure` | `individualprofile`, `individualcorrelations` | Correlation matrix (A) and the five cells that carry the argument (B–F) |

## Figure conventions and output contract

Rules that hold for every figure in the battery, and that any new one must
obey. They are cheap to state and expensive to retrofit.

### Invariants

1. **The plotting layer never computes an inferential statistic.** Any `t`,
   `p`, `F`, `df`, `dz`, `ηp²`, `r` printed on a figure is read from a
   derivative. The plotting layer may compute only display quantities: means,
   SEMs, within-subject confidence intervals, kernel densities, binned means,
   quantiles of already-persisted subject-level values, and an OLS line with
   its band. This is what keeps a figure from ever disagreeing with the
   Findings above.
2. **Decision time is `dt_ms`, never `rawRT`.** The motor baseline is already
   subtracted in Stage 2.
3. **Trial-level layers select `primary_analysis_eligible == True` and finite
   `dt_ms`** — the same selection the analyses used. Any deviation is recorded
   in the sidecar.
4. **`unclassified` trials are excluded from class-keyed panels**, and the
   exclusion is stated in the caption and sidecar. They are 43.0% of trials;
   silently dropping them is not acceptable, and neither is drawing them as a
   fourth class. **One deliberate exception, F10** (`conditionalaccuracy-caf`):
   its claim is that the pooled line *is* the class lines weighted by their
   share of each bin, and that arithmetic cannot be shown while omitting
   41–45% of the weight in every bin. Any further exception needs the same
   kind of argument, not this one as precedent.
5. **Case normalisation.** `trialfeatures.condition` is `Fast`/`Slow`/`RT`;
   every group derivative uses `fast`/`slow`. Normalise with `.str.lower()` at
   every join. `RT`-condition rows never enter a behavioral figure.
6. **Repeated-measures contrasts are drawn as paired/slope plots, not bars.**
   Bars hide the within-subject pairing every contrast is built on.
7. **Identity is never carried by colour alone.** Every multi-series panel has
   a legend, and ≤ 4 series are additionally direct-labelled or distinguished
   by marker/line style.
8. **No multiplicity correction is applied, anywhere**, matching
   `docs/data_contract.md` ("Contrasts are never chosen from observed results
   and receive no multiplicity correction"). Every sidecar records
   `"multiplicity_correction": "none"`, and panels with more than three
   contrasts state it in the annotation block or their caption.
9. **Non-convergence is shown, not hidden.** Where a derivative carries a
   `converged` column (`ssmcomparison` among the live ones), non-converged
   rows are excluded from the drawn estimate and the retained *n* is printed
   on the panel.

### Drawing conventions

- Numeric tick labels stay visible; reduce their *number* rather than hiding
  them. Tick marks themselves are removed globally.
- Use `Δ = value [unit] marker` for compact comparison brackets, and only for
  a genuine contrast — a correlation or a rate labelled `Δ` claims a
  comparison that was never made. Significance is a marker (`***`, `n.s.`),
  not a spelled-out p; the exact `t/df/p/dz` belongs in the derivative and
  sidecar.
- Correlations print without their leading zero (`-.81`), as p-values do.
- Long annotations must be designed for the actual panel width. They can make
  `constrained_layout` collapse an axis even when the text is technically
  drawable. Two short lines usually beat one long one.
- Put exclusions and methodological caveats in the sidecar or the caption
  rather than overcrowding a title.
- SEM is used for descriptive subject-balanced bin means. Repeated-measures
  trajectories and subject-coefficient summaries use appropriate
  within-subject or group 95% CIs.
- The type scale lives in `reports/style.py`'s rc block. Axis labels, titles,
  ticks and legends are never resized at a call site; only the three in-panel
  constants (`SIGNIFICANCE_PT`, `ANNOTATION_PT`, `DENSE_PT`) apply there.
- **A passing synthetic report test is necessary but insufficient.** Regenerate
  every changed figure from the real derivatives, render its PDF, and look at
  the image.
- A display rule belongs to the *variable* it filters, not to the figure it was
  first written for. A cohort-fraction threshold suits a **continuous**
  predictor whose tail thins gradually; applied to a predictor with a handful
  of **design levels** it deletes whole stimulus configurations. Before reusing
  a threshold, check which kind the new variable is.

### Result families with no figure — and why

| Family / derivative | Disposition |
| :--- | :--- |
| `trialfeatures` | Not a result. It is the trial-level layer behind the distribution, drift, criterion, posterior, and continuous-evidence figures. |
| `groupstats` | Not a figure of its own; its rows are drawn as figure annotations. Drawing a table as a chart adds nothing. |
| `extremedttrials` | Summarized as subject counts in the cohort-composition panel; the per-trial derivative needs no figure of its own. |
| `speciescomparison` | Every one of its ten rows is already drawn in the act that produced it (F07, F08, F09, F13, F14); why it gets no figure of its own is in "Cross-species translation". The derivative is still computed. |
| `ssmtrialpredictions` | Trial-level model-derived regressors for the MEG join (`criterion_at_decision`, `decision_variable_at_decision`, `predicted_accuracy`), not a behavioural result. Revisit when the source-space features they join to exist. |
| `individualprofile` × MEG metrics | No figure: there is no subject-level MEG table, and its r/p would have to be computed in the plotting layer against rule 1. `behavior characterization --neural-metrics` performs the analysis-side join. |
| Response vigor | Movement time is not recorded (Known Issues, Stage 1); nothing must be plotted against the null field. |

### Output format, naming, sidecars

**Formats.** Two files per figure, same stem: `.pdf` (vector,
`pdf.fonttype = 42`, the publication artefact) and `.png` (raster at
`dpi = 400`, for quick viewing). `--formats` may narrow this; `.svg` is
accepted for figures destined for hand-editing.

**Path.** `<root>/derivatives/sub-group/fig/sub-group_task-tokens_desc-<analysis>-<view>_behavior.pdf`,
where the figure key `<analysis>-<view>` becomes the `desc` entity so a figure
file names its own source, and `<analysis>` is the derivative that supplies its
statistics. Both segments are lowercase alphanumeric and are recorded verbatim
in the sidecar (`analysis`, `view`), under the `desc` rule of
`docs/data_contract.md` ("Paths and filenames"), which a test enforces.

**Sidecars.** One per figure, written with `save_sidecar`. Note the collision
and use it deliberately: `io.sidecar_path` calls `Path.with_suffix(".json")`,
so `…_behavior.pdf` and `…_behavior.png` map to the *same* `…_behavior.json`.
Write the sidecar once, after all formats, and list them in a `formats` field;
never call `save_sidecar` per format.

```json
{
  "stage": "behavior_report",
  "figure": "<analysis>-<view>",
  "analysis": "<analysis>",
  "view": "<view>",
  "kind": "<what the figure is, not what it found>",
  "title": "<figure title, matching the registry>",
  "source_derivatives": ["/abs/.../desc-<analysis>_beh.tsv"],
  "columns_read": {"<derivative>": ["<exactly the columns the builder reads>"]},
  "statistics_source": "<derivative>[<the filter that selects the rows drawn>]",
  "subjects": ["H01", "..."],
  "n_subjects": 32,
  "palette": {"<series>": "#008300"},
  "significance_convention": "*** p<.001, ** p<.01, * p<.05, n.s. otherwise",
  "multiplicity_correction": "none",
  "caveat": null,
  "formats": [".pdf", ".png"],
  "dpi": 400
}
```

`columns_read` is not decoration: it makes "which variable is this figure
actually showing?" answerable without reading code. A test asserts that `dt_ms`
appears, and `rawRT` does not, in the decision-time figures' sidecars.

## Known Issues (Stage 0–1)

Data-quality issues inherited from the original recording and parsing code,
how each was handled, and its impact.

### Stage 0: Raw BIDSification

| Issue | Handling | Impact |
| :--- | :--- | :--- |
| 3 non-standard `.tdms` filenames (scratch fragments: H03 `temp_180214.tdms`, H18 `temp_181024.tdms`, H23 `temp_181121.tdms`) | Explicit `behavior_ignore_files` allowlist; any other non-matching file still errors | 320/323 retained runs |
| H01, H05 raw-session duration counts don't match 8-Slow/Fast + 2-RT | `KNOWN_SESSION_OVERRIDES`, cross-referenced against the acquisition notebook (H01 labeled `Pilot01` there); fingerprint independently reproduces all 17 entries | H05 RT1 confirmed unmatched — real gap, not ambiguity |
| H01 headshape not found on any known drive | Left `not_found` / review | No headshape derivative for H01 |
| H01 Slow3, H06 Slow2, H10 Fast2: Start-pulse count mismatch at the trailing boundary | `KNOWN_TRAILING_TRIAL_MISMATCHES`, staged with a note instead of sent to review | 3 runs recovered |
| H06 Start/Go trigger codes swapped vs. every other subject | `SUBJECT_TRIGGER_CODES` in `meg_bids.py` (event code `524288` is Start for H06, `262144` elsewhere) | H06 matches and epochs correctly |

Permanent gaps — H05 RT1 (no raw MEG counterpart; excluded from H05's
MEG-joined analyses) and empty-room, H07/H10 anatomy, and H26/H27's corrupt
empty-room (redirected to `_02.ds`) — are in `docs/data_contract.md`, "Known
dataset gaps".

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

Ordered as a scientific argument, not by figure number:

- Cohort composition and data quality first.
- Checks on block order, response side, and time on task next.
- The classification reference frame used by the class-based results.
- Finally the condition, difficulty, distribution, and confidence results.

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

Figure: `summary-cohort` (F01) — panel E carries the quality census, folded
in from a standalone figure because it asked the same question of the same
subject rows.

### Block order does not explain the Slow − Fast effect

**Result.**

- Subjects were counterbalanced on which condition they met first: 15
  Fast-first, 17 Slow-first (`conditionorder`, assigned from
  `initial_time_ms`, not run numbers).
- **The Slow − Fast effect holds independently in both order groups**, which
  is what counterbalancing has to establish (`conditionorderstats`, rows with
  `test = one_sample_vs_zero`; these are the markers drawn on the figure):

| First block | n | Slow − Fast | 95% CI | t | p | positive |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: |
| Fast-first | 15 | +104.1 ms | [24, 185] | 2.77 | .015 | 12/15 |
| Slow-first | 17 | +147.3 ms | [105, 189] | 7.44 | 1.4e-6 | 16/17 |

- The two groups do not differ reliably (Welch t(21.4) = −1.02, p = .32), and
  neither does baseline decision time in either condition (Fast p = .63,
  Slow p = .84; `conditionorderstats`).
- **That null is not evidence of balance, and is reported as a bound.** The
  90% CI on the between-group difference is [−116, +30] ms
  (`conditionorderstats`, `test = welch_bound`), so the test
  excludes only order effects larger than **~116 ms — about the size of the
  Slow − Fast effect itself (127 ms)**. It fails a TOST at d = ±0.8 and
  passes only at d = ±1.0. At n = 15 vs 17 this comparison cannot demonstrate
  balance and should not be quoted as if it had.

**Interpretation.**

- The design claim that survives is the within-group one: whichever condition
  a subject started on, they showed the effect. That is sufficient for F06 —
  the proportional stretch is not an artifact of practice or order.
- The between-group comparison is uninformative in both directions. It gives
  no reason to suspect an order effect and no power to rule a moderate one
  out; the honest statement is that this dataset cannot resolve it. A
  numerically larger effect in the Slow-first group (147 vs 104 ms) is well
  inside that uncertainty.

Figure: `conditionorder-balance` (F02).

### Response side: no bias in choice, accuracy, or decision time

**Result.**

- Side 1 is the right target, answered with the right hand; side 2 the left.
  Every task log places target 1 at 0°, 65 px right of centre, and target 2
  at 180° (323 of 323 files), and tokens of direction 1 land on the right;
  presses on side 1 drive the left motor cortex in 31 of 32 subjects
  (`docs/meg.md`, Source reconstruction); the original analysis scripts
  name side 2 left.
- Left and right are balanced on all three measures
  (`choiceside`, `choicesidestats`, paired per subject, n = 32):

| Measure (left − right) | Δ | t(31) | p | 90% CI |
| :--- | ---: | ---: | ---: | :--- |
| choice proportion | −0.016 | −1.14 | .263 | [−0.040, +0.008] |
| accuracy | −0.002 | −0.16 | .877 | [−0.020, +0.017] |
| decision time | +9.8 ms | 0.94 | .354 | [−7.9, +27.5] |
| decision time, Fast | +0.3 ms | — | .979 | — |
| decision time, Slow | +18.9 ms | — | .250 | — |

- All three are bounded, not merely non-significant: the choice bias is
  within ±0.04 (≈4 percentage points), the accuracy difference within ±0.02,
  and the decision-time difference within ±28 ms. None passes a TOST at
  dz = ±0.5, so a small asymmetry is not excluded — but nothing approaching
  a side preference survives.
- One subject sits outside the plotted decision-time range: H21 at −239 ms
  under Slow, running opposite to the group (right *slower*), which pulls the
  group mean toward zero.

**Interpretation.**

- Subjects did not prefer a side, were not more accurate on one, and are not
  reliably faster with one hand once the motor baseline accounts for the
  hand. The two-alternative design is balanced in every way that would bias
  a choice analysis.
- **This is the result of a correction, not the raw data.** A single pooled
  motor baseline gives +22.8 ms, p = .023 — the right hand's shorter motor
  latency, not slower deciding on the left. See "Deliberate deviations from the preprint's analysis,"
  below.
- A residual ~10 ms remains, and the bound cannot exclude it. It is small
  relative to the Fast/Slow effect (127 ms) and common to both conditions,
  so it cannot generate that effect — but lateralised motor preparation is
  measured directly in the MEG analyses, and a ~10 ms per-hand offset is
  worth carrying there.

Figure: `choiceside-asymmetry` (F03).

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
  contrast in F06.
- **The confound is confined to within-block position.** Trial class is
  balanced across *session blocks* — 25.1% easy / 21.4% ambiguous / 10.5%
  misleading in every one of the eight — and the session slope is unchanged
  by class controls (−35.90 → −35.82 ms/block). Panel A is clean.
- **Composition accounts for at most 13% of the within-block slope.** Trial
  class is *not* uniformly distributed across within-block positions — the
  first decile is 29% easy with almost no ambiguous trials, the second is 25%
  ambiguous with almost no easy — and easy trials are ~450 ms faster.
  Re-fitting with trial-class controls drops the slope from **+3.08 to +2.68
  ms/trial** (95% CI [1.85, 3.52], t(31) = 6.56, p = 2.5e-7; the reduction
  itself reliable at t(31) = 6.13, p = 8.5e-7). The uncontrolled +3.08 is that
  re-fit's own baseline; the +3.06 in the table above comes from the pooled
  model, which also carries the Fast/Slow indicator.
  Both numbers are bounds rather than point estimates. Class and position are
  partially confounded, so removing per-class means also removes genuine drift
  lying between the classes — the adjustment can understate real fatigue but
  never overstate it. Read **+2.68 ms/trial as a floor on the positional
  effect and 13% as a ceiling on composition's share of the slope.**
- **Panel B shows a larger correction than 13%, and both are right.** The
  panel draws the class-adjusted profile against the observed one, so the
  compositional contribution is the gap between them: it shrinks the
  first-to-last rise from +254 to +150 ms (41%) and the group profile's total
  excursion from 296 to 190 ms (36%). The slope is a linear fit weighting all
  ten deciles equally; the rise and the excursion are endpoint and range
  measures, dominated by exactly the deciles where composition is most skewed
  — the class-imbalanced first two and the decile-6 peak. Quote the 13% for
  the slope and the panel numbers for the shape; neither is the compositional
  share of the other.
- **The within-block profile is not linear.** Each subject's decile means
  have their own least-squares line in decile number removed; the residual
  profile still varies reliably across deciles (F(9, 279) = 16.24,
  p = 2.2e-21, ηp² = .34), and 6 of 10 deciles depart from the linear
  prediction at Bonferroni α = .005, the largest being +123 ms at decile 6.
  The linear coefficient is a summary, not a description of the shape.

**Interpretation.**

- For the story, the load-bearing point is the second one: time on task is
  common to Fast and Slow, so no result in F06, F07, F08 can be attributed to
  practice or fatigue. The bounds make that checkable rather than assumed.
- The class-scheduling confound is worth carrying forward as a caution about
  *any* within-block positional analysis on this dataset — position and
  difficulty are entangled by design, so a raw positional effect always
  mixes the two.
- The non-linear profile is not interpreted here. Its shape does not
  replicate across conditions — the group profile peaks at decile 9 in Fast
  and decile 6 in Slow (`withinblockdecilestats`, term `peak_decile`) — so it
  warrants a dedicated analysis before any account is offered. That peak
  decile is descriptive and deliberately carries no test: an argmax over ten
  noisy cells is grounds for withholding an interpretation, not for making
  one.

Figure: `timeontask-drift` (F04) — two panels: session drift by condition,
and the within-block profile observed against class-adjusted. The
per-subject coefficient strips were folded into the panel annotations. Each
panel's Fast-minus-Slow line carries its 90% interval rather than a
significance marker, since the informative content of both nulls is how
tightly the difference is bounded against the slope above it. Panel B reads
its deciles and class adjustment from `withinblockdecile`, so the numbers
quoted above and the drawn curve come from the same table.

### Trial-classification reference frame

**Method.**

- Two candidate frames for classifying random (`x`) trials: the recorded
  runtime `nProb`, referenced to the *chosen* target (the original
  notebook's method), vs. a profile derived from `sTokenDirs`, referenced to
  the *correct* target.
- `SP_chosen = 1 - SP_correct` whenever the subject erred, so the chosen
  frame's class assignment depends on the response it is later used to
  explain — a circularity the design frame avoids.

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

**Interpretation.**

- The correct-target design frame is the valid method.
- Chosen-frame "misleading" is 53.7% genuine errors on easy/ambiguous
  stimuli, not misleading stimuli — the confound inflates its mean DT and
  produces the preprint's ambiguous-vs-misleading sign as an artifact, not a
  real effect.
- No random trial can be classified misleading under the design frame: after
  3 jumps the correct target never trails in a random sequence — a property
  of the stimulus set (`e` splits 3-0 always, `a` splits 2-1/1-2, `m` splits
  1-2 [n=1,706] or 0-3 [n=12], `x` only ever 2-1 or 3-0), not an error.
- `infer_random_classes=false` (recorded labels only, symmetric with the
  preprint's treatment of the random 60% as a non-category) gives the
  closest Easy-vs-Ambiguous match to the preprint (t=-15.12 vs -15.04);
  under it misleading is slightly *faster* than ambiguous — coherent, since
  misleading evidence invites early commitment while ambiguous evidence
  delays it.

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
  condition ηp² = .51, difficulty ηp² = .90, both p < .001. **These are the
  log-scale effects**, matching panel A's annotation and the scale the
  interaction is judged on below; on raw milliseconds they are .50 and .86.
- Decision time is **non-monotonic in difficulty**: ambiguous is the slowest
  class, slower than misleading by 63 ms (Fast) and 91 ms (Slow), even though
  misleading is by far the least accurate. Time-to-decide and
  probability-correct come apart — pursued in F07.
- The interaction depends entirely on the scale it is tested on, and only one
  scale asks the right question:

| Interaction tested on | F(2, 62) | p | ηp² |
| :--- | ---: | ---: | ---: |
| decision time in ms | 1.20 | .307 | **.037** |
| **log decision time** | 0.06 | .938 | **.0021** |

- Condition shifts decision time by a *factor*, not by a constant (F06), so
  the millisecond interaction tests a model already known to be wrong. On the
  log scale — where the interaction asks whether Fast→Slow scales every
  difficulty by the same amount — the residual effect is 18× smaller and
  essentially zero.
- Stated positively, the per-class stretch factors are near-identical while
  the millisecond effects are not (`conditionclasscontrasts`; the factor is a
  geometric mean, tested and bounded on log ratios to match the log-scale
  interaction above):

| Class | Slow − Fast | Slow ÷ Fast | 95% CI on the factor |
| :--- | ---: | ---: | ---: |
| easy | +113 ms | 1.126 | [1.081, 1.172] |
| ambiguous | +151 ms | 1.121 | [1.080, 1.164] |
| misleading | +122 ms | 1.117 | [1.052, 1.187] |

- The three intervals overlap almost entirely and all exclude 1.0: each class
  stretches reliably, and by an indistinguishable amount.
- Bound on the residual additive interaction: easy-vs-misleading
  difference-of-differences = −9.3 ms, 90% CI [−54.1, +35.5] ms
  (`conditionclasscontrasts`, term `additive_interaction_ms`; annotated on
  panel A in place of a significance marker). Quote it as a bound on an
  effect argued to be absent, never as evidence for the null.
- **Accuracy shows no condition effect** (ηp² = .04, p = .26), and no
  interaction (ηp² = .06, p = .15). Difficulty dominates it (ηp² = .95,
  p < .001), as designed: easy ≈ 0.99, ambiguous ≈ 0.77, misleading ≈ 0.37.
- Misleading accuracy is **significantly below chance** (0.375, 95% CI
  [0.331, 0.418], t(31) = −5.83, p = 2.0e-6; `conditionclasscontrasts`, term
  `accuracy_vs_chance`, pooled over Fast and Slow since the class definition
  is a stimulus property), which is what the class definition demands rather
  than a performance failure: early evidence points at the wrong target
  (`SP3 < 0.40`), and behaviour follows it. This is a direct check that the
  design-frame classification is selecting the trials it claims to.

**Interpretation.**

- The urgency manipulation rescales the deliberation clock **by the same
  factor regardless of evidence quality** — 1.117 to 1.126 across three
  classes that differ by 400 ms in mean decision time. Urgency and evidence
  quality are separable knobs, which is what licenses treating F06 (rate) and
  F07 (shape) as distinct effects rather than one effect seen twice.
- This is a positive claim, not a failure to detect an interaction: the
  log-scale result shows the multiplicative description leaves essentially
  nothing unexplained.
- Subjects took ~11% longer under Slow and were **no more accurate for it**.
  Read cautiously: accuracy here is largely fixed by the stimulus (easy is at
  ceiling, misleading below chance by construction), so there may have been
  little accuracy available to buy. The defensible statement is that the
  manipulation moved time without moving accuracy — see "Deliberate
  deviations from the preprint's analysis," below, for why the preprint's
  error-count comparison suggests otherwise.

Figure: `conditionclass-anova` (F05).

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
  log-ratio q90 vs. q10, t(31) = −1.60, p = .119, 95% CI [−0.073, +0.009].
  Both tests, the spread comparison below, the KS ladder and the stretch
  factor come from `dtgeometry`/`dtgeometrystats`, which run this same
  delay-versus-stretch-versus-shape protocol on every contrast in F06 and
  F07. The stretch row is a non-rejection and must be quoted as the bound
  above, never as evidence the ratios are identical.
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
  favored over bounded integration (ΔBIC = −239.6, t(31) = −8.14, p = 3.4e-9,
  30/32 subjects individually favoring it), and its growth rate differs Fast
  vs. Slow (`urgency_scale` Δ = −0.114, t(31) = −2.60, p = .014;
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

Figure: `dtdistribution-condition` (F06).

### Decision time by trial class: difficulty changes the shape of the distribution, not just its location

**Result.**

- Easy trials are faster than both harder classes at every quantile, and the
  gap grows toward the tail (`dtdistribution`, `dtdistributionstats`,
  `stratum_type == "class"`):

| Contrast | q10 | q50 | q90 |
| :--- | ---: | ---: | ---: |
| Easy − Ambiguous | −316 ms, t(31)=−9.15, p=2.6e-10 | −420 ms, t(31)=−13.40, p=1.9e-14 | −459 ms, t(31)=−13.07, p=3.8e-14 |
| Easy − Misleading | −209 ms, t(31)=−6.32, p=4.9e-7 | −355 ms, t(31)=−11.30, p=1.6e-12 | −449 ms, t(31)=−10.55, p=8.8e-12 |

- The per-class distributions those contrasts are taken from, as group means
  of the subject-level statistics (`dtdistribution`, `stratum_type = class`):

| Class | Mean | q10 | q50 | q90 | Skew |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Easy | 1033 ms | 607 | 968 | 1510 | 1.21 |
| Ambiguous | 1434 ms | 923 | 1389 | 1969 | 0.73 |
| Misleading | 1358 ms | 816 | 1324 | 1959 | 0.46 |

  Easy is both the fastest and the most right-skewed; ambiguous and misleading
  are close in the tail and separate at the fast end, which is the contrast
  the next table tests.
- Ambiguous vs. misleading differs in kind, not just size — misleading is
  faster through the fast and typical range, but the two converge in the
  tail:

| Quantile | Δ (ambiguous − misleading) | t(31) | p |
| :--- | ---: | ---: | ---: |
| q10 | +107 ms | 5.65 | 3.3e-6 |
| q50 | +65 ms | 4.17 | 2.3e-4 |
| q90 | +10 ms | 0.31 | .76 (n.s.) |

- Applying the F06 geometry tests to each contrast (fixed delay, proportional
  stretch, or neither?) separates them cleanly (`dtgeometrystats`; 1.6 of 32
  subjects differ by chance at α = .05, recorded there as
  `n_expected_by_chance`). The per-subject KS ladder —
  subjects whose two distributions still differ after removing a per-class
  additive offset, or a multiplicative factor — is the decisive column:

| Contrast | Fixed delay? | Stretch? | KS raw → −median → ÷median |
| :--- | :--- | :--- | :--- |
| Easy vs. Ambiguous | no (p = .004) | no (p = .003) | 32/32 → 10/32 → 6/32 |
| Easy vs. Misleading | no (p < 1e-4) | **yes** (p = .67) | 29/32 → 10/32 → **0/32** |
| Ambiguous vs. Misleading | no (p = .014) | no (p < 1e-4) | 5/32 → 0/32 → 0/32 |

- **Easy vs. misleading is a clean ~1.33× stretch** (`dtgeometrystats`,
  term `stretch_factor`: 1.331, 95% CI [1.281, 1.381]; the least-squares
  slope of misleading's quantile ladder on easy's, through the origin).
  Dividing out a per-class factor leaves the two distributions
  indistinguishable in every subject (0/32).
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
  p = 6.3e-6 on the full-sample mean). Mechanism and the 28-subject
  frame-by-frame comparison: "Trial-classification reference frame," above.

Figure: `dtdistribution-class` (F07).

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

- The cumulative distributions behind those means show the shape, not just the
  location (`spdcumulative`, `view = all_logged`; pooled proportion of trials
  committed at or below each SPD):

| SPD at or below | Easy | Ambiguous | Misleading |
| :--- | ---: | ---: | ---: |
| 0.40 | 0.009 | 0.076 | 0.062 |
| 0.60 | 0.045 | 0.284 | 0.247 |
| 0.80 | 0.439 | 0.779 | **0.861** |

  Ambiguous and misleading agree closely at the low thresholds and separate at
  0.80, where misleading is the most tightly concentrated of the three: 86% of
  those commitments are made below 0.80 SPD. The 15-row sensitivity view is the
  same table under `view = validated_15row`.
- Easy vs. either harder class is about as large an effect as behavioral data
  gets (dz ≈ 3, near-zero group overlap).
- **Ambiguous vs. misleading is stated as a bound, not an absence** —
  p = .684 is not by itself evidence of equivalence. The 90% CI on the paired
  difference is [−0.012, +0.020] SPD (`groupstats`, `ci_low`/`ci_high`).
- Read directly: **the data rule out any difference larger than 0.0202 SPD
  (dz = 0.372); a smaller one remains possible and this sample cannot exclude
  it.** That is `groupstats`'s `equivalence_bound`/`equivalence_bound_dz`: the
  smallest bound a two-one-sided-tests check passes at α = .05, which is the
  90% interval's furthest endpoint. It is reported as one number rather than
  as a list of candidate bounds that do and do not pass.

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

Figure: `spdcumulative-class` (F08).

### The declining evidence criterion is an artifact of the evidence scale

**Result.**

- The criterion is how much evidence favouring the chosen target a subject held
  at the moment of commitment. A criterion that *falls* with decision time —
  settling for weaker support the longer a trial runs — is the classic
  behavioral signature of urgency gating, and is what the first-order measure
  reports here.
- Two scorings of "evidence" are available. On the same 10,849 trials, the same
  32 subjects and the same regression, they disagree in sign
  (`criterionscale`, `criterionscalestats`):

| Evidence scale | Slope (per s) | 95% CI | t(31) | p | dz |
| :--- | ---: | :--- | ---: | ---: | ---: |
| First-order SumLogLR | **−0.133** | [−0.191, −0.075] | −4.66 | 5.8e-5 | −0.82 |
| Exact posterior (log odds) | **+0.305** | [+0.186, +0.423] | 5.24 | 1.1e-5 | 0.93 |

- The contrast is paired within subject: **Δ = −0.438 per s**, t(31) = −9.92,
  p = 3.9e-11, dz = −1.75. The two measures nonetheless correlate across
  subjects (r = 0.68, p = 2.2e-5) — the same people rank alike on both while
  the population sign is opposite, which is what a systematic change of scale
  looks like and what noise or an outlying subgroup does not.
- **The reversal follows from the finite horizon.** SumLogLR is the
  chosen-target token lead times `log(0.604736 / 0.395264)`, so it cannot see
  how many tokens remain. The trial always ends at 15, so the same lead means
  progressively more:

| A lead of 3 | Exact P(win) | First-order SumLogLR |
| :--- | ---: | ---: |
| after token 5 (10 remain) | 0.83 | 1.276 |
| after token 11 (4 remain) | 0.94 | 1.276 |
| after token 13 (2 remain) | **1.00** | 1.276 |

  A subject holding one *unchanging* posterior threshold must therefore accept
  a smaller and smaller token lead as the chance to overturn it disappears.
  Measured in SumLogLR their criterion declines; their policy never moved.
- **Neither scale identifies the true sign.** The first-order scale is
  confounded by the horizon. The exact posterior is read off the design's
  success-probability grid, which coarsens sharply late in a trial, so part of
  the positive slope is discretization rather than a rising criterion. The two
  are contaminated in opposite directions, and this diagnostic cannot
  adjudicate between them.
- **Neither scale shows a Fast/Slow difference, and both nulls are reported as
  bounds.** First-order Fast − Slow = +0.041, 90% CI [−0.024, +0.106]; exact
  −0.047, 90% CI [−0.226, +0.133]. The first-order bound is 0.8× the pooled
  slope it is measured against: this test could have missed a condition
  difference nearly the size of the entire effect, so `p = .29` is close to
  uninformative. Quote the bounds.
- **The profile is not a straight line.** Removing each subject's own linear
  fit and testing the residual profile: first-order F(8, 200) = 2.80,
  p = .0059, ηp² = .10; exact F(8, 200) = 1.95, p = .055, ηp² = .07. The fitted
  slope is a summary, not a claim that the criterion moves by the same amount
  at every token.

**Method.**

- Both scales are fitted on one trial set, which is what makes the difference
  between them attributable to the scale rather than to the sample: complete
  15-token logs with a commitment after the first token, 10,849 of 16,324
  primary trials. A 14-row log cannot be aligned to the token present at
  commitment, so the criterion is *undefined* there rather than noisy; a
  pre-first-token commitment has no sensory evidence to score. The exclusion is
  near-neutral for the time axis — mean decision time differs by −9.6 ms
  (t = −1.0), and the excluded fraction runs 27–38% across decision-time
  deciles with no monotone trend.
- Trial-level OLS per subject against decision time in seconds. The raw success
  probability is deliberately not fitted: it is bounded at 1 and compresses
  exactly where late commitments sit, which flattens the contrast instead of
  testing it. Both reported scales are log-odds, so their slopes are
  comparable.
- Binning is 200 ms. Panels A and B draw every bin held by at least half the
  cohort; the linearity omnibus requires 90%. The two rules differ on purpose:
  a repeated-measures ANOVA is listwise, and at the display rule it retains 14
  bins but only 10 of 32 subjects — those with unusually wide decision-time
  distributions. The stricter rule tests nine bins spanning 0.5–2.1 s, holding
  88% of trials, on 26 subjects.

**Interpretation.**

- **A declining first-order criterion is not by itself evidence of urgency on
  this task.** Any fixed-horizon token design will produce one whether or not
  the decision policy changes, so the measure cannot distinguish a falling
  bound from an unchanging one. This is a general caution for replications of
  Cisek et al. (2009), not a peculiarity of this dataset.
- **The urgency account is unaffected.** It rests on the sequential-sampling
  comparison — urgency gating beats bounded integration by ΔBIC −239.6
  (t(31) = −8.14, p = 3.4e-9, 30/32 subjects), confirmed out of sample. What is
  withdrawn is the claim that a model-free measure independently corroborates
  it; the model comparison never depended on this figure.
- A model-free criterion plot is interpretable only once the evidence scale
  accounts for the evidence still to come. Neither scale available here does
  so cleanly, which is the honest limit of the diagnostic.

Figure: `criterionscale-comparison` (F09) — A and B are the same trials and the
same subjects, scored two ways, so the reversal between them is attributable to
the scale alone; C shows it holding within subject rather than only in the group
mean. The figure carries no colour scope: black is the observed series and grey
the derived one in every panel.

### The conditional accuracy function measures trial composition, not a speed–accuracy trade-off

**Result.**

- Pooled over classes, accuracy falls steadily across within-subject
  decision-time quintiles (`conditionalaccuracy`,
  `conditionalaccuracystats`, n = 32, ~102 trials per subject per bin):

| Quintile | Bin mean DT | Accuracy | 95% CI |
| :--- | ---: | ---: | :--- |
| 1 | 678 ms | 0.897 | [0.880, 0.914] |
| 2 | 964 ms | 0.872 | [0.857, 0.887] |
| 3 | 1183 ms | 0.843 | [0.826, 0.861] |
| 4 | 1444 ms | 0.798 | [0.775, 0.822] |
| 5 | 1949 ms | 0.777 | [0.758, 0.795] |

- The trend is **−0.0314 accuracy per quintile**, 95% CI [−0.0372, −0.0255],
  t(31) = −10.92, p = 3.8e-12 — equivalently **−0.112 per second** of the
  drawn axis, [−0.136, −0.088], t(31) = −9.53, p = 9.9e-11. Both units are
  registered (`accuracy_slope_per_bin`, `accuracy_slope_per_second`); the
  figure annotates the per-second one because its x-axis is time.
- **Resolved by trial class the function does not survive.** The same bins,
  the same subjects, split by what the trials actually are:

| Class | Bin 1 → bin 5 accuracy | Slope per quintile | 95% CI | p |
| :--- | :--- | ---: | :--- | ---: |
| easy | 0.996 → 0.900 | −0.0203 | [−0.0290, −0.0116] | 4.3e-5 |
| ambiguous | 0.726 → 0.772 | +0.0013 | [−0.0180, +0.0207] | .89 |
| misleading | **0.085 → 0.599** | **+0.1288** | [+0.1104, +0.1471] | 3.4e-15 |
| unclassified | 0.963 → 0.807 | −0.0391 | [−0.0474, −0.0309] | 7.1e-11 |

- **The one class whose early evidence actively misleads runs the other way,
  and by far the largest amount.** Misleading trials start at 8.5% correct —
  reliably *below* chance, t(31) = −21.88, p = 2.1e-20, which is what
  `SP3 < 0.40` demands — and end at 59.9%, reliably *above* it,
  t(31) = 3.62, p = .0010. Accuracy on these trials is a direct function of
  how long the subject waited, and the crossing of chance is the effect.
- **The ambiguous null is stated as a bound.** Its slope excludes any effect
  larger than **0.0174 accuracy per quintile** (`equivalence_bound`, the
  furthest endpoint of the 90% interval), against a pooled slope of 0.0314.
  Restricting to bins holding at least five ambiguous trials — a sensitivity
  check, since dropping thin cells selects on the axis under test — gives
  −0.0087, 95% CI [−0.0237, +0.0063], p = .25. Flat either way.
- **The pooled decline is composition.** Which trials occupy a bin changes
  drastically across the function (`class_proportion_first_to_last`,
  bin 1 → bin 5):

| Class | Share of bin 1 | Share of bin 5 | Δ | p |
| :--- | ---: | ---: | ---: | ---: |
| easy | 43.2% | 10.4% | −0.328 | 9.8e-18 |
| ambiguous | 7.6% | 34.1% | +0.264 | 1.4e-17 |
| misleading | 7.2% | 14.7% | +0.075 | 1.6e-8 |
| unclassified | 42.0% | 40.8% | −0.012 | .52 |

  The fastest bin is nearly half easy trials, which are at ceiling; the
  slowest is a third ambiguous, which are not. Removing each subject's
  per-class mean and adding back their grand mean — the same adjustment F04's
  panel B uses — cuts the slope from **−0.0314 to −0.0067 per quintile**,
  95% CI [−0.0109, −0.0024], t(31) = −3.18, p = .0033. **Composition accounts
  for 79% of it.**
- **Attributed by class, the same split says which stratum carries what**
  (`accuracy_change_decomposition`). Each subject's first-to-last change
  (−0.1201) splits exactly into a composition part and a within-class part —
  for one class, `p₅a₅ − p₁a₁ ≡ Δp·ā + p̄·Δa` — so the two terms sum to that
  subject's own observed change with no residual:

| | Composition | Within class |
| :--- | ---: | ---: |
| easy | −0.3096 | −0.0278 |
| ambiguous | +0.1973 | +0.0097 (n.s.) |
| misleading | +0.0277 | **+0.0558** |
| unclassified | −0.0091 (n.s.) | **−0.0641** |
| **total** | **−0.0937**, 78.1% | **−0.0263** |

- **The second route agrees, but not to the precision the two numbers
  suggest.** The composition share of the endpoint change is **78.1%** and the
  share of the slope removed by the class adjustment is **78.8%** — different
  quantities on different supports, one splitting two bins and the other
  fitting five. Both are ratios of subject means, so neither has a per-subject
  value to average (one subject's own accuracy change is positive, and their
  share is unbounded); the comparison is a paired bootstrap over subjects
  instead, 20,000 resamples, both routes driven by the same resample
  (`composition_share_agreement`). The routes differ by **−0.007**, 95% CI
  [−0.155, +0.165], p = .89, and the same test in per-second slope units gives
  −0.008. So the two routes are not distinguishable — but the comparison
  excludes only differences larger than **0.135**, and each route's own
  interval is wide (endpoint [0.61, 1.00], slope [0.70, 0.90]). Agreement to
  0.7 of a percentage point is finer than anything here can resolve and should
  not be read as the routes corroborating each other to the decimal. What is
  supported is the claim actually made: most of the pooled decline is
  composition, by either construction.
- **The within-class part is carried entirely by `unclassified`.** The three
  designed classes' within-class terms sum to **+0.0377**, 95% CI
  [+0.0156, +0.0599], t(31) = 3.48, p = .0015 — reliably *positive*. Holding
  the mixture constant, then, accuracy among the named classes **rises** with
  decision time, and the whole of the residual decline comes from
  `unclassified`. (This is a statement about the within-class term, not about
  what a three-class figure would draw: such a figure would still fall, at
  −0.027 per quintile, because easy → ambiguous mixing still drives it down.)
- **That is why the figure draws four strata and not three.** Each bin is a
  mixture, and the pooled line is the class lines weighted by their share of
  that bin — arithmetic, not interpretation. The four strata reproduce it
  (0.896 / 0.873 / 0.841 / 0.794 / 0.774 against an observed 0.897 / 0.872 /
  0.843 / 0.798 / 0.777); three cannot, because `unclassified` is 41–45% of
  the weight in every bin. A figure claiming the pooled decline *is* the class
  lines plus their mixing cannot demonstrate that while omitting the largest
  class. `unclassified` also contributes almost nothing to the composition
  term — its share of a bin is flat — so it is in the figure for the residual,
  not for the mixture.
- **The residual is real but is an upper bound, not an estimate.** The
  adjustment removes *between*-class composition only, and `unclassified` is a
  residual category rather than a designed one: evidence strength varies
  inside it, so part of its −0.0641 is composition the adjustment cannot see.
  The analysis behind F11 measures that directly
  (`continuousevidencestats`, term `evidence_contrast_within_class`): within
  `unclassified`, accuracy rises **+0.110** between the two early-evidence
  levels it spans, against +0.011 within easy and nothing within ambiguous.
  F11's panels draw only the three designed classes, so this number is quoted
  from the derivative rather than read off that figure. So −0.0067 per quintile bounds any
  genuine accuracy-versus-time relation from above rather than estimating one.
- The endpoint form of that residual is weaker than the slope form: the
  within-class term above is −0.0263, 95% CI [−0.0539, +0.0012], p = .060.
  The slope is the better-powered statement — it uses all five bins, the
  endpoint change uses two — and it is the one quoted. Both describe the same
  residual, as F04's panel-B note describes for its own two measures.
- **Neither the pooled nor the adjusted slope differs Fast vs. Slow, and here
  the null is tight.** Pooled Δ = +0.0005 per quintile, 90% CI
  [−0.0048, +0.0057]; class-adjusted Δ = +0.0017, [−0.0026, +0.0059]. The
  bound is 0.18× the pooled slope it is measured against, so unlike F09's
  condition null this one genuinely excludes a difference of consequence.

**Interpretation.**

- **A falling conditional accuracy function is not evidence about the decision
  policy on this design.** It is what a stimulus schedule that front-loads easy
  trials and back-loads ambiguous ones produces regardless of policy. The
  earlier reading of this figure — that slow decisions are not more accurate —
  quoted the pooled slope as if the bins held comparable trials, and they do
  not.
- **Read by class, the sign is set by the evidence, not by the clock.** Where
  early evidence points the right way (easy) accuracy is at ceiling and can
  only fall; where it points the wrong way (misleading) waiting is what
  recovers the correct answer, and accuracy climbs from below chance to above
  it. Where it points nowhere (ambiguous) time buys nothing, bounded at
  0.017 per quintile. That is three different relations between time and
  accuracy in one dataset, and the pooled curve shows none of them.
- **The misleading trajectory is the informative one for the urgency account.**
  It says the accumulated evidence really does keep improving on these trials,
  so a subject who commits early on them is not failing to extract information
  that is there — they are committing before it arrives. That is a cost of a
  time-dependent commitment policy, and it is measured here rather than
  assumed. It is not by itself evidence *for* urgency gating: a fixed bound
  reached at different times would also produce it. The mechanistic
  adjudication is F13–F14.
- The tight Fast/Slow null adds a small constraint on that adjudication: the
  urgency manipulation moves *when* subjects commit (F06, +11%) without
  measurably changing how accuracy relates to commitment time.

Figure: `conditionalaccuracy-caf` (F10) — A is the four strata against the same
bins with chance drawn in; B is the pooled function against its class-adjusted
counterpart, the same construction as F04's panel B, so the gap between the two
lines *is* the compositional part. Slopes are annotated per second because the
axis is time; the per-quintile values quoted above are registered alongside
them. The ambiguous slope carries its equivalence bound in place of a
significance marker.

### Early evidence constrains the trial but does not determine it — and it is not a continuous predictor

**Result.**

- Early evidence is the design's success probability after **3 of 15 tokens** —
  the earliest point the classification rule uses (`SP3 < 0.40` defines
  misleading) and one where 71.4% of trials are still running, against 45.0%
  by jump 5 and 16.6% by jump 8.
- **Three tokens admit exactly four splits, so the predictor has exactly four
  values**: 0.194, 0.387, 0.613, 0.806. `sum_log_lr_design_early` is exactly
  `logit(sp_design_early)`, so it has the same four. Nothing about it is
  continuous. The per-unit coefficients in `continuousevidence` are contrasts
  between levels — the decision-time fit sees **2** distinct predictor values
  and the accuracy fit **4** (`n_predictor_levels`) — and the registered
  −270.0 ms per unit is arithmetically identical to each subject's difference
  of two means over the gap between them. Quote
  `evidence_contrast_within_class`, in milliseconds, instead.
- **Each class occupies exactly two adjacent levels, forced by the
  classification rule**, so no series spans the axis and the gaps are a
  property of the stimulus set rather than missing data:

| Level | easy | ambiguous | misleading | unclassified |
| :--- | ---: | ---: | ---: | ---: |
| 0.194 (lead −3) | – | – | 12 | – |
| 0.387 (lead −1) | – | 641 | 1,706 | – |
| 0.613 (lead +1) | 2,179 | 2,842 | – | 6,175 |
| 0.806 (lead +3) | 1,919 | – | – | 850 |

  Ambiguous requires `SP2 = 0.50`, so its first two tokens split 1-1 and the
  third can only make the lead ±1. Easy requires `SP2 > 0.60`, so its lead is
  +1 or +3. Misleading is *defined* by `SP3 < 0.40` — the plotted variable is
  in its own rule — so its lead is negative. No random sequence leaves the
  correct target trailing after three tokens, so unclassified is
  positive-only. The 0-3 level holds 12 trials in the whole dataset and is
  drawn anyway, with its count on the axis.
- **At a fixed level, class still decides the trial** — the figure's result,
  and a within-subject comparison in both clusters that hold one
  (`class_contrast_at_level`, n = 32):

| Level | Contrast | Δ decision time | Δ accuracy |
| :--- | :--- | ---: | ---: |
| 0.61 | ambiguous − easy | **+350.5 ms**, dz = 2.80 | **−0.207**, dz = −3.11 |
| 0.39 | ambiguous − misleading | +78.5 ms, p = .0018 | **+0.372**, dz = 2.50 |

- The 0.39 pair is the sharpest test: those trials share the
  success-probability profile through the first eight tokens and separate only
  at `SP11` (see "Trial-classification reference frame"). Their timing barely
  differs while their accuracy differs by 0.372 — 0.749 against 0.377, either
  side of chance.
- **Holding class fixed, the early read moves behaviour in some classes and
  not others** (`evidence_contrast_within_class`; misleading has one testable
  level and so no contrast):

| Class | Levels | Δ decision time | Δ accuracy |
| :--- | :--- | ---: | ---: |
| easy | 0.61 → 0.81 | −105.9 ms, t(31) = −7.57 | +0.011, p = .034 |
| ambiguous | 0.39 → 0.61 | −3.8 ms, \|Δ\| < 35.6 ms | +0.027, \|Δ\| < 0.094 |

  `unclassified`, which the figure does not draw, gives −119.7 ms and
  **+0.110** accuracy over 0.61 → 0.81; that accuracy effect is what F10 cites
  when it calls its class-adjusted residual an upper bound.
- **Neither coefficient differs Fast vs. Slow, and both nulls are bounded.**
  Decision-time slope Δ = +1.5 ms per unit, excluding anything above 27.5
  against a slope of 270; accuracy slope Δ = −0.064, bound 0.238 against 2.39.

**Token order: a confound that runs the wrong way**
(`continuousevidenceorder`, `continuousevidencestats`).

- "The same early evidence" holds for the *strength* after three tokens, not
  for the sequence. A 2-1 lead is reachable three ways, and at SP = 0.61 the
  classes hold disjoint sets of them (`order_composition`): **every** easy
  trial is correct-correct-wrong, and **no** ambiguous trial is (71.3%
  correct-wrong-correct, 28.7% wrong-correct-correct). Given the primacy
  kernel, those are not the same evidence to a subject.
- The order effect is real and sizeable. Holding class *and* level fixed and
  contrasting a correct first jump against a wrong one
  (`first_token_contrast`), a correct first jump makes the decision
  **slower**: +82.7 ms within ambiguous at 0.61 (t(31) = 4.66, p = .0001) and
  +109.5 ms within unclassified at the same level (t(31) = 4.48, p = .0001).
- **It therefore suppresses the reported gaps rather than creating them.** Easy
  is the all-correct-first group and should be ~80–110 ms *slower* on order
  alone, yet it is 350 ms faster than ambiguous. The class contrasts above are
  floors, not inflated estimates.

**Interpretation.**

- **This figure does not license a "continuous evidence" claim.** A
  four-level factor fitted with OLS
  returns a slope, but there is no linearity to test and no shape to see.
- **It also cannot corroborate the class analysis, because it is largely a
  recoding of it.** All 1,718 misleading trials sit below chance and no easy or
  unclassified trial does; the 0.81 level holds only easy and unclassified
  trials; and the cut itself is the one the classification rule reads. That
  the two analyses "agree, so the three-class reduction is not creating the DT
  effect" therefore carries much less than it appears to — they are largely
  the same variable.
- **What it establishes is the size of what early evidence leaves open.** Three
  tokens fix a great deal — accuracy runs 0.38 to 0.99 across the levels — and
  still leave 350 ms of decision time and 0.21 of accuracy to be settled by
  the tokens that follow. Subjects track the sequence rather than committing to
  the early read.
- **That is the measurement F10's residual needed.** F10 removes between-class
  composition and leaves −0.0067 per quintile, reported as an upper bound
  because evidence strength still varies *within* a class. Here that variation
  is measured: real in easy and unclassified, absent in ambiguous.

Figure: `continuousevidence-effects` (F11) — A and B are **clustered by
evidence level**, one marker per class within each cluster, chance marked on B.
The encoding follows the claim: the comparison is *within* a level, so it is
the vertical spread inside a cluster rather than a gap between the ends of two
lines. Error bars are within-subject across the classes in a cluster, which is
the contrast each marker reports; the faint connectors carry the weaker
across-level claim. Only the three designed classes are drawn. The key and the
derivative names still say "continuous": keys are the durable reference, and
derivative names follow the key rather than the claim.

### The choice kernel shows primacy; the Fast/Slow reweighting survives correction only at jumps 2–4

**Result.**

- Per-subject logistic weights for each of the first eight token jumps, with
  tokens after commitment coded as unseen (`reversecorrelation`,
  `reversecorrelationstats`, n = 32). **Every jump carries weight**: 8 of 8
  depart from zero at the corrected threshold (`all_jumps_vs_zero`).
- **The kernel shows clear primacy**, falling monotonically and more than
  halving across the window:

| Jump | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Weight | 2.69 | 2.39 | 2.40 | 2.07 | 1.75 | 1.75 | 1.65 | 1.26 |

- **The condition effect is confined to jumps 2–4.** Eight independent
  per-jump tests need a corrected threshold; at Bonferroni α = .00625 the
  Fast-minus-Slow difference survives at **jumps 2, 3 and 4 only**. Earlier
  drafts reported "jumps 1–5", which holds only at an uncorrected .05 and is
  withdrawn.
- **"Identical from jump 6" is not established.** The late differences are
  small but their bounds are wide relative to the weights they sit on:
  jump 6 Δ = +0.29 excluding only differences above 0.74, jump 7 Δ = +0.24
  excluding above 0.72, jump 8 Δ = +0.005 excluding above 0.48 — against a
  jump-8 weight of 1.26. The data are consistent with Fast and Slow weighting
  late tokens alike and cannot demonstrate it.
- Fits converged for all 32 subjects pooled and 28 of 32 within each condition;
  the rest are separated data and are reported as non-converged rather than as
  large arbitrary weights.

**Interpretation.**

- Primacy is the solid result: early tokens dominate the choice, and the
  weight on the eighth jump is less than half that on the first. That is the
  expected signature of a decision that commits before the sequence ends, and
  it is consistent with the urgency account without being specific to it.
- The reweighting result should be stated as **jumps 2–4**, not 1–5, and
  described as "Fast weights early tokens more heavily" rather than as a
  crossover at a particular jump. Where the difference vanishes is not
  identified by this analysis.

Figure: `reversecorrelation-kernel` (F12) — A is the kernel with within-subject
intervals, B the Fast-minus-Slow difference at each jump with 95% intervals on
the paired difference. Rings mark the corrected threshold, printed on the panel
so it cannot be mistaken for .05. Per-subject traces are not drawn: one subject
reaches −3 while the group lines sit between 1.26 and 2.69, so the traces set
the y-range and flatten the three series they exist to contextualise.

### Urgency gating beats bounded integration, and beats a collapsing bound — but the mechanism is not thereby identified

**Result.**

- Four sequential-sampling models were fitted to the same
  `primary_analysis_eligible` trials, with the same observation model and
  validation folds: a fixed-bound integrator (`ddm`), the 2012 urgency-gating
  reproduction (`urgency`), a hyperbolically collapsing bound (`collapsing`),
  and a label-symmetric additive-drive variant (`additive_urgency`).
  Specification, shared conventions, and why each is included: "Model
  specification," below.
- **In sample**, ΔBIC against the integrator (negative favours the candidate;
  `ssmcomparisonstats`, `condition = all`, n = 32):

| Model | ΔBIC vs. `ddm` | t(31) | p | subjects favouring |
| :--- | ---: | ---: | ---: | ---: |
| `urgency` | **−239.6** | −8.14 | 3.4e-9 | 30/32 |
| `collapsing` | **−169.0** | −10.34 | 1.4e-11 | 31/32 |
| `additive_urgency` | −19.0 | −1.34 | .19 | 16/32 |

- **Out of sample**, three-fold held-out log-likelihood per trial (positive
  favours the candidate; `ssmheldout`, `ssmheldoutstats`):

| Model | Δ per-trial LL vs. `ddm` | 95% CI | p | subjects favouring |
| :--- | ---: | :--- | ---: | ---: |
| `urgency` | **+0.244** | [+0.183, +0.306] | 4.0e-9 | 30/32 |
| `collapsing` | **+0.174** | [+0.139, +0.209] | 1.9e-11 | 31/32 |
| `additive_urgency` | +0.009 | [−0.020, +0.038] | .52 | 13/32 |

- The two time-dependent accounts are therefore both real, and **the direct
  contrast separates them**: urgency beats the collapsing bound out of sample
  by +0.070 per-trial log-likelihood, 95% CI [+0.041, +0.100], t(31) = 4.87,
  p = 3.1e-5, in 28 of 32 subjects (`ssmheldoutpairwise`). This is the
  comparison that matters, because a collapsing bound can mimic a declining
  behavioral criterion just as well as urgency gating can.
- **Every comparison among the candidates points the same way**
  (`ssmheldoutpairwise`, `condition = all`, per-trial held-out
  log-likelihood, positive favours the first model):

| Contrast | Δ per-trial LL | 95% CI | t(31) | p | dz | subjects favouring |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: |
| urgency − collapsing | **+0.070** | [+0.041, +0.100] | 4.87 | 3.1e-5 | 0.86 | 28/32 |
| urgency − additive | **+0.235** | [+0.179, +0.291] | 8.54 | 1.2e-9 | 1.51 | 30/32 |
| collapsing − additive | **+0.165** | [+0.128, +0.201] | 9.24 | 2.0e-10 | 1.63 | 31/32 |

  These three are the *complete* set of comparisons among the candidates. Note
  what that leaves: the six pairwise means over four models are exactly
  additive (urgency − collapsing plus collapsing − `ddm` equals urgency −
  `ddm`, to the last digit), so the three candidate-versus-`ddm` means in the
  table above are arithmetic on these three. Their intervals and p-values are
  separately estimated and so are not redundant as tests, but they are not
  independent evidence: four models yield three independent held-out
  differences, not six.
- **These margins are group effects, not per-subject constants.** The
  urgency-minus-collapsing difference ranges from −0.166 to +0.201 across
  subjects, and absolute held-out fit varies six-fold (−1.27 to −0.186 per
  trial). The ordering is consistent — 28 to 31 of 32 subjects in every
  contrast — but no individual subject's margin is a measurement, for the
  same reason the fitted urgency parameters are not (below).
- **Additive urgency is not supported, and that is a weak conclusion by
  construction.** A common additive drive to two race channels cancels in a
  signed difference (see "Model specification"), so this variant is a
  sensitivity analysis, not a test of additive mechanisms. Its null should not
  be read as evidence against additive urgency in general.
- **Model selection and parameter recovery both hold in the region the fits
  actually occupy** (F15; `ssmparameterrecoverystats`, `ssmmodelrecoverystats`).
  Truths are drawn from the accepted per-subject fits rather than from a
  hypercube over the optimizer bounds, at the production estimator's three
  starts, with each synthetic dataset sized to the real median subject ×
  condition cell (240 trials over eight empirical lead paths).

  The correct model wins in **47 of 48** cases — one true-`ddm` case selects
  `collapsing` — with a median BIC margin between best and second of **22.8
  points**; 18/48 cases are decided by under 10 points.

  The urgency parameters recover essentially perfectly:

| parameter | r | 95% CI |
| :--- | ---: | :--- |
| `urgency_scale` | **0.994** | [+0.98, +1.00] |
| `urgency_onset_s` | **0.986** | [+0.95, +1.00] |
| `additive_scale` | **0.904** | [+0.69, +0.97] |
| `drift_scale` (urgency) | 0.983 | [+0.94, +1.00] |

  **This reverses an earlier finding that the urgency parameters are
  unidentified**, which came from a recovery study that drew its truths from
  the optimizer bounds, a region disjoint from the fitted values, and so asked
  whether the models are separable *somewhere* in parameter space rather than
  where the fits lie. The limits that remain (`nondecision_s`,
  `boundary_rate`) are under "What is identified, and what is not," below.
- **Six-configuration robustness: the winner never changes**
  (`ssmrobustness`, `ssmrobustnessstats`). Varying the filter time constant
  (100/300 ms), the solver grid (20 ms), the post-horizon evidence
  convention, and the parameter bounds, `urgency` wins in every configuration
  at 30/32 subjects, ΔBIC −231 to −247. Shifts relative to that run's own
  baseline are real but single-digit (−7.8 to +8.7 BIC points) against
  effects of 150–250 points. `expanded_bounds` does not change the winner
  despite many urgency-scale and onset estimates sitting near the original
  bound.
- **Censoring at the bound attenuates the Fast/Slow urgency contrast rather
  than creating it.** The obvious objection to the `urgency_scale` condition
  contrast is that estimates pinned at the upper bound manufacture it. Relaxing
  the bound makes it **larger**, not smaller: Fast-minus-Slow goes from −0.118
  (t(31) = −2.74, p = .010) at baseline to −0.181 (t(31) = −2.56, p = .016)
  under `expanded_bounds`, where no cell is censored
  (`ssmrobustnessstats`, `analysis = ssm_robustness_parameter_contrast`). It
  holds in all six configurations (−0.101 to −0.181, p = .010 to .021). This
  bounds the censoring objection, and the recovery result corroborates it:
  with `urgency_scale` recovering at r = 0.99 in the fitted region, a
  subject's estimate is a measurement of that subject, so the contrast is a
  contrast between measured values.
- **The win does not come from the subjects whose fit has no
  observed-information SE.** The result does not depend on which subjects
  yielded a finite Hessian (`ssmheldoutrobustness`,
  `analysis = ssm_heldout_identifiability_subset`):

| Subset | n | urgency − collapsing | t | p |
| :--- | ---: | ---: | ---: | ---: |
| all subjects | 32 | +0.0703 | 4.87 | 3.1e-5 |
| `urgency_scale` not pinned at its bound | 24 | +0.0671 | 4.11 | .00043 |
| finite observed-information SE | 17 | +0.0654 | 3.52 | .0029 |

  Subjects with a pinned estimate and subjects without one show
  indistinguishable advantages (Welch t = −0.36, p = .73), and the effect is
  the same size in all three subsets.
- **No single subject carries it, and it does not need normality**
  (`analysis = ssm_heldout_influence`). Leave-one-subject-out on
  urgency − collapsing: the least favourable of all 32 deletions still gives
  t = 4.61, p = 6.9e-5. Wilcoxon p = 2.27e-5; sign test p = 1.93e-5. H20 — the
  subject flagged as influential elsewhere in this write-up — is the *least*
  favourable one to drop, so removing it strengthens rather than weakens the
  contrast.
- **Pooling Fast and Slow does not manufacture the advantage**
  (`analysis = ssm_heldout_condition_pooling`). If the flexible models were
  being rewarded for absorbing a condition mixture the fixed bound cannot
  represent, the pooled advantage would exceed the sum of the within-condition
  ones. It does not: urgency −239.60 pooled against −249.59 for Fast + Slow,
  collapsing −169.04 against −172.16. Pooling is very slightly conservative.
- **Urgency is the only model that reproduces the error-RT distribution.**
  Selection says which model predicts best; it does not say whether any of
  them fits. Observed minus predicted decision time in seconds,
  subject-averaged (`ssmdistributioncheckstats`, `condition = all`,
  `trial_class = all`; negative means the model is too slow):

| Model | correct q10 | correct q50 | correct q90 | error q10 | error q50 | error q90 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `urgency` | −0.023 | −0.001 | +0.054 | +0.090 | +0.048 | +0.009 |
| `collapsing` | +0.009 | +0.042 | −0.055 | +0.169 | +0.178 | −0.085 |
| `ddm` | +0.109 | +0.035 | **−0.375** | **+0.373** | **+0.337** | −0.157 |
| `additive_urgency` | +0.150 | +0.014 | −0.111 | **+0.499** | +0.292 | +0.007 |

  The fixed bound fails **qualitatively**, not marginally: it overshoots the
  90th correct percentile by 375 ms and misses the fast errors by 373 ms.
  Urgency is within ~50 ms everywhere and within 10 ms at the median. This is
  an absolute-fit argument and is stronger than the BIC table, because it does
  not depend on the model set being complete.
- **Every model under-predicts accuracy, and the winner is the worst at it.**
  Held out, observed accuracy is 0.8373; predicted: `ddm` 0.8200,
  `additive_urgency` 0.8129, `collapsing` 0.8073, `urgency` 0.8027
  (`ssmheldoutpredictions`, `condition = all`). The ordering on accuracy is the
  **reverse** of the ordering on likelihood. The gap is small (3.5 pp for
  urgency) and systematic across all four models, which points at a shared
  feature of the observation model — the 2% lapse mixture, or the non-decision
  floor — rather than at a defect specific to urgency. It is stated here rather
  than absorbed into a fit-quality panel, because a reader entitled to the
  likelihood ranking is also entitled to know it does not transfer to accuracy.

**What is identified, and what is not.**

- **Identified: which of the four models predicts these data best**, in and
  out of sample, among four models that recover 47 of 48 in the region these
  fits occupy, at this design's trial count and estimator.
- **Identified: the individual fitted urgency parameters** —
  `urgency_scale` r = 0.994, `urgency_onset_s` r = 0.986, `drift_scale`
  r = 0.983, `additive_scale` r = 0.904, `collapse_rate` r = 0.939
  (`ssmparameterrecoverystats`, F15).

  Two cautions stand. Each correlation is
  estimated from **12 recovery cases**, so the intervals are wide enough that
  the parameters should not be ranked against one another by r. And
  `boundary_rate` is 0.33–0.67, because truths drawn from real fits inherit
  fits that sit at a bound — a subject whose `urgency_scale` is pinned at 2.0
  is still censored, and F14 draws those points as rings for that reason.
- **Not identified: `nondecision_s`, for `urgency`, `ddm` and `collapsing`**
  (r = −0.29, 0.24, −0.44). This is a resolution limit rather than a statement
  about the models: across the fitted cells those parameters span 0.007–0.009 s,
  which is 0.7–0.9 of the solver's 10 ms step, while `additive_urgency`'s
  non-decision spans 0.976 s — 98 steps — and recovers at r = 1.000. A
  quantity finer than the discretisation that generated it cannot be estimated
  from it. What is fitted is the
  residual after motor correction, a few milliseconds, so this is the expected
  result rather than a surprising one — but no individual non-decision estimate
  should be read for those three models.
- **Not identified: that the time dependence is multiplicative.** Both
  `urgency` and `collapsing` implement a criterion that falls with time; the
  held-out contrast prefers the multiplicative form, but "urgency-like time
  dependence" is what the data establish, not "a growing urgency signal
  multiplying a leaky evidence trace." F09 makes the complementary point from
  the other side: a declining first-order criterion is produced by the finite
  horizon whether or not the policy changes.
- **Not addressed: reward-rate optimality.** Verified payoff, error and
  timeout penalties, and the complete inter-trial timing were never
  established from the acquisition material, so no optimality claim is made.
  The remaining-token schedule is in the task log (`tTime`, every token's
  time): after the press the tokens jump every 20 ms in Fast blocks and every
  170 ms in Slow blocks, and the trial ends 143 ms and 1,024 ms after the
  press (medians, all 16,324 trials). This is an omission by decision, not an oversight;
  it should not be added without that material found and documented.

**The strict-exclusion control.**

- Refitting on the strict subset — primary trials that additionally have a
  complete 15-row token log and valid design alignment, **10,958 of 16,320
  trials** — leaves urgency's raw `delta_bic` advantage smaller by 61.5 BIC
  points (`ssmexclusionrobustness`, `ssmexclusionrobustnessstats`, term
  `strict_delta_bic_minus_primary_delta_bic`; t(31) = 7.18, p = 4.6e-8, 30/32
  subjects) and collapsing's by 39.7 (t(31) = 8.08, p = 4.0e-9, 31/32).
- **That shrinkage is the trial count, not the effect — and once it is removed
  the evidence per trial goes *up*.** `delta_bic` scales with n, so comparing
  it across trial sets of different size measures the size difference. The
  decomposition `delta_bic = -2 delta_ll + delta_k ln n`
  (`ssmexclusiondecomposition`, `analysis =
  ssm_exclusion_evidence_decomposition`, `condition = all`) holds evidence per
  trial at its primary value and re-evaluates at the strict trial count:

| Model | ΔBIC primary | ΔBIC strict | expected from n | excess | per trial primary | per trial strict |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `urgency` | −239.60 | −178.05 | −158.59 | **−19.46** | 0.2451 | **0.2730** |
| `collapsing` | −169.04 | −129.33 | −111.63 | **−17.70** | 0.1733 | **0.1987** |
| `additive_urgency` | −19.00 | −6.43 | −10.64 | +4.21 | 0.0258 | 0.0180 |

  Had evidence per trial been unchanged, urgency's advantage would have fallen
  by 79.4 points; it fell by 61.5. The excess is significant
  (t(31) = −4.96, p = 2.4e-5), and measured per trial with the in-sample
  overfitting bias `delta_k / 2n` subtracted, urgency's advantage **rises**
  from 0.2451 to 0.2730 nats (t(31) = 4.90, p = 2.9e-5, 26/32 subjects);
  collapsing's rises from 0.1733 to 0.1987 (p = 1.6e-8). The overfitting
  correction is 19x smaller than the effect and does not drive this.
  **The strict subset therefore carries the same conclusion on more evidence
  per trial, not less.** A raw `delta_bic` difference between two trial sets of
  unequal size must never be quoted as a change in the size of an effect.
- **It is a pure subset of the trials and a different mix of the design.** The
  strict population is a pure subset of the primary one, not a second
  independently assembled sample (`n_mask_symmetric_difference = 0`,
  `mask_equivalent = true`), so the two results are not independent
  replications and must never be quoted as though they were. But identity of
  *trials* is not identity of *composition*: the 14-row token logs are almost
  all Slow trials (`ssm_exclusion_condition_composition`):

| Condition | primary | strict | retained | share of primary | share of strict |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Fast | 8,980 | 8,613 | 95.9% | 55.0% | 78.6% |
| Slow | 7,340 | 2,345 | **31.9%** | 45.0% | 21.4% |
| Total | 16,320 | 10,958 | 67.1% | — | — |

  This holds in **every** subject (Fast 93.7–98.3% retained, Slow 24.1–40.5%),
  and it explains the pattern that pooling hides: within Fast, which barely
  loses trials, the raw shrinkage is 3.2 BIC points; within Slow it is 73.4.
  Per trial the Slow evidence rises most of all (0.2663 → 0.3232, p = .0081) —
  what one expects if the trials with anomalous logs are the badly timed ones.
  So this control doubles as a **Fast-weighted reanalysis**: it supports the
  conclusion under a substantially different condition mix, which is a stronger
  result than a completeness check. "Pure subset" must not be left to imply
  "same design".
- Boundary-hit rates rise to 91–100% on the smaller subset. **A near-universal
  boundary flag is expected here and is not by itself a fitting failure**: the
  non-decision parameter has nothing left to absorb once `dt_ms` is
  motor-corrected, so its residual sits below 10 ms in 32/32 `urgency` cells,
  32/32 `collapsing`, 31/32 `additive_urgency` and 30/32 `ddm`
  (`ssmcomparison`, `condition = all`; the exceptions are `ddm` cells at 0.605
  and 0.902 s). The flag must still be decomposed by parameter before it is
  interpreted — that is what makes it readable rather than alarming.

**The matched-sequence diagnostic.**

- A frozen stimulus-only matching rule (opposite early evidence sign at jump 3
  with identical lead, equal lead at jump 6, response only after jump 6,
  paired by RMSE) yields 310 Fast and 302 Slow pairs. **No behavioral effect is
  statistically established**: decision-time and accuracy contrasts run
  p = .49 and .67 (Fast) and p = .33 and .14 (Slow)
  (`ssmmatchedsequencestats`).
- **The direction of that null points away from the winning model, and must be
  stated.** This is the only prespecified, model-free test of what separates
  urgency gating from integration — whether old evidence still counts once the
  later evidence is matched — so reporting it as "no effect" without its sign
  lets a null read as support. On the against-minus-for **accuracy** contrast
  (`ssmmatchedsequencepredstats`, `ssmmatchedsequenceppc`):

| Condition | observed | `urgency` | `collapsing` | `ddm` | `additive` | obs − urgency | p |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fast (27 subj, 310 pairs) | −0.020 | **+0.031** | −0.031 | −0.055 | +0.006 | −0.050 | .238 |
| Slow (28 subj, 302 pairs) | −0.091 | **+0.023** | −0.074 | −0.101 | +0.008 | −0.114 | .070 |

  The observed contrast is negative in both conditions. `urgency` is the only
  model predicting a *significantly* positive contrast (p = .010 Fast,
  p = .042 Slow); `additive_urgency` is also positive but indistinguishable
  from zero (p = .79, p = .77), and its null is weak by construction. The
  observed point estimates therefore sit closest to `ddm` and `collapsing` —
  the models the likelihood comparison rejects.
- **Tested properly as a posterior-predictive check, no model is rejected.**
  Pairing observed and predicted contrasts within subject
  (`ssmmatchedsequenceppc`), every observed-minus-predicted residual is
  consistent with zero. `urgency` comes closest to rejection in both
  conditions and is the only model whose predicted sign is wrong in both. The
  honest summary is that this diagnostic cannot separate the models at this
  n — not that it is silent, and not that it supports the winner.
- Some fitted-model contrasts on the same pairs are significant. That does not
  turn a null behavioral diagnostic into mechanism-identifying evidence, and
  the diagnostic is reported as complementary rather than as support.

**Interpretation.**

- The mechanistic comparison converges with the two model-free results that
  point the same way: the proportional Fast/Slow stretch (F06), which bounded
  integration with a moved threshold does not produce, and the
  confidence-at-commitment pattern (F08). Three independent routes, one
  conclusion — **the commitment policy on this task is time-dependent, and the
  multiplicative form fits best.**
- What this does *not* license is a claim that subjects behave optimally: the
  missing reward specification makes that unaskable. How fast a given
  subject's urgency grows *is* readable — `urgency_scale` recovers at
  r = 0.994 in the fitted region (F15) — except where an estimate is censored
  at its bound.

**Figure F13 `ssmcomparison-model`** — six panels, mechanism above
adjudication.

| | Panel | Shows |
| :--- | :--- | :--- |
| **A** | Criterion vs. time | What each model claims the policy is: flat integrator bound, multiplicatively falling urgency criterion, hyperbolically collapsing bound |
| **B** | Decision variable by trial class, with the urgency criterion dashed | Where commitment happens, and why harder trials commit later |
| **C** | Observed − predicted decision time at the 10th/50th/90th percentile; solid correct, dashed error | Absolute fit. `ddm` misses the 90th correct percentile by −375 ms and fast errors by +373 ms; urgency is within 54 ms and 90 ms |
| **D** | Per-subject in-sample ΔBIC vs. the fixed bound | The in-sample ranking |
| **E** | The same comparison, held out | The out-of-sample ranking |
| **F** | The three candidate-pair held-out contrasts | Which candidate beats which |

Reading rules, each of which is a decision rather than a default:

- **`ddm` is the zero line in D and E**, not a fourth column of zeros — every
  delta is measured against it, and the axis label names it.
- **`additive_urgency` carries no significance marker in D.** Its null is weak
  by construction (see the additive-urgency bullet above) and no equivalence
  bound is registered for it, so a marker would assert more than the analysis
  supports.
- **Colour identifies a model in A, C, D and E only.** A paired difference
  belongs to neither model, so F is neutral and row-labelled. B colours by
  trial class, a disjoint palette scope, and carries its own legend.
- **C is residuals, not an overlay of predicted densities.** Overlaid, the four
  predictions are near-superimposed and the panel reads as "all four fit about
  equally well", which is false. Correct and error share the x positions so a
  model's two outcomes sit side by side: a model can track the correct
  distribution and still miss the errors badly, which is precisely the fixed
  bound's failure. The observation is C's zero line and has no legend entry.
- **A and B are trimmed** to the last time every subject's fit contributes;
  past it the group mean is taken over a shrinking, self-selected subset.

Statistics are read from `ssmcomparisonstats` (D),
`ssmdistributioncheckstats` (C), `ssmheldoutstats` (E) and
`ssmheldoutpairwise` (F); intervals are the registered ones, never refitted
from the drawn points.


**Figure F14 `ssmcomparison-urgency`** — what does the speed instruction
change about urgency?

| | Panel | Shows |
| :--- | :--- | :--- |
| **A** | Drift scale, Fast vs. Slow | Δ = +0.0264, p < .001 — gain rises under time pressure |
| **B** | Criterion scale | Δ = −0.114 s, p = .014 — urgency rises **faster** under time pressure |
| **C** | Urgency onset | Null, and bounded: \|Δ\| < 0.114 s |
| **D** | Non-decision residual | Null, and bounded: \|Δ\| < 0.00226 s |
| **E** | The criterion-scale contrast: production fit, then six refits | The effect size, and that it survives every refit |

The argument is the *pattern* across A–D: the speed instruction acts on gain
and on how fast urgency grows, **not** on when urgency starts or on motor
delay. That claim rests on two nulls, so C and D carry the smallest effect
each excludes rather than a bare "n.s." — a bound is a result, a p is not.
Both bounds are read from `equivalence_bound` on the registered contrast; the
figure derives no inferential quantity of its own.

The panels carry no explanatory text, so what they encode is recorded here:

- **Lower criterion scale means faster-rising urgency.** It is threshold ÷
  urgency slope, so B's Fast mean sitting *below* Slow **is** the effect. This
  is the one genuine sign trap in Act 3.
- **Ringed points are censored, not estimated** — the optimizer left them
  within 1% of the parameter's upper bound — and pairs touching one are
  **dashed**. A pair censored at both ends would otherwise draw a flat line
  that reads as a measured "no change" the fit never claimed.
- **The censoring is asymmetric in the direction of the effect**: 3 cells in
  Fast against 8 in Slow on criterion scale. E answers that objection: the
  coloured row is the refit with no censored cell (`expanded_bounds`), whose
  contrast is the **largest** of the seven (see the censoring bullet under
  Result, above).
- **E's top row is the production fit, below the rule; the rest are checks on
  it.** Its baseline refit (−0.118) and the production fit (−0.114) are
  different runs agreeing, which is why both are drawn rather than one being
  quoted for the other.

**What this does license, and what it does not.** In the region these fits
occupy, criterion scale recovers at r = 0.994 and urgency onset at 0.986
(F15), so a dot in A–C *is* that subject's fitted value and the panels are
contrasts between measured quantities. Two limits remain: ringed points are
censored at the bound and are not estimates at all, and D's non-decision
residual does **not** recover for this model ("What is identified, and what is
not"), so no individual value there should be read.


### Post-error slowing is real, but the measure everyone quotes contains two effects

**Result.**

- Against each subject's own mean decision time, trials **after** an error run
  **+105.7 ms** slower, 95% CI [74.5, 136.9], t(31) = 6.90, p = 9.6e-8,
  dz = 1.22. That is post-error slowing.
- Trials **before** an error run **−38.3 ms** *faster*, 95% CI [−58.7, −18.0],
  t(31) = −3.84, p = 5.7e-4, dz = −0.68. That is pre-error speeding, and it is
  a separate effect.
- The conventional local-pair measure, `DT(error+1) − DT(error−1)`, is their
  difference: **+144.0 ms**, 95% CI [105.9, 182.1], t(31) = 7.71, p = 1.1e-8.
  **27% of it is pre-error speeding, not post-error slowing.**
- The classical contrast, post-error minus post-correct, is **+117.6 ms**,
  95% CI [80.9, 154.3], t(31) = 6.54, p = 2.7e-7. The two agree at r = 0.82
  but are not interchangeable: the local pair is positive in **32 of 32**
  subjects, the classical contrast in 29.
- Neither differs between Fast and Slow (post-error slowing Δ = +14.5 ms,
  p = .47, excluding effects larger than 48.6 ms).

Estimates rest on 47–109 complete error pairs per subject, median 81.5
(`posterror`, `posterrorstats`; F16).

**Interpretation.** The local-pair measure exists to remove slow drift in
time-on-task, which the classical contrast is exposed to. It does that, but it
buys the property at the cost of a second effect: it is only post-error slowing
if pre-error trials are typical, and here they are not. This also explains the
otherwise odd result that the "stricter" measure returns the *larger* number —
it is not stricter, it is broader, because the classical contrast never touches
pre-error trials. Neither measure is clean, which is why both are reported.
`robust` in the derivative names local temporal *matching*, not a robust
estimator; no trimming or median is involved.

### The previous outcome shifts which side is chosen, and choices mildly alternate

**Result.**

- Subjects repeat their previous side **0.498** of the time after a correct
  trial and **0.441** after an error: Δ = **+0.057**, t(31) = 5.10, p = 1.6e-5,
  dz = 0.90. Both sit near chance because the correct side is counterbalanced,
  so this is a lose-shift bias rather than a side preference.
- The lag-1 autocorrelation of chosen side is **−0.029**, 95% CI
  [−0.052, −0.007], t(31) = −2.65, p = .012, dz = −0.47 — small but reliable,
  and negative in 24 of 32 subjects (`choicehistory`, `choicehistorystats`;
  F17).

**Interpretation.** Both effects are alternation, not perseveration: an error
makes a switch more likely, and side alternates slightly overall. The
autocorrelation is a group tendency rather than a strategy every subject shows,
and at −0.029 it is far too small to confound the condition contrasts, which is
the reason it is measured.

### Individual differences are one speed axis, and the Fast/Slow adjustment is not on it

**Result.**

- The seven per-subject measures spread widely across the cohort
  (`individualprofile`, n = 32; min–median–max):

| Measure | Min | Median | Max |
| :--- | ---: | ---: | ---: |
| Mean decision time (ms) | 582 | 1199 | 2024 |
| Accuracy (%) | 78.5 | 83.5 | 90.1 |
| Evidence sensitivity (accuracy log odds per unit) | 4.74 | 9.21 | 17.97 |
| Urgency slope (per second) | −0.050 | +0.011 | +0.133 |
| Criterion slope (SumLogLR per second) | −0.550 | −0.142 | +0.183 |
| SAT adjustment, Slow − Fast (ms) | −137 | +115 | +454 |
| Lapse rate | 0 | 0 | 0.0056 |

  The two slope measures disagree on sign across subjects — the urgency slope
  is positive in 21 of 32 and the criterion slope in 7 of 32 — because they are
  fitted on different evidence scales, not because subjects disagree about the
  direction of the effect. The Fast/Slow adjustment is positive in 28 of 32.
- **Five of the 21 pairwise correlations reach p < .05**, uncorrected
  (`individualcorrelations`, Pearson, n = 32, df = 30; Spearman ρ alongside):

| Pair | r | p | ρ |
| :--- | ---: | ---: | ---: |
| urgency slope × criterion slope | **+0.887** | 1.4e-11 | +0.860 |
| mean DT × evidence sensitivity | **−0.814** | 1.5e-8 | −0.842 |
| mean DT × urgency slope | −0.527 | .0019 | −0.503 |
| mean DT × accuracy | +0.460 | .0080 | +0.435 |
| mean DT × criterion slope | −0.372 | .036 | −0.357 |

- **Mean decision time is in four of the five.** Slower subjects are more
  accurate, have flatter urgency and criterion slopes, and take less accuracy
  per unit of early evidence.
- **The two slope diagnostics converge, at r = +0.887.** The first-order
  chosen-target criterion slope and the exact-posterior urgency slope are
  fitted on explicitly different evidence scales (F09), and across subjects
  they order the cohort almost identically. Rank agreement is as strong
  (ρ = +0.860), so the correlation is not carried by the extremes.
- **The Fast/Slow adjustment correlates with nothing.** Its largest
  correlation with any other measure is **+0.282** (lapse rate, p = .12), and
  with mean decision time it is −0.200, p = .27. Across the six pairs it
  enters, no p falls below **.12**.
- **Rank and product-moment agree throughout.** Spearman reproduces every one
  of the five significant Pearson correlations and adds none; the largest
  disagreement in either direction is the urgency-slope/evidence-sensitivity
  pair (r = +0.262, p = .15 against ρ = +0.324, p = .070), which neither
  method calls significant.
- **No multiplicity correction is applied**, here or anywhere in this document
  (`docs/data_contract.md`). Twenty-one tests at α = .05 expect about one false
  positive by chance, and the five are not equally secure: three sit at 1.4e-11,
  1.5e-8 and .0019, while +0.460 (p = .0080) and −0.372 (p = .036) are the two
  that would not survive any correction. The strongest three are the ones the
  argument leans on.
- The strongest correlation that does **not** reach significance is accuracy ×
  criterion slope (−0.291, p = .106), followed by accuracy × urgency slope
  (−0.282, p = .117). Accuracy relates to the slope measures in the direction
  the speed axis predicts, but not reliably at this n.

**Interpretation.**

- **The dominant structure is a single speed axis, and it is mean decision
  time.** Four of the five reliable correlations involve it, and the fifth is
  two measures of the same thing agreeing with each other. This cohort does not
  show independent "accuracy" and "speed" traits that trade off; it shows one
  ordering of subjects that everything else projects onto.
- **The r = +0.887 convergence is a validity check, not a finding about
  subjects.** It says the first-order and exact-posterior criterion diagnostics
  identify the same subjects as steep or flat despite being different scales,
  which is what licenses Acts 2 and 3 quoting either. It is not evidence that
  the two are interchangeable: F09 shows their *group* slopes differ, and
  agreeing on the ranking of subjects is a weaker claim than agreeing on a
  value.
- **The Fast/Slow adjustment being unrelated to everything else is the useful
  null here.** The manipulation Acts 2–3 lean on is an independent trait in
  this cohort: how much a subject speeds up under time pressure is not
  predictable from how fast, how accurate, or how steep they are. That is what
  makes the Fast/Slow contrast a manipulation rather than another expression of
  the speed axis — and it is why F06's condition effect cannot be re-described
  as a between-subject difference.
- **This is a correlational description of 32 subjects and should be read as
  one.** At this n the individual coefficients are imprecisely estimated, the
  tests are uncorrected, and four of five share a variable, so their apparent
  agreement is partly arithmetic. The claim the document rests on is the
  qualitative one — a single axis, with the SAT adjustment off it — not any
  particular r.

Figure: `individualcorrelations-structure` (F18) — A is the full 7 × 7
correlation matrix, lower triangle without the diagonal, colour for r and the
significance marker in each cell; B–F magnify the five significant pairs,
largest |r| first, with an OLS line and its 95% band (display only; r and its
marker are read from the derivative). B–F are **selected on significance**, so
they are a gallery of significant results by construction and cannot say how
many pairs reached the bar — A is the only panel that can, and it says five of
twenty-one. The figure carries no note of its own, so its caption must state:
Pearson r, n = 32, uncorrected across 21 pairs; in A colour is r and a blank
cell is tested and not significant.

### Cross-species translation: where these numbers meet the monkey literature

**Not a finding — a cross-reference.** `speciescomparison` collects the ten
measures the monkey tokens-task literature states its results in, computed on
this cohort (n = 32), so the two can be read against each other without
re-analysing either. Every value in it is already reported in the act that
produced it, with the context that makes it interpretable. What this table adds
is only the last column: which published analysis each one lines up against.

| Measure (`speciescomparison`) | Reported in | Comparable published measure |
| :--- | :--- | :--- |
| `decision_time_{easy,ambiguous,misleading}_ms` | "Decision time by trial class" (F07) | DT by class (Cisek et al. 2009) |
| `success_probability_at_decision_{easy,ambiguous,misleading}` | "Success probability at decision" (F08) | SP at decision by class (Thura et al. 2012) |
| `criterion_slope_sum_log_lr_per_second` | "The declining evidence criterion is an artifact of the evidence scale" (F09) | evidence at commitment (Cisek et al. 2009) |
| `urgency_minus_integrator_bic` | "Urgency gating beats bounded integration" (F13) | urgency gating preferred over integration (Cisek 2009; Thura 2012) |
| `urgency_scale_criterion_seconds` | same, Act 3 (F14) | urgency signal fit (Thura & Cisek 2014; Carland et al. 2019) |
| `urgency_scale_fast_minus_slow` | same, Act 3 (F14) | urgency differs Fast/Slow (Cisek et al. 2009) |

Three things this table is not, and one it cannot be:

- **It is not a comparison.** Published monkey values are absent from the
  derivative and from this document by design. Transcribing them would mean
  lifting numbers from figures fitted under different task parameters, session
  structures and subject counts, and any side-by-side display invites reading
  them as a common analysis. The comparison is textual and belongs to the
  reader.
- **Its `t` and `p` columns are not results.** The derivative tests every row
  against zero. That is meaningful for the criterion slope, the BIC margin and
  the Fast − Slow contrast, and meaningless for the other seven: a decision
  time cannot be zero, and a 2AFC success probability is at chance at 0.5, not
  0. Read those rows in the acts that report them, where they are tested
  against the null that applies.
- **It has no figure.** Ten numbers reported elsewhere, drawn as a chart,
  add nothing the acts do not already carry — and a forest plot of
  incommensurate measures sharing a zero reference that is wrong for seven of
  them misleads actively.
- **Two comparisons in that literature have no row at all.** Movement duration
  and the decision/movement covariation both need a movement measure, and the
  logs do not record one: `tEnterCenter` and `tExitCenter` are written from the
  same LabVIEW event (Known Issues, Stage 1). They are absent, not
  approximated.

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
  ηp² = .04, `p=.26`; F05). **The speed manipulation changed decision time
  without changing accuracy.**
- **The motor baseline is estimated per response side.** The preprint
  subtracts one pooled RT latency per subject. The RT runs contain no
  deliberation, yet they show a reliable hand difference — right 442.1 ms vs.
  left 456.0 ms, Δ = 13.9 ms, t(31) = 3.29, p = .003 — so a pooled baseline
  leaves a purely motor asymmetry inside `dt_ms`. Each trial is now corrected
  with the baseline for the hand that answered it (a hand with no usable RT
  response falls back to the pooled value). Effect: the left-minus-right
  asymmetry in decision time drops from +22.8 ms (p = .023) to +9.8 ms
  (p = .354), while **every conclusion elsewhere is unchanged** (Slow − Fast
  +127.4 → +127.1 ms with identical t and p). The commitment time that
  `logged_spd` is read at shifts by 10.4 ms on average, which is why SPD
  had to be re-derived rather than assumed safe.
- **Trial classification uses the correct-target design frame**, not the
  recorded chosen-target `nProb`. This is the deviation that reverses the
  ambiguous-vs-misleading result; mechanism and evidence in
  "Trial-classification reference frame," above.

## Act 3 method and provenance: the Thura et al. (2012) mechanistic test

The result and its interpretation are above, under "Urgency gating beats
bounded integration, and beats a collapsing bound." What follows is method and
provenance, not result.

### Model specification

Primary source: Thura, Beauregard-Racine, Fradet, and Cisek (2012), "Decision
making by urgency gating: theory and experimental support," *Journal of
Neurophysiology*, 108, 2912-2930, doi:10.1152/jn.01071.2011. The paper
proposes a mechanistic account (short low-pass evidence filter x growing
urgency signal vs. a fixed threshold), not a replacement for the descriptive
criterion plot; our tokens arrive every 200 ms and remain visible, so the
filter time constant and urgency form were estimated/tested over a grid rather
than copied from the paper's 100-ms random-dot values.

Four models are implemented in
`meg_tokens/behavior/analyses/sequential_sampling.py`, selected with
`behavior ssm-fit`, fit to identical trials, response definition, motor-time
correction, likelihood, and validation folds:

1. `ddm`: fixed-bound, un-leaky evidence integration baseline.
2. `urgency`: the primary 2012 reproduction (leaky-filtered evidence,
   `FILTER_TAU_S = 0.2`, multiplied by a linear urgency signal, fixed
   threshold; equivalent criterion `T / (b + m t)`).
3. `collapsing`: un-leaky integrator with a hyperbolically collapsing bound,
   included because it can mimic the same declining criterion.
4. `additive_urgency`: a label-symmetric additive-drive sensitivity variant,
   explicitly not claimed as the 2012 mechanism. A common additive drive to two
   race channels cancels in a signed difference, so the scalar representation
   cannot identify one; the variant instead adds a time-growing drive in the
   *current evidence-preference* direction, which is odd under target-label
   reversal and never uses the post-hoc correct target. It is not treated as
   evidence for a unique additive mechanism, and the paper itself notes that
   current behavioral paradigms do not conclusively distinguish additive and
   multiplicative urgency.

Shared conventions: 200-ms token interval; PyDDM absorbing finite-grid first
passage (`dt=0.01`, `dx=0.01`, no separate overshoot correction); 2% uniform
contaminant/lapse mixture; hand-specific motor-corrected nondecision time per
subject x condition x model; the canonical `primary_analysis_eligible` trial
set; maximum-likelihood fitting with three deterministic starts for production
full-data fits, finite-difference Hessian SEs on final fits.

The comparison is not an optimality test unless reward values,
movement/nondecision timing, and intertrial timing are verified from the
acquisition/task specification. Those inputs are not part of the current
trial-feature contract, so no reward-rate claim is made by default.

### Scope and outputs

**Scope.** Production fit (32 subjects x 3 conditions x 4 models, three starts,
384/384 converged), held-out three-fold evaluation (1,152 cells, all
converged), parameter and model recovery (12 repetitions), six-configuration
robustness (2,304 rows), strict-exclusion sensitivity (10,961 of 16,324
trials), and the frozen matched-sequence diagnostic. Workflow code lives in
`meg_tokens/workflows/sequential_sampling.py`. Each stage's sidecar records
its `git_commit` and `source_tree_sha256`.

**Outputs.** Derivatives at
`BIDS/derivatives/sub-group/beh/` (`desc-ssm*`, 230 files). Figures F13-F15:
`sub-group_task-tokens_desc-ssmcomparison-{model,urgency}_behavior.*` and
`desc-ssmparameterrecovery-identifiability_behavior.*` in `.../sub-group/fig/`.

## References

- Thiery T., Rainville P., Cisek P., Jerbi K. (2022). *Distinct trajectories in
  low-dimensional neural oscillation state space track dynamic decision-making
  in humans.* bioRxiv
  [10.1101/2022.06.14.494674](https://doi.org/10.1101/2022.06.14.494674).
- Cisek P., Puskas G.A., El-Murr S. (2009). *Decisions in changing conditions:
  the urgency-gating model.* J Neurosci 29(37):11560–11571.
  [10.1523/JNEUROSCI.1844-09.2009](https://doi.org/10.1523/JNEUROSCI.1844-09.2009).
- Thura D., Beauregard-Racine J., Fradet C.-W., Cisek P. (2012). *Decision
  making by urgency gating: theory and experimental support.* J Neurophysiol
  108:2912–2930.
- Thura D., Cisek P. (2014). *Deliberation and commitment in the premotor and
  primary motor cortex during dynamic decision making.* Neuron 81:1401–1416.
- Carland M.A., Thura D., Cisek P. (2019). *The urge to decide and act:
  implications for brain function and dysfunction.* The Neuroscientist.
- PyDDM — generalized drift-diffusion modelling:
  [pyddm.readthedocs.io](https://pyddm.readthedocs.io/).
