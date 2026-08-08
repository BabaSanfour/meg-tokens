# Behavioral Analysis Roadmap

## Status

Tiers A and B are implemented, together with the parts of Tier C that do not
require a hierarchical model fit or MEG source features. Run them with:

```bash
meg-tokens --config tokens.toml behavior analyze     # trial features
meg-tokens --config tokens.toml behavior characterization    # this roadmap
```

Each item below carries its implementing module and the derivative it writes
(`sub-group_task-tokens_desc-<name>_beh.tsv`, plus `<name>stats` for the group
test). Measured results on the 32-subject dataset are in
`docs/behavior_roadmap_results.md`.

Group inference throughout is two-stage: one fit inside each subject, then a
test across subjects on the fitted values. For the balanced within-subject
designs used here that is the random-effects equivalent of a mixed model with
by-subject random slopes, and it keeps the package free of a modelling
dependency. Tier C1-C2 are the exception and are deferred to HSSM.

## Tier A — direct, low-cost extensions

1. **SPD distributions by class.** ✅ `behavior/analyses/distributions.py` →
   `spdcumulative`. Cumulative distributions for both the all-trial and 15-row
   sensitivity views, reported pooled and as the mean of per-subject curves.
2. **DT distributions.** ✅ `behavior/analyses/distributions.py` → `dtdistribution`.
   Subject-level quantiles, skewness, kurtosis, and ex-Gaussian
   (`mu`, `sigma`, `tau`) fits per condition, class, and condition × class
   cell; class and Fast/Slow contrasts on each statistic.
3. **Condition × class breakdown.** ✅ `behavior/analyses/design_effects.py` →
   `conditionclass`. DT and accuracy for all six cells with a fully
   within-subject 2 × 3 ANOVA (main effects and interaction, each against its
   own `effect × subject` error term).
4. **Choice-side bias.** ✅ `behavior/analyses/design_effects.py` → `choiceside`.
   Left/right choice proportions, DT asymmetry, and accuracy asymmetry,
   overall and per condition.
5. **Time-on-task effects.** ✅ `behavior/analyses/design_effects.py` → `timeontask`,
   `conditionorder`. Block order comes from the LabVIEW session clock
   (`nInitialTime`), the only field that recovers it — Fast and Slow blocks
   interleave, and `nTrialIndex` restarts at 1 in each run.
6. **Lapses and extreme DT review.** ✅ `behavior/analyses/design_effects.py` →
   `lapses`, `extremedt`, `extremedttrials`. No-response outcomes are
   summarized by LabVIEW code; extreme DTs are flagged at a robust
   median-absolute-deviation cutoff and listed by trial, never removed.

## Tier B — exploratory behavioral analyses

1. **SumLogLR.** ✅ `behavior/math/evidence.py` and `behavior/features.py`,
   written into the trial-feature table. Success probability is the exact posterior from Equation 1, so with
   equal priors the cumulative log-likelihood ratio is its log posterior odds.
   Certainty (SP exactly 0 or 1) is reported at ±log 255 — the most extreme
   non-degenerate state the 15-token design can reach — and flagged, rather
   than propagated as an infinity that would drop the most decisive trials
   from every regression. `token_lead_at_decision` is the always-finite
   sufficient statistic.
2. **Accuracy-criterion decline.** ✅ `behavior/analyses/evidence.py` →
   `criteriondecline`. Evidence at decision against the number of tokens
   observed, fitted on both the probability and the log-odds scale.
3. **Psychophysical reverse correlation.** ✅ `behavior/analyses/evidence.py` →
   `reversecorrelation`. Per-subject logistic weights for each of the first
   eight jumps, with tokens that fell after commitment coded as unseen, plus a
   model-free kernel and a Fast/Slow comparison per jump.
4. **Conditional accuracy functions.** ✅ `behavior/analyses/evidence.py` →
   `conditionalaccuracy`. Accuracy across within-subject DT quantiles per
   condition, with a test of the slope across bins.
5. **Continuous evidence effects.** ✅ `behavior/analyses/evidence.py` →
   `continuousevidence`. DT on evidence strength and accuracy on signed
   evidence, over every task trial including the unclassified random ones.
6. **Robust post-error slowing.** ✅ `behavior/analyses/sequential.py` → `posterror`.
   `DT(error + 1) − DT(error − 1)`, with the classical contrast reported
   beside it. Adjacency is broken by any gap in `run_trial_index`.
7. **Choice-history effects.** ✅ `behavior/analyses/sequential.py` → `choicehistory`.
   Win-stay/lose-shift, lag-1 side autocorrelation, and DT as a function of
   the previous trial's outcome and class.
8. **Individual differences.** ✅ `behavior/analyses/individual.py` →
   `individualprofile`, `individualcorrelations`. Subject-level SAT
   adjustment, urgency slope, criterion slope, evidence sensitivity, accuracy,
   and lapse rate, correlated pairwise. `--neural-metrics` joins any
   subject-level MEG table into the same matrix.

## Tier C — model-based and MEG-linked analyses

1. **Urgency-gating versus drift-diffusion models.** ⏳ Deferred. Needs a
   fitted sequential-sampling model and WAIC/LOO comparison; see item 2.
2. **Hierarchical sequential-sampling models.** ⏳ Deferred. HSSM remains the
   primary candidate for the roughly 350 trials per subject, with `pyddm` as a
   lighter custom-bound alternative. Neither is a dependency of this package.
3. **Explicit urgency extraction.** ✅ `behavior/analyses/evidence.py` → `urgency`.
   Per-subject intercept and slope of evidence at decision against decision
   time, by condition, on both evidence scales.
4. **Mixed-effects inference.** ✅ via the two-stage design described above:
   every roadmap regression is fitted within subject and tested across
   subjects, and the paired t-tests are retained for direct preprint
   comparison. `statsmodels` supplies the per-subject OLS and Logit fits; it is
   not currently used to replace the planned two-stage design with a
   single-model `DT ~ class * condition + (class * condition | subject)` fit.
5. **Model-based MEG regressors.** ◐ Behavioral half done. The trial-feature
   table carries SP at decision, `sum_log_lr_at_decision`,
   `token_lead_at_decision`, and the per-subject urgency parameters, all keyed
   by `subject`/`condition`/`run`/`run_trial_index`. The join to source-space
   features waits on those features existing.
6. **Monkey–human comparison.** ✅ `behavior/analyses/individual.py` →
   `speciescomparison`. DT by class, SP at decision by class, and criterion
   decline, each row naming the published measure it corresponds to. The
   movement-related comparisons in that literature are omitted for the reason
   given in B8. Values from the papers are not reproduced in code.

## Suggested order after the active plan

4. Tier C1–C2 and C5 once trial-level MEG regressors and group infrastructure
   are stable.

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
- Carland M.A., Thura D., Cisek P. (2019). *The urge to decide and act:
  implications for brain function and dysfunction.* The Neuroscientist.
- HSSM — Hierarchical Sequential Sampling Modeling:
  [lnccbrown.github.io/HSSM](https://lnccbrown.github.io/HSSM/).
