# Behavioral Analysis Roadmap

Deferred analyses for the Tokens-task dataset. This document is intentionally
separate from the active readiness plan in
`docs/behavior_metrics_readiness.md`.

T0-1 through T0-5 are complete. Subject exclusions remain deferred until MEG
quality review. The analyses below are not current blockers unless promoted
into the active plan.

## Tier A — direct, low-cost extensions

1. **SPD distributions by class.** Add cumulative distributions for the existing
   logged all-trial and 15-row sensitivity summaries.
2. **DT distributions.** Add subject-level quantiles, skewness, and optionally
   ex-Gaussian fits; means alone hide right-tail changes.
3. **Condition × class breakdown.** Report DT and accuracy for all six cells and
   test the interaction.
4. **Choice-side bias.** Report left/right choice proportions and DT asymmetry;
   use this to check balance in MEG’s choice cells.
5. **Time-on-task effects.** Test block index and within-block position for
   fatigue, learning, or condition-order effects.
6. **Lapses and extreme DT review.** Summarize no-response outcomes (`7006`/`7011`)
   and inspect extreme positive DTs; negative DT counts are already emitted.

## Tier B — exploratory behavioral analyses

1. **SumLogLR.** Compute cumulative log-likelihood evidence from token directions
   as a continuous complement to class and SPD.
2. **Accuracy-criterion decline.** Relate evidence at decision to the number of
    tokens observed; a declining criterion is the behavioral signature of
    urgency gating.
3. **Psychophysical reverse correlation.** Regress choice on token direction at
    each jump to estimate temporal weighting, and compare Fast/Slow kernels.
4. **Conditional accuracy functions.** Plot accuracy across DT quantiles by
    condition to distinguish threshold and drift changes.
5. **Continuous evidence effects.** Regress DT and accuracy on early SP,
    SumLogLR, or another continuous evidence measure instead of discarding
    unclassified trials.
6. **Robust post-error slowing.** Compare each post-error trial with the trial
    immediately preceding its error, and split by condition to avoid slow-session
    confounds.
7. **Choice-history effects.** Test win-stay/lose-shift behavior, side
    autocorrelation, and effects of previous class/outcome on current DT.
8. **Response vigor.** Separate movement time (`tTrialEnd - tEnterTarget`) from
    deliberation and test decision-speed/movement-speed covariation.
9. **Individual differences.** Relate subject-level Fast/Slow SAT adjustment,
    urgency slope, evidence sensitivity, accuracy, and MEG measures.

## Tier C — model-based and MEG-linked analyses

1. **Urgency-gating versus drift-diffusion models.** Fit both accounts of
    non-stationary evidence and compare them with WAIC/LOO.
2. **Hierarchical sequential-sampling models.** Prefer hierarchical estimation
    for the approximately 350 trials per subject; HSSM is the primary candidate,
    with `pyddm` as a lighter custom-bound alternative.
3. **Explicit urgency extraction.** Estimate per-subject urgency slope and
    intercept by condition from evidence-at-decision versus decision time.
4. **Mixed-effects inference.** Use models such as
    `DT ~ class * condition + (class * condition | subject)` alongside paired
    t-tests retained for direct preprint comparison.
5. **Model-based MEG regressors.** Join trial-level SP(t), SumLogLR(t), and
    urgency(t) to source-space features for single-trial neural tests.
6. **Monkey–human comparison.** Report DT by class, SPD by class, criterion
    decline, and decision/movement-speed covariation using comparable statistics
    from the Cisek-lab literature.

## Suggested order after the active plan

1. Tier A items 1–2 for distributional descriptions.
2. Tier A items 3–6 for behavioral QC and descriptive completeness.
3. Tier B items 2, 3, and 8 as the highest-value mechanistic tests.
4. Tier C once trial-level MEG regressors and group infrastructure are stable.

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
