# Behavior Reporting - Session Handoff

This handoff preserves the scientific and implementation history needed to
continue safely, then gives the forward plan. Completed numerical results and
their interpretation remain canonical in `docs/behavior.md`; this file keeps
the decisions, failure modes, machine boundaries, and next-analysis protocol.

## Current state

- `docs/behavior.md` is the canonical behavioral results narrative.
- The reporting layer reads group derivatives from
  `BIDS/derivatives/sub-group/beh/` and writes figures to
  `BIDS/derivatives/sub-group/fig/`.
- Laptop data root:
  `/Users/hamzaabdelhedi/Projects/data/meg-tokens`.
- Cluster checkout: `~/meg-tokens`.
- Cluster data root: `/scratch/hamza97/meg-tokens`.
- The current reporting/behavior test set passes locally. Optional `pyddm`
  failures in the broader suite are unrelated until the sequential-sampling
  models are deliberately revisited.
- The completed criterion/reporting work is the commit immediately after
  `2dbd2a4`; confirm it with `git log -1 --oneline` at the start of the next
  session. The attached preprint PDF and `uv.lock` remain deliberately
  untracked and outside that commit.

## Development and machine history

- Session 1 built and styled F04-F06 and wrote their findings.
- Session 2 removed F07, corrected the science in F04-F06, and brought the
  decision-time computation to preprint parity. Two of the first three prose
  findings were contradicted by numbers already present in the same sections;
  remaining unaudited claims must therefore be treated as provisional.
- The laptop continuation audited F08-F13, corrected response-side decision
  time for hand-specific motor baselines, merged the former F12 quality census
  into F13 panel E, and closed the descriptive behavioral foundation.
- The two-session reporting refactor and Act 1 work were committed as
  `2dbd2a4` before the current uncommitted evidence/criterion work began.
- The original desktop data root was `/media/karim/Hamza/meg-tokens`. The real
  dataset did not travel with Git. Only the group behavioral derivatives and
  rendered figures were copied to the laptop; raw behavioral/MEG data remain
  on the cluster.
- The raw/per-subject sequential-sampling fit inputs remain cluster-only, but
  the pooled local derivatives are present: `ssmcomparison`,
  `ssmcomparisonstats`, `ssmtrialpredictions`, `ssmtimecourse`,
  `ssmpopulation`, and `ssmpopulationstats`. They are sufficient for a local
  first-pass figure. Any refit still requires a cluster `behavior ssm-fit`
  job because `pyddm` is not installed locally.

## Scientific act map

1. **Behavioral foundation.** Cohort/QC, the Fast/Slow manipulation,
   stimulus-frame difficulty, decision-time distributions, and confidence at
   commitment. F04-F06, F08-F11, and F13 have been audited.
2. **Dynamic commitment policy.** How changing evidence becomes a choice:
   criterion across time, conditional accuracy, continuous-evidence effects,
   and the temporal choice kernel. The exact-posterior/time plot must be
   audited separately as a possible sensitivity analysis.
3. **Mechanistic adjudication.** Formal comparison of urgency gating,
   bounded integration, collapsing/adaptive bounds, and plausible urgency
   variants. This is where the 2012 analysis below belongs.
4. **Sequential adaptation.** Post-error slowing and choice/history effects.
5. **Individual differences and translation.** Cross-subject structure,
   cross-species comparison, and the later behavioral-MEG join.

This is the scientific Results order, not the historical implementation-phase
order in `docs/behavior_reporting_plan.md`.

## Behavioral foundation already established

1. All 32 subjects contribute. Decision-time retention is 99.92%; lapses,
   anticipations, and robust-MAD extremes are rare. H20 is influential for
   tail-sensitive work but is not an exclusion.
2. The Fast/Slow contrast is not explained by response side, session drift,
   within-block slowing, or counterbalancing order. The between-order test is
   too weak to establish equivalence and must not be described as proof of no
   order effect.
3. Difficulty must be classified in the stimulus/correct-target frame. The
   former response-frame definition was circular and reversed the
   ambiguous-versus-misleading result.
4. Slow is approximately a proportional stretch of Fast, not a fixed added
   delay. Easy-versus-ambiguous changes distribution shape;
   easy-versus-misleading is close to a stretch; ambiguous and misleading
   converge in the tail.
5. Decision time and confidence dissociate: easy trials finish at higher SPD,
   while ambiguous and misleading finish at similar SPD despite different
   timing distributions.

F07's ex-Gaussian detour was removed. F12 was merged into F13, not merely
hidden. The registry contains 24 figures after those changes.

## Figure-specific decisions worth preserving

- **F04:** keep the condition quantile curves and raincloud; no pooled-trial
  KDE. The Fast/Slow finding is a proportional stretch. The panel ratio was
  intentionally asymmetric and the comparison uses a compact bracket.
- **F05:** use a raincloud plus quantile functions, without a duplicate KDE.
  Subject-line direction is evaluated per adjacent-category leg. The upper
  left was the only legend corner that cleared the real curves.
- **F06:** the `all_logged` and `validated_15row` views were numerically
  checked before collapsing to the higher-integrity view. The custom 5.7-inch
  width, x-range starting at 0.2, and explicit solid/dashed legend are
  deliberate.
- **F07:** removed because the ex-Gaussian decomposition was unstable and
  added no identifiable scientific conclusion.
- **F08-F11:** use within-subject uncertainty for repeated trajectories,
  compact statistics, and design-validity questions that the data can
  actually answer.
- **F13:** combines cohort composition and the quality census on a shared
  subject axis. Do not recreate the removed standalone lapse figure.

## Reusable audit lessons

For any two decision-time distributions, distinguish a fixed delay, a
proportional stretch, and a genuine shape change:

1. Compare quantile differences and ratios. A delay predicts stable
   differences; a stretch predicts stable ratios.
2. Use the per-subject KS ladder on raw, median-subtracted, and median-divided
   distributions. Subtraction should remove a delay; division should remove a
   stretch.
3. Check SD and CV. A delay preserves SD and changes CV; a stretch scales SD
   and approximately preserves CV.
4. For a null result, report an equivalence bound or interval. `p > .05` is
   never evidence that two quantities are identical.

Session 2 implemented these as temporary scripts (`shift_vs_scale.py`,
`norm.py`, `rescale.py`, `cv_vs_skew.py`, `cv_persubject.py`, `ks_check.py`,
and `audit_f05_f06.py`). They may no longer exist. If reused, promote the
analysis into a persisted derivative instead of creating another throwaway
script.

## Reporting conventions

- Numeric tick labels stay visible; reduce their number rather than hiding
  them. Tick marks themselves are removed globally.
- `style.SAVEFIG_DPI` must be passed explicitly to saves made after the style
  context exits. This is fixed centrally in `behavior_summary.py`.
- Use `Delta = value [unit] marker` for compact comparison brackets. Full
  `t/df/p/dz` text belongs in the derivative and sidecar when it would distort
  a panel.
- Long annotations must be designed for the actual panel width. They can make
  `constrained_layout` collapse an axis even when the text is technically
  drawable.
- Put exclusions and methodological caveats in the sidecar/registry rather
  than overcrowding the title.
- Color is never the sole carrier of identity. Use labels, line style, or
  marker form as well.
- SEM is used for descriptive subject-balanced bin means. Repeated-measures
  trajectories and subject-coefficient summaries use appropriate
  within-subject or group 95% CIs.
- A passing synthetic report test is necessary but insufficient. Regenerate
  every changed figure with the real derivatives, render its PDF, and inspect
  the resulting image.
- Inferential statistics must be persisted before plotting. The report layer
  may compute display summaries only.
- `docs/behavior.md` is the only home for completed numerical Result and
  Interpretation prose. Planning and handoff files retain implementation and
  audit instructions without duplicating the result narrative.

## Next analysis 1: exact posterior evidence against decision time

### Ready at session start

No data transfer or cluster job is needed to begin this audit. The laptop
already has:

- `trialfeatures` (19,090 rows, including the eligibility and alignment flags);
- `urgency` (192 subject-fit rows = 32 subjects x 3 conditions x 2 response
  scales);
- `urgencystats` (probability and log-odds inference rows); and
- the current 2 x 2 report builder under the `urgency-decisiontime` key.

`logged_spd` is documented in `behavior/features.py` as the acquisition-file
probability referenced to the chosen target. The fit already uses
`task_trials(features)`, i.e. `primary_analysis_eligible`; complete token-log
alignment is not currently required because the probability is logged rather
than reconstructed.

The current figure is not ready for interpretation. Its observed layer reads
all trial-feature rows, pools trials rather than first averaging within
subject, and chooses its range from the unfiltered table. Therefore the first
task is to make the plotted data use exactly the same eligible trials as the
fits and to compute subject-balanced display summaries.

### Scientific question

Determine whether recorded success probability at commitment changes with
decision time, and whether that pattern contains information beyond the
first-order chosen-target evidence result already documented in
`docs/behavior.md`.

This is a scale audit, not an automatic second urgency finding. Exact
posterior success probability is horizon-dependent: the same token lead maps
to a different posterior as the number of remaining tokens shrinks. Later
decisions can therefore overshoot in larger posterior steps even if the
underlying policy is unchanged.

### Required audit

1. Inspect the persisted `urgency` and `urgencystats` derivatives and trace
   every response, predictor, eligibility flag, and unit back to
   `trialfeatures`.
2. Confirm that `logged_spd` is the success probability for the selected
   target on both correct and error trials. Do not silently convert it to the
   correct-target frame.
3. Use continuous `dt_ms` for fitting. Any bins are display-only.
4. State explicitly whether anomalous 14-row logs are usable. Directly logged
   SPD may not require token-to-commitment reconstruction; do not inherit a
   complete-log exclusion without demonstrating that the analysis needs it.
5. Fit each subject separately, then perform group inference on subject
   coefficients. Do not pool trials across subjects for the inferential fit.
6. Audit both probability and log-odds scales. Report which features are
   invariant to the transformation and which are not.
7. Quantify the horizon/discretization contribution. At minimum, simulate or
   enumerate the posterior step size at each token index and compare the
   observed time slope with the slope expected from step-size overshoot alone.
8. Repeat the prespecified sensitivities used elsewhere only when they answer
   a real ambiguity: token-0 inclusion, complete-log eligibility, and a 3-s
   display or fit horizon. Keep the no-cutoff analysis primary unless the
   paper or task design supplies a principled cutoff.
9. Compare subject coefficients with the first-order criterion coefficients,
   but treat correlation as convergence between distinct measures, not proof
   that they are interchangeable.
10. Decide only after this audit whether the result belongs in the main
    narrative, a sensitivity figure, or no figure. A positive posterior slope
    cannot by itself be called a rising criterion.

### Ordered work for the next session

1. Run the existing figure once as a diagnostic, render it, and record every
   mismatch between the observed layer and `task_trials` eligibility.
2. Add a reusable eligible observed-data table: subject, condition,
   continuous decision time, decision token index, `logged_spd`, and
   `logged_spd_log_odds`.
3. Rebuild the 200-ms or prespecified display bins by averaging within subject
   first, then calculate the group mean and SEM. Bins remain display-only.
4. Enumerate the exact posterior grid by token index and remaining horizon.
   Quantify how much positive time slope a fixed underlying criterion acquires
   from discrete posterior overshoot alone.
5. Run the probability/log-odds, token-0, complete-log, and 3-s sensitivities.
   Persist every inferential comparison rather than calculating it in the
   report builder.
6. Only then redesign and style the figure. Decide whether both response
   scales are informative or whether one belongs in a sensitivity panel.
7. Write a plain-language result in `docs/behavior.md` only after deciding
   whether the corrected effect survives the discretization audit.

### Figure requirements

- Show the observed subject-balanced data and subject-level fits.
- Put the overall slope result in the data/fits panel.
- Use a paired Fast/Slow coefficient panel only if the condition contrast is
  scientifically interpretable after the horizon audit.
- Use SEM for descriptive subject-balanced bins and 95% CI for subject-level
  coefficient summaries; label both explicitly.
- Apply the shared palette, legend, bracket, and outlier conventions from the
  already reviewed figures.
- Render the PDF to PNG and inspect it before sign-off.

## Next analysis 2: Thura et al. (2012) mechanistic test

Primary source: Thura, Beauregard-Racine, Fradet, and Cisek (2012),
"Decision making by urgency gating: theory and experimental support,"
*Journal of Neurophysiology*, 108, 2912-2930,
doi:10.1152/jn.01071.2011.

### Ready at session start

The local derivatives already contain a complete first-pass two-model run:

- `ssmcomparison`: 192/192 converged cells (32 subjects x all/Fast/Slow x
  bounded integrator/urgency);
- `ssmcomparisonstats`: per-subject AIC/BIC comparisons and urgency parameter
  contrasts;
- `ssmtrialpredictions`: 65,276 trial-model prediction rows; and
- `ssmtimecourse`: 208,960 rows containing criteria, predicted/observed
  decision-time densities, and noise-free decision-variable trajectories.

The present code already drives both models with each trial's token-by-token
evidence path, fixes the token-task filter time constant at 200 ms, and compares
the Thura et al. bounded integrator with multiplicative urgency gating. This is
enough to prototype a figure locally without fitting anything.

Do not yet treat the derivatives as definitive. Their sidecars identify the
cluster inputs but do not record a Git commit or the complete fit settings.
Moreover, 14/96 urgency-scale estimates are within 1% of the upper parameter
bound, 5/96 urgency-onset estimates approach their upper bound, and standard
errors are missing for 38/96 urgency-scale and 41/96 urgency-onset cells. The
first session task is therefore provenance and identifiability audit, not
immediate publication interpretation.

### What the 2012 paper adds

The paper is not a replacement formula for the descriptive criterion plot.
It proposes a mechanistic account: sensory evidence is represented through a
short low-pass/novelty-sensitive process, multiplied by a growing urgency
signal, and compared with a fixed neural threshold. Its decisive comparison
uses stimuli whose evidence changes within a trial, because constant evidence
makes urgency gating and bounded integration difficult to distinguish.

The paper's random-dot implementation used a 100-ms filter and a linear
urgency example. Those values are not constants of nature. Our tokens arrive
every 200 ms and remain visible, so the filter time constant and urgency form
must be estimated or tested over a prespecified grid rather than copied.

### Models to compare

Fit all models to the same eligible trials, response definition, motor-time
correction, likelihood, and validation folds:

1. **Bounded integration:** accumulate all signed evidence samples until a
   fixed bound is crossed.
2. **Urgency gating:** apply a short low-pass filter to the current/novel
   evidence signal, multiply by an evidence-independent urgency function
   (start with `u(t) = b + mt`), and cross a fixed threshold.
3. **Collapsing/adaptive bound:** accumulate evidence while allowing the
   decision bound to fall with time. This is required because it can mimic the
   same declining behavioral criterion.
4. **Additive urgency sensitivity:** include or formally discuss an additive
   urgency variant. The 2012 paper itself did not conclusively identify
   multiplicative over additive urgency.

Use held-out predictive likelihood as the primary comparison when feasible;
retain BIC as a secondary summary for compatibility with existing
derivatives. Perform simulation-based model recovery before interpreting a
winner. Report parameter uncertainty and per-subject model preferences, not
only a group mean.

### Ordered work for the next session

**Stage A - audit and prototype from existing derivatives**

1. Confirm the cluster commit and constants that generated the local SSM
   tables (`FILTER_TAU_S`, parameter bounds, solver step, contaminant, seed,
   eligibility, and motor-time definition). If provenance cannot be recovered,
   regard the tables as a visual prototype only.
2. Audit boundary hits, missing Hessian standard errors, subject-level model
   wins, and whether the large BIC advantage is driven by a few cells.
3. Regenerate the existing model-comparison and model-internals figures
   locally. Use them to choose the first 2012 composition, not as the final
   result.
4. First figure draft: (A) fixed integrator criterion versus urgency-equivalent
   falling criterion, (B) observed and predicted correct/error decision-time
   densities, (C) model decision-variable trajectories by trial class, and
   (D) per-subject predictive or information-criterion difference. Every
   repeated field in `ssmtimecourse` must be de-duplicated before averaging.

**Stage B - make the comparison scientifically discriminating**

5. Specify the collapsing/adaptive-bound and additive-urgency alternatives
   before looking at their performance. Fit every model to identical trials
   and response coding.
6. Add deterministic train/test folds or another prespecified held-out
   scheme. Keep BIC only as a compatibility summary; use held-out predictive
   likelihood for the primary comparison when feasible.
7. Run parameter recovery and model recovery. Widen or reparameterize bounds
   only on the basis of that recovery audit, not because a preferred model
   loses.
8. Persist model diagnostics, fold-level scores, recovery results, and new
   predictions as derivatives.

**Stage C - paper-specific behavioral diagnostic**

9. Create a prespecified matched-sequence derivative for early bias followed
   by neutralized/reversed late evidence. Report sequence counts and subject
   coverage before testing behavior.
10. Compare choices, decision-time distributions, and evidence at commitment
    between matched histories, then compare each fitted model's prediction of
    the same contrast.
11. Treat reward-rate optimality as a separate optional panel. The local
    trial-feature table does not contain the full payoff/intertrial timing
    schedule. Recover those details from the cluster raw logs and task
    specification first; otherwise label the result a mechanistic comparison,
    not an optimality test.
12. Run all refits through Slurm on a compute node. Local work is limited to
    derivative inspection, figure prototyping, and tests until `pyddm` is
    deliberately installed.

### Token-task diagnostic analogous to the 2012 experiment

Construct matched sequence families in the stimulus/correct-target frame:

- sequences with an early bias that is later neutralized or reversed;
- matched sequences with the opposite early bias but comparable late evidence;
- decisions made only after the histories converge enough for the models to
  make different predictions.

Then compare decision-time distributions, choices/accuracy, and evidence at
commitment. A long-timescale integrator predicts a persistent influence of
the early samples; a short-filter urgency model predicts that sufficiently old
evidence contributes little once the later evidence is matched. Matching must
be declared before looking at the behavioral difference, and the analysis
must check trial counts and subject coverage before testing.

### Reward-rate analysis

The 2012 optimality claim concerns reward rate, not merely a better curve fit.
Reconstruct the actual payoff and timing schedule for Fast and Slow blocks,
including decision time, movement/nondecision time, remaining-token time, and
intertrial interval. Compare the best fixed criterion with a time-varying
criterion under the empirical sequence distribution. If any payoff or timing
component is unavailable, label the result a mechanistic fit rather than an
optimality test.

### Acceptance criteria

- Prespecified eligibility and matching rules.
- Identical observation model and validation data across candidate models.
- Parameter-recovery and model-recovery simulations.
- Subject-level and population-level predictive comparisons.
- Posterior-predictive checks for decision-time distributions, accuracy,
  condition effects, and early-bias sequence families.
- Explicit statement of what is identified: urgency-like time dependence is
  not automatically proof of multiplicative urgency or of reward-rate
  optimality.
- A plain-language result entry in `docs/behavior.md` only after the audit is
  complete.

## Execution rules

- Run lightweight report regeneration locally from the copied group
  derivatives.
- Run characterization or model fitting against the full cluster data through
  Slurm on a compute node, never on a login node.
- Cluster command pattern:
  `uv run meg-tokens --config tokens.toml <command>` inside `~/meg-tokens`.
- Regenerate a report figure with:
  `uv run meg-tokens --config tokens.toml report behavior --figures <key>`.
- Before syncing code, inspect `git status`; do not overwrite unrelated user
  changes in the cluster checkout.
- Persist every inferential result to a derivative before plotting it. The
  plotting layer may compute display summaries only.

## Current review boundary

| Figure family | Layout/science state |
| :--- | :--- |
| F01-F03 | Pooled derivatives are local and can be plotted; provenance, parameter identifiability, and science are not audited. Refits are cluster-only. |
| F04-F06 | Audited and rewritten. |
| F07 | Removed. |
| F08-F11 | Audited and rewritten. |
| F12 | Removed after merging its quality census into F13. |
| F13 | Audited. |
| Exact-posterior and later behavioral figures | Not audited; follow the protocols above rather than trusting current prose. |

Known unrelated issues should not derail the reporting work:

- Optional `pyddm` is absent locally, so some sequential-sampling prediction
  tests cannot run until that dependency is deliberately installed.
- `tests/test_batch_erp_parcellation.py` has pre-existing MEG/ERP fixture
  failures around token-direction validation. Do not treat these as evidence
  of a behavior-report regression.
- Before every cluster sync, inspect both local and remote `git status` and
  preserve unrelated changes. Submit all characterization/model jobs through
  Slurm on a compute node, never directly on the login node.
