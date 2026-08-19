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

## Machine transition: desktop -> laptop (2026-08-18)

This session ran on the desktop (`/home/karim/Projects/meg-tokens`, Linux).
The next session is planned for the laptop instead. Read this before trusting
anything else in this document as "current."

- **The Thura2012 work through step 5 is committed locally on the desktop as
  `0856a47` ("feat(behavior): implement Thura et al. (2012) mechanistic
  model comparison"), on top of `df783f5`, but it has deliberately NOT been
  pushed to `origin/main`.** Karim will handle getting it to the laptop
  himself (push/pull or another method) rather than this session pushing it.
  **The laptop session must not assume this commit is present** - start by
  running `git log -1 --oneline` and `git status --short --branch` on the
  laptop checkout. If HEAD is still `df783f5` (or anything other than
  `0856a47` or a descendant of it), the sync has not happened yet; stop and
  ask Karim rather than redoing or re-deriving any of the validated
  recovery/robustness/exclusion/held-out-pairwise work above, all of which
  is already committed and does not need to be rerun.
- **Everything through step 5 is done and validated on the cluster
  already**, independent of which laptop the next session runs on: recovery
  (job `55358783`, aggregate `55383746`), robustness (job `55386493`,
  aggregate `55428328`), strict-exclusion (job `55431692`, aggregate
  `55450044`), and the new persisted `ssmheldoutpairwise` derivative
  (evaluate-aggregate rerun `55454191`). None of this needs to be
  regenerated from the laptop; it is sitting in
  `/scratch/hamza97/meg-tokens/BIDS/derivatives/sub-group/beh` on the
  cluster, reachable the same way from any machine with the `fir` SSH alias
  configured. **Confirm the laptop session actually has that SSH access and
  alias before assuming any cluster command in this document will work
  unchanged** - it was not verified from the laptop specifically.
- **Local data roots differ by machine** and this document has both: laptop
  `/Users/hamzaabdelhedi/Projects/data/meg-tokens`, desktop
  `/media/karim/Hamza/meg-tokens`, cluster `/scratch/hamza97/meg-tokens`.
  Step 6 (retrieve accepted derivatives, next) was deliberately left un-run
  on the desktop this session specifically so the laptop session can
  retrieve directly into its own data root without a redundant desktop copy
  first. Use the laptop path for step 6 and everything after it. `uv.lock`
  stays untracked on every machine independently; do not commit it from the
  laptop either.
- Next action for the laptop session, once the commit sync above is
  confirmed: **step 6** - retrieve accepted group derivatives, then continue
  the ordered "Remaining steps" list below from step 7 onward (F01-F03
  regeneration, F27 rendering, the `docs/behavior.md` write-up, final tests,
  then the three deferred cleanup steps 11-13).

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

## Thura et al. (2012) mechanistic analysis: live continuation handoff

This section is the durable source of truth for the unfinished F27 analysis.
It was updated while production fitting was running on 2026-08-17. Do not
replace the production outputs with the older pooled SSM derivatives, a
one-start smoke fit, or a local prototype. Do not commit or push any of this
work before Karim reviews it.

### Verified repository, data, software, and literature state

- The required starting commit is present locally and on the cluster:
  `df783f5fe31a2800bb0319cbc4a3280ab86973f4` (`df783f5`,
  `feat(behavior): finalize first-order criterion analysis`) on `main`.
- Local checkout: `/home/karim/Projects/meg-tokens`. The only unrelated local
  untracked file is `uv.lock`; preserve it.
- Cluster checkout: `fir:~/meg-tokens`. Preserve the unrelated cluster
  `logs/`, `scripts/render_figs.sh`, and `scripts/rerun_behavior.sh`. The
  checkout was deliberately fast-forwarded from `302c342` to `df783f5` before
  production work. Never use `rsync --delete`.
- Cluster data root: `/scratch/hamza97/meg-tokens`; the data are not on the
  external drive for fitting. The local configured data root remains
  `/media/karim/Hamza/meg-tokens` for retrieved final derivatives/reporting.
- The original 208 active SSM derivative files were copied before fitting to
  `/scratch/hamza97/meg-tokens/thura2012_preanalysis_ssm_20260817`. Treat those
  files as historical only: their generating commit/settings could not be
  established.
- A wrong-base cluster sync was detected and cancelled before execution. Its
  preserved diagnostic patch is
  `/scratch/hamza97/meg-tokens/thura_sync_wrongbase_20260817.patch`.
- Cluster environment: Python 3.11.4, PyDDM 0.9.0, NumPy
  2.4.2+computecanada, pandas 3.0.0+computecanada, SciPy
  1.17.0+computecanada, and Slurm 24.11.6. There are 32 eligible subject IDs.
- `2022.06.14.494674v1.full.pdf` was not present locally. The primary paper,
  Thura et al. (2012), DOI `10.1152/jn.01071.2011`, was read directly. Cite
  Thura et al. (2012) and the 2009 paper directly; never call either analysis
  “Paul's method.”

### Implemented model equations and conventions

The explicit four-model API is in
`meg_tokens/behavior/analyses/sequential_sampling.py`, selected with
`behavior ssm-fit --model-set mechanistic`:

1. `ddm`: fixed-bound, un-leaky evidence integration baseline.
2. `urgency`: the primary Thura et al. (2012) reproduction. Token evidence is
   low-pass/leaky filtered with fixed `FILTER_TAU_S = 0.2`, multiplied by a
   linearly increasing urgency signal, and compared with a fixed threshold.
   The equivalent criterion is hyperbolically declining, `T / (b + m t)`.
3. `collapsing`: an un-leaky evidence integrator with a hyperbolically
   collapsing/adaptive bound, included as an alternative explanation.
4. `additive_urgency`: a sensitivity model with time-growing drive toward the
   current evidence-preference sign. A common additive drive to two race
   channels cancels in a signed difference, so this label-symmetric
   parameterization is explicitly not claimed to be the 2012 mechanism.

Shared numerical/statistical conventions are explicit and persisted:

- 200-ms token interval; decision time is relative to the first token jump;
  evidence is held after the final token in the baseline fits;
- PyDDM absorbing finite-grid first passage with `dt = 0.01`, `dx = 0.01`, and
  no separate boundary-overshoot correction;
- 2% uniform contaminant/lapse mixture (`MIXTURE_COEF = 0.02`);
- hand-specific motor-corrected nondecision time, fitted per
  subject × condition × model cell;
- the same canonical `primary_analysis_eligible` trials used elsewhere, with
  all exclusions/counts persisted rather than silently changed;
- maximum-likelihood fitting with three deterministic starts for production
  full-data fits, start objectives/convergence recorded, optimizer and
  boundary diagnostics persisted, and finite-difference observed-information
  Hessian SEs only for final full-data fits;
- condition cells are `all`, `fast`, and `slow`, with parameters fit separately
  by subject, condition, and model.

The 2009 chosen-target SumLogLR result is a model-light first-order behavioral
measurement. This 2012 analysis is a mechanistic process-model comparison.
That distinction is implemented in the code and must remain explicit in the
figure text and interpretation.

### Implemented restartable workflows and checks

- `behavior ssm-aggregate` rejects stale one-start/smoke derivatives, requires
  exact 32 × 3 × 4 coverage (384 rows), three start diagnostics per row,
  sidecars, uniform source hash/commit, and pools subject derivatives only
  after those gates pass.
- `behavior ssm-evaluate` and the evaluation array implement deterministic,
  response-stratified three-fold held-out evaluation. Expected coverage is
  32 × 3 folds × 4 models × 3 conditions = 1,152 rows. Failed cells remain
  explicit. Per-trial predictions, weighted held-out statistics, fitted
  correct/error decision-time quantiles, accuracy, condition contrasts,
  boundary/SE/convergence audits, and subject/group posterior-predictive checks
  are persisted.
- Recovery is a restartable 12-repetition array. Each repetition uses a unique
  deterministic interior Latin-hypercube-like truth point for every model
  parameter and two fit starts. It produces 15 parameter-recovery and 16
  model-recovery rows per repetition, followed by an aggregate/confusion stage.
- Robustness is a 192-task subject × configuration array, not a monolithic
  login-node job. Six configurations are fit: baseline, `tau100`, `tau300`,
  `solver20`, `post_horizon_evidence_zero`, and `expanded_bounds`. The
  post-horizon configuration is an evidence-horizon test, not an overshoot
  correction. Expected merged fit coverage is 2,304 rows.
- Exclusion robustness refits the strict subset
  `primary & token_log_rows == 15 & design_time_alignment_valid` with two
  starts for all 32 subjects. The verified real-data census is 10,961 strict
  trials versus 16,324 primary trials, with 32 subjects, 64 Fast/Slow cells,
  minimum 56 trials/cell, and no cell below 50. The complete-token and timing
  masks are identical on current data, so no duplicate second refit is run.
  The aggregate compares within-population model contrasts rather than raw BIC
  values across different sample sizes. H20 leave-one-subject group
  sensitivity remains a separate exclusion diagnostic.
- Every new sidecar records the base commit, dirty status, deterministic source
  SHA-256, stage/eligibility scope, software versions, parameter ranges, and
  solver/timing conventions. Slurm logs record command, host, environment,
  commit, dirty state, start/end time, and output paths.
- The current F27 builder requires the held-out group statistics and creates a
  publication-quality four-panel figure: criterion comparison; observed versus
  predicted correct/error decision-time distributions; decision-variable
  trajectories; and per-subject held-out model comparison. It deduplicates
  repeated time-course fields and uses subject-balanced summaries.

Local validation before production submission:

- broad behavior/report suite: 516 passed, 5 skipped;
- workflow/CLI suite: 51 passed, 1 skipped;
- latest focused audit: 51 passed, 1 skipped;
- `compileall`, every Thura shell-script syntax check, and `git diff --check`
  passed. Rerun the broad relevant suites after retrieving real results and
  after the final documentation/figure fixes.

### Frozen matched-sequence diagnostic

The matching rule was frozen before inspecting outcomes and never matches on
the final response:

- represent stimuli in the correct-target frame;
- require opposite early evidence signs at jump 3 with identical absolute
  lead;
- require equal lead at jump 6 (600 ms/three filter time constants later);
- require maximum trajectory distance ≤ 2 over jumps 6–15;
- globally sort candidates by trajectory RMSE and stable trial IDs, then greedily
  form deterministic one-to-one pairs from stimulus information only;
- analyze the response subset only after pairing, requiring both decisions to
  occur after jump 6, and compute model predictions conditional on first
  passage after convergence.

The real-data census check retained 1,955 stimulus-only pairs and 614
post-convergence response pairs across 29 subjects: Fast 310 pairs/27 subjects,
Slow 304 pairs/28 subjects. Early and convergence evidence mean differences
were exactly zero. These counts are implementation validation only until the
Slurm-generated evaluation derivatives are merged. Treat this as a
complementary behavioral diagnostic, never proof of a unique urgency mechanism.

### Reward-rate/optimality decision

Do not run or claim a reward-rate optimum from the present data contract. The
repository/acquisition material inspected so far does not verify the reward
values, error/timeout penalties, full movement/nondecision contribution, or
the complete remaining-token/intertrial timing schedule needed for a defensible
reward-rate calculation. The existing ITI fingerprint code is not a reward
schedule. If authoritative acquisition/task code supplying every missing value
is later found, document it and then add the analysis; otherwise the final
report must explicitly omit the optimality claim.

### Cluster job ledger and current state

Cancelled before producing accepted results:

- `55221612`: wrong-base attempt, caught/cancelled before execution;
- `55224933`: obsolete metadata-correction attempt, cancelled before execution.

Diagnostic smoke:

- `55225675_0`: H01 one-start smoke, completed in 17:26 and validated the
  cluster pathway. It is intentionally rejected by the production aggregator
  and is not an accepted production fit.

Production primary fit:

- parent array `55227660`, submitted as
  `sbatch --array=0-31%8 scripts/thura2012_fit_array.sh tokens.toml 3`;
- requested resources: eight CPUs/task, 8-hour limit; logs are
  `fir:~/meg-tokens/logs/thura12-fit-55227660_%a.out`;
- exact generating source hash:
  `979b517caf2427d424aa541226f04e2a10948ab7957066981d4d0494f2fc1d2f`;
- tasks 0–7 (H01–H08) completed with exit 0 in 36:51–58:15 and passed row,
  three-start, optimizer, convergence, fit-error, and source-hash checks;
- task 11 (H12) completed with exit 0 in 34:02 and passed the same checks;
- all 32 tasks subsequently completed with exit 0. Task runtimes were
  31:24--1:13:38. The accepted aggregate contains 384 rows (32 subjects x
  three conditions x four models), exactly three starts per cell, 384/384
  optimizer successes, 384/384 converged fits, no fit errors, and uniform
  source/commit provenance.

The exact primary source was archived before any downstream resync:

- patch:
  `/scratch/hamza97/meg-tokens/thura2012_primary_source_979b517caf2427d424aa541226f04e2a10948ab7957066981d4d0494f2fc1d2f.patch`
  (SHA-256
  `5ae35e093a9d886619e5e2a6fdd91ab4aad3223cb983a9a24bc34ccd2b24cdb4`);
- source tarball:
  `/scratch/hamza97/meg-tokens/thura2012_primary_source_979b517caf2427d424aa541226f04e2a10948ab7957066981d4d0494f2fc1d2f.tar.gz`
  (SHA-256
  `21da0b73fa38d4fd59934f2018e5660dcd9fb54dc2a29083fd4051afa12214b8`).

The first eight subjects already show why recovery/expanded-bound work cannot
be skipped: all 24 urgency cells converged, but 12/24 urgency-scale SEs and
12/24 urgency-onset SEs were missing; several urgency-scale estimates were at
or near 2.0. Most generic `boundary_hit` flags also include a nondecision-time
lower-bound flag, so final reporting must separate upper urgency-bound hits,
nondecision lower-bound hits, optimizer convergence, and Hessian uncertainty
failure. Do not interpret this partial cohort as a scientific result.

### Authoritative stop point for the next model (2026-08-18)

This section supersedes the earlier live snapshots. The current model is being
stopped at the user's request. Do not restart completed primary or held-out
fits. Do not cancel the active recovery array. There has been no commit or
push. No agent-side monitoring loop or scheduled wait remains active in this
session; the recovery array continues independently under Slurm. The next
session must inspect Slurm directly before acting on the snapshot below.

Repository and synchronization state:

- local repository: `/home/karim/Projects/meg-tokens`, branch `main`, HEAD
  `df783f5fe31a2800bb0319cbc4a3280ab86973f4`;
- cluster repository: `fir:~/meg-tokens`, same HEAD; data root
  `/scratch/hamza97/meg-tokens`;
- all Thura implementation and Slurm files needed by the active recovery array
  are already on the cluster. Do **not** sync either documentation or code
  while recovery tasks remain queued: every recovery task must see the same
  frozen source tree and source hash;
- preserve local unrelated `uv.lock` and cluster unrelated/untracked `logs/`,
  `scripts/render_figs.sh`, and `scripts/rerun_behavior.sh`. Never use a
  deleting rsync, reset, overwrite, commit, or push.

Completed production jobs and failures that must remain in the audit trail:

- primary array `55227660`:
  `sbatch --array=0-31%8 scripts/thura2012_fit_array.sh tokens.toml 3`; all 32
  tasks completed with exit 0 (31:24--1:13:38). Its aggregate job `55262390`
  completed with exit 0 in 1:52;
- initial held-out array `55272741` was deliberately cancelled after about two
  hours because the original four-hour limit could not accommodate observed
  progress (only 11--18/36 fits per task). It produced no accepted outputs.
  `scripts/thura2012_evaluate_array.sh` now has a 12-hour limit;
- replacement held-out array `55290226`:
  `sbatch --array=0-31%16 scripts/thura2012_evaluate_array.sh tokens.toml`; all
  32 tasks completed with exit 0 (2:48:44--4:59:26), and all 1,152 held-out
  cells converged without fit errors;
- first evaluation merge `55342091` failed with exit 2 after 1:36 because
  legitimate empty subject TSVs (subjects with no post-convergence matched
  pairs) raised `pandas.errors.EmptyDataError`. The merger now skips empty raw
  contributions and recomputes pooled summaries; no fits were affected;
- corrected evaluation merge `55350240` completed with exit 0 in 1:29;
- earlier cancelled/diagnostic jobs remain: `55221612` wrong-base attempt,
  `55224933` obsolete metadata attempt, and successful one-start H01 smoke
  `55225675_0`, which the production aggregator correctly rejects.

Accepted provenance hashes:

- primary subject fits:
  `979b517caf2427d424aa541226f04e2a10948ab7957066981d4d0494f2fc1d2f`;
- primary group aggregation:
  `0e2969b86181ec1ab8bc42d8f6e7bbad8def5683729a05d6323e05255f8f5fc1`;
- held-out subject outputs:
  `ccb5dc49929c85cb3c6d23050069dc7c59696992fc508be9701bff7250344fa8`;
- evaluation group outputs:
  `4b8e6b1725d653bdc43adee82cf0be5f113a1ba2c02a0f5b8fea8434c83691bc`.

The exact primary patch and tarball remain under the cluster data root at the
paths and checksums recorded above. The recovery source hash must be read from
its outputs after the first task finishes and must be identical across all 12
tasks.

Validated primary aggregate:

- `ssmcomparison`: 384 x 35, 32 subjects, three conditions, four models;
- `ssmtimecourse`: 419,920 x 12, with no exact duplicate rows. Repeated key
  values across `trial_class` are intentional and must not be collapsed across
  that field;
- `ssmtrialpredictions`: 130,560 x 9;
- `ssmcomparisonstats`: 28 rows; `ssmexclusionsensitivity`: 18 rows;
  `ssmpopulation`: 1,257 rows; `ssmpopulationstats`: 45 rows.

In-sample candidate-minus-DDM BIC contrasts (negative favors the candidate):

- multiplicative urgency: all -239.600 (30/32 subjects, p=3.39e-9), fast
  -132.214 (30/32, p=8.46e-9), slow -117.375 (30/32, p=1.26e-9);
- collapsing bound: all -169.036 (31/32, p=1.42e-11), fast -94.944 (30/32,
  p=1.01e-11), slow -77.216 (31/32, p=1.40e-11);
- additive urgency: all -18.999 (16/32, p=.191), fast -8.025 (17/32,
  p=.343), slow -9.626 (16/32, p=.250).

The accepted boundary audit has 96 cells per model and all 384 optimizer fits
converged, but generic boundary flags are frequent and must be decomposed:

- multiplicative urgency: 19/96 urgency-scale estimates near the upper bound,
  6/96 urgency-onset estimates near the upper bound, 47/96 missing SEs for
  each of urgency scale and onset, and 96/96 nondecision estimates near the
  lower bound;
- collapsing bound: 36/96 bound estimates near the upper bound, 3/96 collapse
  rates near the lower and 2/96 near the upper bound, with 94/96 nondecision
  estimates near the lower bound;
- additive urgency: 39/96 additive-scale estimates near the upper bound and
  89/96 nondecision estimates near the lower bound;
- fixed DDM: 90/96 nondecision estimates near the lower bound.

Therefore optimizer convergence does not establish parameter identification.
Expanded-bound robustness and recovery are mandatory before interpreting the
urgency parameters. Missing Hessian SEs may be numerical/identifiability
failures rather than optimizer failures.

Validated evaluation aggregate:

- eligibility audit 256 x 9; exclusion robustness 3 x 15; boundary audit
  12 x 28; mechanistic statistics 18 x 14;
- held-out fits 1,152 x 25 with exact subject x condition x model x fold
  coverage, 384-row fold audit, all fits converged, and no errors;
- held-out predictions 130,560 x 10 with no duplicate key;
- distribution checks 11,520 x 15, covering 32 subjects, all three conditions,
  four models, and both outcomes. Observed quantiles are exact raw-trial
  quantiles. Predicted densities/quantiles reuse `ssmtimecourse` on its 20-ms
  grid; integrated decision mass ranges 0.818704--1.001748, with the slight
  value above one attributable to numerical grid integration. Do not claim
  greater precision than this representation supports;
- matched-sequence outputs: 3,910 stimulus rows = 1,955 pairs, 64 audit rows,
  4,896 model-prediction rows = 612 post-convergence pairs x two trials x four
  models, 29 represented subjects, four observed-stat rows, and 16
  model-stat rows. Exact early/convergence balance is zero by construction.

Held-out candidate-minus-DDM per-trial log-likelihood contrasts (positive
favors the candidate):

- multiplicative urgency: all 0.2442, 95% CI [0.1825, 0.3059], 30/32 subjects,
  p=4.03e-9; fast 0.2512 [0.1848, 0.3176], 30/32, p=1.06e-8; slow 0.2695
  [0.2067, 0.3324], 30/32, p=7.12e-10;
- collapsing bound: all 0.1739 [0.1391, 0.2086], 31/32, p=1.93e-11; fast
  0.1850 [0.1472, 0.2229], 32/32, p=3.43e-11; slow 0.1838 [0.1469,
  0.2207], 32/32, p=2.18e-11;
- additive urgency: all 0.0092 [-0.0197, 0.0381], 13/32, p=.521; fast 0.0122
  [-0.0248, 0.0491], 17/32, p=.507; slow 0.0120 [-0.0344, 0.0584], 13/32,
  p=.601.

Thus both multiplicative urgency and collapsing bounds outperform the fixed
integrator in held-out prediction; additive urgency does not. A direct,
subject-paired urgency-versus-collapsing held-out contrast should be computed
before stating which flexible mechanism predicts better. Recovery and
robustness remain outstanding, so these are not yet final causal/mechanistic
conclusions.

Matched-sequence diagnostic:

- Fast: 310 post-convergence pairs across 27 subjects; observed
  against-minus-for decision time +22.9 ms (p=.488) and accuracy -0.0198
  (p=.674);
- Slow: 302 pairs across 28 subjects; decision time -55.1 ms (p=.326) and
  accuracy -0.0912 (p=.142).

No observed matched effect is statistically established. Some fitted-model
contrasts are significant (for example urgency-predicted accuracy differences
of +0.0306 Fast and +0.0230 Slow), but these do not turn a null behavioral
diagnostic into proof of urgency gating. Treat the diagnostic as complementary,
not mechanism-identifying.

Implementation fix made after the first evaluation attempt: raw trial labels
`Fast`/`Slow` did not match normalized fit labels `fast`/`slow`, and direct
equality could not produce the pooled `all` cell, leaving old distribution
tables empty. `fitted_distribution_checks()` now uses canonical condition
groups, preserves exact observed quantiles, and reuses primary time-course
densities rather than refitting. The aggregate merger also recomputes the group
exclusion audit and tolerates legitimate empty subject matched-prediction files
(H03, H15, H26). Focused tests now pass 45/45 with one skipped; the broad
behavior/report suite passes 519 tests with five skipped and 109 warnings.
Compile, shell syntax, and `git diff --check` checks pass.

Active cluster work at stop time:

- recovery parent `55358783`, submitted as
  `sbatch --array=0-11%4 scripts/thura2012_recovery_array.sh tokens.toml 12`;
- at the final query on 2026-08-18, tasks 0, 2, and 3 had completed with exit
  0 in 32:07--32:28; task 1 was still running at 32:30; tasks 4--11 were
  pending only because of the `%4` throttle. No task had failed. Actual
  first-wave job IDs are 55359093--55359096;
- do not cancel and do not sync any changed file to `fir:~/meg-tokens` until all
  12 tasks are terminal. The next session should query `squeue`, `sacct`, and
  the task logs when it starts rather than assuming this snapshot is current.
  Validate each replicate has 15 parameter rows, 16 model-recovery rows,
  exactly two starts, one unique truth design, no errors, and identical source
  hash. Resubmit only failed indices after diagnosis;
- after successful validation, submit
  `sbatch scripts/thura2012_recovery_aggregate.sh tokens.toml 12`; require 180
  parameter rows, 192 model rows, parameter-recovery summaries, and a complete
  model-confusion table.

### Remaining steps, in exact order

1. Finish and aggregate recovery as specified immediately above. Record final
   task states, runtimes, logs, environment, commit, source hash, commands, and
   output paths. Only after the recovery array is entirely terminal may the
   source tree be synchronized again.
2. Re-check `git status --short --branch` locally and on `fir`. Deliberately
   sync only required changed Thura files, with no deletion and no `uv.lock`.
   Preserve all unrelated files. Lightweight validation may run on the login
   node; every substantial analysis must use Slurm compute nodes.
3. Run robustness:
   `sbatch --array=0-191%8 scripts/thura2012_robustness_array.sh tokens.toml 1`,
   monitor as appropriate in the new session, validate exactly 2,304 rows
   across baseline, tau100, tau300,
   solver20, `post_horizon_evidence_zero`, and expanded bounds, then submit
   `sbatch scripts/thura2012_robustness_aggregate.sh tokens.toml`. The
   post-horizon configuration is not an overshoot manipulation; describe it
   precisely. Compare configurations with subject-paired contrasts.
4. Run strict-exclusion sensitivity:
   `sbatch --array=0-31%8 scripts/thura2012_exclusion_array.sh tokens.toml`, then
   `sbatch scripts/thura2012_exclusion_aggregate.sh tokens.toml`. The strict
   mask is primary eligibility plus `token_log_rows == 15` plus valid design
   alignment. Its frozen census is 10,961 trials versus 16,324 primary trials,
   32 subjects, 64 Fast/Slow cells, minimum 56 trials, and no cell below 50.
   Compare models within the same retained population; do not silently change
   exclusions.
5. Audit recovery/confusion, robustness, strict-exclusion, exact direct
   urgency-versus-collapsing held-out contrasts, subject-level results,
   convergence, boundary hits, and uncertainty. State that model comparison
   distinguishes predictive accounts but cannot uniquely prove a neural
   mechanism. Reward-rate optimality remains omitted because verified payoff,
   reward/error penalties, complete ITI, and task timing are missing; do not
   invent them.

   **Get the strict-exclusion population relationship right, in the code and
   in any write-up.** The primary population (16,324 trials) already pools
   both complete 15-row and anomalous 14-row token logs with no distinction -
   the model fit itself never reads `token_log_rows` (only
   `exclusion_robustness_audit` does; the fitting path uses the designed
   `token_directions` sequence and continuous `dt_ms`, which do not depend on
   acquisition-log completeness). The strict population (10,961 trials) is
   not a second, independently assembled 15-row-only sample added alongside
   primary - it is `primary_analysis_eligible AND token_log_rows==15 AND
   design_time_alignment_valid`, a pure subset carved out of the same 16,324
   primary trials (10,961 + 5,363 dropped 14-row trials = 16,324 exactly).
   Never describe this as "ran with 14+15, then added 15" or as two separate
   populations - it is one population (primary) versus a strict narrowing of
   it (drop the 14-row third), which is exactly why the paired
   `strict_delta_bic_minus_primary_delta_bic` contrast in
   `ssmexclusionrobustnessstats` is a clean within-subject, same-underlying-
   trials-where-retained comparison. State this explicitly wherever the
   strict-exclusion result is interpreted, including the eventual F27
   write-up in `docs/behavior.md` (step 9).

   **Direct urgency-vs-collapsing held-out contrast: done, persisted as a
   real derivative (2026-08-18).** `heldout_model_statistics` only ever
   contrasted each candidate against `ddm`; two candidates that both beat
   `ddm` are not thereby shown to differ from each other. Added
   `heldout_pairwise_model_statistics` to
   `meg_tokens/behavior/analyses/sequential_sampling.py`, refactored to
   share `_heldout_subject_scores` (the weighted-by-`n_test` subject score)
   and `_paired_score_contrast` (the one-sample-vs-zero CI/favouring-count
   logic) with the existing `heldout_model_statistics` rather than
   duplicating either - both were extracted from what was previously
   `heldout_model_statistics`'s own body, so this is a pure refactor for the
   existing function, not a behavior change (confirmed by two passing
   pre-existing tests plus two new ones,
   `tests/behavior/test_analyses_sequential_sampling.py`). Wired into
   `aggregate_mechanistic_evaluation` in `thura2012.py` as a new persisted
   derivative `ssmheldoutpairwise` (added to `_EVALUATION_TABLES` and the
   `recomputed` set, since it's computed at merge time from the already-
   collected `ssmheldout` table, not per-subject). Local suites: 521 passed,
   5 skipped, no regressions. Synced the three changed files
   (`sequential_sampling.py`, `thura2012.py`, the test file) to `fir` -
   safe, since nothing was queued on the cluster at that point - smoke-
   tested the new function directly against the cluster's numpy
   2.4.2/pandas 3.0.0/scipy 1.17.0 stack (pytest itself is not installed in
   the cluster venv), then reran only the merge job:
   `sbatch scripts/thura2012_evaluate_aggregate.sh tokens.toml` -> job
   `55454191`, completed exit `0` in 33s (cheap, since it recomputes only
   from the already-collected per-subject `ssmheldout` array outputs, no
   refitting). `ssmheldoutpairwise` now has 18 rows (6 model pairs x 3
   conditions), and its `collapsing`-vs-`urgency` row matches the earlier
   ad-hoc script's number exactly (mean -0.0703 for `condition=all`, sign
   flipped from "urgency vs collapsing" to "collapsing vs urgency").
   **Result, now on the record as a persisted, reproducible derivative
   rather than an ad-hoc script:** urgency significantly outpredicts
   collapsing directly in every condition (all: mean=+0.0703, t=4.87,
   p=3.1e-5, 28/32 subjects favoring urgency; fast: p=2.1e-4; slow:
   p=6.9e-6). Both urgency and collapsing directly and decisively beat
   additive_urgency in every condition too (p<3e-8 throughout, 30-31/32
   subjects), consistent with additive_urgency's weak showing everywhere
   else in this analysis.
6. Retrieve only accepted group derivatives from
   `/scratch/hamza97/meg-tokens/BIDS/derivatives/sub-group/beh` to the local
   data root of whichever machine this step runs on - `/media/karim/Hamza/
   meg-tokens` on the desktop, `/Users/hamzaabdelhedi/Projects/data/
   meg-tokens` on the laptop (see "Machine transition" below; this step was
   deliberately not yet run on the desktop specifically so the laptop
   session can do it directly into its own data root without a second
   redundant copy). Use explicit non-deleting rsync paths either way.
7. **Regenerate and re-audit F01-F03 before touching F27.** The retrieved
   `ssmcomparison` file is the *same filename* the headline two-model figures
   (`ssmcomparison-deltabic`, `ssmcomparison-urgencyscale`,
   `ssmcomparison-urgencyparams`) already read from, and it was overwritten in
   place by the mechanistic production fit: it now holds 384 rows (32
   subjects x 3 conditions x 4 models, `n_starts=3`, commit `df783f5`,
   `stage=thura2012_mechanistic_evaluation`) instead of the 192-row two-model
   file currently on the laptop, whose sidecar has no `models`, `n_starts`,
   `git_commit`, or `source_tree_sha256` at all (it predates that provenance
   convention; its start count is unknown). The `ddm`/`urgency` rows were
   therefore refit, not merely carried over, when the mechanistic set was
   fitted. Before rendering F27: regenerate F01-F03 against the retrieved
   file, diff their `ddm`/`urgency` parameter estimates, deltaBIC, and
   urgency-scale Fast-vs-Slow contrast against the numbers currently reported
   for H1/H2 in `docs/behavior.md`, and flag any material change to Karim
   before silently overwriting that prose. Do not skip this because the file
   already exists locally under the same name. "Regenerate" means re-run the
   local report build (`report behavior --figures
   ssmcomparison-deltabic,ssmcomparison-urgencyscale,ssmcomparison-urgencyparams`)
   against the retrieved derivatives, not a new PyDDM fit: each model's fit is
   independent of which other models were requested in the same
   `ssm-fit` call (`_fit_all_cells` tasks one `(cell, model)` pair at a time),
   so the `ddm`/`urgency` rows already sitting in the retrieved 384-row file
   are the correct refit numbers to use - do not re-run `ssm-fit` with
   `models=SSM_MODELS` separately, since its default `n_starts=1` would not
   match the `n_starts=3` the mechanistic set was fit with and would silently
   introduce a second, non-comparable pair of `ddm`/`urgency` estimates.
   H1/H2 numbers are not confined to the F01-F03 panels - the same ΔBIC and
   `urgency_scale` values are separately quoted as corroborating evidence in
   two other Findings sections that must be updated in the same pass:
     - "Fast vs. Slow stretches decision time proportionally..." currently
       states `ΔBIC = -238.9, t(31) = -8.12, p = 3.6e-9, 30/32 subjects` and
       `urgency_scale Δ = -0.108, t(31) = -2.52, p = .017`. Replace both with
       the values read from the regenerated `ssmcomparisonstats` (the same
       table the figures read, so the prose and the figure cannot disagree).
     - The confidence-at-commitment finding ("Success probability at
       decision...") references "H1 (model comparison)" without repeating
       numbers - no edit needed there beyond confirming the qualitative
       direction (urgency favored) still holds.
   Also cross-check whichever ΔBIC F27's own write-up (step 9) reports for
   `urgency` vs `ddm`, `condition=all` - it must be numerically identical to
   what F01-F03 and the two Findings citations above now report, since all
   four are the same fitted rows from the same retrieved file. Two different
   numbers for the same comparison anywhere in `docs/behavior.md` is a bug to
   catch here, not something to reconcile later.
8. Render the registered `ssmcomparison-mechanistic` F27 on a Slurm compute
   node.
   Produce final PDF, PNG, JSON/statistical sidecars, and required TSVs. Its
   core panels are criterion comparison, observed/predicted correct/error RT
   distributions, decision-variable trajectories, and per-subject model
   comparison; extend with held-out/recovery/matched results only where clear.
   Use subject-balanced summaries and deduplicate time-course rows using all
   semantically relevant fields. Rasterize the PDF at high resolution, inspect
   it visually, fix clipping/spacing/legends/annotations/alignment, and rerender
   until final.
9. Finish the accessible F27 interpretation in `docs/behavior.md`, then replace
   this live handoff section with the final job ledger, outputs, numerical
   conclusions, limitations, and remaining decisions. Preserve the F16
   instructions below.
10. Run the full relevant local tests with retrieved derivatives, compile and
    shell checks, `git diff --check`, and a requirement-by-requirement A--G
    audit. List all changed files and exact output paths. Do not commit or push
    until the user has reviewed the work.
11. **Deferred: reorder the figure registry to match the scientific act map.**
    Not part of this analysis and must not block it; do only after step 10 and
    after Karim has reviewed the F27 work. `ssmcomparison-mechanistic` was
    given F27 (2026-08-18) as a minimal, non-colliding placeholder — the
    registry's F-numbers currently follow historical implementation order
    (module-by-module: F01-F03/F21/F22/F27 in `modeling.py`, F04-F06 in
    `distributions.py`, F08-F13 in `design.py`, F14-F18 in `evidence.py`,
    F19-F20 in `sequential.py`, F23-F26 in `individual.py`), not the
    "Scientific act map" order documented above (foundation -> commitment
    policy -> mechanistic adjudication -> sequential adaptation -> individual
    differences). A full reorder means renumbering every figure to read in
    that narrative order and updating every place an F-number appears: the
    `REGISTRY` tuple order in `meg_tokens/reports/behavior/__init__.py`,
    every module/function docstring F-number in
    `meg_tokens/reports/behavior/*.py`, `docs/behavior_reporting_plan.md`
    (itself already stale - it jumps from F05 straight to F15 and never
    labeled F06/F08-F14 with `####` headers), `docs/behavior.md`, and any
    script/comment referencing a specific number. Treat retired numbers
    (F07, F12) as permanently retired, not reusable, consistent with how
    this document already preserves them as historical record rather than
    silently renumbering into their slots. Do this as one deliberate pass,
    not incrementally, so no file is left with a stale number mid-way.

    **Also fold in: retire the old F21 in favor of F27, and let F27 become
    the new F21 (decided with Karim, 2026-08-18).** `ssmtimecourse-fit`
    (`build_ssmtimecourse_fit` in `modeling.py`) is limited to
    `style.MODEL_ORDER = ("urgency", "ddm")` and its three panels - criterion
    time course, observed/predicted correct/error densities, urgency
    decision-variable trajectory - use flat pooled means with no uncertainty
    band. `ssmcomparison-mechanistic`/F27 (`build_ssmcomparison_mechanistic`)
    reproduces those same three panels for all four models with proper
    subject-balanced within-subject 95% CI bands, and adds a fourth
    (held-out per-subject model comparison) that F21 never had. F21 is
    therefore not a companion figure to F27, it is a strict, worse subset of
    it - unlike F07 (content removed, no replacement) this is a genuine
    upgrade-in-place, so unlike the "retired numbers are not reusable" rule
    above, F21's *number* should be kept and now refer to the *new* content
    (mirroring how F12 was folded into F13, i.e. absorb the superseded
    figure's slot rather than leaving two entries). Concretely, in the same
    deliberate pass:
      - Delete the `FigureSpec(key="ssmtimecourse-fit", ...)` entry from
        `meg_tokens/reports/behavior/__init__.py` and the
        `build_ssmtimecourse_fit` function from
        `meg_tokens/reports/behavior/modeling.py` entirely - do not leave it
        as unused dead code.
      - Rename `build_ssmcomparison_mechanistic`'s F-number from F27 to F21
        in the `modeling.py` module docstring and everywhere else F27 is
        used as this figure's number (this handoff, `docs/behavior.md`,
        `docs/behavior_reporting_plan.md`'s F21 entry, and the "Current
        review boundary" table). The FigureSpec `key`
        (`ssmcomparison-mechanistic`) and function name
        (`build_ssmcomparison_mechanistic`) do not change - only the F-number
        label changes, since the key already correctly matches its own
        `analysis`/`view` fields.
      - Update or remove the test for the old F21
        (`tests/reports/test_modeling.py`), and re-run the full report test
        suite afterward.
      - Verify no other file/derivative/script depends on the
        `ssmtimecourse-fit` key specifically (only on the `ssmtimecourse`
        derivative table, which F27 already reads) before deleting it.
12. **Deferred: consolidate SSM modeling workflow code into a new
    `meg_tokens/workflows/sequential_sampling.py` (decided with Karim,
    2026-08-18).** Separate task from step 11 - this is a code-architecture
    change, not a reporting/figure-numbering one, and should be done as its
    own deliberate pass. Not part of this analysis and must not block it; do
    only after step 10 and Karim's review, and not before `thura2012.py` is
    safe to touch (frozen while any array task is queued, same as always).
    Motivation: `_package_version`, `_source_tree_sha256`
    (`_ssm_source_tree_sha256` in `behavior_characterization.py`), and
    `_selected_features` are each duplicated near-identically between
    `meg_tokens/workflows/behavior_characterization.py` and
    `meg_tokens/workflows/thura2012.py` - same logic, written independently
    twice, already at risk of silently drifting apart. The split also
    doesn't match what the two files actually contain: `thura2012.py`
    already holds every *downstream* SSM stage (aggregate, evaluate,
    recovery, robustness, exclusion), while the *primary* fit
    (`fit_subject_sequential_sampling`, driven by the `ssm-fit` CLI command)
    is stranded inside `behavior_characterization.py` alongside many
    unrelated lightweight behavioral analyses (cohort, distributions, SPD,
    criterion, sequential effects, individual differences) that have
    nothing to do with sequential-sampling modeling. Plan:
      - Create `meg_tokens/workflows/sequential_sampling.py`, mirroring the
        existing `meg_tokens/behavior/analyses/sequential_sampling.py`
        naming convention for this concern.
      - Move `fit_subject_sequential_sampling` and its provenance helpers
        (`_ssm_fit_metadata`, `_ssm_source_tree_sha256`, `_package_version`)
        out of `behavior_characterization.py` into the new file.
      - Move everything currently in `thura2012.py` into the same new file
        (or rename `thura2012.py` itself and fold the fit function in - either
        way, end with exactly one file), keeping `thura2012.py`'s versions of
        `_selected_features`/`_package_version`/`_source_tree_sha256` since
        they already correctly use the shared `require_file()` helper that
        `behavior_characterization.py`'s copy does not.
      - End with exactly one `_selected_features`, one `_package_version`,
        one `_source_tree_sha256`, and one provenance-metadata builder,
        shared by every SSM stage (fit, aggregate, evaluate, recovery,
        robustness, exclusion) - no duplication by construction.
      - Update `meg_tokens/cli/main.py`'s import of
        `fit_subject_sequential_sampling` (currently imported from
        `behavior_characterization.py`) to the new module.
      - Move/update the corresponding tests, and re-run the full
        workflow/CLI and behavior test suites afterward.
13. **Deferred: deduplicate the paired-contrast favouring-count boilerplate
    in `sequential_sampling.py` (decided with Karim, 2026-08-18).** Separate,
    smaller cleanup than steps 11/12; do whenever convenient, no ordering
    dependency on them. The 3-line pattern
    `int((differences < 0).sum())` / `int((differences > 0).sum())`
    immediately after an `one_sample_statistics(differences)` call is
    copy-pasted independently 6 times across 4 functions, each just
    relabelling the two ends: `model_comparison_statistics` (original
    2-model in-sample BIC, feeds F01-F03), `mechanistic_model_statistics`
    (4-model in-sample BIC), `robustness_statistics` (twice in the same
    function), and `exclusion_robustness_statistics` (the strict-vs-primary
    comparison from step 4). This is the same shape of duplication as the
    held-out favouring-count logic that step 5 already extracted into
    `_paired_score_contrast` - extend or reuse that helper (it currently
    returns `n_subjects_favoring_a`/`n_subjects_favoring_b` plus the CI, so
    each call site would just relabel those two keys as it already does in
    `heldout_model_statistics`/`heldout_pairwise_model_statistics`) rather
    than adding a second, differently-named generic helper. Note
    `_paired_score_contrast` also computes a 95% CI half-width via
    `stats.t.ppf`, which none of these 4 in-sample functions currently do
    (they only report the favouring counts and the `one_sample_statistics`
    summary) - confirm whether they should gain a CI too as part of this
    pass, or whether the helper needs a CI-optional variant. Re-run the full
    behavior test suite afterward; this changes internal structure only, so
    no test's asserted values should change.

### Session continuation (2026-08-18, after the stop point)

This section records what happened in the session that resumed from the
"Authoritative stop point" above. It supersedes the "Active cluster work at
stop time" snapshot for recovery; that snapshot is preserved above only as a
historical record. Do not commit or push any of this work before Karim
reviews it.

**Step 1 (recovery) - complete and validated.**

- Recovery array `55358783` (all 12 tasks, `%4` throttle) finished with all
  12 tasks `COMPLETED`, exit `0:0`, runtimes 21:56-34:03.
- Per-replicate validation passed for all 12 reps: exactly 15 parameter-
  recovery rows and 16 model-recovery rows each; `n_starts=2` and
  `converged=True` on every row; zero `fit_error` rows; one unique
  `truth_design_index` per replicate (0-11, all distinct, matching
  `recovery_repetition_indices`); identical `source_tree_sha256`
  (`4b8e6b1725d653bdc43adee82cf0be5f113a1ba2c02a0f5b8fea8434c83691bc`) and
  `git_commit` (`df783f5...`) across all 24 sidecars.
- Aggregate job `55383746`
  (`sbatch scripts/thura2012_recovery_aggregate.sh tokens.toml 12`) completed
  exit `0`, elapsed `1:44`. Merged outputs validated: `ssmparameterrecovery`
  180 rows, `ssmmodelrecovery` 192 rows, `ssmparameterrecoverystats` 15 rows,
  `ssmmodelrecoverystats` (the model-confusion table) 16 rows (full 4x4
  true-vs-selected matrix).
- **Model recovery is perfect**: every model (`ddm`, `urgency`, `collapsing`,
  `additive_urgency`) is selected in all 12/12 repetitions when it is the true
  generating model; 100% on-diagonal, 0% off-diagonal everywhere. Model
  *selection* from this design is trustworthy.
- **Parameter recovery is uneven and confirms the boundary/SE concerns already
  flagged from the real-data fits.** Most parameters recover well (r = 0.79-
  0.999: `bound`, `drift_scale`, `collapse_rate`, `nondecision_s` across all
  four models, boundary-hit rate 0.0 except as noted below). Two `urgency`
  parameters recover poorly: `urgency_scale` r=0.444, bias -0.083;
  `urgency_onset_s` r=0.325, bias -0.221. `additive_urgency`'s `additive_scale`
  also recovers weakly (r=0.463, bias +0.399) with a 0.25 boundary-hit rate,
  versus 0.0 for every other model's parameters. This must be stated
  explicitly wherever the fitted urgency-scale/urgency-onset or additive-scale
  point estimates are interpreted: model comparison/selection is reliable,
  but those specific mechanistic parameter values are only weakly identified
  by this design and should not be over-interpreted individually.

**Step 2 (git sync) - complete, code required no sync.**

- Re-checked `git status --short --branch` locally and on `fir`: identical
  sets of modified/untracked Thura-related paths on both sides.
- Verified with `sha256sum` on all 21 changed/untracked Thura implementation,
  workflow, and shell-script files: every one was already byte-identical
  between local and cluster. The frozen source tree the recovery array
  validated against was already correct; no code sync was necessary.
- The only divergent file was `docs/behavior_reporting_session_handoff.md`
  itself (cluster was 756 lines/pre-stop-point, local was 920 lines). Synced
  only that single file with `rsync --checksum` (no deletion, nothing else
  touched); post-sync hashes match
  (`70f20309f915e55da383de0ae1a2bb4062af2b368370bdba4d98307a79a26186`).
  `uv.lock` (local-only) and `logs/`, `scripts/render_figs.sh`,
  `scripts/rerun_behavior.sh` (cluster-only, unrelated) were preserved
  untouched on both sides.

**Step 3 (robustness) - complete and validated.**

- Submitted `sbatch --array=0-191%8 scripts/thura2012_robustness_array.sh
  tokens.toml 1` -> parent array `55386493` (192 tasks: 32 subjects x 6
  configurations - `baseline`, `tau_100ms`, `tau_300ms`, `solver_20ms`,
  `post_horizon_evidence_zero`, `expanded_bounds`).
- At Karim's request the throttle was raised live, without cancelling the
  job, via `scontrol update ArrayTaskThrottle=192 JobId=55386493`, removing
  the self-imposed concurrency limit early in the run.
- All 192 tasks finished `COMPLETED`, exit `0:0` (elapsed 4:16-22:41). No
  task failed or needed resubmission.
- Per-file validation passed: 32 subjects x 6 configurations x 12 rows
  (3 conditions x 4 models) = exactly 2,304 rows, no missing files, no
  `fit_error` rows, identical `source_tree_sha256`
  (`25475454af3be95035f9a7021d7557ee4fcfcb0c7dd816d3ce2858240f1cf997`) and
  `git_commit` (`df783f5...`) across all 192 sidecars, `n_starts=[1]` as
  submitted.
- Aggregate job `55428328`
  (`sbatch scripts/thura2012_robustness_aggregate.sh tokens.toml`) completed
  exit `0`, elapsed `1:33`. Merged outputs validated: `ssmrobustness` 2,304
  rows; `ssmrobustnessstats` 132 rows, covering both an `ssm_robustness_summary`
  view (`delta_bic` per configuration/condition/model) and an
  `ssm_robustness_paired_sensitivity` view (`delta_bic_change_vs_baseline`,
  the actual subject-paired sensitivity test against the baseline
  configuration).
- **Result: the primary comparison is robust across every configuration.**
  For `condition=all`: `urgency` beats `ddm` by delta-BIC -230 to -247 across
  all six configurations (p<1e-8, 30/32 subjects favoring urgency in every
  configuration); `collapsing` beats `ddm` by -169 to -170 (p~1e-11, 31/32
  subjects in every configuration - and identical to machine precision across
  `baseline`/`tau_100ms`/`tau_300ms`, exactly as expected since the
  collapsing-bound model does not use the filter time constant, a useful
  internal-consistency check). `additive_urgency` is never significant in any
  configuration (p=0.09-0.54, 14-18/32 subjects - near chance throughout),
  consistent with the held-out result. The paired-vs-baseline table shows a
  few statistically detectable shifts under some configurations (e.g.
  `urgency` under `tau_300ms`, p=4.3e-6) but they are small in magnitude
  (single-digit delta-BIC points) against effects of ~150-250 points, i.e.
  real but scientifically negligible. `expanded_bounds` in particular does
  not change which models win, even though many individual urgency-scale/
  urgency-onset point estimates sit near the original parameter bound -
  model *selection* is robust to that even though (per the recovery audit
  above) the individual parameter values are only weakly identified.
- Do not sync code or resync the source tree while any exclusion-sensitivity
  task (step 4, next) remains non-terminal, for the same frozen-source-tree
  reason recovery and robustness required it.

**Step 4 (strict-exclusion sensitivity) - complete and validated.**

- Before submitting, re-checked `git status --short --branch` on both sides:
  the fitting-relevant files (`sequential_sampling.py`, `thura2012.py`, both
  exclusion shell scripts) were already identical by `sha256sum`; only the
  report-layer F-number/key/title fixes and this handoff doc had diverged, so
  those three files were synced with `rsync --checksum` (verified matching
  hashes after) before submission.
- Submitted `sbatch --array=0-31%8 scripts/thura2012_exclusion_array.sh
  tokens.toml` -> parent array `55431692` (32 tasks, one per subject, strict
  mask = primary eligibility + `token_log_rows == 15` + valid design
  alignment; frozen census 10,961 strict trials vs. 16,324 primary trials, 32
  subjects, 64 Fast/Slow cells, minimum 56 trials/cell, no cell below 50).
- All 32 tasks finished `COMPLETED`, exit `0:0`. No task failed.
- Per-file validation passed: 32 subjects x 12 rows (3 conditions x 4 models)
  = exactly 384 rows in `ssmexclusioncompletetokenlogalignment`, no missing
  files, no `fit_error` rows, identical `source_tree_sha256`
  (`ab1168542cf8afb28b9d01c739651f7d209acd19ab5b6a66dc9784aca436de58`) and
  `git_commit` (`df783f5...`) across all 32 sidecars, `n_starts=(2,)` as
  submitted.
- Aggregate job `55450044`
  (`sbatch scripts/thura2012_exclusion_aggregate.sh tokens.toml`) completed
  exit `0`, elapsed `1:40`. Merged outputs validated: `ssmexclusionrefit` 384
  rows; `ssmexclusionrobustnessstats` 9 rows (3 conditions x 3 candidate
  models), confirming `mask_equivalent=True` and the exact frozen census
  (`n_complete_token_log`/`n_alignment_valid`/`n_mask_intersection` = 10,961,
  `n_mask_symmetric_difference` = 0) - no duplicate refit was needed, as
  expected. All 32/32 subjects converged in every row.
- **Result: direction holds, magnitude shrinks, and a real audit item
  surfaces.** `urgency` and `collapsing` still strongly beat `ddm` under the
  strict subset; `additive_urgency` is still never significant (p=.08-.52
  across conditions, unchanged qualitative picture). But the strict-subset
  delta-BIC is measurably *less negative* than the full-population
  delta-BIC (`strict_delta_bic_minus_primary_delta_bic`): urgency's
  separation from `ddm` shrinks by ~62-73 points (`condition=all`: mean
  61.5, p=4.6e-8, cohens_dz=1.27; similar in fast/slow), collapsing's by
  ~40-48 points (`condition=all`: mean 39.7, p=4.0e-9, cohens_dz=1.43) -
  real, well-powered shifts (not a mere trend), plausibly reflecting the
  ~33% smaller trial count changing the BIC complexity penalty and/or
  per-subject variance. Not enough to overturn the headline comparison, but
  not negligible either, unlike the robustness-configuration shifts in step
  3 which were small relative to their own effect size. **Boundary-hit
  rates are also notably high under this smaller, stricter subset**: 91-100%
  across urgency/collapsing/additive_urgency (`n_boundary_hit`/`n_subjects`
  in the stats table), well above what the primary/robustness fits showed.
  Step 5's audit must treat this as a real finding to reconcile, not a
  clean pass to wave through.

## Next figure F16 instructions

F16 is `reversecorrelation-kernel`, not a second mechanistic-fit figure. Build
it only after the corrected derivatives and F27 are audited. Use the canonical
`reversecorrelation`/`reversecorrelationstats` tables, preserve the
correct-target stimulus frame, and do not infer mechanism from the kernel
alone. Plot subject-balanced token-jump weights with within-subject uncertainty
and an explicit Fast-minus-Slow contrast; show unseen-after-commitment coding
and the number of eligible subjects/trials. Keep the first-order chosen-target
SumLogLR criterion (F14) distinct from the 2012 process model.

Persist the inferential table before plotting, regenerate PDF/PNG/JSON sidecar,
render the PDF at high resolution, and inspect panel alignment, zero line,
legend, token labels, and uncertainty. Add tests for token-0/unseen coding,
subject balancing, and sidecar columns. Do not reuse pooled SSM time-course
fields or call F16 evidence that additive/multiplicative urgency is uniquely
identified.
