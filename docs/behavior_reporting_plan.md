# Behavioral Reporting Plan

Implementation plan for the figure layer of the behavioral battery. It is
self-contained: every figure names the derivative and columns it reads, every
file to create or delete is named, and every statistic printed on a figure is
cited to an existing column. Nothing here requires re-deriving the analysis
code.

Measured values quoted throughout come from `docs/behavior_roadmap_results.md`
(N = 32, `subject_exclusions = []`); they are stated so the implementer knows
what each panel has to make legible, not to be hard-coded anywhere.

## 1. Why this exists

The analysis battery (`docs/behavior.md`, "Analyses") writes 40 group
derivatives. Exactly two figures are reachable from the CLI, and the primary
one is wrong:

| Problem | Where | Consequence |
| :--- | :--- | :--- |
| Plots `rawRT`, not `dt_ms` | `reports/behavior_summary.run_behavior_plotting` | The only Fast/Slow figure shows a variable no analysis in the battery uses. `dt_ms = rawRT - motor_baseline_ms` is the canonical decision time (`docs/data_contract.md`, Stage 2). |
| Pools trials across subjects | same | The group density is a mixture weighted by trial count per subject; it cannot support the paired test that is actually reported. |
| Two plotting functions never called | `reports/behavior.plot_trial_class_distributions`, `reports/behavior.plot_comparison_bars` | Dead code with passing unit tests. |
| Bars + SEM for paired data | `plot_comparison_bars` | Hides the within-subject pairing that every contrast in the battery is built on. |
| No statistical annotation anywhere | all | Every `*stats` derivative already carries `mean, sem, t, p, df, cohens_dz`; none of it reaches a figure. |
| No shared palette or style | all | Hard-coded `'orange'`/`'blue'`, `'green'`/`'blue'`/`'red'`, `'lime'`/`'red'`; not colorblind-safe, not consistent, not print-calibrated. |
| No figure for the headline result | — | H1 (urgency gating vs. bounded integration, ΔBIC = −238.9, t(31) = −8.12, p = 3.6e-9) and H2 (`urgency_scale` Fast − Slow = −0.108, t(31) = −2.52, p = 0.017) have no visual at all. |

## 2. Rules that hold for every figure

These are the invariants the implementer must not break. They are cheap to
state and expensive to retrofit.

1. **The plotting layer never computes an inferential statistic.** Any `t`,
   `p`, `F`, `df`, `dz`, `ηp²`, `r` printed on a figure is read from a
   derivative. The plotting layer may compute only display quantities: means,
   SEMs, within-subject confidence intervals, kernel densities, binned means,
   and quantiles of already-persisted subject-level values. This is what keeps
   a figure from ever disagreeing with `docs/behavior_roadmap_results.md`.
2. **Decision time is `dt_ms`, never `rawRT`.** `rawRT` appears in exactly one
   place in the figure set: nowhere. The motor baseline is already subtracted
   in Stage 2.
3. **Trial-level layers select `primary_analysis_eligible == True` and finite
   `dt_ms`.** That is the same selection the analyses used. Any deviation is
   recorded in the sidecar.
4. **`unclassified` trials are excluded from class-keyed panels** and the
   exclusion is stated in the caption text and sidecar. They are 43.0 % of
   trials; silently dropping them is not acceptable, and neither is drawing
   them as a fourth class.
5. **Case normalisation.** `trialfeatures.condition` is `Fast`/`Slow`/`RT`;
   every group derivative uses `fast`/`slow`. Normalise with `.str.lower()` at
   every join. `RT`-condition rows never enter a behavioral figure.
6. **Repeated-measures contrasts are drawn as paired/slope plots, not bars.**
   Justification in §4.3.
7. **Identity is never carried by colour alone.** Every multi-series panel has
   a legend, and ≤ 4 series are additionally direct-labelled or distinguished
   by marker/line style.
8. **No multiplicity correction is applied, anywhere**, matching
   `docs/data_contract.md` ("Contrasts are never chosen from observed results
   and receive no multiplicity correction"). Every sidecar records
   `"multiplicity_correction": "none"`, and panels with more than three
   contrasts state it in the annotation block.
9. **Non-convergence is shown, not hidden.** Where a derivative carries a
   `converged` column (`criteriondecline`, `urgency`, `reversecorrelation`,
   `continuousevidence`, `timeontask`, `ssmcomparison`), non-converged rows are
   excluded from the drawn estimate and the retained *n* is printed on the
   panel.

---

## 3. Figure inventory

26 figures. Every result family in `docs/behavior.md` has a disposition here;
§3.9 lists the families that get no figure, with reasons.

Naming: each figure has a key `<analysis>-<view>`, where `<analysis>` is the
derivative that supplies its statistics. That key becomes the `desc` entity
(§5) so a figure file names its own source.

### 3.1 Headline — sequential sampling (`behavior/analyses/sequential_sampling.py`)

#### F01 `ssmcomparison-deltabic` — H1: urgency gating beats bounded integration

*Reads*
- `ssmcomparison`: `subject`, `condition`, `model`, `bic`, `delta_bic`,
  `n_trials`, `n_token_sequences`, `converged`. Filter `model == "urgency"`
  for the per-subject ΔBIC (`delta_bic` on a `ddm` row is 0 by construction —
  the criterion is always differenced against the integrator fit).
- `ssmcomparisonstats`: `analysis == "ssm_model_comparison"`,
  `criterion == "bic"`, one row per `condition`; columns `mean`, `sem`, `t`,
  `p`, `df`, `cohens_dz`, `n_subjects`, `n_subjects_favoring_urgency`,
  `n_subjects_favoring_ddm`.

*Layout* — two panels, double-column width.

- **A. Per-subject ΔBIC.** Horizontal ordered lollipop: one row per subject,
  sorted by `delta_bic`; x = ΔBIC (urgency − integrator); vertical reference
  line at 0; a shaded half-plane label ("urgency preferred" left / "integrator
  preferred" right). Group mean ± 95 % CI drawn as a separate summary marker
  below the subject rows, visually separated by a rule.
- **B. BIC_urgency vs. BIC_integrator scatter** with the identity line: one
  point per subject, `bic` from the two `model` rows joined on
  `subject`; points below the identity line are urgency wins. Log-scaled axes
  are unnecessary (BIC spans ~400–3000); use a shared linear scale and force
  equal aspect so the identity line is at 45°.

*Chart-type justification.* A group bar of mean ΔBIC would communicate one
number and conceal the two things a reviewer will ask for: how many subjects
individually prefer each model (30/2) and whether the group mean is driven by
a tail. Ordered per-subject dot plots plus an identity-line scatter are the
standard model-comparison presentation and answer both directly. Facet by
`condition` only in a supplementary variant; the main figure uses
`condition == "all"` with `fast`/`slow` reported in the annotation.

*Annotation.* Panel A, upper-left block:
`ΔBIC = −238.9 ± 29.4, t(31) = −8.12, p < .001, dz = −1.44` plus
`urgency preferred in 30/32 subjects`; then two shorter lines for `fast`
(−131.9, t = −7.85) and `slow` (−117.5, t = −8.49). All values read from the
stats rows, never recomputed.

#### F02 `ssmcomparison-urgencyscale` — H2: urgency scale differs Fast vs. Slow

*Reads*
- `ssmcomparison`: `model == "urgency"`, `condition in {fast, slow}`;
  `subject`, `urgency_scale`, `urgency_scale_se`, `converged`.
- `ssmcomparisonstats`: `analysis == "ssm_urgency_condition_contrast"`,
  `parameter == "urgency_scale"`; columns `mean_a` (fast), `sem_a`, `mean_b`
  (slow), `sem_b`, `mean_difference`, `t`, `p`, `df`, `cohens_dz`,
  `n_subjects`.

*Layout* — Gardner-Altman estimation plot (two axes sharing a row, width ratio
2:1).

- **Left axis:** paired slope plot. Two x positions (Fast, Slow); one thin grey
  line per subject connecting its two `urgency_scale` values; markers coloured
  by condition; group mean ± within-subject 95 % CI overlaid in ink.
  y-label: `urgency_scale (criterion-seconds; smaller = urgency rises faster)`.
- **Right axis:** the paired difference (Fast − Slow), one point per subject
  as a jittered strip, with the group mean and its 95 % CI as a gapped
  vertical bar, aligned to a zero reference tied to the Slow mean on the left
  axis.

*Chart-type justification.* dz = −0.45 with heavily overlapping marginal
distributions: a bar chart of the two means with SEM would look like no effect.
The signal is the consistency of within-subject direction, which only a paired
plot shows, and the estimation-plot difference axis is the current
publication norm for a within-subject contrast (Ho et al. 2019, *Nat Methods*)
because it puts the effect size and its uncertainty on the page rather than a
significance verdict.

*Annotation.* On the difference axis:
`Fast − Slow = −0.108, t(31) = −2.52, p = .017, dz = −0.45 (n = 32)`.
Note in the caption text that smaller `urgency_scale` means faster-growing
urgency, so the negative difference means **urgency grows faster in Fast
blocks** — the sign is easy to misread and must be spelled out.

#### F03 `ssmcomparison-urgencyparams` — all four urgency parameters

*Reads* same two tables; `parameter in {drift_scale, urgency_scale,
urgency_onset_s, nondecision_s}`.

*Layout* — 1 × 4 small multiples, each a compact paired slope panel (no
difference axis; the F02 treatment is reserved for the tested hypothesis).
Each panel annotated with its own `t`, `p`, `dz` from the matching stats row.
`drift_scale` (t = 5.03, p = 2.0e-5) and `urgency_scale` are significant;
`urgency_onset_s` (p = .88) and `nondecision_s` (p = .84) are nulls and are
annotated `n.s.` with the actual t and p, never left blank.

*Justification for including the nulls.* The claim that the Fast/Slow
manipulation acts on urgency growth rather than on non-decision time or
urgency onset is only supported if the other parameters are shown not to move.

### 3.2 Corrected core distributions (`behavior/analyses/distributions.py`)

#### F04 `dtdistribution-condition` — Fast vs. Slow decision time

Replaces the current (wrong-variable, pooled) Fast/Slow KDE.

*Reads*
- `trialfeatures`: `subject`, `condition`, `dt_ms`,
  `primary_analysis_eligible` — trial-level density layer only.
- `dtdistribution`: `stratum_type == "condition"`, `stratum in {fast, slow}`;
  `subject`, `mean`, `q10`, `q25`, `q50`, `q75`, `q90`, `n_trials`.
- `dtdistributionstats`: `stratum_type == "condition"`,
  `contrast == "fast_vs_slow"`, `metric in {q10, q50, q90, skewness}`.
- `groupstats`: `analysis == "decision_time"`, `contrast == "fast_vs_slow"`
  for the mean-DT contrast (1186 ± 70 vs. 1313 ± 66, t = −6.19).

*Layout* — two panels.

- **A. Vincentized quantile functions.** x = group mean of the subject-level
  quantile values; y = quantile probability (0.10, 0.25, 0.50, 0.75, 0.90);
  one line + markers per condition with horizontal SEM bars at each quantile.
  A trial-pooled KDE of `dt_ms` per condition is drawn behind at low alpha as
  a shape reference only, explicitly labelled "pooled trials" in the legend so
  it is not mistaken for the group estimate.
- **B. Paired subject means** using `dtdistribution.mean` for
  `stratum in {fast, slow}`, one line per subject, colored by direction
  (with vs. against the group shift) rather than the Gardner-Altman
  companion-axis construction originally planned here — panel A already
  carries the distributional/shape story, so B's job narrowed to just the
  paired location contrast.

*Chart-type justification.* Averaging subject quantile functions
(vincentizing) is the standard way to summarise RT distributions at the group
level; pooling trials produces a mixture dominated by high-trial-count
subjects and matches no test that is reported. Panel A also makes the actual
finding legible — q10, q50, and q90 all differ (p < 2e-4) with skewness
unchanged (p = .53) — which a pair of overlaid densities does not.
(Whether that pattern is an additive shift or a multiplicative rescaling
is under revision; see docs/behavior.md, F04.)

*Annotation.* Per-quantile significance markers above panel A's q10/q50/q90
positions; a shape-null note (skew) on panel A; a bracket directly on
panel B's Fast-Slow comparison with the mean contrast, t, marker, and dz —
same bracket-on-the-comparison idiom as F05 panel A, not a floating text
block.

#### F05 `dtdistribution-class` — decision time by trial class

Replaces the dead `plot_trial_class_distributions`.

*Reads*
- `trialfeatures`: `subject`, `trial_class_name`, `dt_ms`,
  `primary_analysis_eligible` (exclude `unclassified`).
- `dtdistribution`: `stratum_type == "class"`, `stratum in {easy, ambiguous,
  misleading}`; `subject`, `mean`, `q10`…`q90`.
- `dtdistributionstats`: `stratum_type == "class"`, `contrast in
  {easy_vs_ambiguous, easy_vs_misleading, ambiguous_vs_misleading}`.
- `groupstats`: `analysis == "decision_time"`, the three class contrasts.

*Layout* — two panels.

- **A. Raincloud of subject mean DT per class**: half-violin of the 32 subject
  means + individual points + a thin box (median/IQR), one column per class in
  the fixed order easy → ambiguous → misleading. Subject points connected
  across classes by faint lines (the design is within-subject).
- **B. Vincentized quantile functions per class**, same construction as F04A.

*Chart-type justification.* Panel B is not optional decoration: the
ambiguous-vs-misleading result — the one that reverses the preprint's sign —
lives entirely in the leading edge (q10 t = 5.39, q50 t = 4.37, q90 t = 0.42,
p = .68). A mean-only presentation would show a difference whose location the
reader cannot check. A violin of *subject means* (not trials) is honest at
n = 32; a violin of pooled trials would not be.

*Annotation.* Three significance brackets on panel A (easy–ambiguous,
easy–misleading, ambiguous–misleading) with `Δ` and a significance marker —
same convention as F04 panel B, settled on after the full-`t`/`p` text
proved too wide for a bracket spanning just one category. No pooled-trial
KDE or caption on panel B in the shipped version: panel A's violins already
carry the distribution-shape story, so B stays to the quantile functions
alone, zoomed to 500-2000 ms.

### 3.3 Design effects (`behavior/analyses/design_effects.py`)

#### F12 `lapses-quality` — lapses and extreme decision times (supplementary/QC)

*Reads*
- `lapses`: `subject`, `condition`, `n_started_trials`, `n_lapse_trials`,
  `lapse_rate`, `n_outcome_7006_reaction_time_too_long`,
  `n_outcome_7011_delay_1_error`, `n_lapse_other_outcomes`.
- `lapsestats`: `measure == "lapse_rate"`, `condition in {all, fast, slow,
  fast_vs_slow}`.
- `extremedt`: `subject`, `n_dt_trials`, `median_dt_ms`, `mad_dt_ms`,
  `n_extreme_dt`, `n_extreme_slow`, `n_extreme_fast`, `n_negative_dt`,
  `max_dt_ms`, `min_dt_ms`.
- `extremedttrials`: `subject`, `trial_id`, `condition`, `run`,
  `run_trial_index`, `trial_class_name`, `dt_ms`, `robust_z`, `nOutcome`.

*Layout* — three panels.

- **A.** Per-subject lapse census: horizontal dot plot of `n_lapse_trials`
  (integer axis), subjects on y, split by outcome code as two marker shapes
  (7006 / 7011). Most subjects are 0.
- **B.** Per-subject `n_extreme_dt`, with `n_extreme_slow` / `n_extreme_fast`
  as a paired marker and `n_negative_dt` overlaid as a distinct marker.
- **C.** Every flagged trial: strip of `robust_z` by subject, with the 5-MAD
  cutoff as a vertical rule and negative-DT (anticipation) trials marked
  separately.

*Chart-type justification.* A histogram of a 0.08 % lapse rate communicates
nothing; the useful object is a per-subject census that lets a reviewer see
that 13 trials in 16,337 are concentrated nowhere in particular. Panel C is a
census too — 56 trials — and each one is individually inspectable, which is
the point of the "flag, never remove" policy.

*Annotation.* Panel A: `13 lapse trials / 16,337 started (0.08 %), 7 Fast /
6 Slow, p = .96`. Panel C: `56 / 16,324 (0.34 %) at 5 MAD, in 14 of 32
subjects; nothing removed`.

#### F14 `criteriondecline-tokens` — evidence at decision vs. tokens observed

*Reads*
- `criteriondecline`: `subject`, `condition`, `predictor ==
  "decision_token_index"`, `response in {logged_spd, logged_spd_log_odds}`,
  `n_trials`, `intercept`, `slope`, `slope_se`, `converged`.
- `criteriondeclinestats`: `term in {intercept, slope}` × `response` ×
  `condition in {all, fast, slow, fast_vs_slow}`.
- `trialfeatures`: `decision_token_index`, `logged_spd`,
  `logged_spd_log_odds`, `condition`, `primary_analysis_eligible` — for the
  binned observed overlay.

*Layout* — 2 × 2. Rows = response scale (probability, log odds).

- **Left column:** per-subject fitted lines (`intercept + slope · x`) drawn
  over the observed range of `decision_token_index` (0–15) in faint grey, the
  group mean line in ink, and binned observed means (one point per integer
  token index, ± SEM across subjects) as markers.
- **Right column:** `slope` strip plot vs. zero at three x positions (all,
  fast, slow) with Fast/Slow subject connectors; group mean ± 95 % CI.
  Non-converged subjects excluded, retained n printed.

*Chart-type justification.* Fitted lines reconstructed from persisted
`intercept`/`slope` are exactly the model that was tested; overlaying binned
observed means is the standard check that the linear fit is not the whole
story. Drawing per-subject lines rather than a single group regression makes
the two-stage inference visible.

*Annotation.* Right column: `+0.100 log odds per token, t(31) = 10.80,
p < .001` (log-odds row, `condition == "all"`), `+0.0049 per token,
t = 3.67, p < .001` (probability row), and the Fast-vs-Slow paired contrast
(t = −3.25, p = .003). **Caption text must carry the discretization caveat**
from `docs/behavior_roadmap_results.md` (B1–B2): SP moves in larger steps at
later jumps, so part of the positive slope is overshoot, not a rising
criterion. Put this string in the figure caption and in the sidecar under
`"caveat"`; a positive slope drawn without it invites the opposite
interpretation of the urgency account.

#### F15 `urgency-decisiontime` — evidence at decision vs. decision time

Identical construction to F14, reading the `urgency` and `urgencystats`
derivatives (`predictor == "dt_ms"`, same column set; note `urgencystats`
carries `analysis == "criterion_decline"` — filter on `predictor`, not
`analysis`). x-axis is decision time in seconds; the observed overlay is
binned into deciles of `dt_ms`.

*Annotation.* `+0.451 log odds per second, t(31) = 9.88, p < .001`; Fast
+0.314 vs. Slow +0.498, paired t = −2.68, p = .012.

*Why both F14 and F15.* They are two parameterisations of the same criterion
(tokens vs. seconds) and their subject-level slopes correlate at r = 0.98
(F23). Keeping them as separate figures — rather than one merged panel —
matches how the derivatives are written and lets the redundancy be the finding
in F23 instead of an assumption in the figure.

#### F16 `reversecorrelation-kernel` — psychophysical kernel

*Reads*
- `reversecorrelation`: `subject`, `condition`, `jump` (1–8), `n_trials`,
  `n_trials_token_seen`, `logistic_weight`, `mean_signed_direction`,
  `converged`.
- `reversecorrelationstats`: `metric == "logistic_weight"`, `condition in
  {all, fast, slow, fast_vs_slow}`, `jump`, `test in {one_sample_vs_zero,
  paired_t_test}`.

*Layout* — two panels.

- **A.** x = `jump` (1–8), y = group mean `logistic_weight` ± within-subject
  SEM; three lines: `all` in ink, `fast` and `slow` in the condition palette;
  faint per-subject kernels behind for `condition == "all"`. A small row of
  retained n under the x-axis (32 pooled; 28/32 per condition — separated-data
  subjects are reported non-converged, not drawn as large arbitrary weights).
- **B.** The Fast − Slow difference kernel: `mean_difference` per jump with a
  95 % CI band from `sem`, zero reference, significance markers at jumps 1–5.

*Chart-type justification.* `jump` is an ordered variable and the claim is a
monotone decay (primacy: 2.68 → 1.26, more than halving), so a line is right
and a bar chart per jump would obscure the trend. The separate difference
panel is preferable to eight brackets between two lines.

*Annotation.* Panel A: a single line stating that every weight differs from
zero (`all p < 6e-7`) rather than eight markers. Panel B: per-jump markers
from the `paired_t_test` rows, and a caption line reading the pattern —
Fast weights jumps 1–5 more heavily, identical from jump 6, consistent with
Fast subjects committing before later tokens can matter.

#### F17 `conditionalaccuracy-caf` — conditional accuracy function

*Reads*
- `conditionalaccuracy`: `subject`, `condition`, `dt_bin` (1–5), `n_trials`,
  `mean_dt_ms`, `accuracy`.
- `conditionalaccuracystats`: `test == "mean_accuracy"` (per `dt_bin`,
  columns `mean_dt_ms`, `mean`, `sem`, …) and
  `test == "accuracy_slope_across_bins"`, × `condition in {all, fast, slow}`.

*Layout* — one panel. x = group mean of the subject-level `mean_dt_ms` per
bin (**not** the bin index — the real time axis is the informative one and is
already in the table), y = group mean `accuracy` ± within-subject SEM, one
line per condition plus `all` in ink; faint per-subject lines.

*Chart-type justification.* CAFs are conventionally plotted against bin RT in
this literature, which also makes the unequal bin spacing (677 → 1948 ms)
visible; plotting against bin index would linearise it and misrepresent the
decline.

*Annotation.* `slope = −0.031 accuracy per bin, t(31) = −11.07, p < .001`,
plus `Fast −0.033, Slow −0.033` on one line. Caption: slow decisions are not
more accurate — the signature the urgency account predicts, and the measure
that disagrees with F14's slope sign.

#### F18 `continuousevidence-effects` — continuous early evidence

*Reads*
- `continuousevidence`: `subject`, `condition`, `predictor in
  {sp_design_early, sum_log_lr_design_early}`, `n_dt_trials`,
  `dt_intercept_ms`, `dt_slope_ms_per_unit`, `n_accuracy_trials`,
  `accuracy_log_odds_per_unit`, `converged`.
- `continuousevidencestats`: `term in {dt_slope_ms_per_unit,
  accuracy_log_odds_per_unit}` × `predictor` × `condition in {all, fast, slow,
  fast_vs_slow}`.
- `trialfeatures`: `sp_design_early`, `sum_log_lr_design_early`, `dt_ms`,
  `isCorrect`, `condition`, `primary_analysis_eligible` — observed layer.

*Layout* — 2 × 2. Row 1: the observed relationships — (A) chronometric,
binned mean `dt_ms` against deciles of evidence *strength*
(`|sum_log_lr_design_early|`), (B) psychometric, binned proportion correct
against deciles of *signed* evidence, both with per-subject faint lines and a
group mean. Row 2: the fitted coefficients — (C) `dt_slope_ms_per_unit` and
(D) `accuracy_log_odds_per_unit` as strips vs. zero at three x positions with
Fast/Slow connectors.

*Chart-type justification.* Same "data then coefficient" principle as F14. The scientific value of this analysis is that it agrees with the
three-class reduction over *all* trials including the 43 % unclassified, so
the observed panels must be drawn over every task trial — state that
explicitly in the panel subtitle, since it is the one place in the figure set
where `unclassified` trials are included.

*Annotation.* Panels C/D report the `sum_log_lr_design_early` scale
(−269 ms per unit, t = −13.60; +2.39 log odds per unit, t = 16.73) with the
`sp_design_early` scale as a second annotated series or a small inset; plus
the two Fast-vs-Slow nulls (p = .90, p = .51).

### 3.5 Sequential effects (`behavior/analyses/sequential.py`)

#### F19 `posterror-slowing` — robust post-error slowing

*Reads*
- `posterror`: `subject`, `condition`, `n_error_pairs`, `robust_pes_ms`,
  `mean_pre_error_dt_ms`, `mean_post_error_dt_ms`, `n_post_error_trials`,
  `n_post_correct_trials`, `classical_pes_ms`.
- `posterrorstats`: `measure in {robust_pes_ms, classical_pes_ms}` ×
  `condition in {all, fast, slow, fast_vs_slow}`.

*Layout* — two panels.

- **A.** Paired slope + difference axis of `mean_pre_error_dt_ms` vs.
  `mean_post_error_dt_ms` — this *is* the robust definition, drawn as the
  pairing it is.
- **B.** `robust_pes_ms` and `classical_pes_ms` as two adjacent strips vs.
  zero, with a connector per subject between the two definitions.

*Chart-type justification.* Panel B's connectors are the whole point: the
claim is that the effect survives the stricter definition and is in fact
slightly larger under it (+143.4 vs. +117.4 ms), which is a within-subject
comparison *between measures* and therefore also paired.

*Annotation.* `robust +143.4 ms, t(31) = 7.72, p < .001`;
`classical +117.4 ms, t(31) = 6.55, p < .001`; Fast/Slow nulls (p = .61,
p = .33).

#### F20 `choicehistory-effects` — choice history

*Reads*
- `choicehistory`: `subject`, `condition`, `n_linked_trials`, `win_stay`,
  `lose_stay`, `lose_shift`, `side_autocorrelation_lag1`,
  `mean_dt_after_correct_ms`, `mean_dt_after_error_ms`,
  `mean_dt_after_easy_ms`, `mean_dt_after_ambiguous_ms`,
  `mean_dt_after_misleading_ms`.
- `choicehistorystats`: the six `measure` values
  (`win_stay_vs_lose_stay`, `side_autocorrelation_lag1`,
  `dt_after_error_vs_correct`, `dt_after_easy_vs_ambiguous`,
  `dt_after_easy_vs_misleading`, `dt_after_ambiguous_vs_misleading`) ×
  `condition`.

*Layout* — 2 × 2.

- **A.** Paired slope `win_stay` vs. `lose_stay` with a 0.5 reference and a
  difference axis.
- **B.** `side_autocorrelation_lag1` strip vs. zero.
- **C.** Paired slope `mean_dt_after_correct_ms` vs.
  `mean_dt_after_error_ms`.
- **D.** DT by previous-trial class: x = previous class (easy, ambiguous,
  misleading), one faint line per subject, group mean ± within-subject 95 % CI,
  class palette on the markers.

*Annotation.* A: `0.498 vs 0.441, t = 5.10, p < .001`. B: `−0.029,
t = −2.65, p = .012`. C: `1348 vs 1230 ms, t = 6.55, p < .001`. D: the two
tested contrasts (easy vs ambiguous t = −14.13, p < .001; ambiguous vs
misleading t = 1.79, p = .083 → `n.s.`).

### 3.6 Model internals (`behavior/analyses/sequential_sampling.py`)

#### F21 `ssmtimecourse-fit` — what the two models actually do

*Reads* `ssmtimecourse`: `subject`, `condition`, `model`, `trial_class`,
`n_trials`, `time_s`, `criterion`, `mean_decision_variable`,
`predicted_density_correct`, `predicted_density_error`,
`observed_density_correct`, `observed_density_error`.

**Implementation warning — de-duplicate before averaging.** The criterion does
not depend on `trial_class` and the observed densities do not depend on
`model`; both are repeated across rows (`docs/data_contract.md`, Stage 2b). A
naive `groupby(["condition", "model", "time_s"]).mean()` weights those
duplicates by the number of trial classes. Before averaging:
`criterion` → `drop_duplicates(["subject", "condition", "model", "time_s"])`;
`observed_density_*` → `drop_duplicates(["subject", "condition",
"trial_class", "time_s"])`.

*Layout* — three panels, `condition == "all"` in the main figure with
`fast`/`slow` as a supplementary variant.

- **A. Criterion time course.** Group mean `criterion` vs. `time_s`, one line
  per `model` — the flat integrator bound against the hyperbolically declining
  urgency criterion. This is the visual definition of the model contrast and
  the natural companion to F01.
- **B. Predicted vs. observed DT densities.** Observed
  (`observed_density_correct` up, `observed_density_error` mirrored below the
  axis) as a filled neutral-grey area; predicted densities as coloured lines
  per model on the same axes. Mirrored correct/error densities are the
  standard fit-quality display for a two-boundary accumulator.
- **C. Mean decision variable** (`mean_decision_variable`) under the urgency
  model, one line per `trial_class` in the class palette, with the criterion
  from panel A overlaid dashed. Caption must note that this is the noise-free
  trajectory of the *unabsorbed* process, so it can and does cross the
  criterion — otherwise the crossing reads as a bug.

*Annotation.* No new statistics; a pointer line to F01 for the formal
comparison. Panel B carries the per-condition trial count.

#### F22 `ssmpopulation-shrinkage` — empirical-Bayes population model

*Reads*
- `ssmpopulation`: `subject`, `condition`, `model`, `parameter`, `estimate`,
  `standard_error`, `population_informed_estimate`, `own_data_weight`.
- `ssmpopulationstats`: `condition`, `model`, `parameter`, `n_subjects`,
  `population_mean`, `population_mean_se`, `between_subject_sd`, `z`, `p`.

*Layout* — a forest/caterpillar grid: one panel per `parameter` (urgency
model: `drift_scale`, `urgency_scale`, `urgency_onset_s`, `nondecision_s`;
integrator: `drift_scale`, `bound`, `nondecision_s`), `condition == "all"`.
In each: subjects on y ordered by `estimate`; x = `estimate` with an
`± standard_error` bar; a second, smaller marker at
`population_informed_estimate` with a light connector showing the shrinkage;
a vertical band at `population_mean ± between_subject_sd` and a line at
`population_mean`. Marker opacity or size keyed to `own_data_weight`.

*Chart-type justification.* A shrinkage estimator has no honest bar-chart
representation — the point is the *displacement* of each subject's estimate
toward the population mean in proportion to its own uncertainty, which needs
both values plotted and connected. Forest plots are the standard form.

*Annotation.* Per panel: `population mean ± SE, between-subject SD, z, p,
n = <n_subjects>`. **The `n_subjects` column is below 32 for several
parameters** (e.g. `urgency_scale` n = 15 in `condition == "all"`, driven by
non-finite observed-information standard errors); print it on every panel and
state the reason in the caption. A forest plot silently missing half the
cohort is a serious misread.

### 3.7 Individual differences and cross-species (`behavior/analyses/individual.py`)

#### F23 `individualcorrelations-matrix` — pairwise correlation matrix

*Reads* `individualcorrelations`: `measure_a`, `measure_b`, `n_subjects`,
`pearson_r`, `pearson_p`, `spearman_rho`, `spearman_p`, `df`.

*Layout* — lower-triangular heatmap over the seven measures
(`mean_dt_ms`, `sat_adjustment_ms`, `urgency_slope_per_second`,
`criterion_slope_per_token`, `accuracy_log_odds_per_unit`, `percent_correct`,
`lapse_rate`), diverging blue↔grey↔red ramp centred on r = 0, `pearson_r`
printed in each cell, cells with `pearson_p < .05` outlined.

*Chart-type justification.* A diverging ramp with a neutral midpoint is the
only correct encoding for a signed quantity centred at zero; never a rainbow
or a single-hue ramp here. Lower triangle only — a symmetric matrix drawn in
full doubles the ink for no information.

*Annotation.* An in-figure note: `Pearson r, n = 32, uncorrected` and the
outlining convention. Because these are uncorrected, the matrix must not carry
asterisks — outlining plus the stated convention is the honest treatment.

#### F24 `individualprofile-scatter` — the correlations that matter

*Reads* `individualprofile` (`subject`, `mean_dt_ms`, `percent_correct`,
`sat_adjustment_ms`, `urgency_slope_per_second`, `criterion_slope_per_token`,
`accuracy_log_odds_per_unit`, `dt_slope_ms_per_unit`, `lapse_rate`) for the
points, and `individualcorrelations` for the annotated `pearson_r`/`pearson_p`.

*Layout* — 2 × 3 small multiples: `mean_dt_ms` against
`accuracy_log_odds_per_unit` (r = −0.81), `criterion_slope_per_token`
(−0.52), `urgency_slope_per_second` (−0.50) and `percent_correct` (+0.46);
`urgency_slope_per_second` against `criterion_slope_per_token` (r = 0.98, the
redundancy panel); and `sat_adjustment_ms` against `mean_dt_ms` as the
representative null. Each: points + OLS line + 95 % CI band.

*Chart-type justification.* At n = 32 a correlation matrix alone cannot show
whether an r is outlier-driven; the scatter grid is the required companion,
not a duplicate. The r = 0.98 panel is included specifically to make the
near-redundancy of the two urgency predictors visible rather than asserted.

*Annotation.* `r = …, p = …, n = 32` per panel, read from
`individualcorrelations`. The OLS line and band are display-only; the reported
r is never recomputed.

#### F25 `speciescomparison-forest` — comparison-ready statistics

*Reads* `speciescomparison`: `measure`, `n_subjects`, `mean`, `sem`, `t`, `p`,
`df`, `cohens_dz` (10 rows).

*Layout* — a forest plot faceted by **unit group**, because the measures are
incommensurate and must never share an axis: (i) milliseconds
(`decision_time_easy_ms`, `_ambiguous_ms`, `_misleading_ms`);
(ii) probability (`success_probability_at_decision_*`, three rows);
(iii) log odds per token (`criterion_slope_log_odds_per_token`);
(iv) BIC (`urgency_minus_integrator_bic`);
(v) criterion-seconds (`urgency_scale_criterion_seconds`,
`urgency_scale_fast_minus_slow`). Each row: `mean` with a 95 % CI computed as
`mean ± t.ppf(0.975, df) · sem` (a display quantity, permitted by rule 1). A
right-hand text column names the comparable published measure, taken verbatim
from the C6 mapping table in `docs/behavior_roadmap_results.md` — the
derivative carries no citation column, so the mapping is a constant in the
figure module, not invented per run.

*Explicit prohibition.* Do **not** plot published monkey values. They are not
in any derivative, are not reproduced in code by design
(`docs/behavior.md`), and a forest plot that mixes measured rows with
transcribed literature values invites reading them as a common analysis. The
figure reports our side in the same statistics; the comparison is textual.

*Annotation.* Per row: `t(df) = …, p = …`. Note under the BIC facet that a
one-sample test against zero on a criterion difference is a model-preference
test, not a parameter estimate.

### 3.8 Optional, gated on the MEG join

#### F26 `individualprofile-neural` — behaviour vs. neural metric

The existing `--neural-metrics` scatter, kept but rebuilt: reads
`individualprofile` (subject-level behavioural measures, **not** pooled
`rawRT`) joined to the supplied subject-level MEG table on `subject`, and
drawn with the shared correlation-scatter helper used by F24. Remains gated
on `--neural-metrics` and raises the existing clear error when the table is
absent or lacks the required columns. Correlation r/p for this figure is
computed in the plotting layer — the documented exception to rule 1, because
no derivative holds it yet; the sidecar records
`"statistics_source": "computed_in_report"` so the exception is auditable.
Revisit once the Tier C5 join lands and a derivative owns the statistic.

### 3.9 Result families with no figure — and why

| Family / derivative | Disposition |
| :--- | :--- |
| `trialfeatures` | Not a result. It is the trial-level layer behind F04, F05, F10, F14, F15, F18. |
| `groupstats` | Not a figure of its own; its rows are the annotations on F04, F05, F06. Drawing a table as a chart adds nothing. |
| `extremedttrials` | Folded into F12 panel C; a 56-row table needs no second figure. |
| `ssmtrialpredictions` | **No figure.** It is the trial-level model-derived regressor for the Tier C5 MEG join (`criterion_at_decision`, `decision_variable_at_decision`, `predicted_accuracy`), not a behavioural result. Revisit when the source-space features it joins to exist. |
| `ssmhierarchical`, `ssmhierarchicalstats` | **No figure.** These files exist in the data root but are *not* written by the current workflow (`workflows/behavior_characterization.py` writes `ssmpopulation`/`ssmpopulationstats`). They are stale outputs of a previous name. Do not build a figure against them; flag them for deletion from the data root as a separate cleanup, not from the plotting code. |
| Response vigor (Tier B8) | Dropped upstream — movement time is not recorded (`docs/behavior.md`, Known Issues). Nothing to plot, and nothing must be plotted against the null field. |

---

## 4. Shared infrastructure

### 4.1 File layout

**New files**

| Path | Contents |
| :--- | :--- |
| `meg_tokens/reports/style.py` | Palettes, rcParams, panel geometry. Project-wide, not behavior-specific, so MEG figures can adopt it later. |
| `meg_tokens/reports/annotations.py` | `StatResult`, `stat_from_row`, formatters, `annotate_*` helpers. |
| `meg_tokens/reports/panels.py` | Reusable panel builders that draw onto a caller-supplied `Axes`. |
| `meg_tokens/reports/behavior/__init__.py` | Figure registry, `FigureSpec`, public API. |
| `meg_tokens/reports/behavior/_tables.py` | `BehaviorTableSet` — derivative loader/cache with contract-shaped errors. |
| `meg_tokens/reports/behavior/distributions.py` | F04, F05, F06. |
| `meg_tokens/reports/behavior/design.py` | F08, F09, F10, F11, F12, F13. |
| `meg_tokens/reports/behavior/evidence.py` | F14, F15, F16, F17, F18. |
| `meg_tokens/reports/behavior/sequential.py` | F19, F20. |
| `meg_tokens/reports/behavior/modeling.py` | F01, F02, F03, F21, F22. |
| `meg_tokens/reports/behavior/individual.py` | F23, F24, F25, F26. |

The figure-module names mirror `meg_tokens/behavior/analyses/` one-for-one, so
the figure for an analysis is found by module name.

**Deleted**

- `meg_tokens/reports/behavior.py` — replaced by the package of the same
  import path (`meg_tokens.reports.behavior`), so no import elsewhere breaks
  on the package name itself. All three of its functions are removed:

  | Function | Disposition | Reason |
  | :--- | :--- | :--- |
  | `plot_fast_slow_distributions` | **Remove.** | Plots `rawRT` and pools trials. Superseded by F04, which reads `dt_ms` and keeps the pairing. No caller wants the old semantics. |
  | `plot_trial_class_distributions` | **Remove.** | Never called; superseded by F05. Its pooled tri-KDE cannot show the q10/q90 dissociation that is the actual result. |
  | `plot_comparison_bars` | **Remove.** | Never called; bar + SEM on repeated-measures data is precisely the pattern §4.3 replaces. Superseded by `panels.paired_slope` + `panels.estimation_axis`. |

**Rewritten**

- `meg_tokens/reports/behavior_summary.py` — keeps its module path and its
  entry point name `run_behavior_plotting` (so the CLI import line at
  `meg_tokens/cli/main.py:846` changes only to add `list_behavior_figures`),
  but its body becomes the orchestration layer of §5. Its Stage-1 per-run
  loading (`layout.behavior_tables(...)`, `_behavior_entities`,
  `started_trials`, `rawRT`) is deleted; it now reads Stage 2/2b group
  derivatives.

**Unchanged**

- `meg_tokens/reports/statistics.py` — MEG group statistics, out of scope.
  Its `mean_raw_rt_by_subject` still uses `rawRT`; that is a separate defect
  (it should read `summary.mean_dt_ms`) and is **noted here but not fixed by
  this plan** to keep the change reviewable.
- `meg_tokens/reports/meg.py::plot_correlation` — kept; F24/F26 call it
  through a thin wrapper in `panels.py` that applies the shared style, rather
  than duplicating a scatter implementation.

### 4.2 `reports/style.py`

Palette validated with the `dataviz` skill's `validate_palette.js` in light
mode with `--pairs all`; every group of hues that can co-occur in one panel
passes the lightness band, chroma floor, CVD separation, and normal-vision
floor checks.

```python
CLASS_ORDER: Final[tuple[str, ...]] = ("easy", "ambiguous", "misleading")
CLASS_COLORS: Final[Mapping[str, str]] = {
    "easy":       "#1baf7a",   # aqua
    "ambiguous":  "#2a78d6",   # blue
    "misleading": "#eb6834",   # orange
}
# validate_palette.js "#1baf7a,#2a78d6,#eb6834" --mode light --pairs all
#   CVD ΔE 9.2 (deutan) PASS · normal-vision ΔE 24.0 PASS
#   easy (#1baf7a) is 2.74:1 on white -> relief rule: it always carries a
#   direct label or legend text, never colour alone.

CONDITION_ORDER: Final[tuple[str, ...]] = ("fast", "slow")
CONDITION_COLORS: Final[Mapping[str, str]] = {
    "fast": "#e34948",   # red
    "slow": "#4a3aa7",   # violet
    "all":  "#0b0b0b",   # ink, for the pooled reference line
}
# CVD ΔE 22.7 PASS · normal-vision ΔE 33.6 PASS · both >= 3:1 contrast

MODEL_ORDER: Final[tuple[str, ...]] = ("urgency", "ddm")
MODEL_COLORS: Final[Mapping[str, str]] = {
    "urgency": "#008300",   # green
    "ddm":     "#e87ba4",   # magenta
}
MODEL_LABELS: Final[Mapping[str, str]] = {
    "urgency": "Urgency gating",
    "ddm":     "Bounded integrator",
}
# CVD ΔE 17.6 PASS · normal-vision ΔE 35.9 PASS

OBSERVED: Final[str] = "#52514e"        # observed data behind model fits
SUBJECT_LINE: Final[str] = "#9a9992"    # per-subject traces
SUBJECT_ALPHA: Final[float] = 0.30
INK: Final[str] = "#0b0b0b"
INK_SECONDARY: Final[str] = "#52514e"
INK_MUTED: Final[str] = "#9a9992"
SURFACE: Final[str] = "#ffffff"
DIVERGING: Final[tuple[str, str, str]] = ("#2a78d6", "#f0efec", "#e34948")
```

**Scoping rule (must be documented in the module docstring).** The three
palettes are *scoped*: class hues, condition hues, and model hues are pairwise
disjoint, and no panel mixes two scopes as colour. Where class and condition
both appear (F08), class is on the x-axis and only condition is coloured.
Where model and condition both appear (F21), condition is a facet and only
model is coloured. This is why the three-hue class palette can be validated
independently of the others.

**rcParams** — applied by `apply_publication_style()`, and only ever inside a
`matplotlib.rc_context` in library code so it cannot leak into a caller's
session:

- `font.family: "sans-serif"`, `font.sans-serif: ["DejaVu Sans"]` —
  matplotlib's bundled default. **Do not require an external font**; a figure
  layer that fails on a cluster node without Arial is not shippable.
- `font.size: 8`, `axes.labelsize: 8`, `axes.titlesize: 9`,
  `xtick.labelsize: 7`, `ytick.labelsize: 7`, `legend.fontsize: 7`,
  `axes.titleweight: "bold"`.
- `axes.linewidth: 0.8`, `lines.linewidth: 1.6`, `lines.markersize: 4`,
  `xtick.major.width: 0.8`, `ytick.major.width: 0.8`.
- `axes.spines.top: False`, `axes.spines.right: False`.
- `axes.grid: False` by default; panels that need one call
  `ax.grid(axis="y", color=INK_MUTED, alpha=0.35, linewidth=0.6)`.
- `figure.constrained_layout.use: True`.
- `savefig.dpi: 400`, `savefig.bbox: "tight"`, `savefig.transparent: False`,
  `figure.facecolor: SURFACE`, `axes.facecolor: SURFACE`.
- `pdf.fonttype: 42`, `ps.fonttype: 42`, `svg.fonttype: "none"` — TrueType
  text stays editable in Illustrator/Inkscape. This is a hard journal
  requirement and the single most common omission in matplotlib figure code.
- **No `axes.prop_cycle`.** Colour is always assigned explicitly by entity;
  a cycler would let a filter that changes the series count repaint the
  survivors.
- **Never call `seaborn.set_theme()`.** Seaborn stays a dependency and is used
  only for density primitives (`kdeplot`, `violinplot`) inside `panels.py`,
  always with an explicit `ax=` and `color=`; its global theming would fight
  these rcParams.

**Geometry**

```python
WIDTH_SINGLE_IN: Final[float] = 3.42    # 87 mm, single column
WIDTH_DOUBLE_IN: Final[float] = 7.09    # 180 mm, double column
PANEL_HEIGHT_IN: Final[float] = 2.40

def figure_grid(
    n_rows: int = 1,
    n_cols: int = 1,
    *,
    width: Literal["single", "double"] = "double",
    panel_height_in: float = PANEL_HEIGHT_IN,
    width_ratios: Sequence[float] | None = None,
    height_ratios: Sequence[float] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Create a styled figure and a 2-D array of Axes at publication size."""
```

Panel letters (A, B, C…) are stamped by
`style.panel_label(ax, "A")` at a fixed offset in bold, so every multi-panel
figure labels identically.

### 4.3 `reports/panels.py`

Every helper draws onto a caller-supplied `Axes` and returns `None`; the
figure modules own layout. This keeps the panel library composable and
testable without file I/O.

```python
def paired_slope(
    ax, *,
    values_a: np.ndarray, values_b: np.ndarray,
    label_a: str, label_b: str,
    color_a: str, color_b: str,
    ylabel: str = "",
    subject_line_color: str = style.SUBJECT_LINE,
    show_group_mean: bool = True,
    reference: float | None = None,
) -> None:
    """Two x positions, one line per subject, group mean +/- within-subject CI."""

def estimation_axis(
    ax_diff, *,
    differences: np.ndarray,
    result: StatResult,
    unit: str = "",
    color: str = style.INK,
) -> None:
    """Gardner-Altman difference axis: subject differences, mean, 95% CI, zero rule."""

def subject_strip(
    ax, *,
    groups: Mapping[str, np.ndarray],
    colors: Mapping[str, str],
    reference: float | None = 0.0,
    ylabel: str = "",
    connect: Sequence[tuple[str, str]] = (),
) -> None:
    """Jittered per-subject estimates per group, group mean +/- 95% CI,
    optional subject connectors between named group pairs."""

def group_line(
    ax, *,
    x: np.ndarray,
    subject_matrix: np.ndarray,      # (n_subjects, n_x), NaN-tolerant
    color: str, label: str,
    error: Literal["within_sem", "within_ci95", "none"] = "within_sem",
    draw_subjects: bool = False,
) -> None:
    """Group mean line with a within-subject error band, optional subject traces."""

def quantile_function(
    ax, *,
    probabilities: Sequence[float],
    subject_quantiles: Mapping[str, np.ndarray],   # label -> (n_subjects, n_q)
    colors: Mapping[str, str],
) -> None:
    """Vincentized quantile functions: mean quantile on x, probability on y,
    horizontal SEM bars."""

def raincloud(
    ax, *,
    groups: Mapping[str, np.ndarray],
    colors: Mapping[str, str],
    ylabel: str = "",
    connect_subjects: bool = True,
) -> None:
    """Half-violin + box + individual points per group, subject connectors."""

def forest(
    ax, *,
    labels: Sequence[str],
    centres: np.ndarray, lows: np.ndarray, highs: np.ndarray,
    colors: Sequence[str] | str = style.INK,
    reference: float | None = None,
    secondary: np.ndarray | None = None,     # e.g. shrunk estimates
) -> None:

def correlation_heatmap(
    ax, *,
    r_matrix: pd.DataFrame,
    p_matrix: pd.DataFrame,
    alpha: float = 0.05,
    lower_triangle_only: bool = True,
) -> None:

def scatter_fit(
    ax, *,
    x: np.ndarray, y: np.ndarray,
    xlabel: str, ylabel: str,
    color: str = style.INK,
) -> None:
    """Styled wrapper over reports.meg.plot_correlation; OLS line + 95% band."""

def within_subject_error(
    values: np.ndarray, *, axis: int = 0, kind: Literal["sem", "ci95"] = "sem"
) -> np.ndarray:
    """Cousineau (1994) normalisation with the Morey (2008) bias correction.

    DISPLAY ONLY. Never used to derive a reported statistic; every inferential
    number on a figure comes from a *stats derivative (rule 1).
    """
```

**Why paired/slope over bar + SEM.** Every contrast in the battery is a
within-subject paired test (`behavior/math/inference.paired_subject_statistics`
pairs by subject and drops incomplete pairs). A bar of two group means with
between-subject SEM plots an error term that the test does not use, so a
reliable effect with dz = 0.45 (F02) looks like overlap. A slope plot shows
the quantity the test operates on — the per-subject difference — and the
estimation-plot difference axis shows its magnitude and uncertainty on their
own scale. Bars survive only where counts have a meaningful zero (F13,
where they are stacked per subject).

### 4.4 `reports/annotations.py`

```python
@dataclass(frozen=True)
class StatResult:
    label: str
    n_subjects: int
    mean: float | None          # `mean` or `mean_difference`
    sem: float | None
    t: float | None
    p: float | None
    df: float | None
    cohens_dz: float | None
    mean_a: float | None = None
    mean_b: float | None = None
    extra: Mapping[str, float] = field(default_factory=dict)   # F, eta_p2, r ...

def stat_from_row(row: pd.Series, *, label: str) -> StatResult:
    """Build a StatResult from any battery *stats row.

    Reads the shared column names emitted by
    `behavior/math/inference.one_sample_statistics` and
    `paired_subject_statistics`: n_subjects, mean, sem, mean_a, sem_a, mean_b,
    sem_b, mean_difference, t, p, df, cohens_dz. Prefers `mean`; falls back to
    `mean_difference`. Absent or non-finite fields become None -- never 0, and
    never rendered as "nan".
    """

def significance_marker(p: float | None) -> str:
    """'***' p < .001 | '**' p < .01 | '*' p < .05 | 'n.s.' otherwise |
    '' when p is None or non-finite."""

def format_p(p: float) -> str:
    """'p < .001' below 1e-3; otherwise 'p = .017' -- two significant digits,
    APA leading zero dropped."""

def format_stat(
    result: StatResult, *, unit: str = "", include_effect_size: bool = True
) -> str:
    """'Δ = −108, t(31) = −2.52, p = .017, dz = −0.45'"""

def annotate_contrast(
    ax, *, x_a: float, x_b: float, y: float, result: StatResult,
    unit: str = "", show_text: bool = True,
) -> None:
    """Significance bracket between two x positions: marker plus optional stat text."""

def annotate_stat_block(
    ax, *, lines: Sequence[str], loc: str = "upper left", y: float | None = None
) -> None:
    """Multi-line stat text in ink (never a series colour), in axes coordinates.
    `y` overrides the corner's vertical position for occupied corners."""

def annotate_anova(
    ax, rows: pd.DataFrame, *, loc: str = "upper left", y: float | None = None
) -> None:
    """Render ANOVA rows as compact 'condition ηp² = .51 ***' lines."""

SIGNIFICANCE_CONVENTION: Final[str] = "*** p<.001, ** p<.01, * p<.05, n.s. otherwise"
```

Conventions, enforced by these helpers rather than by discipline:

- Marker thresholds live in exactly one place and the string
  `SIGNIFICANCE_CONVENTION` is written into every figure's sidecar.
- A non-finite `p` yields no marker and no test text — the descriptive part is
  still drawn. Nothing prints `p = nan`.
- Text always uses ink tokens, never a series colour.
- `n.s.` is rendered explicitly wherever a null is part of the claim (F03,
  F16, F18, F19) — an absent annotation and a tested null must not
  look the same. It carries an effect size, not spelled-out `t`/`p`: full
  stat text collapses `constrained_layout` on anything narrower than a
  double-width panel, and the exact values stay in the sidecar.
- Effect sizes (`cohens_dz`, `partial_eta_squared`) are included by default;
  a p-value alone is not sufficient for publication.

### 4.5 `reports/behavior/_tables.py`

```python
@dataclass
class BehaviorTableSet:
    """Lazy, cached access to Stage 2/2b group derivatives for the figure layer."""
    layout: DerivativeLayout
    subjects: tuple[str, ...] | None = None

    def analysis(self, name: str) -> pd.DataFrame:
        """Load `sub-group_task-tokens_desc-<name>_beh.tsv` via
        `DerivativeLayout.behavior_analysis(name)`, applying the subject filter
        when the table has a `subject` column. Cached per name.

        Raises FileNotFoundError naming the exact path and the command that
        writes it ('meg-tokens behavior characterization', or
        'meg-tokens behavior ssm-fit' for any ssm* table), matching the
        project's error convention.
        """

    def trial_features(self) -> pd.DataFrame:
        """`behavior_trial_features()`, filtered to primary_analysis_eligible
        rows with finite dt_ms, condition lower-cased. Cached."""

    def summary(self) -> pd.DataFrame: ...
    def group_statistics(self) -> pd.DataFrame: ...

    @property
    def sources(self) -> dict[str, str]:
        """Every path actually read, for the sidecar."""
```

`trial_features()` uses `behavior.tables.read_trial_features` (the same reader
the workflow uses) so sequence columns deserialize identically.

---

## 5. Output format, naming, sidecars

**Formats.** Two files per figure, same stem:

- `.pdf` — vector, `pdf.fonttype = 42`, the publication artefact.
- `.png` — raster at `dpi = 400`, for quick viewing and for pasting into
  notes/slides.

`--formats` may narrow this (e.g. `--formats .pdf`), and `.svg` is accepted
for figures destined for hand-editing.

**Path.** Follows the existing `run_behavior_plotting` pattern exactly:

```python
figure_path = output_layout.path(
    subject="group",
    datatype="fig",
    description=f"{spec.analysis}-{spec.view}",   # e.g. "ssmcomparison-deltabic"
    suffix="behavior",
    extension=".pdf",
)
# -> <root>/derivatives/sub-group/fig/
#      sub-group_task-tokens_desc-ssmcomparison-deltabic_behavior.pdf
```

`desc` is a hyphen-joined growing tag list per `docs/data_contract.md`; both
segments are lowercase alphanumeric and both are recorded verbatim in the
sidecar (`"analysis"`, `"view"`). The contract's rule — *every value chained
into `desc` is also recorded in the sidecar metadata, and the two must never
disagree* — is enforced by a test (§7).

**Sidecars.** One per figure, written with `save_sidecar`.

> **Note the collision, and use it deliberately:** `io.sidecar_path` calls
> `Path.with_suffix(".json")`, so `…_behavior.pdf` and `…_behavior.png` map to
> the *same* `…_behavior.json`. Write the sidecar once, after all formats, and
> list them in a `formats` field. Do not call `save_sidecar` per format.

Sidecar schema (superset of what the current code writes):

```json
{
  "stage": "behavior_report",
  "figure": "ssmcomparison-deltabic",
  "analysis": "ssmcomparison",
  "view": "deltabic",
  "kind": "per_subject_criterion_difference",
  "title": "Urgency gating versus bounded integration",
  "source_derivatives": ["/abs/.../desc-ssmcomparison_beh.tsv",
                         "/abs/.../desc-ssmcomparisonstats_beh.tsv"],
  "columns_read": {
    "ssmcomparison": ["subject", "condition", "model", "bic", "delta_bic", "converged"],
    "ssmcomparisonstats": ["analysis", "condition", "criterion", "mean", "sem",
                           "t", "p", "df", "cohens_dz", "n_subjects",
                           "n_subjects_favoring_urgency", "n_subjects_favoring_ddm"]
  },
  "statistics_source": "ssmcomparisonstats[analysis=ssm_model_comparison,criterion=bic]",
  "subjects": ["H01", "..."],
  "n_subjects": 32,
  "palette": {"urgency": "#008300", "ddm": "#e87ba4"},
  "significance_convention": "*** p<.001, ** p<.01, * p<.05, n.s. otherwise",
  "multiplicity_correction": "none",
  "caveat": null,
  "formats": [".pdf", ".png"],
  "dpi": 400
}
```

`columns_read` is not decoration: it makes "which variable is this figure
actually showing?" answerable without reading code, which is precisely the
question the `rawRT`/`dt_ms` defect answers wrongly today. A test asserts that
`dt_ms` appears — and `rawRT` does not — in the decision-time figures'
sidecars.

---

## 6. CLI wiring

Extend the existing `report behavior` subcommand. **Do not add one subcommand
per figure group**: 26 figures would mean 26 parsers duplicating the same four
path arguments, and the group boundaries would drift from the registry. A
`--figures` selector over a registry keeps one dispatch path and makes the
figure list discoverable at runtime.

**Parser** — replace `meg_tokens/cli/main.py:356-360` with:

```python
behavior_report = report_commands.add_parser(
    "behavior", help="Render the behavioral figure battery."
)
behavior_report.add_argument("--subjects", nargs="+")
behavior_report.add_argument("--behavior-root")
behavior_report.add_argument("--output-root")
behavior_report.add_argument("--neural-metrics")
behavior_report.add_argument(
    "--figures", nargs="+", default=["all"],
    help=(
        "Figure keys and/or group names to render (default: all). "
        "Groups: headline, core, distributions, design, evidence, sequential, "
        "modeling, individual, qc. Use --list-figures to see every key."
    ),
)
behavior_report.add_argument(
    "--formats", nargs="+", default=[".pdf", ".png"],
    choices=[".pdf", ".png", ".svg"],
)
behavior_report.add_argument(
    "--list-figures", dest="list_figures", action="store_true",
    help="Print the figure registry (key, source derivatives, title) and exit.",
)
behavior_report.add_argument(
    "--skip-missing", dest="skip_missing", action="store_true",
    help="Skip figures whose source derivative is absent instead of raising.",
)
```

**Dispatch** — replace `meg_tokens/cli/main.py:845-852` with:

```python
elif args.domain == "report" and args.report_command == "behavior":
    from meg_tokens.reports.behavior_summary import (
        list_behavior_figures,
        run_behavior_plotting,
    )

    if args.list_figures:
        result = list_behavior_figures()
    else:
        result = run_behavior_plotting(
            str(args.behavior_root or project.bids_root),
            str(args.output_root or project.bids_root),
            subjects_list=args.subjects,
            neural_metrics_csv=args.neural_metrics,
            figures=tuple(args.figures),
            formats=tuple(args.formats),
            skip_missing=args.skip_missing,
        )
```

**Orchestrator signature** (`reports/behavior_summary.py`) — positional
arguments preserved so nothing else has to change:

```python
matplotlib.use("Agg", force=True)   # module level, as in reports/statistics.py

@dataclass(frozen=True)
class FigureSpec:
    key: str                       # "ssmcomparison-deltabic"
    analysis: str                  # desc segment 1 / statistics derivative
    view: str                      # desc segment 2
    title: str
    kind: str
    groups: tuple[str, ...]        # ("headline", "modeling")
    requires: tuple[str, ...]      # derivative names that must exist
    builder: Callable[[BehaviorTableSet], tuple[Figure, dict[str, object]]]
    caveat: str | None = None

def list_behavior_figures() -> list[FigureSpec]: ...

def run_behavior_plotting(
    behavior_dir: str,
    output_figures_dir: str,
    subjects_list: list[str] | None = None,
    neural_metrics_csv: str | None = None,
    *,
    figures: Sequence[str] = ("all",),
    formats: Sequence[str] = (".pdf", ".png"),
    skip_missing: bool = False,
) -> list[Path]:
    """Render the selected behavioral figures and return every file written."""
```

The builder returns `(figure, extra_sidecar_metadata)`; the orchestrator owns
`figure_grid` sizing conventions only insofar as the builder uses them, and
owns all file I/O, sidecar assembly, and `plt.close(figure)`.

Behaviour change to call out in the commit message: `report behavior` now
reads Stage 2/2b **group** derivatives from `--behavior-root`, not Stage 1
per-run tables. A missing derivative raises `FileNotFoundError` naming the
path and `meg-tokens behavior characterization` (or `behavior ssm-fit` for
`ssm*`), unless `--skip-missing`.

---

## 7. Phased build order

Each phase is independently shippable: it ends with figures on disk, tests
green, and a working `--figures` selector.

### Phase 1 — infrastructure + headline + corrected core

*Ships:* `meg-tokens report behavior --figures headline core` → F01, F02, F04,
F05.

1. `reports/style.py`, `reports/annotations.py`, `reports/panels.py`.
2. `reports/behavior/` package: `__init__.py` (registry + `FigureSpec`),
   `_tables.py`, `modeling.py` (F01, F02), `distributions.py` (F04, F05).
3. Delete `reports/behavior.py` and its three functions.
4. Rewrite `reports/behavior_summary.py`.
5. CLI changes (§6).
6. Rewrite `tests/test_behavior_plotting.py` → the `tests/reports/` suite
   (§8); delete the three dead-function tests.

*Why first:* H1/H2 is the paper's central claim and has no figure; the
`rawRT` defect is a correctness bug in the only figure that exists. Everything
after this is additive.

### Phase 2 — remaining distributions and design effects

F03, F06 (`distributions.py`); F08–F13 (`design.py`). Adds the
`distributions`, `design`, and `qc` groups.

### Phase 3 — evidence and criterion

F14–F18 (`evidence.py`). Adds the `evidence` group. Largest phase; the
per-subject-fitted-line + binned-observed construction is shared by F14, F15
and F18, so build F14 first and reuse.

### Phase 4 — sequential effects

F19, F20 (`sequential.py`). Small; depends only on Phase 1 infrastructure.

### Phase 5 — model internals

F21, F22 (`modeling.py`). F21 carries the de-duplication trap (§3.6); F22
carries the reduced-`n_subjects` trap. Both need care, neither blocks anything
else.

### Phase 6 — individual differences, cross-species, neural join

F23–F25 (`individual.py`), then F26 as the restyled, gated neural scatter.
Last because F26's statistic is the one exception to rule 1 and should not set
a precedent before the rest of the layer exists.

### Documentation, at the end of each phase

Append the phase's figures to a single "Figures" table in `docs/behavior.md`
(figure key → derivative(s) → what it shows), so the analysis list and the
figure list stay adjacent. Do not create a second figure-catalogue document.

---

## 8. Testing plan

**Structure.** New directory `tests/reports/` with
`__init__.py`, `factories.py`, and one test module per source module.
`tests/test_behavior_plotting.py` is deleted; its only surviving test — the
end-to-end "reads staged tables and writes a derivative + sidecar" — is
rewritten against group derivatives in
`tests/reports/test_behavior_report_workflow.py`.

**Fixtures.**

- A session-scoped autouse fixture forcing `matplotlib.use("Agg")` and
  closing all figures after each test (`plt.close("all")`), so the suite never
  trips matplotlib's open-figure warning or needs a display.
- `tests/reports/factories.py`: `write_group_derivative(layout, name, frame)`
  plus one builder per derivative that produces a **schema-exact** minimal
  table — the exact column names listed in §3, with 3–4 synthetic subjects.
  Schema exactness is the point: a factory that invents a column would let a
  figure pass its test and fail on real data.

**What is smoke-tested.**

| Target | Assertion |
| :--- | :--- |
| `style` | Every key in `CLASS_ORDER`/`CONDITION_ORDER`/`MODEL_ORDER` has a colour; every value is a 7-character hex; the three palettes are pairwise disjoint (the scoping rule); `apply_publication_style()` sets `pdf.fonttype == 42` and `ps.fonttype == 42`; rcParams are restored after the context exits. |
| `annotations` | Exact strings: `format_p(3.6e-9) == "p < .001"`, `format_p(0.017) == "p = .017"`, `significance_marker(0.083) == "n.s."`, `significance_marker(float("nan")) == ""`. `stat_from_row` on a real-shaped row returns the right fields, prefers `mean` over `mean_difference`, and maps non-finite to `None`. `format_stat` never emits the substring `"nan"`. |
| `panels` | Each helper draws onto a supplied `Axes` and returns `None`; artist counts scale with input (`paired_slope` with 5 subjects adds 5 subject `Line2D`s plus the group line); all-NaN input draws nothing and does not raise; a single subject does not raise. `within_subject_error` matches a hand-computed Cousineau–Morey value on a 3×2 fixture. |
| each figure builder | Returns a `plt.Figure`; the expected number of axes; at least one `ax.texts` entry contains `"p ="` or `"p <"` (the annotation reached the figure); the returned metadata dict names the derivatives it read. |
| decision-time figures specifically | Their `columns_read` metadata contains `"dt_ms"` and does **not** contain `"rawRT"`. This is the regression guard for the defect this plan fixes. |
| class-keyed figures | `unclassified` does not appear in any tick label or legend entry. |
| `ssmtimecourse-fit` | Built from a fixture with duplicated criterion rows across `trial_class`; assert the plotted criterion equals the de-duplicated mean, not the row-weighted mean. This trap is invisible without a dedicated test. |
| orchestrator | `run_behavior_plotting(..., figures=("headline",))` writes `.pdf` + `.png` + exactly one `.json` per figure; the `desc` segments in each filename equal `analysis`/`view` in its sidecar (the `docs/data_contract.md` desc↔sidecar rule); a missing derivative raises `FileNotFoundError` whose message names `behavior characterization`; `skip_missing=True` returns a shorter list without raising; an unknown figure key raises `ValueError` listing the valid keys. |
| CLI | `main.build_parser().parse_args([...])` accepts `--figures`, `--formats`, `--list-figures`, `--skip-missing`; `--formats bogus` exits non-zero. Add alongside the existing CLI parse tests. |
| real data | In `tests/test_behavior_real_data_regression.py`, a test marked to skip when the configured `data_root` is absent: render the full registry against the real derivative tree and assert every registered figure produced a file and a sidecar. This is the only test that exercises real column values, and it is the one that would have caught the current defect. |

**What is not tested.**

- No pixel comparison, no image hashing, no `pytest-mpl` baseline images.
  They are brittle across matplotlib and freetype versions and would make
  every style tweak a baseline regeneration.
- No assertions on exact artist coordinates, tick positions, or layout.
- No assertion that a figure "looks right" — that is a review step, and the
  phase checklist should include opening the rendered PDFs once per phase.
