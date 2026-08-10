# Behavioral Roadmap Results

Measured results for the analyses described in `docs/behavior.md`
("Analyses"), produced by:

```bash
meg-tokens --config tokens.toml behavior ingest
meg-tokens --config tokens.toml behavior analyze
meg-tokens --config tokens.toml behavior ssm-fit --n-jobs 4
meg-tokens --config tokens.toml behavior characterization
```

Pending merge into `docs/behavior.md` once the full analysis battery
(including the C1/C2 sequential-sampling refit) has been run to completion
on all 32 subjects; kept separate for now since these numbers are still
partial.

All values below are **N = 32 with no subject exclusions applied**
(`subject_exclusions = []`), covering 16,324 started-and-chosen task trials.
Subject exclusions are not yet populated (`docs/meg.md`, Subject Exclusion),
so these are not the N = 28 preprint-comparison numbers; every table
regenerates under an exclusion list without code changes.
Group tests are two-stage: one fit per subject, then a test across subjects.
C2 is the exception: its group step is a normal population model over the
subject-level fits. The C1–C2 sequential-sampling fits need the optional
`pyddm` dependency (`pip install -e .[modeling]`); they are the runtime of
Stage 2b and are a separate command, `behavior ssm-fit`, which the
characterization pools. `scripts/ssm_fit_array.sh` runs them as a cluster
array job, one subject per task on four CPUs.

Each section names the derivative that holds the full table
(`sub-group/beh/sub-group_task-tokens_desc-<name>_beh.tsv`).

## A1 — SPD distributions by class (`spdcumulative`)

Cumulative distribution of logged chosen-target SPD, pooled proportion at or
below each threshold, for the all-logged view:

| Threshold | Easy (n=4,098) | Ambiguous (n=3,483) | Misleading (n=1,718) |
|---|---:|---:|---:|
| 0.40 | 0.009 | 0.076 | 0.063 |
| 0.60 | 0.046 | 0.286 | 0.247 |
| 0.80 | 0.439 | 0.780 | 0.861 |

Easy decisions are made at much higher success probability; misleading
decisions are the most tightly concentrated below 0.8. The 15-row sensitivity
view is in the same table under `view = validated_15row`.

## A2 — DT distributions (`dtdistribution`, `dtdistributionstats`)

Group means of the per-subject statistics (ms):

| Stratum | Mean | q10 | q50 | q90 | Skew | ex-Gaussian μ | σ | τ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Easy | 1033 | 606 | 968 | 1510 | 1.21 | 698 | 190 | 335 |
| Ambiguous | 1433 | 920 | 1387 | 1971 | 0.73 | 1164 | 292 | 269 |
| Misleading | 1357 | 813 | 1323 | 1958 | 0.46 | 1076 | 328 | 281 |
| Fast | 1186 | 677 | 1127 | 1768 | 0.89 | 857 | 254 | 329 |
| Slow | 1313 | 756 | 1255 | 1916 | 1.05 | 984 | 302 | 330 |

The ex-Gaussian fit converged on all 384 subject × stratum cells.

What the means hide:

- The class effect is a **shift of the whole distribution**, not a tail
  effect: easy is faster than ambiguous at q10 (`t=-9.11`), q50 (`t=-13.29`)
  and q90 (`t=-12.88`).
- **Ambiguous versus misleading separates only in the fast half.** q10
  `t=5.39, p=7e-6` and q50 `t=4.37, p=1e-4`, but q90 `t=0.42, p=0.68`. The
  positive ambiguous-vs-misleading mean difference discussed in
  `docs/behavior.md` ("Preprint Replication") comes entirely from the
  leading edge of the distribution.
- Fast/Slow is also a pure shift: q10, q50, q90 all differ (`p<2e-4`), while
  skewness (`p=0.53`) and τ (`p=0.97`) do not. The speed instruction moves the
  body of the distribution and leaves its shape alone.

## A3 — Condition × class (`conditionclass`, `conditionclassstats`)

Group mean DT (ms) and accuracy per cell:

| Condition | Easy | Ambiguous | Misleading |
|---|---|---|---|
| Fast | 981 / 0.992 | 1367 / 0.767 | 1303 / 0.390 |
| Slow | 1094 / 0.983 | 1517 / 0.779 | 1426 / 0.359 |

Within-subject 2 × 3 ANOVA:

| Measure | Effect | F | df | p | ηp² |
|---|---|---:|---|---:|---:|
| DT | condition | 31.45 | 1, 31 | 3.8e-6 | 0.50 |
| DT | class | 184.45 | 2, 62 | 7.9e-27 | 0.86 |
| DT | condition × class | 1.19 | 2, 62 | 0.31 | 0.04 |
| Accuracy | condition | 1.32 | 1, 31 | 0.26 | 0.04 |
| Accuracy | class | 602.41 | 2, 62 | 2.4e-41 | 0.95 |
| Accuracy | condition × class | 1.94 | 2, 62 | 0.15 | 0.06 |

**No interaction on either measure.** The Fast/Slow instruction adds a roughly
constant amount of decision time to every class, and does not change accuracy
at all. Misleading-trial accuracy is 0.36–0.39, well below chance, which is
what the class definition demands: the early evidence points at the wrong
target.

## A4 — Choice-side bias (`choiceside`, `choicesidestats`)

Left/right choice proportions are balanced (0.508 vs 0.492, `t=1.14, p=0.26`),
and accuracy does not differ by side (`p=0.88`). Left-hand choices are
**22.8 ms faster** than right (`t=-2.39, p=0.023`), carried by the Slow
condition (−31.9 ms, `p=0.050`) rather than Fast (−13.3 ms, `p=0.19`). The
asymmetry is small relative to the 127 ms Fast/Slow effect but should be
carried into any MEG choice-cell contrast.

## A5 — Time on task (`timeontask`, `conditionorder`)

Fitted per subject with block order, within-block position, and a Fast/Slow
indicator in one model:

| Term | Mean | t (df=31) | p |
|---|---:|---:|---:|
| DT per block | −35.9 ms | −4.14 | 2.5e-4 |
| DT per within-block trial | +3.1 ms | 8.00 | 5.0e-9 |

Subjects get **faster across the session** (about 36 ms per block, roughly
290 ms over eight task blocks) and **slower within each block** (about 3 ms
per trial, roughly 180 ms across a 60-trial block). Neither slope differs
between Fast and Slow (`p=0.24`, `p=0.74`). This reads as practice between
blocks against fatigue or disengagement within them.

Condition order is balanced: 15 subjects began with a Fast block and 17 with a
Slow block, and the Fast/Slow adjustment does not differ between those groups
(105 vs 147 ms, Welch `t=-1.00, p=0.33`).

Block order is derived from the LabVIEW session clock (`nInitialTime`); see
`docs/meg.md` for why no other field can do it.

## A6 — Lapses and extreme DTs (`lapses`, `extremedt`, `extremedttrials`)

Only **13 lapse trials** in the whole dataset (started task trials with a go
cue and no choice), a rate of 0.08 %, split 7 Fast / 6 Slow (`p=0.96`). Six
are `7006` (reaction time too long) and seven are `7011` (delay-1 error); no
other outcome code appears among them. 16,337 task trials started, 16,324 of
which produced a choice.

At a 5-MAD robust cutoff, **56 of 16,324 task DTs (0.34 %)** are extreme, in 14
of 32 subjects, all in the slow tail; 5 negative DTs (anticipations) are
retained and flagged, as before. Every flagged trial is listed with its
`trial_id`, condition, run, class, and robust z-score. Nothing is removed: the
DT contract still retains all finite values.

## B1–B2, C3 — Evidence, criterion, and urgency (`criteriondecline`, `urgency`)

Slope of evidence at decision, fitted per subject on both scales:

| Predictor | Scale | Condition | Slope | t (df=31) | p |
|---|---|---|---:|---:|---:|
| tokens observed | probability | all | +0.0049 | 3.67 | 9e-4 |
| tokens observed | log odds | all | +0.100 | 10.80 | 4.9e-12 |
| tokens observed | log odds | Fast | +0.072 | 6.30 | 5.2e-7 |
| tokens observed | log odds | Slow | +0.113 | 9.03 | 3.5e-10 |
| decision time (s) | log odds | all | +0.451 | 9.88 | 4.3e-11 |
| decision time (s) | log odds | Fast | +0.314 | 5.26 | 1.0e-5 |
| decision time (s) | log odds | Slow | +0.498 | 7.86 | 7.1e-9 |

The slope is **positive**, i.e. later commitments happen at *stronger*
evidence, and it is steeper in Slow than in Fast (log-odds per token
`t=-3.25, p=0.003`; per second `t=-2.68, p=0.012`). That is the opposite sign
from the declining accuracy criterion the urgency-gating account predicts, and
the difference deserves care before it is interpreted:

- Success probability moves in **larger steps at later jumps** (the remaining
  token count shrinks), so a subject holding a fixed criterion overshoots it
  more the later they commit. Part of the positive slope is this
  discretization, not a rising criterion.
- The conditional accuracy function (B4 below) falls steeply with DT, which is
  the signature the same account predicts. The two measures disagree here, and
  the criterion slope is the one with the known confound.

A regression that removes the overshoot — for example, fitting the evidence
available at the *previous* jump, or fitting against a model-derived bound —
is the natural next step and is not implemented.

## B3 — Reverse correlation (`reversecorrelation`)

Per-subject logistic weights of each token jump on the eventual choice, with
post-commitment tokens coded as unseen:

| Jump | All | Fast | Slow | Fast − Slow p |
|---|---:|---:|---:|---:|
| 1 | 2.68 | 2.98 | 2.47 | 0.008 |
| 2 | 2.39 | 2.69 | 2.19 | 6e-4 |
| 3 | 2.39 | 2.75 | 2.13 | 2e-4 |
| 4 | 2.07 | 2.23 | 1.86 | 0.002 |
| 5 | 1.75 | 1.99 | 1.50 | 0.033 |
| 6 | 1.76 | 1.71 | 1.56 | 0.23 |
| 7 | 1.67 | 1.71 | 1.75 | 0.70 |
| 8 | 1.26 | 1.53 | 1.46 | 0.69 |

Every weight is positive (`p<6e-7`). The kernel shows clear **primacy**:
weight falls monotonically from jump 1 to jump 8, more than halving. Fast
weights early tokens more heavily than Slow at jumps 1–5 and identically from
jump 6 on — consistent with Fast subjects committing before the later tokens
can matter. The fit converged for all 32 subjects pooled and for 28 of 32 in
each condition separately; the rest are separated data and are reported as
non-converged rather than as large arbitrary weights.

## B4 — Conditional accuracy (`conditionalaccuracy`)

Accuracy across within-subject DT quintiles, pooled over conditions:

| Bin | Mean DT | Accuracy |
|---|---:|---:|
| 1 | 677 ms | 0.897 |
| 2 | 964 ms | 0.872 |
| 3 | 1183 ms | 0.839 |
| 4 | 1444 ms | 0.802 |
| 5 | 1948 ms | 0.776 |

The slope is −0.031 accuracy per bin (`t=-11.07, p=2.7e-12`) and is the same
in Fast (−0.033) and Slow (−0.033). Accuracy falls monotonically with decision
time: slow decisions are not more accurate ones.

## B5 — Continuous evidence (`continuousevidence`)

Over all task trials, including the unclassified random ones that the
threshold rule discards:

| Response | Predictor | Coefficient | t (df=31) | p |
|---|---|---:|---:|---:|
| DT | SP after 3 jumps, strength | −1344 ms per unit | −13.60 | 1.3e-14 |
| Accuracy | SP after 3 jumps, signed | +9.87 log odds per unit | 17.04 | 2.7e-17 |
| DT | log odds after 3 jumps, strength | −269 ms per unit | −13.60 | 1.3e-14 |
| Accuracy | log odds after 3 jumps, signed | +2.39 per unit | 16.73 | 4.6e-17 |

Stronger early evidence produces faster and more accurate decisions, and
neither coefficient differs between Fast and Slow (`p=0.90`, `p=0.51`). The
class analysis and this continuous one agree, so the three-class reduction is
not creating the DT effect.

## B6 — Robust post-error slowing (`posterror`)

| Measure | Mean | t (df=31) | p |
|---|---:|---:|---:|
| Robust: DT(error+1) − DT(error−1) | +143.4 ms | 7.72 | 1.0e-8 |
| Classical: post-error − post-correct | +117.4 ms | 6.55 | 2.6e-7 |

Post-error slowing survives the stricter definition and is in fact slightly
larger under it, so it is not an artifact of comparing against a globally
faster pool of post-correct trials. It does not differ between Fast and Slow
under either definition (`p=0.61`, `p=0.33`).

## B7 — Choice history (`choicehistory`)

| Measure | Value | t (df=31) | p |
|---|---:|---:|---:|
| Win-stay vs lose-stay | 0.498 vs 0.441 | 5.10 | 1.6e-5 |
| Lag-1 side autocorrelation | −0.029 | −2.65 | 0.012 |
| DT after error vs after correct | 1348 vs 1230 ms | 6.55 | 2.6e-7 |
| DT after easy vs after ambiguous | 1139 vs 1304 ms | −14.13 | 4.7e-15 |
| DT after ambiguous vs after misleading | 1304 vs 1266 ms | 1.79 | 0.083 |

Subjects repeat a side about half the time after a correct trial and less than
half after an error: a small but reliable lose-shift bias, matched by a
slightly negative side autocorrelation. The previous trial's class also
predicts the current DT, which is a carry-over effect worth holding on to when
trial-level MEG regressors are built.

## B8 — Response vigor — dropped

Not implemented: the logs record no movement duration. Movement time
(`tEnterTarget - tExitCenter`) is 0 ms on 18,833 of 18,846 chosen trials
because LabVIEW writes both timestamps from the same event, and
`tTrialEnd - tEnterTarget` is the post-choice token replay rather than a
movement (r = -0.98 with DT within condition). See `docs/behavior.md`,
Known Issues (Stage 1), "Movement time not recorded" for the evidence;
nothing in the package computes it.


## B9 — Individual differences (`individualprofile`, `individualcorrelations`)

Across the 32 subjects (Pearson r, uncorrected):

| Measure A | Measure B | r | p |
|---|---|---:|---:|
| mean DT | evidence sensitivity (accuracy log odds) | −0.81 | 1.6e-8 |
| mean DT | criterion slope per token | −0.52 | 0.002 |
| mean DT | urgency slope per second | −0.50 | 0.004 |
| mean DT | percent correct | +0.46 | 0.008 |
| SAT adjustment (Slow − Fast) | — | no correlation reaches p<0.05 | |
| urgency slope | criterion slope | +0.98 | 1.3e-23 |

The dominant axis is a single speed factor: slower subjects are more accurate,
show flatter criterion and urgency slopes, and are *less* sensitive to early
evidence per unit. The urgency/criterion correlation of 0.98 confirms the two
predictors are near-redundant, as expected — decision time and tokens observed
are nearly the same clock. The Fast/Slow adjustment itself is not related to
any other behavioral measure, so it is an independent trait in this cohort.

## C1 — Urgency gating versus bounded integration (`ssmcomparison`, `ssmcomparisonstats`)

**The previously published numbers here are withdrawn.** They came from a
collapsing-bound drift-diffusion model driven by one scalar per trial, which
is not the urgency-gating model and, with within-trial evidence held constant,
is algebraically equivalent to the integrator it was compared against. See the
module docstring of `meg_tokens/behavior/analyses/sequential_sampling.py` for
the model specification.

The corrected fit is pending: it is far more expensive per cell than the
withdrawn one, because each likelihood evaluation solves the diffusion once
per distinct token sequence in the cell (~76) rather than once per evidence
level (4). Measured on one core for H01 Fast (293 trials, 76 sequences): 172 s
for the integrator and 1,700 s for the urgency model, so the 192 fits are
about 50 core-hours — a cluster job, run with `--n-jobs`.

Two things to check once it lands:

- **B2's positive accuracy-criterion slope already disagrees with the
  urgency-gating prediction of a declining criterion** (see B1–B2 above), but
  that measure has a known discretization confound. If this fit says urgency
  gating wins decisively, that tension is the thing to explain, and the
  model-derived criterion here is the tool for it, since it is not subject to
  the same confound.
- `tau` (the low-pass filter time constant) is fixed at 200 ms, Cisek et al.
  2009's value for this task, not fitted. A sensitivity refit at the other
  published values, 100 ms (Thura 2012) and 250 ms (Carland 2015), is a
  one-constant change worth doing once the main run exists.

## C2 — Population parameters (`ssmpopulation`, `ssmpopulationstats`)

Pending with C1, and withdrawn on the same grounds. The fitted quantities
change with the model: the urgency fit now reports `drift_scale`,
`urgency_scale` (threshold over urgency slope, criterion-seconds -- smaller
means urgency rises faster), `urgency_onset_s`, and `nondecision_s`; the
integrator reports `drift_scale`, `bound`, and `nondecision_s`.

## C6 — Comparison-ready statistics (`speciescomparison`)

The `speciescomparison` table itself carries no citation column — each row
is only `measure` plus the descriptive/one-sample statistics. This table is
the canonical mapping from `measure` to the published analysis it aligns
with.

| Measure | Value | Comparable published measure |
|---|---|---|
| DT easy / ambiguous / misleading | 1033 / 1433 / 1357 ms | DT by class (Cisek et al. 2009) |
| SP at decision, easy / ambiguous / misleading | 0.791 / 0.663 / 0.650 | SP at decision by class (Thura et al. 2012) |
| Criterion slope | +0.100 log odds per token | evidence at commitment (Thura et al. 2012) |
| `urgency_minus_integrator_bic` | pending with C1 | urgency gating preferred over integration (Cisek et al. 2009, Fig. 7; Thura et al. 2012, Figs. 10-11) |
| `urgency_scale_criterion_seconds` | pending with C1 | linear urgency signal fitted to monkey tokens-task behavior (Thura and Cisek 2014; Carland et al. 2019) |
| `urgency_scale_fast_minus_slow` | pending with C1 | urgency time-course differs between fast and slow blocks (Cisek et al. 2009, Fig. 8C-D) |

Values from the monkey papers are not reproduced here; the table exists so
that our side is reported in the same statistics. The movement-duration and
decision/movement-covariation comparisons in that literature have no row: the
logs do not record movement (B8).

## What is not implemented

- **Tier C5 join**: the trial-level regressors exist and carry the MEG join
  key; the source-space features they join to do not exist yet.
- **Tier B8**: dropped. Not implementable from these logs, as above.
