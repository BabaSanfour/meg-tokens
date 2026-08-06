# Behavioral Metrics: Readiness Review and Analysis Roadmap

Scope of this document:

1. **Readiness of `meg_tokens/behavior/metrics.py`** relative to the TDMS/ingestion
   work just completed (`docs/behavior_qc_report.md`).
2. **Fidelity to the source preprint** — Thiery, Rainville, Cisek & Jerbi (2022),
   *Distinct trajectories in low-dimensional neural oscillation state space track
   dynamic decision-making in humans*, bioRxiv
   [10.1101/2022.06.14.494674](https://doi.org/10.1101/2022.06.14.494674)
   (referred to below as **the preprint**), including new empirical findings from
   the real dataset.
3. **A prioritised remediation plan.**
4. **A researched roadmap of additional behavioural analyses**, grounded in the
   dynamic-decision-making literature in monkeys and humans.

Everything in §2's *Empirical findings* subsection was measured on the real local
dataset: 256 Fast/Slow runs, **16,324** started-and-chosen trials, 32 subjects.

---

## 0. Verdict at a glance

| Area | State | Blocking for the metrics phase? |
|---|---|---|
| Consistency with the new started-trial contract | **Good** — `analyze_behavior` filters `OUTCOME_NEVER_STARTED`, `nChoiceMade > 0` is respected throughout | No |
| Trial-class assignment (`sTrialClass`) | **Implemented** — designed labels are preserved and only random trials use the deterministic correct-target rule | No |
| Success probability at decision (SPD) | **Implemented with sensitivity analysis** — logged SPD is reported for all trials and separately for validated 15-row logs | No |
| Statistical design | **Wrong level** — within-subject tests on pooled trials, where the preprint uses group-level paired tests across subjects | **Yes** |
| Fast-vs-slow error rates | **Absent** — the preprint's speed–accuracy result is not reproduced | **Yes** |
| Trial-level feature export | **Absent** — only one summary row per subject; nothing for MEG to join on | **Yes** |
| Subject exclusions (N=32 → N=28) | **Absent** — no exclusion list exists | Yes (for replication) |
| Reporting layer | **Mislabelled** — plots raw RT under a "Decision Time" title | No, but misleading |
| Test coverage | 97 focused behavior, TDMS, SPD, and downstream compatibility tests pass; real-data validation covers all 16,324 started-and-chosen trials | No |

**Bottom line.** T0-1 and T0-2 are implemented. The parser now preserves designed
labels, classifies only random trials from the complete correct-target profile,
and records classification provenance. Logged chosen-target SPD is reported for
all runtime logs and for the validated 15-row-only sensitivity set; design-derived
time alignment is blocked for short logs. The remaining replication blockers are
the group-level statistics, condition-specific error rates, trial-level feature
table, and subject exclusions in T0-3 through T0-6.

---

## 1. Readiness relative to the completed TDMS/behavior work

### 1.1 What the TDMS work changed, and how `metrics.py` fared

`docs/behavior_qc_report.md` changed four things that touch behavioural metrics:

| Change | Effect on `metrics.py` | Status |
|---|---|---|
| `nOutcome` became a required column; `OUTCOME_NEVER_STARTED` (7003) rows excluded from analysis | Handled *outside* `metrics.py`, in `workflows/behavior.py::_started_condition_runs` | Correct, but see §1.2 |
| `rawRT` / `isCorrect` set to `NaN` on no-choice trials | `metrics.py` never reads these columns — it recomputes `tEnterTarget - tGO` inline and re-derives correctness | Works, but duplicated logic (§1.2) |
| Relaxed `nTrialIndex` validation (session-scoped counter) | No impact | Fine |
| `sTrialClass` classification and provenance updated | Designed labels are preserved; only `'x'` trials use the deterministic correct-target rule; RT trials remain not applicable | Correct |

The QC report's before/after replay confirmed that motor baseline, mean Fast/Slow
DT, percent correct, correct/error DT and all three trial-class means were
**identical** for every subject, because those functions already required
`nChoiceMade > 0`. Only post-error sequencing shifted (16 subjects' post-correct
means, 5 subjects' post-error means, largest shift ≈ 9.9 ms). That is the expected
and correct outcome.

**So: no regressions.** The metrics layer survived the ingestion hardening intact.

### 1.2 Where `metrics.py` has *not* kept up with the new contract

These are consistency debts, not bugs that change current numbers:

1. **Duplicated derivation.** `tdms.py::add_run_metadata` now produces canonical
   `rawRT` and `isCorrect` columns with correct NaN semantics. `metrics.py` ignores
   both and recomputes `(tEnterTarget - tGO)` and `nChoiceMade == nCorrectChoice`
   in five separate places (`calculate_motor_baseline`,
   `calculate_decision_times`, `analyze_trial_classes`, `compare_correct_error`,
   `analyze_post_error_slowing`). Two sources of truth for the same quantity is
   exactly the pattern the QC work removed elsewhere.

2. **Never-started filtering lives in the wrong layer.** `metrics.py` functions
   accept raw DataFrames and trust the caller to have removed `nOutcome == 7003`
   rows. `workflows/behavior.py` does this correctly; `reports/behavior_summary.py`
   **does not** — it reads behavior TSVs directly and never filters `nOutcome`.
   A future caller will get this wrong. The filter belongs behind a single
   `started_trials(df)` helper.

3. **Silent zero-returns.** Every function returns `0.0` for an empty input
   (`calculate_motor_baseline` returns `0.0` if there are no RT trials;
   `compare_fast_slow` returns `t_stat=0.0, p_value=1.0` when a group has ≤1
   trial). A subject with a missing RT run therefore gets `motor_baseline_ms = 0`
   and silently inflated decision times across every downstream column, with no
   error and no flag. This directly contradicts the "no silent defaults" principle
   the TDMS work established. It should raise, or emit `NaN` plus an explicit
   count column.

4. **Dead parameters.** `compare_fast_slow(..., n_permutations=1000)` and
   `compare_correct_error(..., n_permutations=1000)` both accept and ignore
   `n_permutations` — a leftover from the legacy permutation approach. Tests still
   pass it (`tests/test_behavior_analysis.py`), which makes it look load-bearing.

5. **`reports/behavior.py` is not on the tested path in this environment.**
   `seaborn` is declared in `pyproject.toml` but is not installed in `.venv`, so
   `tests/test_behavior_plotting.py` fails at import. The 97 focused behavior,
   TDMS, SPD, and downstream compatibility tests pass.

---

## 2. Fidelity to the 2022 preprint

### 2.1 Task and variable definitions (preprint, Methods)

| Element | Preprint definition | Repo |
|---|---|---|
| Tokens | 15, jumping one-by-one every **200 ms** (predecision) | Implicit only |
| Post-decision interval | **50 ms** (fast blocks) / **150 ms** (slow blocks) | Not represented |
| Blocks | 8 blocks, alternating fast/slow, 5 min each | `condition` ∈ {Fast, Slow}; block order not modelled |
| RT task | Two 2-min blocks, before and after; all 15 tokens jump at once ("GO") | `condition == "RT"` ✔ |
| Reaction time | Button press relative to **first token jump** | `tEnterTarget - tGO` ✔ |
| Motor baseline | "the subject's mean reaction time from the visuo-motor reaction-time task" | `calculate_motor_baseline` ✔ (per subject) |
| **Decision time (DT)** | `RT_tokens − mean RT_visuomotor` | `calculate_decision_times` ✔ |
| **Success probability p(t)** | Equation 1 (below) | Implemented for designed and logged profiles |
| **SP at decision (SPD)** | p(t) evaluated at DT | Logged SPD implemented; paired all-log and 15-row-only summaries |
| Trial classes | Easy / ambiguous / misleading, a posteriori from SP | `sTrialClass`, from logged `nProb` — see §2.3 |
| Subjects | 32 recruited, **28 analysed** (2 head movement >20 mm, 1 myographic artifact, 1 system interruption) | **No exclusion list** |

**Equation 1** (preprint), the probability that the right target ultimately wins
given `NR` tokens right, `NL` left, `NC` still in the centre:

```
                    N_C!      min(N_C, 7−N_L)        1
p(R | NR, NL, NC) = ----  ·        Σ           ─────────────────
                    2^N_C          k=0          k! (N_C − k)!
```

I implemented this and validated it against the dataset: it reproduces the
LabVIEW-logged `nProb` **exactly** (to 4 decimal places, all 15 values) on 7,382
trials. The equation and its interpretation are confirmed.

### 2.2 Behavioural results the preprint reports, and whether the repo can produce them

All preprint statistics are **paired t-tests across subjects (N=28)**, on
per-subject means.

| Preprint result | Reported value | Repo status |
|---|---|---|
| DT easy vs ambiguous | 1028 ± 59 vs 1405 ± 74 ms; t = −15.04, p = 1.19e-14 | Means computed per subject; **no test, no group aggregation** |
| DT easy vs misleading | 1028 ± 59 vs 1433 ± 79 ms; t = −13.1, p = 3.25e-13 | Same |
| DT ambiguous vs misleading | t = −1.84, p = 0.077 (n.s.) | Same |
| DT fast vs slow blocks | 1166 ± 71 vs 1293 ± 68 ms; t = −6.08, p = 1.71e-6 | `compare_fast_slow` runs a **Welch independent** test on **pooled trials within one subject** — wrong unit of analysis and wrong test |
| Errors fast vs slow | 45.1 ± 9.9 vs 36 ± 6; t = 6.1, p = 1.7e-6 | **Not computed at all** — `percent_correct` pools Fast+Slow |
| SP at decision, by trial class (Fig 1E) | Cumulative distributions; higher in easy | **Not computed at all** |

Three structural gaps follow:

- **No group level exists.** `analyze_behavior` writes one row per subject and
  stops. Nothing in the codebase aggregates across subjects or runs a paired test,
  so none of the preprint's six headline numbers can currently be produced.
- **Statistics are at the trial/run level within subject.** `compare_fast_slow`
  pools every Fast trial against every Slow trial for one subject and runs Welch's
  t-test. With ~350 trials per subject this yields tiny p-values that describe
  trial-to-trial variance, not the subject-level effect the preprint tests. The
  docstring justifies this as avoiding the legacy downsampling — reasonable as far
  as it goes, but it does not substitute for the group-level paired test.
- **`analyze_trial_classes` runs no statistics at all** — it returns three means.
  The preprint's central behavioural claim (easy < ambiguous ≈ misleading) is
  precisely the comparison that is missing.

Additionally, **post-error slowing is not a preprint analysis.** It comes from the
legacy notebook `44_Behavior_post_error_slowing.ipynb`. Keeping it is fine, but it
should not be presented as replication, and its current form has a known
methodological weakness (§4, item 14).

### 2.3 Historical `nProb` and `sTrialClass` concern — superseded

> **Superseded in full.** T0-1 has since been investigated in full; use
> **`docs/t01_nprob_trial_class_investigation.md`** for the current findings.
> The unchanged code used for the preprint applies the thresholds to the
> chosen-referenced `nProb` profile and reproduces the published analysis. For
> this project, the recommended revised definition preserves recorded design
> labels and classifies only `'x'` trials from a complete, correct-target profile
> reconstructed from `sTokenDirs`. Correct-target classification is the method
> adopted here, not a claim about how the preprint code operated. The initial
> review below is retained only as an investigation record and must not be used
> as the current specification.

#### Historical initial review (superseded)

This is the most consequential finding in this review. `sTrialClass` is assigned in
`tdms.py::parse_single_trial` by indexing the logged `nProb` array:

```python
if len(n_prob) >= 8:
    if n_prob[1] > 0.6 and n_prob[4] > 0.75 and n_prob[7] > 0.75:
        s_trial_class = 1   # Easy
    elif n_prob[1] == 0.5 and 0.38 < n_prob[2] < 0.65 and 0.35 < n_prob[4] < 0.65:
        s_trial_class = 2   # Ambiguous
    elif n_prob[2] < 0.4:
        s_trial_class = 3   # Misleading
```

This assumes `n_prob[i]` is the success probability **after `i+1` token jumps**
(so `[1]`→2 jumps, `[2]`→3, `[4]`→5, `[7]`→8), matching the preprint's criteria.
The thresholds themselves match the preprint, with one exception: ambiguous uses
`> 0.38` where the preprint says 0.4.

**What the data show.**

*Finding 1 — the headline proportions look right.*

| Class | Repo (16,324 trials) | Preprint |
|---|---:|---:|
| Easy | 26.40% | 26% |
| Ambiguous | 22.73% | 22% |
| Misleading | 9.72% | 11% |
| Unclassified | 41.15% | ~41% (implied) |

Encouraging — but as the next findings show, this agreement is not evidence that
the per-trial assignment is correct.

*Finding 2 — `nProb` is referenced to the **chosen** target, not a fixed or correct
target.* Verified by exact Equation-1 reconstruction: on every trial that
reconstructs cleanly, the match is to the subject's chosen target (7,382 trials,
exact to 4 dp). Consequence: `nProb` ends at 1.0 on correct trials and 0.0 on
errors, so a criterion like `n_prob[2] < 0.4` ("misleading") preferentially selects
trials the subject got **wrong**. Measured on the clean trials: misleading trials
are only **7.6%** correct, ambiguous 61.0%, easy 91.4%.

This matters scientifically for the revised analysis adopted here. We define
trial class as a property of the **stimulus** — the token sequence — using the
correct-target profile. This is a project-level analytical choice, not an
attribution to the preprint implementation. The chosen-referenced definition
makes "misleading" partly synonymous with "error trial", making DT-by-class,
accuracy-by-class, and MEG class contrasts partly circular.

*Finding 3 — a third of trials log only 14 SP values for a 15-token trial.*

| `len(nProb)` | Trials | Share |
|---|---:|---:|
| 15 | 10,950 | 67.1% |
| 14 | 5,374 | **32.9%** |

In 5,363 of the 5,374 short trials, the **first** logged SP value is already a
two-jump state (0.709 / 0.5 / 0.291 — values that are arithmetically impossible
after a single jump, where only 0.605 and 0.395 can occur). The array is therefore
offset relative to what the classifier assumes, and neither `sTokenDirs` nor the
logged `nTokenDir` column reproduces these arrays at offset 0 or 1.

The consequence is measurable and stark — accuracy within each assigned class:

| Assigned class | `len(nProb)==15` | `len(nProb)==14` |
|---|---:|---:|
| Easy | 91.4% correct | **100.0%** |
| Ambiguous | 61.0% correct | **100.0%** |
| Misleading | 7.6% correct | **96.6%** |

For the short-array trials the class assignment is degenerate: it separates
essentially nothing. Roughly a third of all trials carry a trial-class label that
does not mean what the remaining two thirds' labels mean — and these trials are
pooled together in `analyze_trial_classes` and in every downstream MEG contrast
that conditions on `sTrialClass`.

*Finding 4 — for the 15-value trials, the classification window is essentially
sound.* Divergence between logged `nProb` and Equation 1:

| | Trials | Share of len-15 |
|---|---:|---:|
| Matches Equation 1 exactly, all 15 values | 7,382 | 67.4% |
| Diverges only at/after the button press | 2,605 | 23.8% |
| Diverges before the decision | 963 | 8.8% |
| …of which within the classification window (indices ≤ 7) | **22** | **0.2%** |

Post-decision divergence is expected — the remaining tokens accelerate once a
choice is made. The important number is the last row: only 22 of 10,950 long-array
trials have a corrupted classification window. **The 15-value trials are fine; the
14-value trials are the problem.**

*Finding 5 — `sTokenDirs` is a clean, authoritative fallback.* Across all 16,324
trials: always exactly 15 characters, only `'1'`/`'2'`, and its majority equals
`nCorrectChoice` in **16,324 / 16,324** cases. It is a sound basis for recomputing
a complete 15-point SP profile for every trial under an explicit reference
convention.

**Caveat on Finding 5.** `sTokenDirs` reproduces the logged `nProb` on exactly the
same 7,382 trials as the logged `nTokenDir` column does, and fails on the same
ones — i.e. the two orderings agree with each other. That is consistent with
`sTokenDirs` being the true temporal jump order and the short arrays being the
defective record, but it does not *prove* it. Before adopting recomputation as the
fix, one run's short-array trials should be checked against MEG token-jump
triggers to confirm the token order and the missing jump. That check is scoped in
§3, T0-1.

---

## 3. Prioritised remediation plan

### Tier 0 — required before the metrics phase can be trusted

**T0-1. Resolve the `nProb` / `sTrialClass` defect — implemented.**
See `docs/t01_nprob_trial_class_investigation.md` for the complete validation.
Outcome: a scientific-notation parsing bug was found and **fixed** (870 trials);
Equation 1 now reconciles 100% of trials; MEG carries no per-token triggers so it
cannot arbitrate token order. The unchanged preprint code is confirmed to use
the chosen-referenced profile. The revised method recommended here uses a full
correct-target profile reconstructed from `sTokenDirs`, preserves recorded
designed labels, and classifies only `'x'` trials. Its SP(2/3/5/8/11) rule
recovers 100% of designed labels deterministically. The 14-row classification
issue is handled; only the cause of the one-step `nTokenDir`/`nProb` offset
remains unknown.

**T0-2. Implement success probability and SPD — implemented with safeguards.**
Equation 1 and per-trial SP profiles are tested. Logged chosen-target `SPD` is
the last `nProb` value in effect at decision time. Subject summaries always
report all logged trials and the validated 15-row-only sensitivity set.
Design-derived time alignment and SPD are blocked for 14-row runtime logs.

**T0-3. Add a group-level analysis stage.**
`analyze_behavior` currently ends at one row per subject. Add a group stage
computing the preprint's paired t-tests across subjects: DT easy/ambiguous/
misleading (3 pairwise), DT fast vs slow, errors fast vs slow, SPD by class.
Report t, p, df, mean ± SEM, and effect size. Add a regression test asserting the
published values are recovered within tolerance on the real dataset — that is the
only real proof of replication.

**T0-4. Compute errors per block condition.**
`percent_correct` currently pools Fast and Slow. Split it, and add per-condition
error counts to reproduce Fig 1H.

**T0-5. Emit a trial-level feature table.**
One row per trial: subject, condition, run, block index, trial index, class,
choice side, correctness, `rawRT`, `DT`, `SPD`, `SumLogLR` at decision, plus the
never-started/QC flags. This is the join key for every brain-behaviour analysis
downstream (Stage 8's `--correlate-behavior` currently has nothing better than
subject-mean `rawRT` to work with) and is the single highest-leverage artifact in
this plan.

**T0-6. Add the subject exclusion list.**
The preprint analyses 28 of 32 subjects. Encode the four exclusions (2 head
movement > 20 mm, 1 myographic artifact, 1 interrupted session) in config with
per-subject reasons, as `behavior_ignore_files` already does for scratch files.
Without this the group numbers cannot match the paper.

### Tier 1 — consistency and correctness cleanups

- **T1-1.** Make `metrics.py` consume canonical `rawRT` / `isCorrect` instead of
  recomputing them in five places.
- **T1-2.** Move never-started filtering behind one `started_trials(df)` helper and
  apply it in `reports/behavior_summary.py`, which currently omits it.
- **T1-3.** Replace silent `0.0` returns with explicit errors or `NaN` + count
  columns, especially `calculate_motor_baseline`.
- **T1-4.** Delete the unused `n_permutations` parameters and their test arguments.
- **T1-5.** Fix `reports/behavior_summary.py`, which plots `rawRT` under the title
  "Decision Time Distribution" and correlates subject-mean `rawRT` with neural
  peaks while calling it "Behavioral Decision Time". Subtract the motor baseline
  or retitle.
- **T1-6.** Decide the ambiguous-class lower threshold: code uses `0.38`, preprint
  says `0.4`. Document whichever is chosen and why.
- **T1-7.** Install `seaborn` (already a declared dependency) so
  `tests/test_behavior_plotting.py` collects.
- **T1-8.** Define DT outlier policy. `DT = rawRT − baseline` can be negative;
  nothing currently guards against anticipations or extreme values.

---

## 4. Roadmap: additional behavioural analyses

Researched against the dynamic-decision-making literature in monkeys and humans —
principally the tokens-task line of work from the Cisek lab (Cisek, Puskas &
El-Murr 2009; Thura et al. 2012; Thura & Cisek 2014, 2017; Carland, Thura & Cisek
2019), which is the direct lineage of this dataset, plus standard practice in
human sequential-sampling research.

### Tier A — simple, high value, low effort

**1. Group-level paired statistics.** (= T0-3.) The preprint's own analysis.

**2. Errors and accuracy by block condition.** (= T0-4.) Preprint Fig 1H.

**3. SP at decision time, by trial class.** (= T0-2, summary implemented.) Logged
chosen-target SPD counts and means are produced for easy, ambiguous, and
misleading trials using both all logs and the validated 15-row-only subset.
Preprint Fig 1E cumulative-distribution plots and group-level inference remain
to be added.

**4. Full DT distributions rather than means.** Report per-subject quantiles
(vincentised across subjects), skewness, and optionally an ex-Gaussian fit (μ, σ,
τ). Preprint Fig 1F shows a single subject's distribution; the mean alone hides the
right-tail changes that speed–accuracy manipulations mainly produce.

**5. Crossed condition × class breakdown.** DT and accuracy for all six
block × class cells. The preprint collapses over the other factor in each figure;
the interaction (does the fast/slow effect differ by difficulty?) is not reported
anywhere and is cheap to obtain.

**6. Choice-side bias and lateral asymmetry.** Proportion of left vs right choices,
and DT difference between them, per subject. A basic QC check that also matters for
MEG: the preprint's PCA groups trials into 12 cells including left/right choice, so
a subject with a strong side bias produces unbalanced cells.

**7. Time-on-task and block-order effects.** DT and accuracy as a function of block
index (1–8) and of trial position within block. Fast/slow blocks alternate, so any
monotonic drift (fatigue, learning) is partly confounded with condition unless
checked.

**8. Anticipations, lapses and outliers.** Count and characterise trials with
negative DT, implausibly long DT, and no-response trials (`nOutcome` 7006/7011,
already parsed but never analysed). Report per subject as a data-quality column.

### Tier B — exploratory

**9. SumLogLR (evidence at decision).** The Cisek-lab evidence measure: the sum of
log-likelihood ratios of individual token jumps in favour of the chosen target,
Σ log[p(eⱼ|C)/p(eⱼ|U)], which is proportional to the difference in token counts
between the two targets. Simple to compute from `sTokenDirs` and a natural
complement to SPD.

**10. The accuracy-criterion decline — the signature urgency analysis.** Group
trials by the number of tokens that jumped before DT, and plot mean SumLogLR (or
SPD) at decision as a function of that. Both monkeys and humans show a **decline**:
later decisions are made on *less* evidence, which is the behavioural fingerprint of
a growing urgency signal (Thura & Cisek 2014, Fig 1F; Cisek et al. 2009). This is
the single most informative analysis missing from the repo. It is now unblocked
by the completed T0-2 work and is the natural behavioural anchor for the preprint's claim that subcortical
high-frequency components track speed–accuracy policy.

**11. Psychophysical reverse correlation / temporal weighting kernel.** Regress the
subject's choice on the direction of each individual token jump (logistic
regression, one weight per jump index). The resulting kernel shows whether
evidence early or late in the trial dominates the decision — primacy indicates
integration to a bound, recency indicates leaky integration or urgency gating.
This is a standard discriminator between accumulator models and is a strong
per-subject individual-difference measure. Compare kernels between fast and slow
blocks.

**12. Conditional accuracy functions (CAF).** Accuracy as a function of DT
quantile, per condition. Together with the DT quantiles from A-4 this gives the
full speed–accuracy picture rather than two scalar means, and CAF shape differences
between fast and slow blocks distinguish threshold changes from drift changes.

**13. Regression of DT on continuous evidence strength.** The three-class scheme
discards 41% of trials as unclassified. Regressing DT (and accuracy) on a
continuous early-evidence measure — e.g. SP after 3 jumps, or the SumLogLR slope —
uses every trial and gives a per-subject sensitivity slope.

**14. Post-error slowing, done robustly.** The current implementation compares
post-correct vs post-error trials, which is confounded with slow fluctuations in
performance over the session. The standard robust estimator compares each
post-error trial with the trial *immediately preceding* the same error
(pre-error → post-error), which cancels the fluctuation. Worth switching to, and
worth also splitting by block condition — post-error adjustment under time pressure
is itself an urgency-modulation question.

**15. Sequential and choice-history effects.** Win-stay/lose-shift rates,
autocorrelation of choice side, and the effect of the previous trial's class and
outcome on current DT. Prior-trial effects are an established input to the urgency
signal (the preprint's own introduction cites context- and prior-trial-dependent
urgency adjustment).

**16. Response vigor / movement time.** `tTrialEnd − tEnterTarget` and the
`tGO → tEnterTarget` decomposition give a proxy for movement execution separate
from deliberation. Thura et al. (2014, *J Neurosci*) showed that decision speed and
movement speed covary within subjects and shift together with the speed–accuracy
context — a shared urgency/vigor signal. Testing that covariation here is a direct,
cheap replication of a monkey finding in this human dataset.

**17. Individual differences.** Per-subject SAT adjustment magnitude
(DT_slow − DT_fast), urgency slope from B-10, and evidence sensitivity from B-13,
then correlate these with overall accuracy and with MEG measures. The preprint
correlates behaviour with neural peaks at the subject level; richer per-subject
behavioural parameters make that correlation far more informative than mean DT.

### Tier C — advanced / model-based

**18. Urgency-gating model vs drift-diffusion model.** The central theoretical
question in this literature: does evidence get integrated to a fixed bound (DDM),
or low-pass filtered and multiplied by a growing urgency signal (UGM)? The tokens
task was designed specifically to dissociate them, because the evidence is
non-stationary. Fitting both per subject and comparing (WAIC/LOO) would let this
dataset speak to its own premise — and the preprint's discussion interprets its
neural results in UGM terms without a behavioural model fit to lean on.

**19. Hierarchical Bayesian sequential-sampling models.** Use **HSSM**
(lnccbrown/HSSM, built on PyMC + Bambi) — it is the maintained successor to HDDM,
supports mixed within/between-subject effects, collapsing bounds, and likelihood
approximation for non-analytic models. `pyddm` is a lighter alternative for
custom-bound models. Hierarchical estimation is the right call here: ~350 trials
per subject is thin for per-subject maximum-likelihood fits.

**20. Explicit urgency-signal extraction.** Thura et al. (2012) derive a
per-subject urgency function from the relationship between evidence at decision and
decision time. This yields an interpretable per-subject urgency slope and intercept
per block condition — directly comparable to the monkey values in the literature,
and the natural behavioural regressor for the preprint's speed–accuracy neural
components.

**21. Mixed-effects models instead of pooled or paired t-tests.** A
`DT ~ class * condition + (class * condition | subject)` LMM uses all trials,
respects the nesting, handles the unbalanced class counts that the a-posteriori
classification produces, and estimates the interaction that A-5 asks for. Keep the
paired t-tests for direct comparability with the published numbers; add the LMM as
the primary inferential model.

**22. Model-based MEG regressors — the payoff.** T0-2 is complete; once T0-5 is
also done, every
trial carries time-resolved SP(t), SumLogLR(t), and (from C-20) urgency(t). These
become single-trial parametric regressors for source-level power, letting you ask
where and when the brain tracks evidence versus urgency, rather than only
contrasting three discrete trial classes. The preprint correlates *condition-mean*
PC trajectories with *condition-mean* SP profiles (R = 0.78 for alpha PC3); a
single-trial parametric analysis is strictly more powerful and is the obvious
extension of the published work.

**23. Monkey–human comparison as an explicit framing.** The behavioural results in
this dataset were designed to replicate Cisek et al. (2009) in humans and Thura &
Cisek (2014) in monkeys — the preprint says so directly. Reporting the same
quantities the monkey papers report (DT by class, SPD by class, accuracy-criterion
decline, decision–movement speed covariation) with the same statistics (those
papers use KS tests on trial distributions alongside subject-level tests) makes the
cross-species comparison explicit rather than implied.

---

## 5. Suggested sequencing

1. **Completed: T0-1 and T0-2** — label-preserving deterministic classification,
   SP reconstruction, paired SPD reporting, and 14-row safeguards.
2. **Next: T0-5** — emit the trial-level feature table for behavioral and MEG
   analyses.
3. **T0-3, T0-4, T0-6** — group stage, per-condition errors, exclusions. At this
   point the preprint's six headline numbers become reproducible, and a regression
   test can lock them in.
4. **Tier 1** cleanups alongside the above.
5. **Tier A**, then **B-10 / B-11 / B-16** as the highest-value exploratory
   additions.
6. **Tier C** once the MEG stages need model-based regressors.

---

## References

- Thiery T., Rainville P., Cisek P., Jerbi K. (2022). *Distinct trajectories in
  low-dimensional neural oscillation state space track dynamic decision-making in
  humans.* bioRxiv [10.1101/2022.06.14.494674](https://doi.org/10.1101/2022.06.14.494674)
- Cisek P., Puskas G.A., El-Murr S. (2009). *Decisions in changing conditions: the
  urgency-gating model.* J Neurosci 29(37):11560–11571.
  [jneurosci.org](https://www.jneurosci.org/content/29/37/11560)
- Thura D., Beauregard-Racine J., Fradet C.-W., Cisek P. (2012). *Decision making by
  urgency gating: theory and experimental support.* J Neurophysiol 108:2912–2930.
- Thura D., Cisek P. (2014). *Deliberation and commitment in the premotor and
  primary motor cortex during dynamic decision making.* Neuron 81:1401–1416.
  [PDF](https://www.cisek.org/pavel/Pubs/ThuraCisek2014.pdf)
- Thura D., Cos I., Trung J., Cisek P. (2014). *Context-dependent urgency influences
  speed–accuracy trade-offs in decision-making and movement execution.* J Neurosci
  34(49):16442–16454. [jneurosci.org](https://www.jneurosci.org/content/34/49/16442)
- Thura D., Cisek P. (2017). *The basal ganglia do not select reach targets but
  control the urgency of commitment.* Neuron 95:1160–1170.
- Carland M.A., Thura D., Cisek P. (2019). *The urge to decide and act: implications
  for brain function and dysfunction.* The Neuroscientist.
  [PDF](https://www.cisek.org/pavel/Pubs/CarlandThuraCisek2019.pdf)
- HSSM — Hierarchical Sequential Sampling Modeling.
  [lnccbrown.github.io/HSSM](https://lnccbrown.github.io/HSSM/)

### Reproducing the empirical findings

The Equation-1 implementation, deterministic class rule, SPD safeguards, and
real-data validator are committed in `meg_tokens/behavior/success_probability.py`
and `meg_tokens/validation/spd.py`. Validation covers 256 Fast/Slow runs and
16,324 started-and-chosen trials, excluding the three documented scratch files.
Exposing the validator through a reproducible `meg-tokens behavior qc` command
remains pending.
