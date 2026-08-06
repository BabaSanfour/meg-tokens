# Behavioural Metrics & Trial Labelling — Master Record

Consolidated record of the whole investigation: the readiness review of
`meg_tokens/behavior/metrics.py`, the fidelity check against the source
preprint, the `nProb` / `sTrialClass` defect, the code-history question, the
resolution of the short-log trials, and the implemented specification.

**Companion documents**
- `docs/behavior_qc_report.md` — the earlier TDMS/MEG-alignment QC work (unaffected; see §3.3)
- `docs/behavior_metrics_readiness.md` — readiness review and the analysis roadmap (§9 here summarises it)
- `docs/t01_nprob_trial_class_investigation.md` — the T0-1 investigation in detail

**Source preprint.** Thiery T., Rainville P., Cisek P., Jerbi K. (2022), *Distinct
trajectories in low-dimensional neural oscillation state space track dynamic
decision-making in humans*, bioRxiv
[10.1101/2022.06.14.494674](https://doi.org/10.1101/2022.06.14.494674).

**Dataset.** All measurements: 256 Fast/Slow runs, **16,324** started-and-chosen
trials, 32 subjects, plus the raw CTF recordings.

---

## 1. The question we started from

After finishing the TDMS ingestion work, is `metrics.py` ready, and does it
implement what the preprint describes?

Short answer: T0-1 and T0-2 are implemented. The parser preserves designed
labels, classifies only random trials from the complete correct-target profile,
and reports logged SPD using both all runtime logs and a validated 15-row-only
sensitivity set. Group-level statistics and the remaining metrics gaps are
listed in §9.

---

## 2. Two defects, found in order

| | Defect | Severity |
|---|---|---|
| **A** | Probabilities written in scientific notation were silently mis-parsed | Real bug, **zero** effect on any reported number (§3) |
| **B** | The preprint code derives classes from the chosen frame and overwrites some recorded design labels | Reproduced historically; revised analytical definition implemented here (§4–§8) |

---

## 3. Defect A — the parser

### 3.1 Cause

`parse_single_trial` read probabilities with `nProb:\s*([\d\.]+)`. LabVIEW writes
values near 0 and 1 in scientific notation, which breaks that pattern two ways:

| Raw value in log | Old result | Why |
|---|---|---|
| `5.46875E-1` (= 0.546875) | **5.46875** | pattern matches the mantissa, ignores the exponent |
| `-2.22044604925031E-16` (= 0) | *no value at all* | `[\d\.]` cannot match the leading `-` |

The second case is worse: the parser appended a `tTime` for that row but no
`nProb`, silently shortening the array and shifting every later index.

### 3.2 Scale

| | Trials | Share |
|---|---:|---:|
| ≥1 out-of-range `nProb` | 881 | 5.40% |
| Array changed in value | 870 | 5.33% |
| Entry **dropped**, desyncing `nProb` from `tTime` | 11 | 0.07% |

Corruption clusters at array indices 9–10 — late in the trial, where SP nears 0
or 1 and LabVIEW switches notation. Indices 1 and 2, which drive the ambiguous
and misleading tests, were **never** corrupted.

### 3.3 Impact: exactly zero

| Quantity | Change |
|---|---|
| `sTrialClass` | **0 / 16,324 trials** |
| Per-subject mean DT by class | max \|Δ\| = **0.000000 ms**, 0 of 96 cells |
| Trial counts per class | **0** |
| Motor baseline, DT, accuracy, post-error slowing | structurally unaffected — never read `nProb` |

**`docs/behavior_qc_report.md` stands entirely unchanged.** That is partly luck:
the corrupted values sat outside the indices inspected by the historical
override.

It still mattered to fix, because the corrupted column *is* written into the
Stage-1 TSVs and propagates into `epochs.metadata`, and because all of the
success-probability work (T0-2) would have been built on it.

### 3.4 Fix — applied

`meg_tokens/behavior/tdms.py`: sign/exponent-aware pattern; `[0,1]` validation
(tolerating and clamping LabVIEW's `-2.22e-16` zero); `len(tTime) == len(nProb)`
guard. **Verified**: 97 focused behavior/TDMS/SPD/downstream tests pass; the
broader nonplot suite has 231 passes and four unrelated `specparam`/archive
failures. All 320 retained real runs parse cleanly.

### 3.5 It was inherited

The pattern is byte-identical in every commit back to `13e07c6` "feat: add robust
TDMS parser" — the first parser commit, before the refactors. Not introduced by
recent work.

---

## 4. Adopted change 1 — the reference frame

### 4.1 What it means

Success probability is defined **per target**, and the two traces are mirror
images: `p_left(t) = 1 − p_right(t)`. A profile is meaningless until you say
*whose*.

- `nProb` is logged in the **chosen** frame — evidence for the button pressed.
- The unchanged code used for the preprint applies the class thresholds to that
  chosen-referenced profile.
- The implemented analysis classifies a complete **correct-target** profile
  reconstructed from `sTokenDirs`.

Difficulty class is treated as a property of the token sequence rather than of
the participant's response. That is the analytical definition implemented here,
not a claim about how the preprint code operated.

### 4.2 Why the change matters

On correct trials the two frames coincide. On error trials they are mirror
images — so a genuinely misleading stimulus (where the subject *was* misled, i.e.
answered wrongly) has a **high** chosen-referenced SP early and fails the test.

Measured: the unchanged preprint code recovers only **41.3%** of the
designed-misleading trials, and its "misleading" class is **25.8%** correct. The
label therefore depends strongly on the participant's response, making class
contrasts partly circular. Correct-target classification removes that response
dependence.

### 4.3 The runtime conversion is exact

```
sp_correct[j] = sp_chosen[j]      if nChoiceMade == nCorrectChoice
                1 - sp_chosen[j]  otherwise
```

This conversion is available when a correct-referenced runtime analysis is
explicitly required. The reported SPD retains the logged chosen-target frame.
Trial classification instead uses a complete design-time profile reconstructed
from `sTokenDirs`, as specified in §7–§8.

### 4.4 Independent rule evaluation

Rule agreement must be measured without falling back to the recorded
designed label when no rule fires. With `unclassified` as the outcome in that
case:

| Probability source and rule | 14-row designed | 15-row designed | All designed |
|---|---:|---:|---:|
| Logged `nProb`, legacy indices, empirical four-point rule | 40.3% | 91.6% | 76.4% |
| Logged `nProb`, row-aware indices, empirical four-point rule | 38.8% | 91.6% | 75.9% |
| Eq. 1 on `sTokenDirs`, correct-referenced, empirical four-point rule | 95.4% | 91.6% | 92.7% |
| Eq. 1 on `sTokenDirs` plus the SP(11) boundary rule | **100%** | **100%** | **100%** |

The logged profile is therefore not a sound basis for classifying the 14-row
trials. A complete, correct-target profile reconstructed from `sTokenDirs` is
both consistent across logging groups and independently validated by the
designed labels.

### 4.5 The design-derived boundary

At the traditional jumps 2, 3, 5, and 8, one signature occurs in two designed
classes:

```
SP(2)=0.5, SP(3)=0.3872, SP(5)=0.623, SP(8)=0.7734
```

It covers 641 designed-ambiguous and 381 designed-misleading trials, so no
four-point rule can distinguish them. At jump 11 the ambiguous trials have
`SP(11) == 1.0`, while the misleading trials have 0.6875 or 0.9375. The fixed
deterministic rule using SP at jumps 2, 3, 5, 8, and 11 reproduces all 5,224
designed labels.

The SP(11) boundary is derived from this dataset rather than stated in the
preprint and must be reported as such.

---

## 5. Adopted change 2 — preserve recorded design labels

This is the larger and simpler half.

The preprint's Methods text says:

> "We used SP to classify **random trials** into a posteriori classes of trials
> (e.g. 'easy,' 'ambiguous,' and 'misleading' trials) embedded in the sequence"

The retained dataset has 5,224 trials constructed to be
easy/ambiguous/misleading, and LabVIEW **records which** in `sTrialClass`
(`'e'`/`'a'`/`'m'`). The `'x'` trials are random Fast/Slow trials to which the
a-posteriori rule applies. The `'r'` trials belong to the separate RT task and
are not difficulty-classified.

The unchanged code used for the preprint runs the `elif` chain on **every**
trial, so recorded labels get overwritten:

| Overwrite | n |
|---|---:|
| misleading → ambiguous | 783 |
| misleading → easy | 226 |
| ambiguous → misleading | 99 |
| easy → misleading | 11 |
| **total** | **1,119 / 5,224 designed trials (21.4%)** |

### 5.1 Why the recorded labels remain authoritative

The §4.5 deterministic rule independently recovers all designed labels, which validates
the complete `sTokenDirs` representation and the added SP(11) boundary. It does
not create a reason to overwrite information already recorded by LabVIEW.
Preserving the designed label is exact, retains provenance, and avoids making a
data-derived rule part of the definition of its own validation set.

---

## 6. Historical implementation

The unchanged override code was used to produce the preprint results. From
`13e07c6` onward it appeared as:

```python
# Replicate DDM logic rules for sTrialClass override
...
elif n_prob[2] < 0.4:
    s_trial_class = 3  # Fixed bug from previous member codebase (TrialClass -> s_trial_class)
```

The production parser no longer applies this override. Historical results remain
reproducible through the explicit legacy calculation in
`meg_tokens/validation/spd.py`. This report makes no inference about why the
historical implementation was chosen.

---

## 7. The short-log trials, resolved as far as the data allow

5,363 trials (32.9%) log 14 token rows instead of 15, and their two records
disagree: `nProb` behaves like a profile starting at jump 2, while `nTokenDir`
and `sTokenDirs` look like jumps 1–14.

### 7.1 What is now established

| Test | Result |
|---|---|
| Is the implied first token unique? | **Yes — uniquely determined for 5,363 / 5,363** |
| Does `nProb` reconcile with Eq. 1? | **Yes — 100%**, as a 15-token profile seen from jump 2 |
| Would the next jump have fitted before `tTrialEnd`? | **No — 100% of 14-row trials** |
| Accuracy of 14-row trials | **100.0% correct — all 5,363** |
| `majority(sTokenDirs) == nCorrectChoice` | 16,324 / 16,324 |
| `majority(nProb-implied sequence) == nCorrectChoice` | 5,117 / 5,363 (95.4%) |
| Same token composition, both records | 2,635 / 5,363 (49%) |

### 7.2 What is known about the logging

The logs terminate at trial end: the extrapolated next jump falls after
`tTrialEnd` in every 14-row trial. This explains why only 14 runtime rows were
recorded. It does not explain why the direction and probability fields within
those rows are offset by one jump.

The two records are each **internally consistent but mutually incompatible**:

- Logged `nProb`, paired with `tTime`, reconstructs exactly under Eq. 1 and is
  retained as the runtime **evidence profile**.
- `sTokenDirs` is always 15 directions long, and its majority equals
  `nCorrectChoice` on 16,324 / 16,324 trials. It supplies the complete
  design-time sequence used for classification.

The exact LabVIEW cause of the one-step `nTokenDir`/`nProb` offset cannot be
determined from these records. The raw CTF data contain no per-token triggers
that could independently arbitrate the ordering.

### 7.3 Practical resolution

Use the representation appropriate to each analysis:

- **Trial classification:** reconstruct a full 15-point, correct-target profile
  from `sTokenDirs`. The §4.5/§8 rule independently recovers 100% of designed
  labels in both the 14- and 15-row logging groups.
- **Runtime SP and SPD:** retain logged `nProb` paired with `tTime`; carry a
  `token_log_rows == 14` QC flag. Report SPD once for all logged trials and once
  for the validated 15-row-only sensitivity set.
- **Design-derived time-resolved SP/SPD:** do not align the design profile to
  runtime time for a 14-row log. The production helper raises an error, and the
  validator exposes no analysis-ready design SPD for these trials.

The termination pattern is understood; only the cause of the one-step offset
remains unknown. Classification no longer requires an index-shift decision.

---

## 8. The specification

**Step 1 — parse correctly.** *Applied.*

**Step 2 — preserve separate runtime and design profiles.** *Applied.*

```python
sp_runtime_chosen = n_prob
sp_runtime_correct = (
    sp_runtime_chosen
    if nChoiceMade == nCorrectChoice
    else [1 - value for value in sp_runtime_chosen]
)
sp_design_correct = equation_1_profile(sTokenDirs, target=nCorrectChoice)
```

Keep logged `nProb` paired with `tTime` for runtime SP/SPD. Use
`sp_design_correct` only for trial classification; do not overwrite the runtime
profile with it.

**Step 3 — apply the revised labelling definition.** *Applied.*

```python
if s_trial_class in ('e', 'a', 'm'):
    cls, source = {'e': 1, 'a': 2, 'm': 3}[s_trial_class], 'design'
elif s_trial_class == 'x':
    cls = classify(sp_design_correct)
    source = 'inferred' if cls else 'unclassified'
else:
    cls, source = 0, 'not_applicable'  # RT task ('r')
```

For an `'x'` trial, apply this rule to the correct-target design profile:

```python
if SP2 > 0.60 and SP5 > 0.75 and SP8 > 0.75:
    return EASY
if SP2 == 0.50 and 0.40 < SP3 < 0.65 and 0.35 < SP5 < 0.65:
    return AMBIGUOUS
if (
    SP2 == 0.50
    and 0.38 < SP3 < 0.40
    and 0.35 < SP5 < 0.65
    and SP11 == 1.0
):
    return AMBIGUOUS
if SP3 < 0.40:
    return MISLEADING
return UNCLASSIFIED
```

Use numerical tolerance for comparisons with 0.5 and 1.0.

**Step 4 — carry provenance and QC.** *Applied.* Store `sTrialClassRaw`,
`sp_design_correct`, `trial_class_source` (`design`, `inferred`, `unclassified`,
or `not_applicable`), the rule that fired, `token_log_rows`, and
`token_log_short`.

**Step 5 — enforce the short-log SPD policy.** *Applied.* Compute logged
chosen-target SPD from the last `nProb` value available at or before the decision
time. Report `all_logged` and `validated_15row` summaries. Design-derived runtime
alignment and SPD require exactly 15 runtime rows and raise an error otherwise.

### Why this is right

| Class | Old method | Implemented method | Change | Preprint |
|---|---:|---:|---:|---:|
| Easy | 4,309 (26.4%) | 4,098 (25.1%) | −211 | 26% |
| Ambiguous | 3,711 (22.7%) | 3,483 (21.3%) | −228 | 22% |
| Misleading | 1,587 (9.7%) | 1,718 (10.5%) | +131 | 11% |
| Unclassified | 6,717 (41.1%) | 7,025 (43.0%) | +308 | ~41% |

The deterministic rule reproduces all 5,224 designed labels. Production retains
those labels directly and applies the rule only to the 11,100 random trials,
which contribute 2,244 easy, 1,831 ambiguous, 0 misleading, and 7,025
unclassified trials.

### Caveat on validation

The SP(11) boundary is data-derived and should be reported as such. Random trials
have no recorded class ground truth, so their inferred labels cannot be directly
verified. No random sequence in this dataset meets the correct-target
misleading rule; the designed-misleading trials nevertheless supply 10.5% of
the final dataset. An unmatched random trial remains valid and contributes to
all analyses not explicitly restricted to the three difficulty classes.

---

## 9. Remaining readiness work

Full detail in `docs/behavior_metrics_readiness.md`. Summary of what else blocks
the metrics phase:

| Gap | State |
|---|---|
| Success probability & SP-at-decision (SPD) | **Implemented with safeguards** — logged SPD is reported for all trials and for the validated 15-row-only subset; design-derived time alignment is blocked for short logs |
| Group-level statistics | **Absent** — `analyze_behavior` writes one row per subject and stops; none of the paper's six headline numbers can currently be produced |
| Statistical level | Within-subject Welch tests on pooled trials, where the paper uses paired t-tests across subjects (N=28) |
| Fast-vs-slow error rates | **Absent** — `percent_correct` pools both conditions |
| Trial-level feature table | **Absent** — nothing for MEG to join on |
| Subject exclusions (32 → 28) | **Absent** |
| `reports/behavior_summary.py` | Plots raw RT under a "Decision Time" title; omits the `nOutcome == 7003` filter |
| Silent zero-returns | A subject with no RT run silently gets `motor_baseline = 0`, inflating every DT |

The analysis roadmap — additional behavioural measures drawn from the
dynamic-decision-making literature in monkeys and humans (accuracy-criterion
decline, psychophysical reverse correlation, urgency-gating vs drift-diffusion
model comparison, single-trial model-based MEG regressors) — is in §4 of that
document.

---

## 10. Status

**Applied:** scientific-notation parsing, strict token-field validation,
Equation-1 reconstruction, deterministic classification, preservation of
designed labels, classification provenance, short-log flags, paired all-log and
15-row-only SPD reporting, and the design-time alignment guard.

**Real-data validation:** all 5,224 designed labels remain unchanged.
Analysis-ready design SPD is withheld for 5,363 / 5,363 short logs and available
for 10,961 / 10,961 complete logs.

**Still pending:** re-ingest Stage-1 behavioral derivatives so they contain the
new fields, and expose the committed validator through a reproducible
`meg-tokens behavior qc` command. The remaining behavioral-analysis gaps are in
§9 and `docs/behavior_metrics_readiness.md`.
