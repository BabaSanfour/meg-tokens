# T0-1: `nProb`, `sTokenDirs`, and `sTrialClass`

Final analytical specification for the `nProb`/trial-class issue. Results cover
256 Fast/Slow runs, 16,324 started-and-chosen trials, and 32 subjects.

## Decision summary

| Item | Final decision |
|---|---|
| Runtime `nProb` | Treat as the chosen-target success-probability profile, paired with `tTime`. |
| Trial classification | Use a complete correct-target profile reconstructed from `sTokenDirs`. |
| Designed labels (`e`, `a`, `m`) | Preserve the recorded LabVIEW label. |
| Random labels (`x`) | Apply the deterministic SP(2/3/5/8/11) rule; otherwise use `unclassified`. |
| RT labels (`r`) | Mark as not applicable for difficulty classification. |
| Short logs | Keep for classification and logged runtime analyses; do not use for design-derived time alignment. |
| Parser | Accept signed scientific notation, validate probabilities, and require equal runtime-array lengths. |

## 1. Success probability and reference frame

Equation 1 gives the probability that the right target will receive the
majority of the 15 tokens:

```text
                         N_C!      min(N_C, 7 − N_L)        1
p(R | N_R, N_L, N_C) = ──────  ·          Σ           ─────────────
                         2^N_C             k=0          k!(N_C − k)!
```

`N_R` and `N_L` are the tokens already at the right and left targets, and
`N_C = 15 - N_R - N_L` is the number remaining in the centre. Each remaining
token is assumed equally likely to jump left or right. Swapping left and right
gives the opposite target’s profile.

Equation 1 reconciles every runtime profile:

| Runtime rows | Trials | Result |
|---:|---:|---|
| 15 | 10,961 | 100% — chosen-target profile |
| 14 | 5,363 | 100% — chosen-target profile beginning at jump 2 |

`nProb` therefore remains the logged probability of the target selected by the
participant. The two target frames are mirrors:

```python
sp_correct[j] = sp_chosen[j] if nChoiceMade == nCorrectChoice else 1 - sp_chosen[j]
```

For runtime evidence and logged SPD, retain `nProb` unchanged and paired with
`tTime`. For difficulty classification, use the complete profile reconstructed
from `sTokenDirs` and referenced to `nCorrectChoice`. This makes difficulty a
property of the token sequence rather than of the participant’s response.

Using the chosen frame for class labels recovers only 41.3% of designed-
misleading trials, and its assigned misleading class is 25.8% correct. The
correct-target design frame removes that response dependence.

## 2. Classification rule

The four traditional class points contain one shared signature:

```text
SP(2)=0.5, SP(3)=0.3872, SP(5)=0.623, SP(8)=0.7734
```

It occurs in 641 designed-ambiguous and 381 designed-misleading trials. The
SP(11) value separates them: ambiguous trials have `SP(11) == 1.0`, while the
misleading trials have `SP(11) == 0.6875` or `0.9375`.

Use numerical tolerance for comparisons with 0.5 and 1.0:

```python
if SP2 > 0.60 and SP5 > 0.75 and SP8 > 0.75:
    trial_class = EASY
elif SP2 == 0.50 and 0.40 < SP3 < 0.65 and 0.35 < SP5 < 0.65:
    trial_class = AMBIGUOUS
elif (
    SP2 == 0.50
    and 0.38 < SP3 < 0.40
    and 0.35 < SP5 < 0.65
    and SP11 == 1.0
):
    trial_class = AMBIGUOUS
elif SP3 < 0.40:
    trial_class = MISLEADING
else:
    trial_class = UNCLASSIFIED
```

This rule reproduces all 5,224 recorded designed labels. The narrow
`0.38 < SP(3) < 0.40`/SP(11) branch is a data-derived extension not reported in
the preprint. It currently affects zero random trials in the 320 retained runs,
but is retained so any future matching random sequence is classified
consistently with the task's designed labels.

Production classification is:

```python
if sTrialClassRaw in ('e', 'a', 'm'):
    use recorded label; source = 'design'
elif sTrialClassRaw == 'x':
    apply the rule; source = 'inferred' or 'unclassified'
elif sTrialClassRaw == 'r':
    class = 0; source = 'not_applicable'
```

The resulting counts are:

| Class | Historical method | Implemented method | Preprint |
|---|---:|---:|---:|
| Easy | 4,309 (26.4%) | 4,098 (25.1%) | 26% |
| Ambiguous | 3,711 (22.7%) | 3,483 (21.3%) | 22% |
| Misleading | 1,587 (9.7%) | 1,718 (10.5%) | 11% |
| Unclassified | 6,717 (41.1%) | 7,025 (43.0%) | ~41% |

The implemented random-trial counts are 2,244 easy, 1,831 ambiguous, 0
misleading, and 7,025 unclassified. Random trials have no recorded class
ground truth, so their inferred labels cannot be directly validated; the
designed trials provide the 100% rule validation.

Unclassified random trials remain valid trials. They remain in Stage 1 and MEG
metadata and contribute to Fast/Slow DT, accuracy, post-error, continuous-SP,
SumLogLR, and regression analyses. They are omitted only from analyses
restricted to the three named difficulty classes.

## 3. Short runtime logs

The task has 15 tokens, but 5,363 trials contain 14 runtime rows. In every case,
the next expected jump falls after `tTrialEnd`, so the log ends at trial end.
Within those rows, `nTokenDir` represents jumps 1–14 while `nProb` behaves like
the probabilities after jumps 2–15. The cause of this one-step offset is
unknown.

The records are therefore used as follows:

| Analysis | 14-row trials |
|---|---|
| Choice, accuracy, RT, DT, Fast/Slow comparisons | Include |
| Event-aligned MEG without token-level evidence | Include |
| Classification and final token composition | Include; use `sTokenDirs` |
| Logged `nProb`/SPD | Include with `token_log_short`; report a 15-row sensitivity set |
| Design-derived time-resolved SP or SPD | Exclude; alignment is not validated |

Profile agreement confirms the limitation:

| Runtime rows | Trials | Legal Eq.-1 path | Full design-profile match | Point agreement |
|---:|---:|---:|---:|---:|
| 15 | 10,961 | 100% | 100% | 100% |
| 14 | 5,363 | 100% | 0% unshifted / 0% shifted | 20.6% unshifted / 60.3% shifted |

For the 14-row group, logged SPD alignment outcomes are:

| Outcome | Trials | Share |
|---|---:|---:|
| Matches both candidate alignments | 122 | 2.3% |
| Matches shifted alignment only | 2,613 | 48.7% |
| Matches neither | 2,628 | 49.0% |
| Matches unshifted only | 0 | 0.0% |

The 122 matches are trivial values: 31 decisions occur before the first token
(`SPD = 0.5`) and 91 have reached `SPD = 1.0`. They are not representative and
must not be selected for inference. No short-log trial has a complete match to
`sTokenDirs`.

For the 32 available subjects, pooled mean chosen-target SPD is:

| Labels and runtime data | Easy | Ambiguous | Misleading |
|---|---:|---:|---:|
| Historical labels, all logs | 0.809 (n=4,309) | 0.653 (n=3,711) | 0.576 (n=1,587) |
| Revised labels, all logs | 0.788 (n=4,098) | 0.660 (n=3,483) | 0.648 (n=1,718) |
| Revised labels, 15-row logs | 0.777 (n=2,460) | 0.647 (n=2,452) | 0.645 (n=1,503) |

The all-log row retains the full sample but includes the short-log mismatch;
the 15-row row is a sensitivity analysis, not a corrected estimate for omitted
trials. Exact reproduction of preprint Figure 1E is not established because
the paper’s numerical SPD values, analysis code, and four-subject exclusion
list are unavailable.

## 4. Parser hardening

LabVIEW writes probabilities near 0 and 1 in scientific notation. The old
parser misread or dropped these values. Across the real dataset:

| Effect | Trials |
|---|---:|
| At least one out-of-range parsed value | 881 |
| Value changed by parsing | 870 |
| Entry dropped and misaligned with `tTime` | 11 |

The corruption occurred at late token rows and changed no historical labels,
decision-time results, or trial counts. `meg_tokens/behavior/tdms.py` now:

- accepts signs and exponents;
- validates and clamps floating-point zero within tolerance;
- rejects probabilities outside `[0, 1]`; and
- requires equal lengths for `nTokenNum`, `nTokenDir`, `tTime`, and `nProb`.

The defect is inherited from the first parser commit (`13e07c6`), not from the
refactor. The fix was verified across all 320 retained runs.

## 5. Implemented fields and status

Stage 1 carries the raw label and provenance fields needed downstream:
`sTrialClassRaw`, `trial_class_source`, `trial_class_rule`,
`sp_design_correct`, `token_log_rows`, and `token_log_short`. Runtime `nProb`
and `tTime` remain unchanged.

Implemented: parser hardening, correct-target design profiles, deterministic
classification, designed-label preservation, random-trial inference,
classification provenance, short-log flags, paired all-log/15-row SPD
reporting, and the design-time alignment guard.

Existing Stage 1 derivatives must be regenerated from TDMS:

```bash
meg-tokens --config tokens.toml behavior ingest
```

The committed validator can be run with:

```bash
meg-tokens --config tokens.toml behavior qc
```

