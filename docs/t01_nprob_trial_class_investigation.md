# T0-1: `nProb`, `sTokenDirs`, and `sTrialClass` — Investigation and Recommendation

Follow-up to `docs/behavior_metrics_readiness.md` §2.3 and §3 (T0-1). All numbers
measured on the real dataset: **256 Fast/Slow runs, 16,324 started-and-chosen
trials, 32 subjects**, with the raw CTF recordings mounted for cross-checking.

With a tolerance appropriate for values rounded to four decimal places,
**Equation 1 reconciles 100% of trials**. The classification consequence of the
14-row logs is handled. These logs terminate at trial end; only the cause of the
one-step offset between `nTokenDir` and `nProb` remains unknown (§4).

---

## Summary

| # | Finding | Status |
|---|---|---|
| 1 | `nProb` is the success probability of the **chosen** target; reconciles with Eq. 1 for **100%** of trials | Confirmed |
| 2 | The preprint code computes trial class from the chosen-referenced profile; this analysis instead uses a correct-target design profile | Recommended analytical correction (§2) |
| 3 | Classifier accuracy is evaluated with no fallback to the recorded designed label | Confirmed (§3) |
| 4 | The 14-row logs terminate at trial end, but the cause of the one-step `nTokenDir`/`nProb` offset remains unknown | Classification handled with `sTokenDirs`; timing caveat retained (§4) |
| 5 | Correct-target SP from the full `sTokenDirs` sequence separates designed classes using SP at jumps 2, 3, 5, 8, and 11 | 100% recovery with the deterministic §5 rule |
| 6 | The code re-derives labels that LabVIEW already records, overwriting 1,119 / 5,224 designed trials | Confirmed — preserve designed labels and classify only `'x'` trials (§6) |
| 7 | Random sequences matching no class template remain valid, retained, and explicitly unclassified | Expected, not data loss (§6–§7) |

---

## 1. What `nProb` is

Equation 1 gives the probability that the **right target** will ultimately
receive the majority of the 15 tokens, given the state of the trial at time
`t`:

```
                         N_C!      min(N_C, 7 − N_L)        1
p(R | N_R, N_L, N_C) = ──────  ·          Σ           ─────────────
                         2^N_C             k=0          k!(N_C − k)!
```

where:

- `R` is the event that the right target ultimately wins;
- `N_R` is the number of tokens already at the right target;
- `N_L` is the number of tokens already at the left target;
- `N_C = 15 - N_R - N_L` is the number of tokens still in the centre;
- `k` is a possible number of the remaining centre tokens that will jump left;
- `7 - N_L` is the largest number of additional leftward tokens that still
  leaves the left target with at most 7 tokens, allowing the right target to
  win; and
- `N_C! / (k!(N_C-k)!)` is the binomial coefficient: the number of ways exactly
  `k` of the `N_C` remaining tokens can go left.

Each remaining token is assumed equally likely to jump left or right, so each
complete assignment has probability `1 / 2^N_C`. The sum therefore counts all
future assignments in which the left target finishes with at most 7 tokens—and
the right target consequently finishes with at least 8. If the left target
already has 8 tokens the probability is 0; if the right target already has 8 it
is 1. Swapping left and right gives the probability for the left target.

Implementing Equation 1 with exact integer binomials and reconstructing the SP
profile from the per-row `nTokenDir` values logged in each `Tokens.Data` block,
with 1e-6 tolerance:

| Trial group | n | Reconciles with Eq. 1 |
|---|---:|---|
| 15 token rows | 10,961 | **100%** — profile of the **chosen** target |
| 14 token rows | 5,363 | **100%** — chosen-target profile, starting at jump 2 |

Zero unexplained trials. `nProb` is the success probability of the target the
subject chose, ending at exactly 1.0 on correct trials and 0.0 on errors.

---

## 2. Correct-referencing, explained

### 2.1 What "reference target" means

Success probability is defined **per target**. The preprint's Equation 1 gives

> p(R | N_R, N_L, N_C) — "the probability that the target on the right will
> ultimately be the correct one"

so `p_right(t)` and `p_left(t)` are two mirror-image traces with
`p_left(t) = 1 − p_right(t)` at every instant. A "success probability profile"
is meaningless until you say *whose*. There are three natural choices:

| Reference target | Meaning of `SP(t)` | Use in this analysis |
|---|---|---|
| **Chosen target** | Probability that the target selected by the participant will receive the majority of tokens | Runtime evidence and SPD |
| **Correct target** | Probability that the target that ultimately receives the majority is currently favoured | Trial classification: easy, ambiguous, or misleading |
| **Fixed side** (left or right) | Probability that one specified screen-side target will receive the majority | Side-specific analyses only |

`nProb` is logged in the **chosen** frame (§1), and the unchanged code used for
the preprint applied the class thresholds to that chosen-referenced profile. The
preprint lists the thresholds as follows:

> "…15% of trials were so-called 'easy' trials, in which success probability (SP)
> had to exceed 0.6 after two token jumps, 0.75 after five token jumps and 0.75
> after eight token jumps… Another 10% of trials were called 'misleading' trials,
> in which the SP had to be below 0.4 after three token jumps."

For the analysis recommended here, difficulty class is treated as a property of
the token sequence rather than of the participant's response. The thresholds
are therefore evaluated on the complete, correct-target SP profile reconstructed
from `sTokenDirs`.

### 2.2 Why the recommended change matters

Read `SP(3) < 0.4` off the **chosen** profile and you are asking "did the
evidence initially oppose the button the subject pressed?" On a trial the subject
got *right*, the two frames coincide and nothing changes. On a trial the subject
got *wrong*, they are mirror images — so a genuinely misleading stimulus (where
the subject was misled, i.e. answered wrongly) has a **high** chosen-referenced
SP early and fails the test entirely.

The measured consequence is that the unchanged preprint code recovers only
**41.3%** of LabVIEW's designed-misleading trials, and its assigned
"misleading" class is **25.8%** correct. The label therefore depends strongly on
the participant's response, making DT-by-class and MEG-by-class contrasts partly
circular. Correct-target classification removes that response dependence.

### 2.3 The conversion is exact and needs no token sequence

Because the two profiles sum to 1 at every jump:

```
sp_correct[j] = sp_chosen[j]        if nChoiceMade == nCorrectChoice
                1 - sp_chosen[j]    otherwise
```

This conversion remains the correct way to obtain a correct-referenced version
of the *logged* probability profile for SPD and other runtime analyses. Trial
classification uses the full designed sequence instead, for the reasons in §4.

---

## 3. Classifier evaluation

Classifier accuracy is evaluated independently of the recorded designed label.
Each trial starts as `unclassified` and receives a class only when a rule fires,
matching how the classifier must operate on an `'x'` trial with no recorded
class label. This gives:

| Probability source and rule | 14-row designed | 15-row designed | All designed |
|---|---:|---:|---:|
| Logged `nProb`, legacy indices, empirical four-point rule | 40.3% | 91.6% | 76.4% |
| Logged `nProb`, row-aware indices, empirical four-point rule | 38.8% | 91.6% | 75.9% |
| Eq. 1 on `sTokenDirs`, correct-referenced, empirical four-point rule | 95.4% | 91.6% | 92.7% |
| Eq. 1 on `sTokenDirs` plus the SP(11) boundary rule in §5 | **100%** | **100%** | **100%** |

**Preprint note.** The preprint did not report classifier accuracy against the
recorded designed labels. The legacy code used the logged, chosen-referenced
`nProb` profile with the legacy indices and four-point rule, while retaining the
existing label when no rule fired. That procedure produces 26.4% easy, 22.7%
ambiguous, 9.7% misleading, and 41.1% unclassified, close to the reported 26%,
22%, 11%, and approximately 41%. The 76.4% accuracy shown here is this report's
independent evaluation, not a preprint result.

Logged `nProb` reproduces the preprint implementation, but the designed-label
validation does not support using it as the basis of the revised classifier,
particularly for 14-row trials. Runtime analyses still retain `nProb` unshifted
and paired with `tTime`; trial classification instead uses the complete profile
reconstructed from `sTokenDirs`.

---

## 4. The 14-row logs

The task always uses 15 tokens, but 5,363 trials contain only 14 runtime log
rows. These are not 14-token trials:

| Group | Trials |
|---|---:|
| All 14-row trials | 5,363 |
| Designed (`'e'`/`'a'`/`'m'`) | 1,549 |
| Random (`'x'`) | 3,814 |

In every short-log trial, the next expected jump would occur after `tTrialEnd`.
This explains why the runtime log stops at 14 rows.

There is still a one-jump mismatch inside those rows: `nTokenDir` represents
jumps 1–14, while `nProb` behaves like the probabilities after jumps 2–15. The
available data do not explain why LabVIEW recorded the two fields this way.

Use each record for a different purpose:

- **Trial classification:** compute the complete 15-jump, correct-target SP
  profile from `sTokenDirs`. This avoids the row mismatch.
- **Runtime SP and SPD:** keep `nProb` paired with its recorded `tTime`. Flag the
  14-row trials and repeat timing-sensitive analyses without them as a check.

### SP/SPD validation

SPD was defined as the last SP available at or before the estimated decision
timestamp, `tEnterTarget - subject motor baseline`. The logged chosen-target
`nProb` profile was compared with Equation 1 applied to `sTokenDirs`, using a
1e-6 tolerance, across all 16,324 started-and-chosen trials:

Trial-level profile validation:

| Log rows | Trials | Legal Eq.-1 path | Full unshifted design-profile match | Full shifted design-profile match |
|---:|---:|---:|---:|---:|
| 15 | 10,961 | 10,961 (100%) | 10,961 (100%) | Not applicable |
| 14 | 5,363 | 5,363 (100%) | 0 (0%) | 0 (0%) |

Point-level agreement uses individual logged SP positions as its denominator;
the unshifted and shifted matches can overlap and do not represent percentages
of trials:

| Alignment | 15 rows: 164,415 positions | 14 rows: 75,082 positions |
|---|---:|---:|
| Unshifted point agreement | 100% | 20.6% |
| Shifted point agreement | Not applicable | 60.3% |

All 10,961 SPDs in the 15-row group match the complete design profile. For the
5,363 short-log trials, the mutually exclusive SPD outcomes are:

| 14-row SPD alignment outcome | Trials | Share |
|---|---:|---:|
| Matches both alignments | 122 | 2.3% |
| Matches unshifted alignment only | 0 | 0.0% |
| Matches shifted alignment only | 2,613 | 48.7% |
| Matches neither alignment | 2,628 | 49.0% |
| **Total** | **5,363** | **100%** |

The 15-row results validate both the Equation-1 implementation and the SPD
timing rule. For the 14-row trials, neither aligning logged row 1 with design
jump 1 nor shifting it to design jump 2 reproduces the complete profile or SPD
reliably. Each short `nProb` series is internally consistent with some legal
Equation-1 path covering jumps 2–15, but that latent path does not match the
recorded `sTokenDirs`. It is therefore an internal consistency check, not a
recovery of the actual design sequence. The logged `nProb`/`tTime` pair can be
used to compute a log-based runtime SPD, but it includes the short-log mismatch
and cannot be treated as a verified design-derived SP trace. Exact reproduction
of preprint Figure 1E has not been established because the preprint does not
report numerical SPD values and its SPD analysis code and four-subject exclusion
list are unavailable here.

For the 32 available subjects, the pooled mean chosen-target SPD is:

| Labels and runtime data | Easy | Ambiguous | Misleading |
|---|---:|---:|---:|
| Historical labels, all logged trials | 0.809 (n=4,309) | 0.653 (n=3,711) | 0.576 (n=1,587) |
| Revised labels, all logged trials | 0.788 (n=4,098) | 0.660 (n=3,483) | 0.648 (n=1,718) |
| Revised labels, validated 15-row trials only | 0.777 (n=2,460) | 0.647 (n=2,452) | 0.645 (n=1,503) |

The all-logged revised row retains the complete sample but includes the short-log
mismatch. The 15-row row is the validation sensitivity analysis, not a corrected
estimate for the omitted trials.

### Where the 14-row trials can be used

The 5,363 short-log trials remain valid 15-token trials and should not be
discarded from every analysis:

| Analysis | Use the 14-row trials? | Usable trials |
|---|---|---:|
| Choice, accuracy, RT, DT, and condition comparisons | Yes | All 5,363 |
| Event-aligned MEG analyses that do not use token-level evidence | Yes | All 5,363 |
| Recorded or reconstructed trial class | Yes, using the raw label and `sTokenDirs` | All 5,363 |
| Final token majority or sequence composition | Yes | All 5,363 |
| Log-based `nProb` or SPD | Yes, with `token_log_short` flagged and a 15-row-only sensitivity analysis | All 5,363 |
| Design-derived time-resolved SP, SPD, or evidence at decision | No reliable temporal alignment | 0 |

For 122 trials, the SPD value is unchanged by the two candidate alignments and
also agrees with logged `nProb`: 31 have a decision before the first logged token
and therefore `SPD = 0.5`, while 91 have already reached the absorbing value
`SPD = 1.0`. These trials are individually robust to the offset, but the subset
contains only the trivial values 0.5 and 1.0 and is not representative enough for
class comparisons or SPD distributions.

Shifted SPD matches in 2,735 trials overall: the 122 trials matching both
alignments and 2,613 matching only after shifting by one jump. These trials
should not be treated as validated. None of the 5,363 short-log trials has a
complete profile match to `sTokenDirs`, and isolated equality is common because
SP takes a small set of discrete values. Selecting either matching subset for
inferential analyses would introduce selection bias.

The short logs therefore do not prevent classification. Only the cause of the
one-jump mismatch remains unknown, and SPD results must be reported both with
all trials and with the 14-row group excluded.

---

## 5. Deterministic rule derived from the designed labels

For this section, `SP(j)` means Equation 1 after jump `j`, reconstructed from
`sTokenDirs` and referenced to `nCorrectChoice`.

The four traditional points contain one mixed signature:

```
SP(2)=0.5, SP(3)=0.3872, SP(5)=0.623, SP(8)=0.7734
```

It occurs in 641 designed-ambiguous and 381 designed-misleading trials. A
rule using only these four points cannot distinguish them. Assigning this shared
signature to ambiguous gives the highest possible four-point agreement with the
recorded labels: 92.7%.

Jump 11 separates the collision:

| Recorded designed class | SP(11) |
|---|---|
| Ambiguous | 1.0 |
| Misleading | 0.6875 or 0.9375 |

The following fixed rule retains the preprint's main windows and adds the
observed jump-11 boundary case:

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

Applied to the 5,224 designed trials, this deterministic rule reproduces all
recorded labels (100% agreement).

Production code should use numerical tolerance for the comparisons with 0.5
and 1.0. The boundary rule is data-derived and should be reported as such; it is
not stated in the preprint.

---

## 6. Preserve designed labels; classify only random trials

The preprint's Methods text describes SP-based classification **only for random
trials**:

> "We used SP to classify **random trials** into a posteriori classes of trials
> (e.g. 'easy,' 'ambiguous,' and 'misleading' trials) embedded in the sequence"

The unchanged code used for the preprint nevertheless applies the override to
both designed and random trials. In the retained dataset, which contains 5,224
trials with a recorded designed label and 11,100 random trials, this changes
1,119 / 5,224 designed labels:

| Overwrite | n |
|---|---:|
| misleading → ambiguous | 783 |
| misleading → easy | 226 |
| ambiguous → misleading | 99 |
| easy → misleading | 11 |

The flow recommended here is:

1. `'e'`, `'a'`, or `'m'`: retain the recorded label; source = `design`.
2. `'x'`: apply the §5 deterministic rule; source = `inferred` or
   `unclassified`.
3. `'r'`: RT task, for which difficulty class is not applicable; do not run the
   rule.

### Comparison of the two methods

The old method applies the legacy rule to logged, chosen-referenced `nProb` on
both designed and random trials. The new method retains the recorded designed
labels and applies the §5 deterministic rule only to random trials.

| Class | Old method, n (%) | New method, n (%) | Change (new − old) | Preprint |
|---|---:|---:|---:|---:|
| Easy | 4,309 (26.4%) | 4,098 (25.1%) | −211 | 26% |
| Ambiguous | 3,711 (22.7%) | 3,483 (21.3%) | −228 | 22% |
| Misleading | 1,587 (9.7%) | 1,718 (10.5%) | +131 | 11% |
| Unclassified | 6,717 (41.1%) | 7,025 (43.0%) | +308 | ~41% |

Under the new method, the 5,224 designed labels remain unchanged. The 11,100
random trials contribute 2,244 easy, 1,831 ambiguous, 0 misleading, and 7,025
unclassified trials.

No random sequence in this dataset meets the correct-target misleading rule.
This is not itself an error: the recorded designed-misleading trials already
make up 10.5% of the final dataset.

### Unclassified does not mean lost

The three categories are narrow templates, not an exhaustive partition of all
random evidence trajectories. An unmatched random trial remains a valid trial
with class 0. It stays in the Stage-1 table and MEG epoch metadata and contributes
to Fast/Slow DT, accuracy, post-error, continuous-SP, SumLogLR, and regression
analyses. It is excluded only from contrasts explicitly restricted to easy,
ambiguous, and misleading trials.

---
