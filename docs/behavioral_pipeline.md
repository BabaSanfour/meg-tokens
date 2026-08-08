# Behavioral Pipeline: Stage 0 → Stage 1

Data issues found, how each was handled, and its impact. Grows through
Stage 2 and results. Contract (schema/columns/validation) lives in
`docs/data_contract.md` — not duplicated here.

## Stage 0: Raw BIDSification

| Issue | Handling | Impact |
| :--- | :--- | :--- |
| 3 non-standard `.tdms` filenames (scratch fragments: H03 `temp_180214.tdms`, H18 `temp_181024.tdms`, H23 `temp_181121.tdms`) | Explicit `behavior_ignore_files` allowlist; any other non-matching file still errors | 320/323 retained runs |
| H01, H05 raw-session duration counts don't match 8-Slow/Fast + 2-RT | `KNOWN_SESSION_OVERRIDES`, cross-referenced against the acquisition notebook (H01 labeled `Pilot01` there); fingerprint independently reproduces all 17 entries | H05 RT1 confirmed unmatched — real gap, not ambiguity |
| H01 headshape not found on any known drive | Left `not_found` / review | No headshape derivative for H01 |
| H05 RT1 has no raw MEG counterpart | Left `ambiguous` / review, 0 candidates | RT1 excluded from H05's MEG-joined analyses |
| H05 has no empty-room recording, any date | Left `not_found` / review | No empty-room for H05 |
| H07, H10 have no FreeSurfer reconstruction | Left `not_found` / review | No anat/BEM for H07, H10 |
| H26/H27 shared empty-room `_01.ds` `.hc` corrupt (`RuntimeError: HPI information not available`) | `KNOWN_NOISE_OVERRIDES` redirects to the valid retake `_02.ds` | H26/H27 empty-room resolved |
| H01 Slow3, H06 Slow2, H10 Fast2: Start-pulse count mismatch at the trailing boundary | `KNOWN_TRAILING_TRIAL_MISMATCHES`, staged with a note instead of sent to review | 3 runs recovered |
| H06 Start/Go trigger codes swapped vs. every other subject | `SUBJECT_EVENT_OVERRIDES` (event code `524288` is Start for H06, `262144` elsewhere) | H06 matches and epochs correctly |

## Stage 1: Behavioral Log Parsing

| Issue | Handling | Impact |
| :--- | :--- | :--- |
| 226 trials never started (`nOutcome==7003`, no go cue) | Retained in Stage 1; excluded from behavioral counts and MEG event matching | Post-error adjacency corrected: post-correct means changed for 16 subjects, post-error for 5 (largest: H11 post-correct −9.30 ms, H08 post-error +9.85 ms) |
| Movement time not recorded — `tEnterCenter`/`tExitCenter` written from the same LabVIEW event | Both fields retained; no movement measure computed | Response-vigor analysis omitted, not implemented against a null field |
| `nTrialIndex` doesn't order blocks within a session (session-scoped counter, resets inconsistently) | Block/session-drift analyses use `nInitialTime` (session clock) instead | Correct block order; confirms Fast/Slow blocks interleave |
| LabVIEW scientific-notation probabilities near 0/1 misparsed by the original parser | Parser hardened: signed/exponent notation, float-zero tolerance, `[0,1]` range check, equal-length array check | 881 out-of-range values found, 870 corrected, 11 dropped/misaligned; no historical label, DT result, or trial count changed |
| 5,363/16,324 trials have 14-row (not 15-row) token logs — cause unknown | Included for classification/logged-SPD; excluded from design-time-resolved SP; 15-row sensitivity set reported alongside the all-log figure | Defines which trials feed which analyses |
| Classifying by chosen-target `nProb` (original/preprint method) is response-confounded | Classify by the correct-target profile from `sTokenDirs` instead; recorded labels (`e`/`a`/`m`) preserved verbatim, deterministic rule applied only to random (`x`) trials | See Findings, below |

## Findings

### Trial-classification reference frame

**Method.** Two candidate frames for classifying random (`x`) trials: the
recorded runtime `nProb`, referenced to the *chosen* target (the original
notebook's method), vs. a profile derived from `sTokenDirs`, referenced to
the *correct* target. `SP_chosen = 1 - SP_correct` whenever the subject
erred, so the chosen frame's class assignment depends on the response it is
later used to explain — a circularity the design frame avoids.

Classification rule (reproduces all 5,224 recorded designed labels):

```python
if SP2 > 0.60 and SP5 > 0.75 and SP8 > 0.75:
    trial_class = EASY
elif SP2 == 0.50 and 0.40 < SP3 < 0.65 and 0.35 < SP5 < 0.65:
    trial_class = AMBIGUOUS
elif SP2 == 0.50 and 0.38 < SP3 < 0.40 and 0.35 < SP5 < 0.65 and SP11 == 1.0:
    trial_class = AMBIGUOUS
elif SP3 < 0.40:
    trial_class = MISLEADING
else:
    trial_class = UNCLASSIFIED
```

**Result.** Real dataset counts (32 subjects, 16,324 started-and-chosen
trials):

| Class | Implemented | Preprint |
| :--- | ---: | ---: |
| Easy | 4,098 (25.1%) | 26% |
| Ambiguous | 3,483 (21.3%) | 22% |
| Misleading | 1,718 (10.5%) | 11% |
| Unclassified | 7,025 (43.0%) | ~41% |

Random (`x`) trials, inferred: 2,244 easy, 1,831 ambiguous, **0 misleading**.

Mean decision time by class and frame (28-subject list, matching the
preprint's exclusions):

| Classification | Easy | Ambiguous | Misleading | Amb − Mis |
| :--- | ---: | ---: | ---: | ---: |
| Design frame, `'x'` only (implemented) | 1023.8 | 1415.2 | 1336.2 | +5.09 |
| Recorded labels only | 971.8 | 1400.4 | 1336.2 | +3.31 |
| Chosen frame, all trials (legacy) | 1037.1 | 1399.4 | 1563.6 | −5.18 |
| Preprint | 1028±59 | 1405±74 | 1433±79 | −1.84 |

**Interpretation.** The correct-target design frame is the valid method.
Chosen-frame "misleading" is 53.7% genuine errors on easy/ambiguous stimuli,
not misleading stimuli — the confound inflates its mean DT and produces the
preprint's ambiguous-vs-misleading sign as an artifact, not a real effect.
No random trial can be classified misleading under the design frame: after
3 jumps the correct target never trails in a random sequence — a property
of the stimulus set (`e` splits 3-0 always, `a` splits 2-1/1-2, `m` splits
1-2 [n=1,706] or 0-3 [n=12], `x` only ever 2-1 or 3-0), not an error.
`infer_random_classes=false` (recorded labels only, symmetric with the
preprint's treatment of the random 60% as a non-category) gives the closest
Easy-vs-Ambiguous match to the preprint (t=-15.12 vs -15.04); under it
misleading is slightly *faster* than ambiguous — coherent, since misleading
evidence invites early commitment while ambiguous evidence delays it.

**Supporting evidence.**

- Chosen-target `nProb` reconciles 100% against Equation 1
  (`p(R|N_R,N_L,N_C)`, binomial random-walk over the `N_C` centre tokens)
  for both 15-row (n=10,961) and 14-row (n=5,363) logs; frames mirror via
  `sp_correct = sp_chosen if nChoiceMade==nCorrectChoice else 1-sp_chosen`.
  Classifying by the chosen frame recovers only 41.3% of designed-misleading
  trials (25.8% of its assigned "misleading" trials are correct).
- Classification-rule signature `SP(2)=0.5, SP(3)=0.3872, SP(5)=0.623,
  SP(8)=0.7734` occurs in 641 designed-ambiguous and 381 designed-misleading
  trials; `SP(11)` separates them (`1.0` ambiguous vs. `0.6875`/`0.9375`
  misleading).
- 14-row logs (one token short, cause unknown): 0% full design-profile
  match either shifted or unshifted (vs. 100% for 15-row logs); of their
  logged-SPD alignment, 48.7% match only a shifted read, 49.0% match
  neither, and the 2.3% matching both are trivial (pre-first-token or
  post-final-token values) — none has a complete `sTokenDirs` match.
  Confirms the exclusion-from-design-time-alignment policy above.
- Parser defect (scientific-notation probabilities near 0/1) traced to the
  first parser commit (`13e07c6`), not the refactor; fix verified across
  all 320 retained runs.
