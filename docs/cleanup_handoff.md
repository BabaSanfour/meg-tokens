# Behavior cleanup — handoff

Working session of 2026-08-23. **The session is finished and committed** in
eight commits, `58e4412..d1a3284`, plus the one that carries this rewrite. Every
commit was checked out in turn and run against the full suite in isolation —
all nine are green. §2 records what each of the three planned steps turned out
to be worth; the verification commands are in §4.

---

## 0. Read this first

The one thing that makes the rest safe: **the behavioral pipeline can prove its
own outputs unchanged.** It re-runs end to end on the real 32-subject data in
about two minutes, and figure rendering is bit-deterministic, so a refactor is
*measured*, not argued:

```bash
python -m scripts.equivalence.harness prepare        # once, copies fixed inputs
python -m scripts.equivalence.harness run baseline   # before a change
#   ... make the change ...
python -m scripts.equivalence.harness run candidate
python -m scripts.equivalence.harness compare baseline candidate
```

It diffs 72 group tables, 320 Stage 1 subject tables, 18 figure PNGs byte for
byte, and every sidecar. `EQUIVALENT` means the science did not move. Snapshots
live under `~/.cache/meg-tokens-equivalence` (override with
`MEG_TOKENS_EQUIV_ROOT`).

It does **not** cover per-subject model fitting (`ssm-fit`, `ssm-evaluate`,
`ssm-recovery`, `ssm-robustness`, `ssm-exclusion-robustness`) or the recovery
merge, whose outputs it consumes. Changes to the likelihood, the solver or the
model set still have to be verified on the cluster.

Current state: **EQUIVALENT**, 513 passed / 27 skipped / 6 failed, and all seven
`scripts/verify/` scripts pass against the real derivatives. All six failures
are in `tests/test_batch_erp_parcellation.py`, predate this work, and are
outside behavior scope.

---

## 1. What was done

### 1.1 Two defects found and fixed

**The production fit-array script could not run.** `thura2012_fit_array.sh`
passed `--model-set mechanistic`, a flag removed from `behavior ssm-fit` some
time ago. Every Slurm array task would have exited with argparse status 2
before fitting anything. The flag is gone; the CLI already fits exactly the
model set it asked for, so no scientific behaviour changed. A new test now
parses every `meg-tokens` invocation in `scripts/*.sh` against the real parser.

**The local recovery derivatives had inconsistent provenance — and it was
producing a wrong figure.** The 96 `ssm*recovery*rep*` files on the data disk
were from an earlier run (`n_starts=2`, `n_per_sequence=40`); the published
merged tables came from the cluster (`n_starts=3`, `n_per_sequence=30`). Every
local `ssm-recovery-aggregate` had been overwriting the published tables with
the wrong simulation, so **F15 as rendered locally was not the F15 in the
write-up** (`urgency_scale` recovery r = .44 against the published .994).

The 96 files were deleted from `/media/karim/Hamza/meg-tokens/...`. The command
now fails naming the missing repetition instead of producing a plausible wrong
answer. Verified after the fix: F15 rendered from a fresh fixture is
byte-identical to the published PNG, and `verify_ssm_doc` passes against the
fixture — both previously failed.

### 1.2 Fourteen unread derivatives removed

`behavior characterization` went from 53 tables to 39. Nothing read these — no
figure, no `scripts/verify/` script, no line of `docs/behavior.md`, no code:

| Removed | Why |
| :--- | :--- |
| `posteriorgrid`, `posteriorgridstats`, `posteriordiscretization`, `posteriordiscretizationstats`, `posteriorevidencesensitivity`, `posteriorevidencesensitivitystats`, `posteriorevidenceindex`, `posteriorevidenceindexstats`, `posteriorevidenceagreement` | Nine tables diagnosing the SP grid's coarsening; supported one *qualitative* sentence in the write-up, no number quoted anywhere. −444 lines of `evidence.py`. |
| `criteriondeclinestats`, `urgencystats`, `lapsestats` | Group tests over subject-level fits nothing read. The subject-level tables they summarized are all still written. |
| `ssmmechanisticstats` | The same `mechanistic_model_statistics` call that already fills `ssmcomparisonstats`, written twice under two names. |
| `ssmexclusionsensitivity` | Leave-one-subject-out recomputation, read by nothing. `ssmrobustness*` and `ssmexclusionrobustness*` stay. |

Removing the posterior family also stranded `criterion_decline_statistics`,
whose only remaining callers were inside it. Every name is now recorded in
`docs/behavior.md`'s "Result families with no figure" table with the reason it
went, so none gets reintroduced by someone noticing a gap.

**Verified:** the harness reported exactly those 14 tables missing and *nothing
else changed*.

### 1.3 CLI dispatch chain replaced

44 `elif args.domain == "X" and args.Y_command == "Z":` branches became
`_HANDLERS[(domain, command)]`. `main()` went from ~620 lines to ten; each
command is a named function; 39 collapsed further from `result = f(...)` /
`return result` to `return f(...)`. `build_parser` was not touched, so the CLI
surface is provably unchanged.

**Be aware:** `cli/main.py` grew by 152 lines (1,185 → 1,337). Forty-four `def`
lines cost more than forty-four `elif` lines saved. The win is structural, not
volumetric.

### 1.4 Every cluster script renamed for the command it submits

`thura2012_*` said which paper the model came from — the one thing all ten had
in common. Now `ssm_<command>[_array].sh` submits `behavior <command>`, and a
test enforces it.

| Was | Now |
| :--- | :--- |
| `thura2012_fit_array.sh` | `ssm_fit_array.sh` |
| `thura2012_aggregate.sh` | `ssm_aggregate.sh` |
| `thura2012_evaluate_array.sh` | `ssm_evaluate_array.sh` |
| `thura2012_evaluate_aggregate.sh` | `ssm_evaluate_aggregate.sh` |
| `thura2012_recovery_array.sh` | `ssm_recovery_array.sh` |
| `thura2012_recovery_aggregate.sh` | `ssm_recovery_aggregate.sh` |
| `thura2012_robustness_array.sh` | `ssm_robustness_array.sh` |
| `thura2012_robustness_aggregate.sh` | `ssm_robustness_aggregate.sh` |
| `thura2012_exclusion_array.sh` | `ssm_exclusion_robustness_array.sh` |
| `thura2012_exclusion_aggregate.sh` | `ssm_exclusion_robustness_aggregate.sh` |
| `render_figs.sh` | `behavior_report.sh` |
| `rerun_behavior.sh` | `behavior_analyze.sh` |

Deleted: `ssm_fit_array.sh` (old one — described the retired two-model design,
defaulted to `n_starts=1`, output rejected by `ssm-aggregate`'s three-start
gate), `thura2012_evaluate.sh` (duplicate of the recovery merge reserving 12 h
and 24 GB for a 1 h job), `thura2012_robustness.sh` (a stub that printed "do
not submit this" and exited 2).

`behavior_report.sh` also lost a stale workaround: it built the figure list by
piping `--list-figures` through `awk` to skip a figure "F26" that no longer
exists. It now says `--figures all`.

### 1.5 Dead code and redundant documentation

- `equivalence_bound_from_summary`, `annotate_contrast`, `fit_fooof`,
  `run_psd_fooof` — four functions with zero callers anywhere.
- One `subject_condition_groups` in `behavior/trials.py` replacing three inline
  copies in `design_effects.py` and a private helper in `evidence.py` that
  `sequential_sampling.py` imported across module boundaries. The four differed
  in whether an empty condition cell is yielded, which was load-bearing in both
  directions; it is now a documented `skip_empty` parameter.
- `BehaviorTableSet`: four near-identical load-cache-filter methods → one
  `_load` with a single missing-file policy.
- 25 numpydoc `Parameters` blocks whose every entry restated the type
  annotation ("Output of :func:`x`.", "Canonical trial-feature table."). This
  follows the convention already present in those files — `read_raw_behavior_table`
  was written that way. **No explanatory comment was touched.**

### 1.6 Test suite trimmed

**875 → 625 collected** at the point this section was written; the seven files
of Step 2 took it to **546**. Sixteen files were grouped in all.

Removed outright: 26 tests that hand-listed command names and asserted argparse
prints `--help`; the N = 28 preprint regression that could never run (it needs
a config with four subject exclusions; the project runs 32 with none); the
tests attached to the deleted analyses.

Added: `tests/test_cli.py` (parser-derived, covers all 44 commands instead of
the 26 someone remembered to list, and pins that every command has a handler)
and `tests/test_cluster_scripts.py` (script ↔ CLI agreement).

Nine files rewritten as **grouped** suites — one test per behaviour rather than
per assertion, sharing the expensive fixtures, keeping every distinct claim:

| File | tests | lines |
| :--- | ---: | ---: |
| `test_analyses_evidence.py` | 59 → 23 | 1257 → 964 |
| `test_analyses_sequential_sampling.py` | 45 → 22 | 1042 → 874 |
| `test_analyses_design_effects.py` | 42 → 18 | 772 → 530 |
| `test_schema.py` | 35 → 17 | 347 → 255 |
| `test_math_inference.py` | 35 → 12 | 427 → 320 |
| `test_features.py` | 35 → 15 | 396 → 250 |
| `test_math_probability.py` | 29 → 11 | 282 → 180 |
| `test_analyses_distributions.py` | 25 → 10 | 426 → 300 |
| `test_math_evidence.py`, `test_trials.py`, `test_tables.py` | 50 → 21 | 529 → 400 |

### 1.7 Repository hygiene

- The stale worktree at `.claude/worktrees/agent-a993d156c19a4c105` is gone. It
  held 8 uncommitted files, so they were committed first to their own branch —
  **`worktree-agent-a993d156c19a4c105`, commit `e1e4cd1`**. Nothing was
  discarded; delete the branch once you have looked at it.
- Orphaned `verify_f18_doc` bytecode removed.
- `README.md`: stale warning that `render_figs.sh`/`rerun_behavior.sh` are
  untracked (they are tracked) corrected; the equivalence harness documented;
  the recovery-merge hazard documented so nobody "fixes" the error by
  restoring the repetition files.

**Net over the whole session (`554adc5..d1a3284`): 62 files changed, 4,303
insertions, 6,220 deletions — −1,917 tracked lines. 316 of the insertions are
this document.**

---

## 2. How the three planned steps turned out

### Step 1 — Commit the verified state ✅

Eight commits, `58e4412..d1a3284`. The planned split was seven; two pairs had to
merge because the *files* do not separate — the fit-array flag fix lives in the
script that was renamed in the same breath, and the tests for the deleted
analyses sit in files that were also regrouped. Splitting those would have meant
hunk surgery producing red intermediate commits, which is worse than a slightly
coarser history.

Each commit was then checked out in turn and run against the full suite
(`git rebase --exec`): 861 → 863 → 730 → 676 → 589 → 589 → 513 → 513 passed,
**zero failures at any step**. The history bisects.

    58e4412  fix(scripts): repair the ssm-fit array and name every launcher…
    c9a7aef  refactor(cli): replace the dispatch chain with a handler table
    b4ce299  refactor(behavior): drop fourteen derivatives nothing reads
    bf828e3  refactor(behavior): remove dead helpers and signature-restating…
    9263ab6  test(behavior): group the suites by behaviour rather than by…
    cfe9b85  feat(scripts): add the before/after equivalence harness
    73767a5  test(behavior): group the last seven suites by behaviour
    d1a3284  refactor: finish the source trim — rename the figure battery…

Still open: delete branch `worktree-agent-a993d156c19a4c105` once reviewed.

### Step 2 — Finish the test trim ✅ (625 → 546 collected, not ~450)

All seven remaining files were grouped: **147 test functions → 81**, 2,577 →
2,358 lines. Per-file numbers are in commit `73767a5`.

The ~450 target was not reached and should not be chased. What is left is not
scaffolding: the suite is now 546 collected of which 130 are `pytest.raises`
guards over the LabVIEW-log parsers, and the rest are one test per distinct
behavioural claim. Going lower means deleting claims.

`tests/reports/factories.py` was left alone. Grouping the report tests stranded
nothing — every factory it exports is still called.

### Step 3 — Finish the source trim ✅ (two of three items were already empty)

- **Docstring scaffolding — nothing left of that class.** The behavior stack is
  11,975 lines with 4,001 docstring lines (33%). Scanned with `ast`: **zero**
  `Returns` sections name a type without saying something about it, and **zero**
  `Parameters` blocks restate every annotation — commit `bf828e3` took the last
  of them. What remains earns its space (`individual_profile` explains which
  slope becomes which column and that joins never drop subjects). Do not sweep
  further, and **do not touch `Notes` sections**: that is where the scientific
  reasoning lives, and it is why the pipeline is auditable.
- **Defensive branches — audited, nothing dead.** Measured rather than read, with
  a filtered `sys.settrace` recording every line the real 32-subject pipeline and
  the whole test suite execute, then subtracted from the statements `ast` says
  exist:

  | Module | statements | never executed |
  | :--- | ---: | ---: |
  | `design_effects.py` | 264 | 5 (2%) |
  | `evidence.py` | 452 | 9 (2%) |
  | `sequential_sampling.py` | 1,021 | 258 (25%) |

  The 14 in the two smaller modules are `continue`/`pass` guards, every one of
  them protecting a cohort shape this dataset does not produce — a subject with
  no Slow block, a class with no trials, a `sm.Logit` that fails to separate.
  Removing them turns a skipped row into a `KeyError` on a differently shaped
  dataset. They stay.

  The 258 in `sequential_sampling.py` are not dead either: they are the
  cluster-only machinery — `_fit_one`, the Hessian standard errors, the
  cross-validation, the matched-sequence predictions, the recovery and
  robustness refits. Neither the harness nor the suite reaches them *by design*
  (§0). Their coverage lives on the cluster; do not read that 25% as slack.

  The one genuine find: `_population_normal`, 50 lines with no callers, removed
  in `d1a3284` along with the module's last `scipy.optimize` import.
- **`reports/behavior_summary.py` renamed** to `reports/behavior_battery.py`,
  which is what its own docstring calls it. The old name collided with
  `layout.behavior_summary()`, a different thing entirely (the Stage 1 subject
  table).

### What is left for a future session

- Delete branch `worktree-agent-a993d156c19a4c105` once reviewed.
- Four MEG-side functions have no callers anywhere and were left alone as out of
  scope for a behavior cleanup: `psd_description` (`workflows/spectral.py`),
  `run_batch_dpca` and two `_*_desc` helpers (`workflows/decomposition.py`),
  `get_peak_latency` / `get_significance_onset` / `compute_motor_lateralization`
  (`analysis/statistics.py`), `extract_roi_time_courses`
  (`features/connectivity.py`), `plot_tfr_spectrogram` / `plot_sensor_topomap`
  (`reports/meg.py`), `_parse_baseline` (`workflows/power.py`). The equivalence
  harness cannot vouch for any of these, so they need the MEG stack's own
  verification before anyone removes them.
- The 6 `tests/test_batch_erp_parcellation.py` failures. Still a separate job.

---

## 3. How to do the test grouping (the method that worked)

1. Read the whole file. The test *names* describe the behaviours; the bodies
   hold the expected values, which must be preserved verbatim.
2. Group by **the contract under test**, not by the function called. Held-out
   fold weighting, for instance, is one behaviour whether it is reached through
   the baseline comparison, the pairwise comparison, or the per-subject
   differences — one test, three entry points.
3. Hoist the expensive fixtures into module-level helpers and assert every
   claim a fixture can support on the one build.
4. Keep parametrized tables for genuinely distinct inputs (eight feature flags
   that must reject serialized booleans); collapse to a loop inside one test
   when the cases are the *same* behaviour (44 commands whose `--help` exits 0).
5. Keep every explanatory docstring and comment. They carry the *why* — "a
   2AFC accuracy of 0.5 carries no information; against zero it is huge" — and
   are the most valuable thing in the file.
6. Run that file, then the whole suite, then the harness.

Do **not** delete guard tests to hit a number. 130 tests use `pytest.raises`,
concentrated in `test_schema.py` and `test_math_probability.py`; those modules
exist to reject malformed LabVIEW logs, and they are the only protection the
fitting layer has, because the harness cannot reach it.

---

## 4. Verification commands

Run all four before committing anything. Expected results are stated, and are
what `d1a3284` produces today.

```bash
# 1. Test suite — expect 513 passed, 27 skipped, 6 failed.
#    All six failures are pre-existing, in tests/test_batch_erp_parcellation.py.
python -m pytest -q

# 2. Pipeline equivalence — expect "EQUIVALENT".
python -m scripts.equivalence.harness run candidate
python -m scripts.equivalence.harness compare step3_baseline candidate

# 3. The write-up against the real derivatives — expect seven PASS.
for m in verify_doc verify_act4_doc verify_act5_doc \
         verify_f11_doc verify_f12_doc verify_f13_doc verify_ssm_doc; do
    python -m scripts.verify.$m >/dev/null && echo "PASS $m" || echo "FAIL $m"
done

# 4. Every scripted CLI call — the cluster-script tests cover this.
python -m pytest tests/test_cluster_scripts.py -q
```

---

## 5. Traps

- **The cluster checkout still has the old script names.** Ten renames and
  three deletions land there on the next sync. `README.md` already warns
  against a deleting rsync — pull rather than push, and check `git status` on
  both sides first.
- **Never restore the `ssm*recovery*rep*` files to the desktop.** Their absence
  is the fix. Run the recovery merge where the repetitions actually are.
- **A removed output is invisible without clearing the fixture first.** The
  harness had this bug: the first run of the fourteen removals came back
  `EQUIVALENT`, which was false, because the previous run's files were still on
  disk. `clear_outputs()` fixes it — do not weaken it.
- **`--figures all` renders all 18.** Any script or habit that enumerates
  figure keys is working around a problem that no longer exists.
- **`reports.behavior_battery` is the figure battery; `layout.behavior_summary()`
  is the Stage 1 subject table.** They were one word apart until `d1a3284`, and
  the import path is the half that moved.
- The 6 MEG `erp_parcellation` failures are untouched and out of scope. They
  are a separate job.
