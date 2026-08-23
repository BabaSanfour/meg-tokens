# Behavior cleanup — handoff

Working session of 2026-08-23. **Nothing is committed.** The working tree holds
48 modified/deleted files plus three new untracked paths. Everything below was
verified before this file was written; the verification commands are in §4.

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

Current state: **EQUIVALENT**, 590 passed / 29 skipped / 6 failed. All six
failures are in `tests/test_batch_erp_parcellation.py`, predate this work, and
are outside behavior scope.

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

**875 → 625 collected** (754 → 537 test functions; 15,046 → 13,728 lines).

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

**Net: 48 files changed, 3,160 insertions, 5,610 deletions (−2,450 tracked
lines), plus three new untracked paths.**

---

## 2. Next steps, in priority order

### Step 1 — Commit the verified state (do this first)

The tree is verified but bisectable only if it lands in coherent pieces. The
suggested split, in dependency order:

1. `fix(behavior): repair the ssm-fit array flag and pin scripts to the CLI` —
   `scripts/ssm_fit_array.sh`, `tests/test_cluster_scripts.py`
2. `refactor(behavior): drop fourteen derivatives nothing reads` —
   `workflows/behavior_characterization.py`, `workflows/sequential_sampling.py`,
   `analyses/evidence.py`, `analyses/design_effects.py`,
   `analyses/sequential_sampling.py`, `docs/behavior.md`
3. `refactor(cli): replace the dispatch chain with a handler table` —
   `cli/main.py`, `tests/test_cli.py`, minus `tests/reports/test_cli_wiring.py`
4. `refactor(scripts): name every launcher for the command it submits`
5. `refactor(behavior): remove dead helpers and signature-restating docstrings`
6. `test(behavior): group the suites by behaviour rather than by assertion`
7. `feat(scripts): add the before/after equivalence harness` +
   `docs: document the harness and the recovery-merge hazard`

Also delete branch `worktree-agent-a993d156c19a4c105` once reviewed.

### Step 2 — Finish the test trim (~625 → ~450 collected)

Six files have not had the grouping treatment. Same method as §3.

| File | tests | lines |
| :--- | ---: | ---: |
| `tests/reports/test_remaining_figures.py` | 21 | 473 |
| `tests/behavior/test_analyses_individual.py` | 26 | 411 |
| `tests/test_behavior_workflow.py` | 15 | 403 |
| `tests/behavior/test_classification.py` | 21 | 361 |
| `tests/behavior/test_analyses_sequential.py` | 20 | 366 |
| `tests/behavior/test_analyses_performance.py` | 25 | 345 |
| `tests/behavior/test_analyses_summary.py` | 19 | 218 |

`tests/reports/factories.py` (638 lines, no tests) is the largest single file
left and is pure fixture construction — worth a read for redundancy after the
report tests are grouped, since grouping may strand parts of it.

### Step 3 — Finish the source trim

Only partly done. What is left, in descending value:

- **Docstring scaffolding.** ~4,500 docstring lines remain in the behavior
  stack (35% of it). The 25 pure-restatement `Parameters` blocks are gone;
  `Returns` sections that only name the type (`pandas.DataFrame`) with no
  content are the same class of thing and were not swept. **Do not touch
  `Notes` sections** — that is where the scientific reasoning lives, and it is
  why the pipeline is auditable.
- **Unreachable defensive branches** inside `sequential_sampling.py` (3,700
  lines), `evidence.py` (2,030) and `design_effects.py` (1,179). Not yet
  audited. Note that the 60 `require_columns` calls are *not* this: each names
  the columns that analysis needs and produces the "re-run `behavior analyze`"
  message, so they were deliberately kept.
- **`reports/behavior_summary.py`** is misnamed — it is the figure-battery
  orchestrator, not a summary report. Renaming touches `cli/main.py` and two
  test files.

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

Run all four before committing anything. Expected results are stated.

```bash
# 1. Test suite — expect 590 passed, 29 skipped, 6 failed.
#    All six failures are pre-existing, in tests/test_batch_erp_parcellation.py.
python -m pytest -q

# 2. Pipeline equivalence — expect "EQUIVALENT".
python -m scripts.equivalence.harness run candidate
python -m scripts.equivalence.harness compare baseline candidate

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
- The 6 MEG `erp_parcellation` failures are untouched and out of scope. They
  are a separate job.
