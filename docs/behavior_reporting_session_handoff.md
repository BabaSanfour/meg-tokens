# Behavior Reporting — Session Handoff

Notes for whoever (human or agent) picks this up next. Two sessions so far,
iterating on the behavior figure battery (`meg_tokens/reports/behavior/`)
figure by figure, on real data.

- **Session 1** built and styled F04-F06 and wrote their findings.
- **Session 2** removed F07, corrected the science in F04-F06 (two of the
  three findings were wrong as written), and brought the DT computation to
  preprint parity.
- **End of session 2 / machine handoff:** this work was developed on a
  desktop reachable only via remote desktop (AnyDesk) and had never been
  committed. Everything staged in `git status` at that point (see commit
  below) is this whole two-session arc — the `reports/behavior/` package
  split, the style/annotation modules, F04-F06's relayout and rewritten
  findings, F07's removal, and the motor-baseline preprint-parity fix.
  Committed and pushed to `origin/main` so work can continue from a laptop.
  The real dataset does **not** travel with git: `data_root` pointed at an
  external drive (`/media/karim/Hamza/meg-tokens`, exfat, not accessible
  from the laptop), so only the two derivative folders the whole reporting
  layer actually reads/writes were carried over by hand —
  `BIDS/derivatives/sub-group/beh/` (61M, Stage 2/2b group derivative
  tables — required, everything in `meg_tokens/reports/` reads only these)
  and `BIDS/derivatives/sub-group/fig/` (19M, already-rendered PNG/PDF/JSON
  outputs — reference only, fully regenerable from `beh/`). Confirmed by
  grep that no report builder reads per-subject or raw MEG/BIDS paths.
  `tokens.toml` is gitignored (local machine config) and was **not**
  copied — recreate it on the laptop from `config/tokens.toml.template`,
  pointing `data_root` at wherever `meg-tokens/` was placed locally (not
  the original `/media/karim/Hamza/...` path, which won't exist there).
  Full SSM fits (F01-F03's source data) were **not** brought over — if F01
  or F02 need regenerating rather than just reading existing tables, that
  requires `behavior ssm-fit`, which is expensive and wasn't in scope for
  this transfer.

**Start here if you are picking this up:** read "The audit that has to
happen next" below. F04-F06 have now been checked against the literature and
against the data; F01-F03 and F08-F26 have not, and two of the three
findings that *were* checked contained the same class of error. Do not treat
the unreviewed write-ups as trustworthy.

## Where session 1 started

The behavior report (`meg-tokens report behavior`) had just been split from
one file into `meg_tokens/reports/behavior/*.py` (distributions, design,
evidence, sequential, modeling, individual — 26 figures, F01–F26, indexed in
`docs/behavior_reporting_plan.md`). The user wanted to go through the
figures one at a time, on the real dataset (`tokens.toml`, data on
`/media/karim/Hamza/meg-tokens`, run via
`uv run meg-tokens --config tokens.toml report behavior --figures <key>`),
fixing layout problems and improving how results are presented, then writing
the actual scientific findings into `docs/behavior.md`.

Nothing here was reviewable from synthetic test fixtures alone — every real
bug in session 1 (axes collapsing to zero, text overlapping data, legends
covering curves) only showed up when a figure was actually regenerated on
the real 32-subject dataset and the PNG was read and visually checked,
including close crops. `pytest tests/reports/` passing is necessary but not
sufficient; it uses tiny synthetic fixtures that never exercise real layout
pressure.

Session 2 added the analogous lesson for the *science*: a finding written in
prose is not reviewable from the prose. Two of the three findings said
something the numbers printed in the same paragraph contradicted.

## What we did, per figure

### F04 `dtdistribution-condition` (Fast vs. Slow decision time) — done

- Scrubbed `dt_ms` out of all display labels ("Decision time (ms)", not
  "Decision time, dt_ms (ms)").
- X-axis capped at 0–3000ms, decluttered to 2–3 ticks; tick *marks* removed
  globally but numeric labels kept (see style rules below).
- Session 1 fixed a real bug here (the pooled-trial KDE overlay went
  flat once the x-axis was clipped, because each condition had its own
  `twinx()` autoscaled against the full unclipped range). **Session 2 then
  deleted the overlay entirely** and `_pooled_kde_overlay` with it: the
  finding is that the *gap between the two curves widens*, and a filled
  density behind them is the wrong background for judging a widening gap.
- Found and fixed a pipeline-wide bug (not F04-specific): every figure in
  the whole battery was silently saving at matplotlib's 100dpi default
  instead of the intended 400dpi, because `savefig.dpi` only applies inside
  `style.apply_publication_style()`'s `rc_context`, and the actual
  `fig.savefig()` call happens later, in `behavior_summary.py`, after that
  context has exited. Fixed by exporting `style.SAVEFIG_DPI` and passing
  `dpi=style.SAVEFIG_DPI` explicitly in `behavior_summary._render_one`.
- Panel B iterated through several designs: Gardner-Altman companion axis
  (rejected — user meant to reference F05's style, not F02's) → per-quantile
  full-text annotation (broke: text wider than the panel, collapsed
  `constrained_layout`) → settled on a single bracket between Fast/Slow with
  `Δ = value marker` (see conventions below).
- Added two-color subject lines to `paired_slope` (`panels.py`):
  `increase_color`/`decrease_color` params, gray for "with the group
  direction", red (`CONDITION_COLORS["fast"]`) for "against it" — makes
  individual reversals visible at a glance.
- Panel width ratio settled at A:B = 2.7:1.3 (not equal split), panel taller
  than the shared default (4.6in vs. 2.85in).
- **Session 2 relayout.** A is now the two quantile curves alone, x capped
  to 500-2000 ms; B became a `raincloud` (was `paired_slope`) so the
  distribution stays in the figure and B matches F05 panel A's chart type.
  The old `skew n.s.` note is gone -- skewness is invariant under *any*
  positive linear transform, so it cannot distinguish a fixed delay from a
  stretch and printing it invited reading a null as evidence. It is replaced
  by the CV note (`CV 0.40 -> 0.39, fixed delay predicts 0.35`), which does
  discriminate. Also fixed a latent bug: the significance-marker x-clamp was
  hardcoded at 2950 ms for the old window and is now derived from
  `display_range`.
- Finding written to `docs/behavior.md` → "Findings". It was originally
  written as "a pure shift, not a tail effect"; that was **wrong and has
  been rewritten** as a proportional stretch (see the F07 section below for
  how it surfaced). The section is now "Fast vs. Slow stretches decision
  time proportionally, rather than adding a fixed delay".

### F05 `dtdistribution-class` (decision time by trial class) — done

- Same style pass as F04 (bigger fonts, decluttered ticks, `dt_ms` scrubbed).
- `raincloud()` (`panels.py`) extended with the same `increase_color`/
  `decrease_color` params as `paired_slope`, but **per-leg**, not
  per-subject — a subject can go up easy→ambiguous and down
  ambiguous→misleading in the same figure, and both legs matter.
- Brackets on panel A: same bug as F04 (full `t(31)=…, p=…` text collapsed
  the axes — two of the three brackets span only one category, ~1/3 of the
  panel). Fixed with the `Δ = value marker` convention.
- Panel B went through a detour: added a pooled-trial KDE overlay + zoomed
  x-range (500–2200ms), then the user asked to remove it again — panel A's
  violins already carry the distribution-shape story, so B stays to just
  the quantile functions. (Its x-range is 500–2100 ms in the shipped code;
  F04's panel A now uses 500–2000, so the two are near but not identical.
  Worth unifying if anyone touches both.)
- Legend placement needed an actual visual check, not just a guess:
  "lower right" clips through the ambiguous/misleading curves for this
  chart's shape (the curves rise toward high-x at moderate y); moved to
  upper-left, which is genuinely empty for this specific curve shape.
- Suptitle: dropped "(unclassified trials excluded)" from the plotted title;
  moved to the registry's `caveat` field instead (still surfaces in
  `--list-figures` and the JSON sidecar).
- Finding written to `docs/behavior.md` → "Decision time by trial class:
  difficulty changes the shape of the distribution, not just its location"
  — includes the preprint-reversal writeup (misleading is reliably faster
  than ambiguous here; the preprint found the opposite, marginally, under a
  confounded classification — see "Trial-classification reference frame" in
  the same doc for the mechanism).
- **Retitled after an audit.** It was originally "easy is a whole-
  distribution shift…", using the same loose "shift" wording that turned
  out to be wrong for F04. Running F04's geometry tests on each class
  contrast showed easy-vs-ambiguous is *neither* a delay nor a stretch (a
  real shape change; 9-11/32 subjects still differ after either
  normalization), while easy-vs-misleading is a clean ~1.30x stretch
  (0/32 after dividing by the median). Also flagged there: only 5/32
  subjects show any distributional ambiguous-vs-misleading difference, so
  that contrast is a small consistent effect, not a large one.

### F06 `spdcumulative-class` (SPD at decision, cumulative by class) — done

- Started as two panels (`all_logged` vs. `validated_15row` views). Checked
  whether they actually said the same thing before dropping one — they do
  (proportions land within a few points of each other despite
  `validated_15row` excluding up to 40% fewer trials per class from
  unvalidated "14-row" logs). Collapsed to one panel, kept
  `validated_15row` (the higher-integrity subset).
  **`all_logged` is still used elsewhere** — `summary`, `individual.py`
  (deliberately, a separate choice), `performance.py` — untouched, flagged
  as TBD in the registry `caveat`, not changed.
- Same `Δ = value marker` fix needed again (full stat text collapsed the
  axes on the narrower single-width panel).
- Width iterated: single (3.42in) → double (7.09in, "make it wider") →
  a custom 5.7in ("reduce a bit"). This required extending
  `style.figure_grid`'s `width` param to accept a raw float in inches, not
  just the `"single"`/`"double"` literals — backward compatible, every
  other call site unaffected.
- X-axis progressively cropped (0 → 0.2 → 0.3 → back to 0.2) as the curves
  are flat/uninformative below there for every class; guide lines at
  0.4/0.6/0.8 removed per request.
- Legend: replaced a text note ("dashed = pooled...") with actual gray
  `Line2D` proxy legend entries (solid = mean of subjects, dashed = pooled
  across trials) — cleaner than prose, matches how class rows already key
  color.
- Its entire section was **removed** from `docs/behavior_reporting_plan.md`
  (not trimmed to a pointer — deleted outright), per explicit request,
  since that doc is transitional and slated for eventual removal.
- Finding written to `docs/behavior.md` → "Success probability at decision:
  easy resolves at higher confidence; ambiguous and misleading resolve at
  the same confidence despite different timing".

### F07 `dtdistribution-exgaussian` — removed, question closed

**Resolved: the figure and its whole analysis layer were deleted.** The
open question recorded here (was the shared-sigma fix correct, given that
ambiguous got worse?) dissolved rather than being answered, so the
`fixed_sigma` / `_subject_shared_sigma` work described in earlier drafts of
this file no longer exists in the tree.

Three independent reasons, any one sufficient:

- **No precedent.** The tokens-task literature never fits a distribution to
  DT. Cisek, Puskas & El-Murr (2009), Thura et al. (2012), Thura & Cisek
  (2014, 2017) and Carland et al. (2016) all compare whole DT distributions
  with per-subject two-sample KS tests and report the count of significant
  subjects; parametric machinery goes into the *model* (UGM vs. DDM), never
  into a descriptive fit. The preprint (Thiery, Rainville, Cisek & Jerbi,
  2022) uses paired t-tests on subject means and no distribution fitting at
  all.
- **Not identifiable at these trial counts.** Strata here hold 54-128 trials
  per subject. Lacouture & Cousineau's (2008) recovery study puts sigma's
  95 % interval at [0, 141] for a true sigma of 100 at n = 50. The
  corr(mu, tau) ≈ -0.6 table recorded in earlier drafts of this file is the
  textbook small-n signature, not a discovery about this cohort.
- **Not interpretable even when well estimated.** Matzke & Wagenmakers
  (2009) fit the ex-Gaussian to diffusion-generated data: tau moves with
  both drift rate *and* boundary separation. It licenses no claim about
  which mechanism changed.

DT here is additionally right-censored by the 15-jump deadline, which biases
exactly the tail parameter tau estimates.

**What replaced it.** Quantile contrasts (already in `dtdistributionstats`)
answer the same shape questions without the identifiability problem, and
per-subject KS tests are available cheaply from `trialfeatures` if a formal
shape test is wanted — that is Cisek's own idiom (2009 Fig. 3C: "Most
subjects (20 of 22) showed significantly faster responses in fast versus
slow blocks").

**Knock-on: F04's finding was wrong and has been rewritten.** Removing tau
also removed the prop under F04's "pure shift" claim, and checking it
directly showed the claim was wrong as written -- the quantile shifts are
79/129/148 ms, not "the same amount" (p = .021), while the *ratios* are
constant (p = .167), CV is invariant (p = .62), and per-subject KS goes
18/32 raw -> 6/32 median-subtracted -> 2/32 median-*divided* (chance 1.6).
Fast -> Slow is a ~1.12x stretch of the deliberation clock, not an additive
delay -- a stronger result for urgency gating, since a constant added to
every trial would be non-decision time, which the motor-baseline subtraction
has already removed. Rewritten in `docs/behavior.md`; the per-subject model
comparison is underpowered (p = .38) and the write-up says so.

**Lesson worth keeping.** The bad claim was visible in its own numbers the
whole time -- "all three move together by about the same amount: 677->756,
1127->1255, 1768->1916" lists 79, 129 and 148 ms in the same sentence that
calls them the same. When a finding states a pattern, check the pattern
against the numbers printed next to it before trusting the sentence.

## Session 2: decision time now matches the preprint exactly

`calculate_motor_baseline` (`meg_tokens/behavior/features.py`) previously
pooled every RT trial and took one mean. It now averages *per-run* means,
which is what the preprint notebook does
(`archive/replicated/DDM_scripts/scripts_new/00_44_Behavior_Trial_Types.ipynb`:
`mean_RT_mean = np.mean(np.array(mean_RT))` over per-run means). The two
differ whenever the RT runs hold unequal numbers of usable responses.

Deliberately **not** changed: H02 still uses both RT runs, where the preprint
uses RT2 only. That is a documented data-quality call (`docs/meg.md`, "H02
motor baseline") and the user reaffirmed it. Effect of the whole
motor-baseline question is ~1.4 ms on absolute group means and exactly zero
on any paired within-subject contrast.

Confirmed while verifying this: the project's DT definition already matched
both Cisek and the preprint. `rawRT = tEnterTarget - tGO` (`tables.py:62`),
`dt = rawRT - motor_baseline` (`features.py`), which is Thura et al. (2012)
Eq. 31, `DT = RT_VMD - RT_CMD`. Note the consequence, which matters for
interpretation: **non-decision time is already subtracted out**, so a
constant added to every DT has no mechanism left to point at.

## The audit that has to happen next

Session 2 found that **two of the three reviewed findings asserted a
mechanism from a statistic that cannot distinguish the mechanisms.** All
three errors were visible in numbers already printed in the same paragraph.
This is the single most important thing to carry forward.

| Where | Claimed | Actually |
|---|---|---|
| F04 | "all three quantiles move by about the same amount" | 79 / 129 / 148 ms — Δq90 vs Δq10 p = .021. It is a ~1.12x stretch, not a delay |
| F04 figure | "shape unchanged: skew n.s." | skewness is invariant under both models; it was never evidence |
| F05 | "easy is a whole-distribution shift" | easy-vs-ambiguous is neither a delay nor a stretch — a real shape change |
| F06 | "completely null", "statistically identical" | p > .05 is not equivalence; TOST bounds it at dz < 0.36 |

**The reusable method.** For any two decision-time distributions, ask which
of three things the difference is — a fixed delay, a proportional stretch,
or a genuine shape change — using:

1. **Quantile deltas vs. ratios.** Is Δ constant across quantiles (delay)?
   Is the ratio constant (stretch)? Test Δq90 vs Δq10 and log-ratio q90 vs
   q10 as paired t-tests.
2. **The per-subject KS ladder** — the decisive one. Count subjects whose
   two distributions differ (a) raw, (b) after subtracting each side's
   median, (c) after dividing by it. A delay collapses (b) to chance; a
   stretch collapses (c). Chance is 0.05 x n_subjects. This is also Cisek's
   own idiom (2009 Fig. 3C reports "20 of 22 subjects").
3. **Spread.** A delay leaves SD alone and drives CV down; a stretch scales
   SD and leaves CV alone. Compare observed against each model's per-subject
   prediction.
4. **For any null**, report a TOST/equivalence bound, never "no difference".

Session 2 implemented all of this as throwaway scripts against the real
derivatives (`shift_vs_scale.py`, `norm.py`, `rescale.py`, `cv_vs_skew.py`,
`cv_persubject.py`, `ks_check.py`, `audit_f05_f06.py`). They lived in a
session-scoped scratchpad and are **probably gone by the time you read
this** — they were ~40 lines each reading
`sub-group_task-tokens_desc-trialfeatures_beh.tsv` and
`...desc-dtdistribution_beh.tsv` directly with pandas/scipy, so rewriting
them is quicker than hunting for them. If this audit becomes routine, the
right move is to promote the KS ladder into `behavior/analyses/` as a real
derivative rather than keep rewriting it.

**Nothing in F01-F03 or F08-F26 has had this treatment.** Given the hit rate
so far, assume those write-ups contain the same class of error until checked.

## Rejected: merging F04 and F05

Prototyped at `scratchpad/merged_f04_f05.py` (self-contained, reads the real
derivatives, writes a PNG). One raincloud panel — Fast, Slow, dashed
divider, Easy, Ambiguous, Misleading — plus one panel with all five quantile
functions, conditions dashed.

It works and looks good as an *overview*, but the user chose to keep F04 and
F05 separate, and the reasons are worth recording:

- Both findings are about the gap between two specific curves, and both
  become unreadable among five. F04's stretch and F05's
  ambiguous-vs-misleading convergence are shown by their own figures and
  merely asserted by the merged one.
- Conditions and classes are **not disjoint** — every class trial is also a
  Fast or Slow trial — so one axis invites a Fast-vs-Easy read that is not a
  contrast. Dashed-vs-solid mitigates but does not prevent it.
- Cisek keeps them separate too (2009 Fig. 3 = condition, Fig. 4 = class,
  identical grammar, two figures).

Two implementation gotchas found there, if anyone revives it: `raincloud`
connects subjects across *adjacent* categories, so a single five-position
axis draws a meaningless connector from each subject's Slow mean to their
Easy mean (use two sub-axes sharing a y-scale); and `errorbar` puts the
series label on the `ErrorbarContainer`, not the `Line2D`, so
`ax.get_lines()` reports every curve as `_nolegend_`.

## Conventions established across both sessions (apply to F08+ too)

These aren't one-off fixes — they're patterns that showed up more than once
and should be the starting point for every remaining figure, not
rediscovered from scratch.

**Style (`meg_tokens/reports/style.py`)**
- Fonts are bigger than the original design across the board: `font.size`
  13, `axes.labelsize` 16, `axes.titlesize` 18, tick labels 13, legend 13
  (all in `_PUBLICATION_RC`). `panel_label` (A/B/C letters) at fontsize 17.
  `annotate_stat_block`'s default text size is 9 (`annotations.py`).
- Tick *marks* are removed globally (`xtick.major.size`/`ytick.major.size`
  = 0) but numeric tick *labels* are always kept — declutter by reducing
  *how many* ticks (usually 2–3 per axis), never by hiding the numbers.
- `SAVEFIG_DPI` (400) must be passed explicitly to every `fig.savefig()`
  call outside `apply_publication_style()`'s context — the rc value doesn't
  survive past the `with` block. Already fixed centrally in
  `behavior_summary.py`; don't reintroduce a bare `fig.savefig(path)` call
  anywhere in the reporting pipeline.
- `style.figure_grid`'s `width` param accepts `"single"`, `"double"`, or a
  raw float (inches) for one-off widths that don't fit either preset.
- Default `panel_height_in` (2.85) is usually too short at these font
  sizes — most figures touched this session bumped it to 3.4–4.6 depending
  on content. Check the real render before assuming the default is enough.

**Annotation**
- **The standard convention for any bracket/comparison annotation is
  `Δ = value [unit] marker`** (marker from `significance_marker()`, e.g.
  `***`/`n.s.`) — never spelled-out `t(df) = …, p = …` text on the figure
  itself. Hit the same bug three separate times (F04 panel B, F05 panel A,
  F06) before this became the fixed rule: full stat text is wide enough to
  collapse `constrained_layout`'s axes to zero width on anything narrower
  than a full double-width panel. The exact t/p/dz values still live in the
  derivative table and the JSON sidecar — they don't need to be spelled out
  on the plot.
- Any multi-line or long annotation string must be wrapped/shortened to fit
  the *specific panel's actual column width*, not the full figure width.
  This is the single most common bug this session — an unwrapped long line
  doesn't just visually overflow, it distorts `constrained_layout`'s space
  allocation for that axes (can happen vertically *or* horizontally
  depending on where the text sits).
- Non-essential methodological detail (e.g. "unclassified trials excluded",
  "validated_15row only, all_logged still used elsewhere") belongs in the
  `FigureSpec.caveat` field, not the plotted title — it still reaches
  `--list-figures` and the JSON sidecar.
- Before placing a legend or text box in a corner, check what's *actually*
  empty for that specific chart's data shape — don't assume any corner is
  safe by default. "Lower right" was safe for F04 (paired scatter, two fixed
  x-positions) but clipped curves on F05 (rising quantile functions) and F06
  (rising cumulative curves with a wide legend). A legend's own footprint
  (more entries or longer labels = bigger box) can intrude into a corner
  that's nominally empty of data.
- Two-color "direction" line convention (`paired_slope`, `raincloud` in
  `panels.py`): gray (`style.SUBJECT_LINE`) = consistent with the group's
  direction, red (`style.CONDITION_COLORS["fast"]`, reused deliberately to
  stay inside an already-validated color scope rather than introducing a
  new one) = against it. For `raincloud` this is per *leg* (each pair of
  adjacent categories), not per subject overall.

**Scientific claims (added session 2)**
- Never describe a set of numbers with a word the numbers do not support.
  Before writing "the same", "unchanged", "identical", or "shift", compute
  the comparison that word implies and cite its p-value.
- A statistic that is invariant under the hypotheses being compared is not
  evidence for either. Skewness cannot separate a delay from a stretch; CV
  can. Check that the statistic *moves* under one hypothesis and not the
  other before it goes in a figure or a Result paragraph.
- `p > .05` is never "no effect". Report an equivalence bound (TOST, or the
  90% CI) so the claim is "we exclude effects larger than X", which is
  checkable.
- Group-level significance and per-subject consistency are different claims.
  Report both when they diverge: ambiguous-vs-misleading is p = 7e-6 at the
  group level and detectable in only 5/32 individual subjects, and a reader
  seeing only the first would infer the wrong thing.
- Cross-references rot. When a finding is rewritten, grep for every other
  section that describes it ("shape-invariant shift", "same signature as
  ...") and fix those too.

**Process**
- Real bugs only show up on real data. After any change: run
  `uv run pytest tests/reports/`, then regenerate the actual figure
  (`uv run meg-tokens --config tokens.toml report behavior --figures <key>`),
  then `Read` the resulting PNG — and crop closely around anything
  suspicious (legends, brackets, corners) rather than trusting a full-figure
  thumbnail glance.
- When a builder stops reading a derivative or column, remove it from that
  `FigureSpec`'s `requires` tuple and the metadata's `columns_read` — keep
  the registry honest.
- When two "views"/panels look redundant, *check they actually agree*
  numerically before dropping one (done for F06's `all_logged` vs.
  `validated_15row`) — don't assume redundancy from a visual glance alone.
- `docs/behavior.md` → "## Findings" is the **only** place Result/
  Interpretation write-ups with real numbers belong (bold **Result.** /
  **Interpretation.** paragraphs, ending with `Figure: <key> (F0X).`).
  `docs/behavior_reporting_plan.md` is design/plan documentation only
  (Reads/Layout/Chart-type justification/Annotation) — keep it in sync with
  what actually shipped when it drifts, but never duplicate the Results
  numbers there, not even as a cross-reference pointer (that doc is
  transitional and will eventually go away — for F06 its whole section was
  deleted outright, not pointer-trimmed, once the finding had a home in
  `behavior.md`). `docs/behavior_roadmap_results.md` is being trimmed of
  duplicated narrative the same way as it's found.

## Status: layout reviewed vs. science audited

| Figure | Key | Layout | Science |
|---|---|---|---|
| F01 | `ssmcomparison-deltabic` | not reviewed | **not audited** — source of the H1 numbers cited in F04 |
| F02 | `ssmcomparison-urgencyscale` | not reviewed | **not audited** — source of the H2 numbers cited in F04 |
| F03 | `ssmcomparison-urgencyparams` | not reviewed | **not audited** |
| **F04** | `dtdistribution-condition` | done (relaid out session 2) | **audited & rewritten** — proportional stretch |
| **F05** | `dtdistribution-class` | done | **audited & rewritten** — shape change, not a shift |
| **F06** | `spdcumulative-class` | done, one panel/view | **audited & rewritten** — equivalence bound |
| F07 | — | **removed** | figure, analysis layer, tests and plan section all deleted |
| **F08** | `conditionclass-anova` | done (CI95 bands, no traces) | **audited & rewritten** — log-scale interaction; separability |
| **F09** | `choiceside-asymmetry` | done (3 difference panels, was 3x3) | **audited** — no side bias; 23 ms left-hand speed advantage |
| F12 | see `docs/behavior_reporting_plan.md` | not reviewed | **not audited** |
| **F10** | `timeontask-drift` | done (CI bands, within-block deciles) | **audited & rewritten** — drift bounded; class-scheduling confound |
| **F11** | `conditionorder-balance` | done (widened, compact stats) | **audited & rewritten** — within-group tests; null bounded |
| **F13** | `summary-cohort` | done (4 panels, shared subject axis) | **audited** — cohort composition |
| F14–F26 | see `docs/behavior_reporting_plan.md` | not reviewed | **not audited** |

Registry is 25 figures after F07's removal (`--list-figures` to confirm).

## Practical pointers

- Real data root: `tokens.toml` → `data_root = "/media/karim/Hamza/meg-tokens"`.
  Derivatives live under `<data_root>/BIDS/derivatives/sub-group/{beh,fig}/`.
- Regenerate one figure: `uv run meg-tokens --config tokens.toml report behavior --figures <key>`.
- Regenerate the distributional/design derivatives (not the expensive SSM
  fits, which are pooled from a prior `behavior ssm-fit` run, not
  recomputed): `uv run meg-tokens --config tokens.toml behavior characterization`.
- List all figure keys/groups: `uv run meg-tokens --config tokens.toml report behavior --list-figures`.
- Report test suite: `uv run pytest tests/reports/ -q` (58 passed, all
  synthetic-fixture based).
- Behavior analysis test suite: `uv run pytest tests/behavior/ -q`
  (422 passed, 5 skipped).
- **Known-failing, unrelated:** `tests/test_batch_erp_parcellation.py` has 6
  failures in the MEG/ERP path (`workflows/erp.py` -> `behavior/tables.py`
  -> `schema.py`, "Token directions must contain only 1 and 2" — a bad test
  fixture). Confirmed pre-existing by stashing the behavior-module changes
  and re-running. Do not chase these from the reporting side.
- After changing anything upstream of `dt_ms` (e.g. the motor baseline),
  rerun `behavior analyze` *then* `behavior characterization` — the first
  rebuilds `trialfeatures`, the second everything derived from it.
