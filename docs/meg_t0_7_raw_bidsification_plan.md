# T0-7: Raw BIDSification (acquisition trees -> BIDS/sub-*/{meg,beh,anat})

Implemented in `meg_tokens/meg/raw_staging.py`, `meg_tokens/meg/meg_bids.py`,
`meg_tokens/meg/anat_bids.py`, `meg_tokens/behavior/tdms_bids.py`,
`meg_tokens/workflows/raw_staging.py`,
`meg-tokens meg stage-raw` / `meg-tokens meg apply-raw-staging`. The current
contract (manifest schema, BIDS entities, matching algorithm) is documented
in `docs/data_contract.md`'s "Stage 0: Raw BIDSification" section; this file
holds the underlying dataset evidence that section points back to.

`meg-tokens meg stage-raw` reads the acquisition sessions from
`ProjectConfig.raw_meg_root` (`data_root/raw`), the behavioral logs from
`ProjectConfig.behavior_root` (`data_root/tdms`), and the FreeSurfer
reconstructions from `ProjectConfig.subjects_dir`. All three come from the
project TOML with no per-command path override, so a staging plan is
reproducible from that file alone.

The manifest covers **all four staged kinds** -- `run`, `noise`,
`headshape`, `anat`. Only runs need matching; the other three are
single-file-per-subject discoveries where finding the file *is* the
decision, reported `found`/`not_found`. Keeping anat here rather than in a
separate `stage-anat` command is what makes one review pass sufficient: a
missing FreeSurfer reconstruction (H07, H10) appears as a `review` row
beside every other gap, instead of in a second command's separate output.

## Dataset evidence

- Raw sessions: `<raw-root>/<Subject>_DDM-tthiery_<YYYYMMDD>_<NN>.ds`,
  `NOISE_noise_<YYYYMMDD>_01.ds`, `<Subject>_DDM-tthiery_<YYYYMMDD>_HEADSHAPE.eeg`
  (+ fiducial `.jpg`s, ignored). Each subject has exactly one acquisition
  date (spot-checked H02/H10/H12/H20/H32).
- Each `.ds`'s `.hist` file has a `Trial duration: <seconds>` line -- the
  **nominal/configured** duration, unaffected by mid-recording truncation
  (confirmed against the one known truncated file,
  `H05_DDM-tthiery_20180220_02.ds`, 138s actual vs 315s nominal per
  `docs/meg_t0_6_subject_exclusion_qc.md`) -- plus a `DATE:` collection
  timestamp and `Sample rate: 1200`.
- Scanning all 412 real subject sessions' nominal durations gives exactly
  three clusters, matching `scripts/qc/meg_session_qc.py`'s
  `SLOWFAST_TRIAL_DURATION = 315.0`:
  - `315` (259 sessions) = Slow/Fast candidates (32 x 8 = 256 expected;
    **H01 has 10, H05 has 9** -- the only two subjects with extra
    candidates).
  - `135` (63 sessions) = RT candidates (32 x 2 = 64 expected; **H05 has
    only 1** -- the only subject short one).
  - `360`/`420` (90 sessions) = other protocols, never part of the 10
    canonical runs, always excluded from matching.
  - 30 of 32 subjects have exactly the expected 8-SF/2-RT candidate counts
    with zero ambiguity.
- The ingested behavior TSVs carry `nInitialTime` per trial. Its per-run
  minimum gives a clean, gapless chronological order across a subject's
  *entire* session (verified on H02: RT1, Slow1, Fast1, Slow2, Fast2, Slow3,
  Fast3, Slow4, Fast4, RT2 -- strict alternation, RT bookending the block, no
  clock resets).
- A raw session's Start-pulse count (`mne.find_events`, event code `262144`
  per `meg_tokens.meg.epoching.DEFAULT_EVENT_IDS`, respecting
  `SUBJECT_EVENT_OVERRIDES`) is the cross-validation signal recorded in
  every match's manifest row (spot-checked on `H02_..._03.ds` vs. the TDMS
  `Slow1` run: 56 Start pulses vs. 56 started trials from
  `meg_tokens.behavior.tdms.started_trials`, an exact match).

## The inter-trial-interval fingerprint

Duration class narrows a run's candidates to one protocol but cannot say
*which* Slow/Fast recording is `Slow2`. The identifying signal is timing:
the behavior log records each trial's `nInitialTime`, the recording emits a
trial-start pulse per trial, and although the two clocks share no origin,
the gaps between consecutive trials are the same physical intervals
measured twice. Scoring a candidate is therefore just comparing two
interval sequences (`meg_tokens.meg.raw_staging.fingerprint_candidates`),
scored over every relative shift so that a trailing dropout, a late
recording start, or both at once still score on the intervals the two
sequences genuinely share. Each shift's overlap must cover at least half
the shorter sequence, so a two-interval coincidence at the edge cannot
outscore a true full-length match.

Measured across the 17 `KNOWN_SESSION_OVERRIDES` runs on the real dataset,
the gap between right and wrong is not a gradient: the correct match scores
**0.39-0.57 ms** mean absolute error and beats its runner-up by
**144-892x**. That is why acceptance uses fixed thresholds rather than
"pick the best": `fingerprint_max_error_ms = 5.0` and
`fingerprint_min_separation = 20.0` sit in the empty middle of that gap,
and a run whose best candidate misses either one is reported `ambiguous`
rather than matched. A candidate that two runs both score best is likewise
refused for both -- self-contradictory evidence resolves to review, not to
the higher score.

The guards earn their keep on H05 RT1, whose recording simply does not
exist: its only RT-duration candidate is RT2's session, which it scores at
141.5 ms. A best-available matcher would have taken it. This one rejects it
and reports the run unmatched, which is the correct answer.

Reproduce with:

```bash
python scripts/qc/meg_tdms_fingerprint_matching.py \
    <raw-root> <tdms-root>
```

## H01 and H05: resolving the two subjects with extra/missing candidates

The original researcher's own conversion notebook,
[`55_CTF_to_FIF.ipynb`](../archive/replicated/DDM_scripts/scripts_new/55_CTF_to_FIF.ipynb)
(cell 6), gives an explicit, complete, session-by-session mapping for H01
(the notebook's own subject label there is `Pilot01`):

```text
01.ds=RT1  05.ds=Slow1  06.ds=Fast1  07.ds=Slow2  08.ds=Fast2
09.ds=Slow3  10.ds=Fast3  11.ds=Slow4  12.ds=Fast4  13.ds=RT2
```

This resolves H01's `Slow3` row exactly (`09.ds`) and confirms sessions
`02`/`03`/`04` are the extra SLOWFAST-duration candidates the matcher
correctly refuses to guess among (Thomas's own pipeline skips them too --
they're absent from his `filenames` list entirely). Cell 1's per-file trial
counts (e.g. `09.ds = Slow3 116`) also line up with the real Start-pulse
counts the matcher reports for that session. No equivalent per-session
listing exists in the archive for `H05` or any other subject (cells 2-4
cover the general subject group by date only, not by session number).

H05's case is structurally different: the real media only has **one**
135s-duration (RT) session for H05 (`{'135': 1, '315': 9, '360': 5}`), not
two. One of RT1/RT2 has no raw MEG counterpart on the media at all -- a
genuine recording gap, not an ambiguity to resolve. RT1 stays genuinely
unmatched (behavior-only, no MEG raw layer for that run).

Both mappings are hard-coded as `KNOWN_SESSION_OVERRIDES` in
`meg_tokens/meg/raw_staging.py`, the same pattern
`meg_tokens/meg/epoching.py`'s
`KNOWN_TRAILING_TRIAL_MISMATCHES`/`SUBJECT_EVENT_OVERRIDES` use for other
verified, subject-specific dataset exceptions.

They are no longer what resolves these subjects, though. The fingerprint
reproduces **all 17 entries** (8 for H01, 9 for H05) from the recordings
alone, with no reference to the notebook or to the count/position argument
above -- so the overrides now serve as a *pinned expectation*:
`match_subject` cross-checks every fingerprint result against them and
flags any disagreement for review rather than silently preferring either
source. They remain a fallback for a run the fingerprint cannot score at
all (e.g. an unreadable TDMS file). H05 RT1 is unaffected either way: it
has no raw counterpart on the media, so there is nothing for any method to
match it to.

## Testing

Unit tests build synthetic `.ds`-like directories (a `.hist` file with
controlled `Trial duration:`/`DATE:` text -- no real MEG binary data needed
for bucketing/ordering logic) and synthetic behavior-run tables to exercise:
fast-path matching, the H01-shape case (extra same-class candidate), the
H05-shape case (missing candidate, must report `ambiguous` not guess),
noise/headshape discovery, and manifest read/apply round-tripping with a
hand-edited `action` column -- none of which needs the raw acquisition tree,
so CI does not depend on it being reachable.

**Fingerprint behavior is tested on real recordings, not constructed pulse
trains**, because the property under test -- two independent clocks agreeing
to sub-millisecond precision on the same run and disagreeing by orders of
magnitude on any other -- is a fact about the acquisition hardware that a
constructed input would merely assume. These tests read the dataset directly
and skip unless both roots are exported:

```bash
MEG_TOKENS_RAW_ROOT=<raw-root> MEG_TOKENS_TDMS_ROOT=<tdms-root> \
    python -m pytest tests/test_meg_raw_staging.py
```

They cover: a real session's pulse train; H05, where every pinned override is
reproduced from the recordings alone; H05 RT1, where the only candidate must
be *rejected* rather than taken; and H12 Slow2, whose recording starts late
and ends long, so its pulse train is not a sub-sequence of the log in either
direction. `_best_window_error`'s shift semantics are additionally pinned by
two arithmetic unit tests on plain number sequences. `write_meg_bids` is tested
against a tiny synthetic `mne.io.RawArray` with a synthetic STIM channel,
asserting the resulting `BIDSPath` round-trips through
`mne_bids.read_raw_bids`. `write_beh_bids` is tested against a small
real-shaped synthetic `.tdms`-parsed DataFrame fixture.

Real-data verification:

```bash
meg-tokens --config <your tokens.toml> meg stage-raw --subjects H01 H02 H05 H10
```

Confirm the printed summary and manifest show H02/H10 fast-pathed and
H01/H05 resolved via `KNOWN_SESSION_OVERRIDES`, then run
`meg-tokens meg apply-raw-staging --subjects H01 H02 H05 H10` and confirm
`BIDS/sub-H02/meg/sub-H02_task-tokens_acq-slow_run-2_meg.ds` resolves via
`mne_bids.read_raw_bids`, `BIDS/sub-H02/beh/..._beh.tsv` has the expected
trial rows, and `meg-tokens meg preprocess --raw-path
.../BIDS/sub-H02/meg/..._meg.ds ...` runs unchanged.

Plus the standard phase-gate: `python -m pytest -q`,
`python -m compileall -q meg_tokens`.
