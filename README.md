# MEG Tokens Task Analysis (DDM Project)

Analysis library for decision-making dynamics on the **Tokens Task** with MEG. `archive/` iEEG notebooks are out of scope.

## Project Structure

*   **`meg_tokens/`**: package.
    *   `core/`: subject/run identifiers, project config, workflow result models.
    *   `behavior/`: TDMS parsing, behavioral metrics.
    *   `meg/`: raw staging, coregistration, two-stream preprocessing (DSS and ICA), epoching, source localization.
    *   `features/`: ERP, power, spectral, Hilbert, PAC, connectivity estimators.
    *   `analysis/`: statistics, decoding, PCA, dPCA estimators.
    *   `workflows/`: filesystem-aware processing/analysis stages.
    *   `reports/`: figures, report tables.
    *   `cli/`: the `meg-tokens` command.
    *   `io/`: derivative layout, `.npy`/JSON/xarray/table contracts.
    *   `validation/`: golden-reference and SPD comparisons (`meg-tokens validate`).
*   **`workflow/`**: Snakemake DAG, local/Slurm profiles.
*   **`tests/`**: unit tests.
*   **`docs/`**: [`behavior.md`](docs/behavior.md) (behavioral results, figures and their conventions), [`meg.md`](docs/meg.md) (MEG pipeline specification, deviations from the paper, open decisions, alignment QC and subject exclusion), [`data_contract.md`](docs/data_contract.md) (persisted data and metadata).
*   **`pyproject.toml`**: package metadata/deps.
*   **`archive/`**: raw scripts copied from the external drives -- `DDM_scripts/` (`scripts_new/` Python/Jupyter + `matlab_scripts/`), `DDM_analysis_scripts/` (Jupyter, from `DDM/scripts/`).

Dependencies point down this list and never back up: `core` → `io` → domain
packages (`behavior`, `meg`, `features`, `analysis`, `reports`) → `workflows` →
`cli` → `workflow/`. The CLI holds no scientific calculation or derivative
naming; domain modules never import workflows or the CLI; the Snakefile invokes
only `meg-tokens` commands, and its profiles change resources, never scientific
parameters. There are no `utils` modules. The boundary test enforces this.

## Pipeline Execution Flow & Module Map

Module map, in pipeline order. Stage numbers live only under [Running the Pipeline](#running-the-pipeline-cli), not here.

| Stage | Module | Purpose |
| :--- | :--- | :--- |
| **Raw BIDSification** | [`raw_staging.py`](meg_tokens/meg/raw_staging.py), [`meg_bids.py`](meg_tokens/meg/meg_bids.py), [`anat_bids.py`](meg_tokens/meg/anat_bids.py), [`tdms_bids.py`](meg_tokens/behavior/tdms_bids.py) | Identifies which raw session is which run, then stages the whole raw layer: `BIDS/sub-*/{meg,beh,anat}` and `sub-emptyroom`. |
| **Behavioral Log Parsing** | [`classification.py`](meg_tokens/behavior/classification.py), [`schema.py`](meg_tokens/behavior/schema.py), [`behavior_ingest.py`](meg_tokens/workflows/behavior_ingest.py) | Reads the staged raw behavior, infers trial classes, validates the Stage 1 contract, and writes behavior derivatives. |
| **Behavioral Metrics Extraction** | [`features.py`](meg_tokens/behavior/features.py) and [`performance.py`](meg_tokens/behavior/analyses/performance.py) | Computes choice RTs, accuracy, difficulty levels, and behavioral summaries. |
| **Behavioral Characterization Analyses** | [`behavior_characterization.py`](meg_tokens/workflows/behavior_characterization.py) and [`behavior/analyses/`](meg_tokens/behavior/analyses) | Runs the full analysis battery described in `docs/behavior.md` over the trial-feature table: distributions, design/session effects, evidence, criterion decline, individual differences, and more. |
| **Behavioral Reporting** | [`behavior.py`](meg_tokens/reports/behavior.py) | Renders performance diagnostics and RT distributions. |
| **MEG Preprocessing** | [`preprocessing.py`](meg_tokens/meg/preprocessing.py) | One shared trunk per recording (compensation, ZapLine, band-pass, bad channels and segments), then two cleaners -- DSS and ICA -- each persisted with its artifact estimate. |
| **Epoch Extraction & Event Alignment** | [`epoching.py`](meg_tokens/meg/epoching.py) | Segments MEG data into trial epochs aligned to triggers and behavior. |
| **Neural Source Localization** | [`sources.py`](meg_tokens/meg/sources.py) | Computes noise covariance, BEM models, sets up source spaces, and applies minimum-norm inverses. |
| **Source-Space Time-Frequency Power** | [`time_frequency.py`](meg_tokens/features/time_frequency.py) | Extracts band power from source estimates using modern MNE calls. |
| **ERP Slicing, Parcellation, & Export** | [`erp.py`](meg_tokens/features/erp.py) | Aligns, pads, parcellates, and exports labeled `.npy` arrays with JSON sidecars. |
| **Group Analyses** | [`analysis/`](meg_tokens/analysis) | Provides permutation statistics, decoding, PCA, and dPCA estimators. |
| **Workflow Orchestration** | [`Snakefile`](workflow/Snakefile) | Schedules subjects, runs, shared source models, features, analyses, and reports. |

## 💾 Data Locations

> [!NOTE]
> Raw data lives outside this repo.

Data root `meg-tokens`:

*   **`raw/`** — raw CTF sessions (`.ds`), `NOISE_noise_*.ds` empty-room, headshapes, fiducial
    photos. Read-only; only copied into `BIDS/` (`meg-tokens meg stage-raw`).
*   **`tdms/`** — LabVIEW behavioral logs, all 32 subjects (`H1`-`H32`).
*   **`BIDS/`** — Stage 0's raw layer (`sub-*/meg`, `sub-*/beh`, `sub-*/anat`,
    `sub-emptyroom`) + `BIDS/derivatives/`.
*   **`IRM/`** — FreeSurfer `recon-all` output (30/32; H07, H10 missing). Default
    `subjects_dir` (`data_root/IRM`); override in the project TOML.

`data_root` set once in a TOML ([`config/tokens.toml.template`](config/tokens.toml.template)),
passed via `--config`; `raw_meg_root`/`behavior_root`/`bids_root`/`subjects_dir` derive from it.
Relative paths resolve from the config file's location.


## Machines, environments, and execution rules

Two machines run this project, and the differences between them have caused
real failures. Read this before running anything on the cluster.

### `fir` (Alliance cluster)

Checkout `~/meg-tokens`, data root `/scratch/hamza97/meg-tokens`. The venv is a
**real directory** at `~/meg-tokens/.venv` — not a symlink; it was rebuilt on
2026-08-22 after the previous one turned out to be a dangling symlink into a
deleted `/project/.../meg-tokens-venv`:

```bash
module load StdEnv/2023 python/3.13
python -m venv .venv
./.venv/bin/pip install -e '.[modeling,dev]'
```

Python 3.11 is the floor (`mne-denoise` 0.0.3 requires it) and 3.13 is what
the desktop runs; add `workflow` to the extras to get Snakemake 9 and its Slurm
executor plugin.

Two traps worth stating plainly:

- **`uv` is not available on `fir`**, so `uv.lock` is unusable there. Install
  from `pyproject.toml` with pip.
- **`pyddm` lives in the `modeling` extra.** A bare `pip install -e .` yields an
  environment that imports fine and cannot fit a single model.

### Configuration

**`tokens.toml` is per machine and gitignored. Never sync it.** The desktop's
`data_root` is `/media/karim/Hamza/meg-tokens`; the cluster's is
`/scratch/hamza97/meg-tokens`.

### Syncing between checkouts

**Never use a deleting rsync, `git reset`, or a force-overwrite against either
checkout.** Inspect both local and remote `git status` first — the cluster
carries its own untracked `logs/`, and losing those costs a re-derivation.

### Where each stage runs

- **Report regeneration is local**, from the copied group derivatives:
  `uv run meg-tokens --config tokens.toml report behavior --figures <key>`.
- **Characterization and model fitting run on the cluster**, against the full
  data, **through Slurm on a compute node — never on a login node**.
- Cluster command pattern: `uv run meg-tokens --config tokens.toml <command>`
  from inside `~/meg-tokens`.

### Stage ordering for the sequential-sampling models

`ssm-aggregate` is the sole writer of every `ssm*` group derivative;
`behavior characterization` consumes the per-subject fits but writes none of
them back, so the run order between the two is free. A characterization-only
run produces no `ssm*` group table, so the model-comparison figures need
`ssm-fit` and `ssm-aggregate` to have run first.

Recovery is the exception to "copy the group derivatives down and re-run
locally". `ssm-recovery-aggregate` merges twelve repetition-scoped tables that
only the cluster array writes, and merging a *partial* or *stale* set silently
replaces the published `ssmparameterrecovery` / `ssmmodelrecovery` tables — and
therefore F15 — with results from a different simulation. The desktop
deliberately keeps only the merged tables, so the command fails naming the
missing repetition instead of producing a plausible wrong answer. Run the merge
where the repetitions are.

### Proving a refactor changed nothing

The behavioral stages re-run end to end on the real data in about two minutes,
so a change to them can be *measured* rather than argued:

```bash
python -m scripts.equivalence.harness prepare        # copy the fixed inputs once
python -m scripts.equivalence.harness run baseline   # before the change
python -m scripts.equivalence.harness run candidate  # after it
python -m scripts.equivalence.harness compare baseline candidate
```

It diffs every derivative table, every figure PNG (rendering is deterministic,
so they compare byte for byte) and every sidecar. It does not cover the
per-subject model fitting or the recovery merge, whose outputs it consumes;
those still have to be verified on the cluster.

### Verifying the write-up against the derivatives

`scripts/verify/` holds seven scripts that assert every number
`docs/behavior.md` quotes against the derivative it came from. They read
`data_root` from this checkout's `tokens.toml`, so they run unchanged on any
machine where the data is configured:

```bash
python -m scripts.verify.verify_doc          # Acts 1-2
python -m scripts.verify.verify_act4_doc     # sequential effects
python -m scripts.verify.verify_act5_doc     # individual differences
```

Run them after any derivative regeneration.

## Running the Pipeline (CLI)

`meg-tokens` is the supported execution API; every stage also has a callable workflow
function under `meg_tokens.workflows`.

### Stage 0: Raw BIDSification
Plans the raw layer per subject (MEG runs, empty-room, headshape, anat) into a reviewable
manifest. Reads `raw/`, `tdms/`, `subjects_dir`; no Stage 1 dependency; never touches `BIDS/`:
```bash
meg-tokens --config tokens.toml meg stage-raw --subjects H01 H02
```

Matching: **inter-trial-interval fingerprint** — logged `nInitialTime` gaps vs. real MEG
trial-start pulse gaps, the same physical intervals on two unrelated clocks. Accepted only
within an error threshold and a margin over the runner-up, with no session claimed twice;
otherwise `review`.

Resolve a flagged row: edit the manifest's `source_path`/`action`, re-run
`apply-raw-staging` (applies the manifest as saved, never recomputes). Applying copies staged
rows into `BIDS/sub-*/{meg,beh,anat}` and `BIDS/sub-emptyroom`; originals under `raw/`,
`tdms/`, `IRM/` are untouched:
```bash
meg-tokens --config tokens.toml meg apply-raw-staging --subjects H01 H02
```

### Stage 1: Behavioral Log Parsing
Builds trial tables from `BIDS/sub-*/beh/` (trial-class inference + validation). Requires
Stage 0.
```bash
meg-tokens --config tokens.toml behavior ingest
```

### Stage 2: Behavioral Metrics Extraction
Computes subject summaries, paired group statistics, and the trial-level
MEG-joinable feature table from Stage 1's staged TSV files:

```bash
meg-tokens --config tokens.toml behavior analyze
```

### Stage 2b: Behavioral Characterization Analyses
Runs the full battery of behavioral analyses described in `docs/behavior.md`
over Stage 2's trial-feature table — distributions, condition × class effects, session
drift, lapses, continuous evidence, criterion decline and urgency, reverse
correlation, conditional accuracy, trial history, and individual differences.
Each writes its own group derivative; the results on the current dataset, and
the figures built from them, are reported in `docs/behavior.md`. Requires
Stage 2 to have run first:

```bash
meg-tokens --config tokens.toml behavior characterization
```

**Behavior QC** (not a pipeline stage — a standalone validation utility):
re-run the source-log success-probability/SPD validation and print its summary:

```bash
meg-tokens --config tokens.toml behavior qc
```


### Stage 3: MEG Preprocessing
Builds one shared trunk per staged CTF recording (compensation, ZapLine,
0.5–150 Hz, bad channels and segments), then cleans it with DSS and with ICA
and writes one `proc-<stream>` derivative per cleaner. The specification is
`docs/meg.md`. Defaults to every staged subject and both
streams (`--streams` narrows):
```bash
meg-tokens --config tokens.toml meg preprocess
meg-tokens --config tokens.toml meg preprocess --subjects H01 H02
```
`--run` narrows to a single recording, for one subject:
```bash
meg-tokens --config tokens.toml meg preprocess --subjects H01 --run Slow1
```

### Stage 4: Epoch Extraction & Event Alignment
Slices one Stage 3 stream (`ica` until the canonical stream is decided) into event-locked trial epochs, aligned to Stage 1 behavior TSV derivatives:
```bash
meg-tokens --config tokens.toml meg epoch \
  --alignment go \
  --subjects H01 H02
```

### Stage 5: Behavioral Figures
Renders the behavioral figure battery from the group derivatives (`--list-figures` names them):
```bash
meg-tokens --config tokens.toml report behavior --figures all
```

### Stages 6 onward: source reconstruction, features, group analyses

Commands exist for every later stage -- `meg source`, `features
{erp,power,spectral,hilbert,pac,connectivity}`, `analyze
{statistics,lateralization,decoding,decomposition,spatial-decoding}`, the
matching `report` figures, and `validate golden` -- but they were ported from
the original notebooks and **none has run on this project's data**. Each is
rebuilt against [`docs/meg.md`](docs/meg.md) when
its stage is reached; until then treat them as unverified, and read their
options from `meg-tokens <domain> <command> --help`.

## Workflow Orchestration

Snakemake owns local and Slurm execution. Edit `workflow/config.yaml` to list
the real subjects, runs, raw-data template, and analysis settings; keep path
roots in the referenced `config/tokens.toml`.

```bash
pip install -e '.[workflow]'

# Inspect the complete DAG without running data.
snakemake --snakefile workflow/Snakefile \
  --configfile workflow/config.yaml \
  --profile workflow/profiles/local \
  --dry-run

# Run locally.
./cluster/run_workflow_local.sh workflow/config.yaml

# Submit jobs through the Snakemake Slurm executor.
./cluster/submit_workflow.sh workflow/config.yaml
```

Shared covariance, BEM, and source-space models are separate subject-level
rules. Run-level forward/inverse/source estimates depend on those models, so
parallel execution does not rebuild or overwrite shared files.
