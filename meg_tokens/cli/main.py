"""Unified MEG Tokens command-line interface."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional, Sequence

import mne

from meg_tokens import __version__
from meg_tokens.core import (
    EPOCH_LOCKS,
    STREAMS,
    ConnectivityConfig,
    DecodingConfig,
    CoregistrationConfig,
    EpochingConfig,
    HilbertConfig,
    LateralizedStatisticsConfig,
    PACConfig,
    PowerConfig,
    PreprocessingConfig,
    ProjectConfig,
    RunSpec,
    SpatialDecodingConfig,
    SpectralConfig,
    BandChoiceConfig,
    StatisticsConfig,
    TRAJECTORY_MEASURES,
    TRAJECTORY_SPACES,
    TrajectoryConfig,
    normalize_subject_id,
)

Handler = Callable[[ProjectConfig, argparse.Namespace], object]


def build_parser() -> argparse.ArgumentParser:
    """Build the side-effect-free project command parser."""
    parser = argparse.ArgumentParser(prog="meg-tokens")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", required=True, help="Project TOML configuration.")
    domains = parser.add_subparsers(dest="domain", required=True)

    behavior = domains.add_parser("behavior", help="Behavior ingestion and analysis.")
    behavior_commands = behavior.add_subparsers(dest="behavior_command", required=True)

    ingest = behavior_commands.add_parser(
        "ingest",
        help="Parse TDMS runs into behavior derivatives.",
    )
    ingest.add_argument("--subjects", nargs="+")
    ingest.add_argument("--dry-run", dest="dry_run", action="store_true")

    analyze = behavior_commands.add_parser(
        "analyze",
        help="Compute subject behavioral summaries from staged TSV files.",
    )
    analyze.add_argument("--subjects", nargs="+")

    characterization = behavior_commands.add_parser(
        "characterization",
        help=(
            "Stage 2b: run the behavioral characterization analyses over the "
            "staged trial-feature table."
        ),
    )
    characterization.add_argument("--subjects", nargs="+")
    characterization.add_argument(
        "--neural-metrics",
        dest="neural_metrics",
        help=(
            "Optional subject-level MEG metrics table (TSV or CSV with a "
            "'subject' column) to join into the individual-difference profile."
        ),
    )

    ssm_fit = behavior_commands.add_parser(
        "ssm-fit",
        help=(
            "Stage 2b, sequential-sampling fit only: fit the models and "
            "write one table per subject. Split out because it is the whole "
            "runtime of the stage; 'behavior characterization' pools the "
            "tables it writes."
        ),
    )
    ssm_fit.add_argument("--subjects", nargs="+")
    ssm_fit.add_argument(
        "--n-jobs",
        dest="n_jobs",
        type=int,
        default=1,
        help=(
            "Worker processes. Negative uses every CPU. Each subject is six "
            "independent fits, so pass a batch of subjects to fill a node "
            "with more cores than that."
        ),
    )

    ssm_evaluate = behavior_commands.add_parser(
        "ssm-evaluate",
        help="Evaluate complete Thura et al. mechanistic fits (held-out, recovery, matching, diagnostics).",
    )
    ssm_evaluate.add_argument("--subjects", nargs="+")
    ssm_evaluate.add_argument("--folds", type=int, default=3)
    ssm_evaluate.add_argument("--n-starts", dest="n_starts", type=int, default=1)
    ssm_evaluate.add_argument("--recovery-repetitions", dest="recovery_repetitions", type=int, default=0)
    ssm_evaluate_aggregate = behavior_commands.add_parser(
        "ssm-evaluate-aggregate",
        help="Validate and merge subject-array held-out/matching/distribution diagnostics.",
    )
    ssm_evaluate_aggregate.add_argument("--folds", type=int, default=3)
    behavior_commands.add_parser(
        "ssm-aggregate",
        help="Validate and pool complete subject-level mechanistic SSM outputs only.",
    )
    ssm_recovery = behavior_commands.add_parser(
        "ssm-recovery",
        help="Run parameter/model recovery simulations for the mechanistic model set.",
    )
    ssm_recovery.add_argument("--repetitions", type=int, default=12)
    # Three, to match the production fit's estimator; recovery is cited as
    # evidence about those fits.
    ssm_recovery.add_argument("--n-starts", dest="n_starts", type=int, default=3)
    ssm_recovery.add_argument(
        "--repetition-index", type=int,
        help="Run one truth-design repetition for a restartable Slurm array task.",
    )
    behavior_commands.add_parser(
        "ssm-recovery-aggregate",
        help="Validate and merge all repetition-scoped recovery derivatives.",
    ).add_argument("--repetitions", type=int, default=12)
    ssm_robustness = behavior_commands.add_parser(
        "ssm-robustness",
        help="Run the prespecified tau, solver, evidence-horizon, and bound robustness grid.",
    )
    ssm_robustness.add_argument("--n-jobs", dest="n_jobs", type=int, default=1)
    ssm_robustness.add_argument("--n-starts", dest="n_starts", type=int, default=1)
    ssm_robustness.add_argument("--subjects", nargs="+")
    ssm_robustness.add_argument("--configuration")
    behavior_commands.add_parser(
        "ssm-robustness-aggregate",
        help="Validate and merge subject-by-configuration robustness outputs.",
    )
    ssm_exclusion = behavior_commands.add_parser(
        "ssm-exclusion-robustness",
        help="Refit the mechanistic set under complete-token-log/alignment sensitivities.",
    )
    ssm_exclusion.add_argument("--n-jobs", dest="n_jobs", type=int, default=1)
    ssm_exclusion.add_argument("--n-starts", dest="n_starts", type=int, default=1)
    ssm_exclusion.add_argument("--subjects", nargs="+")
    ssm_exclusion.add_argument(
        "--rule", choices=("complete_token_log_alignment",)
    )
    behavior_commands.add_parser(
        "ssm-exclusion-robustness-aggregate",
        help="Validate and merge subject-by-rule exclusion-refit outputs.",
    )
    ssm_fit.add_argument(
        "--n-starts",
        dest="n_starts",
        type=int,
        default=1,
        help="Independent seeded optimizer starts per model/cell.",
    )

    behavior_commands.add_parser(
        "subjects",
        help=(
            "Print the Stage 2b subject list, one per line, in the order an "
            "array job should batch it."
        ),
    )

    behavior_commands.add_parser(
        "qc",
        help="Validate source TDMS success-probability and SPD profiles.",
    )

    meg = domains.add_parser("meg", help="MEG preprocessing and epoching.")
    meg_commands = meg.add_subparsers(dest="meg_command", required=True)

    stage_raw = meg_commands.add_parser(
        "stage-raw",
        help=(
            "Stage 0: plan the whole raw layer -- match CTF sessions to "
            "behavior runs, and locate each subject's empty-room, head shape "
            "and anatomical -- into a reviewable manifest. Plan only; never "
            "touches BIDS/."
        ),
    )
    stage_raw.add_argument("--subjects", nargs="+", required=True)

    apply_raw_staging_parser = meg_commands.add_parser(
        "apply-raw-staging",
        help=(
            "Stage 0: materialize BIDS/sub-*/{meg,beh,anat} and "
            "BIDS/sub-emptyroom from a Stage 0 manifest (fresh from "
            "`stage-raw`, or hand-reviewed). Only action=stage rows."
        ),
    )
    apply_raw_staging_parser.add_argument("--subjects", nargs="+")
    apply_raw_staging_parser.add_argument(
        "--manifest",
        help="Manifest TSV to apply. Defaults to the project's Stage 0 manifest path.",
    )

    preprocess = meg_commands.add_parser(
        "preprocess",
        help="Filter staged CTF recordings and optionally apply ICA.",
    )
    preprocess.add_argument(
        "--subjects",
        nargs="+",
        help="Subjects to preprocess. Defaults to every staged subject.",
    )
    preprocess.add_argument(
        "--run",
        help=(
            "Narrow to one recording. Requires exactly one subject, and a "
            "condition -- either labelled into the run ('Fast1') or via "
            "--condition."
        ),
    )
    preprocess.add_argument("--condition")
    preprocess.add_argument("--streams", dest="streams", nargs="+",
                            choices=STREAMS,
                            help="cleaners to run; default both")

    amplitudes = meg_commands.add_parser(
        "epoch-amplitudes",
        help="Per-epoch peak-to-peak on every magnetometer, for the cohort's outlier thresholds.",
    )
    amplitudes.add_argument("--subjects", nargs="+", required=True)
    amplitudes.add_argument("--locks", nargs="+", choices=EPOCH_LOCKS, default=list(EPOCH_LOCKS))
    amplitudes.add_argument("--streams", nargs="+", choices=STREAMS, default=list(STREAMS))

    thresholds = meg_commands.add_parser(
        "epoch-thresholds",
        help="The cohort's outlier threshold per stream, lock and magnetometer.",
    )
    thresholds.add_argument("--subjects", nargs="+", help="default: every staged subject")
    thresholds.add_argument("--locks", nargs="+", choices=EPOCH_LOCKS, default=list(EPOCH_LOCKS))
    thresholds.add_argument("--streams", nargs="+", choices=STREAMS, default=list(STREAMS))

    epoch = meg_commands.add_parser(
        "epoch",
        help="Build event-locked epochs from the cleaned recordings, dropping marked and outlier epochs.",
    )
    epoch.add_argument("--subjects", nargs="+", required=True)
    epoch.add_argument("--locks", nargs="+", choices=EPOCH_LOCKS, default=list(EPOCH_LOCKS))
    epoch.add_argument("--streams", nargs="+", choices=STREAMS, default=list(STREAMS))

    coreg = meg_commands.add_parser(
        "coreg",
        help="Fit MEG-MRI transforms for staged subjects, with QC.",
    )
    coreg.add_argument("--subjects", nargs="+")
    coreg.add_argument("--subjects-dir")
    coreg.add_argument("--output-root")
    coreg.add_argument("--template-subject", dest="template_subject", default="fsaverage")
    coreg.add_argument(
        "--no-scale-template",
        dest="scale_template",
        action="store_false",
        help="Use the template anatomy at its own size instead of fitting it.",
    )
    coreg.add_argument("--omit-distance", dest="omit_distance", type=float, default=0.005)

    source_models = meg_commands.add_parser(
        "source-models",
        help="Noise covariance, BEM and source space for selected subjects.",
    )
    source_models.add_argument("--subjects", nargs="+", required=True)

    source = meg_commands.add_parser(
        "source",
        help="Forward model per recording and inverse operator per stream.",
    )
    source.add_argument("--subjects", nargs="+", required=True)
    source.add_argument("--run", help="one recording; default every Fast and Slow run")
    source.add_argument("--condition")
    source.add_argument("--streams", nargs="+", choices=STREAMS, default=list(STREAMS))

    band_power = meg_commands.add_parser(
        "band-power",
        help="Single-trial band power in source space for one run, stream and band, every lock.",
    )
    band_power.add_argument("--subjects", nargs=1, required=True)
    band_power.add_argument("--run", required=True)
    band_power.add_argument("--condition", required=True)
    band_power.add_argument("--stream", choices=STREAMS, required=True)
    band_power.add_argument("--band", required=True, help="theta, alpha, beta, gamma or highgamma")
    band_power.add_argument("--locks", nargs="+", choices=EPOCH_LOCKS, help="default: go, decision, enter")
    band_power.add_argument("--n-jobs", type=int, default=1, help="cores for filtering")
    band_power.add_argument("--induced", action="store_true",
                            help="take every task event's response out first and write the second copy")
    band_power.add_argument("--output-root", help="write the region part under this root instead of the BIDS root")
    band_power.add_argument(
        "--fsaverage-root",
        help="also write every trial's power on fsaverage under this root (about 24 TB for the cohort); "
        "empty for none",
    )

    deep_power = meg_commands.add_parser(
        "deep-power",
        help="Single-trial band power at every deep source for one run, stream and band, every lock.",
    )
    deep_power.add_argument("--subjects", nargs=1, required=True)
    deep_power.add_argument("--run", required=True)
    deep_power.add_argument("--condition", required=True)
    deep_power.add_argument("--stream", choices=STREAMS, required=True)
    deep_power.add_argument("--band", required=True, help="theta, alpha, beta, gamma or highgamma")
    deep_power.add_argument("--locks", nargs="+", choices=EPOCH_LOCKS, help="default: go, decision, enter")
    deep_power.add_argument("--n-jobs", type=int, default=1, help="cores for filtering")
    deep_power.add_argument("--output-root", required=True, help="the BIDS-like root of the deep power (in /project)")

    sensor_spectra = meg_commands.add_parser(
        "sensor-spectra",
        help="A subject's trial-averaged magnetometer spectra around one lock, every Fast and Slow run (Welch).",
    )
    sensor_spectra.add_argument("--subjects", nargs="+", required=True)
    sensor_spectra.add_argument("--streams", nargs="+", choices=STREAMS, default=list(STREAMS))
    sensor_spectra.add_argument("--lock", choices=EPOCH_LOCKS, default="enter")
    sensor_spectra.add_argument("--output-root")

    band_power_means_fsaverage = meg_commands.add_parser(
        "band-power-means-fsaverage",
        help="A subject's mean fsaverage band power per condition and decision-time quintile, one band, every lock.",
    )
    band_power_means_fsaverage.add_argument("--subjects", nargs="+", required=True)
    band_power_means_fsaverage.add_argument("--streams", nargs="+", choices=STREAMS, default=list(STREAMS))
    band_power_means_fsaverage.add_argument("--bands", nargs="+", help="default: every band")
    band_power_means_fsaverage.add_argument("--locks", nargs="+", choices=EPOCH_LOCKS, help="default: go, decision, enter")
    band_power_means_fsaverage.add_argument("--fsaverage-root", required=True, help="where the fsaverage band power is")
    band_power_means_fsaverage.add_argument("--output-root", help="write the means under this root instead of the fsaverage root")
    band_power_means_fsaverage.add_argument("--induced", action="store_true", help="the second copy, event responses taken out")

    deep_morph = meg_commands.add_parser(
        "deep-morph",
        help="Each subject's morph of its deep sources onto fsaverage's grid of the same structures, once.",
    )
    deep_morph.add_argument("--subjects", nargs="+", required=True)

    deep_power_means = meg_commands.add_parser(
        "deep-power-means",
        help="A subject's mean deep-source band power per condition and quintile on fsaverage's grid, one band.",
    )
    deep_power_means.add_argument("--subjects", nargs="+", required=True)
    deep_power_means.add_argument("--streams", nargs="+", choices=STREAMS, default=list(STREAMS))
    deep_power_means.add_argument("--bands", nargs="+", help="default: every band")
    deep_power_means.add_argument("--locks", nargs="+", choices=EPOCH_LOCKS, help="default: go, decision, enter")
    deep_power_means.add_argument("--deep-root", required=True, help="where the deep power is")
    deep_power_means.add_argument("--output-root", help="write the means under this root instead of the deep root")

    band_power_prepare = meg_commands.add_parser(
        "band-power-prepare",
        help="Build each subject's morph map with fsaverage, once, before the band-power jobs.",
    )
    band_power_prepare.add_argument("--subjects", nargs="+", help="default: every staged subject")

    band_power_tasks = meg_commands.add_parser(
        "band-power-tasks",
        help="Print one line per band-power job (subject run condition stream band), for Slurm arrays.",
    )
    band_power_tasks.add_argument("--subjects", nargs="+", help="default: every staged subject")
    band_power_tasks.add_argument("--streams", nargs="+", choices=STREAMS, default=list(STREAMS))

    erp = meg_commands.add_parser(
        "erp",
        help="Event-related fields per condition, in sensors and in source regions, per subject and stream.",
    )
    erp.add_argument("--subjects", nargs="+", required=True)
    erp.add_argument("--streams", nargs="+", choices=STREAMS, default=list(STREAMS))
    erp.add_argument("--locks", nargs="+", choices=EPOCH_LOCKS, help="default: go, decision, enter")

    band_power_means = meg_commands.add_parser(
        "band-power-means",
        help="A subject's mean band power per condition and per decision-time quintile.",
    )
    band_power_means.add_argument("--subjects", nargs="+", required=True)
    band_power_means.add_argument("--streams", nargs="+", choices=STREAMS, default=list(STREAMS))
    band_power_means.add_argument("--locks", nargs="+", choices=EPOCH_LOCKS, help="default: go, decision, enter")
    band_power_means.add_argument("--induced", action="store_true", help="the second copy, event responses taken out")

    band_power_merge = meg_commands.add_parser(
        "band-power-merge",
        help="Assemble a subject's band power per stream and lock from its run and band parts.",
    )
    band_power_merge.add_argument("--subjects", nargs="+", required=True)
    band_power_merge.add_argument("--streams", nargs="+", choices=STREAMS, default=list(STREAMS))
    band_power_merge.add_argument("--locks", nargs="+", choices=EPOCH_LOCKS, help="default: go, decision, enter")
    band_power_merge.add_argument("--induced", action="store_true", help="the second copy, event responses taken out")

    features = domains.add_parser("features", help="Neural feature extraction.")
    feature_commands = features.add_subparsers(dest="feature_command", required=True)

    power = feature_commands.add_parser("power", help="Extract source-space band power.")
    power.add_argument("--subjects", nargs="+", required=True)
    power.add_argument("--run", required=True)
    power.add_argument("--condition")
    power.add_argument("--alignment", choices=EPOCH_LOCKS, default="go")
    power.add_argument("--source-method", default="dSPM")
    power.add_argument("--method", choices=["hilbert", "morlet", "multitaper"], default="hilbert")
    power.add_argument("--bands", nargs="+")
    power.add_argument("--width", type=int, default=400)
    power.add_argument("--step", type=int, default=110)
    power.add_argument("--baseline", type=float, nargs=2)
    power.add_argument("--baseline-method", choices=["percent", "ratio", "logratio", "zscore", "difference"], default="percent")
    power.add_argument("--n-jobs", type=int, default=1)
    power.add_argument("--source-root")

    hilbert = feature_commands.add_parser("hilbert", help="Extract analytic-signal features.")
    hilbert.add_argument("--subjects", nargs="+", required=True)
    hilbert.add_argument("--conditions", nargs="+", required=True)
    hilbert.add_argument("--alignment", choices=EPOCH_LOCKS, default="go")
    hilbert.add_argument("--source-method", default="dSPM")
    hilbert.add_argument("--parc", default="HCPMMP1")
    hilbert.add_argument("--labels", nargs="+")
    hilbert.add_argument("--bands", nargs="+")
    hilbert.add_argument("--feature-types", nargs="+", default=["amplitude", "power", "phase", "sigfilt"])
    hilbert.add_argument("--sfreq", type=float)
    hilbert.add_argument("--n-jobs", type=int, default=1)
    hilbert.add_argument("--feature-root")

    pac = feature_commands.add_parser("pac", help="Compute phase-amplitude coupling.")
    pac.add_argument("--subjects", nargs="+", required=True)
    pac.add_argument("--conditions", nargs="+", required=True)
    pac.add_argument("--phase-bands", nargs="+", required=True)
    pac.add_argument("--amplitude-bands", nargs="+", required=True)
    pac.add_argument("--alignment", choices=EPOCH_LOCKS, default="go")
    pac.add_argument("--source-method", default="dSPM")
    pac.add_argument("--parc", default="HCPMMP1")
    pac.add_argument("--n-bins", type=int, default=18)
    pac.add_argument("--time-window", type=float, nargs=2)
    pac.add_argument("--feature-root")

    connectivity = feature_commands.add_parser("connectivity", help="Compute spectral connectivity.")
    connectivity.add_argument("--subjects", nargs="+", required=True)
    connectivity.add_argument("--conditions", nargs="+", required=True)
    connectivity.add_argument("--alignment", choices=EPOCH_LOCKS, default="enter")
    connectivity.add_argument("--source-method", default="dSPM")
    connectivity.add_argument("--parc", default="HCPMMP1")
    connectivity.add_argument("--labels", nargs="+")
    connectivity.add_argument("--bands", nargs="+")
    connectivity.add_argument("--method", default="imcoh")
    connectivity.add_argument("--mode", default="fourier")
    connectivity.add_argument("--sfreq", type=float)
    connectivity.add_argument("--before-window", type=float, nargs=2, default=(0.7, 1.4))
    connectivity.add_argument("--after-window", type=float, nargs=2, default=(1.6, 2.3))
    connectivity.add_argument("--n-jobs", type=int, default=1)
    connectivity.add_argument("--feature-root")

    analyze = domains.add_parser("analyze", help="Group statistics and multivariate analyses.")
    analysis_commands = analyze.add_subparsers(dest="analysis_command", required=True)

    statistics = analysis_commands.add_parser("statistics", help="Run a paired group contrast.")
    statistics.add_argument("--conditions", nargs=2, default=["Fast", "Slow"])
    statistics.add_argument("--subjects", nargs="+")
    statistics.add_argument("--alignment", choices=EPOCH_LOCKS, default="go")
    statistics.add_argument("--source-method", default="dSPM")
    statistics.add_argument("--parc", default="HCPMMP1")
    statistics.add_argument("--runs-condition-1", nargs="+")
    statistics.add_argument("--runs-condition-2", nargs="+")
    statistics.add_argument("--permutations", type=int, default=1000)
    statistics.add_argument("--alpha", type=float, default=0.05)
    statistics.add_argument("--tail", type=int, choices=[-1, 0, 1], default=0)
    statistics.add_argument("--n-jobs", type=int, default=1)
    statistics.add_argument("--feature-root")

    lateralization = analysis_commands.add_parser(
        "lateralization",
        help="Test homologous left-minus-right ERP labels.",
    )
    lateralization.add_argument("--condition", required=True)
    lateralization.add_argument("--subjects", nargs="+")
    lateralization.add_argument("--alignment", choices=EPOCH_LOCKS, default="go")
    lateralization.add_argument("--source-method", default="dSPM")
    lateralization.add_argument("--parc", default="HCPMMP1")
    lateralization.add_argument("--runs", nargs="+")
    lateralization.add_argument("--permutations", type=int, default=1000)
    lateralization.add_argument("--alpha", type=float, default=0.05)
    lateralization.add_argument("--tail", type=int, choices=[-1, 0, 1], default=0)
    lateralization.add_argument("--n-jobs", type=int, default=1)
    lateralization.add_argument("--feature-root")

    decoding = analysis_commands.add_parser("decoding", help="Run time-resolved decoding.")
    decoding.add_argument("--conditions", nargs="+", default=["Fast", "Slow"])
    decoding.add_argument("--input-conditions", nargs="+")
    decoding.add_argument("--subjects", nargs="+")
    decoding.add_argument("--feature-source", choices=["erp", "power"], default="erp")
    decoding.add_argument("--alignment", choices=EPOCH_LOCKS, default="go")
    decoding.add_argument("--source-method", default="dSPM")
    decoding.add_argument("--parc", default="HCPMMP1")
    decoding.add_argument("--band")
    decoding.add_argument("--power-method", default="hilbert")
    decoding.add_argument("--runs", nargs="+")
    decoding.add_argument("--labels", "--roi", dest="labels", nargs="+")
    decoding.add_argument("--lateralize", action="store_true")
    decoding.add_argument("--class-column")
    decoding.add_argument("--class-values", nargs="+")
    decoding.add_argument("--permutations", type=int, default=0)
    decoding.add_argument("--n-jobs", type=int, default=4)
    decoding.add_argument("--feature-root")

    trajectories = analysis_commands.add_parser(
        "trajectories",
        help="PCA trajectories of the cohort's condition means, per stream and lock (meg.md §2).",
    )
    trajectories.add_argument("--fit", choices=["grandaverage", "pooled"], required=True)
    trajectories.add_argument("--measure", choices=TRAJECTORY_MEASURES, required=True)
    trajectories.add_argument("--space", choices=TRAJECTORY_SPACES, default="HCPMMP1")
    trajectories.add_argument("--n-components", type=int, default=20)
    trajectories.add_argument("--streams", nargs="+", choices=STREAMS, default=list(STREAMS))
    trajectories.add_argument("--locks", nargs="+", choices=EPOCH_LOCKS, help="default: go, decision, enter")
    trajectories.add_argument("--subjects", nargs="+", help="default: every staged subject")
    trajectories.add_argument("--window", type=float, nargs=2, metavar=("TMIN", "TMAX"),
                              help="seconds from the lock; default: the lock's whole analysis window")
    trajectories.add_argument("--fsaverage-root", help="where the fsaverage band power means are (power on fsaverage)")
    trajectories.add_argument("--structure", help="the deep structure of an fsaverageaseg fit (Caudate: both sides)")
    trajectories.add_argument("--induced", action="store_true",
                              help="the second copy of band power, task events' responses taken out")
    trajectories.add_argument("--with-deep", action="store_true",
                              help="fit the 17 deep structures beside the 360 regions (HCP-MMP1 band power)")

    band_choice = analysis_commands.add_parser(
        "band-choice",
        help="The preprint's band choice from spectral peaks of the cohort's sensor spectra (its Fig S1), and its checks.",
    )
    band_choice.add_argument("--streams", nargs="+", choices=STREAMS, default=list(STREAMS))
    band_choice.add_argument("--lock", choices=EPOCH_LOCKS, default="enter")
    band_choice.add_argument("--aperiodic", choices=["fixed", "knee"], default="fixed",
                             help="the aperiodic component: fixed (the preprint's) or with a knee")
    band_choice.add_argument("--subjects", nargs="+", help="default: every staged subject")
    band_choice.add_argument("--output-root")

    spatial_decoding = analysis_commands.add_parser(
        "spatial-decoding",
        help="Run sensor-space decoding on staged trial arrays.",
    )
    spatial_decoding.add_argument("--data-dir", required=True)
    spatial_decoding.add_argument("--conditions", nargs="+", default=["Fast", "Slow"])
    spatial_decoding.add_argument("--permutations", type=int, default=0)
    spatial_decoding.add_argument("--n-jobs", type=int, default=4)
    spatial_decoding.add_argument("--output-root")

    report = domains.add_parser("report", help="Generate figures and summary tables.")
    report_commands = report.add_subparsers(dest="report_command", required=True)

    behavior_report = report_commands.add_parser(
        "behavior", help="Render the behavioral figure battery."
    )
    behavior_report.add_argument("--subjects", nargs="+")
    behavior_report.add_argument("--behavior-root")
    behavior_report.add_argument("--output-root")
    behavior_report.add_argument(
        "--figures",
        nargs="+",
        default=["all"],
        help=(
            "Figure keys and/or group names to render (default: all). "
            "Groups: headline, core, distributions, design, evidence, sequential, "
            "modeling, individual, qc. Use --list-figures to see every key."
        ),
    )
    behavior_report.add_argument(
        "--formats",
        nargs="+",
        default=[".pdf", ".png"],
        choices=[".pdf", ".png", ".svg"],
    )
    behavior_report.add_argument(
        "--list-figures",
        dest="list_figures",
        action="store_true",
        help="Print the figure registry (key, source derivatives, title) and exit.",
    )
    behavior_report.add_argument(
        "--skip-missing",
        dest="skip_missing",
        action="store_true",
        help="Skip figures whose source derivative is absent instead of raising.",
    )

    band_choice_report = report_commands.add_parser(
        "band-choice", help="specparam's group report and the band-choice figure, per stream."
    )
    band_choice_report.add_argument("--streams", nargs="+", choices=STREAMS, default=list(STREAMS))
    band_choice_report.add_argument("--lock", choices=EPOCH_LOCKS, default="enter")
    band_choice_report.add_argument("--aperiodic", choices=["fixed", "knee"], default="fixed")
    band_choice_report.add_argument("--output-root")

    coreg_report = report_commands.add_parser(
        "coreg", help="Rebuild MEG-MRI coregistration QC figures."
    )
    coreg_report.add_argument("--subjects", nargs="+")
    coreg_report.add_argument("--output-root")
    coreg_report.add_argument("--subjects-dir")

    statistics_report = report_commands.add_parser("statistics", help="Plot group statistics.")
    statistics_report.add_argument("--conditions", nargs=2, default=["Fast", "Slow"])
    statistics_report.add_argument("--alignment", default="go", choices=EPOCH_LOCKS)
    statistics_report.add_argument("--source-method", default="dSPM")
    statistics_report.add_argument("--parc", default="HCPMMP1")
    statistics_report.add_argument("--labels", nargs="+")
    statistics_report.add_argument("--top-n", type=int, default=8)
    statistics_report.add_argument("--alpha", type=float, default=0.05)
    statistics_report.add_argument("--peak-mode", default="min", choices=["min", "max", "abs"])
    statistics_report.add_argument("--correlate-behavior", action="store_true")
    statistics_report.add_argument("--stats-root")
    statistics_report.add_argument("--behavior-root")
    statistics_report.add_argument("--output-root")

    connectivity_report = report_commands.add_parser("connectivity", help="Plot a connectivity circle.")
    connectivity_report.add_argument("--condition", required=True)
    connectivity_report.add_argument("--band", default="alpha")
    connectivity_report.add_argument("--p-threshold", type=float, default=0.05)
    connectivity_report.add_argument("--permutations", type=int, default=1000)
    connectivity_report.add_argument("--node-names")
    connectivity_report.add_argument("--feature-root")
    connectivity_report.add_argument("--output-root")

    seed_report = report_commands.add_parser("seed-connectivity", help="Create a seed-connectivity report.")
    seed_report.add_argument("--condition", required=True)
    seed_report.add_argument("--seed-roi", required=True)
    seed_report.add_argument("--band", default="alpha")
    seed_report.add_argument("--p-threshold", type=float, default=0.05)
    seed_report.add_argument("--permutations", type=int, default=1000)
    seed_report.add_argument("--node-names")
    seed_report.add_argument("--feature-root")
    seed_report.add_argument("--output-root")

    onset_report = report_commands.add_parser("decoding-onset", help="Compare decoding onset times.")
    onset_report.add_argument("--scores", nargs="+", required=True)
    onset_report.add_argument("--thresholds", nargs="+", required=True)
    onset_report.add_argument("--names", nargs="+", required=True)
    onset_report.add_argument("--time-offset", type=float, default=-1000)
    onset_report.add_argument("--time-step", type=float, default=50)
    onset_report.add_argument("--output-root")

    spatial_report = report_commands.add_parser("spatial-decoding", help="Plot sensor decoding scores.")
    spatial_report.add_argument("--score-path", required=True)
    spatial_report.add_argument("--info-fif", required=True)
    spatial_report.add_argument("--output-root")

    validate = domains.add_parser("validate", help="Validate staged derivatives.")
    validation_commands = validate.add_subparsers(dest="validation_command", required=True)
    golden = validation_commands.add_parser("golden", help="Compare derivatives with real references.")
    golden.add_argument("--comparison-config", required=True)
    golden.add_argument("--out-tsv", required=True)
    golden.add_argument("--allow-failures", action="store_true")
    return parser


def _project_from_args(args: argparse.Namespace) -> ProjectConfig:
    project = ProjectConfig.from_toml(args.config)

    if getattr(args, "subjects_dir", None):
        return replace(project, subjects_dir=Path(args.subjects_dir))
    return project


# argparse names the second-level dest per domain rather than uniformly,
# so the lookup needs the mapping stated once.
_COMMAND_DEST: dict[str, str] = {
    "behavior": "behavior_command",
    "meg": "meg_command",
    "features": "feature_command",
    "analyze": "analysis_command",
    "report": "report_command",
    "validate": "validation_command",
}


def _behavior_ingest(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.workflows.behavior_ingest import ingest_behavior

    return ingest_behavior(
        project,
        subjects=args.subjects,
        dry_run=args.dry_run,
    )


def _behavior_analyze(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.workflows.behavior_analysis import analyze_behavior

    return analyze_behavior(project, subjects=args.subjects)


def _behavior_characterization(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.workflows.behavior_characterization import (
        analyze_behavior_characterization,
    )

    return analyze_behavior_characterization(
        project,
        subjects=args.subjects,
        neural_metrics_path=args.neural_metrics,
    )


def _behavior_ssm_fit(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.workflows.sequential_sampling import (
        fit_subject_sequential_sampling,
    )
    from meg_tokens.behavior.analyses.sequential_sampling import (
        MECHANISTIC_MODELS,
    )

    return fit_subject_sequential_sampling(
        project,
        subjects=args.subjects,
        n_jobs=args.n_jobs,
        models=MECHANISTIC_MODELS,
        n_starts=args.n_starts,
    )


def _behavior_ssm_evaluate(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.workflows.sequential_sampling import evaluate_thura2012

    return evaluate_thura2012(
        project,
        subjects=args.subjects,
        folds=args.folds,
        n_starts=args.n_starts,
        recovery_repetitions=args.recovery_repetitions,
    )


def _behavior_ssm_evaluate_aggregate(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.workflows.sequential_sampling import aggregate_mechanistic_evaluation

    return aggregate_mechanistic_evaluation(project, folds=args.folds)


def _behavior_ssm_aggregate(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.workflows.sequential_sampling import aggregate_mechanistic_fits

    return aggregate_mechanistic_fits(project)


def _behavior_ssm_recovery(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.workflows.sequential_sampling import run_recovery_stage

    return run_recovery_stage(
        project,
        repetitions=args.repetitions,
        n_starts=args.n_starts,
        repetition_index=args.repetition_index,
    )


def _behavior_ssm_recovery_aggregate(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.workflows.sequential_sampling import aggregate_recovery_stage

    return aggregate_recovery_stage(project, repetitions=args.repetitions)


def _behavior_ssm_robustness(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.workflows.sequential_sampling import run_robustness_stage

    return run_robustness_stage(
        project,
        subject=args.subjects[0] if args.subjects else None,
        configuration=args.configuration,
        n_jobs=args.n_jobs,
        n_starts=args.n_starts,
    )


def _behavior_ssm_robustness_aggregate(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.workflows.sequential_sampling import aggregate_robustness_stage

    return aggregate_robustness_stage(project)


def _behavior_ssm_exclusion_robustness(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.workflows.sequential_sampling import run_exclusion_robustness_stage

    return run_exclusion_robustness_stage(
        project,
        subject=args.subjects[0] if args.subjects else None,
        rule=args.rule,
        n_jobs=args.n_jobs,
        n_starts=args.n_starts,
    )


def _behavior_ssm_exclusion_robustness_aggregate(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.workflows.sequential_sampling import aggregate_exclusion_robustness_stage

    return aggregate_exclusion_robustness_stage(project)


def _behavior_subjects(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.workflows.sequential_sampling import _selected_features

    _, _, subjects, _ = _selected_features(project, None)
    for subject in subjects:
        print(subject)
    return None


def _behavior_qc(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.validation import run_spd_validation

    summary, details = run_spd_validation(
        project.behavior_root,
        ignore_files=project.behavior_ignore_files,
    )
    print(summary.to_string(index=False))
    print(f"Validated trials: {len(details)}")
    return None


def _meg_stage_raw(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.workflows.raw_staging import plan_raw_staging

    result = plan_raw_staging(project, subjects=args.subjects)
    for subject, counts in result.settings["summary_by_subject"].items():
        counts_text = ", ".join(f"{action}={count}" for action, count in sorted(counts.items()))
        print(f"{subject}: {counts_text}")
    review_rows = result.settings["review_rows"]
    if review_rows:
        print(f"\n{len(review_rows)} row(s) need review before applying:")
        for row in review_rows:
            reason = row["note"] or (
                f"{row['count_agreement']} (MEG start pulses={row['meg_start_pulse_count']}, "
                f"behavior trials={row['behavior_trial_count']})"
            )
            print(f"  {row['subject']} {row['kind']} {row['condition']} {row['run']}: {reason}")
    print(f"\nManifest: {result.outputs[0]}")
    print("Review flagged rows, then run `meg-tokens meg apply-raw-staging`.")
    return None


def _meg_apply_raw_staging(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.workflows.raw_staging import apply_raw_staging

    return apply_raw_staging(
        project,
        manifest_path=args.manifest,
        subjects=args.subjects,
    )


def _meg_preprocess(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.workflows.preprocessing import preprocess_run, preprocess_subjects

    settings = PreprocessingConfig(
        streams=tuple(args.streams) if args.streams else None,
    )
    if args.run is None:
        return preprocess_subjects(project, subjects=args.subjects, settings=settings)
    if not args.subjects or len(args.subjects) != 1:
        raise SystemExit(
            "--run narrows to a single recording, so it needs exactly one --subjects."
        )
    return preprocess_run(
        project,
        RunSpec(subject=args.subjects[0], run=args.run, condition=args.condition),
        settings=settings,
    )


def _meg_epoch_amplitudes(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.workflows.preprocessing import epoch_amplitude_subjects

    return epoch_amplitude_subjects(
        project,
        subjects=args.subjects,
        settings=EpochingConfig(locks=tuple(args.locks), streams=tuple(args.streams)),
    )


def _meg_epoch_thresholds(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.workflows.preprocessing import epoch_thresholds

    return epoch_thresholds(
        project,
        subjects=args.subjects,
        settings=EpochingConfig(locks=tuple(args.locks), streams=tuple(args.streams)),
    )


def _meg_epoch(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.workflows.preprocessing import epoch_subjects

    return epoch_subjects(
        project,
        subjects=args.subjects,
        settings=EpochingConfig(locks=tuple(args.locks), streams=tuple(args.streams)),
    )


def _meg_coreg(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.workflows.coregistration import coregister_subjects

    return coregister_subjects(
        project,
        subjects=args.subjects,
        settings=CoregistrationConfig(
            omit_distance=args.omit_distance,
            template_subject=args.template_subject,
            scale_template=args.scale_template,
        ),
        output_root=args.output_root,
    )


def _meg_source_models(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.workflows.sources import source_models

    return source_models(project, subjects=args.subjects)


def _meg_source(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.workflows.sources import source_operator_subjects, source_operators

    if args.run is None:
        return source_operator_subjects(project, subjects=args.subjects, streams=args.streams)
    if len(args.subjects) != 1:
        raise SystemExit("--run narrows to a single recording, so it needs exactly one --subjects.")
    return source_operators(
        project, RunSpec(subject=args.subjects[0], run=args.run, condition=args.condition), streams=args.streams,
    )


def _meg_band_power(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.meg.band_power import BANDS, LOCKS
    from meg_tokens.workflows.band_power import band_power_part

    if args.band not in BANDS:
        raise SystemExit(f"--band must be one of {', '.join(BANDS)}")
    return band_power_part(
        project, RunSpec(subject=args.subjects[0], run=args.run, condition=args.condition),
        stream=args.stream, band=args.band, locks=args.locks or LOCKS,
        fsaverage_root=args.fsaverage_root or None, output_root=args.output_root, n_jobs=args.n_jobs,
        induced=args.induced,
    )


def _meg_deep_power(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.meg.band_power import BANDS, LOCKS
    from meg_tokens.workflows.band_power import deep_power_part

    if args.band not in BANDS:
        raise SystemExit(f"--band must be one of {', '.join(BANDS)}")
    return deep_power_part(
        project, RunSpec(subject=args.subjects[0], run=args.run, condition=args.condition),
        stream=args.stream, band=args.band, locks=args.locks or LOCKS, output_root=args.output_root,
        n_jobs=args.n_jobs,
    )


def _meg_sensor_spectra(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.workflows.spectral import sensor_spectra

    settings = SpectralConfig(lock=args.lock)
    return [
        sensor_spectra(project, subject, stream=stream, settings=settings, output_root=args.output_root)
        for subject in args.subjects
        for stream in args.streams
    ]


def _meg_band_power_means_fsaverage(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.meg.band_power import BANDS, LOCKS
    from meg_tokens.workflows.band_power import band_power_means_fsaverage

    unknown = sorted(set(args.bands or ()) - set(BANDS))
    if unknown:
        raise SystemExit(f"--bands must be among {', '.join(BANDS)}, got {unknown}")
    return [
        band_power_means_fsaverage(project, subject, stream=stream, band=band, locks=args.locks or LOCKS,
                                   fsaverage_root=args.fsaverage_root, output_root=args.output_root,
                                   induced=args.induced)
        for subject in args.subjects
        for stream in args.streams
        for band in args.bands or BANDS
    ]


def _meg_deep_morph(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.workflows.band_power import deep_morph_subject

    return [deep_morph_subject(project, subject) for subject in args.subjects]


def _meg_deep_power_means(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.meg.band_power import BANDS, LOCKS
    from meg_tokens.workflows.band_power import deep_power_means

    unknown = sorted(set(args.bands or ()) - set(BANDS))
    if unknown:
        raise SystemExit(f"--bands must be among {', '.join(BANDS)}, got {unknown}")
    return [
        deep_power_means(project, subject, stream=stream, band=band, locks=args.locks or LOCKS,
                         deep_root=args.deep_root, output_root=args.output_root)
        for subject in args.subjects
        for stream in args.streams
        for band in args.bands or BANDS
    ]


def _meg_band_power_prepare(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.meg.band_power import prepare_morph_map
    from meg_tokens.meg.meg_bids import staged_subjects

    for subject in args.subjects or staged_subjects(project.bids_root):
        print(prepare_morph_map(subject, project.subjects_dir))
    return None


def _meg_band_power_tasks(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.meg.band_power import BANDS
    from meg_tokens.meg.epoching import EPOCH_CONDITIONS
    from meg_tokens.meg.meg_bids import raw_meg_sessions, staged_subjects

    for subject in args.subjects or staged_subjects(project.bids_root):
        for run, _ in raw_meg_sessions(project.bids_root, subject, task=project.task):
            if run.condition not in EPOCH_CONDITIONS:
                continue
            for stream in args.streams:
                for band in BANDS:
                    print(subject, run.run, run.condition.lower(), stream, band)
    return None


def _meg_erp(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.meg.band_power import LOCKS
    from meg_tokens.workflows.evoked import evoked_subject

    return [
        evoked_subject(project, subject, stream=stream, locks=args.locks or LOCKS)
        for subject in args.subjects
        for stream in args.streams
    ]


def _meg_band_power_means(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.meg.band_power import LOCKS
    from meg_tokens.workflows.band_power import band_power_means

    return [
        band_power_means(project, subject, stream=stream, locks=args.locks or LOCKS, induced=args.induced)
        for subject in args.subjects
        for stream in args.streams
    ]


def _meg_band_power_merge(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.meg.band_power import LOCKS
    from meg_tokens.workflows.band_power import band_power_merge

    return [
        band_power_merge(project, subject, stream=stream, locks=args.locks or LOCKS, induced=args.induced)
        for subject in args.subjects
        for stream in args.streams
    ]


def _features_power(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.workflows.power import extract_power_features, parse_frequency_bands

    parsed_bands = parse_frequency_bands(args.bands) if args.bands else None
    return extract_power_features(
        project,
        subjects=args.subjects,
        settings=PowerConfig(
            run=args.run,
            condition=args.condition,
            alignment=args.alignment,
            source_method=args.source_method,
            method=args.method,
            bands=(
                tuple((name, bounds[0], bounds[1]) for name, bounds in parsed_bands.items())
                if parsed_bands
                else None
            ),
            width=args.width,
            step=args.step,
            baseline=tuple(args.baseline) if args.baseline else None,
            baseline_method=args.baseline_method,
            n_jobs=args.n_jobs,
        ),
        source_root=args.source_root,
    )


def _features_hilbert(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.features.time_frequency import parse_frequency_bands
    from meg_tokens.workflows.hilbert import extract_hilbert_features

    parsed_bands = parse_frequency_bands(args.bands) if args.bands else None
    return extract_hilbert_features(
        project,
        subjects=args.subjects,
        settings=HilbertConfig(
            conditions=tuple(args.conditions),
            alignment=args.alignment,
            source_method=args.source_method,
            parc=args.parc,
            labels=tuple(args.labels) if args.labels else None,
            bands=(
                tuple((name, bounds[0], bounds[1]) for name, bounds in parsed_bands.items())
                if parsed_bands
                else None
            ),
            features=tuple(args.feature_types),
            sfreq=args.sfreq,
            n_jobs=args.n_jobs,
        ),
        feature_root=args.feature_root,
    )


def _features_pac(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.workflows.pac import extract_pac_features

    return extract_pac_features(
        project,
        subjects=args.subjects,
        settings=PACConfig(
            conditions=tuple(args.conditions),
            phase_bands=tuple(args.phase_bands),
            amplitude_bands=tuple(args.amplitude_bands),
            alignment=args.alignment,
            source_method=args.source_method,
            parc=args.parc,
            n_bins=args.n_bins,
            time_window=tuple(args.time_window) if args.time_window else None,
        ),
        feature_root=args.feature_root,
    )


def _features_connectivity(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.workflows.connectivity import _parse_bands, extract_connectivity_features

    names, lower, upper = _parse_bands(args.bands)
    return extract_connectivity_features(
        project,
        subjects=args.subjects,
        settings=ConnectivityConfig(
            conditions=tuple(args.conditions),
            alignment=args.alignment,
            source_method=args.source_method,
            parc=args.parc,
            labels=tuple(args.labels) if args.labels else None,
            bands=tuple(zip(names, lower, upper)),
            method=args.method,
            mode=args.mode,
            sfreq=args.sfreq,
            before_window=tuple(args.before_window),
            after_window=tuple(args.after_window),
            n_jobs=args.n_jobs,
        ),
        feature_root=args.feature_root,
    )


def _analyze_statistics(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.workflows.statistics import run_group_statistics

    return run_group_statistics(
        project,
        subjects=args.subjects,
        settings=StatisticsConfig(
            conditions=tuple(args.conditions),
            alignment=args.alignment,
            source_method=args.source_method,
            parc=args.parc,
            runs_condition_1=tuple(args.runs_condition_1) if args.runs_condition_1 else None,
            runs_condition_2=tuple(args.runs_condition_2) if args.runs_condition_2 else None,
            permutations=args.permutations,
            alpha=args.alpha,
            tail=args.tail,
            n_jobs=args.n_jobs,
        ),
        feature_root=args.feature_root,
    )


def _analyze_lateralization(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.workflows.statistics import run_lateralized_statistics

    return run_lateralized_statistics(
        project,
        subjects=args.subjects,
        settings=LateralizedStatisticsConfig(
            condition=args.condition,
            alignment=args.alignment,
            source_method=args.source_method,
            parc=args.parc,
            runs=tuple(args.runs) if args.runs else None,
            permutations=args.permutations,
            alpha=args.alpha,
            tail=args.tail,
            n_jobs=args.n_jobs,
        ),
        feature_root=args.feature_root,
    )


def _analyze_decoding(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.workflows.decoding import run_decoding

    return run_decoding(
        project,
        subjects=args.subjects,
        settings=DecodingConfig(
            conditions=tuple(args.conditions),
            input_conditions=tuple(args.input_conditions) if args.input_conditions else None,
            feature_source=args.feature_source,
            alignment=args.alignment,
            source_method=args.source_method,
            parc=args.parc,
            band=args.band,
            power_method=args.power_method,
            runs=tuple(args.runs) if args.runs else None,
            labels=tuple(args.labels) if args.labels else None,
            lateralize=args.lateralize,
            class_column=args.class_column,
            class_values=tuple(args.class_values) if args.class_values else None,
            permutations=args.permutations,
            n_jobs=args.n_jobs,
        ),
        feature_root=args.feature_root,
    )


def _analyze_trajectories(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.meg.band_power import LOCKS
    from meg_tokens.workflows.trajectories import trajectories

    settings = TrajectoryConfig(
        fit=args.fit, measure=args.measure, space=args.space, n_components=args.n_components,
        window=tuple(args.window) if args.window else None, deep=args.with_deep, structure=args.structure,
        induced=args.induced,
    )
    return [
        trajectories(project, stream=stream, lock=lock, settings=settings, subjects=args.subjects,
                     fsaverage_root=args.fsaverage_root)
        for stream in args.streams
        for lock in args.locks or LOCKS
    ]


def _analyze_band_choice(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.workflows.spectral import band_choice

    settings = BandChoiceConfig(lock=args.lock, aperiodic=args.aperiodic)
    return [
        band_choice(project, stream=stream, settings=settings, subjects=args.subjects, output_root=args.output_root)
        for stream in args.streams
    ]


def _analyze_spatial_decoding(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.workflows.spatial_decoding import run_spatial_decoding

    return run_spatial_decoding(
        project,
        settings=SpatialDecodingConfig(
            conditions=tuple(args.conditions),
            permutations=args.permutations,
            n_jobs=args.n_jobs,
        ),
        data_dir=args.data_dir,
        output_root=args.output_root,
    )


def _report_behavior(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.reports.behavior_battery import (
        list_behavior_figures,
        run_behavior_plotting,
    )

    if args.list_figures:
        result = [
            f"{spec.key}  [{', '.join(spec.groups)}]  "
            f"requires: {', '.join(spec.requires)}  -- {spec.title}"
            for spec in list_behavior_figures()
        ]
    else:
        result = run_behavior_plotting(
            str(args.behavior_root or project.bids_root),
            str(args.output_root or project.bids_root),
            subjects_list=args.subjects,
            figures=tuple(args.figures),
            formats=tuple(args.formats),
            skip_missing=args.skip_missing,
        )
    return result


def _report_band_choice(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.reports.bands import render_band_choice

    return render_band_choice(project, streams=args.streams,
                              settings=BandChoiceConfig(lock=args.lock, aperiodic=args.aperiodic),
                              output_root=args.output_root)


def _report_coreg(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.reports.coregistration import render_coregistration_qc

    return render_coregistration_qc(
        project, subjects=args.subjects, output_root=args.output_root
    )


def _report_statistics(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.reports.statistics import run_group_statistics_plotting

    return run_group_statistics_plotting(
        str(args.stats_root or project.bids_root),
        str(args.output_root or project.bids_root),
        conditions=args.conditions,
        align_to=args.alignment,
        source_method=args.source_method,
        parc=args.parc,
        labels=args.labels,
        top_n=args.top_n,
        alpha=args.alpha,
        peak_mode=args.peak_mode,
        behavior_dir=(
            str(args.behavior_root or project.bids_root)
            if args.correlate_behavior
            else None
        ),
        correlate_behavior=args.correlate_behavior,
    )


def _report_connectivity(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.reports.connectivity import run_batch_plot_connectivity_circle

    return run_batch_plot_connectivity_circle(
        str(args.feature_root or project.bids_root),
        str(args.output_root or project.bids_root),
        args.condition,
        band=args.band,
        p_threshold=args.p_threshold,
        n_permutations=args.permutations,
        node_names_path=args.node_names,
    )


def _report_seed_connectivity(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.reports.seed_connectivity import run_batch_plot_seed_connectivity

    return run_batch_plot_seed_connectivity(
        str(args.feature_root or project.bids_root),
        str(args.output_root or project.bids_root),
        args.condition,
        args.seed_roi,
        band=args.band,
        p_threshold=args.p_threshold,
        n_permutations=args.permutations,
        node_names_path=args.node_names,
    )


def _report_decoding_onset(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.reports.decoding import run_batch_plot_decoding_onset

    return run_batch_plot_decoding_onset(
        args.scores,
        args.thresholds,
        args.names,
        str(args.output_root or project.bids_root),
        time_offset=args.time_offset,
        time_step=args.time_step,
    )


def _report_spatial_decoding(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.reports.spatial_decoding import plot_spatial_decoding_topomap

    return plot_spatial_decoding_topomap(
        args.score_path,
        args.info_fif,
        str(args.output_root or project.bids_root),
    )


def _validate_golden(project: ProjectConfig, args: argparse.Namespace) -> object:
    from meg_tokens.validation import run_golden_validation

    result = run_golden_validation(
        args.comparison_config,
        out_tsv=args.out_tsv,
    )
    failed = result[result["status"] != "pass"]
    print(f"Comparisons: {len(result)}; failed: {len(failed)}")
    if len(failed) and not args.allow_failures:
        names = ", ".join(failed["name"].astype(str).tolist())
        raise SystemExit(f"Golden validation failed for: {names}")
    print(args.out_tsv)
    return None


_HANDLERS: dict[tuple[str, str], Handler] = {
    ("behavior", "ingest"): _behavior_ingest,
    ("behavior", "analyze"): _behavior_analyze,
    ("behavior", "characterization"): _behavior_characterization,
    ("behavior", "ssm-fit"): _behavior_ssm_fit,
    ("behavior", "ssm-evaluate"): _behavior_ssm_evaluate,
    ("behavior", "ssm-evaluate-aggregate"): _behavior_ssm_evaluate_aggregate,
    ("behavior", "ssm-aggregate"): _behavior_ssm_aggregate,
    ("behavior", "ssm-recovery"): _behavior_ssm_recovery,
    ("behavior", "ssm-recovery-aggregate"): _behavior_ssm_recovery_aggregate,
    ("behavior", "ssm-robustness"): _behavior_ssm_robustness,
    ("behavior", "ssm-robustness-aggregate"): _behavior_ssm_robustness_aggregate,
    ("behavior", "ssm-exclusion-robustness"): _behavior_ssm_exclusion_robustness,
    ("behavior", "ssm-exclusion-robustness-aggregate"): _behavior_ssm_exclusion_robustness_aggregate,
    ("behavior", "subjects"): _behavior_subjects,
    ("behavior", "qc"): _behavior_qc,
    ("meg", "stage-raw"): _meg_stage_raw,
    ("meg", "apply-raw-staging"): _meg_apply_raw_staging,
    ("meg", "preprocess"): _meg_preprocess,
    ("meg", "coreg"): _meg_coreg,
    ("meg", "epoch-amplitudes"): _meg_epoch_amplitudes,
    ("meg", "epoch-thresholds"): _meg_epoch_thresholds,
    ("meg", "epoch"): _meg_epoch,
    ("meg", "source-models"): _meg_source_models,
    ("meg", "source"): _meg_source,
    ("meg", "band-power"): _meg_band_power,
    ("meg", "band-power-merge"): _meg_band_power_merge,
    ("meg", "band-power-tasks"): _meg_band_power_tasks,
    ("meg", "erp"): _meg_erp,
    ("meg", "band-power-means"): _meg_band_power_means,
    ("meg", "band-power-prepare"): _meg_band_power_prepare,
    ("meg", "deep-power"): _meg_deep_power,
    ("meg", "deep-morph"): _meg_deep_morph,
    ("meg", "deep-power-means"): _meg_deep_power_means,
    ("meg", "sensor-spectra"): _meg_sensor_spectra,
    ("meg", "band-power-means-fsaverage"): _meg_band_power_means_fsaverage,
    ("features", "power"): _features_power,
    ("features", "hilbert"): _features_hilbert,
    ("features", "pac"): _features_pac,
    ("features", "connectivity"): _features_connectivity,
    ("analyze", "statistics"): _analyze_statistics,
    ("analyze", "lateralization"): _analyze_lateralization,
    ("analyze", "decoding"): _analyze_decoding,
    ("analyze", "trajectories"): _analyze_trajectories,
    ("analyze", "band-choice"): _analyze_band_choice,
    ("analyze", "spatial-decoding"): _analyze_spatial_decoding,
    ("report", "behavior"): _report_behavior,
    ("report", "band-choice"): _report_band_choice,
    ("report", "coreg"): _report_coreg,
    ("report", "statistics"): _report_statistics,
    ("report", "connectivity"): _report_connectivity,
    ("report", "seed-connectivity"): _report_seed_connectivity,
    ("report", "decoding-onset"): _report_decoding_onset,
    ("report", "spatial-decoding"): _report_spatial_decoding,
    ("validate", "golden"): _validate_golden,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Execute one workflow command."""
    # Set here rather than at module scope in the library: importing a module
    # for a path helper should not reconfigure logging for the whole process.
    mne.set_log_level("ERROR")
    parser = build_parser()
    args = parser.parse_args(argv)
    # The one place a subject label enters the system. Normalizing here means
    # `--subjects H1`, `h01` and `H01` are the same run from this line on, and
    # nothing downstream has to re-check: the disk is canonical after
    # BIDSification, so the only non-canonical labels are the ones typed here.
    if getattr(args, "subjects", None):
        try:
            args.subjects = [normalize_subject_id(s) for s in args.subjects]
        except ValueError as error:
            parser.error(str(error))
    command = getattr(args, _COMMAND_DEST[args.domain])
    try:
        project = _project_from_args(args)
        result = _HANDLERS[(args.domain, command)](project, args)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    if hasattr(result, "outputs"):
        outputs = result.outputs
    elif isinstance(result, dict):
        outputs = result.values()
    elif isinstance(result, (str, Path)):
        outputs = (result,)
    elif result is None:
        outputs = ()
    else:
        outputs = result
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    # Without this, `python -m meg_tokens.cli.main` defines these functions and
    # exits 0 having run nothing, so a reproduction instruction written that way
    # appears to succeed and writes no derivative.
    raise SystemExit(main())
