"""Unified MEG Tokens command-line interface."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

from meg_tokens import __version__
from meg_tokens.core import (
    ConnectivityConfig,
    DecodingConfig,
    DecompositionConfig,
    ERPConfig,
    EpochingConfig,
    HilbertConfig,
    LateralizedStatisticsConfig,
    PACConfig,
    PowerConfig,
    PreprocessingConfig,
    ProjectConfig,
    RunSpec,
    SourceConfig,
    SpatialDecodingConfig,
    SpectralConfig,
    StatisticsConfig,
)


def _add_root_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        dest="data_root",
        help=(
            "Data root containing raw/ (raw MEG), tdms/ (behavioral logs), "
            "and BIDS/ (derivatives) as siblings. Only used without --config."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the side-effect-free project command parser."""
    parser = argparse.ArgumentParser(prog="meg-tokens")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", help="Project TOML configuration.")
    domains = parser.add_subparsers(dest="domain", required=True)

    behavior = domains.add_parser("behavior", help="Behavior ingestion and analysis.")
    behavior_commands = behavior.add_subparsers(dest="behavior_command", required=True)

    ingest = behavior_commands.add_parser(
        "ingest",
        help="Parse TDMS runs into behavior derivatives.",
    )
    _add_root_option(ingest)
    ingest.add_argument("--subjects", nargs="+")
    ingest.add_argument("--dry-run", dest="dry_run", action="store_true")

    analyze = behavior_commands.add_parser(
        "analyze",
        help="Compute subject behavioral summaries from staged TSV files.",
    )
    _add_root_option(analyze)
    analyze.add_argument("--subjects", nargs="+")

    extended = behavior_commands.add_parser(
        "extended",
        help=(
            "Run the docs/behavior_analysis_roadmap.md analyses over the "
            "staged trial-feature table."
        ),
    )
    _add_root_option(extended)
    extended.add_argument("--subjects", nargs="+")
    extended.add_argument(
        "--neural-metrics",
        dest="neural_metrics",
        help=(
            "Optional subject-level MEG metrics table (TSV or CSV with a "
            "'subject' column) to join into the individual-difference profile."
        ),
    )

    qc = behavior_commands.add_parser(
        "qc",
        help="Validate source TDMS success-probability and SPD profiles.",
    )
    _add_root_option(qc)

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
    _add_root_option(stage_raw)
    stage_raw.add_argument("--subjects", nargs="+", required=True)

    apply_raw_staging_parser = meg_commands.add_parser(
        "apply-raw-staging",
        help=(
            "Stage 0: materialize BIDS/sub-*/{meg,beh,anat} and "
            "BIDS/sub-emptyroom from a Stage 0 manifest (fresh from "
            "`stage-raw`, or hand-reviewed). Only action=stage rows."
        ),
    )
    _add_root_option(apply_raw_staging_parser)
    apply_raw_staging_parser.add_argument("--subjects", nargs="+")
    apply_raw_staging_parser.add_argument(
        "--manifest",
        help="Manifest TSV to apply. Defaults to the project's Stage 0 manifest path.",
    )

    preprocess = meg_commands.add_parser(
        "preprocess",
        help="Filter one CTF recording and optionally apply ICA.",
    )
    _add_root_option(preprocess)
    preprocess.add_argument("--raw-path", dest="raw_path", required=True)
    preprocess.add_argument("--subject", required=True)
    preprocess.add_argument("--run", required=True)
    preprocess.add_argument("--condition")
    preprocess.add_argument("--high-pass", dest="high_pass", type=float, default=0.5)
    preprocess.add_argument("--low-pass", dest="low_pass", type=float, default=150.0)
    preprocess.add_argument("--notch-freqs", dest="notch_freqs", type=float, nargs="+")
    preprocess.add_argument("--run-ica", dest="run_ica", action="store_true")
    preprocess.add_argument("--ica-exclude", dest="ica_exclude", type=int, nargs="*")

    epoch = meg_commands.add_parser(
        "epoch",
        help="Build event-locked epochs from staged raw and behavior derivatives.",
    )
    _add_root_option(epoch)
    epoch.add_argument("--subjects", nargs="+", required=True)
    epoch.add_argument("--alignment", "--align-to", dest="alignment", choices=["go", "enter", "feedback"], default="go")
    epoch.add_argument("--tmin", type=float, default=-0.5)
    epoch.add_argument("--tmax", type=float, default=2.0)
    epoch.add_argument("--raw-root")
    epoch.add_argument("--behavior-root")

    source = meg_commands.add_parser(
        "source",
        help="Run source reconstruction stages for selected subjects.",
    )
    _add_root_option(source)
    source.add_argument("--subjects", nargs="+", required=True)
    source.add_argument("--run")
    source.add_argument("--condition")
    source.add_argument("--alignment", choices=["go", "enter", "feedback"], default="go")
    source.add_argument("--stages", nargs="+", choices=["cov", "bem", "src", "fwd", "inv", "apply"], default=["cov", "bem", "src", "fwd", "inv", "apply"])
    source.add_argument("--spacing", default="oct6")
    source.add_argument("--volume-labels", nargs="+")
    source.add_argument("--volume-pos", type=float, default=5.0)
    source.add_argument("--method", default="dSPM")
    source.add_argument("--snr", type=float, default=1.0)
    source.add_argument("--epochs-root")
    source.add_argument("--subjects-dir")
    source.add_argument("--trans-dir")
    source.add_argument("--noise-dir")

    features = domains.add_parser("features", help="Neural feature extraction.")
    feature_commands = features.add_subparsers(dest="feature_command", required=True)

    erp = feature_commands.add_parser("erp", help="Extract ERP source features.")
    _add_root_option(erp)
    erp.add_argument("--subjects", nargs="+", required=True)
    erp.add_argument("--run", required=True)
    erp.add_argument("--condition")
    erp.add_argument("--alignment", choices=["go", "enter", "feedback"], default="go")
    erp.add_argument("--source-method", default="dSPM")
    erp.add_argument("--parc", default="HCPMMP1")
    erp.add_argument("--feature-space", choices=["parcellated", "all_source", "volume"], default="parcellated")
    erp.add_argument("--hemi", choices=["left", "right", "both"], default="both")
    erp.add_argument("--label-subject")
    erp.add_argument("--spacing")
    erp.add_argument("--label-mode", default="mean")
    erp.add_argument("--max-duration-samples", type=int, default=400)
    erp.add_argument("--cutoff-before-enter-ms", type=float, default=300.0)
    erp.add_argument("--min-rt-ms", type=float, default=100.0)
    erp.add_argument("--source-root")
    erp.add_argument("--behavior-root")

    power = feature_commands.add_parser("power", help="Extract source-space band power.")
    _add_root_option(power)
    power.add_argument("--subjects", nargs="+", required=True)
    power.add_argument("--run", required=True)
    power.add_argument("--condition")
    power.add_argument("--alignment", choices=["go", "enter", "feedback"], default="go")
    power.add_argument("--source-method", default="dSPM")
    power.add_argument("--method", choices=["hilbert", "morlet", "multitaper"], default="hilbert")
    power.add_argument("--bands", nargs="+")
    power.add_argument("--width", type=int, default=400)
    power.add_argument("--step", type=int, default=110)
    power.add_argument("--baseline", type=float, nargs=2)
    power.add_argument("--baseline-method", choices=["percent", "ratio", "logratio", "zscore", "difference"], default="percent")
    power.add_argument("--n-jobs", type=int, default=1)
    power.add_argument("--source-root")

    spectral = feature_commands.add_parser("spectral", help="Compute Epochs PSD and specparam.")
    _add_root_option(spectral)
    spectral.add_argument("--subjects", nargs="+", required=True)
    spectral.add_argument("--run")
    spectral.add_argument("--condition")
    spectral.add_argument("--alignment", choices=["go", "enter", "feedback"])
    spectral.add_argument("--method", choices=["welch", "multitaper"], default="welch")
    spectral.add_argument("--fmin", type=float, default=1.0)
    spectral.add_argument("--fmax", type=float, default=100.0)
    spectral.add_argument("--n-fft", type=int, default=2048)
    spectral.add_argument("--n-overlap", type=int, default=150)
    spectral.add_argument("--n-jobs", type=int, default=1)
    spectral.add_argument("--no-specparam", action="store_true")
    spectral.add_argument("--epochs-root")

    hilbert = feature_commands.add_parser("hilbert", help="Extract analytic-signal features.")
    _add_root_option(hilbert)
    hilbert.add_argument("--subjects", nargs="+", required=True)
    hilbert.add_argument("--conditions", nargs="+", required=True)
    hilbert.add_argument("--alignment", choices=["go", "enter", "feedback"], default="go")
    hilbert.add_argument("--source-method", default="dSPM")
    hilbert.add_argument("--parc", default="HCPMMP1")
    hilbert.add_argument("--labels", nargs="+")
    hilbert.add_argument("--bands", nargs="+")
    hilbert.add_argument("--feature-types", nargs="+", default=["amplitude", "power", "phase", "sigfilt"])
    hilbert.add_argument("--sfreq", type=float)
    hilbert.add_argument("--n-jobs", type=int, default=1)
    hilbert.add_argument("--feature-root")

    pac = feature_commands.add_parser("pac", help="Compute phase-amplitude coupling.")
    _add_root_option(pac)
    pac.add_argument("--subjects", nargs="+", required=True)
    pac.add_argument("--conditions", nargs="+", required=True)
    pac.add_argument("--phase-bands", nargs="+", required=True)
    pac.add_argument("--amplitude-bands", nargs="+", required=True)
    pac.add_argument("--alignment", choices=["go", "enter", "feedback"], default="go")
    pac.add_argument("--source-method", default="dSPM")
    pac.add_argument("--parc", default="HCPMMP1")
    pac.add_argument("--n-bins", type=int, default=18)
    pac.add_argument("--time-window", type=float, nargs=2)
    pac.add_argument("--feature-root")

    connectivity = feature_commands.add_parser("connectivity", help="Compute spectral connectivity.")
    _add_root_option(connectivity)
    connectivity.add_argument("--subjects", nargs="+", required=True)
    connectivity.add_argument("--conditions", nargs="+", required=True)
    connectivity.add_argument("--alignment", choices=["go", "enter", "feedback"], default="enter")
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
    _add_root_option(statistics)
    statistics.add_argument("--conditions", nargs=2, default=["Fast", "Slow"])
    statistics.add_argument("--subjects", nargs="+")
    statistics.add_argument("--alignment", choices=["go", "enter", "feedback"], default="go")
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
    _add_root_option(lateralization)
    lateralization.add_argument("--condition", required=True)
    lateralization.add_argument("--subjects", nargs="+")
    lateralization.add_argument("--alignment", choices=["go", "enter", "feedback"], default="go")
    lateralization.add_argument("--source-method", default="dSPM")
    lateralization.add_argument("--parc", default="HCPMMP1")
    lateralization.add_argument("--runs", nargs="+")
    lateralization.add_argument("--permutations", type=int, default=1000)
    lateralization.add_argument("--alpha", type=float, default=0.05)
    lateralization.add_argument("--tail", type=int, choices=[-1, 0, 1], default=0)
    lateralization.add_argument("--n-jobs", type=int, default=1)
    lateralization.add_argument("--feature-root")

    decoding = analysis_commands.add_parser("decoding", help="Run time-resolved decoding.")
    _add_root_option(decoding)
    decoding.add_argument("--conditions", nargs="+", default=["Fast", "Slow"])
    decoding.add_argument("--input-conditions", nargs="+")
    decoding.add_argument("--subjects", nargs="+")
    decoding.add_argument("--feature-source", choices=["erp", "power"], default="erp")
    decoding.add_argument("--alignment", choices=["go", "enter", "feedback"], default="go")
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

    decomposition = analysis_commands.add_parser("decomposition", help="Run PCA trajectories or dPCA.")
    _add_root_option(decomposition)
    decomposition.add_argument("--conditions", nargs="+", default=["Fast", "Slow"])
    decomposition.add_argument("--subjects", nargs="+")
    decomposition.add_argument("--analysis", choices=["pca", "dpca"], default="pca")
    decomposition.add_argument("--feature-source", choices=["erp", "power"], default="erp")
    decomposition.add_argument("--alignment", choices=["go", "enter", "feedback"], default="go")
    decomposition.add_argument("--source-method", default="dSPM")
    decomposition.add_argument("--parc", default="HCPMMP1")
    decomposition.add_argument("--band")
    decomposition.add_argument("--power-method", default="hilbert")
    decomposition.add_argument("--labels", nargs="+")
    decomposition.add_argument("--lateralize", action="store_true")
    decomposition.add_argument("--average-unit", choices=["subject", "trial"], default="subject")
    decomposition.add_argument("--transform", choices=["sqrt", "signed-sqrt"])
    decomposition.add_argument("--n-components", type=int, default=20)
    decomposition.add_argument("--min-variance", type=float)
    decomposition.add_argument("--fit-time-range", type=float, nargs=2)
    decomposition.add_argument("--project-centered", action="store_true")
    decomposition.add_argument("--marginalize-cols", nargs="+")
    decomposition.add_argument("--dpca-labels")
    decomposition.add_argument("--feature-root")

    spatial_decoding = analysis_commands.add_parser(
        "spatial-decoding",
        help="Run sensor-space decoding on staged trial arrays.",
    )
    _add_root_option(spatial_decoding)
    spatial_decoding.add_argument("--data-dir", required=True)
    spatial_decoding.add_argument("--conditions", nargs="+", default=["Fast", "Slow"])
    spatial_decoding.add_argument("--permutations", type=int, default=0)
    spatial_decoding.add_argument("--n-jobs", type=int, default=4)
    spatial_decoding.add_argument("--output-root")

    report = domains.add_parser("report", help="Generate figures and summary tables.")
    report_commands = report.add_subparsers(dest="report_command", required=True)

    behavior_report = report_commands.add_parser("behavior", help="Plot staged behavioral results.")
    _add_root_option(behavior_report)
    behavior_report.add_argument("--subjects", nargs="+")
    behavior_report.add_argument("--behavior-root")
    behavior_report.add_argument("--output-root")
    behavior_report.add_argument("--neural-metrics")

    statistics_report = report_commands.add_parser("statistics", help="Plot group statistics.")
    _add_root_option(statistics_report)
    statistics_report.add_argument("--conditions", nargs=2, default=["Fast", "Slow"])
    statistics_report.add_argument("--alignment", default="go", choices=["go", "enter", "feedback"])
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
    _add_root_option(connectivity_report)
    connectivity_report.add_argument("--condition", required=True)
    connectivity_report.add_argument("--band", default="alpha")
    connectivity_report.add_argument("--p-threshold", type=float, default=0.05)
    connectivity_report.add_argument("--permutations", type=int, default=1000)
    connectivity_report.add_argument("--node-names")
    connectivity_report.add_argument("--feature-root")
    connectivity_report.add_argument("--output-root")

    seed_report = report_commands.add_parser("seed-connectivity", help="Create a seed-connectivity report.")
    _add_root_option(seed_report)
    seed_report.add_argument("--condition", required=True)
    seed_report.add_argument("--seed-roi", required=True)
    seed_report.add_argument("--band", default="alpha")
    seed_report.add_argument("--p-threshold", type=float, default=0.05)
    seed_report.add_argument("--permutations", type=int, default=1000)
    seed_report.add_argument("--node-names")
    seed_report.add_argument("--feature-root")
    seed_report.add_argument("--output-root")

    onset_report = report_commands.add_parser("decoding-onset", help="Compare decoding onset times.")
    _add_root_option(onset_report)
    onset_report.add_argument("--scores", nargs="+", required=True)
    onset_report.add_argument("--thresholds", nargs="+", required=True)
    onset_report.add_argument("--names", nargs="+", required=True)
    onset_report.add_argument("--time-offset", type=float, default=-1000)
    onset_report.add_argument("--time-step", type=float, default=50)
    onset_report.add_argument("--output-root")

    spatial_report = report_commands.add_parser("spatial-decoding", help="Plot sensor decoding scores.")
    _add_root_option(spatial_report)
    spatial_report.add_argument("--score-path", required=True)
    spatial_report.add_argument("--info-fif", required=True)
    spatial_report.add_argument("--output-root")

    trajectory_report = report_commands.add_parser("pca-trajectory", help="Plot PCA trajectories.")
    _add_root_option(trajectory_report)
    trajectory_report.add_argument("--timecourse-path", required=True)
    trajectory_report.add_argument("--conditions", nargs="+")
    trajectory_report.add_argument("--components", type=int, nargs="+", default=[1, 2, 3])
    trajectory_report.add_argument("--output-root")

    timecourse_report = report_commands.add_parser("pca-timecourse", help="Plot PCA component time courses.")
    _add_root_option(timecourse_report)
    timecourse_report.add_argument("--timecourse-path", required=True)
    timecourse_report.add_argument("--components", type=int, default=3)
    timecourse_report.add_argument("--conditions", nargs="+")
    timecourse_report.add_argument("--output-root")

    variance_report = report_commands.add_parser("pca-variance", help="Plot PCA explained variance.")
    _add_root_option(variance_report)
    variance_report.add_argument("--variance-path", required=True)
    variance_report.add_argument("--output-root")

    heatmap_report = report_commands.add_parser("pca-heatmap", help="Plot a PCA ROI heatmap.")
    _add_root_option(heatmap_report)
    heatmap_report.add_argument("--data-path", required=True)
    heatmap_report.add_argument("--rois", nargs="+")
    heatmap_report.add_argument("--output-root")

    loadings_report = report_commands.add_parser("pca-loadings", help="Plot PCA source loadings.")
    _add_root_option(loadings_report)
    loadings_report.add_argument("--loadings-path", required=True)
    loadings_report.add_argument("--subjects-dir")
    loadings_report.add_argument("--threshold-percentile", type=float)
    loadings_report.add_argument("--export-csv", action="store_true")
    loadings_report.add_argument("--save-movie", action="store_true")
    loadings_report.add_argument("--vertices-path")
    loadings_report.add_argument("--volume", action="store_true")
    loadings_report.add_argument("--output-root")

    dpca_report = report_commands.add_parser("dpca", help="Plot dPCA components.")
    _add_root_option(dpca_report)
    dpca_report.add_argument("--component-paths", nargs="+")
    dpca_report.add_argument("--dpca-root")
    dpca_report.add_argument("--n-components", type=int, default=3)
    dpca_report.add_argument("--output-root")

    validate = domains.add_parser("validate", help="Validate staged derivatives.")
    validation_commands = validate.add_subparsers(dest="validation_command", required=True)
    golden = validation_commands.add_parser("golden", help="Compare derivatives with real references.")
    _add_root_option(golden)
    golden.add_argument("--comparison-config", required=True)
    golden.add_argument("--out-tsv", required=True)
    golden.add_argument("--allow-failures", action="store_true")
    return parser


def _project_from_args(args: argparse.Namespace) -> ProjectConfig:
    if args.config:
        project = ProjectConfig.from_toml(args.config)
    else:
        if not args.data_root:
            raise ValueError("Provide --config or --data-root")
        project = ProjectConfig(data_root=Path(args.data_root))

    updates = {}
    for argument, field_name in (
        ("subjects_dir", "subjects_dir"),
        ("trans_dir", "trans_dir"),
        ("noise_dir", "noise_dir"),
    ):
        value = getattr(args, argument, None)
        if value:
            updates[field_name] = Path(value)
    return replace(project, **updates) if updates else project


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Execute one workflow command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        project = _project_from_args(args)
        if args.domain == "behavior" and args.behavior_command == "ingest":
            from meg_tokens.workflows.behavior_ingest import ingest_behavior

            result = ingest_behavior(
                project,
                subjects=args.subjects,
                dry_run=args.dry_run,
            )
        elif args.domain == "behavior" and args.behavior_command == "analyze":
            from meg_tokens.workflows.behavior_analysis import analyze_behavior

            result = analyze_behavior(project, subjects=args.subjects)
        elif args.domain == "behavior" and args.behavior_command == "extended":
            from meg_tokens.workflows.behavior_extended import analyze_behavior_extended

            result = analyze_behavior_extended(
                project,
                subjects=args.subjects,
                neural_metrics_path=args.neural_metrics,
            )
        elif args.domain == "behavior" and args.behavior_command == "qc":
            from meg_tokens.validation import run_spd_validation

            summary, details = run_spd_validation(
                project.behavior_root,
                ignore_files=project.behavior_ignore_files,
            )
            print(summary.to_string(index=False))
            print(f"Validated trials: {len(details)}")
            return 0
        elif args.domain == "meg" and args.meg_command == "stage-raw":
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
            return 0
        elif args.domain == "meg" and args.meg_command == "apply-raw-staging":
            from meg_tokens.workflows.raw_staging import apply_raw_staging

            result = apply_raw_staging(
                project,
                manifest_path=args.manifest,
                subjects=args.subjects,
            )
        elif args.domain == "meg" and args.meg_command == "preprocess":
            from meg_tokens.workflows.preprocessing import preprocess_run

            result = preprocess_run(
                project,
                RunSpec(
                    subject=args.subject,
                    run=args.run,
                    condition=args.condition,
                ),
                raw_path=args.raw_path,
                settings=PreprocessingConfig(
                    high_pass=args.high_pass,
                    low_pass=args.low_pass,
                    notch_freqs=tuple(args.notch_freqs) if args.notch_freqs else None,
                    run_ica=args.run_ica,
                    ica_exclude=tuple(args.ica_exclude) if args.ica_exclude is not None else None,
                ),
            )
        elif args.domain == "meg" and args.meg_command == "epoch":
            from meg_tokens.workflows.preprocessing import epoch_subjects

            result = epoch_subjects(
                project,
                subjects=args.subjects,
                settings=EpochingConfig(
                    tmin=args.tmin,
                    tmax=args.tmax,
                    alignment=args.alignment,
                ),
                raw_root=args.raw_root,
                behavior_root=args.behavior_root,
            )
        elif args.domain == "meg" and args.meg_command == "source":
            from meg_tokens.workflows.sources import reconstruct_sources

            result = reconstruct_sources(
                project,
                subjects=args.subjects,
                settings=SourceConfig(
                    stages=tuple(args.stages),
                    spacing=args.spacing,
                    volume_labels=tuple(args.volume_labels) if args.volume_labels else None,
                    volume_pos=args.volume_pos,
                    run=args.run,
                    condition=args.condition,
                    alignment=args.alignment,
                    method=args.method,
                    snr=args.snr,
                ),
                epochs_root=args.epochs_root,
            )
        elif args.domain == "features" and args.feature_command == "erp":
            from meg_tokens.workflows.erp import extract_erp_features

            result = extract_erp_features(
                project,
                subjects=args.subjects,
                settings=ERPConfig(
                    run=args.run,
                    condition=args.condition,
                    alignment=args.alignment,
                    source_method=args.source_method,
                    parc=args.parc,
                    feature_space=args.feature_space,
                    hemi=args.hemi,
                    label_subject=args.label_subject,
                    spacing=args.spacing,
                    label_mode=args.label_mode,
                    max_duration_samples=args.max_duration_samples,
                    cutoff_before_enter_ms=args.cutoff_before_enter_ms,
                    min_rt_ms=args.min_rt_ms,
                ),
                source_root=args.source_root,
                behavior_root=args.behavior_root,
            )
        elif args.domain == "features" and args.feature_command == "power":
            from meg_tokens.workflows.power import extract_power_features, parse_frequency_bands

            parsed_bands = parse_frequency_bands(args.bands) if args.bands else None
            result = extract_power_features(
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
        elif args.domain == "features" and args.feature_command == "spectral":
            from meg_tokens.workflows.spectral import extract_spectral_features

            result = extract_spectral_features(
                project,
                subjects=args.subjects,
                settings=SpectralConfig(
                    run=args.run,
                    condition=args.condition,
                    alignment=args.alignment,
                    method=args.method,
                    fmin=args.fmin,
                    fmax=args.fmax,
                    n_fft=args.n_fft,
                    n_overlap=args.n_overlap,
                    n_jobs=args.n_jobs,
                    fit_model=not args.no_specparam,
                ),
                epochs_root=args.epochs_root,
            )
        elif args.domain == "features" and args.feature_command == "hilbert":
            from meg_tokens.features.time_frequency import parse_frequency_bands
            from meg_tokens.workflows.hilbert import extract_hilbert_features

            parsed_bands = parse_frequency_bands(args.bands) if args.bands else None
            result = extract_hilbert_features(
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
        elif args.domain == "features" and args.feature_command == "pac":
            from meg_tokens.workflows.pac import extract_pac_features

            result = extract_pac_features(
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
        elif args.domain == "features" and args.feature_command == "connectivity":
            from meg_tokens.workflows.connectivity import _parse_bands, extract_connectivity_features

            names, lower, upper = _parse_bands(args.bands)
            result = extract_connectivity_features(
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
        elif args.domain == "analyze" and args.analysis_command == "statistics":
            from meg_tokens.workflows.statistics import run_group_statistics

            result = run_group_statistics(
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
        elif args.domain == "analyze" and args.analysis_command == "lateralization":
            from meg_tokens.workflows.statistics import run_lateralized_statistics

            result = run_lateralized_statistics(
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
        elif args.domain == "analyze" and args.analysis_command == "decoding":
            from meg_tokens.workflows.decoding import run_decoding

            result = run_decoding(
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
        elif args.domain == "analyze" and args.analysis_command == "decomposition":
            from meg_tokens.workflows.decomposition import run_decomposition

            result = run_decomposition(
                project,
                subjects=args.subjects,
                settings=DecompositionConfig(
                    conditions=tuple(args.conditions),
                    analysis=args.analysis,
                    feature_source=args.feature_source,
                    alignment=args.alignment,
                    source_method=args.source_method,
                    parc=args.parc,
                    band=args.band,
                    power_method=args.power_method,
                    labels=tuple(args.labels) if args.labels else None,
                    lateralize=args.lateralize,
                    average_unit=args.average_unit,
                    transform=args.transform,
                    n_components=args.n_components,
                    min_variance=args.min_variance,
                    fit_time_range=tuple(args.fit_time_range) if args.fit_time_range else None,
                    project_centered=args.project_centered,
                    marginalize_cols=tuple(args.marginalize_cols) if args.marginalize_cols else None,
                    dpca_labels=args.dpca_labels,
                ),
                feature_root=args.feature_root,
            )
        elif args.domain == "analyze" and args.analysis_command == "spatial-decoding":
            from meg_tokens.workflows.spatial_decoding import run_spatial_decoding

            result = run_spatial_decoding(
                project,
                settings=SpatialDecodingConfig(
                    conditions=tuple(args.conditions),
                    permutations=args.permutations,
                    n_jobs=args.n_jobs,
                ),
                data_dir=args.data_dir,
                output_root=args.output_root,
            )
        elif args.domain == "report" and args.report_command == "behavior":
            from meg_tokens.reports.behavior_summary import run_behavior_plotting

            result = run_behavior_plotting(
                str(args.behavior_root or project.bids_root),
                str(args.output_root or project.bids_root),
                subjects_list=args.subjects,
                neural_metrics_csv=args.neural_metrics,
            )
        elif args.domain == "report" and args.report_command == "statistics":
            from meg_tokens.reports.statistics import run_group_statistics_plotting

            result = run_group_statistics_plotting(
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
        elif args.domain == "report" and args.report_command == "connectivity":
            from meg_tokens.reports.connectivity import run_batch_plot_connectivity_circle

            result = run_batch_plot_connectivity_circle(
                str(args.feature_root or project.bids_root),
                str(args.output_root or project.bids_root),
                args.condition,
                band=args.band,
                p_threshold=args.p_threshold,
                n_permutations=args.permutations,
                node_names_path=args.node_names,
            )
        elif args.domain == "report" and args.report_command == "seed-connectivity":
            from meg_tokens.reports.seed_connectivity import run_batch_plot_seed_connectivity

            result = run_batch_plot_seed_connectivity(
                str(args.feature_root or project.bids_root),
                str(args.output_root or project.bids_root),
                args.condition,
                args.seed_roi,
                band=args.band,
                p_threshold=args.p_threshold,
                n_permutations=args.permutations,
                node_names_path=args.node_names,
            )
        elif args.domain == "report" and args.report_command == "decoding-onset":
            from meg_tokens.reports.decoding import run_batch_plot_decoding_onset

            result = run_batch_plot_decoding_onset(
                args.scores,
                args.thresholds,
                args.names,
                str(args.output_root or project.bids_root),
                time_offset=args.time_offset,
                time_step=args.time_step,
            )
        elif args.domain == "report" and args.report_command == "spatial-decoding":
            from meg_tokens.reports.spatial_decoding import plot_spatial_decoding_topomap

            result = plot_spatial_decoding_topomap(
                args.score_path,
                args.info_fif,
                str(args.output_root or project.bids_root),
            )
        elif args.domain == "report" and args.report_command == "pca-trajectory":
            from meg_tokens.reports.decomposition.trajectory import run_batch_plot_pca_trajectory

            result = run_batch_plot_pca_trajectory(
                args.timecourse_path,
                str(args.output_root or project.bids_root),
                args.conditions,
                args.components,
            )
        elif args.domain == "report" and args.report_command == "pca-timecourse":
            from meg_tokens.reports.decomposition.timecourse import run_batch_plot_component_timecourse

            result = run_batch_plot_component_timecourse(
                args.timecourse_path,
                str(args.output_root or project.bids_root),
                args.components,
                args.conditions,
            )
        elif args.domain == "report" and args.report_command == "pca-variance":
            from meg_tokens.reports.decomposition.variance import run_batch_plot_pca_variance

            result = run_batch_plot_pca_variance(
                args.variance_path,
                str(args.output_root or project.bids_root),
            )
        elif args.domain == "report" and args.report_command == "pca-heatmap":
            from meg_tokens.reports.decomposition.heatmap import run_batch_plot_pca_heatmap

            result = run_batch_plot_pca_heatmap(
                args.data_path,
                str(args.output_root or project.bids_root),
                args.rois,
            )
        elif args.domain == "report" and args.report_command == "pca-loadings":
            from meg_tokens.reports.decomposition.loadings import run_batch_plot_pca_loadings

            result = run_batch_plot_pca_loadings(
                args.loadings_path,
                str(args.output_root or project.bids_root),
                args.subjects_dir or (
                    str(project.subjects_dir) if project.subjects_dir else None
                ),
                args.threshold_percentile,
                args.export_csv,
                args.save_movie,
                args.vertices_path,
                args.volume,
            )
        elif args.domain == "report" and args.report_command == "dpca":
            from meg_tokens.reports.decomposition.dpca import run_batch_plot_dpca

            result = run_batch_plot_dpca(
                component_paths=args.component_paths,
                dpca_dir=args.dpca_root,
                output_dir=str(args.output_root or project.bids_root),
                n_components_to_plot=args.n_components,
            )
        elif args.domain == "validate" and args.validation_command == "golden":
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
            return 0
        else:  # pragma: no cover - argparse enforces the command tree
            parser.error("Unknown command")
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
