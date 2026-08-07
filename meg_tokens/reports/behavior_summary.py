"""
Report generation for behavioral distributions and
Stage 8 Brain-Behavior Correlations (e.g. Success Probability & Latencies).
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from meg_tokens.behavior.tdms import started_trials
from meg_tokens.io import DerivativeLayout, require_file, save_sidecar
from meg_tokens.reports.behavior import plot_fast_slow_distributions
from meg_tokens.reports.meg import plot_correlation


def _behavior_entities(path: Path) -> tuple[str, str]:
    subject = path.name.split("_", 1)[0].replace("sub-", "")
    condition = path.stem.split("_desc-", 1)[1].rsplit("_beh", 1)[0]
    return subject, condition


def run_behavior_plotting(
    behavior_dir: str,
    output_figures_dir: str,
    subjects_list: list[str] | None = None,
    neural_metrics_csv: str | None = None,
) -> list[Path]:
    """Plot real staged behavior and optional subject-level neural correlation."""
    input_layout = DerivativeLayout(behavior_dir)
    output_layout = DerivativeLayout(output_figures_dir)
    tables = input_layout.behavior_tables(
        subjects=subjects_list,
        conditions=("Fast", "Slow"),
    )
    values_by_condition = {"fast": [], "slow": []}
    values_by_subject: dict[str, list[float]] = {}
    for path in tables:
        subject, condition = _behavior_entities(path)
        table = pd.read_csv(path, sep="\t")
        if "rawRT" not in table.columns:
            raise ValueError(f"Behavior derivative {path} is missing canonical rawRT")
        table = started_trials(table)
        values = pd.to_numeric(table["rawRT"], errors="coerce").dropna().to_numpy()
        key = condition.lower()
        if key in values_by_condition:
            values_by_condition[key].extend(values)
        values_by_subject.setdefault(subject, []).extend(values)

    if not values_by_condition["fast"] or not values_by_condition["slow"]:
        raise ValueError("Fast and Slow behavior derivatives must both contain finite rawRT values")

    figure_path = output_layout.path(
        subject="group",
        datatype="fig",
        description="fast-vs-slow",
        suffix="behavior",
        extension=".png",
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure = plot_fast_slow_distributions(
        dt_fast=np.asarray(values_by_condition["fast"]),
        dt_slow=np.asarray(values_by_condition["slow"]),
        title="Raw Response Time Distribution: Fast vs Slow",
        save_path=str(figure_path),
    )
    plt.close(figure)
    save_sidecar(
        figure_path,
        {
            "stage": "behavior_report",
            "kind": "raw_response_time_distribution",
            "conditions": ["Fast", "Slow"],
            "metric": "rawRT",
            "inputs": [str(path) for path in tables],
        },
    )
    outputs = [figure_path]

    if neural_metrics_csv:
        metrics = pd.read_csv(require_file(neural_metrics_csv, purpose="neural metrics for behavior correlation"))
        required = {"subject", "neural_peak_ms"}
        missing = required - set(metrics.columns)
        if missing:
            raise ValueError(f"Neural metrics table is missing columns: {sorted(missing)}")

        mean_subject_rts = []
        neural_peaks = []
        for subject, values in sorted(values_by_subject.items()):
            metric_rows = metrics.loc[metrics["subject"] == subject, "neural_peak_ms"]
            if values and not metric_rows.empty:
                mean_subject_rts.append(float(np.mean(values)))
                neural_peaks.append(float(metric_rows.iloc[0]))

        if len(mean_subject_rts) < 3:
            raise ValueError("At least three subjects with behavior and neural metrics are required for correlation")

        fig, ax = plt.subplots(figsize=(8, 8))
        plot_correlation(
            x_data=np.asarray(mean_subject_rts),
            y_data=np.asarray(neural_peaks),
            x_label='Mean raw response time (ms)',
            y_label='Neural Peak Commitment (ms)',
            title='Correlation: Behavior vs. Neural Peak',
            ax=ax
        )
        correlation_path = output_layout.path(
            subject="group",
            datatype="fig",
            description="behavior-neural",
            suffix="correlation",
            extension=".png",
        )
        correlation_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(correlation_path, dpi=300)
        plt.close(fig)
        save_sidecar(
            correlation_path,
            {
                "stage": "behavior_report",
                "kind": "behavior_neural_correlation",
                "behavior_metric": "rawRT",
                "neural_metric": "neural_peak_ms",
                "neural_metrics_table": str(neural_metrics_csv),
                "inputs": [str(path) for path in tables],
            },
        )
        outputs.append(correlation_path)
    return outputs
