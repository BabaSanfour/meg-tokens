"""Canonical behavioral feature and analysis workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from meg_tokens.behavior.analyses.performance import behavior_group_statistics
from meg_tokens.behavior.analyses.summary import summarize_behavior
from meg_tokens.behavior.features import (
    build_trial_features,
    calculate_motor_baseline,
)
from meg_tokens.behavior.tables import read_behavior_table
from meg_tokens.behavior.trials import started_trials
from meg_tokens.core import ProjectConfig, WorkflowResult, normalize_subject_id
from meg_tokens.io import DerivativeLayout, save_table


def _condition_runs(tables: Sequence[pd.DataFrame], condition: str) -> list[pd.DataFrame]:
    return [
        table
        for table in tables
        if not table.empty
        and str(table["condition"].iloc[0]).lower() == condition.lower()
    ]


def _started_condition_runs(
    tables: Sequence[pd.DataFrame],
    condition: str,
) -> list[pd.DataFrame]:
    """Return analysis views containing only trials that received a go cue."""
    return [
        started_trials(table)
        for table in _condition_runs(tables, condition)
    ]


def analyze_behavior(
    project: ProjectConfig,
    *,
    subjects: Optional[Sequence[str]] = None,
) -> WorkflowResult:
    """Build canonical trial features and behavioral analysis derivatives."""
    layout = DerivativeLayout(
        project.bids_root,
        task=project.task,
    )
    paths = layout.behavior_tables(subjects=subjects)
    excluded_subjects = set(project.subject_exclusions)
    by_subject: dict[str, list[pd.DataFrame]] = {}
    source_paths: dict[str, list[Path]] = {}
    for path in paths:
        table = read_behavior_table(path)
        if table.empty:
            continue
        subject = normalize_subject_id(str(table["subject"].iloc[0]))
        if subject in excluded_subjects:
            continue
        by_subject.setdefault(subject, []).append(table)
        source_paths.setdefault(subject, []).append(path)
    if not by_subject:
        raise ValueError("Behavior derivatives do not contain any trials")

    motor_baselines = {
        subject: calculate_motor_baseline(_started_condition_runs(tables, "RT"))
        for subject, tables in sorted(by_subject.items())
    }
    trial_features = build_trial_features(by_subject, motor_baselines)

    subject_summary = summarize_behavior(trial_features)
    output_path = layout.behavior_summary()
    save_table(
        output_path,
        subject_summary,
        metadata={
            "stage": "behavior_analysis",
            "subjects": sorted(by_subject),
            "excluded_subjects": sorted(excluded_subjects),
            "spd": {
                "source": "logged chosen-target nProb paired with tTime",
                "views": ["all_logged", "validated_15row"],
                "short_log_design_alignment": "forbidden",
            },
            "input_files": [
                str(path)
                for subject in sorted(source_paths)
                for path in source_paths[subject]
            ],
        },
    )

    group_path = layout.behavior_group_statistics()
    save_table(
        group_path,
        behavior_group_statistics(subject_summary),
        metadata={
            "stage": "behavior_group_statistics",
            "subjects": sorted(by_subject),
            "excluded_subjects": sorted(excluded_subjects),
            "test": "paired_t_test",
            "effect_size": "cohens_dz",
            "spd_views": ["all_logged", "validated_15row"],
            "subject_summary": str(output_path),
        },
    )

    trial_features_path = layout.behavior_trial_features()
    save_table(
        trial_features_path,
        trial_features,
        metadata={
            "stage": "behavior_trial_features",
            "subjects": sorted(by_subject),
            "excluded_subjects": sorted(excluded_subjects),
            "join_key": ["subject", "condition", "run", "run_trial_index"],
            "dt": "rawRT - subject motor_baseline_ms; task trials only",
            "spd": "logged chosen-target nProb at the motor-corrected decision time",
            "evidence_at_decision": "logged_spd - 0.5",
            "design_evidence": (
                "correct-target success probability and log posterior odds from "
                "the designed sTokenDirs sequence; 'early' is after three jumps"
            ),
            "input_files": [str(path) for path in paths],
        },
    )
    return WorkflowResult(
        stage="behavior_analysis",
        inputs=tuple(paths),
        outputs=(output_path, group_path, trial_features_path),
        settings={
            "subjects": sorted(by_subject),
            "excluded_subjects": sorted(excluded_subjects),
            "n_subjects": len(by_subject),
        },
    )
