import json

import pandas as pd
import pytest

from meg_tokens.behavior.schema import OUTCOME_NEVER_STARTED
from meg_tokens.cli import main
from meg_tokens.core import ProjectConfig
from meg_tokens.io import DerivativeLayout, save_table, sidecar_path
from meg_tokens.workflows import analyze_behavior, ingest_behavior


def _behavior_table(subject, condition, run, enter_times, choices, correct):
    n_trials = len(enter_times)
    return pd.DataFrame(
        {
            "subject": [subject] * n_trials,
            "condition": [condition] * n_trials,
            "run": [run] * n_trials,
            "source_file": [f"{subject}{condition}{run}_recording.tdms"] * n_trials,
            "nTrialIndex": list(range(1, n_trials + 1)),
            "sTrialClass": list(range(1, n_trials + 1)),
            "sTrialClassRaw": ["e", "a"][:n_trials],
            "trial_class_source": ["design"] * n_trials,
            "trial_class_rule": ["recorded_label"] * n_trials,
            "sp_design_correct": ["[0.6, 0.8]"] * n_trials,
            "nInitialTime": [0] * n_trials,
            "nChoiceMade": choices,
            "nCorrectChoice": correct,
            "tGO": [1000] * n_trials,
            "tEnterCenter": [0] * n_trials,
            "tExitCenter": enter_times,
            "tEnterTarget": enter_times,
            "tTrialEnd": [value + 200 for value in enter_times],
            "sTokenDirs": ["121"] * n_trials,
            "nTokenNum": ["[1, 2]"] * n_trials,
            "nTokenDir": ["[1, 2]"] * n_trials,
            "tTime": ["[1100, 1300]"] * n_trials,
            "nProb": ["[0.6, 0.8]"] * n_trials,
            "token_log_rows": [2] * n_trials,
            "token_log_short": [False] * n_trials,
            "nOutcome": [0] * n_trials,
            "rawRT": [value - 1000 for value in enter_times],
            "isCorrect": [
                choice == expected
                for choice, expected in zip(choices, correct)
            ],
        }
    )


def _write_behavior_inputs(bids_root):
    layout = DerivativeLayout(bids_root)
    tables = [
        _behavior_table("H01", "RT", 1, [1300, 1400], [1, 2], [1, 2]),
        _behavior_table("H01", "Fast", 2, [1900, 2100], [1, 2], [1, 1]),
        _behavior_table("H01", "Slow", 3, [2400, 2600], [1, 2], [1, 2]),
    ]
    for table in tables:
        condition = table["condition"].iloc[0]
        run = str(table["run"].iloc[0])
        save_table(
            layout.behavior(subject="H01", run=run, condition=condition),
            table,
            metadata={"stage": "behavior_parsing"},
        )


def test_analyze_behavior_writes_group_summary(tmp_path):
    project = ProjectConfig(data_root=tmp_path)
    _write_behavior_inputs(project.bids_root)

    result = analyze_behavior(project)

    assert result.stage == "behavior_analysis"
    layout = DerivativeLayout(project.bids_root)
    assert result.outputs == (
        layout.behavior_summary(),
        layout.behavior_group_statistics(),
        layout.behavior_trial_features(),
    )
    summary = pd.read_csv(result.outputs[0], sep="\t")
    assert summary["subject"].tolist() == ["H01"]
    assert summary["motor_baseline_ms"].iloc[0] == pytest.approx(350.0)
    assert summary["mean_fast_dt_ms"].iloc[0] == pytest.approx(650.0)
    assert summary["mean_slow_dt_ms"].iloc[0] == pytest.approx(1150.0)
    assert summary["percent_correct"].iloc[0] == pytest.approx(75.0)
    assert summary["n_fast_error_trials"].iloc[0] == 1
    assert summary["n_slow_error_trials"].iloc[0] == 0
    assert summary["n_spd_all_logged"].iloc[0] == 4
    assert summary["mean_spd_all_logged"].iloc[0] == pytest.approx(0.8)
    assert summary["n_spd_validated_15row"].iloc[0] == 0
    assert pd.isna(summary["mean_spd_validated_15row"].iloc[0])

    group = pd.read_csv(result.outputs[1], sep="\t")
    assert len(group) == 11
    assert set(group["analysis"]) == {"decision_time", "error_count", "spd"}
    assert group["n_subjects"].max() == 1

    features = pd.read_csv(result.outputs[2], sep="\t")
    assert len(features) == 6
    assert not features["trial_id"].duplicated().any()
    assert features["primary_analysis_eligible"].sum() == 4
    task_features = features.loc[features["condition"].isin(["Fast", "Slow"])]
    assert task_features["logged_spd"].tolist() == pytest.approx([0.8] * 4)
    assert task_features["evidence_at_decision"].tolist() == pytest.approx([0.3] * 4)
    assert features.loc[features["condition"] == "RT", "dt_ms"].isna().all()
    feature_metadata = json.loads(
        sidecar_path(result.outputs[2]).read_text(encoding="utf-8")
    )["metadata"]
    assert feature_metadata["join_key"] == [
        "subject", "condition", "run", "run_trial_index"
    ]
    assert feature_metadata["evidence_at_decision"] == "logged_spd - 0.5"

    metadata = json.loads(sidecar_path(result.outputs[0]).read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 1
    assert metadata["metadata"]["stage"] == "behavior_analysis"
    assert metadata["metadata"]["spd"]["views"] == [
        "all_logged",
        "validated_15row",
    ]
    assert metadata["metadata"]["spd"]["short_log_design_alignment"] == "forbidden"


def test_analyze_behavior_excludes_never_started_trials_but_preserves_input(tmp_path):
    project = ProjectConfig(data_root=tmp_path)
    _write_behavior_inputs(project.bids_root)
    layout = DerivativeLayout(project.bids_root)
    fast_path = layout.behavior(subject="H01", run="2", condition="Fast")
    fast = pd.read_csv(fast_path, sep="\t")
    fast.loc[0, ["nOutcome", "tGO", "nChoiceMade", "tEnterTarget"]] = [
        OUTCOME_NEVER_STARTED,
        0,
        0,
        0,
    ]
    fast.loc[0, "rawRT"] = float("nan")
    fast["isCorrect"] = fast["isCorrect"].astype("boolean")
    fast.loc[0, "isCorrect"] = pd.NA
    save_table(fast_path, fast, metadata={"stage": "behavior_parsing"})

    result = analyze_behavior(project)

    summary = pd.read_csv(result.outputs[0], sep="\t").iloc[0]
    assert summary["n_fast_trials"] == 1
    assert summary["n_never_started_trials"] == 1
    assert summary["mean_fast_dt_ms"] == pytest.approx(750.0)
    features = pd.read_csv(result.outputs[2], sep="\t")
    never_started = features.loc[features["nOutcome"] == OUTCOME_NEVER_STARTED].iloc[0]
    assert not bool(never_started["is_started"])
    assert pd.isna(never_started["started_trial_index"])
    assert not bool(never_started["primary_analysis_eligible"])
    preserved = pd.read_csv(fast_path, sep="\t")
    assert preserved["nOutcome"].tolist() == [OUTCOME_NEVER_STARTED, 0]


def test_analyze_behavior_applies_configured_subject_exclusions(tmp_path):
    project = ProjectConfig(data_root=tmp_path, subject_exclusions=("H01",))
    _write_behavior_inputs(project.bids_root)

    with pytest.raises(ValueError, match="do not contain any trials"):
        analyze_behavior(project)


def test_ingest_behavior_dry_run_declares_outputs(tmp_path):
    project = ProjectConfig(data_root=tmp_path)
    subject_dir = project.behavior_root / "H1"
    subject_dir.mkdir(parents=True)
    (subject_dir / "H1Slow1_180131.tdms").write_text("", encoding="utf-8")

    result = ingest_behavior(project, dry_run=True)

    assert result.stage == "behavior_ingestion"
    assert result.settings["n_runs"] == 1
    assert result.outputs[0].name == "sub-H01_task-tokens_run-1_desc-slow_beh.tsv"
    assert not result.outputs[0].exists()


def test_unified_cli_runs_behavior_analysis(tmp_path, capsys):
    project = ProjectConfig(data_root=tmp_path)
    _write_behavior_inputs(project.bids_root)

    exit_code = main(
        [
            "behavior",
            "analyze",
            "--data-root",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert str(DerivativeLayout(project.bids_root).behavior_summary()) in capsys.readouterr().out


def test_unified_behavior_qc_help_is_available(capsys):
    with pytest.raises(SystemExit) as error:
        main(["behavior", "qc", "--help"])
    assert error.value.code == 0
    capsys.readouterr()


def test_unified_cli_runs_behavior_qc(tmp_path, monkeypatch, capsys):
    summary = pd.DataFrame({"log_rows": [14, 15], "n_trials": [2, 3]})
    details = pd.DataFrame({"subject": ["H01", "H01", "H02"]})
    monkeypatch.setattr(
        "meg_tokens.validation.run_spd_validation",
        lambda *args, **kwargs: (summary, details),
    )

    exit_code = main(
        [
            "behavior",
            "qc",
            "--data-root",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "log_rows" in output
    assert "Validated trials: 3" in output


def test_unified_cli_help_is_side_effect_free(capsys):
    with pytest.raises(SystemExit) as error:
        main(["behavior", "ingest", "--help"])
    assert error.value.code == 0
    assert "--data-root" in capsys.readouterr().out


def test_unified_meg_cli_help_is_available(capsys):
    with pytest.raises(SystemExit) as error:
        main(["meg", "epoch", "--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "--alignment" in output
    assert "--subjects" in output


def test_unified_source_cli_help_is_available(capsys):
    with pytest.raises(SystemExit) as error:
        main(["meg", "source", "--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "--stages" in output
    assert "--volume-labels" in output


@pytest.mark.parametrize("command", ["erp", "power", "spectral"])
def test_unified_primary_feature_help_is_available(command, capsys):
    with pytest.raises(SystemExit) as error:
        main(["features", command, "--help"])
    assert error.value.code == 0
    assert "--subjects" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["hilbert", "pac", "connectivity"])
def test_unified_advanced_feature_help_is_available(command, capsys):
    with pytest.raises(SystemExit) as error:
        main(["features", command, "--help"])
    assert error.value.code == 0
    assert "--conditions" in capsys.readouterr().out


@pytest.mark.parametrize(
    "command",
    [
        "statistics",
        "lateralization",
        "decoding",
        "decomposition",
        "spatial-decoding",
    ],
)
def test_unified_analysis_help_is_available(command, capsys):
    with pytest.raises(SystemExit) as error:
        main(["analyze", command, "--help"])
    assert error.value.code == 0
    expected = "--condition" if command == "lateralization" else "--conditions"
    assert expected in capsys.readouterr().out


@pytest.mark.parametrize(
    "command",
    [
        "behavior",
        "statistics",
        "connectivity",
        "seed-connectivity",
        "decoding-onset",
        "spatial-decoding",
        "pca-trajectory",
        "pca-timecourse",
        "pca-variance",
        "pca-heatmap",
        "pca-loadings",
        "dpca",
    ],
)
def test_unified_report_help_is_available(command, capsys):
    with pytest.raises(SystemExit) as error:
        main(["report", command, "--help"])
    assert error.value.code == 0
    assert "--output-root" in capsys.readouterr().out


def test_unified_validation_help_is_available(capsys):
    with pytest.raises(SystemExit) as error:
        main(["validate", "golden", "--help"])
    assert error.value.code == 0
    assert "--comparison-config" in capsys.readouterr().out
