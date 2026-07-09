import json

import pandas as pd
import pytest

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
            "nInitialTime": [0] * n_trials,
            "nChoiceMade": choices,
            "nCorrectChoice": correct,
            "tGO": [1000] * n_trials,
            "tEnterTarget": enter_times,
            "tTrialEnd": [value + 200 for value in enter_times],
            "sTokenDirs": ["121"] * n_trials,
            "tTime": ["[1100, 1300]"] * n_trials,
            "nProb": ["[0.6, 0.8]"] * n_trials,
            "rawRT": [value - 1000 for value in enter_times],
            "isCorrect": [
                choice == expected
                for choice, expected in zip(choices, correct)
            ],
        }
    )


def _write_behavior_inputs(root):
    layout = DerivativeLayout(root)
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
    _write_behavior_inputs(tmp_path)
    project = ProjectConfig(bids_root=tmp_path)

    result = analyze_behavior(project)

    assert result.stage == "behavior_analysis"
    assert result.outputs == (DerivativeLayout(tmp_path).behavior_summary(),)
    summary = pd.read_csv(result.outputs[0], sep="\t")
    assert summary["subject"].tolist() == ["H01"]
    assert summary["motor_baseline_ms"].iloc[0] == pytest.approx(350.0)
    assert summary["mean_fast_dt_ms"].iloc[0] == pytest.approx(650.0)
    assert summary["mean_slow_dt_ms"].iloc[0] == pytest.approx(1150.0)
    assert summary["percent_correct"].iloc[0] == pytest.approx(75.0)

    metadata = json.loads(sidecar_path(result.outputs[0]).read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 1
    assert metadata["metadata"]["stage"] == "behavior_analysis"


def test_ingest_behavior_dry_run_declares_outputs(tmp_path):
    input_root = tmp_path / "tdms"
    subject_dir = input_root / "H1"
    subject_dir.mkdir(parents=True)
    (subject_dir / "H1Slow1_180131.tdms").write_text("", encoding="utf-8")
    project = ProjectConfig(bids_root=tmp_path / "bids", behavior_root=input_root)

    result = ingest_behavior(project, dry_run=True)

    assert result.stage == "behavior_ingestion"
    assert result.settings["n_runs"] == 1
    assert result.outputs[0].name == "sub-H01_task-tokens_run-1_desc-slow_beh.tsv"
    assert not result.outputs[0].exists()


def test_unified_cli_runs_behavior_analysis(tmp_path, capsys):
    _write_behavior_inputs(tmp_path)

    exit_code = main(
        [
            "behavior",
            "analyze",
            "--bids-root",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert str(DerivativeLayout(tmp_path).behavior_summary()) in capsys.readouterr().out


def test_unified_cli_help_is_side_effect_free(capsys):
    with pytest.raises(SystemExit) as error:
        main(["behavior", "ingest", "--help"])
    assert error.value.code == 0
    assert "--input-dir" in capsys.readouterr().out


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
