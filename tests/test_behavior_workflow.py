import json

import pandas as pd
import pytest

from meg_tokens.behavior.schema import OUTCOME_NEVER_STARTED
from meg_tokens.cli import main
from meg_tokens.core import ProjectConfig
from meg_tokens.io import DerivativeLayout, save_table, sidecar_path
from meg_tokens.workflows import analyze_behavior, ingest_behavior
from meg_tokens.workflows.behavior_characterization import (
    analyze_behavior_characterization,
)
from meg_tokens.workflows.sequential_sampling import fit_subject_sequential_sampling

from tests.behavior.factories import stage_behavior, stage_raw_behavior


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
    # The surviving Fast trial is a right-hand response (rawRT 1100), and the
    # RT run gives left 300 / right 400, so it loses the right hand's own
    # latency rather than the 350 ms pooled over both hands.
    assert summary["mean_fast_dt_ms"] == pytest.approx(700.0)
    features = pd.read_csv(result.outputs[2], sep="\t")
    never_started = features.loc[features["nOutcome"] == OUTCOME_NEVER_STARTED].iloc[0]
    assert not bool(never_started["is_started"])
    assert pd.isna(never_started["started_trial_index"])
    assert not bool(never_started["primary_analysis_eligible"])
    preserved = pd.read_csv(fast_path, sep="\t")
    assert preserved["nOutcome"].tolist() == [OUTCOME_NEVER_STARTED, 0]


def test_ingest_behavior_dry_run_declares_outputs(tmp_path):
    project = ProjectConfig(data_root=tmp_path)
    stage_raw_behavior(project.bids_root, subject="H01", condition="Slow", run="1")

    result = ingest_behavior(project, dry_run=True)

    assert result.stage == "behavior_ingestion"
    assert result.settings["n_runs"] == 1
    assert result.settings["subjects"] == ["H01"]
    assert result.outputs[0].name == "sub-H01_task-tokens_run-1_desc-slow_beh.tsv"
    assert not result.outputs[0].exists()


def test_a_stage_without_its_prerequisite_names_what_is_missing(tmp_path):
    project = ProjectConfig(data_root=tmp_path)
    project.bids_root.mkdir(parents=True)

    # Stage 1 reads the raw BIDS layer, so an unstaged dataset is a missing
    # prerequisite rather than a dataset with no behavior.
    with pytest.raises(FileNotFoundError, match="No staged subject behavior"):
        ingest_behavior(project, dry_run=True)

    with pytest.raises(FileNotFoundError, match="behavior analyze"):
        analyze_behavior_characterization(project)

    excluded = ProjectConfig(data_root=tmp_path, subject_exclusions=("H01",))
    _write_behavior_inputs(excluded.bids_root)

    with pytest.raises(ValueError, match="do not contain any trials"):
        analyze_behavior(excluded)


def _write_tokens_toml(tmp_path, data_root):
    config_path = tmp_path / "tokens.toml"
    config_path.write_text(f'[project]\ndata_root = "{data_root}"\n', encoding="utf-8")
    return config_path


def test_the_unified_cli_runs_the_behavior_stages(tmp_path, monkeypatch, capsys):
    project = ProjectConfig(data_root=tmp_path)
    _write_behavior_inputs(project.bids_root)
    config_path = _write_tokens_toml(tmp_path, tmp_path)

    exit_code = main(["--config", str(config_path), "behavior", "analyze"])

    assert exit_code == 0
    assert str(DerivativeLayout(project.bids_root).behavior_summary()) in (
        capsys.readouterr().out
    )

    summary = pd.DataFrame({"log_rows": [14, 15], "n_trials": [2, 3]})
    details = pd.DataFrame({"subject": ["H01", "H01", "H02"]})
    monkeypatch.setattr(
        "meg_tokens.validation.run_spd_validation",
        lambda *args, **kwargs: (summary, details),
    )

    exit_code = main(["--config", str(config_path), "behavior", "qc"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "log_rows" in output
    assert "Validated trials: 3" in output


# --- Stage 2b: behavioral characterization analyses ---------------------------


def test_analyze_behavior_characterization_writes_every_analysis_table(tmp_path):
    project = ProjectConfig(data_root=tmp_path)
    stage_behavior(project.bids_root)
    analyze_behavior(project)

    result = analyze_behavior_characterization(project)

    assert result.stage == "behavior_characterization_analysis"
    assert result.settings["n_subjects"] == 3
    layout = DerivativeLayout(project.bids_root)
    # The sequential-sampling derivatives are written by 'behavior ssm-fit';
    # this step only pools them, and none has been fitted here.
    for name in (
        "spdcumulative",
        "dtdistribution",
        "conditionclass",
        "choiceside",
        "timeontask",
        "conditionorder",
        "lapses",
        "extremedt",
        "criteriondecline",
        "reversecorrelation",
        "conditionalaccuracy",
        "continuousevidence",
        "posterror",
        "choicehistory",
        "individualprofile",
        "individualcorrelations",
        "urgency",
        "speciescomparison",
    ):
        assert layout.behavior_analysis(name) in result.outputs
    for path in result.outputs:
        assert path.is_file()
    assert result.settings["sequential_sampling_subjects_missing"] == [
        "H01",
        "H02",
        "H03",
    ]

    cells = pd.read_csv(layout.behavior_analysis("conditionclass"), sep="\t")
    assert set(cells["subject"]) == {"H01", "H02", "H03"}
    criterion = pd.read_csv(layout.behavior_analysis("criteriondecline"), sep="\t")
    assert set(criterion["response"]) == {
        "chosen_sum_log_lr_first_order_at_decision"
    }
    assert set(criterion["predictor"]) == {"dt_ms"}


def test_sequential_sampling_fits_are_written_one_table_per_subject(tmp_path):
    """The fixture's cells are far below the trials an accumulator needs, so the
    fits are reported as unfitted -- the tables, columns, and sidecars are not."""
    project = ProjectConfig(data_root=tmp_path)
    stage_behavior(project.bids_root)
    analyze_behavior(project)

    result = fit_subject_sequential_sampling(project)

    layout = DerivativeLayout(project.bids_root)
    for subject in ("H01", "H02", "H03"):
        path = layout.behavior_subject_analysis(subject, "ssmcomparison")
        assert path in result.outputs
        fits = pd.read_csv(path, sep="\t")
        assert set(fits["subject"]) == {subject}
        assert set(fits["model"]) == {"ddm", "urgency"}
        assert not fits["converged"].any()
        assert {"log_likelihood", "aic", "bic", "delta_bic"} <= set(fits.columns)


def test_a_subject_can_be_fitted_on_its_own(tmp_path):
    """One array-job task writes its own subject's table and nothing else, and
    characterization then pools only the subjects already fitted."""
    project = ProjectConfig(data_root=tmp_path)
    stage_behavior(project.bids_root)
    analyze_behavior(project)

    result = fit_subject_sequential_sampling(project, subjects=["H02"])

    layout = DerivativeLayout(project.bids_root)
    assert result.outputs == (
        layout.behavior_subject_analysis("H02", "ssmcomparison"),
    )
    assert not layout.behavior_subject_analysis("H01", "ssmcomparison").is_file()

    assert analyze_behavior_characterization(project).settings[
        "sequential_sampling_subjects_missing"
    ] == ["H01", "H03"]


def test_characterization_pools_the_fits_without_republishing_them(tmp_path):
    """`ssm-aggregate` is the sole writer of every `ssm*` group derivative.

    Both pipelines once wrote six of these filenames, so whichever ran last
    silently won -- and they disagreed in substance, not just in timing:
    characterization pooled ungated per-subject fits and omitted the
    `ssm_mechanistic_model_comparison` block the Act 3 panel selects on. This
    test fails if a second writer is ever reintroduced, which is what keeps the
    run order from mattering again.
    """
    project = ProjectConfig(data_root=tmp_path)
    stage_behavior(project.bids_root)
    analyze_behavior(project)
    fit_subject_sequential_sampling(project)

    result = analyze_behavior_characterization(project)

    assert result.settings["sequential_sampling_subjects_missing"] == []
    # The fits are consumed, not republished: they reach `speciescomparison`,
    # which is written, while no `ssm*` group table is.
    assert "speciescomparison" in result.settings["analyses"]
    written = {path.name for path in result.outputs}
    assert not [name for name in written if "desc-ssm" in name], written
    assert not [name for name in result.settings["analyses"] if name.startswith("ssm")]

    # And the group tables genuinely are absent rather than merely unlisted.
    layout = DerivativeLayout(project.bids_root)
    for name in (
        "ssmcomparison",
        "ssmcomparisonstats",
        "ssmtimecourse",
        "ssmtrialpredictions",
    ):
        assert not layout.behavior_analysis(name).is_file(), name


def test_behavior_analysis_paths_are_named_from_the_analysis(tmp_path):
    layout = DerivativeLayout(tmp_path)
    assert layout.behavior_analysis("urgency").as_posix().endswith(
        "sub-group/beh/sub-group_task-tokens_desc-urgency_beh.tsv"
    )
    with pytest.raises(ValueError, match="alphanumeric"):
        layout.behavior_analysis("post_error")
