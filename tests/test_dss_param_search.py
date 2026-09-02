"""The DSS parameter search: its grid, its metrics, and its decision rule.

The properties worth pinning here are methodological, not numerical. Three of
them exist because the same mistakes were made by hand first: ranking on a
median over a cohort whose between-subject spread dwarfs the effect, scoring a
candidate against the events the candidate itself chose, and reading a stage
that correctly removed nothing as a stage that missed something.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import mne
import numpy as np
import pytest

from meg_tokens.meg import preprocessing
from scripts.qc.dss_param_search import grid, metrics, runner, summarize

SUBJECTS = [f"H{index:02d}" for index in range(1, 33)]


# --------------------------------------------------------------------------- grid

def test_every_stage_grid_contains_its_production_configuration():
    """A stage with no reference row is a field of candidates with nothing to beat."""
    for stage in grid.STAGES:
        configs = grid.stage_grid(stage)
        assert sum(config.is_production for config in configs) == 1, stage
        assert len({config.label for config in configs}) == len(configs), stage
    assert len(grid.full_grid()) == sum(len(grid.stage_grid(s)) for s in grid.STAGES)


def test_a_config_must_name_exactly_its_stages_axes():
    with pytest.raises(ValueError, match="missing"):
        grid.Config(stage="qrs", params={"window": (-0.1, 0.2)})
    with pytest.raises(ValueError, match="unexpected"):
        grid.Config(stage="muscle", params={**grid.PRODUCTION["muscle"], "window": (0, 1)})
    with pytest.raises(ValueError, match="unknown stage"):
        grid.Config(stage="cardiac", params={})


def test_bias_and_selection_params_partition_a_config():
    """Every axis goes to the bias or to the selector, and none to both."""
    for stage in grid.STAGES:
        for config in grid.stage_grid(stage):
            assert not set(config.bias_params) & set(config.selection_params)
            assert {**config.bias_params, **config.selection_params} == config.params


def test_selection_params_are_accepted_by_the_production_subtract():
    """The search tunes `dss_subtract` itself, so every axis must be a real keyword."""
    import inspect

    accepted = set(inspect.signature(preprocessing.dss_subtract).parameters)
    for stage in grid.STAGES:
        for config in grid.stage_grid(stage):
            assert set(config.selection_params) <= accepted, config.label


def test_production_row_reproduces_the_cleaner_defaults():
    """If these drift apart the search optimises something nothing runs.

    `dss_subtract`'s defaults are the production values, so a production row is
    the row that passes no selection overrides at all.
    """
    defaults = {
        "n_components": preprocessing.DSS_N_COMPONENTS,
        "selection_threshold": preprocessing.DSS_SELECTION_THRESHOLD,
        "knee_min_ratio": preprocessing.DSS_KNEE_MIN_RATIO,
        "min_select": preprocessing.DSS_MIN_SELECT,
        "max_prop_remove": preprocessing.DSS_MAX_PROP_REMOVE,
    }
    import inspect

    signature = inspect.signature(preprocessing.dss_subtract)
    for name, value in defaults.items():
        assert signature.parameters[name].default == value

    for stage in grid.STAGES:
        production = grid.production_config(stage)
        for name, value in production.selection_params.items():
            assert defaults[name] == value, (stage, name)
    assert grid.PRODUCTION["qrs"]["window"] == preprocessing.QRS_WINDOW
    assert grid.PRODUCTION["blink"]["window"] == preprocessing.BLINK_WINDOW
    assert grid.PRODUCTION["blink"]["scale_multiple"] == preprocessing.OCULAR_SCALE_MULTIPLE
    assert grid.PRODUCTION["muscle"]["freq_band"] == preprocessing.MUSCLE_BAND


def test_folds_split_the_cohort_deterministically_and_not_by_subject_number():
    search = {s for s, fold in grid.folds(SUBJECTS).items() if fold == "search"}
    confirm = {s for s, fold in grid.folds(SUBJECTS).items() if fold == "confirm"}
    assert len(search) == 20 and len(confirm) == 12
    assert not search & confirm
    assert search | confirm == set(SUBJECTS)
    assert grid.folds(SUBJECTS) == grid.folds(list(reversed(SUBJECTS)))
    # Subject order is acquisition order, so the first 20 would be a time split.
    assert search != set(SUBJECTS[:20])
    with pytest.raises(ValueError, match="no confirmation fold"):
        grid.folds(SUBJECTS, n_search=len(SUBJECTS))


# ------------------------------------------------------------------------ metrics

def _raw(data: np.ndarray, sfreq: float = 600.0) -> mne.io.BaseRaw:
    info = mne.create_info(
        [f"MEG{index:03d}" for index in range(data.shape[0])], sfreq, "mag"
    )
    return mne.io.RawArray(data, info, verbose=False)


def test_reference_correlation_recovers_an_injected_artifact_and_ignores_an_unrelated_one():
    rng = np.random.default_rng(0)
    reference = np.sin(np.linspace(0, 200, 4000))
    removed = np.outer(rng.normal(size=32), reference) + 0.01 * rng.normal(size=(32, 4000))

    assert metrics.reference_correlation(removed, reference) > 0.99
    assert metrics.reference_correlation(removed, rng.normal(size=4000)) < 0.2
    # DSS fixes neither a component's sign nor a topography's.
    assert metrics.reference_correlation(-removed, reference) == pytest.approx(
        metrics.reference_correlation(removed, reference)
    )
    assert np.isnan(metrics.reference_correlation(np.zeros((4, 100)), np.zeros(100)))


def test_dominant_time_course_matches_the_svd_it_replaces():
    """The covariance eigenvector is the left singular vector, without the 800 MB factor."""
    rng = np.random.default_rng(1)
    removed = np.outer(rng.normal(size=16), np.sin(np.linspace(0, 60, 900)))
    removed += 0.05 * rng.normal(size=removed.shape)

    centred = removed - removed.mean(axis=1, keepdims=True)
    left, _, _ = np.linalg.svd(centred, full_matrices=False)
    expected = left[:, 0] @ removed

    actual = metrics.dominant_time_course(removed)
    assert abs(np.corrcoef(actual, expected)[0, 1]) > 0.9999


def test_band_power_db_reads_the_band_it_was_asked_for():
    times = np.arange(0, 20, 1 / 600.0)
    data = np.tile(np.sin(2 * np.pi * 11.0 * times), (4, 1))
    raw = _raw(data * 1e-12)
    picks = np.arange(4)

    inside = metrics.band_power_db(raw, picks, metrics.ALPHA_BAND)
    outside = metrics.band_power_db(raw, picks, (40.0, 60.0))
    assert inside > outside + 20


def test_evoked_rms_reduction_spans_unchanged_to_fully_removed():
    rng = np.random.default_rng(2)
    sfreq = 600.0
    samples = np.arange(1200, 9000, 1200)
    data = 0.01 * rng.normal(size=(8, 12000))
    pulse = np.hanning(240)
    for sample in samples:
        data[:, sample : sample + 240] += pulse
    raw = _raw(data)
    picks = np.arange(8)

    before = metrics.evoked_rms(raw, picks, samples)
    unchanged = metrics.evoked_rms(raw, picks, samples)
    assert 100.0 * (1.0 - unchanged / before) == pytest.approx(0.0)

    cleaned = _raw(0.01 * rng.normal(size=(8, 12000)))
    reduction = 100.0 * (1.0 - metrics.evoked_rms(cleaned, picks, samples) / before)
    assert reduction > 90.0
    assert np.isnan(metrics.evoked_rms(raw, picks, np.array([], dtype=int)))


def test_artifact_presence_reports_how_much_muscle_a_recording_carries():
    """A stage that correctly removes nothing must be distinguishable from a miss."""
    rng = np.random.default_rng(3)
    raw = _raw(1e-12 * rng.normal(size=(6, 12000)))
    raw.set_annotations(mne.Annotations([2.0, 8.0], [1.0, 0.5], ["BAD_muscle", "BAD_muscle"]))
    events = {
        "ecg": np.arange(0, 12000, 600),
        "blink": np.arange(0, 12000, 2000),
        "saccade": np.arange(0, 12000, 4000),
    }

    presence = metrics.artifact_presence(raw, events, np.arange(6))
    assert presence["muscle_annotated_seconds"] == pytest.approx(1.5)
    assert presence["muscle_annotated_fraction"] == pytest.approx(1.5 / presence["duration_seconds"])
    assert presence["ecg_per_minute"] == pytest.approx(60.0, rel=0.05)


# ------------------------------------------------------------------------- runner

def test_a_bias_averages_only_events_with_a_complete_window():
    sfreq = 600.0
    samples = np.array([10, 600, 6000, 11990])
    trimmed = runner._trim_to_complete_windows(samples, (-0.2, 0.5), sfreq, 12000)
    assert trimmed.tolist() == [600, 6000]


def test_too_few_usable_events_is_an_error_not_an_empty_average():
    """Nine blinks would produce a template, and a meaningless one."""
    events = {"blink": np.arange(1000, 1000 + 5 * 600, 600), "ecg": np.array([])}
    config = grid.production_config("blink")
    with pytest.raises(ValueError, match="too few"):
        runner._build_bias(config, events, 600.0, 12000)


def test_the_bias_uses_the_candidates_event_set():
    """`scale_multiple` is an axis only if the events it changes reach the bias."""
    events = {"blink": np.arange(1000, 11000, 300)}
    config = grid.production_config("blink")
    _, many = runner._build_bias(config, events, 600.0, 12000)
    _, few = runner._build_bias(config, {"blink": events["blink"][::3]}, 600.0, 12000)
    assert many > few >= 10


LAUNCHER = Path(__file__).parents[1] / "scripts" / "dss_param_search_array.sh"


def test_the_launcher_runs_a_command_the_runner_accepts(tmp_path):
    """Run the launcher rather than read it.

    `ssm_fit_array.sh` passed a removed flag for months because nothing ran it.
    Executing the launcher against a recording interpreter checks the whole
    chain -- argument order, quoting, and the argv the runner finally sees --
    where the previous regex over the file only checked the last of those.

    It does not catch a missing interpreter: `./.venv/bin/python` resolves here
    and did not on the cluster, which cost sixteen array tasks. That is what
    the launcher now fails fast on, and what the next test covers.
    """
    recorder = tmp_path / "python"
    recorder.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$0.argv"\n')
    recorder.chmod(0o755)

    result = subprocess.run(
        ["bash", str(LAUNCHER), str(tmp_path / "bids"), str(tmp_path / "out"), str(recorder)],
        cwd=LAUNCHER.parents[1], capture_output=True, text=True,
        env={**os.environ, "SLURM_ARRAY_TASK_ID": "7", "SLURM_JOB_ID": "1"},
    )
    assert result.returncode == 0, result.stderr

    argv = (tmp_path / "python.argv").read_text().split("\n")[:-1]
    assert argv[:2] == ["-m", "scripts.qc.dss_param_search.runner"]
    parsed = runner.build_parser().parse_args(argv[2:])
    assert parsed.index == 7
    assert parsed.bids_root == tmp_path / "bids"


def test_the_launcher_refuses_an_interpreter_that_is_not_there():
    """Thirty-two tasks failing identically is a worse error message than one."""
    result = subprocess.run(
        ["bash", str(LAUNCHER), "/bids", "/out", "/nonexistent/python"],
        cwd=LAUNCHER.parents[1], capture_output=True, text=True,
        env={**os.environ, "SLURM_ARRAY_TASK_ID": "0"},
    )
    assert result.returncode == 2
    assert "/nonexistent/python" in result.stderr


# ---------------------------------------------------------------------- decision

def _record(subject: str, fold: str, rows: list[dict], muscle_fraction: float = 0.02) -> dict:
    return {
        "subject": subject,
        "fold": fold,
        "presence": {"muscle_annotated_fraction": muscle_fraction},
        "rows": rows,
    }


def _row(stage: str, label: str, **kwargs) -> dict:
    row = {
        "stage": stage, "label": label, "is_production": False,
        "n_selected": 8, "alpha_delta_db": -0.1,
        "reference_r": 0.5, "muscle_reduction_db": 0.5, "removal_percent": 50.0,
    }
    row.update(kwargs)
    return row


def test_feasibility_rejects_collateral_over_budget():
    budget = {"alpha_budget": 0.5, "dimension_budget": 20}
    assert summarize.feasible(_row("qrs", "a"), **budget)
    assert not summarize.feasible(_row("qrs", "a", alpha_delta_db=-0.9), **budget)
    assert not summarize.feasible(_row("qrs", "a", n_selected=40), **budget)
    assert not summarize.feasible({"stage": "qrs", "failed": "boom"}, **budget)
    # The budget is a loss budget: a configuration that costs nothing passes.
    assert summarize.feasible(_row("qrs", "a", alpha_delta_db=+0.2), **budget)


def test_ranking_is_by_wins_and_a_better_score_does_not_buy_past_the_budget():
    """The shape of the real QRS result: widening raises the score everywhere and
    the alpha cost with it.

    `aggressive` has the best objective on every subject and breaches the
    collateral budget on six of ten, so it can only win on the four where it is
    affordable. Ranking on the objective's median -- which is what was done by
    hand, twice -- puts it first; ranking on wins puts production first.
    """
    production = grid.production_config("qrs").label
    aggressive = "qrs|aggressive"
    records = []
    for index in range(10):
        over_budget = index >= 4
        records.append(_record(f"H{index:02d}", "search", [
            _row("qrs", production, is_production=True, reference_r=0.50, alpha_delta_db=-0.10),
            _row("qrs", aggressive, reference_r=0.90,
                 alpha_delta_db=-0.90 if over_budget else -0.20),
        ]))

    table = summarize.stage_table(
        records, "qrs", fold="search",
        alpha_budget=0.5, dimension_budget=20, muscle_floor=0.005,
    )
    by_label = {row["label"]: row for row in table}
    assert by_label[aggressive]["objective_median"] > by_label[production]["objective_median"]
    assert by_label[aggressive]["n_feasible"] == 4
    assert by_label[aggressive]["wins"] == 4
    assert by_label[production]["wins"] == 6
    assert table[0]["label"] == production


def test_the_median_breaks_a_tie_in_wins():
    """Wins first, then the objective -- a config that never wins never sorts first."""
    production = grid.production_config("qrs").label
    better, worse = "qrs|better", "qrs|worse"
    records = [
        _record("H01", "search", [
            _row("qrs", production, is_production=True, reference_r=0.90),
            _row("qrs", better, reference_r=0.50),
            _row("qrs", worse, reference_r=0.40),
        ]),
    ]
    table = summarize.stage_table(
        records, "qrs", fold="search",
        alpha_budget=0.5, dimension_budget=20, muscle_floor=0.005,
    )
    assert [row["label"] for row in table] == [production, better, worse]


def test_the_spread_is_reported_so_a_bimodal_configuration_is_visible():
    """A median over a bimodal cohort describes no subject in it.

    ASR was read as weak from a median of 0.91 over `0.00, 0.91, 0.94`: it does
    nothing on a clean recording, correctly, and a great deal on a dirty one.
    """
    label = grid.production_config("muscle").label
    records = [
        _record(f"H{index:02d}", "search",
                [_row("muscle", label, is_production=True, muscle_reduction_db=value)])
        for index, value in enumerate([0.00, 0.91, 0.94])
    ]
    row = summarize.stage_table(
        records, "muscle", fold="search",
        alpha_budget=0.5, dimension_budget=20, muscle_floor=0.005,
    )[0]
    assert row["objective_min"] == pytest.approx(0.00)
    assert row["objective_max"] == pytest.approx(0.94)
    assert row["objective_median"] == pytest.approx(0.91)


def test_a_label_absent_from_the_current_grid_still_reaches_the_table():
    """Editing `grid.py` must not silently shorten an older run's table."""
    records = [_record("H01", "search", [_row("qrs", "qrs|retired-axis=1", reference_r=0.7)])]
    table = summarize.stage_table(
        records, "qrs", fold="search",
        alpha_budget=0.5, dimension_budget=20, muscle_floor=0.005,
    )
    assert [row["label"] for row in table] == ["qrs|retired-axis=1"]
    assert table[0]["wins"] == 1


def test_the_muscle_stage_ignores_recordings_that_carry_no_muscle():
    """Zero components on a clean recording is the correct answer, not a miss."""
    production = grid.production_config("muscle").label
    records = [
        _record("H01", "search", [
            _row("muscle", production, is_production=True, muscle_reduction_db=0.9),
        ], muscle_fraction=0.02),
        _record("H11", "search", [
            _row("muscle", production, is_production=True, muscle_reduction_db=0.0, n_selected=0),
        ], muscle_fraction=0.0),
    ]

    table = summarize.stage_table(
        records, "muscle", fold="search",
        alpha_budget=0.5, dimension_budget=20, muscle_floor=0.005,
    )
    row = next(row for row in table if row["label"] == production)
    assert row["n_subjects"] == 1
    assert row["objective_median"] == pytest.approx(0.9)


def test_the_winner_is_confirmed_only_on_the_held_out_fold():
    production = grid.production_config("qrs").label
    candidate = "qrs|candidate"
    records = [
        # Search fold: the candidate looks enormous here and is not reported.
        _record("H01", "search", [
            _row("qrs", production, is_production=True, reference_r=0.10),
            _row("qrs", candidate, reference_r=0.99),
        ]),
        _record("H02", "confirm", [
            _row("qrs", production, is_production=True, reference_r=0.50),
            _row("qrs", candidate, reference_r=0.60),
        ]),
        _record("H03", "confirm", [
            _row("qrs", production, is_production=True, reference_r=0.50),
            _row("qrs", candidate, reference_r=0.40),
        ]),
    ]

    result = summarize.confirm(
        records, "qrs", candidate,
        alpha_budget=0.5, dimension_budget=20, muscle_floor=0.005,
    )
    assert result["of"] == 2 and result["beats_production_on"] == 1
    assert result["delta_median"] == pytest.approx(0.0)
    assert (result["delta_min"], result["delta_max"]) == pytest.approx((-0.1, 0.1))


def test_summarize_writes_a_table_and_a_summary(tmp_path):
    production = {stage: grid.production_config(stage).label for stage in grid.STAGES}
    for index, fold in enumerate(["search"] * 4 + ["confirm"] * 2):
        rows = [
            _row(stage, label, is_production=True)
            for stage, label in production.items()
        ]
        (tmp_path / f"sub-H{index:02d}_dsssearch.json").write_text(
            json.dumps(_record(f"H{index:02d}", fold, rows))
        )

    out = tmp_path / "summary"
    assert summarize.main([
        "--input-root", str(tmp_path), "--output-root", str(out),
    ]) == 0
    table = (out / "dss_param_search_table.tsv").read_text().splitlines()
    assert table[0].split("\t")[:3] == ["stage", "label", "is_production"]
    # One row per configuration the run produced -- here only the production
    # rows -- not per configuration the current grid defines.
    assert len(table) == 1 + len(grid.STAGES)
    summary = json.loads((out / "dss_param_search_summary.json").read_text())
    assert summary["budget"]["alpha_budget"] == 0.5
    assert set(summary["folds"].values()) == {"search", "confirm"}


def test_summarize_refuses_an_empty_input_root(tmp_path):
    with pytest.raises(SystemExit, match="no sub-"):
        summarize.load(tmp_path)
