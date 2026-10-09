"""The DSS parameter search: its grid, its metrics, and its decision rule.

The properties worth pinning here are methodological, not numerical. Three of
them exist because the same mistakes were made by hand first: ranking on a
median over a cohort whose between-subject spread dwarfs the effect, scoring a
candidate against the events the candidate itself chose, and reading a stage
that correctly removed nothing as a stage that missed something.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from meg_tokens.meg import preprocessing
from scripts.qc.dss_param_search import grid, metrics, runner, streams, summarize


# --------------------------------------------------------------------------- grid

def test_every_stage_grid_contains_its_production_configuration():
    """A stage with no reference row is a field of candidates with nothing to beat."""
    for stage in grid.STAGES:
        configs = grid.stage_grid(stage)
        assert sum(config.is_production for config in configs) == 1, stage
        assert len({config.label for config in configs}) == len(configs), stage
    assert len(grid.full_grid()) == sum(len(grid.stage_grid(s)) for s in grid.STAGES)


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


# ------------------------------------------------------------------------ metrics


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


# ------------------------------------------------------------------------- runner


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


def test_feasibility_is_measured_against_production_on_the_same_subject():
    """An absolute alpha bar is the B5 asymmetry, and a first pass here repeated it.

    0.5 dB read as a reasonable ceiling; production itself exceeds it on 15 of
    32 subjects for the cardiac stage, so the table ranked configurations by
    which subjects fell under a round number. The bar is production's own cost
    on that recording.
    """
    budget = {"alpha_slack": 0.1, "dimension_budget": 20}
    cheap = _row("qrs", "prod", alpha_delta_db=-0.10)
    dear = _row("qrs", "prod", alpha_delta_db=-1.40)

    # 0.30 dB is affordable beside a 1.40 dB incumbent and not beside a 0.10 one.
    candidate = _row("qrs", "a", alpha_delta_db=-0.30)
    assert summarize.feasible(candidate, dear, **budget)
    assert not summarize.feasible(candidate, cheap, **budget)

    # Matching production, or beating it, always passes.
    assert summarize.feasible(_row("qrs", "a", alpha_delta_db=-0.10), cheap, **budget)
    assert summarize.feasible(_row("qrs", "a", alpha_delta_db=+0.20), cheap, **budget)
    # Slack is slack: a little worse is allowed, a lot is not.
    assert summarize.feasible(_row("qrs", "a", alpha_delta_db=-0.19), cheap, **budget)
    assert not summarize.feasible(_row("qrs", "a", alpha_delta_db=-0.21), cheap, **budget)

    assert not summarize.feasible(_row("qrs", "a", n_selected=40), cheap, **budget)
    assert not summarize.feasible({"stage": "qrs", "failed": "boom"}, cheap, **budget)
    # With no incumbent to compare against, only the dimension ceiling applies.
    assert summarize.feasible(_row("qrs", "a", alpha_delta_db=-9.0), None, **budget)


def test_a_subject_with_many_runs_counts_once():
    """Recording counts multiply the apparent evidence by ten.

    `loud` is best on all ten of one subject's runs; `quiet` is best on the
    single run of each of three other subjects. Counted by recording that is
    10 against 3; counted by subject it is 1 against 3, which is the number of
    independent observations there actually are.
    """
    production = grid.production_config("qrs").label
    loud, quiet = "qrs|loud", "qrs|quiet"

    def rows(loud_r, quiet_r):
        return [
            _row("qrs", production, is_production=True, reference_r=0.10),
            _row("qrs", loud, reference_r=loud_r),
            _row("qrs", quiet, reference_r=quiet_r),
        ]

    records = [
        {**_record("H01", "search", rows(0.90, 0.20)), "run": str(index)}
        for index in range(1, 11)
    ]
    records += [
        _record(f"H{index:02d}", "search", rows(0.20, 0.90)) for index in (2, 3, 4)
    ]

    table = summarize.stage_table(
        records, "qrs", fold="search",
        alpha_slack=0.1, dimension_budget=20, muscle_floor=0.005,
    )
    by_label = {row["label"]: row for row in table}
    assert by_label[loud]["wins"] == 10 and by_label[loud]["subject_wins"] == 1
    assert by_label[quiet]["wins"] == 3 and by_label[quiet]["subject_wins"] == 3
    assert by_label[loud]["n_recordings"] == 13
    assert by_label[loud]["n_subjects"] == 4
    assert table[0]["label"] == quiet


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
        alpha_slack=0.1, dimension_budget=20, muscle_floor=0.005,
    )
    assert result["winner_is_production"] is False
    assert result["of"] == 2 and result["beats_production_on"] == 1
    assert result["delta_median"] == pytest.approx(0.0)
    assert (result["delta_min"], result["delta_max"]) == pytest.approx((-0.1, 0.1))


# ------------------------------------------------------------------- cleaners


def test_the_top_pattern_search_finds_a_subdominant_artifact():
    """A combined cleaner removes cardiac and ocular at once.

    Its dominant pattern tracks whichever is larger, so the other scores near
    zero however well it was removed. `reference_correlation_top` asks the fair
    question of both; the single-pattern form used by the per-stage grid cannot.
    """
    rng = np.random.default_rng(0)
    samples = 4000
    loud = np.sin(np.linspace(0, 200, samples))
    quiet = np.sin(np.linspace(0, 37, samples))
    removed = (
        5.0 * np.outer(rng.normal(size=64), loud) + np.outer(rng.normal(size=64), quiet)
    )

    assert metrics.reference_correlation_top(removed, loud, k=1) > 0.99
    assert metrics.reference_correlation_top(removed, quiet, k=1) < 0.2
    assert metrics.reference_correlation_top(removed, quiet, k=3) > 0.99


def test_the_launcher_passes_the_mode_it_was_given(tmp_path):
    """A run that silently produced the grid instead of the comparison would
    look exactly like a run that worked, so the mode is an argument."""
    recorder = tmp_path / "python"
    recorder.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$0.argv"\n')
    recorder.chmod(0o755)

    result = subprocess.run(
        ["bash", str(LAUNCHER), str(tmp_path / "bids"), str(tmp_path / "out"),
         str(recorder), "streams"],
        cwd=LAUNCHER.parents[1], capture_output=True, text=True,
        env={**os.environ, "SLURM_ARRAY_TASK_ID": "3", "SLURM_JOB_ID": "1"},
    )
    assert result.returncode == 0, result.stderr
    argv = (tmp_path / "python.argv").read_text().split("\n")[:-1]
    assert runner.build_parser().parse_args(argv[2:]).mode == "streams"

    # An unknown mode fails at submit, not 319 tasks later.
    bad = subprocess.run(
        ["bash", str(LAUNCHER), "/bids", "/out", str(recorder), "grd"],
        cwd=LAUNCHER.parents[1], capture_output=True, text=True,
        env={**os.environ, "SLURM_ARRAY_TASK_ID": "0"},
    )
    assert bad.returncode == 2 and "grd" in bad.stderr
