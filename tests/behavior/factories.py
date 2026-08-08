"""Table factories shared by the behavioral test modules.

The behavioral pipeline has two canonical table shapes: the Stage 1 run table
written by TDMS ingestion, and the analysis-ready trial-feature table written by
``behavior analyze``. Every behavioral test builds one of them, so they are
constructed here once with valid defaults and overridden per test.

Values here are written out literally rather than computed with the functions
under test, so that a test asserting a scientific quantity is never comparing a
module against itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from meg_tokens.behavior.schema import RAW_TRIAL_COLUMNS
from meg_tokens.io import DerivativeLayout, save_table


# A 15-jump design with a 8-to-7 majority for target 1.
TOKEN_DIRECTIONS = "121121122112212"

# One logged success-probability entry per jump, referenced to the chosen
# target. Written literally: these are inputs, not pipeline outputs.
LOGGED_PROBABILITIES = [
    0.60, 0.55, 0.62, 0.70, 0.65, 0.72, 0.78, 0.72,
    0.68, 0.75, 0.82, 0.78, 0.72, 0.80, 1.00,
]

# Token-log timestamps in the acquisition clock, 200 ms apart after the go cue.
TOKEN_TIMES = [1200 + 200 * jump for jump in range(15)]

GO_CUE_MS = 1000


def stage1_run(
    *,
    subject: str = "H01",
    condition: str = "Fast",
    run: int = 1,
    raw_rts: list[float | None],
    choices: list[int] | None = None,
    correct_choices: list[int] | None = None,
    trial_classes: list[int] | None = None,
    outcomes: list[int] | None = None,
    first_trial_index: int = 1,
    initial_time_ms: int = 0,
    token_directions: str = TOKEN_DIRECTIONS,
    probabilities: list[float] | None = None,
    **columns,
) -> pd.DataFrame:
    """Build one valid Stage 1 behavioral run table.

    ``raw_rts`` drives the trial count and the event timeline: a trial with a
    null RT is a no-response trial, and ``tGO``/``tExitCenter``/``tEnterTarget``
    /``tTrialEnd`` are laid out in the order the Stage 1 schema requires.
    """
    n_trials = len(raw_rts)
    choices = [1] * n_trials if choices is None else choices
    correct_choices = [1] * n_trials if correct_choices is None else correct_choices
    trial_classes = [1] * n_trials if trial_classes is None else trial_classes
    outcomes = [0] * n_trials if outcomes is None else outcomes
    probabilities = LOGGED_PROBABILITIES if probabilities is None else probabilities
    log_rows = len(probabilities)

    enter_target = [
        GO_CUE_MS + float(rt) if rt is not None else 0 for rt in raw_rts
    ]
    class_labels = {0: "x", 1: "e", 2: "a", 3: "m"}
    table = pd.DataFrame({
        "subject": [subject] * n_trials,
        "condition": [condition] * n_trials,
        "run": [run] * n_trials,
        "source_file": [f"{subject}{condition}{run}_180131.tdms"] * n_trials,
        "nTrialIndex": list(range(first_trial_index, first_trial_index + n_trials)),
        "sTrialClass": trial_classes,
        "sTrialClassRaw": [class_labels[code] for code in trial_classes],
        "trial_class_source": ["design"] * n_trials,
        "trial_class_rule": ["recorded_label"] * n_trials,
        "sp_design_correct": [list(probabilities)] * n_trials,
        "nInitialTime": [
            initial_time_ms + 5000 * index for index in range(n_trials)
        ],
        "nChoiceMade": [
            choice if rt is not None else 0
            for choice, rt in zip(choices, raw_rts)
        ],
        "nCorrectChoice": correct_choices,
        "tGO": [GO_CUE_MS] * n_trials,
        "tEnterCenter": [0] * n_trials,
        "tExitCenter": enter_target,
        "tEnterTarget": enter_target,
        "tTrialEnd": [value + 200 for value in enter_target],
        "sTokenDirs": [token_directions] * n_trials,
        "nTokenNum": [list(range(1, log_rows + 1))] * n_trials,
        "nTokenDir": [
            [int(character) for character in token_directions[:log_rows]]
        ] * n_trials,
        "tTime": [list(TOKEN_TIMES[:log_rows])] * n_trials,
        "nProb": [list(probabilities)] * n_trials,
        "token_log_rows": [log_rows] * n_trials,
        "token_log_short": [log_rows == 14] * n_trials,
        "nOutcome": outcomes,
        "rawRT": [float(rt) if rt is not None else None for rt in raw_rts],
        "isCorrect": [
            (choice == expected) if rt is not None else None
            for choice, expected, rt in zip(choices, correct_choices, raw_rts)
        ],
    })
    for column, values in columns.items():
        table[column] = values
    return table


def raw_run(*, raw_rts=(400.0, None), **kwargs) -> pd.DataFrame:
    """Build one valid raw behavioral run: ``RAW_TRIAL_COLUMNS`` and no more.

    This is what Stage 0 stages under ``BIDS/sub-*/beh/`` -- the transcribed
    fields only. It is the Stage 1 table with the derivative-stage additions
    removed (trial class and its provenance, run identity, ``rawRT``,
    ``isCorrect``), so the two shapes cannot drift apart here.
    """
    table = stage1_run(raw_rts=list(raw_rts), **kwargs)
    return table[RAW_TRIAL_COLUMNS].reset_index(drop=True)


def stage_raw_behavior(
    bids_root,
    *,
    subject: str = "H01",
    condition: str = "Slow",
    run: str = "1",
    date: str = "180131",
    task: str = "tokens",
    **kwargs,
) -> Path:
    """Write one raw behavioral run where Stage 0 puts it, sidecar included.

    Stage 1 reads run identity from the sidecar rather than from the BIDS
    entities (which lowercase the condition and drop the date), so a fixture
    that omits it would not be readable by the stage under test.
    """
    table = raw_run(subject=subject, condition=condition, run=int(run), **kwargs)
    path = (
        Path(bids_root) / f"sub-{subject}" / "beh"
        / f"sub-{subject}_task-{task}_acq-{condition.lower()}_run-{run}_beh.tsv"
    )
    save_table(
        path,
        table,
        metadata={
            "stage": "raw_bids",
            "subject": subject,
            "condition": condition,
            "run": int(run),
            "acquisition_date": date,
            "source_file": f"{subject}{condition}{run}_{date}.tdms",
        },
    )
    return path


def trial_features(**columns) -> pd.DataFrame:
    """Build a canonical trial-feature table with valid defaults.

    Every column declared by ``TRIAL_FEATURE_COLUMNS`` is present, so the table
    passes ``validate_trial_features`` unchanged. The trial count comes from the
    first supplied column. ``logged_spd_log_odds``, ``evidence_at_decision``,
    and ``logged_spd_validated_15row`` follow ``logged_spd`` unless overridden,
    so a test that varies SPD does not have to restate its transforms.
    """
    if not columns:
        raise ValueError("supply at least one column to set the trial count")
    length = len(next(iter(columns.values())))
    defaults = {
        "subject": ["H01"] * length,
        "condition": ["Fast"] * length,
        "run": [1] * length,
        "block_index": [1] * length,
        "run_trial_index": list(range(1, length + 1)),
        "started_trial_index": list(range(1, length + 1)),
        "nTrialIndex": list(range(1, length + 1)),
        "initial_time_ms": [1000 * index for index in range(length)],
        "source_file": ["H01Fast1_180131.tdms"] * length,
        "trial_class": [1] * length,
        "trial_class_name": ["easy"] * length,
        "sTrialClassRaw": ["e"] * length,
        "trial_class_source": ["design"] * length,
        "trial_class_rule": ["recorded_label"] * length,
        "nChoiceMade": [1] * length,
        "nCorrectChoice": [1] * length,
        "choice_side": [1] * length,
        "correct_side": [1] * length,
        "isCorrect": [True] * length,
        "nOutcome": [0] * length,
        "outcome_label": ["completed"] * length,
        "motor_baseline_ms": [350.0] * length,
        "rawRT": [1350.0] * length,
        "dt_ms": [1000.0] * length,
        "logged_spd": [0.8] * length,
        "decision_token_index": [5] * length,
        "token_directions": [TOKEN_DIRECTIONS] * length,
        "sp_design_early": [0.6] * length,
        "sum_log_lr_design_early": [0.4054651081081644] * length,
        "token_lead_at_decision": [1] * length,
        "sum_log_lr_at_decision": [1.3862943611198906] * length,
        "sum_log_lr_saturated": [False] * length,
        "token_log_rows": [15] * length,
        "token_log_short": [False] * length,
        "is_started": [True] * length,
        "has_choice": [True] * length,
        "is_no_response": [False] * length,
        "dt_anticipation": [False] * length,
        "spd_available": [True] * length,
        "design_time_alignment_valid": [True] * length,
        "primary_analysis_eligible": [True] * length,
    }
    defaults.update(columns)
    table = pd.DataFrame(defaults)

    spd = pd.to_numeric(table["logged_spd"], errors="coerce")
    if "logged_spd_log_odds" not in columns:
        # ln(p / (1 - p)), left missing wherever SPD is missing.
        with np.errstate(divide="ignore", invalid="ignore"):
            table["logged_spd_log_odds"] = np.log(spd / (1.0 - spd))
    if "evidence_at_decision" not in columns:
        table["evidence_at_decision"] = spd - 0.5
    if "logged_spd_validated_15row" not in columns:
        table["logged_spd_validated_15row"] = spd.where(
            table["token_log_rows"] == 15
        )
    table["trial_id"] = [
        f"sub-{row.subject}_task-tokens_run-{row.run}"
        f"_desc-{str(row.condition).lower()}_trial-{row.run_trial_index:03d}"
        for row in table.itertuples()
    ]
    return table


# Three subjects, each with an RT baseline run plus one Fast and one Slow task
# run. Decision times differ by subject and condition so group-level contrasts
# have something to find, and the Slow blocks are recorded before the Fast
# blocks on the session clock so block-order analyses see a real interleaving.
SESSION_LAYOUT = {
    "H01": {"rt": [340.0, 360.0], "fast": 900.0, "slow": 1300.0},
    "H02": {"rt": [380.0, 420.0], "fast": 1000.0, "slow": 1500.0},
    "H03": {"rt": [300.0, 340.0], "fast": 1100.0, "slow": 1400.0},
}


def stage_behavior(bids_root, layout=SESSION_LAYOUT):
    """Write a small but complete Stage 1 dataset under ``bids_root``."""
    derivatives = DerivativeLayout(bids_root)
    for subject, timing in layout.items():
        runs = [
            stage1_run(
                subject=subject,
                condition="RT",
                run=1,
                raw_rts=timing["rt"],
                initial_time_ms=0,
            ),
            stage1_run(
                subject=subject,
                condition="Slow",
                run=1,
                raw_rts=[
                    timing["slow"], timing["slow"] + 200.0,
                    timing["slow"] + 100.0, timing["slow"] + 300.0,
                ],
                choices=[1, 2, 1, 2],
                correct_choices=[1, 2, 2, 2],
                trial_classes=[1, 2, 3, 1],
                initial_time_ms=100_000,
            ),
            stage1_run(
                subject=subject,
                condition="Fast",
                run=1,
                raw_rts=[
                    timing["fast"], timing["fast"] + 200.0,
                    timing["fast"] + 100.0, timing["fast"] + 300.0,
                ],
                choices=[1, 2, 1, 2],
                correct_choices=[1, 1, 2, 2],
                trial_classes=[1, 2, 3, 1],
                initial_time_ms=200_000,
            ),
        ]
        for table in runs:
            save_table(
                derivatives.behavior(
                    subject=subject,
                    run=str(table["run"].iloc[0]),
                    condition=str(table["condition"].iloc[0]),
                ),
                table,
                metadata={"stage": "behavior_parsing"},
            )
    return derivatives
