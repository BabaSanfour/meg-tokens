"""Real-data validation for success-probability profiles and SPD timing."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from meg_tokens.behavior.math.probability import (
    classify_design_profile,
    probability_at_decision,
    success_probability_profile,
    validate_probability_path,
)
from meg_tokens.behavior.schema import parse_token_directions
from meg_tokens.behavior.tdms import parse_tdms_file, parse_tdms_filename


DEFAULT_IGNORE_FILES = {
    "temp_180214.tdms",
    "temp_181024.tdms",
    "temp_181121.tdms",
}


def _allclose(left: Iterable[float], right: Iterable[float], *, atol: float) -> bool:
    return bool(np.allclose(list(left), list(right), atol=atol, rtol=0.0))


def _point_matches(left: Iterable[float], right: Iterable[float], *, atol: float) -> tuple[int, int]:
    left_values = np.asarray(list(left), dtype=float)
    right_values = np.asarray(list(right), dtype=float)
    if left_values.shape != right_values.shape:
        raise ValueError("Profiles must have equal shapes")
    return int(np.sum(np.isclose(left_values, right_values, atol=atol, rtol=0.0))), int(left_values.size)


def _mean_abs_diff(left: Iterable[float], right: Iterable[float]) -> float:
    return float(np.mean(np.abs(np.asarray(list(left), dtype=float) - np.asarray(list(right), dtype=float))))


def validate_spd_trial(
    trial: pd.Series,
    *,
    motor_baseline_ms: float,
    atol: float = 1e-6,
) -> dict[str, object]:
    """Validate profile and SPD candidates for one started-and-chosen trial."""
    logged = [float(value) for value in trial["nProb"]]
    token_times = [float(value) for value in trial["tTime"]]
    runtime_directions = [int(value) for value in trial["nTokenDir"]]
    design_directions = parse_token_directions(trial["sTokenDirs"])
    chosen_target = int(trial["nChoiceMade"])

    if len(logged) not in (14, 15):
        raise ValueError(f"Expected 14 or 15 token rows, got {len(logged)}")
    if len(runtime_directions) != len(logged) or len(token_times) != len(logged):
        raise ValueError("Runtime token directions, times, and probabilities must align")
    if len(design_directions) != 15:
        raise ValueError("sTokenDirs must contain exactly 15 directions")

    runtime_profile = success_probability_profile(runtime_directions, target=chosen_target)
    design_profile = success_probability_profile(design_directions, target=chosen_target)
    design_unshifted = design_profile[: len(logged)]
    design_shifted = design_profile[1:] if len(logged) == 14 else None
    try:
        validate_probability_path(
            logged,
            first_jump=2 if len(logged) == 14 else 1,
            atol=atol,
        )
        logged_profile_is_legal_path = True
    except ValueError:
        logged_profile_is_legal_path = False

    decision_timestamp = float(trial["tEnterTarget"]) - float(motor_baseline_ms)
    logged_spd, decision_index = probability_at_decision(
        logged,
        token_times,
        decision_time=decision_timestamp,
    )
    design_time_resolved_valid_for_analysis = len(logged) == 15
    if design_time_resolved_valid_for_analysis:
        design_spd_for_analysis, _ = probability_at_decision(
            design_profile,
            token_times,
            decision_time=decision_timestamp,
        )
    else:
        design_spd_for_analysis = np.nan
    if decision_index is None:
        runtime_spd = 0.5
        design_unshifted_spd = 0.5
        design_shifted_spd = 0.5 if design_shifted is not None else np.nan
    else:
        runtime_spd = runtime_profile[decision_index]
        design_unshifted_spd = design_unshifted[decision_index]
        design_shifted_spd = (
            design_shifted[decision_index] if design_shifted is not None else np.nan
        )

    runtime_points, n_points = _point_matches(logged, runtime_profile, atol=atol)
    design_unshifted_points, _ = _point_matches(logged, design_unshifted, atol=atol)
    if design_shifted is None:
        design_shifted_points = np.nan
        design_shifted_full_match = np.nan
        design_shifted_mae = np.nan
        design_shifted_spd_match = np.nan
    else:
        design_shifted_points, _ = _point_matches(logged, design_shifted, atol=atol)
        design_shifted_full_match = _allclose(logged, design_shifted, atol=atol)
        design_shifted_mae = _mean_abs_diff(logged, design_shifted)
        design_shifted_spd_match = bool(np.isclose(logged_spd, design_shifted_spd, atol=atol, rtol=0.0))

    return {
        "log_rows": len(logged),
        "logged_profile_is_legal_path": logged_profile_is_legal_path,
        "n_profile_values": n_points,
        "runtime_profile_point_matches": runtime_points,
        "runtime_profile_full_match": _allclose(logged, runtime_profile, atol=atol),
        "runtime_profile_mae": _mean_abs_diff(logged, runtime_profile),
        "design_unshifted_point_matches": design_unshifted_points,
        "design_unshifted_full_match": _allclose(logged, design_unshifted, atol=atol),
        "design_unshifted_mae": _mean_abs_diff(logged, design_unshifted),
        "design_shifted_point_matches": design_shifted_points,
        "design_shifted_full_match": design_shifted_full_match,
        "design_shifted_mae": design_shifted_mae,
        "decision_timestamp": decision_timestamp,
        "decision_index": -1 if decision_index is None else decision_index,
        "decision_before_first_token": decision_index is None,
        "decision_after_last_token": decision_timestamp >= token_times[-1],
        "logged_spd": logged_spd,
        "design_time_resolved_valid_for_analysis": (
            design_time_resolved_valid_for_analysis
        ),
        "design_spd_for_analysis": design_spd_for_analysis,
        "runtime_spd": runtime_spd,
        "runtime_spd_match": bool(np.isclose(logged_spd, runtime_spd, atol=atol, rtol=0.0)),
        "design_unshifted_spd": design_unshifted_spd,
        "design_unshifted_spd_match": bool(
            np.isclose(logged_spd, design_unshifted_spd, atol=atol, rtol=0.0)
        ),
        "design_shifted_spd": design_shifted_spd,
        "design_shifted_spd_match": design_shifted_spd_match,
    }


def _load_runs(
    behavior_root: str | Path,
    *,
    ignore_files: Iterable[str],
) -> list[pd.DataFrame]:
    root = Path(behavior_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Behavior root does not exist: {root}")
    ignored = set(ignore_files)
    runs: list[pd.DataFrame] = []
    for path in sorted(root.rglob("*.tdms")):
        if path.name in ignored:
            continue
        info = parse_tdms_filename(path.name)
        table = parse_tdms_file(str(path)).copy()
        table.insert(0, "subject", info.subject)
        table.insert(1, "condition", info.condition)
        table.insert(2, "run", int(info.run))
        table.insert(3, "source_file", str(path))
        runs.append(table)
    return runs


def _legacy_trial_class(trial: pd.Series) -> int:
    """Reproduce the unchanged logged-nProb override for comparison."""
    raw_class = str(trial["sTrialClassRaw"])
    trial_class = {"e": 1, "a": 2, "m": 3}.get(raw_class, 0)
    probabilities = [float(value) for value in trial["nProb"]]
    if len(probabilities) < 8:
        return trial_class
    if probabilities[1] > 0.6 and probabilities[4] > 0.75 and probabilities[7] > 0.75:
        return 1
    if (
        probabilities[1] == 0.5
        and 0.38 < probabilities[2] < 0.65
        and 0.35 < probabilities[4] < 0.65
    ):
        return 2
    if probabilities[2] < 0.4:
        return 3
    return trial_class


def _subject_motor_baselines(runs: list[pd.DataFrame]) -> dict[str, float]:
    rt = pd.concat([run for run in runs if run["condition"].iat[0] == "RT"], ignore_index=True)
    chosen = rt["nChoiceMade"] > 0
    values = (rt.loc[chosen, "tEnterTarget"] - rt.loc[chosen, "tGO"]).astype(float)
    baselines = values.groupby(rt.loc[chosen, "subject"]).mean().to_dict()
    subjects = {str(run["subject"].iat[0]) for run in runs}
    missing = sorted(subjects - set(baselines))
    if missing:
        raise ValueError(f"Subjects have no valid RT baseline trials: {missing}")
    return {str(subject): float(value) for subject, value in baselines.items()}


def _summarize(details: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for log_rows, group in details.groupby("log_rows", sort=True):
        profile_values = int(group["n_profile_values"].sum())
        row: dict[str, object] = {
            "log_rows": int(log_rows),
            "n_trials": int(len(group)),
            "n_profile_values": profile_values,
            "logged_legal_path_pct": 100 * float(group["logged_profile_is_legal_path"].mean()),
            "runtime_profile_full_match_pct": 100 * float(group["runtime_profile_full_match"].mean()),
            "runtime_profile_point_match_pct": 100 * float(group["runtime_profile_point_matches"].sum()) / profile_values,
            "runtime_profile_mean_mae": float(group["runtime_profile_mae"].mean()),
            "design_unshifted_full_match_pct": 100 * float(group["design_unshifted_full_match"].mean()),
            "design_unshifted_point_match_pct": 100 * float(group["design_unshifted_point_matches"].sum()) / profile_values,
            "design_unshifted_mean_mae": float(group["design_unshifted_mae"].mean()),
            "runtime_spd_match_pct": 100 * float(group["runtime_spd_match"].mean()),
            "runtime_spd_mean_abs_diff": float(
                np.mean(np.abs(group["logged_spd"] - group["runtime_spd"]))
            ),
            "design_unshifted_spd_match_pct": 100 * float(group["design_unshifted_spd_match"].mean()),
            "design_unshifted_spd_mean_abs_diff": float(
                np.mean(np.abs(group["logged_spd"] - group["design_unshifted_spd"]))
            ),
            "n_decisions_before_first_token": int(group["decision_before_first_token"].sum()),
            "n_decisions_after_last_token": int(group["decision_after_last_token"].sum()),
        }
        if int(log_rows) == 14:
            row.update(
                {
                    "design_shifted_full_match_pct": 100 * float(group["design_shifted_full_match"].mean()),
                    "design_shifted_point_match_pct": 100 * float(group["design_shifted_point_matches"].sum()) / profile_values,
                    "design_shifted_mean_mae": float(group["design_shifted_mae"].mean()),
                    "design_shifted_spd_match_pct": 100 * float(group["design_shifted_spd_match"].mean()),
                    "design_shifted_spd_mean_abs_diff": float(
                        np.mean(np.abs(group["logged_spd"] - group["design_shifted_spd"]))
                    ),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def run_spd_validation(
    behavior_root: str | Path,
    *,
    ignore_files: Iterable[str] = DEFAULT_IGNORE_FILES,
    atol: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate SP profiles and SPD candidates across the real behavior logs."""
    runs = _load_runs(behavior_root, ignore_files=ignore_files)
    baselines = _subject_motor_baselines(runs)
    rows: list[dict[str, object]] = []
    for run in runs:
        if run["condition"].iat[0] not in {"Fast", "Slow"}:
            continue
        selected = run[run["nChoiceMade"] > 0]
        for _, trial in selected.iterrows():
            raw_class = str(trial["sTrialClassRaw"])
            historical_class = _legacy_trial_class(trial)
            recorded_classes = {"e": 1, "a": 2, "m": 3}
            if raw_class in recorded_classes:
                revised_class = recorded_classes[raw_class]
                revised_class_source = "design"
            elif raw_class == "x":
                correct_profile = success_probability_profile(
                    parse_token_directions(trial["sTokenDirs"]),
                    target=int(trial["nCorrectChoice"]),
                )
                revised_class, _ = classify_design_profile(
                    correct_profile,
                    atol=atol,
                )
                revised_class_source = "inferred" if revised_class else "unclassified"
            else:
                revised_class = 0
                revised_class_source = "not_applicable"
            validation = validate_spd_trial(
                trial,
                motor_baseline_ms=baselines[str(trial["subject"])],
                atol=atol,
            )
            rows.append(
                {
                    "subject": trial["subject"],
                    "condition": trial["condition"],
                    "run": int(trial["run"]),
                    "nTrialIndex": int(trial["nTrialIndex"]),
                    "historical_trial_class": historical_class,
                    "raw_trial_class": raw_class,
                    "revised_trial_class": revised_class,
                    "revised_trial_class_source": revised_class_source,
                    "nChoiceMade": int(trial["nChoiceMade"]),
                    "nCorrectChoice": int(trial["nCorrectChoice"]),
                    "is_correct": bool(trial["nChoiceMade"] == trial["nCorrectChoice"]),
                    "motor_baseline_ms": baselines[str(trial["subject"])],
                    **validation,
                }
            )
    details = pd.DataFrame(rows)
    if details.empty:
        raise ValueError("No started-and-chosen Fast/Slow trials were found")
    return _summarize(details), details
