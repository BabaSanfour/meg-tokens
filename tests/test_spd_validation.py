import pandas as pd

from meg_tokens.behavior.success_probability import success_probability_profile
from meg_tokens.validation.spd import validate_spd_trial


def test_validate_spd_trial_matches_a_complete_runtime_profile():
    directions = "112112211121122"
    profile = success_probability_profile(directions, target=1)
    trial = pd.Series(
        {
            "nProb": profile,
            "tTime": list(range(1200, 4200, 200)),
            "nTokenDir": [int(value) for value in directions],
            "sTokenDirs": directions,
            "nChoiceMade": 1,
            "tEnterTarget": 1800,
        }
    )

    result = validate_spd_trial(trial, motor_baseline_ms=500)

    assert result["runtime_profile_full_match"] is True
    assert result["runtime_spd_match"] is True
    assert result["design_time_resolved_valid_for_analysis"] is True
    assert result["design_spd_for_analysis"] == result["design_unshifted_spd"]
    assert result["decision_index"] == 0
    assert result["logged_spd"] == 0.604736328125


def test_validate_spd_trial_withholds_design_spd_for_14_rows():
    directions = "112112211121122"
    profile = success_probability_profile(directions, target=1)
    trial = pd.Series(
        {
            "nProb": profile[1:],
            "tTime": list(range(1200, 4000, 200)),
            "nTokenDir": [int(value) for value in directions[:14]],
            "sTokenDirs": directions,
            "nChoiceMade": 1,
            "tEnterTarget": 1800,
        }
    )

    result = validate_spd_trial(trial, motor_baseline_ms=500)

    assert result["design_time_resolved_valid_for_analysis"] is False
    assert pd.isna(result["design_spd_for_analysis"])
