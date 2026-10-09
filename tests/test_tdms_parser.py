"""Transcription tests for the TDMS parser.

The parser transcribes what LabVIEW wrote and derives the designed
success-probability profile; it assigns no trial class. Everything about
which class a trial belongs to now lives in
``tests/behavior/test_classification.py``, next to the module that decides
it.
"""

import pytest
import pandas as pd
from meg_tokens.behavior.math.probability import success_probability_profile
from meg_tokens.behavior.schema import (
    CLASSIFICATION_COLUMNS,
    RAW_TRIAL_COLUMNS,
    parse_token_directions,
)
from meg_tokens.behavior.tdms import (
    parse_single_trial,
    parse_tdms_file,
)

# Sample Events string mimicking the real TDMS structure
EVENTS_STR = """sTaskType: 'TokensMvt'
dDate: '2018/01/31 04:20:52.738 PM'
nInitialTime: 6232233
nTrialIndex: 1
sNeuralFilename: 'H1Slow2'
nChoiceMade: 2
nCorrectChoice: 2
nSelectedTarget: 4
nUnselectedTarget: 1
tEnterCenter: 3
tClick: 3
tGO: 1041
tExitCenter: 2484
tEnterTarget: 2484
tTrialEnd: 3676
nOutcome: 0
sTrialClass: 'x'
sTokenDirs: '221221122212211'
Tokens.Data: (
[tTime: 1260, nTokenNum: 1, nTokenDir: 2, nProb: 0.70947265625000, nTokenX: 575, nTokenY: 474], 
[tTime: 1460, nTokenNum: 2, nTokenDir: 2, nProb: 0.81234567890123, nTokenX: 576, nTokenY: 475]
)
"""


def make_events(
    prob_values,
    *,
    token_dirs="221221122212211",
    correct_choice=2,
    trial_class="x",
):
    steps = [
        f"[tTime: {1000 + i*200}, nTokenNum: {i+1}, nTokenDir: 2, nProb: {p}]"
        for i, p in enumerate(prob_values)
    ]
    lines = EVENTS_STR.splitlines()
    idx = lines.index("Tokens.Data: (")
    new_lines = lines[:idx+1] + steps + [")"]
    return (
        "\n".join(new_lines)
        .replace("nCorrectChoice: 2", f"nCorrectChoice: {correct_choice}")
        .replace("sTrialClass: 'x'", f"sTrialClass: '{trial_class}'")
        .replace("sTokenDirs: '221221122212211'", f"sTokenDirs: '{token_dirs}'")
    )


def test_design_profile_comes_from_the_token_sequence_not_the_logged_path():
    """``sp_design_correct`` is what class inference later reads. It is derived
    from the designed sequence and the correct target, so a logged probability
    path that disagrees with it cannot leak into a class."""
    events_str = make_events([0.1] * 15)
    trial_dict = parse_single_trial(events_str)

    assert trial_dict['sp_design_correct'] == success_probability_profile(
        parse_token_directions("221221122212211"), target=2
    )
    assert trial_dict['sp_design_correct'] != trial_dict['nProb']


def test_design_profile_is_identical_for_14_and_15_log_rows():
    """A truncated runtime log shortens nProb but not the designed profile,
    which is why a 14-row trial classifies exactly like a 15-row one."""
    kwargs = {
        "token_dirs": "212112111111212",
        "correct_choice": 1,
    }
    short = parse_single_trial(make_events([0.5] * 14, **kwargs))
    complete = parse_single_trial(make_events([0.5] * 15, **kwargs))

    assert short['sp_design_correct'] == complete['sp_design_correct']
    assert len(short['sp_design_correct']) == 15
    assert short['token_log_rows'] == 14
    assert short['token_log_short'] is True
    assert complete['token_log_rows'] == 15
    assert complete['token_log_short'] is False


def test_parse_real_tdms_integration(behavior_logs):
    df = parse_tdms_file(behavior_logs / "H02" / "H02Slow1_180213.tdms")

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == RAW_TRIAL_COLUMNS
    # TrialData indices should be sequential starting at 1
    assert df['nTrialIndex'].iloc[0] == 1
    # A real run records labels the acquisition software wrote, and no
    # column this project inferred.
    assert set(df['sTrialClassRaw']) <= {'x', 'e', 'a', 'm', 'r'}
    assert set(CLASSIFICATION_COLUMNS).isdisjoint(df.columns)


def test_parse_single_trial_rejects_desynced_token_fields():
    """A Tokens.Data row matching only some patterns would silently misalign
    every subsequent index-based criterion."""
    events_str = make_events(["0.5", "0.6"]).replace(
        "[tTime: 1200, nTokenNum: 2, nTokenDir: 2, nProb: 0.6]",
        "[tTime: 1200, nTokenNum: 2, nTokenDir: 2, nProb: nan]",
    )

    with pytest.raises(ValueError, match="Tokens.Data field lengths differ"):
        parse_single_trial(events_str)
