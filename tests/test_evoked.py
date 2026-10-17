import pandas as pd

from meg_tokens.meg.evoked import CONDITIONS, condition_masks


def test_twelve_conditions_from_speed_class_and_side_with_side_1_the_right_hand():
    metadata = pd.DataFrame({
        "condition": ["Fast", "Fast", "Slow", "Slow", "Fast"],
        "trial_class_name": ["easy", "misleading", "ambiguous", "unclassified", "easy"],
        "choice_side": [1.0, 2.0, 1.0, 2.0, 2.0],
    })
    masks = condition_masks(metadata)

    assert list(masks) == list(CONDITIONS) and len(CONDITIONS) == 12
    assert masks["fast-easy-right"].tolist() == [True, False, False, False, False]
    assert masks["fast-misleading-left"].tolist() == [False, True, False, False, False]
    assert masks["slow-ambiguous-right"].tolist() == [False, False, True, False, False]
    assert masks["fast-easy-left"].tolist() == [False, False, False, False, True]
    assert not any(mask[3] for mask in masks.values())  # unclassified belongs to no condition
