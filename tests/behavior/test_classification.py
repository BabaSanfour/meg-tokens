"""Tests for meg_tokens.behavior.classification.

Trial classification is a derivative-stage judgement applied to a table, not
something the acquisition software recorded. Its inputs are two transcribed
fields -- the recorded label ``sTrialClassRaw`` and the designed profile
``sp_design_correct`` -- so most of it is testable as a pure function of
scalars and probability lists, which describe no recording and are built here
from the already-tested probability layer rather than typed out by hand.

The one property that genuinely needs a real behavioral run is the
reproducibility claim: that a real log's classification is recoverable from
the staged raw table alone, with no access to the ``.tdms`` container. Those
tests read a real run and skip cleanly when it is not available. Point
``MEG_TOKENS_TDMS_ROOT`` at the behavioral log root to run them:

    MEG_TOKENS_TDMS_ROOT=<tdms-root> python -m pytest tests/behavior/test_classification.py
"""

from __future__ import annotations

import os

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from meg_tokens.behavior.classification import (
    RECORDED_CLASSES,
    classify_trial,
    classify_trials,
)
from meg_tokens.behavior.math.probability import (
    classify_design_profile,
    success_probability_profile,
)
from meg_tokens.behavior.schema import (
    CLASSIFICATION_COLUMNS,
    RAW_TRIAL_COLUMNS,
    parse_token_directions,
)
from meg_tokens.behavior.tables import read_raw_behavior_table
from meg_tokens.behavior.tdms import parse_tdms_file
from meg_tokens.io import save_table


def _profile(directions: str, target: int) -> list[float]:
    """The designed correct-target profile for one 15-jump token sequence."""
    return success_probability_profile(parse_token_directions(directions), target=target)


# A designed sequence per class, taken from the rules the probability layer
# implements: easy, ambiguous, and misleading, plus one matching no rule.
EASY = "111111112222222"
AMBIGUOUS = "121121122112212"
MISLEADING = "222111111112222"
UNMATCHED = "111122221111111"


# --- one trial's recorded label -----------------------------------------------


@pytest.mark.parametrize(("label", "trial_class"), [("e", 1), ("a", 2), ("m", 3)])
@pytest.mark.parametrize("infer_random_classes", [True, False])
def test_a_recorded_label_is_taken_at_face_value(
    label, trial_class, infer_random_classes
):
    """A class the acquisition software asserted is read straight off the
    label. Inference is a policy for trials logged ``'x'`` only, so switching
    it off must not touch a trial that was never in question."""
    assert classify_trial(
        label, _profile(UNMATCHED, target=1), infer_random_classes=infer_random_classes
    ) == (trial_class, "design", "recorded_label")


def test_a_recorded_label_is_read_case_insensitively():
    assert classify_trial("E", None, infer_random_classes=True) == (
        1,
        "design",
        "recorded_label",
    )


def test_the_recorded_class_map_is_the_three_designed_classes():
    assert RECORDED_CLASSES == {"e": 1, "a": 2, "m": 3}


# --- trials logged 'x': the inference decision --------------------------------


@pytest.mark.parametrize(
    ("directions", "trial_class", "rule"),
    [
        (EASY, 1, "easy_sp2_sp5_sp8"),
        (AMBIGUOUS, 2, "ambiguous_sp2_sp3_sp5"),
        (MISLEADING, 3, "misleading_sp3"),
    ],
)
def test_a_random_trial_takes_its_class_from_the_designed_profile(
    directions, trial_class, rule
):
    assert classify_trial(
        "x", _profile(directions, target=1), infer_random_classes=True
    ) == (trial_class, "inferred", rule)


def test_a_random_trial_matching_no_rule_stays_unclassified():
    """An inferred class is only as good as the rule that produced it, so a
    profile no rule matches yields no class rather than a nearest guess."""
    assert classify_trial(
        "x", _profile(UNMATCHED, target=1), infer_random_classes=True
    ) == (0, "unclassified", "unclassified")


def test_inference_disabled_declines_even_a_profile_that_would_classify():
    """The rule that would have matched is recorded nowhere: with inference
    off the trial is unclassified because the project declined, and the rule
    says so."""
    assert classify_trial(
        "x", _profile(EASY, target=1), infer_random_classes=False
    ) == (0, "unclassified", "inference_disabled")


def test_a_random_trial_without_a_correct_target_is_unclassifiable():
    """No correct target means no designed profile to read a class out of --
    a different fact from having declined to look, and recorded as such."""
    assert classify_trial("x", None, infer_random_classes=True) == (
        0,
        "unclassified",
        "no_correct_target",
    )


def test_declining_outranks_having_nothing_to_infer_from():
    """Both produce class 0; the rule distinguishes them, and the disabled
    switch is checked first because it applies whatever the profile holds."""
    assert (
        classify_trial("x", None, infer_random_classes=False)[2]
        == "inference_disabled"
    )


# --- other labels -------------------------------------------------------------


@pytest.mark.parametrize("infer_random_classes", [True, False])
def test_a_repeat_trial_is_not_applicable_rather_than_unclassified(
    infer_random_classes,
):
    assert classify_trial(
        "r", _profile(EASY, target=1), infer_random_classes=infer_random_classes
    ) == (0, "not_applicable", "not_applicable")


@pytest.mark.parametrize("label", [1, 2, 3, "2"])
def test_a_numeric_label_is_the_class_it_names(label):
    trial_class, source, rule = classify_trial(
        label, None, infer_random_classes=True
    )

    assert (trial_class, source, rule) == (int(label), "design", "recorded_numeric_label")


@pytest.mark.parametrize("label", [0, 7])
def test_a_numeric_label_outside_the_three_classes_carries_no_design(label):
    assert classify_trial(label, None, infer_random_classes=True) == (
        label,
        "unclassified",
        "recorded_numeric_label",
    )


# --- applying it to a table ---------------------------------------------------


def _raw_table() -> pd.DataFrame:
    """Two columns is all classification reads; the rest is carried through."""
    return pd.DataFrame(
        {
            "nTrialIndex": [1, 2, 3, 4],
            "sTrialClassRaw": ["e", "x", "x", "r"],
            "sp_design_correct": [
                _profile(EASY, target=1),
                _profile(MISLEADING, target=1),
                None,
                _profile(AMBIGUOUS, target=1),
            ],
        }
    )


def test_classify_trials_appends_exactly_the_classification_columns():
    table = _raw_table()

    classified = classify_trials(table)

    assert list(classified.columns) == list(table.columns) + list(CLASSIFICATION_COLUMNS)
    assert classified["sTrialClass"].tolist() == [1, 3, 0, 0]
    assert classified["trial_class_source"].tolist() == [
        "design",
        "inferred",
        "unclassified",
        "not_applicable",
    ]
    assert classified["trial_class_rule"].tolist() == [
        "recorded_label",
        "misleading_sp3",
        "no_correct_target",
        "not_applicable",
    ]


def test_classify_trials_carries_the_input_rows_through_unchanged():
    table = _raw_table()

    classified = classify_trials(table)

    assert_frame_equal(classified[table.columns.tolist()], table)


def test_classify_trials_does_not_mutate_its_input():
    """The raw table it reads is the staged file's content; a derivative that
    edited it in place would leave the caller holding a table that is neither
    raw nor derivative."""
    table = _raw_table()
    before = table.copy(deep=True)

    classify_trials(table)

    assert list(table.columns) == list(before.columns)
    assert_frame_equal(table, before)


def test_the_inference_switch_reaches_every_row():
    classified = classify_trials(_raw_table(), infer_random_classes=False)

    assert classified["sTrialClass"].tolist() == [1, 0, 0, 0]
    assert (classified["trial_class_source"] != "inferred").all()
    assert classified.loc[1, "trial_class_rule"] == "inference_disabled"
    assert classified.loc[2, "trial_class_rule"] == "inference_disabled"


@pytest.mark.parametrize("column", ["sTrialClassRaw", "sp_design_correct"])
def test_classify_trials_names_the_input_column_it_cannot_find(column):
    table = _raw_table().drop(columns=[column])

    with pytest.raises(ValueError, match=column):
        classify_trials(table)


def test_classify_trials_preserves_a_non_default_index():
    table = _raw_table().set_index("nTrialIndex")

    classified = classify_trials(table)

    assert classified.index.tolist() == [1, 2, 3, 4]
    assert classified["sTrialClass"].tolist() == [1, 3, 0, 0]


# --- reproducibility from the raw layer, on a real run ------------------------


def _real_run_path() -> str:
    root = os.environ.get("MEG_TOKENS_TDMS_ROOT")
    if not root:
        pytest.skip("Set MEG_TOKENS_TDMS_ROOT to the behavioral log root to run this test.")
    path = os.path.join(root, "H02", "H02Slow1_180213.tdms")
    if not os.path.exists(path):
        pytest.skip(f"Real TDMS data not available at {path}")
    return path


def _expected_classification(label: object, profile: object) -> tuple:
    """The pre-refactor parser's contract, restated against the math layer.

    This is what ``parse_tdms_file`` used to write into every row when
    ``infer_random_classes`` was on. Stating it independently here is what
    makes the refactor checkable: the columns the derivative stage now
    produces must still be these.
    """
    normalized = str(label).lower()
    if normalized in {"e", "a", "m"}:
        return {"e": 1, "a": 2, "m": 3}[normalized], "design", "recorded_label"
    if normalized == "r":
        return 0, "not_applicable", "not_applicable"
    if normalized == "x":
        if profile is None:
            return 0, "unclassified", "no_correct_target"
        trial_class, rule = classify_design_profile(profile)
        return trial_class, ("inferred" if trial_class else "unclassified"), rule
    trial_class = int(label)
    return trial_class, ("design" if trial_class in (1, 2, 3) else "unclassified"), (
        "recorded_numeric_label"
    )


def test_a_real_run_classifies_exactly_as_the_parser_used_to():
    table = parse_tdms_file(_real_run_path())
    assert list(table.columns) == RAW_TRIAL_COLUMNS  # nothing pre-assigned

    classified = classify_trials(table)

    expected = [
        _expected_classification(row["sTrialClassRaw"], row["sp_design_correct"])
        for _, row in table.iterrows()
    ]
    assert [
        tuple(row[column] for column in CLASSIFICATION_COLUMNS)
        for _, row in classified.iterrows()
    ] == expected
    # The run must actually exercise both paths, or the comparison is empty.
    assert (classified["trial_class_source"] == "design").any()
    assert (classified["trial_class_source"] == "inferred").any()


def test_a_real_run_classifies_from_the_two_raw_columns_alone():
    """Classification is a function of ``(sTrialClassRaw, sp_design_correct)``
    and nothing else, which is what lets a derivative be regenerated from the
    staged raw table with the ``.tdms`` container out of reach."""
    table = parse_tdms_file(_real_run_path())

    full = classify_trials(table)
    projected = classify_trials(table[["sTrialClassRaw", "sp_design_correct"]])

    assert_frame_equal(
        full[list(CLASSIFICATION_COLUMNS)], projected[list(CLASSIFICATION_COLUMNS)]
    )


def test_a_real_run_classifies_the_same_after_a_round_trip_through_disk(tmp_path):
    """The staged raw table, read back with the project loader, classifies to
    the same columns as the freshly parsed one -- serialization of the design
    profile included."""
    table = parse_tdms_file(_real_run_path())
    staged = tmp_path / "sub-H02_task-tokens_acq-slow_run-1_beh.tsv"
    save_table(staged, table, metadata={"stage": "raw_bids"})

    reloaded = classify_trials(read_raw_behavior_table(staged))
    direct = classify_trials(table)

    assert_frame_equal(
        reloaded[list(CLASSIFICATION_COLUMNS)],
        direct[list(CLASSIFICATION_COLUMNS)],
    )


def test_a_real_run_infers_nothing_when_inference_is_off():
    table = parse_tdms_file(_real_run_path())

    classified = classify_trials(table, infer_random_classes=False)

    assert (classified["trial_class_source"] != "inferred").all()
    random_trials = classified["sTrialClassRaw"].astype(str).str.lower() == "x"
    assert random_trials.any()
    assert (classified.loc[random_trials, "sTrialClass"] == 0).all()
    assert (
        classified.loc[random_trials, "trial_class_rule"] == "inference_disabled"
    ).all()
    # Recorded labels are unaffected: only the 'x' trials were ever at stake.
    assert (
        classified.loc[~random_trials, "trial_class_rule"] == "recorded_label"
    ).all()
