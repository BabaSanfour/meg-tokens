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
tests read a real run from this checkout's configured data (see
``tests/conftest.py``) and skip where it is absent.
"""

from __future__ import annotations


from pandas.testing import assert_frame_equal

from meg_tokens.behavior.classification import classify_trial, classify_trials
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


# --- trials logged 'x': the inference decision --------------------------------


def test_a_random_trial_takes_its_class_from_the_designed_profile():
    for directions, trial_class, rule in (
        (EASY, 1, "easy_sp2_sp5_sp8"),
        (AMBIGUOUS, 2, "ambiguous_sp2_sp3_sp5"),
        (MISLEADING, 3, "misleading_sp3"),
    ):
        assert classify_trial(
            "x", _profile(directions, target=1), infer_random_classes=True
        ) == (trial_class, "inferred", rule)


# --- other labels -------------------------------------------------------------


def test_a_repeat_trial_is_not_applicable_rather_than_unclassified():
    for infer_random_classes in (True, False):
        assert classify_trial(
            "r", _profile(EASY, target=1), infer_random_classes=infer_random_classes
        ) == (0, "not_applicable", "not_applicable")


# --- reproducibility from the raw layer, on a real run ------------------------


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


def test_a_real_run_classifies_exactly_as_the_parser_used_to(tmp_path, behavior_logs):
    table = parse_tdms_file(behavior_logs / "H02" / "H02Slow1_180213.tdms")
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

    # Classification is a function of ``(sTrialClassRaw, sp_design_correct)``
    # and nothing else, which is what lets a derivative be regenerated from the
    # staged raw table with the ``.tdms`` container out of reach.
    projected = classify_trials(table[["sTrialClassRaw", "sp_design_correct"]])
    assert_frame_equal(
        classified[list(CLASSIFICATION_COLUMNS)], projected[list(CLASSIFICATION_COLUMNS)]
    )

    # The staged raw table, read back with the project loader, classifies to
    # the same columns as the freshly parsed one -- serialization of the design
    # profile included.
    staged = tmp_path / "sub-H02_task-tokens_acq-slow_run-1_beh.tsv"
    save_table(staged, table, metadata={"stage": "raw_bids"})
    reloaded = classify_trials(read_raw_behavior_table(staged))
    assert_frame_equal(
        reloaded[list(CLASSIFICATION_COLUMNS)],
        classified[list(CLASSIFICATION_COLUMNS)],
    )
