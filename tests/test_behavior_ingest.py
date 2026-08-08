"""Stage 1 behavioral ingestion: staged raw BIDS tables -> derivatives.

Ingestion no longer reads ``.tdms`` containers. It reads what Stage 0 staged
under ``BIDS/sub-*/beh/`` and adds what makes a derivative a derivative: trial
classes, run identity, and the response fields derived from them. The tables
staged here are plumbing fixtures for that stage -- they exercise path,
sidecar and column bookkeeping, not any claim about a real recording.
"""

import json

import pandas as pd
import pytest

from meg_tokens.behavior.schema import CLASSIFICATION_COLUMNS
from meg_tokens.behavior.tdms import (
    FILENAME_RE,
    parse_tdms_filename,
)
from meg_tokens.io import DerivativeLayout
from meg_tokens.workflows.behavior_ingest import ingest_subject_behavior

from tests.behavior.factories import stage_raw_behavior


# Test regex match groupings
@pytest.mark.parametrize("filename,expected", [
    ("H1Slow1_180131.tdms", ("Slow", "1", "180131")),
    ("H02Fast4_180213.tdms", ("Fast", "4", "180213")),
    ("H32RT2_190214.tdms", ("RT", "2", "190214")),
])
def test_filename_regex(filename, expected):
    match = FILENAME_RE.match(filename)
    assert match is not None
    assert (match.group("condition"), match.group("run"), match.group("date")) == expected


def test_parse_tdms_filename_normalizes_subject():
    info = parse_tdms_filename("H1Slow1_180131.tdms")
    assert info.subject == "H01"
    assert info.condition == "Slow"
    assert info.run == "1"
    assert info.date == "180131"


def test_ingest_dry_run_declares_one_output_per_staged_run(tmp_path):
    bids_root = tmp_path / "BIDS"
    output_dir = tmp_path / "dataframes"
    for condition, run in (("Slow", "1"), ("Fast", "2"), ("RT", "1")):
        stage_raw_behavior(bids_root, condition=condition, run=run)

    results = ingest_subject_behavior(
        "H1",
        bids_root=bids_root,
        output_root=output_dir,
        dry_run=True,
    )

    assert [record["output"].split("/")[-1] for record in results] == [
        "sub-H01_task-tokens_run-2_desc-fast_beh.tsv",
        "sub-H01_task-tokens_run-1_desc-rt_beh.tsv",
        "sub-H01_task-tokens_run-1_desc-slow_beh.tsv",
    ]
    assert all(record["trials"] == 0 for record in results)
    # Dry run shouldn't write files
    assert not (output_dir / "derivatives").exists()


def test_ingest_raises_when_the_subject_has_no_staged_behavior(tmp_path):
    """An empty result is indistinguishable from success, so a missing Stage 0
    prerequisite has to say so."""
    with pytest.raises(FileNotFoundError, match="No staged raw behavior"):
        ingest_subject_behavior(
            "H1",
            bids_root=tmp_path / "BIDS",
            output_root=tmp_path / "dataframes",
            dry_run=True,
        )


def test_ingest_writes_a_classified_derivative_with_run_identity(tmp_path):
    bids_root = tmp_path / "BIDS"
    output_dir = tmp_path / "dataframes"
    stage_raw_behavior(bids_root, condition="Fast", run="2", trial_classes=[1, 2])

    results = ingest_subject_behavior(
        "H1",
        bids_root=bids_root,
        output_root=output_dir,
        dry_run=False,
    )

    assert len(results) == 1
    run = parse_tdms_filename("H1Fast2_180131.tdms")
    out_path = DerivativeLayout(output_dir).behavior(
        subject=run.subject,
        run=run.run,
        condition=run.condition,
    )
    assert out_path.exists()
    assert out_path.with_suffix(".json").exists()

    written = pd.read_csv(out_path, sep="\t")
    assert written["subject"].tolist() == ["H01", "H01"]
    assert written["condition"].tolist() == ["Fast", "Fast"]
    assert written["run"].tolist() == [2, 2]
    assert written["rawRT"].iloc[0] == 400
    assert pd.isna(written["rawRT"].iloc[1])
    assert bool(written["isCorrect"].iloc[0]) is True
    # What the stage adds on top of the raw layer it read.
    assert set(CLASSIFICATION_COLUMNS) <= set(written.columns)
    assert written["sTrialClass"].tolist() == [1, 2]
    assert written["trial_class_rule"].tolist() == ["recorded_label"] * 2


def test_ingest_records_the_inference_choice_it_was_run_with(tmp_path):
    """The derivative states which policy produced it; the raw table it came
    from carries no such column, so the sidecar is the only record."""
    bids_root = tmp_path / "BIDS"
    output_dir = tmp_path / "dataframes"
    raw_path = stage_raw_behavior(bids_root, condition="Fast", run="2")

    ingest_subject_behavior(
        "H1",
        bids_root=bids_root,
        output_root=output_dir,
        dry_run=False,
        infer_random_classes=False,
    )

    out_path = DerivativeLayout(output_dir).behavior(
        subject="H01", run="2", condition="Fast"
    )
    sidecar = json.loads(out_path.with_suffix(".json").read_text())
    assert sidecar["metadata"]["infer_random_classes"] is False
    assert sidecar["metadata"]["raw_behavior_table"] == str(raw_path)
    assert sidecar["metadata"]["source_file"] == "H01Fast2_180131.tdms"


def test_filename_regex_rejects_non_six_digit_date():
    assert FILENAME_RE.match("H1Slow1_18013.tdms") is None
    assert FILENAME_RE.match("H1Slow1_2018-01-31.tdms") is None
