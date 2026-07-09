import os
import pytest
import pandas as pd

from meg_tokens.behavior.tdms import (
    FILENAME_RE,
    parse_tdms_filename,
)
from meg_tokens.io import DerivativeLayout
from meg_tokens.workflows.behavior import ingest_subject_behavior

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

def test_batch_process_dry_run(tmp_path):
    # Set up input folder hierarchy
    input_dir = tmp_path / "tdms"
    output_dir = tmp_path / "dataframes"
    
    subject_input_dir = input_dir / "H1"
    os.makedirs(subject_input_dir, exist_ok=True)
    
    # Create empty TDMS files
    tdms_files = ["H1Slow1_180131.tdms", "H1Fast2_180131.tdms", "H1RT1_180131.tdms"]
    for f in tdms_files:
        with open(subject_input_dir / f, 'w') as fh:
            fh.write("")
            
    # Run process in dry_run mode
    results = ingest_subject_behavior(
        "H1",
        input_root=input_dir,
        output_root=output_dir,
        dry_run=True,
    )
    
    assert len(results) == 3
    assert results[0]['output'].endswith(
        "derivatives/meg-tokens/sub-H01/beh/sub-H01_task-tokens_run-2_desc-fast_beh.tsv"
    )
    assert results[1]['output'].endswith(
        "derivatives/meg-tokens/sub-H01/beh/sub-H01_task-tokens_run-1_desc-rt_beh.tsv"
    )
    assert results[2]['output'].endswith(
        "derivatives/meg-tokens/sub-H01/beh/sub-H01_task-tokens_run-1_desc-slow_beh.tsv"
    )
    
    # Dry run shouldn't write files
    assert not os.path.exists(output_dir / "derivatives")


def test_batch_process_writes_behavior_derivative(tmp_path, monkeypatch):
    input_dir = tmp_path / "tdms"
    output_dir = tmp_path / "dataframes"
    subject_input_dir = input_dir / "H1"
    os.makedirs(subject_input_dir, exist_ok=True)
    tdms_file = subject_input_dir / "H1Fast2_180131.tdms"
    tdms_file.write_text("")

    parsed_df = pd.DataFrame({
        "nTrialIndex": [1, 2],
        "sTrialClass": [1, 2],
        "nInitialTime": [10, 20],
        "nChoiceMade": [1, 0],
        "nCorrectChoice": [1, 2],
        "tGO": [1000, 2000],
        "tEnterTarget": [1400, 0],
        "tTrialEnd": [1700, 2100],
        "sTokenDirs": ["121", "212"],
        "tTime": [[1100, 1300], 0],
        "nProb": [[0.6, 0.8], 0],
    })

    monkeypatch.setattr("meg_tokens.workflows.behavior.parse_tdms_file", lambda _: parsed_df)

    results = ingest_subject_behavior(
        "H1",
        input_root=input_dir,
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
    sidecar_path = out_path.with_suffix(".json")
    assert out_path.exists()
    assert sidecar_path.exists()

    written = pd.read_csv(out_path, sep="\t")
    assert written["subject"].tolist() == ["H01", "H01"]
    assert written["condition"].tolist() == ["Fast", "Fast"]
    assert written["run"].tolist() == [2, 2]
    assert written["rawRT"].iloc[0] == 400
    assert pd.isna(written["rawRT"].iloc[1])
    assert written["isCorrect"].iloc[0] is True or written["isCorrect"].iloc[0] == True
