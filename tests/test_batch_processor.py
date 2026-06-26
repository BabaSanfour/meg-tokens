import os
import pytest
from meg_tokens.utils.batch_processor import process_subject_tdms, FILENAME_RE

# Test regex match groupings
@pytest.mark.parametrize("filename,expected", [
    ("H1Slow1_180131.tdms", ("Slow", "1", "180131")),
    ("H02Fast4_180213.tdms", ("Fast", "4", "180213")),
    ("H32RT2_190214.tdms", ("RT", "2", "190214")),
])
def test_filename_regex(filename, expected):
    match = FILENAME_RE.match(filename)
    assert match is not None
    assert match.groups() == expected

def test_batch_process_dry_run(tmp_path):
    # Set up mock input folder hierarchy
    input_dir = tmp_path / "tdms"
    output_dir = tmp_path / "dataframes"
    
    subject_input_dir = input_dir / "H1"
    os.makedirs(subject_input_dir, exist_ok=True)
    
    # Create empty mock TDMS files
    mock_files = ["H1Slow1_180131.tdms", "H1Fast2_180131.tdms", "H1RT1_180131.tdms"]
    for f in mock_files:
        with open(subject_input_dir / f, 'w') as fh:
            fh.write("")
            
    # Run process in dry_run mode
    results = process_subject_tdms("H1", str(input_dir), str(output_dir), dry_run=True)
    
    assert len(results) == 3
    assert results[0]['output'] == "H1_Fast2.csv"
    assert results[1]['output'] == "H1_RT1.csv"
    assert results[2]['output'] == "H1_Slow1.csv"
    
    # Dry run shouldn't write files
    assert not os.path.exists(output_dir / "H1" / "H1_Slow1.csv")
