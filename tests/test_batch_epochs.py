import os

from meg_tokens.utils.batch_epochs import infer_run_id_from_raw, find_raw_files


def test_infer_run_id_from_raw_bids_and_legacy_names():
    assert infer_run_id_from_raw("sub-H01_task-tokens_run-1_proc-filt_desc-slow_raw.fif") == "1"
    assert infer_run_id_from_raw("H01_Slow2_filt_raw.fif") == "Slow2"


def test_find_raw_files_prefers_derivative_contract(tmp_path):
    raw_dir = (
        tmp_path
        / "derivatives"
        / "meg-tokens"
        / "sub-H01"
        / "meg"
    )
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = raw_dir / "sub-H01_task-tokens_run-1_proc-filt_desc-slow_raw.fif"
    raw_path.write_text("")

    assert find_raw_files(str(tmp_path), "H1") == [str(raw_path)]
