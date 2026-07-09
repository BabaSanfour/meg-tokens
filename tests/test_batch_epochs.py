import os

from meg_tokens.io import DerivativeLayout


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

    assert DerivativeLayout(tmp_path).raw_files(subject="H1") == [raw_path]
