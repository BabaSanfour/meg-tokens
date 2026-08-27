import os

from meg_tokens.io import DerivativeLayout
from meg_tokens.meg.meg_bids import preprocessed_meg_runs


def test_preprocessed_discovery_prefers_the_derivative_contract(tmp_path):
    raw_dir = (
        tmp_path
        / "derivatives"
        / "sub-H01"
        / "meg"
    )
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = raw_dir / "sub-H01_task-tokens_run-1_proc-filt_desc-slow_raw.fif"
    raw_path.write_text("")

    assert [path for _, path in preprocessed_meg_runs(DerivativeLayout(tmp_path), "H1")] == [
        raw_path
    ]


def test_preprocessed_discovery_prefers_clean_over_filt(tmp_path):
    """Both exist when a run was preprocessed with and without ICA."""
    layout = DerivativeLayout(tmp_path)
    written = {}
    for processing in ("filt", "clean"):
        path = layout.preprocessed_raw(
            subject="H01", run="1", condition="Fast", processing=processing
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        written[processing] = path

    assert [path for _, path in preprocessed_meg_runs(layout, "H01")] == [
        written["clean"]
    ]
