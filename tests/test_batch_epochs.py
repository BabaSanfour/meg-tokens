import pytest
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
    raw_path = raw_dir / "sub-H01_task-tokens_run-1_proc-ica_desc-slow_raw.fif"
    raw_path.write_text("")

    assert [path for _, path in preprocessed_meg_runs(DerivativeLayout(tmp_path), "H1", stream="ica")] == [
        raw_path
    ]


def test_preprocessed_discovery_requires_a_named_stream(tmp_path):
    """Preprocessing writes one derivative per cleaner, so there is no "the"
    preprocessed run: ``stream`` is required by the signature, not defaulted
    and then checked."""
    layout = DerivativeLayout(tmp_path)
    for stream in ("dss", "ica", "ssp"):
        path = layout.preprocessed_raw(
            subject="H01", run="1", condition="Fast", processing=stream
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    with pytest.raises(TypeError, match="stream"):
        preprocessed_meg_runs(layout, "H01")

    for stream in ("dss", "ica", "ssp"):
        found = preprocessed_meg_runs(layout, "H01", stream=stream)
        assert [p.name for _, p in found] == [
            f"sub-H01_task-tokens_run-1_proc-{stream}_desc-fast_raw.fif"
        ]

