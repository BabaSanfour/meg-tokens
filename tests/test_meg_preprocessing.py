from meg_tokens.io import DerivativeLayout
from meg_tokens.meg.preprocessing import STREAMS, stream_paths


def test_every_stream_gets_a_distinct_path(tmp_path):
    """Each cleaner writes its own derivative for one recording."""
    layout = DerivativeLayout(tmp_path)
    paths = [stream_paths(layout, "H1", "Slow", 1, s)["cleaned"] for s in STREAMS]
    assert len(set(paths)) == len(STREAMS)
