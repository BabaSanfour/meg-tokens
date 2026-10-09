import json
import numpy as np
import pytest
from meg_tokens.io import DerivativeLayout
from meg_tokens.meg.preprocessing import STREAMS, stream_paths


def test_every_stream_gets_a_distinct_path(tmp_path):
    """Each cleaner writes its own derivative for one recording."""
    layout = DerivativeLayout(tmp_path)
    paths = [stream_paths(layout, "H1", "Slow", 1, s)["cleaned"] for s in STREAMS]
    assert len(set(paths)) == len(STREAMS)


def test_a_streams_detail_survives_json_without_a_stringifying_fallback():
    """The sidecar is the audit record for what a stream removed.

    `find_bads_eog`/`find_bads_ecg` return numpy scalars, and `save_stream`
    used to carry a `default=str` fallback -- so an `int64` component index was
    written as `"3"` and nothing said so. The fallback is gone, which makes a
    stray numpy type raise here rather than corrupt a record.
    """
    import json

    import numpy as np

    from meg_tokens.meg.preprocessing import StreamResult

    result = StreamResult(
        stream="ica", cleaned=None, artifact=None, n_dimensions_removed=3,
        detail={"n_components": 20, "excluded": [int(i) for i in np.array([3, 7, 11])]},
    )
    dumped = json.dumps(result.detail, sort_keys=True)
    assert json.loads(dumped)["excluded"] == [3, 7, 11]
    assert all(isinstance(index, int) for index in json.loads(dumped)["excluded"])

    with pytest.raises(TypeError, match="not JSON serializable"):
        json.dumps({"excluded": list(np.array([3, 7]))})
