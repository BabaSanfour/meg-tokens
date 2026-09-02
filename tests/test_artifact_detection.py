"""Contracts for threshold derivation and detector validation."""

from __future__ import annotations

import numpy as np
import pytest

mne = pytest.importorskip("mne")

from meg_tokens.meg.artifact_detection import (
    enrichment_db,
    matched_clean_spans,
    robust_scale,
)

SFREQ = 600.0


def _raw(n_seconds: float = 90.0, n_channels: int = 8, noise: float = 1e-13):
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n_channels, int(SFREQ * n_seconds))) * noise
    info = mne.create_info(
        [f"MEG{i:03d}" for i in range(n_channels)], SFREQ, ["mag"] * n_channels
    )
    return mne.io.RawArray(data, info, verbose=False), np.arange(n_channels)


class TestRobustScale:
    def test_the_scale_tracks_the_noise_level(self):
        quiet, picks = _raw(noise=1e-13)
        loud, _ = _raw(noise=4e-13)
        assert robust_scale(loud, picks) / robust_scale(quiet, picks) == pytest.approx(
            4.0, rel=0.1
        )

    def test_slow_drift_does_not_inflate_it(self):
        """Differencing high-passes the recording, which is the whole point:
        the scale must reflect sample-to-sample noise, not signal amplitude."""
        flat, picks = _raw()
        drifting, _ = _raw()
        drifting._data += (np.sin(2 * np.pi * 0.05 * drifting.times) * 5e-11)[None, :]
        assert robust_scale(drifting, picks) == pytest.approx(
            robust_scale(flat, picks), rel=0.05
        )

    def test_a_recording_with_no_variation_raises(self):
        raw, picks = _raw()
        raw._data[:] = 0.0
        with pytest.raises(ValueError, match="zero scale"):
            robust_scale(raw, picks)


class TestMatchedCleanSpans:
    def test_comparison_spans_match_the_flagged_durations(self):
        """Without duration matching a detector could raise its enrichment by
        flagging shorter or longer segments rather than better ones."""
        bad = [(100, 200), (400, 460)]
        clean = matched_clean_spans(bad, 2000)
        assert [b - a for a, b in clean] == [b - a for a, b in bad]

    def test_comparison_spans_never_overlap_a_flagged_one(self):
        bad = [(100, 200), (400, 460)]
        for start, stop in matched_clean_spans(bad, 2000):
            assert all(stop <= a or start >= b for a, b in bad)


class TestEnrichment:
    def _with_band_bursts(self):
        raw, picks = _raw(n_seconds=60.0)
        width = int(SFREQ * 0.5)
        onsets = [int(SFREQ * s) for s in (10, 20, 30)]
        for onset in onsets:
            t = np.arange(width) / SFREQ
            raw._data[:, onset : onset + width] += (
                np.sin(2 * np.pi * 125.0 * t)[None, :] * 8e-13
            )
        annotations = mne.Annotations(
            [o / SFREQ for o in onsets], [0.5] * len(onsets), ["BAD_muscle"] * len(onsets)
        )
        return raw, picks, annotations

    def test_flagging_real_in_band_bursts_gives_positive_enrichment(self):
        raw, picks, annotations = self._with_band_bursts()
        assert enrichment_db(raw, picks, annotations) > 6.0

    def test_flagging_arbitrary_spans_does_not(self):
        """The test must be able to fail, or it certifies any detector."""
        raw, picks, _ = self._with_band_bursts()
        elsewhere = mne.Annotations([2.0, 40.0, 50.0], [0.5] * 3, ["BAD_muscle"] * 3)
        assert enrichment_db(raw, picks, elsewhere) < 3.0
