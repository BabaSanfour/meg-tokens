"""The production cleaners, scored against one trunk on one battery.

This is the comparison B5 decided without ever making: the DSS chain and ICA
each run on the same shared trunk, from the same detected events, and scored
by the same battery. No arm is measured on terms of its own.

It answers a different question from the parameter grid beside it. The grid
asks which settings a DSS stage should use; this asks whether the DSS stream is
worth running at all next to the ICA it was built to challenge. They share the
scoring battery, which is why they share a package.
"""

from __future__ import annotations

import time

import mne
import numpy as np

from meg_tokens.meg.preprocessing import (
    CLEANERS,
    ECG_CHANNEL,
    OCULAR_BAND,
    VEOG_CHANNEL,
)

from scripts.qc.dss_param_search.metrics import score_cleaning

#: In the order they are cheapest to run, so a truncated task still carries the
#: incumbent. ICA is the incumbent and must never be the arm that got dropped.
STREAMS = ("ica", "dss")


def reference_signals(trunk: mne.io.BaseRaw) -> dict[str, np.ndarray]:
    """The channels a removed subspace is checked against."""
    return {
        "ecg": trunk.copy().pick([ECG_CHANNEL]).get_data()[0],
        "veog": (
            trunk.copy()
            .pick([VEOG_CHANNEL])
            .filter(*OCULAR_BAND, fir_design="firwin", verbose=False)
            .get_data()[0]
        ),
    }


def compare_streams(
    trunk: mne.io.BaseRaw,
    events: dict[str, np.ndarray],
    picks: np.ndarray,
    *,
    streams: tuple[str, ...] = STREAMS,
) -> list[dict]:
    """Run each cleaner on the trunk and score it, one row per stream.

    Each stream's cleaned recording and artifact estimate are freed as soon as
    they are scored. Holding three of each alongside the trunk is about six
    copies of a 270-channel 315-second recording, which is where the arena's
    fifteen-gigabyte surprise came from.
    """
    references = reference_signals(trunk)
    rows: list[dict] = []
    for stream in streams:
        started = time.monotonic()
        try:
            cleaned, n_removed, detail = CLEANERS[stream](trunk, events)
        except Exception as error:  # a named failure, not a silent skip
            rows.append({
                "stream": stream, "failed": repr(error),
                "seconds": round(time.monotonic() - started, 1),
            })
            continue
        row = {
            "stream": stream,
            "n_dimensions_removed": int(n_removed),
            "rank": int(len(picks) - n_removed),
            "detail": detail,
            "seconds": round(time.monotonic() - started, 1),
        }
        row.update(score_cleaning(trunk, cleaned, picks, events, references))
        rows.append(row)
        del cleaned
    return rows
