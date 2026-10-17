"""Event-related fields per condition, in sensors and in source regions, through
MNE-Python. Specification: ``docs/meg.md`` (Decisions, Event-related fields).

    Epochs[condition].average()                              (sensors, per run)
    -> apply_inverse, sLORETA at nave = 1, three orientations
    -> cortex: normal component
    -> extract_label_time_course, mode="mean_flip"           (signed, per region)
    -> runs combined, weighted by their trials

The inverse is linear, so the inverse of a run's average is the average of its
trials' estimates, and a weighted mean over runs, each through its own
operator, is the subject's average. sLORETA's noise normalisation scales with
the number of trials averaged; every average is inverted at ``nave=1``, the
kernel band power uses, so conditions with different trial counts compare.
"""

from __future__ import annotations

import mne
import numpy as np

from meg_tokens.meg.sources import LAMBDA2, METHOD

CLASSES = ("easy", "ambiguous", "misleading")
# Response side 1 is the right hand and target, side 2 the left (docs/behavior.md).
SIDES = {1: "right", 2: "left"}
CONDITIONS = tuple(
    f"{speed}-{trial_class}-{side}"
    for speed in ("fast", "slow")
    for trial_class in CLASSES
    for side in SIDES.values()
)


def condition_masks(metadata) -> dict[str, np.ndarray]:
    """Which epochs belong to each of the twelve conditions; unclassified trials belong to none."""
    speed = metadata["condition"].str.lower().to_numpy()
    trial_class = metadata["trial_class_name"].to_numpy()
    side = metadata["choice_side"].map(SIDES).to_numpy()
    return {
        name: (speed == name.split("-")[0]) & (trial_class == name.split("-")[1]) & (side == name.split("-")[2])
        for name in CONDITIONS
    }


def region_time_courses(
    evoked: mne.Evoked, inv: mne.minimum_norm.InverseOperator, labels: list[mne.Label],
) -> np.ndarray:
    """An average's signed time course in each region: sLORETA normal component, sign-flipped mean.

    ``inv`` is prepared at ``nave=1`` (``mne.minimum_norm.prepare_inverse_operator``).
    MNE refuses a normal orientation on a mixed source space, so the estimate is
    taken with three orientations and the cortex projected onto its normals.
    ``mean_flip`` flips each source by its normal's agreement with the region's
    dominant orientation before averaging, so opposite walls of a sulcus add
    rather than cancel; the sign of a region's time course is a convention of
    its anatomy. A region with no source is NaN.
    """
    src = inv["src"]
    single = evoked.copy()
    single.nave = 1
    estimate = mne.minimum_norm.apply_inverse(
        single, inv, LAMBDA2, METHOD, pick_ori="vector", prepared=True, verbose=False,
    )
    normal = estimate.surface().project("normal", src[:2])[0]
    courses = mne.extract_label_time_course(normal, labels, src[:2], mode="mean_flip", allow_empty=True, verbose=False)
    empty = [len(label.restrict(src[:2]).vertices) == 0 for label in labels]
    courses[empty] = np.nan
    return courses


__all__ = ["CLASSES", "SIDES", "CONDITIONS", "condition_masks", "region_time_courses"]
