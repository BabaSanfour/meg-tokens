"""What each DSS stage's search space is, and why each axis is in it.

Every stage's grid contains its production configuration, so each stage has a
reference row rather than a field of candidates with nothing to beat.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Any

from meg_tokens.meg.preprocessing import (
    BLINK_WINDOW,
    DSS_KNEE_MIN_RATIO,
    DSS_N_COMPONENTS,
    DSS_SELECTION_THRESHOLD,
    MUSCLE_BAND,
    OCULAR_SCALE_MULTIPLE,
    QRS_WINDOW,
)

STAGES = ("qrs", "blink", "muscle")

#: The bias-side axes. These change *which* subspace DSS finds.
#:
#: ``window`` is the cycle-average template. Too short truncates the artifact
#: -- the cardiac T wave runs to about 400 ms -- and too long estimates the
#: template from samples that are not artifact.
#:
#: ``scale_multiple`` is the ocular detector's MAD threshold. It changes the
#: *event set*, so it changes the template; the blink window reading flat
#: across a 4x range of widths is as consistent with impure events as with an
#: insensitive window, and only this axis separates the two.
#:
#: ``freq_band``/``order`` are the muscle bias's filter. The stage selects zero
#: components on some recordings, and the standing explanation is that a
#: 110-140 Hz band has no transition room under a 150 Hz low-pass -- which is a
#: statement about filter order as much as about the band.
BIAS_AXES: dict[str, dict[str, tuple]] = {
    "qrs": {
        "window": ((-0.1, 0.2), (-0.2, 0.4), (-0.2, 0.5), (-0.3, 0.7)),
    },
    "blink": {
        "window": ((-0.1, 0.3), (-0.2, 0.5), (-0.3, 0.7)),
        "scale_multiple": (2.5, 3.0, 3.5),
    },
    "muscle": {
        "freq_band": ((110.0, 140.0), (100.0, 148.0), (80.0, 148.0), (60.0, 148.0)),
        "order": (4, 8),
    },
}

#: The selection-side axes. These change *how many* dimensions come out.
#:
#: ``n_components`` truncates the eigenvalue spectrum before selection, so 40
#: decides which components look like outliers against 39 others rather than
#: against 269.
#:
#: ``knee_min_ratio`` is the gate that returns zero: the knee arm requires an
#: eigenvalue drop of this factor to fire at all. A muscle stage selecting
#: *exactly* zero means neither selection arm voted, which is a different
#: failure from a band that finds nothing.
SELECTION_AXES: dict[str, dict[str, tuple]] = {
    "qrs": {
        "n_components": (DSS_N_COMPONENTS, None),
        "selection_threshold": (2.0, DSS_SELECTION_THRESHOLD),
    },
    "blink": {
        "n_components": (DSS_N_COMPONENTS, None),
    },
    "muscle": {
        "knee_min_ratio": (1.5, DSS_KNEE_MIN_RATIO),
    },
}

#: What ``clean_dss`` runs today. Each stage's grid must contain its own row,
#: and the search fixes upstream stages here while it varies one stage.
PRODUCTION: dict[str, dict[str, Any]] = {
    "qrs": {
        "window": QRS_WINDOW,
        "n_components": DSS_N_COMPONENTS,
        "selection_threshold": DSS_SELECTION_THRESHOLD,
    },
    "blink": {
        "window": BLINK_WINDOW,
        "scale_multiple": OCULAR_SCALE_MULTIPLE,
        "n_components": DSS_N_COMPONENTS,
    },
    "muscle": {
        "freq_band": MUSCLE_BAND,
        "order": 4,
        "knee_min_ratio": DSS_KNEE_MIN_RATIO,
    },
}


@dataclass(frozen=True)
class Config:
    """One point in one stage's grid."""

    stage: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"unknown stage: {self.stage}")
        expected = set(PRODUCTION[self.stage])
        if set(self.params) != expected:
            missing = sorted(expected - set(self.params))
            extra = sorted(set(self.params) - expected)
            raise ValueError(f"{self.stage} params missing {missing}, unexpected {extra}")

    @property
    def is_production(self) -> bool:
        """Whether this row is what ``clean_dss`` runs today."""
        return self.params == PRODUCTION[self.stage]

    @property
    def label(self) -> str:
        """A stable identifier, so a row can be matched across subjects."""
        parts = [f"{key}={self.params[key]}" for key in sorted(self.params)]
        return f"{self.stage}|" + "|".join(parts)

    @property
    def bias_params(self) -> dict[str, Any]:
        """The subset the bias is built from."""
        return {k: v for k, v in self.params.items() if k in BIAS_AXES[self.stage]}

    @property
    def selection_params(self) -> dict[str, Any]:
        """The subset passed to :func:`dss_subtract`."""
        return {k: v for k, v in self.params.items() if k in SELECTION_AXES[self.stage]}


def stage_grid(stage: str) -> list[Config]:
    """Every configuration for one stage, production included, no duplicates."""
    axes = {**BIAS_AXES[stage], **SELECTION_AXES[stage]}
    names = sorted(axes)
    configs = [
        Config(stage=stage, params=dict(zip(names, values)))
        for values in itertools.product(*(axes[name] for name in names))
    ]
    if not any(config.is_production for config in configs):
        raise ValueError(f"{stage} grid does not contain its production configuration")
    return configs


def full_grid() -> list[Config]:
    """Every configuration for every stage, in stage order."""
    return [config for stage in STAGES for config in stage_grid(stage)]


def production_config(stage: str) -> Config:
    """The row a stage is measured against, and the input the next stage gets."""
    return Config(stage=stage, params=dict(PRODUCTION[stage]))


def folds(subjects: list[str], *, n_search: int = 20, seed: int = 0) -> dict[str, str]:
    """Split the cohort into a fold to choose on and a fold to confirm on.

    Choosing one configuration from fifty on the same subjects it is then
    reported over is selection on the test set. The split is a deterministic
    shuffle rather than the first N subject numbers, because subject order is
    acquisition order and correlates with everything that drifted over the
    study.
    """
    ordered = sorted(subjects)
    if n_search >= len(ordered):
        raise ValueError(f"n_search={n_search} leaves no confirmation fold of {len(ordered)}")
    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    search = set(shuffled[:n_search])
    return {subject: ("search" if subject in search else "confirm") for subject in ordered}
