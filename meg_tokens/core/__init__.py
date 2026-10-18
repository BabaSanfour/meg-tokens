"""Stable project models shared by domain, workflow, and CLI layers."""

from .config import ProjectConfig
from .entities import RunSpec, normalize_subject_id, parse_run_label
from .results import WorkflowResult
from .settings import (
    EPOCH_LOCKS,
    STREAMS,
    ConnectivityConfig,
    DecodingConfig,
    CoregistrationConfig,
    EpochingConfig,
    HilbertConfig,
    LateralizedStatisticsConfig,
    PACConfig,
    PowerConfig,
    PreprocessingConfig,
    RawStagingConfig,
    SpectralConfig,
    BandChoiceConfig,
    SpatialDecodingConfig,
    StatisticsConfig,
    TRAJECTORY_MEASURES,
    TRAJECTORY_SPACES,
    TrajectoryConfig,
)

__all__ = [
    "ProjectConfig",
    "CoregistrationConfig",
    "EpochingConfig",
    "ConnectivityConfig",
    "DecodingConfig",
    "HilbertConfig",
    "LateralizedStatisticsConfig",
    "PACConfig",
    "PowerConfig",
    "PreprocessingConfig",
    "RawStagingConfig",
    "EPOCH_LOCKS",
    "STREAMS",
    "SpatialDecodingConfig",
    "SpectralConfig",
    "BandChoiceConfig",
    "StatisticsConfig",
    "TRAJECTORY_MEASURES",
    "TRAJECTORY_SPACES",
    "TrajectoryConfig",
    "RunSpec",
    "WorkflowResult",
    "normalize_subject_id",
    "parse_run_label",
]
