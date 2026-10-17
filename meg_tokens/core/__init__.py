"""Stable project models shared by domain, workflow, and CLI layers."""

from .config import ProjectConfig
from .entities import RunSpec, normalize_subject_id, parse_run_label
from .results import WorkflowResult
from .settings import (
    EPOCH_LOCKS,
    STREAMS,
    ConnectivityConfig,
    DecodingConfig,
    DecompositionConfig,
    CoregistrationConfig,
    EpochingConfig,
    HilbertConfig,
    LateralizedStatisticsConfig,
    PACConfig,
    PowerConfig,
    PreprocessingConfig,
    RawStagingConfig,
    SpectralConfig,
    SpatialDecodingConfig,
    StatisticsConfig,
)

__all__ = [
    "ProjectConfig",
    "CoregistrationConfig",
    "EpochingConfig",
    "ConnectivityConfig",
    "DecodingConfig",
    "DecompositionConfig",
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
    "StatisticsConfig",
    "RunSpec",
    "WorkflowResult",
    "normalize_subject_id",
    "parse_run_label",
]
