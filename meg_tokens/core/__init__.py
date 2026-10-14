"""Stable project models shared by domain, workflow, and CLI layers."""

from .config import ProjectConfig
from .entities import RunSpec, normalize_subject_id, parse_run_label
from .results import WorkflowResult
from .settings import (
    EPOCH_LOCKS,
    SOURCE_STAGES,
    STREAMS,
    ConnectivityConfig,
    DecodingConfig,
    DecompositionConfig,
    ERPConfig,
    CoregistrationConfig,
    EpochingConfig,
    HilbertConfig,
    LateralizedStatisticsConfig,
    PACConfig,
    PowerConfig,
    PreprocessingConfig,
    RawStagingConfig,
    SpectralConfig,
    SourceConfig,
    SpatialDecodingConfig,
    StatisticsConfig,
)

__all__ = [
    "ProjectConfig",
    "CoregistrationConfig",
    "EpochingConfig",
    "ERPConfig",
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
    "SOURCE_STAGES",
    "STREAMS",
    "SourceConfig",
    "SpatialDecodingConfig",
    "SpectralConfig",
    "StatisticsConfig",
    "RunSpec",
    "WorkflowResult",
    "normalize_subject_id",
    "parse_run_label",
]
