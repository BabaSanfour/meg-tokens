"""Stable project models shared by domain, workflow, and CLI layers."""

from .config import ProjectConfig
from .entities import RunSpec, normalize_subject_id, parse_run_label
from .results import WorkflowResult
from .settings import (
    SOURCE_STAGES,
    ConnectivityConfig,
    DecodingConfig,
    DecompositionConfig,
    ERPConfig,
    EpochingConfig,
    HilbertConfig,
    LateralizedStatisticsConfig,
    PACConfig,
    PowerConfig,
    PreprocessingConfig,
    SpectralConfig,
    SourceConfig,
    SpatialDecodingConfig,
    StatisticsConfig,
)

__all__ = [
    "ProjectConfig",
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
    "SOURCE_STAGES",
    "SourceConfig",
    "SpatialDecodingConfig",
    "SpectralConfig",
    "StatisticsConfig",
    "RunSpec",
    "WorkflowResult",
    "normalize_subject_id",
    "parse_run_label",
]
