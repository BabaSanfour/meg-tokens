"""Filesystem-aware project workflows with lazy exports."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "analyze_behavior": (
        "meg_tokens.workflows.behavior_analysis",
        "analyze_behavior",
    ),
    "analyze_behavior_characterization": (
        "meg_tokens.workflows.behavior_characterization",
        "analyze_behavior_characterization",
    ),
    "ingest_behavior": (
        "meg_tokens.workflows.behavior_ingest",
        "ingest_behavior",
    ),
    "ingest_subject_behavior": (
        "meg_tokens.workflows.behavior_ingest",
        "ingest_subject_behavior",
    ),
    "epoch_subjects": ("meg_tokens.workflows.preprocessing", "epoch_subjects"),
    "plan_raw_staging": ("meg_tokens.workflows.raw_staging", "plan_raw_staging"),
    "apply_raw_staging": ("meg_tokens.workflows.raw_staging", "apply_raw_staging"),
    "preprocess_run": ("meg_tokens.workflows.preprocessing", "preprocess_run"),
    "preprocess_subjects": ("meg_tokens.workflows.preprocessing", "preprocess_subjects"),
    "extract_power_features": ("meg_tokens.workflows.power", "extract_power_features"),
    "source_models": ("meg_tokens.workflows.sources", "source_models"),
    "source_operators": ("meg_tokens.workflows.sources", "source_operators"),
    "source_operator_subjects": ("meg_tokens.workflows.sources", "source_operator_subjects"),
    "extract_spectral_features": ("meg_tokens.workflows.spectral", "extract_spectral_features"),
    "extract_hilbert_features": ("meg_tokens.workflows.hilbert", "extract_hilbert_features"),
    "extract_pac_features": ("meg_tokens.workflows.pac", "extract_pac_features"),
    "extract_connectivity_features": ("meg_tokens.workflows.connectivity", "extract_connectivity_features"),
    "run_group_statistics": ("meg_tokens.workflows.statistics", "run_group_statistics"),
    "run_lateralized_statistics": ("meg_tokens.workflows.statistics", "run_lateralized_statistics"),
    "run_decoding": ("meg_tokens.workflows.decoding", "run_decoding"),
    "run_spatial_decoding": ("meg_tokens.workflows.spatial_decoding", "run_spatial_decoding"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
