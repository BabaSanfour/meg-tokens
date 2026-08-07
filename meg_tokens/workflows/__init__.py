"""Filesystem-aware project workflows with lazy exports."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "analyze_behavior": ("meg_tokens.workflows.behavior", "analyze_behavior"),
    "analyze_behavior_extended": (
        "meg_tokens.workflows.behavior_extended",
        "analyze_behavior_extended",
    ),
    "ingest_behavior": ("meg_tokens.workflows.behavior", "ingest_behavior"),
    "ingest_subject_behavior": ("meg_tokens.workflows.behavior", "ingest_subject_behavior"),
    "epoch_subjects": ("meg_tokens.workflows.preprocessing", "epoch_subjects"),
    "extract_erp_features": ("meg_tokens.workflows.erp", "extract_erp_features"),
    "preprocess_run": ("meg_tokens.workflows.preprocessing", "preprocess_run"),
    "extract_power_features": ("meg_tokens.workflows.power", "extract_power_features"),
    "reconstruct_sources": ("meg_tokens.workflows.sources", "reconstruct_sources"),
    "extract_spectral_features": ("meg_tokens.workflows.spectral", "extract_spectral_features"),
    "extract_hilbert_features": ("meg_tokens.workflows.hilbert", "extract_hilbert_features"),
    "extract_pac_features": ("meg_tokens.workflows.pac", "extract_pac_features"),
    "extract_connectivity_features": ("meg_tokens.workflows.connectivity", "extract_connectivity_features"),
    "run_group_statistics": ("meg_tokens.workflows.statistics", "run_group_statistics"),
    "run_lateralized_statistics": ("meg_tokens.workflows.statistics", "run_lateralized_statistics"),
    "run_decoding": ("meg_tokens.workflows.decoding", "run_decoding"),
    "run_spatial_decoding": ("meg_tokens.workflows.spatial_decoding", "run_spatial_decoding"),
    "run_decomposition": ("meg_tokens.workflows.decomposition", "run_decomposition"),
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
