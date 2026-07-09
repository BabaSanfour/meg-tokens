"""Neural feature extraction operations with lazy public exports."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "align_and_pad_epochs": ("meg_tokens.features.erp", "align_and_pad_epochs"),
    "parcellate_source_estimates": ("meg_tokens.features.erp", "parcellate_source_estimates"),
    "select_source_feature_data": ("meg_tokens.features.erp", "select_source_feature_data"),
    "compute_band_power": ("meg_tokens.features.time_frequency", "compute_band_power"),
    "compute_hilbert_band_features": ("meg_tokens.features.time_frequency", "compute_hilbert_band_features"),
    "compute_psd": ("meg_tokens.features.time_frequency", "compute_psd"),
    "fit_specparam": ("meg_tokens.features.time_frequency", "fit_specparam"),
    "compute_spectral_connectivity": ("meg_tokens.features.connectivity", "compute_spectral_connectivity"),
    "modulation_index": ("meg_tokens.features.pac", "modulation_index"),
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
