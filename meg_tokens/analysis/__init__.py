"""Group analysis estimators with lazy public exports."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "compute_time_resolved_decoding": ("meg_tokens.analysis.decoding", "compute_time_resolved_decoding"),
    "compute_decoding_permutations": ("meg_tokens.analysis.decoding", "compute_decoding_permutations"),
    "compute_permutation_t_test": ("meg_tokens.analysis.statistics", "compute_permutation_t_test"),
    "compute_cluster_permutation_test": ("meg_tokens.analysis.statistics", "compute_cluster_permutation_test"),
    "fit_condition_pca": ("meg_tokens.analysis.decomposition", "fit_condition_pca"),
    "NeuralPCAResult": ("meg_tokens.analysis.decomposition", "NeuralPCAResult"),
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
