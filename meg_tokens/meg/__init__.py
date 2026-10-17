"""MEG domain functions with lazy public exports."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "read_ctf_headshape": ("meg_tokens.meg.coregistration", "read_ctf_headshape"),
    "coregister": ("meg_tokens.meg.coregistration", "coregister"),
    "build_trunk": ("meg_tokens.meg.preprocessing", "build_trunk"),
    "read_staged_raw": ("meg_tokens.meg.preprocessing", "read_staged_raw"),
    "run_stream": ("meg_tokens.meg.preprocessing", "run_stream"),
    "save_stream": ("meg_tokens.meg.preprocessing", "save_stream"),
    "artifact_events": ("meg_tokens.meg.preprocessing", "artifact_events"),
    "align_go": ("meg_tokens.meg.epoching", "align_go"),
    "epoch": ("meg_tokens.meg.epoching", "epoch"),
    "read_triggers": ("meg_tokens.meg.epoching", "read_triggers"),
    "save_epochs": ("meg_tokens.meg.epoching", "save_epochs"),
    "trial_metadata": ("meg_tokens.meg.epoching", "trial_metadata"),
    "compute_band_power": ("meg_tokens.features.time_frequency", "compute_band_power"),
    "rescale_baseline": ("meg_tokens.features.time_frequency", "rescale_baseline"),
    "DEFAULT_BANDS": ("meg_tokens.features.time_frequency", "DEFAULT_BANDS"),
    "align_and_pad_epochs": ("meg_tokens.features.erp", "align_and_pad_epochs"),
    "parcellate_source_estimates": ("meg_tokens.features.erp", "parcellate_source_estimates"),
    "export_neural_space": ("meg_tokens.features.erp", "export_neural_space"),
    "discover_raw_sessions": ("meg_tokens.meg.raw_staging", "discover_raw_sessions"),
    "discover_noise_session": ("meg_tokens.meg.raw_staging", "discover_noise_session"),
    "discover_headshape": ("meg_tokens.meg.raw_staging", "discover_headshape"),
    "load_behavior_runs": ("meg_tokens.meg.raw_staging", "load_behavior_runs"),
    "read_start_pulse_times": ("meg_tokens.meg.raw_staging", "read_start_pulse_times"),
    "fingerprint_candidates": ("meg_tokens.meg.raw_staging", "fingerprint_candidates"),
    "resolve_by_fingerprint": ("meg_tokens.meg.raw_staging", "resolve_by_fingerprint"),
    "match_subject": ("meg_tokens.meg.raw_staging", "match_subject"),
    "match_raw_to_behavior": ("meg_tokens.meg.raw_staging", "match_raw_to_behavior"),
    "write_meg_bids": ("meg_tokens.meg.meg_bids", "write_meg_bids"),
    "write_emptyroom_bids": ("meg_tokens.meg.meg_bids", "write_emptyroom_bids"),
    "write_anat_bids": ("meg_tokens.meg.anat_bids", "write_anat_bids"),
    "discover_anat": ("meg_tokens.meg.anat_bids", "discover_anat"),
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
