"""MEG domain functions with lazy public exports."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "load_and_filter_raw": ("meg_tokens.meg.preprocessing", "load_and_filter_raw"),
    "run_ica_rejection": ("meg_tokens.meg.preprocessing", "run_ica_rejection"),
    "convert_ctf_headshape_to_pos": ("meg_tokens.meg.preprocessing", "convert_ctf_headshape_to_pos"),
    "realign_epochs": ("meg_tokens.meg.preprocessing", "realign_epochs"),
    "build_epochs_with_metadata": ("meg_tokens.meg.epoching", "build_epochs_with_metadata"),
    "get_event_id": ("meg_tokens.meg.epoching", "get_event_id"),
    "mismatch_policy": ("meg_tokens.meg.epoching", "mismatch_policy"),
    "needs_go_reconstruction": ("meg_tokens.meg.epoching", "needs_go_reconstruction"),
    "exclude_unrecoverable_trials": ("meg_tokens.meg.epoching", "exclude_unrecoverable_trials"),
    "reconstruct_missing_go_events": ("meg_tokens.meg.epoching", "reconstruct_missing_go_events"),
    "save_epochs_and_events": ("meg_tokens.meg.epoching", "save_epochs_and_events"),
    "synchronize_events_and_behavior": ("meg_tokens.meg.epoching", "synchronize_events_and_behavior"),
    "compute_noise_covariance": ("meg_tokens.meg.sources", "compute_noise_covariance"),
    "setup_bem_solution": ("meg_tokens.meg.sources", "setup_bem_solution"),
    "setup_mixed_source_space": ("meg_tokens.meg.sources", "setup_mixed_source_space"),
    "compute_forward_solution": ("meg_tokens.meg.sources", "compute_forward_solution"),
    "build_inverse_operator": ("meg_tokens.meg.sources", "build_inverse_operator"),
    "apply_inverse_operator": ("meg_tokens.meg.sources", "apply_inverse_operator"),
    "morph_source_estimates": ("meg_tokens.meg.sources", "morph_source_estimates"),
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
    "count_start_pulses": ("meg_tokens.meg.raw_staging", "count_start_pulses"),
    "match_subject": ("meg_tokens.meg.raw_staging", "match_subject"),
    "match_media_to_behavior": ("meg_tokens.meg.raw_staging", "match_media_to_behavior"),
    "media_subject_prefix": ("meg_tokens.meg.raw_staging", "media_subject_prefix"),
    "write_meg_bids": ("meg_tokens.meg.bids_raw", "write_meg_bids"),
    "write_emptyroom_bids": ("meg_tokens.meg.bids_raw", "write_emptyroom_bids"),
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
