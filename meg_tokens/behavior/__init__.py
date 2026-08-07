"""Behavior parsing and metrics with lazy public exports."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "align_design_profile_to_runtime": ("meg_tokens.behavior.success_probability", "align_design_profile_to_runtime"),
    "calculate_motor_baseline": ("meg_tokens.behavior.metrics", "calculate_motor_baseline"),
    "classify_design_profile": ("meg_tokens.behavior.success_probability", "classify_design_profile"),
    "classify_design_profile_with_rule": ("meg_tokens.behavior.success_probability", "classify_design_profile_with_rule"),
    "calculate_decision_times": ("meg_tokens.behavior.metrics", "calculate_decision_times"),
    "compare_fast_slow": ("meg_tokens.behavior.metrics", "compare_fast_slow"),
    "design_spd_at_decision": ("meg_tokens.behavior.success_probability", "design_spd_at_decision"),
    "analyze_trial_classes": ("meg_tokens.behavior.metrics", "analyze_trial_classes"),
    "analyze_logged_spd": ("meg_tokens.behavior.metrics", "analyze_logged_spd"),
    "compare_correct_error": ("meg_tokens.behavior.metrics", "compare_correct_error"),
    "analyze_post_error_slowing": ("meg_tokens.behavior.metrics", "analyze_post_error_slowing"),
    "behavior_group_statistics": ("meg_tokens.behavior.metrics", "behavior_group_statistics"),
    "paired_subject_statistics": ("meg_tokens.behavior.metrics", "paired_subject_statistics"),
    "logged_spd_at_decision": ("meg_tokens.behavior.metrics", "logged_spd_at_decision"),
    "probability_at_decision": ("meg_tokens.behavior.success_probability", "probability_at_decision"),
    "implied_target_counts": ("meg_tokens.behavior.success_probability", "implied_target_counts"),
    "success_probability": ("meg_tokens.behavior.success_probability", "success_probability"),
    "success_probability_profile": ("meg_tokens.behavior.success_probability", "success_probability_profile"),
    "FILENAME_RE": ("meg_tokens.behavior.tdms", "FILENAME_RE"),
    "TRIAL_COLUMNS": ("meg_tokens.behavior.tdms", "TRIAL_COLUMNS"),
    "TdmsRunInfo": ("meg_tokens.behavior.tdms", "TdmsRunInfo"),
    "add_run_metadata": ("meg_tokens.behavior.tdms", "add_run_metadata"),
    "parse_single_trial": ("meg_tokens.behavior.tdms", "parse_single_trial"),
    "parse_tdms_filename": ("meg_tokens.behavior.tdms", "parse_tdms_filename"),
    "parse_tdms_file": ("meg_tokens.behavior.tdms", "parse_tdms_file"),
    "started_trials": ("meg_tokens.behavior.tdms", "started_trials"),
    "validate_behavior_dataframe": ("meg_tokens.behavior.tdms", "validate_behavior_dataframe"),
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
