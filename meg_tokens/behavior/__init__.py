"""Behavior parsing and metrics with lazy public exports."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "calculate_motor_baseline": ("meg_tokens.behavior.metrics", "calculate_motor_baseline"),
    "calculate_decision_times": ("meg_tokens.behavior.metrics", "calculate_decision_times"),
    "compare_fast_slow": ("meg_tokens.behavior.metrics", "compare_fast_slow"),
    "analyze_trial_classes": ("meg_tokens.behavior.metrics", "analyze_trial_classes"),
    "compare_correct_error": ("meg_tokens.behavior.metrics", "compare_correct_error"),
    "analyze_post_error_slowing": ("meg_tokens.behavior.metrics", "analyze_post_error_slowing"),
    "FILENAME_RE": ("meg_tokens.behavior.tdms", "FILENAME_RE"),
    "TRIAL_COLUMNS": ("meg_tokens.behavior.tdms", "TRIAL_COLUMNS"),
    "TdmsRunInfo": ("meg_tokens.behavior.tdms", "TdmsRunInfo"),
    "add_run_metadata": ("meg_tokens.behavior.tdms", "add_run_metadata"),
    "parse_single_trial": ("meg_tokens.behavior.tdms", "parse_single_trial"),
    "parse_tdms_filename": ("meg_tokens.behavior.tdms", "parse_tdms_filename"),
    "parse_tdms_file": ("meg_tokens.behavior.tdms", "parse_tdms_file"),
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
