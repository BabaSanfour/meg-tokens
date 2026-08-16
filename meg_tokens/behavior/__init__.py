"""Layered behavioral schema, feature, mathematics, and analysis APIs."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "calculate_motor_baseline": ("meg_tokens.behavior.features", "calculate_motor_baseline"),
    "calculate_motor_baseline_by_side": (
        "meg_tokens.behavior.features", "calculate_motor_baseline_by_side"),
    "classify_design_profile": ("meg_tokens.behavior.math.probability", "classify_design_profile"),
    "calculate_decision_times": ("meg_tokens.behavior.features", "calculate_decision_times"),
    "compare_fast_slow": ("meg_tokens.behavior.analyses.performance", "compare_fast_slow"),
    "analyze_trial_classes": ("meg_tokens.behavior.analyses.performance", "analyze_trial_classes"),
    "compare_correct_error": ("meg_tokens.behavior.analyses.performance", "compare_correct_error"),
    "analyze_post_error_slowing": ("meg_tokens.behavior.analyses.performance", "analyze_post_error_slowing"),
    "behavior_group_statistics": ("meg_tokens.behavior.analyses.performance", "behavior_group_statistics"),
    "choice_history": ("meg_tokens.behavior.analyses.sequential", "choice_history"),
    "choice_history_statistics": ("meg_tokens.behavior.analyses.sequential", "choice_history_statistics"),
    "choice_side_statistics": ("meg_tokens.behavior.analyses.design_effects", "choice_side_statistics"),
    "choice_side_summary": ("meg_tokens.behavior.analyses.design_effects", "choice_side_summary"),
    "comparison_statistics": ("meg_tokens.behavior.analyses.individual", "comparison_statistics"),
    "conditional_accuracy_functions": ("meg_tokens.behavior.analyses.evidence", "conditional_accuracy_functions"),
    "conditional_accuracy_statistics": ("meg_tokens.behavior.analyses.evidence", "conditional_accuracy_statistics"),
    "condition_class_cells": ("meg_tokens.behavior.analyses.design_effects", "condition_class_cells"),
    "condition_class_statistics": ("meg_tokens.behavior.analyses.design_effects", "condition_class_statistics"),
    "condition_order_effects": ("meg_tokens.behavior.analyses.design_effects", "condition_order_effects"),
    "condition_order_statistics": ("meg_tokens.behavior.analyses.design_effects", "condition_order_statistics"),
    "continuous_evidence_effects": ("meg_tokens.behavior.analyses.evidence", "continuous_evidence_effects"),
    "continuous_evidence_statistics": ("meg_tokens.behavior.analyses.evidence", "continuous_evidence_statistics"),
    "criterion_decline": ("meg_tokens.behavior.analyses.evidence", "criterion_decline"),
    "criterion_decline_statistics": ("meg_tokens.behavior.analyses.evidence", "criterion_decline_statistics"),
    "decision_time_distributions": ("meg_tokens.behavior.analyses.distributions", "decision_time_distributions"),
    "decision_time_distribution_statistics": ("meg_tokens.behavior.analyses.distributions", "decision_time_distribution_statistics"),
    "distribution_summary": ("meg_tokens.behavior.analyses.distributions", "distribution_summary"),
    "evidence_after_tokens": ("meg_tokens.behavior.math.evidence", "evidence_after_tokens"),
    "extreme_decision_times": ("meg_tokens.behavior.analyses.design_effects", "extreme_decision_times"),
    "individual_correlations": ("meg_tokens.behavior.analyses.individual", "individual_correlations"),
    "individual_profile": ("meg_tokens.behavior.analyses.individual", "individual_profile"),
    "lapse_statistics": ("meg_tokens.behavior.analyses.design_effects", "lapse_statistics"),
    "lapse_summary": ("meg_tokens.behavior.analyses.design_effects", "lapse_summary"),
    "log_posterior_odds": ("meg_tokens.behavior.math.evidence", "log_posterior_odds"),
    "one_sample_statistics": ("meg_tokens.behavior.math.inference", "one_sample_statistics"),
    "parse_token_directions": ("meg_tokens.behavior.schema", "parse_token_directions"),
    "post_error_statistics": ("meg_tokens.behavior.analyses.sequential", "post_error_statistics"),
    "repeated_measures_anova": ("meg_tokens.behavior.math.inference", "repeated_measures_anova"),
    "reverse_correlation": ("meg_tokens.behavior.analyses.evidence", "reverse_correlation"),
    "reverse_correlation_statistics": ("meg_tokens.behavior.analyses.evidence", "reverse_correlation_statistics"),
    "robust_post_error_slowing": ("meg_tokens.behavior.analyses.sequential", "robust_post_error_slowing"),
    "spd_cumulative_distributions": ("meg_tokens.behavior.analyses.distributions", "spd_cumulative_distributions"),
    "sum_log_lr_profile": ("meg_tokens.behavior.math.evidence", "sum_log_lr_profile"),
    "task_trials": ("meg_tokens.behavior.trials", "task_trials"),
    "time_on_task": ("meg_tokens.behavior.analyses.design_effects", "time_on_task"),
    "time_on_task_statistics": ("meg_tokens.behavior.analyses.design_effects", "time_on_task_statistics"),
    "token_lead_profile": ("meg_tokens.behavior.math.evidence", "token_lead_profile"),
    "paired_subject_statistics": ("meg_tokens.behavior.math.inference", "paired_subject_statistics"),
    "summarize_behavior": ("meg_tokens.behavior.analyses.summary", "summarize_behavior"),
    "validate_behavior_table": ("meg_tokens.behavior.schema", "validate_behavior_table"),
    "validate_trial_features": ("meg_tokens.behavior.schema", "validate_trial_features"),
    "build_trial_features": ("meg_tokens.behavior.features", "build_trial_features"),
    "logged_spd_at_decision": ("meg_tokens.behavior.features", "logged_spd_at_decision"),
    "probability_at_decision": ("meg_tokens.behavior.math.probability", "probability_at_decision"),
    "success_probability": ("meg_tokens.behavior.math.probability", "success_probability"),
    "success_probability_profile": ("meg_tokens.behavior.math.probability", "success_probability_profile"),
    "validate_probability_path": ("meg_tokens.behavior.math.probability", "validate_probability_path"),
    "FILENAME_RE": ("meg_tokens.behavior.tdms", "FILENAME_RE"),
    "TRIAL_COLUMNS": ("meg_tokens.behavior.schema", "TRIAL_COLUMNS"),
    "TRIAL_FEATURE_COLUMNS": ("meg_tokens.behavior.schema", "TRIAL_FEATURE_COLUMNS"),
    "TdmsRunInfo": ("meg_tokens.behavior.tdms", "TdmsRunInfo"),
    "add_run_metadata": ("meg_tokens.behavior.tables", "add_run_metadata"),
    "parse_single_trial": ("meg_tokens.behavior.tdms", "parse_single_trial"),
    "parse_tdms_filename": ("meg_tokens.behavior.tdms", "parse_tdms_filename"),
    "parse_tdms_file": ("meg_tokens.behavior.tdms", "parse_tdms_file"),
    "read_behavior_table": ("meg_tokens.behavior.tables", "read_behavior_table"),
    "read_trial_features": ("meg_tokens.behavior.tables", "read_trial_features"),
    "started_trials": ("meg_tokens.behavior.trials", "started_trials"),
    "write_beh_bids": ("meg_tokens.behavior.tdms_bids", "write_beh_bids"),
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
