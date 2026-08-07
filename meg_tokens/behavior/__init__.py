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
    # Roadmap analyses: docs/behavior_analysis_roadmap.md
    "choice_history": ("meg_tokens.behavior.sequential", "choice_history"),
    "choice_history_statistics": ("meg_tokens.behavior.sequential", "choice_history_statistics"),
    "choice_side_statistics": ("meg_tokens.behavior.design_effects", "choice_side_statistics"),
    "choice_side_summary": ("meg_tokens.behavior.design_effects", "choice_side_summary"),
    "comparison_statistics": ("meg_tokens.behavior.individual", "comparison_statistics"),
    "conditional_accuracy_functions": ("meg_tokens.behavior.evidence", "conditional_accuracy_functions"),
    "conditional_accuracy_statistics": ("meg_tokens.behavior.evidence", "conditional_accuracy_statistics"),
    "condition_class_cells": ("meg_tokens.behavior.design_effects", "condition_class_cells"),
    "condition_class_statistics": ("meg_tokens.behavior.design_effects", "condition_class_statistics"),
    "condition_order_effects": ("meg_tokens.behavior.design_effects", "condition_order_effects"),
    "condition_order_statistics": ("meg_tokens.behavior.design_effects", "condition_order_statistics"),
    "continuous_evidence_effects": ("meg_tokens.behavior.evidence", "continuous_evidence_effects"),
    "continuous_evidence_statistics": ("meg_tokens.behavior.evidence", "continuous_evidence_statistics"),
    "criterion_decline": ("meg_tokens.behavior.evidence", "criterion_decline"),
    "criterion_decline_statistics": ("meg_tokens.behavior.evidence", "criterion_decline_statistics"),
    "decision_time_distributions": ("meg_tokens.behavior.distributions", "decision_time_distributions"),
    "decision_time_distribution_statistics": ("meg_tokens.behavior.distributions", "decision_time_distribution_statistics"),
    "distribution_summary": ("meg_tokens.behavior.distributions", "distribution_summary"),
    "evidence_after_tokens": ("meg_tokens.behavior.evidence", "evidence_after_tokens"),
    "ex_gaussian_parameters": ("meg_tokens.behavior.distributions", "ex_gaussian_parameters"),
    "extreme_decision_times": ("meg_tokens.behavior.design_effects", "extreme_decision_times"),
    "fit_linear": ("meg_tokens.behavior.regression", "fit_linear"),
    "fit_logistic": ("meg_tokens.behavior.regression", "fit_logistic"),
    "individual_correlations": ("meg_tokens.behavior.individual", "individual_correlations"),
    "individual_profile": ("meg_tokens.behavior.individual", "individual_profile"),
    "lapse_statistics": ("meg_tokens.behavior.design_effects", "lapse_statistics"),
    "lapse_summary": ("meg_tokens.behavior.design_effects", "lapse_summary"),
    "log_posterior_odds": ("meg_tokens.behavior.evidence", "log_posterior_odds"),
    "one_sample_statistics": ("meg_tokens.behavior.regression", "one_sample_statistics"),
    "parse_token_directions": ("meg_tokens.behavior.evidence", "parse_token_directions"),
    "post_error_statistics": ("meg_tokens.behavior.sequential", "post_error_statistics"),
    "repeated_measures_anova": ("meg_tokens.behavior.regression", "repeated_measures_anova"),
    "reverse_correlation": ("meg_tokens.behavior.evidence", "reverse_correlation"),
    "reverse_correlation_statistics": ("meg_tokens.behavior.evidence", "reverse_correlation_statistics"),
    "robust_post_error_slowing": ("meg_tokens.behavior.sequential", "robust_post_error_slowing"),
    "spd_cumulative_distributions": ("meg_tokens.behavior.distributions", "spd_cumulative_distributions"),
    "sum_log_lr_profile": ("meg_tokens.behavior.evidence", "sum_log_lr_profile"),
    "task_trials": ("meg_tokens.behavior.trials", "task_trials"),
    "time_on_task": ("meg_tokens.behavior.design_effects", "time_on_task"),
    "time_on_task_statistics": ("meg_tokens.behavior.design_effects", "time_on_task_statistics"),
    "token_lead_profile": ("meg_tokens.behavior.evidence", "token_lead_profile"),
    "urgency_parameters": ("meg_tokens.behavior.evidence", "urgency_parameters"),
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
