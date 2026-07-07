"""Utility modules for data loading, TDMS parsing, and I/O."""

__all__ = [
    "build_epochs_with_metadata",
    "normalize_subject_id",
    "parse_single_trial",
    "parse_tdms_file",
    "parse_tdms_filename",
    "process_subject_tdms",
    "save_epochs_and_events",
]


def parse_single_trial(*args, **kwargs):
    from meg_tokens.utils.tdms_parser import parse_single_trial as func
    return func(*args, **kwargs)


def parse_tdms_file(*args, **kwargs):
    from meg_tokens.utils.tdms_parser import parse_tdms_file as func
    return func(*args, **kwargs)


def normalize_subject_id(*args, **kwargs):
    from meg_tokens.utils.batch_processor import normalize_subject_id as func
    return func(*args, **kwargs)


def parse_tdms_filename(*args, **kwargs):
    from meg_tokens.utils.batch_processor import parse_tdms_filename as func
    return func(*args, **kwargs)


def process_subject_tdms(*args, **kwargs):
    from meg_tokens.utils.batch_processor import process_subject_tdms as func
    return func(*args, **kwargs)


def build_epochs_with_metadata(*args, **kwargs):
    from meg_tokens.utils.epochs_builder import build_epochs_with_metadata as func
    return func(*args, **kwargs)


def save_epochs_and_events(*args, **kwargs):
    from meg_tokens.utils.epochs_builder import save_epochs_and_events as func
    return func(*args, **kwargs)
