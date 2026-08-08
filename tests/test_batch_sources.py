import os

import pytest

from meg_tokens.io import DerivativeLayout
from meg_tokens.meg.sources import source_space_description


def test_find_noise_file_requires_single_match(tmp_path):
    subject_dir = tmp_path / "H01"
    os.makedirs(subject_dir)
    noise_file = subject_dir / "empty_noise.fif"
    noise_file.write_text("")

    assert DerivativeLayout(tmp_path).find_noise(subject="H1") == noise_file


def test_find_noise_file_rejects_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="No empty-room noise file"):
        DerivativeLayout(tmp_path).find_noise(subject="H01")


def test_find_epoch_file_matches_stage2_contract(tmp_path):
    epoch_dir = tmp_path / "derivatives" / "sub-H01" / "meg"
    os.makedirs(epoch_dir)
    epoch_file = epoch_dir / "sub-H01_task-tokens_run-1_desc-slow-go_epo.fif"
    epoch_file.write_text("")

    assert DerivativeLayout(tmp_path).find_epochs(
        subject="H1",
        run="Slow1",
        condition=None,
        alignment="go",
    ) == epoch_file


def test_find_trans_file_matches_subject_and_run(tmp_path):
    trans_dir = tmp_path / "coreg"
    os.makedirs(trans_dir)
    trans_file = trans_dir / "sub-H01_run-1-trans.fif"
    trans_file.write_text("")

    assert DerivativeLayout(tmp_path).find_trans(subject="H01", run="1") == trans_file


def test_find_trans_file_prefers_requested_run(tmp_path):
    trans_dir = tmp_path / "coreg"
    os.makedirs(trans_dir)
    run_one = trans_dir / "sub-H01_run-1-trans.fif"
    run_two = trans_dir / "sub-H01_run-2-trans.fif"
    run_one.write_text("")
    run_two.write_text("")

    assert DerivativeLayout(tmp_path).find_trans(subject="H01", run="1") == run_one


def test_model_paths_are_derivative_paths(tmp_path):
    paths = DerivativeLayout(tmp_path).source_models(subject="H1", spacing="oct6")

    assert paths["cov"].name == "sub-H01_task-tokens_desc-noise_cov.fif"
    assert paths["bem"].name == "sub-H01_task-tokens_desc-singlelayer_bem.fif"
    assert paths["src"].name == "sub-H01_task-tokens_space-subject_desc-oct6_src.fif"


def test_model_paths_keep_mixed_source_spaces_distinct(tmp_path):
    paths = DerivativeLayout(tmp_path).source_models(
        subject="H1",
        spacing="oct6",
        mixed=True,
    )

    assert source_space_description("oct6", ["Left-Putamen"]) == "oct6-mixed"
    assert paths["src"].name == "sub-H01_task-tokens_space-subject_desc-oct6-mixed_src.fif"
