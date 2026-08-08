from meg_tokens.io import DerivativeLayout
from meg_tokens.workflows.decoding import decoding_derivative_path
from meg_tokens.workflows.decomposition import pca_derivative_path
from meg_tokens.workflows.erp import erp_derivative_path
from meg_tokens.workflows.statistics import stats_derivative_path
from meg_tokens.workflows.power import power_derivative_path


def test_layout_builds_core_stage_paths():
    layout = DerivativeLayout("/analysis")
    prefix = "/analysis/derivatives/sub-H01"

    assert layout.behavior(subject="H1", run="Slow1", condition="Slow").as_posix() == (
        f"{prefix}/beh/sub-H01_task-tokens_run-1_desc-slow_beh.tsv"
    )
    assert layout.preprocessed_raw(
        subject="H1",
        run="Slow1",
        processing="filt",
    ).as_posix() == (
        f"{prefix}/meg/sub-H01_task-tokens_run-1_proc-filt_desc-slow_raw.fif"
    )
    assert layout.epochs(
        subject="H1",
        run="Slow1",
        condition=None,
        alignment="go",
    ).as_posix() == (
        f"{prefix}/meg/sub-H01_task-tokens_run-1_desc-slow-go_epo.fif"
    )
    assert layout.source(
        subject="H1",
        suffix="stcmanifest",
        extension=".tsv",
        run="Slow1",
        description="go-dSPM",
    ).as_posix() == (
        f"{prefix}/meg/sub-H01_task-tokens_run-1_desc-slow-go-dSPM_stcmanifest.tsv"
    )


def test_layout_discovers_raw_identity_without_losing_condition(tmp_path):
    layout = DerivativeLayout(tmp_path)
    raw_path = layout.preprocessed_raw(
        subject="H1",
        run="1",
        condition="Slow",
        processing="filt",
    )
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text("", encoding="utf-8")

    assert layout.raw_files(subject="H1") == [raw_path]
    run = layout.raw_run(raw_path)
    assert run.subject == "H01"
    assert run.run == "1"
    assert run.condition == "Slow"


def test_layout_matches_compatibility_path_builders():
    layout = DerivativeLayout("/analysis")

    assert layout.erp(
        subject="H1",
        run="Slow1",
        condition=None,
        alignment="go",
        source_method="dSPM",
        parc="HCPMMP1",
        suffix="erp",
        extension=".npy",
    ) == erp_derivative_path(
        "/analysis",
        subject="H1",
        run="Slow1",
        condition=None,
        align_to="go",
        source_method="dSPM",
        parc="HCPMMP1",
        suffix="erp",
        extension=".npy",
    )
    assert layout.power(
        subject="H1",
        run="Slow1",
        condition=None,
        alignment="go",
        source_method="dSPM",
        power_method="hilbert",
        band="alpha",
    ) == power_derivative_path(
        "/analysis",
        subject="H1",
        run="Slow1",
        condition=None,
        align_to="go",
        source_method="dSPM",
        power_method="hilbert",
        band="alpha",
    )
    assert layout.group_stats(
        conditions=["Fast", "Slow"],
        alignment="go",
        source_method="dSPM",
        parc="HCPMMP1",
        suffix="tstat",
        extension=".npy",
    ) == stats_derivative_path(
        "/analysis",
        conditions=["Fast", "Slow"],
        align_to="go",
        source_method="dSPM",
        parc="HCPMMP1",
        suffix="tstat",
        extension=".npy",
    )
    assert layout.lateralized_stats(
        condition="Fast",
        alignment="go",
        source_method="dSPM",
        parc="HCPMMP1",
        suffix="tstat",
        extension=".npy",
    ).name == (
        "sub-group_task-tokens_desc-fast-lateralized-go-dSPM-HCPMMP1_tstat.npy"
    )
    assert layout.decoding(
        conditions=["Fast", "Slow"],
        feature_source="erp",
        alignment="go",
        source_method="dSPM",
        parc="HCPMMP1",
        band=None,
        suffix="decodingscore",
        extension=".npy",
    ) == decoding_derivative_path(
        "/analysis",
        conditions=["Fast", "Slow"],
        feature_source="erp",
        align_to="go",
        source_method="dSPM",
        parc="HCPMMP1",
        band=None,
        suffix="decodingscore",
        extension=".npy",
    )
    assert layout.pca(
        conditions=["Fast", "Slow", "Error"],
        feature_source="erp",
        alignment="go",
        source_method="dSPM",
        parc="HCPMMP1",
        band=None,
        suffix="pcatrajectory",
        extension=".npy",
    ) == pca_derivative_path(
        "/analysis",
        conditions=["Fast", "Slow", "Error"],
        feature_source="erp",
        align_to="go",
        source_method="dSPM",
        parc="HCPMMP1",
        band=None,
        suffix="pcatrajectory",
        extension=".npy",
    )
