import numpy as np
import pandas as pd

from meg_tokens.io import derivative_path, save_array, save_table, sidecar_path
from meg_tokens.workflows.statistics import stats_derivative_path
from meg_tokens.reports.statistics import (
    build_stats_summary,
    load_group_stats,
    run_group_statistics_plotting,
    select_plot_units,
)


LABELS = ["Label_1-lh", "Label_2-rh"]
TIMES = [-0.1, 0.0, 0.1]
SUBJECTS = ["H01", "H02", "H03"]


def _write_stats_derivatives(root):
    conditions = ("Fast", "Slow")
    tstat = np.array([
        [0.0, 3.0, -4.0],
        [1.0, 2.0, 0.0],
    ])
    pvalue = np.array([
        [0.20, 0.01, 0.04],
        [0.30, 0.06, 0.50],
    ])
    contrast = np.array([
        [[0.0, 1.0, -2.0], [0.0, 0.5, 0.2]],
        [[0.0, -3.0, -1.0], [0.0, 0.2, 0.1]],
        [[0.0, -1.0, -4.0], [0.0, 0.3, 0.4]],
    ])
    coords = {"label": LABELS, "time_sec": TIMES}
    metadata = {
        "stage": "group_statistics",
        "conditions": list(conditions),
        "subjects": SUBJECTS,
        "alignment": "go",
        "source_method": "dSPM",
        "parcellation": "HCPMMP1",
    }

    for suffix, data, dims, extra_coords in [
        ("tstat", tstat, ("label", "time"), coords),
        ("pvalue", pvalue, ("label", "time"), coords),
        ("contrast", contrast, ("subject", "label", "time"), {"subject": SUBJECTS, **coords}),
    ]:
        save_array(
            stats_derivative_path(
                root,
                conditions=conditions,
                align_to="go",
                source_method="dSPM",
                parc="HCPMMP1",
                suffix=suffix,
                extension=".npy",
            ),
            data,
            dims=dims,
            coords=extra_coords,
            metadata=metadata,
        )


def _write_behavior_derivatives(root):
    for subject, rt in zip(SUBJECTS, [600.0, 800.0, 1000.0]):
        for condition in ("Fast", "Slow"):
            path = derivative_path(
                root,
                subject=subject,
                datatype="beh",
                task="tokens",
                run="1",
                description=condition.lower(),
                suffix="beh",
                extension=".tsv",
            )
            table = pd.DataFrame({
                "subject": [subject, subject],
                "condition": [condition, condition],
                "run": [1, 1],
                "source_file": [f"{subject}{condition}1_180131.tdms"] * 2,
                "nTrialIndex": [1, 2],
                "sTrialClass": [1, 1],
                "nInitialTime": [0, 0],
                "nChoiceMade": [1, 1],
                "nCorrectChoice": [1, 1],
                "tGO": [1000.0, 1000.0],
                "tEnterTarget": [1000.0 + rt, 1000.0 + rt],
                "tTrialEnd": [3000.0, 3000.0],
                "sTokenDirs": ["0", "0"],
                "tTime": ["[]", "[]"],
                "nProb": ["[]", "[]"],
                "rawRT": [rt, rt],
                "isCorrect": [True, True],
            })
            save_table(path, table, metadata={"stage": "behavior_parsing", "subject": subject})


def test_load_group_stats_and_select_plot_units(tmp_path):
    _write_stats_derivatives(tmp_path)

    bundle = load_group_stats(
        tmp_path,
        conditions=("Fast", "Slow"),
        align_to="go",
        source_method="dSPM",
        parc="HCPMMP1",
    )
    units = select_plot_units(bundle["tstat"], bundle["dims"], bundle["coords"], top_n=1)

    assert bundle["tstat"].shape == (2, 3)
    assert bundle["contrast"].shape == (3, 2, 3)
    assert bundle["subjects"] == SUBJECTS
    assert units == [{"label_index": 0, "label": "Label_1-lh"}]


def test_build_stats_summary_reports_onset_and_peak(tmp_path):
    _write_stats_derivatives(tmp_path)
    bundle = load_group_stats(
        tmp_path,
        conditions=("Fast", "Slow"),
        align_to="go",
        source_method="dSPM",
        parc="HCPMMP1",
    )

    summary = build_stats_summary(
        bundle["tstat"],
        bundle["pvalue"],
        bundle["dims"],
        bundle["coords"],
        alpha=0.05,
        peak_mode="min",
    )

    first = summary.loc[summary["label"] == "Label_1-lh"].iloc[0]
    assert first["onset_index"] == 1
    assert first["peak_index"] == 2
    assert first["n_significant_windows"] == 1


def test_run_group_statistics_plotting_writes_summary_and_figure(tmp_path):
    _write_stats_derivatives(tmp_path)

    outputs = run_group_statistics_plotting(
        stats_dir=str(tmp_path),
        output_dir=str(tmp_path),
        conditions=("Fast", "Slow"),
        labels=["Label_1-lh"],
        alpha=0.05,
        peak_mode="min",
    )

    summary = pd.read_csv(outputs["summary"], sep="\t")
    figure = outputs["figures"][0]

    assert outputs["summary"].name.endswith("_statsummary.tsv")
    assert summary["label"].tolist() == LABELS
    assert figure.is_file()
    assert sidecar_path(figure).is_file()


def test_run_group_statistics_plotting_can_correlate_behavior(tmp_path):
    _write_stats_derivatives(tmp_path)
    _write_behavior_derivatives(tmp_path)

    outputs = run_group_statistics_plotting(
        stats_dir=str(tmp_path),
        output_dir=str(tmp_path),
        conditions=("Fast", "Slow"),
        labels=["Label_1-lh"],
        behavior_dir=str(tmp_path),
        correlate_behavior=True,
        peak_mode="min",
    )

    correlations = pd.read_csv(outputs["correlations"], sep="\t")
    assert outputs["correlations"].name.endswith("_statcorrelations.tsv")
    assert correlations["label"].tolist() == ["Label_1-lh"]
    assert correlations["n_subjects"].tolist() == [3]
