import numpy as np
import matplotlib.pyplot as plt
import pytest
from matplotlib.collections import PathCollection

from meg_tokens.reports import panels, style


def test_paired_slope_draws_one_line_per_subject():
    fig, ax = plt.subplots()
    values_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    values_b = np.array([1.5, 2.5, 2.8, 4.2, 4.9])
    panels.paired_slope(
        ax, values_a=values_a, values_b=values_b,
        label_a="A", label_b="B", color_a="#000000", color_b="#111111",
    )
    subject_lines = [line for line in ax.get_lines() if len(line.get_xdata()) == 2]
    assert len(subject_lines) >= 5


def test_paired_slope_with_all_nan_input_does_not_raise():
    fig, ax = plt.subplots()
    values_a = np.full(3, np.nan)
    values_b = np.full(3, np.nan)
    panels.paired_slope(
        ax, values_a=values_a, values_b=values_b,
        label_a="A", label_b="B", color_a="#000000", color_b="#111111",
    )


def test_paired_slope_with_a_single_subject_does_not_raise():
    fig, ax = plt.subplots()
    panels.paired_slope(
        ax, values_a=np.array([1.0]), values_b=np.array([2.0]),
        label_a="A", label_b="B", color_a="#000000", color_b="#111111",
    )


def test_within_subject_error_matches_hand_computed_value():
    # 3 subjects x 2 conditions; Cousineau-Morey normalisation by hand.
    values = np.array([[1.0, 3.0], [2.0, 4.0], [3.0, 5.0]])
    subject_means = values.mean(axis=1, keepdims=True)
    grand_mean = values.mean()
    normalized = values - subject_means + grand_mean
    correction = np.sqrt(2 / 1)
    expected_sd = normalized.std(axis=0, ddof=1) * correction
    expected_sem = expected_sd / np.sqrt(3)

    result = panels.within_subject_error(values, kind="sem")
    np.testing.assert_allclose(result, expected_sem)


def test_subject_strip_all_nan_group_does_not_raise():
    fig, ax = plt.subplots()
    panels.subject_strip(
        ax,
        groups={"a": np.full(3, np.nan), "b": np.array([1.0, 2.0, 3.0])},
        colors={"a": "#000000", "b": "#111111"},
    )


def test_subject_strip_connectors_end_on_the_jittered_subject_dots():
    fig, ax = plt.subplots()
    panels.subject_strip(
        ax,
        groups={
            "fast": np.array([0.1, np.nan, 0.3]),
            "slow": np.array([0.2, 0.4, 0.5]),
        },
        colors={"fast": "#000000", "slow": "#111111"},
        connect=(("fast", "slow"),),
    )

    dot_collections = [
        collection
        for collection in ax.collections
        if isinstance(collection, PathCollection)
    ]
    fast_dots = dot_collections[0].get_offsets()
    slow_dots = dot_collections[1].get_offsets()
    subject_lines = [
        line
        for line in ax.lines
        if line.get_color() == style.SUBJECT_LINE
        and line.get_zorder() == 1
        and len(line.get_xdata()) == 2
    ]

    assert len(subject_lines) == 2
    np.testing.assert_allclose(
        subject_lines[0].get_xdata(),
        [fast_dots[0, 0], slow_dots[0, 0]],
    )
    np.testing.assert_allclose(
        subject_lines[1].get_xdata(),
        [fast_dots[1, 0], slow_dots[2, 0]],
    )


def test_forest_draws_one_row_per_label():
    fig, ax = plt.subplots()
    panels.forest(
        ax,
        labels=["drift_scale", "urgency_scale", "urgency_onset_s"],
        centres=np.array([1.0, 1.5, 0.05]),
        lows=np.array([0.8, 1.2, 0.0]),
        highs=np.array([1.2, 1.8, 0.1]),
    )
    assert ax.get_yticks().size == 3


def test_raincloud_with_single_value_group_does_not_raise():
    fig, ax = plt.subplots()
    panels.raincloud(
        ax,
        groups={"easy": np.array([900.0]), "ambiguous": np.array([1000.0, 1100.0, 1200.0])},
        colors=style.CLASS_COLORS,
    )
