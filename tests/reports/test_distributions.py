import matplotlib.pyplot as plt

from meg_tokens.io import DerivativeLayout
from meg_tokens.reports.behavior import distributions
from meg_tokens.reports.behavior._tables import BehaviorTableSet

from tests.reports.factories import (
    dtdistribution,
    dtdistributionstats,
    groupstats,
    trial_features_group,
    write_group_derivative,
)


def _tables(tmp_path):
    layout = DerivativeLayout(tmp_path)
    dist = dtdistribution()
    write_group_derivative(layout, "dtdistribution", dist)
    write_group_derivative(layout, "dtdistributionstats", dtdistributionstats(dist))
    write_group_derivative(layout, "groupstats", groupstats())
    write_group_derivative(layout, "trialfeatures", trial_features_group())
    return BehaviorTableSet(layout=layout)


def test_dtdistribution_condition_reads_dt_ms_not_rawrt(tmp_path):
    """Regression guard for the defect this plan fixes."""
    figure, metadata = distributions.build_dtdistribution_condition(_tables(tmp_path))
    assert isinstance(figure, plt.Figure)
    all_columns = [
        column
        for columns in metadata["columns_read"].values()
        for column in columns
    ]
    assert "dt_ms" in all_columns
    assert "rawRT" not in all_columns


def test_dtdistribution_condition_annotation_reaches_the_figure(tmp_path):
    """Regression guard: some significance annotation (marker or spelled-out
    p) reaches the figure -- not that a specific derivative was computed but
    never rendered. B shows markers (***/n.s.), not spelled-out p, so check
    for either form rather than pinning the exact text."""
    figure, _ = distributions.build_dtdistribution_condition(_tables(tmp_path))
    all_text = " ".join(t.get_text() for ax in figure.axes for t in ax.texts)
    assert "p <" in all_text or "p =" in all_text or "*" in all_text or "n.s." in all_text


def test_dtdistribution_class_excludes_unclassified(tmp_path):
    figure, metadata = distributions.build_dtdistribution_class(_tables(tmp_path))
    assert isinstance(figure, plt.Figure)
    assert metadata["excludes"] == "unclassified"
    tick_labels = [
        label.get_text()
        for ax in figure.axes
        for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels())
    ]
    legend_labels = [
        text.get_text()
        for ax in figure.axes
        if ax.get_legend() is not None
        for text in ax.get_legend().get_texts()
    ]
    assert "unclassified" not in [label.lower() for label in tick_labels + legend_labels]


def test_dtdistribution_class_annotation_reaches_the_figure(tmp_path):
    """Regression guard: some significance annotation (marker or spelled-out
    p) reaches the figure. Panel A's brackets show markers (***/n.s.), not
    spelled-out p -- see distributions.py's F04 test for why."""
    figure, _ = distributions.build_dtdistribution_class(_tables(tmp_path))
    all_text = " ".join(t.get_text() for ax in figure.axes for t in ax.texts)
    assert "p <" in all_text or "p =" in all_text or "*" in all_text or "n.s." in all_text
