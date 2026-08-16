import matplotlib.pyplot as plt

from meg_tokens.io import DerivativeLayout
from meg_tokens.reports.behavior import modeling
from meg_tokens.reports.behavior._tables import BehaviorTableSet

from tests.reports.factories import (
    ssmcomparison,
    ssmcomparisonstats,
    write_group_derivative,
)


def _tables(tmp_path):
    layout = DerivativeLayout(tmp_path)
    fits = ssmcomparison()
    write_group_derivative(layout, "ssmcomparison", fits)
    write_group_derivative(layout, "ssmcomparisonstats", ssmcomparisonstats(fits))
    return BehaviorTableSet(layout=layout)


def test_ssmcomparison_deltabic_returns_a_figure_with_two_axes(tmp_path):
    figure, metadata = modeling.build_ssmcomparison_deltabic(_tables(tmp_path))
    assert isinstance(figure, plt.Figure)
    assert len(figure.axes) == 2
    assert metadata["columns_read"]["ssmcomparison"]


def test_ssmcomparison_deltabic_annotation_reaches_the_figure(tmp_path):
    figure, _ = modeling.build_ssmcomparison_deltabic(_tables(tmp_path))
    all_text = " ".join(t.get_text() for ax in figure.axes for t in ax.texts)
    assert "p <" in all_text or "p =" in all_text


def test_ssmcomparison_urgencyscale_returns_a_figure_with_two_axes(tmp_path):
    figure, metadata = modeling.build_ssmcomparison_urgencyscale(_tables(tmp_path))
    assert isinstance(figure, plt.Figure)
    assert len(figure.axes) == 2
    assert metadata["caveat"] is not None


def test_ssmcomparison_urgencyscale_annotation_reaches_the_figure(tmp_path):
    figure, _ = modeling.build_ssmcomparison_urgencyscale(_tables(tmp_path))
    all_text = " ".join(t.get_text() for ax in figure.axes for t in ax.texts)
    assert "p <" in all_text or "p =" in all_text
