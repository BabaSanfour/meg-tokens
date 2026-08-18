import matplotlib.pyplot as plt

from meg_tokens.io import DerivativeLayout
from meg_tokens.behavior.analyses.sequential_sampling import heldout_model_statistics
from meg_tokens.reports.behavior import modeling
from meg_tokens.reports.behavior._tables import BehaviorTableSet

from tests.reports.factories import (
    ssmcomparison,
    ssmcomparisonstats,
    ssmtimecourse,
    write_group_derivative,
)


def _tables(tmp_path):
    layout = DerivativeLayout(tmp_path)
    fits = ssmcomparison()
    write_group_derivative(layout, "ssmcomparison", fits)
    write_group_derivative(layout, "ssmcomparisonstats", ssmcomparisonstats(fits))
    write_group_derivative(layout, "ssmtimecourse", ssmtimecourse())
    heldout = fits.loc[fits["condition"] == "all", ["subject", "model"]].copy()
    heldout["condition"] = "all"
    heldout["fold"] = 0
    heldout["heldout_log_likelihood_per_trial"] = -0.5
    heldout["predicted_accuracy"] = 0.7
    write_group_derivative(layout, "ssmheldout", heldout)
    write_group_derivative(layout, "ssmheldoutstats", heldout_model_statistics(heldout))
    return BehaviorTableSet(layout=layout)


def test_ssmcomparison_mechanistic_deduplicates_timecourse_and_has_four_panels(tmp_path):
    figure, metadata = modeling.build_ssmcomparison_mechanistic(_tables(tmp_path))
    assert isinstance(figure, plt.Figure)
    assert len(figure.axes) == 4
    assert metadata["deduplication"].startswith("subject-balanced")


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
