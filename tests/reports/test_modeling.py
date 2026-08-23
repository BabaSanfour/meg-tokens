import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection

from meg_tokens.io import DerivativeLayout
from meg_tokens.behavior.analyses.sequential_sampling import (
    heldout_model_statistics,
    heldout_pairwise_model_statistics,
)
from meg_tokens.reports import style
from meg_tokens.reports.behavior import modeling
from meg_tokens.reports.behavior._tables import BehaviorTableSet

from tests.reports.factories import (
    SUBJECTS,
    ssmcomparison,
    ssmboundaryaudit,
    ssmcomparisonstats,
    ssmheldout,
    ssmdistributioncheckstats,
    ssmrobustnessstats,
    ssmtimecourse,
    write_group_derivative,
)


def _tables(tmp_path):
    layout = DerivativeLayout(tmp_path)
    fits = ssmcomparison()
    write_group_derivative(layout, "ssmcomparison", fits)
    write_group_derivative(layout, "ssmcomparisonstats", ssmcomparisonstats(fits))
    write_group_derivative(layout, "ssmboundaryaudit", ssmboundaryaudit(fits))
    write_group_derivative(
        layout, "ssmrobustnessstats", ssmrobustnessstats()
    )
    write_group_derivative(layout, "ssmtimecourse", ssmtimecourse())
    write_group_derivative(
        layout, "ssmdistributioncheckstats", ssmdistributioncheckstats()
    )
    heldout = ssmheldout(fits)
    write_group_derivative(layout, "ssmheldout", heldout)
    write_group_derivative(layout, "ssmheldoutstats", heldout_model_statistics(heldout))
    write_group_derivative(
        layout, "ssmheldoutpairwise", heldout_pairwise_model_statistics(heldout)
    )
    return BehaviorTableSet(layout=layout)


def test_ssmcomparison_model_pairs_mechanism_with_adjudication(tmp_path):
    """Six panels: mechanism (A-C) above the adjudication that rests on it
    (D-F). These were two figures, which put the criterion time course in a
    different figure from the comparison it defines and drew the same held-out
    result twice."""
    figure, metadata = modeling.build_ssmcomparison_model(_tables(tmp_path))
    assert isinstance(figure, plt.Figure)
    assert len(figure.axes) == 6
    assert metadata["deduplication"].startswith("subject-balanced")
    assert metadata["columns_read"]["ssmcomparison"]


def test_ssmcomparison_model_draws_one_shared_model_legend(tmp_path):
    """A and C both name the four models; the legend belongs to the figure."""
    figure, _ = modeling.build_ssmcomparison_model(_tables(tmp_path))
    assert len(figure.legends) == 1
    ax_a, ax_b, ax_c = figure.axes[0], figure.axes[1], figure.axes[2]
    assert ax_a.get_legend() is None
    # The shared legend is the only place model names appear. B and C do carry
    # local legends, but for other scopes: B keys the trial classes and C keys
    # correct-vs-error line style. Neither may name a model, or the figure
    # would identify the same thing in two places.
    model_names = set(style.MODEL_LABELS.values())
    for ax in (ax_b, ax_c):
        legend = ax.get_legend()
        assert legend is not None
        local = {text.get_text() for text in legend.get_texts()}
        assert not (local & model_names), local


def test_ssmcomparison_model_leaves_dense_time_courses_unmarkered(tmp_path):
    """A marker per sample turns a few-hundred-point curve into a rope."""
    figure, _ = modeling.build_ssmcomparison_model(_tables(tmp_path))
    for ax in figure.axes[:3]:
        for line in ax.get_lines():
            if len(line.get_xdata()) > 30:
                assert line.get_marker() in ("None", "", None)


def test_ssmcomparison_model_shows_every_candidate_and_every_candidate_pair(tmp_path):
    figure, _ = modeling.build_ssmcomparison_model(_tables(tmp_path))
    ax_b = figure.axes[5]
    assert [tick.get_text() for tick in figure.axes[3].get_xticklabels()] == [
        "Urgency", "Collapsing", "Additive",
    ]
    # Every candidate pair, so no head-to-head is picked for the reader; the
    # candidate-versus-baseline pairs are A's job and would be arithmetic on
    # these three besides.
    rows = [text.get_text() for text in ax_b.texts if "−" in text.get_text()]
    assert rows == [
        "Urgency − Collapsing", "Urgency − Additive", "Collapsing − Additive",
    ]
    # Per-subject dots, not just the group interval: B lost them once and the
    # panel became three numbers.
    subject_dots = sum(
        len(collection.get_offsets())
        for collection in ax_b.collections
        if isinstance(collection, PathCollection)
    )
    assert subject_dots == 3 * len(SUBJECTS)


def test_ssmcomparison_model_reorients_reverse_stored_pairs(tmp_path):
    """The derivative stores each pair alphabetically; half need flipping."""
    tables = _tables(tmp_path)
    pairwise = tables.analysis("ssmheldoutpairwise")
    stored = pairwise.loc[
        (pairwise["model_a"] == "collapsing")
        & (pairwise["model_b"] == "urgency")
        & (pairwise["condition"] == "all")
    ].iloc[0]
    flipped = modeling._pairwise_contrast(pairwise, "urgency", "collapsing")
    assert flipped["mean"] == -float(stored["mean"])
    # Flipping reverses the interval's ends; it has to be re-sorted, not
    # copied across, or the error bar is drawn inside out.
    assert flipped["ci_low"] < flipped["mean"] < flipped["ci_high"]
    assert flipped["ci_low"] == -float(stored["ci95_high"])
    # Reading the row raw would have put urgency on the losing side.
    assert flipped["mean"] > 0


def test_ssmcomparison_urgency_carries_its_censoring_defence(tmp_path):
    """One row: the four parameters, then the contrast on the one that moves
    together with the refits that answer the censoring objection it raises."""
    figure, metadata = modeling.build_ssmcomparison_urgency(_tables(tmp_path))
    assert isinstance(figure, plt.Figure)
    assert len(figure.axes) == 5
    assert metadata["caveat"] is not None
    rows = [label.get_text() for label in figure.axes[4].get_yticklabels()]
    # The production fit is a row of the same forest as the refits, so the
    # result and the checks on it are read on one axis.
    assert "production fit" in rows, rows
    # The uncensored refit has to be there: it is the row answering the
    # objection that the censoring runs with the effect.
    assert any("wider bounds" in row for row in rows), rows
    assert len(rows) == 7, rows


def test_ssmcomparison_urgency_reads_its_equivalence_bound(tmp_path):
    """A null this figure argues from states what it excludes -- and reads
    that bound from the derivative rather than deriving one."""
    figure, _ = modeling.build_ssmcomparison_urgency(_tables(tmp_path))
    titles = " ".join(ax.get_title() for ax in figure.axes)
    assert "|Δ| <" in titles, titles


def test_ssmcomparison_urgency_annotation_reaches_the_figure(tmp_path):
    figure, _ = modeling.build_ssmcomparison_urgency(_tables(tmp_path))
    all_text = " ".join(
        [t.get_text() for ax in figure.axes for t in ax.texts]
        + [ax.get_title() for ax in figure.axes]
    )
    # The compact convention is `Delta = value unit marker`, so the test
    # result reaches the panel as a marker rather than a numeric p. "n/a" is
    # deliberately not accepted: it means no test was run, which is a
    # different claim from a non-significant one.
    assert "Δ = " in all_text, all_text
    assert any(marker in all_text for marker in ("***", "**", "*", "n.s.")), all_text
