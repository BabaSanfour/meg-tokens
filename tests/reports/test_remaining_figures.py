"""Smoke tests for F03, F06-F26 (Phases 2-6): each builder returns a Figure,
draws at least one statistical annotation where a test exists, and its
columns_read metadata names real derivative columns. Not pixel-tested; see
tests/reports/test_modeling.py and test_distributions.py for the more
detailed Phase 1 tests (including the dt_ms-not-rawRT regression guard).
"""

import matplotlib.pyplot as plt
import pytest

from meg_tokens.io import DerivativeLayout
from meg_tokens.reports.behavior import design, evidence, individual, modeling, sequential
from meg_tokens.reports.behavior._tables import BehaviorTableSet

from tests.reports.factories import (
    group_derivatives_from_rich_features,
    individual_and_species_derivatives,
    rich_trial_features,
    speciescomparison,
    ssmcomparison,
    ssmcomparisonstats,
    ssmpopulation,
    ssmtimecourse,
    write_group_derivative,
)


@pytest.fixture(scope="module")
def design_tables(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("design_tables")
    layout = DerivativeLayout(tmp_path)
    features = rich_trial_features()
    derivatives = group_derivatives_from_rich_features(features)
    for name, frame in derivatives.items():
        if name == "summary":
            from meg_tokens.io import save_table
            save_table(layout.behavior_summary(), frame, metadata={"stage": "test_fixture"})
        else:
            write_group_derivative(layout, name, frame)
    from meg_tokens.io import save_table
    save_table(layout.behavior_trial_features(), features, metadata={"stage": "test_fixture"})

    species = individual_and_species_derivatives(
        derivatives["summary"], derivatives["criteriondecline"], derivatives["urgency"],
        derivatives["continuousevidence"], derivatives["lapses"],
    )
    for name, frame in species.items():
        write_group_derivative(layout, name, frame)
    write_group_derivative(layout, "speciescomparison", speciescomparison())

    return BehaviorTableSet(layout=layout)


@pytest.fixture(scope="module")
def modeling_tables(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("modeling_tables")
    layout = DerivativeLayout(tmp_path)
    fits = ssmcomparison()
    write_group_derivative(layout, "ssmcomparison", fits)
    write_group_derivative(layout, "ssmcomparisonstats", ssmcomparisonstats(fits))
    write_group_derivative(layout, "ssmtimecourse", ssmtimecourse())
    subject_estimates, population = ssmpopulation()
    write_group_derivative(layout, "ssmpopulation", subject_estimates)
    write_group_derivative(layout, "ssmpopulationstats", population)
    return BehaviorTableSet(layout=layout)


def _assert_basic_figure(figure, metadata, *, min_axes=1, require_stat_text=True):
    assert isinstance(figure, plt.Figure)
    assert len(figure.axes) >= min_axes
    assert metadata["columns_read"]
    if require_stat_text:
        all_text = " ".join(t.get_text() for ax in figure.axes for t in ax.texts)
        assert "p <" in all_text or "p =" in all_text, "no stat annotation reached the figure"


# --- modeling.py (F03, F21, F22) -------------------------------------------


def test_ssmcomparison_urgencyparams(modeling_tables):
    figure, metadata = modeling.build_ssmcomparison_urgencyparams(modeling_tables)
    _assert_basic_figure(figure, metadata, min_axes=4)


def test_ssmtimecourse_fit(modeling_tables):
    figure, metadata = modeling.build_ssmtimecourse_fit(modeling_tables)
    _assert_basic_figure(figure, metadata, min_axes=3, require_stat_text=False)


def test_ssmpopulation_shrinkage(modeling_tables):
    figure, metadata = modeling.build_ssmpopulation_shrinkage(modeling_tables)
    _assert_basic_figure(figure, metadata, min_axes=7, require_stat_text=False)


# --- design.py (F08-F13) ----------------------------------------------------


def test_conditionclass_anova(design_tables):
    figure, metadata = design.build_conditionclass_anova(design_tables)
    # F08 annotates effect size plus a significance marker rather than
    # spelled-out 'p = ...' text, so the shared p-value check does not apply.
    _assert_basic_figure(figure, metadata, min_axes=2, require_stat_text=False)

    all_text = " ".join(t.get_text() for ax in figure.axes for t in ax.texts)
    assert "ηp²" in all_text, "no effect size reached the figure"
    assert "interaction" in all_text
    assert "p =" not in all_text and "p <" not in all_text, (
        "spelled-out p-values are what collapsed this figure's axes"
    )
    # The per-class stretch factors are the figure's positive claim.
    assert "Slow ÷ Fast" in all_text
    assert set(metadata["stretch_factors"]) == {"easy", "ambiguous", "misleading"}


def test_choiceside_asymmetry(design_tables):
    figure, metadata = design.build_choiceside_asymmetry(design_tables)
    # Three difference panels, not a 3x3 left-vs-right grid: left and right
    # choice proportions are complements, so plotting both drew one number
    # twice. Markers rather than spelled-out stat text.
    _assert_basic_figure(figure, metadata, min_axes=3, require_stat_text=False)
    assert len(figure.axes) == 3

    all_text = " ".join(text.get_text() for ax in figure.axes for text in ax.texts)
    assert "p =" not in all_text and "t(" not in all_text
    # Every panel states the quantity as a difference.
    assert all(ax.get_ylabel().startswith("Δ") for ax in figure.axes)


def test_timeontask_drift(design_tables):
    figure, metadata = design.build_timeontask_drift(design_tables)
    # Two panels: the per-subject coefficient strips were folded into A and B
    # as annotations, since their only content beyond those numbers was
    # per-subject spread. Compact 'Δ = value marker', so no spelled-out p.
    _assert_basic_figure(figure, metadata, min_axes=2, require_stat_text=False)
    assert len(figure.axes) == 2, "coefficient strip panels should stay folded in"

    all_text = " ".join(text.get_text() for ax in figure.axes for text in ax.texts)
    # Both coefficients and both Fast/Slow contrasts must survive the fold-in.
    assert "ms per block" in all_text, "session coefficient did not reach the figure"
    assert "ms per trial" in all_text, "within-block coefficient did not reach the figure"
    assert all_text.count("Fast − Slow") == 2, "a Fast/Slow contrast did not reach the figure"
    assert "p =" not in all_text and "p <" not in all_text
    # Raw column names must not surface as display labels.
    labels = [ax.get_ylabel() + " " + ax.get_xlabel() for ax in figure.axes]
    assert not any("dt_ms" in text or "dt_per" in text for text in labels)


def test_conditionorder_balance(design_tables):
    figure, metadata = design.build_conditionorder_balance(design_tables)
    # Compact 'Δ = value marker' convention, so the shared p-value check does
    # not apply: spelled-out Welch text is what collapsed this single-panel
    # figure's axes to zero width.
    _assert_basic_figure(figure, metadata, min_axes=1, require_stat_text=False)

    all_text = " ".join(t.get_text() for ax in figure.axes for t in ax.texts)
    assert "Δ =" in all_text, "no between-group contrast reached the figure"
    assert "p =" not in all_text and "Welch" not in all_text

    # The reframe: the claim rests on each group differing from zero, not on
    # the underpowered between-group null.
    assert set(metadata["within_group_vs_zero"]) == {"fast", "slow"}
    for stats_for_group in metadata["within_group_vs_zero"].values():
        assert stats_for_group["p"] < 0.05
    assert "underpowered" in metadata["caveat"]


def test_summary_cohort(design_tables):
    figure, metadata = design.build_summary_cohort(design_tables)
    # Five panels: the standalone QC figure was folded in as panel E, since it
    # asked the same question of the same subject rows and 22 of 32 subjects
    # had at most one flagged trial.
    # Six axes: panel E is split into the two halves of a broken x-axis, one
    # subject contributing 27 of 59 flagged trials against a median of 1.
    _assert_basic_figure(figure, metadata, min_axes=6, require_stat_text=False)
    assert len(figure.axes) == 6

    # The QC counts must reach the figure, or the fold-in lost them.
    labels = [ax.get_xlabel() for ax in figure.axes]
    assert "Flagged" in labels
    assert {"n_extreme", "n_lapse", "n_negative_dt"} <= set(metadata)
    assert "Nothing is removed" in metadata["caveat"]


# --- evidence.py (F14-F18) --------------------------------------------------


def test_criteriondecline_tokens(design_tables):
    figure, metadata = evidence.build_criteriondecline_tokens(design_tables)
    _assert_basic_figure(figure, metadata, min_axes=4)
    assert metadata["caveat"] is not None


def test_urgency_decisiontime(design_tables):
    figure, metadata = evidence.build_urgency_decisiontime(design_tables)
    _assert_basic_figure(figure, metadata, min_axes=4)


def test_reversecorrelation_kernel(design_tables):
    figure, metadata = evidence.build_reversecorrelation_kernel(design_tables)
    _assert_basic_figure(figure, metadata, min_axes=2, require_stat_text=False)


def test_conditionalaccuracy_caf(design_tables):
    figure, metadata = evidence.build_conditionalaccuracy_caf(design_tables)
    _assert_basic_figure(figure, metadata, min_axes=1)


def test_continuousevidence_effects(design_tables):
    figure, metadata = evidence.build_continuousevidence_effects(design_tables)
    _assert_basic_figure(figure, metadata, min_axes=4)
    assert metadata["includes_unclassified"] is True


# --- sequential.py (F19, F20) -----------------------------------------------


def test_posterror_slowing(design_tables):
    figure, metadata = sequential.build_posterror_slowing(design_tables)
    _assert_basic_figure(figure, metadata, min_axes=2)


def test_choicehistory_effects(design_tables):
    figure, metadata = sequential.build_choicehistory_effects(design_tables)
    _assert_basic_figure(figure, metadata, min_axes=4)


# --- individual.py (F23-F26) ------------------------------------------------


def test_individualcorrelations_matrix(design_tables):
    figure, metadata = individual.build_individualcorrelations_matrix(design_tables)
    _assert_basic_figure(figure, metadata, min_axes=1, require_stat_text=False)


def test_individualprofile_scatter(design_tables):
    figure, metadata = individual.build_individualprofile_scatter(design_tables)
    _assert_basic_figure(figure, metadata, min_axes=6)


def test_speciescomparison_forest(design_tables):
    figure, metadata = individual.build_speciescomparison_forest(design_tables)
    assert isinstance(figure, plt.Figure)
    assert len(figure.axes) == 5
    assert "comparable_to_source" in metadata


def test_individualprofile_neural_requires_flag(design_tables):
    with pytest.raises(FileNotFoundError, match="neural-metrics"):
        individual.build_individualprofile_neural(design_tables)


def test_individualprofile_neural_end_to_end(tmp_path):
    import pandas as pd
    from meg_tokens.io import save_table

    layout = DerivativeLayout(tmp_path)
    features = rich_trial_features()
    derivatives = group_derivatives_from_rich_features(features)
    save_table(layout.behavior_summary(), derivatives["summary"], metadata={"stage": "test_fixture"})
    species = individual_and_species_derivatives(
        derivatives["summary"], derivatives["criteriondecline"], derivatives["urgency"],
        derivatives["continuousevidence"], derivatives["lapses"],
    )
    write_group_derivative(layout, "individualprofile", species["individualprofile"])

    neural_path = tmp_path / "neural.csv"
    neural = pd.DataFrame({
        "subject": species["individualprofile"]["subject"],
        "neural_peak_ms": species["individualprofile"]["mean_dt_ms"] * 0.8 + 50,
    })
    neural.to_csv(neural_path, index=False)

    tables = BehaviorTableSet(layout=layout, neural_metrics_path=str(neural_path))
    figure, metadata = individual.build_individualprofile_neural(tables)
    assert isinstance(figure, plt.Figure)
    assert metadata["statistics_source"] == "computed_in_report"
