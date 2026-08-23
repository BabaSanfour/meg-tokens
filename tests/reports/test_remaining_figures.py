"""Smoke tests for F01-F05, F09-F12, F15-F18: each
builder returns a Figure,
draws at least one statistical annotation where a test exists, and its
columns_read metadata names real derivative columns. Not pixel-tested; see
tests/reports/test_modeling.py and test_distributions.py for the more
detailed Phase 1 tests (including the dt_ms-not-rawRT regression guard).
"""

import matplotlib.pyplot as plt
import pandas as pd
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
    ssmrobustnessstats,
    ssmboundaryaudit,
    ssmrecovery,
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
    write_group_derivative(layout, "ssmboundaryaudit", ssmboundaryaudit(fits))
    write_group_derivative(layout, "ssmrobustnessstats", ssmrobustnessstats())
    recovery, recovery_stats, confusion = ssmrecovery()
    write_group_derivative(layout, "ssmparameterrecovery", recovery)
    write_group_derivative(layout, "ssmparameterrecoverystats", recovery_stats)
    write_group_derivative(layout, "ssmmodelrecoverystats", confusion)
    return BehaviorTableSet(layout=layout)


def _assert_basic_figure(figure, metadata, *, min_axes=1, require_stat_text=True):
    assert isinstance(figure, plt.Figure)
    assert len(figure.axes) >= min_axes
    assert metadata["columns_read"]
    if require_stat_text:
        # Titles count as well as free text: a compact stat belongs in the
        # axes title, which is the one place constrained_layout reserves room
        # for, and several panels now put it there.
        all_text = " ".join(
            [t.get_text() for ax in figure.axes for t in ax.texts]
            + [ax.get_title() for ax in figure.axes]
        )
        # Either convention counts: the spelled-out t/df/p under an estimation
        # axis, or the compact "Δ = value marker" -- where the marker on a null
        # the figure argues from is an equivalence bound, not a p at all.
        assert any(
            token in all_text
            for token in ("p <", "p =", "***", "n.s.", "|Δ| <")
        ), "no stat annotation reached the figure"


# --- modeling.py (F14, F15) -------------------------------------------------


def test_ssmcomparison_urgency(modeling_tables):
    figure, metadata = modeling.build_ssmcomparison_urgency(modeling_tables)
    _assert_basic_figure(figure, metadata, min_axes=5)


def test_ssmrecovery_identifiability(modeling_tables):
    figure, metadata = modeling.build_ssmrecovery_identifiability(modeling_tables)
    _assert_basic_figure(figure, metadata, min_axes=4, require_stat_text=False)
    # The panel that explains the non-decision failures needs the solver step
    # it divides by; without it the axis is a bare ratio of nothing.
    assert metadata["solver_step_s"] > 0
    assert metadata["boundary_rate"]


# --- design.py (F01-F05) -----------------------------------------------------


def test_conditionclass_anova(design_tables):
    figure, metadata = design.build_conditionclass_anova(design_tables)
    # F05 annotates effect size plus a significance marker rather than
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
    # Three decimals, not two: the factors differ by <0.01 and the claim is
    # that they are indistinguishable, so two decimals straddles a rounding
    # boundary and shows a step that is not in the data.
    for factor in metadata["stretch_factors"].values():
        assert f"{factor:.3f}" in all_text, "a stretch factor lost its precision"
    # The null the figure argues *from* carries a bound, not a bare marker.
    assert "] ms" in all_text, "the additive-interaction bound did not reach panel A"


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
    # Both Fast/Slow contrasts are null, and the figure's point is how tightly
    # each is bounded -- so they must show an interval, never a bare 'n.s.'.
    assert all_text.count("[") == 2, "a Fast/Slow contrast lost its interval"
    assert "n.s." not in all_text
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


# --- evidence.py (F09-F12) ---------------------------------------------------


def test_criterionscale_comparison(design_tables):
    figure, metadata = evidence.build_criterionscale_comparison(design_tables)
    # Three panels: the two scales and the within-subject reversal.
    _assert_basic_figure(figure, metadata, min_axes=3, require_stat_text=False)
    assert len(figure.axes) == 3

    all_text = " ".join(t.get_text() for ax in figure.axes for t in ax.texts)
    # Both nulls are bounded, never a bare marker -- the Fast/Slow contrast is
    # uninformative without the size of difference the test could detect.
    assert all_text.count("|Δ| <") == 2, "a Fast/Slow bound did not reach the figure"
    # The slope rows keep a significance marker (they test a real effect, and
    # on synthetic fixture data that marker is legitimately 'n.s.'); it is the
    # Fast/Slow rows that must never fall back to one.
    assert "Fast − Slow  n.s." not in all_text
    # The between-scale contrast is the figure's result, so it must be drawn --
    # as a bracket over the two columns, with the paired difference and the
    # across-subject correlation.
    assert "Δ = " in all_text and "r = " in all_text
    # Both scales must be sourced, so the reversal is attributable to the scale.
    assert set(metadata["columns_read"]) == {
        "criterionscale", "criterionscaleprofile", "criterionscalestats",
    }


def test_reversecorrelation_kernel(design_tables):
    figure, metadata = evidence.build_reversecorrelation_kernel(design_tables)
    _assert_basic_figure(figure, metadata, min_axes=2, require_stat_text=False)


def test_conditionalaccuracy_caf(design_tables):
    figure, metadata = evidence.build_conditionalaccuracy_caf(design_tables)
    # Compact 'value marker' annotations, not spelled-out t/df/p, so the
    # generic stat-text probe does not apply here.
    _assert_basic_figure(figure, metadata, min_axes=2, require_stat_text=False)

    text = " ".join(t.get_text() for ax in figure.axes for t in ax.texts)
    # The classes are named by the legend, not by the annotation. Annotating
    # all four repeated the legend and crowded the panel, so only the two a
    # reader cannot get by looking are printed: the class that reverses and
    # the null the figure argues from.
    legends = [ax.get_legend() for ax in figure.axes if ax.get_legend()]
    legends += list(figure.legends)  # the classes sit in a figure-level legend
    labels = {t.get_text() for legend in legends for t in legend.get_texts()}
    assert {"easy", "ambiguous", "misleading", "unclassified"} <= labels
    assert {"observed", "class-adjusted"} <= labels
    assert "misl." in text and "ambig." in text
    assert "easy" not in text, "the legend already names the classes"
    # The pooled panel states the adjusted slope beside the observed one --
    # the claim the figure exists to make.
    assert "observed" in text and "adjusted" in text
    # Accuracy is read against chance, and a null slope is quoted as a bound.
    assert "chance" in text
    assert "|slope| <" in text


def test_continuousevidence_effects(design_tables):
    figure, metadata = evidence.build_continuousevidence_effects(design_tables)
    # Compact 'value marker' annotations, not spelled-out t/df/p.
    _assert_basic_figure(figure, metadata, min_axes=2, require_stat_text=False)
    assert metadata["includes_unclassified"] is True

    text = " ".join(t.get_text() for ax in figure.axes for t in ax.texts)
    # The class contrast at a level is a marker over its cluster, not a block
    # of text: one per cluster that holds two classes, and none over the two
    # that hold one. The effect sizes stay in the derivative.
    # On the real derivatives both are '***'; on this fixture's random data
    # they are legitimately 'n.s.', so the test counts markers rather than
    # assuming significance.
    markers = [t.get_text() for ax in figure.axes for t in ax.texts
               if t.get_text() in {"*", "**", "***", "n.s.", "n/a"}]
    assert len(markers) == 4, f"expected one marker per comparable cluster, got {markers}"
    labels = {
        t.get_text() for ax in figure.axes
        for t in (ax.get_legend().get_texts() if ax.get_legend() else [])
    }
    assert labels == {"easy", "ambiguous", "misleading"}, (
        "F11 draws the designed classes only; F10 is the figure that needs "
        "the unclassified stratum"
    )
    # Accuracy is read against chance, not zero.
    assert "chance" in text
    # The predictor's four-level nature is stated where a reader will find it.
    assert "not continuous" in metadata["caveat"]
    assert "n_predictor_levels" in metadata["caveat"]
    # Nothing is read from trialfeatures any more; the binning moved into the
    # analysis layer, so `requires` and `columns_read` can agree.
    assert set(metadata["columns_read"]) == {
        "continuousevidenceprofile", "continuousevidencestats",
    }


# --- sequential.py (F16, F17) -----------------------------------------------


def test_posterror_slowing(design_tables):
    figure, metadata = sequential.build_posterror_slowing(design_tables)
    _assert_basic_figure(figure, metadata, min_axes=2)


def test_choicehistory_effects(design_tables):
    figure, metadata = sequential.build_choicehistory_effects(design_tables)
    _assert_basic_figure(figure, metadata, min_axes=2)


# --- individual.py (F18) ----------------------------------------------------


def test_the_structure_figure_draws_the_matrix_and_every_scatter(design_tables):
    """The matrix and the scatters are one argument: at n = 32 an r can be
    outlier-driven and the matrix cannot show that, so a merged figure missing
    either half is back to the problem the merge was for."""
    figure, metadata = individual.build_individualcorrelations_structure(design_tables)
    correlations = design_tables.analysis("individualcorrelations")
    pairs = individual._scatter_pairs(correlations)

    # No `require_stat_text`: the fixture's four synthetic subjects reach
    # significance on nothing, so there is no marker to find -- correctly, and
    # the figure note below carries the conventions either way.
    _assert_basic_figure(figure, metadata, min_axes=6, require_stat_text=False)

    # The figure carries no note of its own, so the two things that make an
    # uncorrected matrix honest -- the multiplicity statement and what a blank
    # cell means -- have to survive in the sidecar for the caption to quote.
    assert metadata["multiplicity_correction"] == "none"
    assert "uncorrected" in metadata["caption_required"]
    assert "blank cell is tested and not significant" in metadata["caption_required"]

    # One labelled scatter per selected pair, plus the matrix, which labels
    # neither axis. Counting raw axes would also count the colorbar, whose
    # presence in `figure.axes` depends on how it was attached and says nothing
    # about whether the argument is complete.
    labelled = [
        ax for ax in figure.axes
        if ax.get_visible() and ax.get_xlabel() and ax.get_ylabel()
    ]
    assert len(labelled) == len(pairs)
    assert any(not ax.get_xlabel() for ax in figure.axes), "the matrix is missing"
    # Every scatter names the measures it draws, in this figure's vocabulary.
    for (measure_a, measure_b), ax in zip(pairs, labelled):
        assert ax.get_xlabel() == individual._axis_label(measure_a)
        assert ax.get_ylabel() == individual._axis_label(measure_b)
    assert metadata["scatter_pairs"] == [list(pair) for pair in pairs]
    # Every scatter is a cell of the matrix, so both halves must read the same
    # derivative and neither may invent a pair the other does not hold.
    for measure_a, measure_b in pairs:
        assert individual._correlation_row(correlations, measure_a, measure_b) is not None

    # An empty framed axes reads as a panel whose data failed to draw, and the
    # fixture's four synthetic subjects reach significance on nothing -- which
    # is the case the real cohort will hit under any stricter selection. The
    # spare cells are hidden, not left as empty frames.
    hidden = [ax for ax in figure.axes if not ax.get_visible()]
    assert len(hidden) == individual.SCATTER_PANEL_COUNT - len(pairs)


def test_the_scatters_are_the_strongest_significant_pairs_in_order(design_tables):
    """Which cells deserve magnifying is a property of the data, so the panels
    are a rule read off the derivative rather than a list that silently keeps
    drawing yesterday's answer."""
    correlations = pd.DataFrame(
        [
            # Big but not significant: must not be selected.
            {"measure_a": "mean_dt_ms", "measure_b": "lapse_rate",
             "pearson_r": 0.95, "pearson_p": 0.40, "n_subjects": 32},
            {"measure_a": "mean_dt_ms", "measure_b": "percent_correct",
             "pearson_r": -0.30, "pearson_p": 0.01, "n_subjects": 32},
            {"measure_a": "urgency_slope_per_second",
             "measure_b": "criterion_slope_sum_log_lr_per_second",
             "pearson_r": 0.70, "pearson_p": 0.001, "n_subjects": 32},
            {"measure_a": "mean_dt_ms", "measure_b": "sat_adjustment_ms",
             "pearson_r": -0.50, "pearson_p": 0.02, "n_subjects": 32},
        ]
    )

    pairs = individual._scatter_pairs(correlations)

    # Largest |r| first, and the r = .95 pair is absent because p = .40.
    assert pairs == [
        ("criterion_slope_sum_log_lr_per_second", "urgency_slope_per_second"),
        ("mean_dt_ms", "sat_adjustment_ms"),
        ("mean_dt_ms", "percent_correct"),
    ]

    # "None of them reached the bar" is a result, and panel A is the one that
    # can say it: the matrix is still drawn with no scatter beside it.
    none_significant = design_tables.analysis("individualcorrelations").copy()
    none_significant["pearson_p"] = 0.9

    assert individual._scatter_pairs(none_significant) == []


def test_a_pair_with_mean_dt_always_puts_it_on_x():
    """So every panel holding it reads as another view of the same speed
    factor rather than an unrelated plane."""
    assert individual._orient("percent_correct", "mean_dt_ms")[0] == "mean_dt_ms"
    assert individual._orient("mean_dt_ms", "percent_correct")[0] == "mean_dt_ms"
    # Without it, the longer label goes on x, where labels have room.
    x, y = individual._orient(
        "urgency_slope_per_second", "criterion_slope_sum_log_lr_per_second"
    )
    assert len(individual._axis_label(x)) > len(individual._axis_label(y))


def test_the_correlation_matrix_is_symmetric_whichever_way_a_pair_was_stored(
    design_tables,
):
    """The derivative stores each pair once, in whichever order the analysis
    emitted. The heatmap draws one triangle, so a pair stored the other way
    round used to land in the masked half and disappear -- as a blank cell
    reading "not measured" rather than "measured, near zero". Four of the
    twenty-one real pairs went that way."""
    correlations = pd.DataFrame(
        [
            # Deliberately stored with the *later* measure first, which is the
            # orientation that used to be dropped.
            {"measure_a": "percent_correct", "measure_b": "mean_dt_ms",
             "pearson_r": 0.46, "pearson_p": 0.008, "n_subjects": 32},
            {"measure_a": "mean_dt_ms", "measure_b": "lapse_rate",
             "pearson_r": 0.08, "pearson_p": 0.68, "n_subjects": 32},
        ]
    )

    r_matrix, p_matrix = individual._correlation_matrices(correlations)

    for matrix, forward, reverse in (
        (r_matrix, 0.46, 0.46), (p_matrix, 0.008, 0.008),
    ):
        assert matrix.loc["percent_correct", "mean_dt_ms"] == pytest.approx(forward)
        assert matrix.loc["mean_dt_ms", "percent_correct"] == pytest.approx(reverse)
    assert r_matrix.loc["lapse_rate", "mean_dt_ms"] == pytest.approx(0.08)
    assert r_matrix.loc["mean_dt_ms", "lapse_rate"] == pytest.approx(0.08)

    # Whatever triangle the heatmap draws, no pair the derivative holds may be
    # missing from it.
    correlations = design_tables.analysis("individualcorrelations")
    r_matrix, _ = individual._correlation_matrices(correlations)
    measures = list(r_matrix.columns)

    for _, row in correlations.iterrows():
        a, b = row["measure_a"], row["measure_b"]
        if a not in measures or b not in measures:
            continue
        lower = max(measures.index(a), measures.index(b))
        upper = min(measures.index(a), measures.index(b))
        # The cell the lower triangle actually renders for this pair.
        assert r_matrix.iloc[lower, upper] == pytest.approx(row["pearson_r"])
