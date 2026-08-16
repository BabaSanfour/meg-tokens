import json

import pytest

from meg_tokens.io import DerivativeLayout, save_table, sidecar_path
from meg_tokens.reports.behavior_summary import run_behavior_plotting

from tests.reports.factories import (
    ssmcomparison,
    ssmcomparisonstats,
    trial_features_group,
    write_group_derivative,
)


def _stage_headline_derivatives(tmp_path):
    layout = DerivativeLayout(tmp_path)
    fits = ssmcomparison()
    write_group_derivative(layout, "ssmcomparison", fits)
    write_group_derivative(layout, "ssmcomparisonstats", ssmcomparisonstats(fits))


def test_writes_pdf_png_and_one_json_per_figure(tmp_path):
    _stage_headline_derivatives(tmp_path)
    outputs = run_behavior_plotting(
        str(tmp_path), str(tmp_path), figures=("headline",)
    )
    assert len(outputs) == 2  # F01, F02
    for base in outputs:
        assert base.with_suffix(".pdf").is_file()
        assert base.with_suffix(".png").is_file()
        assert sidecar_path(base).is_file()


def test_desc_segments_match_sidecar_analysis_and_view(tmp_path):
    _stage_headline_derivatives(tmp_path)
    outputs = run_behavior_plotting(
        str(tmp_path), str(tmp_path), figures=("ssmcomparison-deltabic",)
    )
    sidecar = json.loads(sidecar_path(outputs[0]).read_text(encoding="utf-8"))
    desc = outputs[0].name.split("_desc-", 1)[1].rsplit("_behavior", 1)[0]
    assert desc == f"{sidecar['analysis']}-{sidecar['view']}"


def test_missing_derivative_raises_filenotfounderror_naming_characterization(tmp_path):
    # Stage the Stage-2 trial-feature table (behavior analyze's output) so the
    # first FileNotFoundError this hits is the missing Stage-2b `dtdistribution`
    # table specifically, not the earlier trial-feature dependency.
    layout = DerivativeLayout(tmp_path)
    save_table(
        layout.behavior_trial_features(),
        trial_features_group(),
        metadata={"stage": "test_fixture"},
    )
    with pytest.raises(FileNotFoundError, match="characterization"):
        run_behavior_plotting(str(tmp_path), str(tmp_path), figures=("core",))


def test_missing_derivative_raises_filenotfounderror_naming_ssmfit(tmp_path):
    with pytest.raises(FileNotFoundError, match="ssm-fit"):
        run_behavior_plotting(str(tmp_path), str(tmp_path), figures=("headline",))


def test_skip_missing_returns_a_shorter_list_without_raising(tmp_path):
    _stage_headline_derivatives(tmp_path)
    outputs = run_behavior_plotting(
        str(tmp_path), str(tmp_path), figures=("all",), skip_missing=True
    )
    # headline (2) written; core (dtdistribution-based) skipped for lack of data.
    assert 0 < len(outputs) < 4


def test_unknown_figure_key_raises_valueerror_listing_valid_keys(tmp_path):
    with pytest.raises(ValueError, match="ssmcomparison-deltabic"):
        run_behavior_plotting(str(tmp_path), str(tmp_path), figures=("not-a-real-key",))


def test_a_figures_own_sources_do_not_leak_across_figures_in_the_same_run(tmp_path):
    """Regression guard: BehaviorTableSet caches across the whole run, so a
    figure's sidecar must list only what its OWN builder reads, not every
    derivative any figure in this run happened to load."""
    from tests.reports.factories import (
        dtdistribution,
        dtdistributionstats,
        groupstats,
        trial_features_group,
    )

    _stage_headline_derivatives(tmp_path)
    layout = DerivativeLayout(tmp_path)
    dist = dtdistribution()
    write_group_derivative(layout, "dtdistribution", dist)
    write_group_derivative(layout, "dtdistributionstats", dtdistributionstats(dist))
    write_group_derivative(layout, "groupstats", groupstats())
    save_table(
        layout.behavior_trial_features(), trial_features_group(),
        metadata={"stage": "test_fixture"},
    )

    outputs = run_behavior_plotting(
        str(tmp_path), str(tmp_path),
        figures=("dtdistribution-condition",),
    )
    sidecar = json.loads(sidecar_path(outputs[0]).read_text(encoding="utf-8"))
    assert not any("ssmcomparison" in path for path in sidecar["source_derivatives"])
