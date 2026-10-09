
from meg_tokens.core import ProjectConfig
from meg_tokens.io import DerivativeLayout
from meg_tokens.workflows import analyze_behavior
from meg_tokens.workflows.behavior_characterization import (
    analyze_behavior_characterization,
)
from meg_tokens.workflows.sequential_sampling import fit_subject_sequential_sampling

from tests.behavior.factories import stage_behavior


# --- Stage 2b: behavioral characterization analyses ---------------------------


def test_characterization_pools_the_fits_without_republishing_them(tmp_path):
    """`ssm-aggregate` is the sole writer of every `ssm*` group derivative.

    Both pipelines once wrote six of these filenames, so whichever ran last
    silently won -- and they disagreed in substance, not just in timing:
    characterization pooled ungated per-subject fits and omitted the
    `ssm_mechanistic_model_comparison` block the Act 3 panel selects on. This
    test fails if a second writer is ever reintroduced, which is what keeps the
    run order from mattering again.
    """
    project = ProjectConfig(data_root=tmp_path)
    stage_behavior(project.bids_root)
    analyze_behavior(project)
    fit_subject_sequential_sampling(project)

    result = analyze_behavior_characterization(project)

    assert result.settings["sequential_sampling_subjects_missing"] == []
    # The fits are consumed, not republished: they reach `speciescomparison`,
    # which is written, while no `ssm*` group table is.
    assert "speciescomparison" in result.settings["analyses"]
    written = {path.name for path in result.outputs}
    assert not [name for name in written if "desc-ssm" in name], written
    assert not [name for name in result.settings["analyses"] if name.startswith("ssm")]

    # And the group tables genuinely are absent rather than merely unlisted.
    layout = DerivativeLayout(project.bids_root)
    for name in (
        "ssmcomparison",
        "ssmcomparisonstats",
        "ssmtimecourse",
        "ssmtrialpredictions",
    ):
        assert not layout.behavior_analysis(name).is_file(), name
