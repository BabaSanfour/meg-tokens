from pathlib import Path

import pytest

from meg_tokens.io import DerivativeLayout


ROOT = Path(__file__).resolve().parents[1]

RUN_LABEL = "Fast3"
WILDCARDS = {"subject": "H09", "run_number": "3", "condition": "fast", "band": "alpha"}


def load_snakefile_constants(bids_root):
    """Execute the Snakefile's pure-Python header and return its namespace.

    Snakemake needs literal ``{wildcard}`` strings in rule outputs, so the
    path patterns cannot be produced by calling the layout builders and are
    necessarily restated. Executing the header lets the tests below compare
    those restatements against the layout that actually owns the paths.
    """
    text = (ROOT / "workflow" / "Snakefile").read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    header = [line for line in lines[: next(
        index for index, line in enumerate(lines) if line.startswith("rule ")
    )]]
    namespace = {
        "config": {
            "project_config": "tokens.toml",
            "bids_root": str(bids_root),
            "subjects": ["H09"],
            "runs": {"Fast": ["Fast3"], "Slow": ["Slow3"]},
        }
    }
    exec(compile("".join(header), str(ROOT / "workflow" / "Snakefile"), "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def snakefile(tmp_path_factory):
    return load_snakefile_constants(tmp_path_factory.mktemp("bids"))


def assert_pattern_matches_layout(snakefile, name, expected):
    """Substitute the concrete wildcard values and compare against the layout."""
    actual = snakefile[name].format(**WILDCARDS)
    assert actual == str(expected), (
        f"{name} has drifted from DerivativeLayout.\n"
        f"  Snakefile pattern -> {actual}\n"
        f"  DerivativeLayout  -> {expected}"
    )


def test_snakefile_uses_unified_cli_and_no_compatibility_modules():
    snakefile = (ROOT / "workflow" / "Snakefile").read_text(encoding="utf-8")

    assert "meg-tokens --config" in snakefile
    assert "meg_tokens.utils" not in snakefile
    for stage in (
        "behavior ingest",
        "meg preprocess",
        "meg epoch",
        "meg source",
        "features erp",
        "features power",
        "analyze statistics",
        "analyze decoding",
        "analyze decomposition",
        "report behavior",
        "report statistics",
    ):
        assert stage in snakefile


def test_snakefile_declares_persisted_stage_dependencies():
    snakefile = (ROOT / "workflow" / "Snakefile").read_text(encoding="utf-8")

    assert "rules.source_models.output" in snakefile
    assert "SOURCE_MANIFEST_PATTERN" in snakefile
    assert "ERP_TARGETS" in snakefile
    assert "POWER_TARGETS" in snakefile
    assert "STATS_TARGET" in snakefile
    assert "DECODING_TARGET" in snakefile
    assert "PCA_TARGET" in snakefile


def test_cluster_launchers_delegate_to_snakemake_profiles():
    local = (ROOT / "cluster" / "run_workflow_local.sh").read_text(encoding="utf-8")
    slurm = (ROOT / "cluster" / "submit_workflow.sh").read_text(encoding="utf-8")

    assert "--profile workflow/profiles/local" in local
    assert "--profile workflow/profiles/slurm" in slurm
    assert "python -m meg_tokens.utils" not in local + slurm


def test_snakefile_derivative_root_matches_the_path_contract(snakefile):
    layout = DerivativeLayout(snakefile["BIDS_ROOT"])
    contract_root = layout.path(
        subject="H09", suffix="raw", extension=".fif"
    ).parents[2]

    assert Path(snakefile["DERIVATIVES"]) == contract_root, (
        "DERIVATIVES has drifted from the derivative_path contract.\n"
        f"  Snakefile         -> {snakefile['DERIVATIVES']}\n"
        f"  derivative_path   -> {contract_root}"
    )


def test_snakefile_state_directory_cannot_collide_with_a_subject(snakefile):
    state = Path(snakefile["STATE"])
    derivatives = Path(snakefile["DERIVATIVES"])

    assert derivatives in state.parents or state.parent == derivatives
    assert state.name.startswith("."), (
        "STATE must be a dot-prefixed marker directory so it can never be "
        f"mistaken for a sub-* derivative directory, got {state}"
    )
    assert not state.name.startswith("sub-")


def test_clean_pattern_matches_the_layout(snakefile):
    layout = DerivativeLayout(snakefile["BIDS_ROOT"])
    assert_pattern_matches_layout(
        snakefile,
        "CLEAN_PATTERN",
        layout.preprocessed_raw(
            subject=WILDCARDS["subject"],
            run=RUN_LABEL,
            condition=WILDCARDS["condition"],
            processing=snakefile["PROCESSING"],
        ),
    )


def test_source_manifest_pattern_matches_the_layout(snakefile):
    layout = DerivativeLayout(snakefile["BIDS_ROOT"])
    assert_pattern_matches_layout(
        snakefile,
        "SOURCE_MANIFEST_PATTERN",
        layout.source(
            subject=WILDCARDS["subject"],
            run=RUN_LABEL,
            condition=WILDCARDS["condition"],
            description=f"{snakefile['ALIGNMENT']}-{snakefile['SOURCE_METHOD']}",
            suffix="stcmanifest",
            extension=".tsv",
        ),
    )


def test_erp_pattern_matches_the_layout(snakefile):
    layout = DerivativeLayout(snakefile["BIDS_ROOT"])
    assert_pattern_matches_layout(
        snakefile,
        "ERP_PATTERN",
        layout.erp(
            subject=WILDCARDS["subject"],
            run=RUN_LABEL,
            condition=WILDCARDS["condition"],
            alignment=snakefile["ALIGNMENT"],
            source_method=snakefile["SOURCE_METHOD"],
            parc=snakefile["PARC"],
            suffix="erp",
            extension=".npy",
        ),
    )


def test_power_pattern_matches_the_layout(snakefile):
    layout = DerivativeLayout(snakefile["BIDS_ROOT"])
    assert_pattern_matches_layout(
        snakefile,
        "POWER_PATTERN",
        layout.power(
            subject=WILDCARDS["subject"],
            run=RUN_LABEL,
            condition=WILDCARDS["condition"],
            alignment=snakefile["ALIGNMENT"],
            source_method=snakefile["SOURCE_METHOD"],
            power_method=snakefile["POWER_METHOD"],
            band=WILDCARDS["band"],
        ),
    )


@pytest.mark.parametrize(
    "name, key",
    [("NOISE_COV_PATTERN", "cov"), ("BEM_PATTERN", "bem"), ("SRC_PATTERN", "src")],
)
def test_source_model_patterns_match_the_layout(snakefile, name, key):
    layout = DerivativeLayout(snakefile["BIDS_ROOT"])
    models = layout.source_models(
        subject=WILDCARDS["subject"], spacing=snakefile["SPACING"]
    )
    assert_pattern_matches_layout(snakefile, name, models[key])
