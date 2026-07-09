from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
