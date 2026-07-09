import ast
from pathlib import Path

from meg_tokens.cli import build_parser
from meg_tokens.workflows.connectivity import connectivity_derivative_path
from meg_tokens.workflows.decoding import decoding_derivative_path
from meg_tokens.workflows.decomposition import pca_derivative_path
from meg_tokens.workflows.erp import erp_derivative_path
from meg_tokens.workflows.statistics import stats_derivative_path
from meg_tokens.workflows.hilbert import hilbert_feature_derivative_path
from meg_tokens.workflows.pac import pac_derivative_path
from meg_tokens.workflows.spectral import psd_derivative_path
from meg_tokens.workflows.power import power_derivative_path


ROOT = Path(__file__).parents[1]


def _module_tree(relative_path: str) -> ast.Module:
    return ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))


def _project_imports(relative_path: str) -> set[str]:
    imports = set()
    for node in ast.walk(_module_tree(relative_path)):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("meg_tokens"):
            imports.add(node.module)
        elif isinstance(node, ast.Import):
            imports.update(
                alias.name
                for alias in node.names
                if alias.name.startswith("meg_tokens")
            )
    return imports


def test_compatibility_modules_are_removed():
    assert not list((ROOT / "meg_tokens" / "utils").glob("*.py"))
    assert not list((ROOT / "meg_tokens" / "cli").glob("legacy_*.py"))
    for path in (
        "meg_tokens/behavior/analysis.py",
        "meg_tokens/behavior/plotting.py",
        "meg_tokens/meg/erp.py",
        "meg_tokens/meg/time_frequency.py",
        "meg_tokens/meg/connectivity.py",
        "meg_tokens/meg/pac.py",
        "meg_tokens/meg/decoding.py",
        "meg_tokens/meg/pca.py",
        "meg_tokens/meg/stats.py",
        "meg_tokens/meg/plotting.py",
    ):
        assert not (ROOT / path).exists()


def test_final_package_dependency_direction():
    violations = []
    for path in (ROOT / "meg_tokens").rglob("*.py"):
        relative = path.relative_to(ROOT).as_posix()
        imports = _project_imports(relative)
        forbidden = {"meg_tokens.utils"}
        if relative.startswith("meg_tokens/core/"):
            forbidden.update(
                {
                    "meg_tokens.io",
                    "meg_tokens.behavior",
                    "meg_tokens.meg",
                    "meg_tokens.features",
                    "meg_tokens.analysis",
                    "meg_tokens.reports",
                    "meg_tokens.workflows",
                    "meg_tokens.cli",
                }
            )
        elif relative.startswith(
            (
                "meg_tokens/behavior/",
                "meg_tokens/meg/",
                "meg_tokens/features/",
                "meg_tokens/analysis/",
                "meg_tokens/reports/",
            )
        ):
            forbidden.update({"meg_tokens.workflows", "meg_tokens.cli"})
        elif relative.startswith("meg_tokens/workflows/"):
            forbidden.update({"meg_tokens.reports", "meg_tokens.cli"})
        for imported in imports:
            if any(
                imported == prefix or imported.startswith(prefix + ".")
                for prefix in forbidden
            ):
                violations.append((relative, imported))
    assert violations == []


def test_unified_cli_is_the_only_command_tree():
    parser = build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if hasattr(action, "choices") and action.choices
    )
    assert set(subparsers.choices) == {
        "behavior",
        "meg",
        "features",
        "analyze",
        "report",
        "validate",
    }


def test_representative_derivative_names_are_frozen():
    root = "/analysis"
    common = "/analysis/derivatives/meg-tokens"

    assert erp_derivative_path(
        root,
        subject="H1",
        run="Slow1",
        condition=None,
        align_to="go",
        source_method="dSPM",
        parc="HCPMMP1",
        suffix="erp",
        extension=".npy",
    ).as_posix() == (
        f"{common}/sub-H01/meg/"
        "sub-H01_task-tokens_run-1_desc-slow-go-dSPM-HCPMMP1_erp.npy"
    )
    assert power_derivative_path(
        root,
        subject="H1",
        run="Slow1",
        condition=None,
        align_to="go",
        source_method="dSPM",
        power_method="hilbert",
        band="gamma_low",
    ).as_posix() == (
        f"{common}/sub-H01/meg/"
        "sub-H01_task-tokens_run-1_desc-slow-go-dSPM-hilbert-gamma-low_power.npy"
    )
    assert psd_derivative_path(
        root,
        subject="H1",
        run="Slow1",
        condition="Slow",
        align_to="go",
        method="welch",
        fmin=1.0,
        fmax=100.0,
        suffix="psd",
        extension=".npy",
    ).as_posix() == (
        f"{common}/sub-H01/meg/"
        "sub-H01_task-tokens_run-1_desc-slow-go-welch-1to100hz_psd.npy"
    )
    assert stats_derivative_path(
        root,
        conditions=["Fast", "Slow"],
        align_to="go",
        source_method="dSPM",
        parc="HCPMMP1",
        suffix="tstat",
        extension=".npy",
    ).as_posix() == (
        f"{common}/sub-group/meg/"
        "sub-group_task-tokens_desc-fast-vs-slow-go-dSPM-HCPMMP1_tstat.npy"
    )
    assert decoding_derivative_path(
        root,
        conditions=["Fast", "Slow"],
        feature_source="erp",
        align_to="go",
        source_method="dSPM",
        parc="HCPMMP1",
        band=None,
        suffix="decodingscore",
        extension=".npy",
    ).as_posix() == (
        f"{common}/sub-group/meg/"
        "sub-group_task-tokens_desc-fast-vs-slow-erp-go-dSPM-HCPMMP1_decodingscore.npy"
    )
    assert pca_derivative_path(
        root,
        conditions=["Fast", "Slow"],
        feature_source="erp",
        align_to="go",
        source_method="dSPM",
        parc="HCPMMP1",
        band=None,
        suffix="pcatrajectory",
        extension=".npy",
    ).as_posix() == (
        f"{common}/sub-group/meg/"
        "sub-group_task-tokens_desc-fast-vs-slow-erp-go-dSPM-HCPMMP1-pca_pcatrajectory.npy"
    )
    assert connectivity_derivative_path(
        root,
        subject="H1",
        run="Slow1",
        condition="Slow",
        align_to="go",
        source_method="dSPM",
        parc="HCPMMP1",
        method="coh",
    ).as_posix() == (
        f"{common}/sub-H01/meg/"
        "sub-H01_task-tokens_run-1_desc-slow-go-dSPM-HCPMMP1-coh_connectivity.npy"
    )
    assert hilbert_feature_derivative_path(
        root,
        subject="H1",
        run="Slow1",
        condition="Slow",
        align_to="go",
        source_method="dSPM",
        parc="HCPMMP1",
        band="gamma_low",
        feature="phase",
    ).as_posix() == (
        f"{common}/sub-H01/meg/"
        "sub-H01_task-tokens_run-1_desc-slow-go-dSPM-HCPMMP1-gamma-low-phase_hilbertfeature.npy"
    )
    assert pac_derivative_path(
        root,
        subject="H1",
        run="Slow1",
        condition="Slow",
        align_to="go",
        source_method="dSPM",
        parc="HCPMMP1",
        phase_bands=["theta"],
        amplitude_bands=["gamma_low"],
        method="tort_mi",
    ).as_posix() == (
        f"{common}/sub-H01/meg/"
        "sub-H01_task-tokens_run-1_desc-slow-go-dSPM-HCPMMP1-theta-to-gamma-low-tort-mi_pac.npy"
    )
