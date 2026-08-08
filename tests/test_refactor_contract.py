import ast
from pathlib import Path

from meg_tokens.cli import build_parser


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
