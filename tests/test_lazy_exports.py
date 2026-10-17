"""Every lazy export names an object that exists.

The packages export through a string table (`_EXPORTS`), which no linter reads:
a function removed or renamed leaves its entry behind, and the error appears
only when someone uses that name. This resolves every entry.
"""

from importlib import import_module
from pkgutil import iter_modules

import pytest

import meg_tokens

PACKAGES = [
    module for module in (
        import_module(f"meg_tokens.{info.name}")
        for info in iter_modules(meg_tokens.__path__) if info.ispkg
    )
    if hasattr(module, "_EXPORTS")
]


@pytest.mark.parametrize(
    ("package", "name"),
    [(package, name) for package in PACKAGES for name in package._EXPORTS],
    ids=lambda value: getattr(value, "__name__", value),
)
def test_lazy_export_resolves(package, name):
    getattr(package, name)
