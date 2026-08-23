"""Make the CLI package runnable as ``python -m meg_tokens.cli``.

``python -m meg_tokens.cli.main`` reaches the submodule directly, which this
package has already imported in ``__init__``, so Python re-executes it and
warns. Running the package instead avoids that. The installed ``meg-tokens``
console script remains the documented entry point; this exists so that a
``-m`` invocation behaves rather than misleading.
"""

from __future__ import annotations

from .main import main

if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
