"""The real dataset, for the tests whose claims are facts about it.

Found the way every command finds it: ``data_root`` in this checkout's own
``tokens.toml``. Any machine that has the data and a configured checkout runs
these tests; one without either skips them, naming what is missing.
"""

from pathlib import Path

import pytest

from meg_tokens.core import ProjectConfig

CONFIG = Path(__file__).resolve().parents[1] / "tokens.toml"


@pytest.fixture(scope="session")
def project() -> ProjectConfig:
    if not CONFIG.is_file():
        pytest.skip(
            f"no {CONFIG.name}: copy config/tokens.toml.template and set data_root "
            "to run the real-data tests"
        )
    return ProjectConfig.from_toml(CONFIG)


def _present(path: Path) -> Path:
    if not path.is_dir():
        pytest.skip(f"{path} is not on this machine")
    return path


@pytest.fixture(scope="session")
def raw_data(project) -> ProjectConfig:
    """Raw CTF recordings and behavioral logs, the two sides Stage 0 matches."""
    _present(project.raw_meg_root)
    _present(project.behavior_root)
    return project


@pytest.fixture(scope="session")
def behavior_logs(project) -> Path:
    return _present(project.behavior_root)


@pytest.fixture(scope="session")
def subjects_dir(project) -> Path:
    return _present(project.subjects_dir)
