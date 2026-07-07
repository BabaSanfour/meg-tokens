from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = REPO_ROOT / "archive" / "replicated"
TRACEABILITY_PATH = REPO_ROOT / "docs" / "legacy_traceability.md"
EXECUTABLE_SUFFIXES = {".ipynb", ".py", ".m", ".sh"}
ALLOWED_STATUSES = {"implemented", "partial", "out_of_scope", "archive"}


def _legacy_executables() -> list[Path]:
    return sorted(
        path
        for path in LEGACY_ROOT.rglob("*")
        if path.is_file() and path.suffix in EXECUTABLE_SUFFIXES
    )


def _matrix_rows(text: str) -> list[list[str]]:
    rows = []
    for line in text.splitlines():
        if not line.startswith("| [`archive/replicated/"):
            continue
        rows.append([cell.strip() for cell in line.split("|")[1:-1]])
    return rows


def test_legacy_traceability_covers_every_legacy_executable_once():
    text = TRACEABILITY_PATH.read_text(encoding="utf-8")

    missing = []
    duplicated = []
    for path in _legacy_executables():
        rel = path.relative_to(REPO_ROOT).as_posix()
        marker = f"[`{rel}`]"
        count = text.count(marker)
        if count == 0:
            missing.append(rel)
        elif count > 1:
            duplicated.append(rel)

    assert missing == []
    assert duplicated == []
    assert f"Total legacy executables covered: `{len(_legacy_executables())}`." in text


def test_legacy_traceability_statuses_and_summary_are_consistent():
    text = TRACEABILITY_PATH.read_text(encoding="utf-8")
    rows = _matrix_rows(text)
    statuses = Counter()

    for row in rows:
        status = row[2].strip("`")
        assert status in ALLOWED_STATUSES
        statuses[status] += 1

        replacement = row[3]
        validation = row[4]
        if status in {"implemented", "partial"}:
            assert "No production command" not in replacement
            assert validation != "Traceability coverage only"
        else:
            assert validation == "Traceability coverage only"

    assert sum(statuses.values()) == len(_legacy_executables())
    for status in ALLOWED_STATUSES:
        assert f"- `{status}`: {statuses[status]}" in text
