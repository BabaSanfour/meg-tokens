from pathlib import Path


PROHIBITED_BATCH_PATTERNS = (
    "mo" + "ck",
    "sim" + "ulat",
    "demon" + "stration",
    "dummy" + " data",
    "np.random." + "normal",
    "np.random." + "randn",
    "np.random." + "uniform",
)


def test_batch_utilities_do_not_generate_fake_project_data():
    utils_dir = Path("meg_tokens/utils")
    offenders = []

    for path in sorted(utils_dir.glob("*.py")):
        text = path.read_text(encoding="utf-8").lower()
        for pattern in PROHIBITED_BATCH_PATTERNS:
            if pattern in text:
                offenders.append(f"{path}: {pattern}")

    assert offenders == []
