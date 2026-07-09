from pathlib import Path


PROHIBITED_PRODUCTION_PATTERNS = (
    "mo" + "ck",
    "sim" + "ulat",
    "demon" + "stration",
    "dummy" + " data",
    "np.random." + "normal",
    "np.random." + "randn",
    "np.random." + "uniform",
)


def test_production_modules_do_not_generate_fake_project_data():
    offenders = []

    for path in sorted(Path("meg_tokens").rglob("*.py")):
        text = path.read_text(encoding="utf-8").lower()
        for pattern in PROHIBITED_PRODUCTION_PATTERNS:
            if pattern in text:
                offenders.append(f"{path}: {pattern}")

    assert offenders == []


def test_production_modules_do_not_write_matlab_or_hdf5_data():
    prohibited = (".savemat(", "h5py.", "to_hdf(", "format=\"hdf5\"")
    offenders = []
    for path in sorted(Path("meg_tokens").rglob("*.py")):
        text = path.read_text(encoding="utf-8").lower()
        offenders.extend(
            f"{path}: {pattern}"
            for pattern in prohibited
            if pattern in text
        )
    assert offenders == []
