import matplotlib
import matplotlib.pyplot as plt
import pytest

matplotlib.use("Agg", force=True)


@pytest.fixture(autouse=True)
def _close_all_figures():
    yield
    plt.close("all")
