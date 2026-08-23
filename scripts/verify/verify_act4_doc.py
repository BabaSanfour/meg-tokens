"""Check every number the F16 and F17 entries in docs/behavior.md quote."""

from scripts.verify._root import group_beh
import math

import pandas as pd


root = group_beh()
failures: list[str] = []


def p_tolerance(doc: float) -> float:
    """Half the last digit of a two-significant-figure p-value.

    A flat percentage is wrong here: 1.1e-8 quoted against a derivative of
    1.0646e-8 is correct rounding, but 2% of 1.1e-8 is smaller than the
    rounding step itself.
    """
    return 0.05 * 10 ** math.floor(math.log10(doc))


def check(name: str, derivative: float, doc: float, tol: float) -> None:
    if abs(float(derivative) - doc) > tol:
        failures.append(f"{name}: derivative {derivative} vs doc {doc}")


# --- F16: post-error slowing, decomposed -------------------------------------
pe = pd.read_csv(root / "sub-group_task-tokens_desc-posterrorstats_beh.tsv", sep="\t")
pooled = pe[pe.condition == "all"].set_index("measure")

# measure -> (mean, ci_low, ci_high, t, p, dz)
expected = {
    "post_error_minus_baseline_ms": (105.7, 74.5, 136.9, 6.90, 9.6e-8, 1.22),
    "pre_error_minus_baseline_ms": (-38.3, -58.7, -18.0, -3.84, 5.7e-4, -0.68),
    "robust_pes_ms": (144.0, 105.9, 182.1, 7.71, 1.1e-8, None),
    "classical_pes_ms": (117.6, 80.9, 154.3, 6.54, 2.7e-7, None),
}
for measure, (mean, low, high, t, p, dz) in expected.items():
    row = pooled.loc[measure]
    check(f"{measure} mean", row["mean"], mean, 0.1)
    check(f"{measure} ci low", row["ci95_low"], low, 0.1)
    check(f"{measure} ci high", row["ci95_high"], high, 0.1)
    check(f"{measure} t", row["t"], t, 0.005)
    check(f"{measure} p", row["p"], p, p_tolerance(p))
    if dz is not None:
        check(f"{measure} dz", row["cohens_dz"], dz, 0.005)

# The doc's "27% of it is pre-error speeding" is arithmetic on those means, and
# the composite is their difference by construction.
post = float(pooled.loc["post_error_minus_baseline_ms", "mean"])
pre = float(pooled.loc["pre_error_minus_baseline_ms", "mean"])
composite = float(pooled.loc["robust_pes_ms", "mean"])
check("local pair = post - pre", post - pre, composite, 1e-6)
check("pre-error share", 100.0 * abs(pre) / composite, 27.0, 0.5)

contrast = pe[pe.condition == "fast_vs_slow"].set_index("measure")
row = contrast.loc["post_error_minus_baseline_ms"]
check("Fast-Slow slowing", row["mean_difference"], 14.5, 0.1)
check("Fast-Slow bound", row["equivalence_bound"], 48.6, 0.1)
check("Fast-Slow p", row["p"], 0.47, 0.01)

subjects = pd.read_csv(root / "sub-group_task-tokens_desc-posterror_beh.tsv", sep="\t")
subjects = subjects[subjects.condition == "all"]
check("error pairs min", subjects["n_error_pairs"].min(), 47, 0)
check("error pairs max", subjects["n_error_pairs"].max(), 109, 0)
check("error pairs median", subjects["n_error_pairs"].median(), 81.5, 0)

# The measure agreement is a registered row, not arithmetic this script does.
# It used to be recomputed here, which made three implementations of the same
# number -- this script, the figure's sidecar, and the doc -- with nothing
# holding them together.
agreement = pooled.loc["robust_vs_classical_agreement"]
check("measure agreement r", agreement["pearson_r"], 0.82, 0.005)
check("local pair positive", agreement["robust_positive"], 32, 0)
check("classical positive", agreement["classical_positive"], 29, 0)
check("agreement subjects", agreement["n_subjects"], 32, 0)

# --- F17: choice history -----------------------------------------------------
ch = pd.read_csv(
    root / "sub-group_task-tokens_desc-choicehistorystats_beh.tsv", sep="\t"
)
pooled = ch[ch.condition == "all"].set_index("measure")

stay = pooled.loc["win_stay_vs_lose_stay"]
check("win-stay", stay["mean_a"], 0.498, 0.0005)
check("lose-stay", stay["mean_b"], 0.441, 0.0005)
check("stay difference", stay["mean_difference"], 0.057, 0.0005)
check("stay t", stay["t"], 5.10, 0.005)
check("stay p", stay["p"], 1.6e-5, p_tolerance(1.6e-5))
check("stay dz", stay["cohens_dz"], 0.90, 0.005)

auto = pooled.loc["side_autocorrelation_lag1"]
check("autocorrelation", auto["mean"], -0.029, 0.0005)
check("autocorrelation ci low", auto["ci95_low"], -0.052, 0.0005)
check("autocorrelation ci high", auto["ci95_high"], -0.007, 0.0005)
check("autocorrelation t", auto["t"], -2.65, 0.005)
check("autocorrelation p", auto["p"], 0.012, 0.0005)
check("autocorrelation dz", auto["cohens_dz"], -0.47, 0.005)

history = pd.read_csv(
    root / "sub-group_task-tokens_desc-choicehistory_beh.tsv", sep="\t"
)
history = history[history.condition == "all"]
check(
    "autocorrelation negative subjects",
    (history["side_autocorrelation_lag1"] < 0).sum(),
    24,
    0,
)

if failures:
    print("MISMATCHES:")
    for line in failures:
        print(" ", line)
    raise SystemExit(1)
print("every quoted F16 and F17 value matches its derivative")
