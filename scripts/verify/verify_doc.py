"""Assert every number the new F10 section quotes against the derivative."""

from scripts.verify._root import group_beh

import numpy as np
import pandas as pd

ROOT = group_beh()
stats = pd.read_csv(ROOT / "sub-group_task-tokens_desc-conditionalaccuracystats_beh.tsv",
                    sep="\t")
functions = pd.read_csv(ROOT / "sub-group_task-tokens_desc-conditionalaccuracy_beh.tsv",
                        sep="\t")

failures = []


def check(label, actual, expected, tol):
    if abs(float(actual) - expected) > tol:
        failures.append(f"{label}: derivative {float(actual):.6g} vs doc {expected}")


def slope(term, stratum, measure="accuracy", condition="all", min_cells=1):
    row = stats.loc[
        (stats["term"] == term) & (stats["measure"] == measure)
        & (stats["condition"] == condition) & (stats["trial_class_name"] == stratum)
        & (stats["min_cell_trials"] == min_cells)
    ]
    assert len(row) == 1, (term, stratum, measure, condition, len(row))
    return row.iloc[0]


def bin_row(stratum, dt_bin):
    row = stats.loc[
        (stats["term"] == "mean_accuracy") & (stats["condition"] == "all")
        & (stats["trial_class_name"] == stratum) & (stats["dt_bin"] == dt_bin)
    ]
    assert len(row) == 1
    return row.iloc[0]


# --- the pooled table -------------------------------------------------------
for dt_bin, dt, acc, lo, hi in (
    (1, 678, 0.897, 0.880, 0.914),
    (2, 964, 0.872, 0.857, 0.887),
    (3, 1183, 0.843, 0.826, 0.861),
    (4, 1444, 0.798, 0.775, 0.822),
    (5, 1949, 0.777, 0.758, 0.795),
):
    row = bin_row("all", dt_bin)
    check(f"pooled bin{dt_bin} dt", row["mean_dt_ms"], dt, 0.5)
    check(f"pooled bin{dt_bin} acc", row["estimate"], acc, 5e-4)
    check(f"pooled bin{dt_bin} ci_low", row["ci_low"], lo, 5e-4)
    check(f"pooled bin{dt_bin} ci_high", row["ci_high"], hi, 5e-4)

check("mean trials per pooled bin", functions.loc[
    (functions["condition"] == "all") & (functions["trial_class_name"] == "all"),
    "n_trials"].mean(), 102, 0.5)

# --- pooled slope, both units ----------------------------------------------
row = slope("accuracy_slope_per_bin", "all")
check("pooled slope/bin", row["estimate"], -0.0314, 5e-5)
check("pooled slope/bin ci_low", row["ci_low"], -0.0372, 5e-5)
check("pooled slope/bin ci_high", row["ci_high"], -0.0255, 5e-5)
check("pooled slope/bin t", row["t"], -10.92, 5e-3)
check("pooled slope/bin p", row["p"], 3.8e-12, 5e-14)

row = slope("accuracy_slope_per_second", "all")
check("pooled slope/s", row["estimate"], -0.112, 5e-4)
check("pooled slope/s ci_low", row["ci_low"], -0.136, 5e-4)
check("pooled slope/s ci_high", row["ci_high"], -0.088, 5e-4)
check("pooled slope/s t", row["t"], -9.53, 5e-3)
check("pooled slope/s p", row["p"], 9.9e-11, 5e-13)

# --- per class --------------------------------------------------------------
for stratum, first, last, est, lo, hi, p in (
    ("easy", 0.996, 0.900, -0.0203, -0.0290, -0.0116, 4.3e-5),
    ("ambiguous", 0.726, 0.772, +0.0013, -0.0180, +0.0207, 0.89),
    ("misleading", 0.085, 0.599, +0.1288, +0.1104, +0.1471, 3.4e-15),
    ("unclassified", 0.963, 0.807, -0.0391, -0.0474, -0.0309, 7.1e-11),
):
    check(f"{stratum} bin1", bin_row(stratum, 1)["estimate"], first, 5e-4)
    check(f"{stratum} bin5", bin_row(stratum, 5)["estimate"], last, 5e-4)
    row = slope("accuracy_slope_per_bin", stratum)
    check(f"{stratum} slope", row["estimate"], est, 5e-5)
    check(f"{stratum} ci_low", row["ci_low"], lo, 5e-5)
    check(f"{stratum} ci_high", row["ci_high"], hi, 5e-5)
    check(f"{stratum} p", row["p"], p, abs(p) * 0.02 + 5e-3 * (p > 0.01))

check("ambiguous bound", slope("accuracy_slope_per_bin", "ambiguous")["equivalence_bound"],
      0.0174, 5e-5)
row = slope("accuracy_slope_per_bin", "ambiguous", min_cells=5)
check("ambiguous min5", row["estimate"], -0.0087, 5e-5)
check("ambiguous min5 ci_low", row["ci_low"], -0.0237, 5e-5)
check("ambiguous min5 ci_high", row["ci_high"], +0.0063, 5e-5)
check("ambiguous min5 p", row["p"], 0.25, 5e-3)

# misleading against chance at the two ends
row = bin_row("misleading", 1)
check("misleading bin1 t", row["t"], -21.88, 5e-3)
check("misleading bin1 p", row["p"], 2.1e-20, 5e-22)
row = bin_row("misleading", 5)
check("misleading bin5 t", row["t"], 3.62, 5e-3)
check("misleading bin5 p", row["p"], 0.0010, 5e-5)

# --- composition ------------------------------------------------------------
composition = stats.loc[
    (stats["term"] == "class_proportion_first_to_last") & (stats["condition"] == "all")
].set_index("trial_class_name")
for stratum, first, last, delta, p in (
    ("easy", 0.432, 0.104, -0.328, 9.8e-18),
    ("ambiguous", 0.076, 0.341, +0.264, 1.4e-17),
    ("misleading", 0.072, 0.147, +0.075, 1.6e-8),
    ("unclassified", 0.420, 0.408, -0.012, 0.52),
):
    row = composition.loc[stratum]
    check(f"{stratum} share bin1", row["first_bin_proportion"], first, 5e-4)
    check(f"{stratum} share bin5", row["last_bin_proportion"], last, 5e-4)
    check(f"{stratum} share delta", row["estimate"], delta, 5e-4)
    check(f"{stratum} share p", row["p"], p, abs(p) * 0.02 + 5e-3 * (p > 0.01))

# --- class adjustment -------------------------------------------------------
row = slope("accuracy_slope_per_bin", "all", measure="accuracy_class_adjusted")
check("adjusted slope", row["estimate"], -0.0067, 5e-5)
check("adjusted ci_low", row["ci_low"], -0.0109, 5e-5)
check("adjusted ci_high", row["ci_high"], -0.0024, 5e-5)
check("adjusted t", row["t"], -3.18, 5e-3)
check("adjusted p", row["p"], 0.0033, 5e-5)
share = 1 - abs(row["estimate"]) / abs(slope("accuracy_slope_per_bin", "all")["estimate"])
check("composition share of the slope", share, 0.79, 5e-3)

# --- the first-to-last decomposition ----------------------------------------
def decomposition(measure, stratum):
    row = stats.loc[
        (stats["term"] == "accuracy_change_decomposition")
        & (stats["measure"] == measure) & (stats["condition"] == "all")
        & (stats["trial_class_name"] == stratum)
    ]
    assert len(row) == 1, (measure, stratum, len(row))
    return row.iloc[0]


row = decomposition("composition", "all")
check("observed first-to-last change", row["observed_change"], -0.1201, 5e-4)
check("composition term", row["estimate"], -0.0937, 5e-4)
check("composition ci_low", row["ci_low"], -0.1172, 5e-4)
check("composition ci_high", row["ci_high"], -0.0702, 5e-4)
check("composition t", row["t"], -8.14, 5e-3)
check("composition share", row["share_of_observed_change"], 0.781, 5e-4)
row = decomposition("within_class", "all")
check("within-class term", row["estimate"], -0.0263, 5e-4)
check("within-class ci_low", row["ci_low"], -0.0539, 5e-4)
check("within-class ci_high", row["ci_high"], +0.0012, 5e-4)
check("within-class p", row["p"], 0.060, 5e-4)
for stratum, comp_term, within in (
    ("easy", -0.3096, -0.0278),
    ("ambiguous", +0.1973, +0.0097),
    ("misleading", +0.0277, +0.0558),
    ("unclassified", -0.0091, -0.0641),
):
    check(f"{stratum} composition", decomposition("composition", stratum)["estimate"],
          comp_term, 5e-4)
    check(f"{stratum} within-class", decomposition("within_class", stratum)["estimate"],
          within, 5e-4)
row = decomposition("within_class", "designed_classes")
check("designed-classes within", row["estimate"], +0.0377, 5e-4)
check("designed-classes ci_low", row["ci_low"], +0.0156, 5e-4)
check("designed-classes ci_high", row["ci_high"], +0.0599, 5e-4)
check("designed-classes t", row["t"], 3.48, 5e-3)
check("designed-classes p", row["p"], 0.0015, 5e-5)
# --- the two routes to the composition share --------------------------------
# Whether they agree is a registered test, not arithmetic the doc performs, so
# the derivative's own rows are what is checked here. The arithmetic is checked
# too: each route's bootstrap point estimate has to be the number the rest of
# this table already reports, or the test compares something nobody claimed.
def agreement(measure, slope_unit="accuracy_slope_per_bin"):
    row = stats.loc[
        (stats["term"] == "composition_share_agreement")
        & (stats["measure"] == measure) & (stats["condition"] == "all")
        & (stats["slope_unit"] == slope_unit)
    ]
    assert len(row) == 1, (measure, slope_unit, len(row))
    return row.iloc[0]


slope_share = 1 - abs(slope("accuracy_slope_per_bin", "all",
                            measure="accuracy_class_adjusted")["estimate"]) / abs(
    slope("accuracy_slope_per_bin", "all")["estimate"])
check("slope-route share", slope_share, 0.788, 5e-4)

row = agreement("endpoint_share")
check("endpoint-route share", row["estimate"], 0.781, 5e-4)
check("endpoint-route share ci_low", row["ci_low"], 0.61, 5e-3)
check("endpoint-route share ci_high", row["ci_high"], 1.00, 5e-3)
check("endpoint route equals the decomposition's own share", row["estimate"],
      float(decomposition("composition", "all")["share_of_observed_change"]), 1e-9)
row = agreement("slope_share")
check("slope-route share, registered", row["estimate"], 0.788, 5e-4)
check("slope-route share ci_low", row["ci_low"], 0.70, 5e-3)
check("slope-route share ci_high", row["ci_high"], 0.90, 5e-3)
check("slope route equals the arithmetic on the two slopes", row["estimate"],
      slope_share, 1e-9)
row = agreement("endpoint_minus_slope_share")
check("route difference", row["estimate"], -0.007, 5e-4)
check("route difference ci_low", row["ci_low"], -0.155, 5e-3)
check("route difference ci_high", row["ci_high"], +0.165, 5e-3)
check("route difference p", row["p"], 0.89, 5e-3)
check("route difference bound", row["equivalence_bound"], 0.135, 5e-3)
check("route difference, per-second slopes",
      agreement("endpoint_minus_slope_share",
                "accuracy_slope_per_second")["estimate"], -0.008, 5e-4)
check("route difference is the difference of the two routes", row["estimate"],
      float(agreement("endpoint_share")["estimate"]
            - agreement("slope_share")["estimate"]), 1e-9)

# --- the pooled line is the class lines weighted by their share --------------
pooled_rows = functions.loc[(functions["condition"] == "all")]
share = pooled_rows.pivot_table(index="dt_bin", columns="trial_class_name",
                                values="proportion_of_bin")
acc = pooled_rows.pivot_table(index="dt_bin", columns="trial_class_name",
                              values="accuracy")
strata = ["easy", "ambiguous", "misleading", "unclassified"]
rebuilt = (share[strata] * acc[strata]).sum(axis=1)
observed = acc["all"]
for dt_bin, expected in zip(range(1, 6), (0.896, 0.873, 0.841, 0.794, 0.774)):
    check(f"bin{dt_bin} rebuilt from four strata", rebuilt.loc[dt_bin], expected, 5e-4)
three = ((share[strata[:3]] * acc[strata[:3]]).sum(axis=1)
         / share[strata[:3]].sum(axis=1))
check("three-stratum slope", np.polyfit(range(1, 6), three, 1)[0], -0.027, 5e-4)
missing = 1 - share[strata[:3]].sum(axis=1)
check("least weight missing", missing.min(), 0.41, 5e-3)
check("most weight missing", missing.max(), 0.45, 5e-3)

# --- Fast/Slow bounds -------------------------------------------------------
def contrast(measure):
    row = stats.loc[
        (stats["term"] == "accuracy_slope_fast_minus_slow")
        & (stats["slope_unit"] == "accuracy_slope_per_bin")
        & (stats["measure"] == measure) & (stats["trial_class_name"] == "all")
        & (stats["min_cell_trials"] == 1)
    ]
    assert len(row) == 1
    return row.iloc[0]


row = contrast("accuracy")
check("fast-slow pooled", row["estimate"], +0.0005, 5e-5)
check("fast-slow pooled ci_low", row["ci_low"], -0.0048, 5e-5)
check("fast-slow pooled ci_high", row["ci_high"], +0.0057, 5e-5)
check("fast-slow bound ratio", row["equivalence_bound"]
      / abs(slope("accuracy_slope_per_bin", "all")["estimate"]), 0.18, 5e-3)
row = contrast("accuracy_class_adjusted")
check("fast-slow adjusted", row["estimate"], +0.0017, 5e-5)
check("fast-slow adjusted ci_low", row["ci_low"], -0.0026, 5e-5)
check("fast-slow adjusted ci_high", row["ci_high"], +0.0059, 5e-5)

if failures:
    print("MISMATCHES:")
    for line in failures:
        print(" ", line)
    raise SystemExit(1)
print("all quoted values match the derivative")
