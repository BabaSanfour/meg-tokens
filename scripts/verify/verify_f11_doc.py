"""Assert every number the F11 section quotes against its derivatives."""

from scripts.verify._root import group_beh

import numpy as np
import pandas as pd


R = group_beh()
profile = pd.read_csv(R / "sub-group_task-tokens_desc-continuousevidenceprofile_beh.tsv",
                      sep="\t")
stats = pd.read_csv(R / "sub-group_task-tokens_desc-continuousevidencestats_beh.tsv",
                    sep="\t")
fits = pd.read_csv(R / "sub-group_task-tokens_desc-continuousevidence_beh.tsv", sep="\t")
features = pd.read_csv(R / "sub-group_task-tokens_desc-trialfeatures_beh.tsv",
                       sep="\t", low_memory=False)

LEVELS = [0.193848, 0.387207, 0.612793, 0.806152]
failures = []


def check(label, actual, expected, tol):
    if pd.isna(actual) or abs(float(actual) - expected) > tol:
        failures.append(f"{label}: derivative {actual} vs doc {expected}")


def rel(p):
    return abs(p) * 0.02 + (5e-4 if p > 0.01 else 0.0)


pooled = profile.loc[profile["condition"] == "all"]

# --- the predictor's support ------------------------------------------------
levels = sorted(pooled["sp_design_early"].unique())
check("number of levels", len(levels), 4, 0)
for index, expected in enumerate(LEVELS):
    check(f"level {index}", levels[index], expected, 5e-6)
logit = np.log(pooled["sp_design_early"] / (1 - pooled["sp_design_early"]))
assert np.allclose(logit, pooled["sum_log_lr_design_early"]), \
    "sum_log_lr_design_early is not logit(sp_design_early)"

# the 0-3 level's size
lowest = pooled.loc[np.isclose(pooled["sp_design_early"], LEVELS[0])
                    & (pooled["trial_class_name"] == "all")]
check("0-3 level trials", lowest["n_trials"].sum(), 12, 0)
check("0-3 level subjects", lowest["subject"].nunique(), 7, 0)
shared = pooled.loc[np.isclose(pooled["sp_design_early"], LEVELS[2])
                    & (pooled["trial_class_name"] == "all")]
check("shared level trials", shared["n_trials"].sum(), 11196, 0)

# --- commitment timing ------------------------------------------------------
primary = features.loc[features["primary_analysis_eligible"] == True]  # noqa: E712
k = pd.to_numeric(primary["decision_token_index"], errors="coerce")
check("still running at jump 3", 100 * (k > 3).mean(), 71.4, 0.05)
check("still running at jump 5", 100 * (k > 5).mean(), 45.0, 0.05)
check("still running at jump 8", 100 * (k > 8).mean(), 16.6, 0.05)

# --- SP < 0.5 iff misleading ------------------------------------------------
below = primary.loc[primary["sp_design_early"] < 0.5, "trial_class_name"]
check("misleading trials total", (primary["trial_class_name"] == "misleading").sum(),
      1718, 0)
# All misleading trials are below chance; no easy or unclassified trial is.
assert (primary.loc[primary["trial_class_name"] == "misleading",
                    "sp_design_early"] < 0.5).all()
assert set(below) == {"ambiguous", "misleading"}, sorted(set(below))
mixed = primary.loc[np.isclose(primary["sp_design_early"], LEVELS[1]),
                    "trial_class_name"].value_counts()
check("misleading at the mixed level", mixed["misleading"], 1706, 0)
check("ambiguous at the mixed level", mixed["ambiguous"], 641, 0)
top = set(primary.loc[np.isclose(primary["sp_design_early"], LEVELS[3]),
                      "trial_class_name"])
assert top == {"easy", "unclassified"}, sorted(top)

# --- each class occupies exactly two adjacent levels --------------------------
occupancy = pd.crosstab(primary["sp_design_early"].round(6), primary["trial_class_name"])
for stratum, counts in (
    ("easy", {2: 2179, 3: 1919}),
    ("ambiguous", {1: 641, 2: 2842}),
    ("misleading", {0: 12, 1: 1706}),
    ("unclassified", {2: 6175, 3: 850}),
):
    present = {i for i, level in enumerate(LEVELS)
               if occupancy.loc[round(level, 6), stratum] > 0}
    assert present == set(counts), (stratum, sorted(present), sorted(counts))
    for index, expected in counts.items():
        check(f"{stratum} at level {index}",
              occupancy.loc[round(LEVELS[index], 6), stratum], expected, 0)
# The share that makes `unclassified` load-bearing at the omnibus level.
shared_counts = occupancy.loc[round(LEVELS[2], 6)]
check("unclassified share of the omnibus level",
      100 * shared_counts["unclassified"] / shared_counts.sum(), 55.2, 0.05)

# --- level counts on the coefficients ---------------------------------------
signed = fits.loc[fits["predictor"] == "sum_log_lr_design_early"]
check("dt coefficient levels", signed["n_strength_levels"].max(), 2, 0)
check("accuracy coefficient levels", signed["n_signed_levels"].max(), 4, 0)
dt_row = stats.loc[(stats["term"] == "dt_slope_ms_per_unit")
                   & (stats["predictor"] == "sum_log_lr_design_early")
                   & (stats["condition"] == "all")].iloc[0]
check("registered dt slope", dt_row["mean"], -270.0, 0.05)

# the slope really is the two-point contrast
two_point = []
for _, group in primary.assign(
    ev=pd.to_numeric(primary["sum_log_lr_design_early"], errors="coerce"),
    dt=pd.to_numeric(primary["dt_ms"], errors="coerce"),
).dropna(subset=["ev", "dt"]).groupby("subject"):
    strength = group["ev"].abs()
    low, high = sorted(strength.round(9).unique())
    two_point.append((group.loc[strength.round(9) == high, "dt"].mean()
                      - group.loc[strength.round(9) == low, "dt"].mean()) / (high - low))
assert np.isclose(np.mean(two_point), dt_row["mean"]), \
    "the registered slope is not the two-point contrast"

# --- within-class evidence contrasts ----------------------------------------
within = stats.loc[stats["term"] == "evidence_contrast_within_class"]
for stratum, measure, delta, extra in (
    ("easy", "mean_dt_ms", -105.9, ("t", -7.57)),
    ("unclassified", "mean_dt_ms", -119.7, ("t", -7.13)),
    ("ambiguous", "mean_dt_ms", -3.8, ("equivalence_bound", 35.6)),
    ("easy", "accuracy", +0.011, ("p", 0.034)),
    ("unclassified", "accuracy", +0.110, ("t", 12.81)),
    ("ambiguous", "accuracy", +0.027, ("equivalence_bound", 0.094)),
):
    row = within.loc[(within["trial_class_name"] == stratum)
                     & (within["measure"] == measure)].iloc[0]
    check(f"within {stratum} {measure}", row["mean_difference"], delta,
          0.05 if measure == "mean_dt_ms" else 5e-4)
    column, expected = extra
    check(f"within {stratum} {measure} {column}", row[column], expected,
          rel(expected) if column == "p" else (5e-3 if column == "t" else 0.05))
check("ambiguous levels spanned low",
      within.loc[within["trial_class_name"] == "ambiguous",
                 "sp_design_early_reference"].iloc[0], LEVELS[1], 5e-6)
check("easy levels spanned low",
      within.loc[within["trial_class_name"] == "easy",
                 "sp_design_early_reference"].iloc[0], LEVELS[2], 5e-6)

# --- the shared-level table -------------------------------------------------
cells = pooled.loc[np.isclose(pooled["sp_design_early"], LEVELS[2])]
means = cells.pivot_table(index="trial_class_name", values=["mean_dt_ms", "accuracy"])
for stratum, dt, accuracy in (
    ("easy", 1083, 0.983), ("unclassified", 1257, 0.882), ("ambiguous", 1433, 0.776),
):
    check(f"{stratum} dt at shared level", means.loc[stratum, "mean_dt_ms"], dt, 0.5)
    check(f"{stratum} accuracy at shared level", means.loc[stratum, "accuracy"],
          accuracy, 5e-4)

omnibus = stats.loc[(stats["term"] == "class_omnibus_at_level")
                    & np.isclose(stats["sp_design_early"], LEVELS[2])].set_index("measure")
check("dt omnibus F", omnibus.loc["mean_dt_ms", "F"], 196.2, 0.05)
check("dt omnibus eta", omnibus.loc["mean_dt_ms", "partial_eta_squared"], 0.86, 5e-3)
check("dt omnibus range", omnibus.loc["mean_dt_ms", "estimate"], 350, 0.5)
check("dt omnibus df_error", omnibus.loc["mean_dt_ms", "df_error"], 62, 0)
check("accuracy omnibus F", omnibus.loc["accuracy", "F"], 181.3, 0.05)
check("accuracy omnibus eta", omnibus.loc["accuracy", "partial_eta_squared"], 0.85, 5e-3)
check("accuracy omnibus range", omnibus.loc["accuracy", "estimate"], 0.207, 5e-4)

contrasts = stats.loc[stats["term"] == "class_contrast_at_level"]


def contrast(level, pair, measure):
    rows = contrasts.loc[np.isclose(contrasts["sp_design_early"], level)
                         & (contrasts["trial_class_name"] == pair)
                         & (contrasts["measure"] == measure)]
    assert len(rows) == 1, (level, pair, measure, len(rows))
    return rows.iloc[0]


for pair, delta, t_stat in (
    ("ambiguous_vs_easy", 350.5, 15.84),
    ("ambiguous_vs_unclassified", 175.9, 13.93),
    ("easy_vs_unclassified", -174.6, -10.26),
):
    row = contrast(LEVELS[2], pair, "mean_dt_ms")
    check(f"{pair} dt", row["mean_difference"], delta, 0.05)
    check(f"{pair} dt t", row["t"], t_stat, 5e-3)

row = contrast(LEVELS[1], "ambiguous_vs_misleading", "mean_dt_ms")
check("ambig-misl dt", row["mean_difference"], 78.5, 0.05)
check("ambig-misl dt t", row["t"], 3.42, 5e-3)
check("ambig-misl dt p", row["p"], 0.0018, rel(0.0018))
row = contrast(LEVELS[1], "ambiguous_vs_misleading", "accuracy")
check("ambig-misl accuracy", row["mean_difference"], 0.372, 5e-4)
check("ambig-misl accuracy t", row["t"], 14.15, 5e-3)
check("ambig-misl accuracy dz", row["cohens_dz"], 2.50, 5e-3)
check("ambiguous accuracy at 0.39", row["mean_a"], 0.749, 5e-4)
check("misleading accuracy at 0.39", row["mean_b"], 0.377, 5e-4)

row = contrast(LEVELS[3], "easy_vs_unclassified", "accuracy")
check("easy-unclass accuracy at 0.81", row["mean_difference"], 0.002, 5e-4)
check("easy-unclass accuracy bound", row["equivalence_bound"], 0.008, 5e-4)
row = contrast(LEVELS[3], "easy_vs_unclassified", "mean_dt_ms")
check("easy-unclass dt at 0.81", row["mean_difference"], -160.8, 0.05)
check("easy-unclass dt t at 0.81", row["t"], -6.49, 5e-3)

# --- Fast/Slow bounds -------------------------------------------------------
fvs = stats.loc[(stats["condition"] == "fast_vs_slow")
                & (stats["predictor"] == "sum_log_lr_design_early")].set_index("term")
check("fast-slow dt slope", fvs.loc["dt_slope_ms_per_unit", "mean_difference"], 1.5, 0.05)
check("fast-slow dt bound", fvs.loc["dt_slope_ms_per_unit", "equivalence_bound"],
      27.5, 0.05)
check("fast-slow accuracy slope",
      fvs.loc["accuracy_log_odds_per_unit", "mean_difference"], -0.064, 5e-4)
check("fast-slow accuracy bound",
      fvs.loc["accuracy_log_odds_per_unit", "equivalence_bound"], 0.238, 5e-4)
check("accuracy coefficient", stats.loc[
    (stats["term"] == "accuracy_log_odds_per_unit")
    & (stats["predictor"] == "sum_log_lr_design_early")
    & (stats["condition"] == "all"), "mean"].iloc[0], 2.39, 5e-3)

if failures:
    print("MISMATCHES:")
    for line in failures:
        print(" ", line)
    raise SystemExit(1)
print("every quoted F11 value matches its derivative")
