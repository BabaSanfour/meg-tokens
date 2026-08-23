"""Assert every number the F13 and F14 write-ups quote against the ssm derivatives."""

from scripts.verify._root import group_beh

import pandas as pd


R = group_beh()


def table(name):
    return pd.read_csv(R / f"sub-group_task-tokens_desc-{name}_beh.tsv", sep="\t")


failures = []


def check(label, actual, expected, tol):
    if pd.isna(actual) or abs(float(actual) - expected) > tol:
        failures.append(f"{label}: derivative {actual} vs doc {expected}")


def rel(p):
    return abs(p) * 0.02 + (5e-4 if p > 0.01 else 0.0)


# --- in-sample delta BIC ----------------------------------------------------
c = table("ssmcomparisonstats")
insample = c.loc[(c["condition"] == "all") & (c["criterion"] == "bic")
                 & c["model"].notna()].set_index("model")
for model, delta, t_stat, p, favouring in (
    ("urgency", -239.6, -8.14, 3.4e-9, 30),
    ("collapsing", -169.0, -10.34, 1.4e-11, 31),
    ("additive_urgency", -19.0, -1.34, 0.19, 16),
):
    row = insample.loc[model]
    check(f"{model} deltaBIC", row["mean"], delta, 0.05)
    check(f"{model} t", row["t"], t_stat, 5e-3)
    check(f"{model} p", row["p"], p, rel(p))
    check(f"{model} favouring", row["n_subjects_favoring_candidate"], favouring, 0)

# --- urgency_scale condition contrast ---------------------------------------
row = c.loc[c["parameter"] == "urgency_scale"].iloc[0]
check("urgency_scale delta", row["mean_difference"], -0.114, 5e-4)

# --- the urgency-parameter entry (F14) --------------------------------------
# Every number the F14 entry quotes, including the two equivalence bounds it
# argues its nulls from. A bound is a claim; if it drifts, the entry is wrong.
urgency = c.loc[c["analysis"] == "ssm_urgency_condition_contrast"].set_index("parameter")
for parameter, delta, p_value, bound in (
    ("drift_scale", +0.0264, 8.23e-5, 0.0364),
    ("urgency_scale", -0.1139, 0.0142, 0.1882),
    ("urgency_onset_s", -0.0141, 0.8115, 0.1139),
    ("nondecision_s", -0.00104, 0.1590, 0.00226),
):
    urow = urgency.loc[parameter]
    check(f"{parameter} delta", urow["mean_difference"], delta, 5e-4)
    check(f"{parameter} p", urow["p"], p_value, rel(p_value))
    check(f"{parameter} equivalence bound", urow["equivalence_bound"], bound, 5e-4)
# The paired interval must be registered, not rebuilt by the figure.
check("urgency_scale ci low", urgency.loc["urgency_scale", "ci95_low"], -0.2033, 5e-4)
check("urgency_scale ci high", urgency.loc["urgency_scale", "ci95_high"], -0.0245, 5e-4)

# The censoring is asymmetric in the direction of the effect: that asymmetry is
# the objection the refit panel exists to answer, so it is pinned too.
ba = table("ssmboundaryaudit")
ba = ba.loc[ba["model"] == "urgency"].set_index("condition")
check("urgency_scale pinned fast", ba.loc["fast", "urgency_scale_near_upper"], 3, 0)
check("urgency_scale pinned slow", ba.loc["slow", "urgency_scale_near_upper"], 8, 0)

# And the defence itself: the refit with no censored cell gives the largest
# contrast. If that ever reverses, the entry's argument inverts.
rp = table("ssmrobustnessstats")
rp = rp.loc[
    (rp["analysis"] == "ssm_robustness_parameter_contrast")
    & (rp["parameter"] == "urgency_scale")
].set_index("configuration")
check("expanded_bounds delta", rp.loc["expanded_bounds", "mean_difference"], -0.1809, 5e-4)
check("expanded_bounds censored", rp.loc["expanded_bounds", "n_cells_near_upper_bound"], 0, 0)
check("robustness baseline delta", rp.loc["baseline", "mean_difference"], -0.1181, 5e-4)
assert (
    abs(rp.loc["expanded_bounds", "mean_difference"]) == rp["mean_difference"].abs().max()
), "the uncensored refit is no longer the largest contrast"
check("urgency_scale t", row["t"], -2.60, 5e-3)
check("urgency_scale p", row["p"], 0.014, 5e-4)

# --- held-out ---------------------------------------------------------------
h = table("ssmheldoutstats")
heldout = h.loc[h["condition"] == "all"].set_index("model")
for model, delta, lo, hi, p, favouring in (
    ("urgency", +0.244, +0.183, +0.306, 4.0e-9, 30),
    ("collapsing", +0.174, +0.139, +0.209, 1.9e-11, 31),
    ("additive_urgency", +0.009, -0.020, +0.038, 0.52, 13),
):
    row = heldout.loc[model]
    check(f"heldout {model}", row["mean"], delta, 5e-4)
    check(f"heldout {model} lo", row["ci95_low"], lo, 5e-4)
    check(f"heldout {model} hi", row["ci95_high"], hi, 5e-4)
    check(f"heldout {model} p", row["p"], p, rel(p))
    check(f"heldout {model} favouring", row["n_subjects_favoring_candidate"], favouring, 0)

# --- urgency vs collapsing, direct ------------------------------------------
hp = table("ssmheldoutpairwise")
row = hp.loc[(hp["condition"] == "all") & (hp["model_a"] == "collapsing")
             & (hp["model_b"] == "urgency")].iloc[0]
# The doc states it urgency-minus-collapsing; the derivative stores a-minus-b.
check("urgency vs collapsing", -row["mean"], +0.070, 5e-4)
check("urgency vs collapsing lo", -row["ci95_high"], +0.041, 5e-4)
check("urgency vs collapsing hi", -row["ci95_low"], +0.100, 5e-4)
check("urgency vs collapsing t", -row["t"], 4.87, 5e-3)
check("urgency vs collapsing p", row["p"], 3.1e-5, rel(3.1e-5))
check("urgency vs collapsing favouring",
      row["n_subjects"] - row["n_subjects_favoring_model_a"], 28, 0)

# --- model recovery ---------------------------------------------------------
# Rewritten 2026-08-22 for the re-run study (jobs 56148178 + 56176860), whose
# truths come from the accepted fits rather than a hypercube over the optimizer
# bounds. The previous assertions encoded the superseded claim that the urgency
# parameters were unidentified.
m = table("ssmmodelrecoverystats")
diagonal = m.loc[m["true_model"] == m["selected_model"]]
assert len(diagonal) == 4, len(diagonal)
check("recovery repetitions", diagonal["n_repetitions"].min(), 12, 0)
# 47 of 48: one true-ddm case selects collapsing, so the diagonal is no longer
# uniformly 1.0 and the doc says so.
check("recovery correct cases", diagonal["n_selected"].sum(), 47, 0)
check("recovery ddm diagonal",
      diagonal.set_index("true_model")["selection_rate"]["ddm"], 0.9167, 5e-4)

# --- parameter recovery -----------------------------------------------------
p_rec = table("ssmparameterrecoverystats")
weak = p_rec.set_index(["true_model", "parameter"])["correlation"]
check("urgency_scale r", weak[("urgency", "urgency_scale")], 0.994, 5e-4)
check("urgency_onset_s r", weak[("urgency", "urgency_onset_s")], 0.986, 5e-4)
check("additive_scale r", weak[("additive_urgency", "additive_scale")], 0.904, 5e-4)
check("urgency drift_scale r", weak[("urgency", "drift_scale")], 0.983, 5e-4)
check("collapse_rate r", weak[("collapsing", "collapse_rate")], 0.939, 5e-4)
# The three non-decision failures are a claim in their own right, so they are
# asserted rather than merely tolerated.
for model, r in (("urgency", -0.289), ("ddm", 0.235), ("collapsing", -0.435)):
    check(f"{model} nondecision r", weak[(model, "nondecision_s")], r, 5e-4)
check("additive nondecision r", weak[("additive_urgency", "nondecision_s")], 1.0, 5e-4)
# Everything except those three recovers well; the doc states no exceptions.
for (model, parameter), r in weak.items():
    if parameter == "nondecision_s" and model in {"urgency", "ddm", "collapsing"}:
        continue
    if r < 0.90:
        failures.append(f"a 'recovers well' parameter is below 0.90: {model}.{parameter} r={r:.3f}")
# The doc quotes this range for boundary_rate; truths drawn from real fits
# inherit fits that sit at a bound.
rates = p_rec["boundary_rate"]
if not (0.33 <= rates.min() and rates.max() <= 0.67):
    failures.append(f"boundary_rate outside the quoted 0.33-0.67: {rates.min()}-{rates.max()}")

# --- robustness -------------------------------------------------------------
rb = table("ssmrobustnessstats")
urgency = rb.loc[(rb["condition"] == "all") & (rb["model"] == "urgency")
                 & (rb["criterion"] == "delta_bic")]
check("robustness configurations", urgency["configuration"].nunique(), 6, 0)
check("robustness favouring (min)", urgency["n_subjects_favoring_candidate"].min(), 30, 0)
check("robustness favouring (max)", urgency["n_subjects_favoring_candidate"].max(), 30, 0)
check("robustness deltaBIC worst", urgency["mean"].max(), -231, 0.6)
check("robustness deltaBIC best", urgency["mean"].min(), -247, 0.1)
shifts = rb.loc[(rb["condition"] == "all") & (rb["model"] == "urgency")
                & (rb["criterion"] == "delta_bic_change_vs_baseline"), "mean"]
check("robustness shift low", shifts.min(), -7.8, 0.05)
check("robustness shift high", shifts.max(), +8.7, 0.05)

# --- strict-exclusion -------------------------------------------------------
er = table("ssmexclusionrobustnessstats")
strict = er.loc[(er["condition"] == "all")
                & (er["criterion"] == "strict_delta_bic_minus_primary_delta_bic")
                ].set_index("model")
for model, shift, t_stat, p, favouring in (
    ("urgency", 61.5, 7.18, 4.6e-8, 30),
    ("collapsing", 39.7, 8.08, 4.0e-9, 31),
):
    row = strict.loc[model]
    check(f"strict {model} shift", row["mean"], shift, 0.05)
    check(f"strict {model} t", row["t"], t_stat, 5e-3)
    check(f"strict {model} p", row["p"], p, rel(p))
    check(f"strict {model} favouring", row["n_subjects_favoring_primary"], favouring, 0)
boundary = er.loc[er["criterion"] == "strict_delta_bic_minus_primary_delta_bic",
                  "boundary_rate"]
check("strict boundary rate low", boundary.min(), 0.91, 0.005)
check("strict boundary rate high", boundary.max(), 1.00, 1e-6)
check("strict trial count", er["n_mask_intersection"].max(), 10961, 0)
check("strict is a pure subset", er["n_mask_symmetric_difference"].max(), 0, 0)
assert bool(er["mask_equivalent"].all()), "mask_equivalent is not true everywhere"

# --- matched sequence -------------------------------------------------------
ms = table("ssmmatchedsequencestats").set_index(["condition", "measure"])
for key, p, pairs in (
    (("Fast", "mean_dt_ms"), 0.49, 310),
    (("Fast", "mean_accuracy"), 0.67, 310),
    (("Slow", "mean_dt_ms"), 0.33, 302),
    (("Slow", "mean_accuracy"), 0.14, 302),
):
    row = ms.loc[key]
    check(f"matched {key} p", row["p"], p, 5e-3)
    check(f"matched {key} pairs", row["n_pairs"], pairs, 0)

# --- strict-exclusion decomposition (the corrected caveat) ------------------
ed = table("ssmexclusiondecomposition")
dec = ed.loc[
    (ed["analysis"] == "ssm_exclusion_evidence_decomposition")
    & (ed["condition"] == "all")
].set_index("model")
for model, primary, strict, expected, per_primary, per_strict in (
    ("urgency", -239.60, -178.05, -158.59, 0.2451, 0.2730),
    ("collapsing", -169.04, -129.33, -111.63, 0.1733, 0.1987),
    ("additive_urgency", -19.00, -6.43, -10.64, 0.0258, 0.0180),
):
    row = dec.loc[model]
    check(f"{model} primary delta_bic", row["mean_primary_delta_bic"], primary, 5e-3)
    check(f"{model} strict delta_bic", row["mean_strict_delta_bic"], strict, 5e-3)
    check(f"{model} expected from n", row["mean_expected_strict_delta_bic"], expected, 5e-3)
    check(f"{model} per trial primary", row["mean_primary_per_trial"], per_primary, 5e-4)
    check(f"{model} per trial strict", row["mean_strict_per_trial"], per_strict, 5e-4)
# The correction that reversed the caveat: evidence per trial rises.
check("urgency per-trial rise t", dec.loc["urgency", "per_trial_change_t"], 4.90, 5e-2)
check("urgency excess t", dec.loc["urgency", "t"], -4.96, 5e-2)
assert dec.loc["urgency", "per_trial_change_n_subjects_rising"] == 26, (
    "urgency per-trial rise is no longer 26/32"
)

comp = ed.loc[
    ed["analysis"] == "ssm_exclusion_condition_composition"
].set_index("condition")
for condition, primary, strict, retained, share_primary, share_strict in (
    ("fast", 8980, 8613, 0.959, 0.550, 0.786),
    ("slow", 7340, 2345, 0.319, 0.450, 0.214),
):
    row = comp.loc[condition]
    check(f"{condition} primary trials", row["n_trials_primary"], primary, 0)
    check(f"{condition} strict trials", row["n_trials_strict"], strict, 0)
    check(f"{condition} retention", row["retention_fraction"], retained, 5e-4)
    check(f"{condition} share of primary", row["share_of_primary"], share_primary, 5e-4)
    check(f"{condition} share of strict", row["share_of_strict"], share_strict, 5e-4)
check("strict total trials", comp.loc["all", "n_trials_strict"], 10958, 0)
check("primary total trials", comp.loc["all", "n_trials_primary"], 16320, 0)

# --- held-out contrast robustness (S1-S3) -----------------------------------
hr = table("ssmheldoutrobustness")
subsets = hr.loc[
    hr["analysis"] == "ssm_heldout_identifiability_subset"
].set_index("subset")
for subset, n, mean, t, p in (
    ("all_subjects", 32, 0.0703, 4.87, 3.1e-5),
    ("urgency_scale_not_pinned", 24, 0.0671, 4.11, 4.3e-4),
    ("finite_observed_information_se", 17, 0.0654, 3.52, 2.9e-3),
):
    row = subsets.loc[subset]
    check(f"{subset} n", row["n_subjects"], n, 0)
    check(f"{subset} mean", row["mean"], mean, 5e-4)
    check(f"{subset} t", row["t"], t, 5e-3)
    check(f"{subset} p", row["p"], p, max(p * 0.05, 1e-9))

influence = hr.loc[hr["analysis"] == "ssm_heldout_influence"].set_index("criterion")
check("worst leave-one-out t", influence.loc["leave_one_subject_out", "least_favourable_t"], 4.61, 5e-2)
check("wilcoxon p", influence.loc["distribution_free", "wilcoxon_p"], 2.27e-5, 1e-6)
check("sign test p", influence.loc["distribution_free", "sign_test_p"], 1.93e-5, 1e-6)

pooling = hr.loc[
    hr["analysis"] == "ssm_heldout_condition_pooling"
].set_index("model")
check("urgency pooled", pooling.loc["urgency", "pooled_delta_bic"], -239.60, 5e-3)
check("urgency within sum", pooling.loc["urgency", "summed_within_condition_delta_bic"], -249.59, 5e-3)
check("collapsing pooled", pooling.loc["collapsing", "pooled_delta_bic"], -169.04, 5e-3)
check("collapsing within sum", pooling.loc["collapsing", "summed_within_condition_delta_bic"], -172.16, 5e-3)

# --- absolute fit (S4) ------------------------------------------------------
dc = table("ssmdistributioncheckstats")
dc = dc.loc[(dc["condition"] == "all") & (dc["trial_class"] == "all")]
residual = dc.pivot_table(
    index="model", columns=["outcome", "quantile"], values="mean_residual_s"
)
for model, values in (
    ("urgency", [-0.023, -0.001, 0.054, 0.090, 0.048, 0.009]),
    ("collapsing", [0.009, 0.042, -0.055, 0.169, 0.178, -0.085]),
    ("ddm", [0.109, 0.035, -0.375, 0.373, 0.337, -0.157]),
    ("additive_urgency", [0.150, 0.014, -0.111, 0.499, 0.292, 0.007]),
):
    keys = [("correct", q) for q in (0.1, 0.5, 0.9)] + [("error", q) for q in (0.1, 0.5, 0.9)]
    for key, expected in zip(keys, values):
        check(f"{model} residual {key}", residual.loc[model, key], expected, 1e-3)

# --- matched-sequence posterior predictive (C4) -----------------------------
ppc = table("ssmmatchedsequenceppc")
ppc = ppc.loc[ppc["measure"] == "mean_accuracy"].set_index(["condition", "model"])
for key, predicted, residual_mean, p in (
    (("Fast", "urgency"), 0.031, -0.050, 0.238),
    (("Slow", "urgency"), 0.023, -0.114, 0.070),
    (("Fast", "collapsing"), -0.031, 0.011, 0.795),
    (("Slow", "ddm"), -0.101, 0.010, 0.867),
):
    row = ppc.loc[key]
    check(f"ppc {key} predicted", row["mean_predicted"], predicted, 1e-3)
    check(f"ppc {key} residual", row["mean_observed_minus_predicted"], residual_mean, 1e-3)
    check(f"ppc {key} p", row["p"], p, 5e-3)

# --- censoring attenuates the condition contrast rather than creating it ----
rs = table("ssmrobustnessstats")
contrast = rs.loc[
    (rs["analysis"] == "ssm_robustness_parameter_contrast")
    & (rs["parameter"] == "urgency_scale")
].set_index("configuration")
for configuration, difference, t_value, p_value in (
    ("baseline", -0.118, -2.74, 0.010),
    ("expanded_bounds", -0.181, -2.56, 0.016),
):
    row = contrast.loc[configuration]
    check(f"{configuration} urgency_scale contrast", row["mean_difference"], difference, 1e-3)
    check(f"{configuration} urgency_scale t", row["t"], t_value, 5e-3)
    check(f"{configuration} urgency_scale p", row["p"], p_value, 5e-4)
assert (
    abs(contrast.loc["expanded_bounds", "mean_difference"])
    > abs(contrast.loc["baseline", "mean_difference"])
), "relaxing the bound no longer enlarges the contrast"

# --- recovery correlations are intervals, not point estimates (C5) ----------
pr = table("ssmparameterrecoverystats").set_index(["true_model", "parameter"])
for key, r, low, high in (
    (("urgency", "urgency_scale"), 0.994, 0.976, 0.998),
    (("urgency", "urgency_onset_s"), 0.986, 0.949, 0.996),
    (("additive_urgency", "additive_scale"), 0.904, 0.687, 0.973),
    (("collapsing", "collapse_rate"), 0.939, 0.793, 0.983),
):
    row = pr.loc[key]
    check(f"recovery {key} r", row["correlation"], r, 1e-3)
    check(f"recovery {key} ci low", row["correlation_ci95_low"], low, 1e-3)
    check(f"recovery {key} ci high", row["correlation_ci95_high"], high, 1e-3)

if failures:
    print("MISMATCHES:")
    for line in failures:
        print(" ", line)
    raise SystemExit(1)
print("every quoted F13 and F14 value matches its derivative")
