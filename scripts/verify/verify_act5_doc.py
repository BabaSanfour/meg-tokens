"""Check every number Act 5 quotes in docs/behavior.md: F18 and the
cross-species translation table, which has no figure."""

from scripts.verify._root import group_beh
import math

import pandas as pd


root = group_beh()
failures: list[str] = []


def p_tolerance(doc: float) -> float:
    """Half the last digit of a two-significant-figure p-value."""
    return 0.05 * 10 ** math.floor(math.log10(doc))


def check(name: str, derivative: float, doc: float, tol: float) -> None:
    if abs(float(derivative) - doc) > tol:
        failures.append(f"{name}: derivative {derivative} vs doc {doc}")


profile = pd.read_csv(
    root / "sub-group_task-tokens_desc-individualprofile_beh.tsv", sep="\t"
)
correlations = pd.read_csv(
    root / "sub-group_task-tokens_desc-individualcorrelations_beh.tsv", sep="\t"
)

# --- the cohort spread table -------------------------------------------------
# measure -> (min, median, max, tolerance)
spread = {
    "mean_dt_ms": (582.0, 1199.0, 2024.0, 0.5),
    "percent_correct": (78.5, 83.5, 90.1, 0.05),
    "accuracy_log_odds_per_unit": (4.74, 9.21, 17.97, 0.005),
    "urgency_slope_per_second": (-0.050, 0.011, 0.133, 0.0005),
    "criterion_slope_sum_log_lr_per_second": (-0.550, -0.142, 0.183, 0.0005),
    "sat_adjustment_ms": (-137.0, 115.0, 454.0, 0.5),
    "lapse_rate": (0.0, 0.0, 0.0056, 5e-5),
}
for measure, (low, median, high, tol) in spread.items():
    check(f"{measure} min", profile[measure].min(), low, tol)
    check(f"{measure} median", profile[measure].median(), median, tol)
    check(f"{measure} max", profile[measure].max(), high, tol)

check("n subjects", len(profile), 32, 0)

# The sign counts behind "the two slope measures disagree on sign".
check(
    "urgency slope positive",
    (profile["urgency_slope_per_second"] > 0).sum(), 21, 0,
)
check(
    "criterion slope positive",
    (profile["criterion_slope_sum_log_lr_per_second"] > 0).sum(), 7, 0,
)
check(
    "SAT adjustment positive", (profile["sat_adjustment_ms"] > 0).sum(), 28, 0
)


# --- the correlation table ---------------------------------------------------
def pair(measure_a: str, measure_b: str):
    rows = correlations.loc[
        (
            (correlations["measure_a"] == measure_a)
            & (correlations["measure_b"] == measure_b)
        )
        | (
            (correlations["measure_a"] == measure_b)
            & (correlations["measure_b"] == measure_a)
        )
    ]
    assert len(rows) == 1, (measure_a, measure_b, len(rows))
    return rows.iloc[0]


check("pair count", len(correlations), 21, 0)
check("df", correlations["df"].max(), 30, 0)
check("correlation n subjects", correlations["n_subjects"].min(), 32, 0)

# measure_a, measure_b -> (r, p, rho)
quoted = {
    ("urgency_slope_per_second", "criterion_slope_sum_log_lr_per_second"): (
        0.887, 1.4e-11, 0.860
    ),
    ("mean_dt_ms", "accuracy_log_odds_per_unit"): (-0.814, 1.5e-8, -0.842),
    ("mean_dt_ms", "urgency_slope_per_second"): (-0.527, 0.0019, -0.503),
    ("mean_dt_ms", "percent_correct"): (0.460, 0.0080, 0.435),
    ("mean_dt_ms", "criterion_slope_sum_log_lr_per_second"): (
        -0.372, 0.036, -0.357
    ),
}
for (measure_a, measure_b), (r, p, rho) in quoted.items():
    row = pair(measure_a, measure_b)
    check(f"{measure_a} x {measure_b} r", row["pearson_r"], r, 5e-4)
    check(f"{measure_a} x {measure_b} p", row["pearson_p"], p, p_tolerance(p))
    check(f"{measure_a} x {measure_b} rho", row["spearman_rho"], rho, 5e-4)

# "Five of the 21 reach p < .05" is the count the whole entry rests on, and the
# figure selects its panels by it.
check("significant pairs", (correlations["pearson_p"] < 0.05).sum(), 5, 0)
# And the five quoted *are* those five, not five chosen to look good.
significant = {
    frozenset((row.measure_a, row.measure_b))
    for row in correlations[correlations["pearson_p"] < 0.05].itertuples()
}
if significant != {frozenset(key) for key in quoted}:
    failures.append("the five quoted pairs are not the five significant ones")

# --- the SAT-adjustment null -------------------------------------------------
sat = correlations[
    (correlations["measure_a"] == "sat_adjustment_ms")
    | (correlations["measure_b"] == "sat_adjustment_ms")
]
check("SAT adjustment pairs", len(sat), 6, 0)
check("SAT adjustment largest |r|", sat["pearson_r"].abs().max(), 0.282, 5e-4)
# "no p falls below .12" across the six pairs it enters. Quoted to two
# decimals in the doc, so checked there.
check("SAT adjustment smallest p", sat["pearson_p"].min(), 0.12, 5e-3)
if not (sat["pearson_p"] >= 0.115).all():
    failures.append("a SAT-adjustment pair fell below the quoted p floor")
check(
    "SAT adjustment vs mean DT r",
    pair("sat_adjustment_ms", "mean_dt_ms")["pearson_r"], -0.200, 5e-4,
)
check(
    "SAT adjustment vs mean DT p",
    pair("sat_adjustment_ms", "mean_dt_ms")["pearson_p"], 0.273, 5e-4,
)
check(
    "SAT adjustment vs lapse rate r",
    pair("sat_adjustment_ms", "lapse_rate")["pearson_r"], 0.282, 5e-4,
)

# --- the strongest non-significant pairs -------------------------------------
check(
    "accuracy x criterion r",
    pair("percent_correct", "criterion_slope_sum_log_lr_per_second")["pearson_r"],
    -0.291, 5e-4,
)
check(
    "accuracy x criterion p",
    pair("percent_correct", "criterion_slope_sum_log_lr_per_second")["pearson_p"],
    0.106, 5e-4,
)
check(
    "accuracy x urgency r",
    pair("percent_correct", "urgency_slope_per_second")["pearson_r"], -0.282, 5e-4,
)
check(
    "accuracy x urgency p",
    pair("percent_correct", "urgency_slope_per_second")["pearson_p"], 0.117, 5e-4,
)
# The doc names these two as the strongest that miss; nothing non-significant
# may outrank them.
not_significant = correlations[correlations["pearson_p"] >= 0.05]
check(
    "largest non-significant |r|",
    not_significant["pearson_r"].abs().max(), 0.291, 5e-4,
)

# --- Pearson and Spearman agree ----------------------------------------------
# The doc claims rank agreement reproduces every significant Pearson and adds
# none. That is a statement about the derivative, so it is tested, not asserted.
pearson_significant = {
    frozenset((row.measure_a, row.measure_b))
    for row in correlations[correlations["pearson_p"] < 0.05].itertuples()
}
spearman_significant = {
    frozenset((row.measure_a, row.measure_b))
    for row in correlations[correlations["spearman_p"] < 0.05].itertuples()
}
if pearson_significant != spearman_significant:
    failures.append(
        "Pearson and Spearman disagree on which pairs are significant: "
        f"{pearson_significant ^ spearman_significant}"
    )
row = pair("urgency_slope_per_second", "accuracy_log_odds_per_unit")
check("urgency x evidence r", row["pearson_r"], 0.262, 5e-4)
check("urgency x evidence p", row["pearson_p"], 0.148, 5e-4)
check("urgency x evidence rho", row["spearman_rho"], 0.324, 5e-4)
check("urgency x evidence rho p", row["spearman_p"], 0.070, 5e-4)

if failures:
    print("MISMATCHES:")
    for line in failures:
        print(" ", line)
    raise SystemExit(1)
# --- the cross-species cross-reference (no figure, no quoted values) ---------
# The doc quotes no number from this table any more -- it is a mapping from
# measure to the act that reports it and the published analysis it lines up
# against. So the only thing to verify is that the ten measures it enumerates
# are still the ten the derivative holds: a row added or renamed upstream would
# silently leave the mapping incomplete, and nothing else would notice.
species = pd.read_csv(
    root / "sub-group_task-tokens_desc-speciescomparison_beh.tsv", sep="\t"
)
mapped = {
    "decision_time_easy_ms",
    "decision_time_ambiguous_ms",
    "decision_time_misleading_ms",
    "success_probability_at_decision_easy",
    "success_probability_at_decision_ambiguous",
    "success_probability_at_decision_misleading",
    "criterion_slope_sum_log_lr_per_second",
    "urgency_minus_integrator_bic",
    "urgency_scale_criterion_seconds",
    "urgency_scale_fast_minus_slow",
}
check("species rows", len(species), 10, 0)
check("species n subjects", species["n_subjects"].min(), 32, 0)
if set(species["measure"]) != mapped:
    failures.append(
        "speciescomparison rows no longer match the doc's mapping: "
        f"{set(species['measure']) ^ mapped}"
    )

if failures:
    print("MISMATCHES:")
    for line in failures:
        print(" ", line)
    raise SystemExit(1)
# --- the cross-species translation table (no figure) -------------------------
species = pd.read_csv(
    root / "sub-group_task-tokens_desc-speciescomparison_beh.tsv", sep="\t"
).set_index("measure")

check("species rows", len(species), 10, 0)
check("species df", species["df"].max(), 31, 0)
check("species n subjects", species["n_subjects"].min(), 32, 0)

# measure -> (value, ci_low, ci_high, t, p)
translation = {
    "decision_time_easy_ms": (1033, 913, 1153, 17.55, 1.2e-17),
    "decision_time_ambiguous_ms": (1434, 1290, 1577, 20.43, 1.5e-19),
    "decision_time_misleading_ms": (1358, 1199, 1517, 17.39, 1.5e-17),
    "success_probability_at_decision_easy": (0.791, 0.769, 0.813, 74.44, 1.6e-36),
    "success_probability_at_decision_ambiguous": (
        0.664, 0.637, 0.690, 51.50, 1.3e-31
    ),
    "success_probability_at_decision_misleading": (
        0.649, 0.636, 0.663, 99.70, 1.9e-40
    ),
    "criterion_slope_sum_log_lr_per_second": (
        -0.133, -0.191, -0.075, -4.66, 5.8e-5
    ),
    "urgency_minus_integrator_bic": (-239.6, -300, -180, -8.14, 3.4e-9),
    "urgency_scale_criterion_seconds": (1.528, 1.381, 1.675, 21.22, 5.1e-20),
    "urgency_scale_fast_minus_slow": (-0.114, -0.203, -0.025, -2.60, 0.014),
}
for measure, (value, low, high, t_stat, p_value) in translation.items():
    row = species.loc[measure]
    scale = 0.5 if abs(value) > 10 else 5e-4
    check(f"{measure} mean", row["mean"], value, scale)
    check(f"{measure} ci low", row["ci95_low"], low, max(scale, 0.5))
    check(f"{measure} ci high", row["ci95_high"], high, max(scale, 0.5))
    check(f"{measure} t", row["t"], t_stat, 0.005)
    check(f"{measure} p", row["p"], p_value, p_tolerance(p_value))

# The doc says these ten are the whole table; a new row added upstream without
# a doc entry is exactly what this catches.
if set(species.index) != set(translation):
    failures.append(
        f"speciescomparison rows changed: {set(species.index) ^ set(translation)}"
    )

if failures:
    print("MISMATCHES:")
    for line in failures:
        print(" ", line)
    raise SystemExit(1)
print("every quoted Act 5 value matches its derivative")
