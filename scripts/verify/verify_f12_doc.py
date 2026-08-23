"""Assert every number the F12 section quotes against its derivative."""

from scripts.verify._root import group_beh

import pandas as pd


R = group_beh()
kernels = pd.read_csv(R / "sub-group_task-tokens_desc-reversecorrelation_beh.tsv", sep="\t")
stats = pd.read_csv(R / "sub-group_task-tokens_desc-reversecorrelationstats_beh.tsv", sep="\t")
weights = stats.loc[stats["metric"] == "logistic_weight"]
failures = []

pooled = weights.loc[
    (weights["condition"] == "all") & (weights["test"] == "one_sample_vs_zero")
].set_index("jump")["mean"]
for jump, expected in zip(range(1, 9), [2.69, 2.39, 2.40, 2.07, 1.75, 1.75, 1.65, 1.26]):
    if round(float(pooled[jump]), 2) != expected:
        failures.append(f"jump {jump} weight: {pooled[jump]:.4f} vs {expected}")

summary = weights.loc[
    (weights["test"] == "all_jumps_vs_zero") & (weights["condition"] == "all")
].iloc[0]
if (summary["n_jumps_significant"], summary["n_jumps_tested"]) != (8, 8):
    failures.append("summary is not 8/8")

contrast = weights.loc[weights["condition"] == "fast_vs_slow"].sort_values("jump")
if round(float(contrast["p_threshold"].iloc[0]), 5) != 0.00625:
    failures.append("threshold is not .00625")
if contrast.loc[contrast["p"] < contrast["p_threshold"], "jump"].tolist() != [2.0, 3.0, 4.0]:
    failures.append("Bonferroni-surviving jumps are not 2-4")
if contrast.loc[contrast["p"] < 0.05, "jump"].tolist() != [1.0, 2.0, 3.0, 4.0, 5.0]:
    failures.append("uncorrected jumps are not 1-5")
for jump, delta, bound in ((6, 0.29, 0.74), (7, 0.24, 0.72), (8, 0.005, 0.48)):
    row = contrast.loc[contrast["jump"] == jump].iloc[0]
    if round(float(row["mean_difference"]), 3 if jump == 8 else 2) != delta:
        failures.append(f"jump {jump} difference: {row['mean_difference']:.4f} vs {delta}")
    if round(float(row["equivalence_bound"]), 2) != bound:
        failures.append(f"jump {jump} bound: {row['equivalence_bound']:.4f} vs {bound}")

converged = kernels.groupby("condition")["converged"].sum()
if int(converged["all"]) != 8 * 32:
    failures.append("pooled fits did not all converge")
for condition in ("fast", "slow"):
    if int(converged[condition]) != 8 * 28:
        failures.append(f"{condition}: expected 28 of 32 subjects to converge")

if failures:
    print("MISMATCHES:")
    for line in failures:
        print(" ", line)
    raise SystemExit(1)
print("every quoted F12 value matches its derivative")
