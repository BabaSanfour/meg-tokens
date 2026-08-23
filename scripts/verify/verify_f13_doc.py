"""Check every number the F13 entry in docs/behavior.md quotes."""

from scripts.verify._root import group_beh
import pandas as pd


root = group_beh()
pw = pd.read_csv(root / "sub-group_task-tokens_desc-ssmheldoutpairwise_beh.tsv", sep="\t")
ho = pd.read_csv(root / "sub-group_task-tokens_desc-ssmheldout_beh.tsv", sep="\t")
a = pw[pw.condition == "all"]

# doc table: contrast -> (mean, ci_low, ci_high, t, p, dz, favouring)
expected = {
    ("urgency", "collapsing"): (0.070, 0.041, 0.100, 4.87, 3.1e-5, 0.86, 28),
    ("urgency", "additive_urgency"): (0.235, 0.179, 0.291, 8.54, 1.2e-9, 1.51, 30),
    ("collapsing", "additive_urgency"): (0.165, 0.128, 0.201, 9.24, 2.0e-10, 1.63, 31),
}
for (first, second), quoted in expected.items():
    fwd = a[(a.model_a == first) & (a.model_b == second)]
    rev = a[(a.model_a == second) & (a.model_b == first)]
    row, sign = (fwd.iloc[0], 1.0) if len(fwd) else (rev.iloc[0], -1.0)
    low, high = sorted([sign * row.ci95_low, sign * row.ci95_high])
    fav = row.n_subjects_favoring_model_a if sign > 0 else row.n_subjects_favoring_model_b
    got = (sign * row["mean"], low, high, sign * row.t, row.p, sign * row.cohens_dz, int(fav))
    for name, q, g in zip(("mean", "ci_low", "ci_high", "t", "p", "dz", "fav"), quoted, got):
        tol = abs(q) * 0.02 if name == "p" else 0.005
        assert abs(q - g) <= tol, f"{first}-{second} {name}: doc {q} vs derivative {g}"

w = ho[ho.condition == "all"].groupby(["subject", "model"])[
    "heldout_log_likelihood_per_trial"
].mean().unstack()
d = w["urgency"] - w["collapsing"]
assert round(float(d.min()), 3) == -0.166, d.min()
assert round(float(d.max()), 3) == 0.201, d.max()
assert round(float(w["urgency"].min()), 3) == -1.270, w["urgency"].min()
assert round(float(w["urgency"].max()), 3) == -0.186, w["urgency"].max()

def mean(x, y):
    fwd = a[(a.model_a == x) & (a.model_b == y)]
    if len(fwd):
        return float(fwd.iloc[0]["mean"])
    return -float(a[(a.model_a == y) & (a.model_b == x)].iloc[0]["mean"])

residual = abs(mean("urgency", "collapsing") + mean("collapsing", "ddm") - mean("urgency", "ddm"))
assert residual < 1e-12, residual

print("every quoted F13 value matches its derivative")
