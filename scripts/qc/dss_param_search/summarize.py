"""Choose one configuration per stage, and say how well the choice holds up.

The rule is a constraint, not a weighted score. Maximise the reference
correlation -- the only metric that says the removed subspace *is* the artifact
-- subject to a collateral budget in the alpha band and a ceiling on how many
of 270 dimensions come out. Removal percentage is reported and never ranked
on: a larger subspace always removes more, signal included.

Configurations are compared by **how many subjects they win on**, never by the
median of their scores. Between-subject variance here dwarfs between-config
differences -- one arena metric ranged -0.5% to 79.8% across three subjects --
and a median over a bimodal cohort describes no subject in it.

The winner is chosen on the search fold and reported on the confirmation fold,
because picking one row of fifty and quoting its margin over the same subjects
that chose it is selection on the test set.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from scripts.qc.dss_param_search import grid as grid_module

#: Objective per stage, and whether a subject has to carry the artifact for its
#: row to count. Cardiac and ocular are present in every recording; muscle is
#: not, and a stage that correctly removes nothing on a clean recording must
#: not be scored as having missed something.
OBJECTIVE = {
    "qrs": "reference_r",
    "blink": "reference_r",
    "muscle": "muscle_reduction_db",
}


def feasible(row: dict, *, alpha_budget: float, dimension_budget: int) -> bool:
    """Whether a configuration's collateral is inside budget."""
    if "failed" in row:
        return False
    if row.get("n_selected", 0) > dimension_budget:
        return False
    return row.get("alpha_delta_db", float("-inf")) >= -alpha_budget


def load(input_root: Path) -> list[dict]:
    """Every per-subject result, in subject order."""
    records = [
        json.loads(path.read_text())
        for path in sorted(Path(input_root).glob("sub-*_dsssearch.json"))
    ]
    if not records:
        raise SystemExit(f"no sub-*_dsssearch.json under {input_root}")
    return records


def stage_table(
    records: list[dict],
    stage: str,
    *,
    fold: str,
    alpha_budget: float,
    dimension_budget: int,
    muscle_floor: float,
) -> list[dict]:
    """One row per configuration: wins, feasibility, and the objective's spread."""
    objective = OBJECTIVE[stage]
    eligible = [
        record for record in records
        if record["fold"] == fold
        and (stage != "muscle" or record["presence"]["muscle_annotated_fraction"] >= muscle_floor)
    ]
    wins: dict[str, int] = {}
    for record in eligible:
        rows = [
            row for row in record["rows"]
            if row["stage"] == stage
            and feasible(row, alpha_budget=alpha_budget, dimension_budget=dimension_budget)
            and row.get(objective) == row.get(objective)  # not NaN
        ]
        if rows:
            best = max(rows, key=lambda row: row[objective])
            wins[best["label"]] = wins.get(best["label"], 0) + 1

    # Rows come from the labels the run actually produced, not from the current
    # grid. Deriving them from the grid means editing `grid.py` and then
    # re-summarising an older run silently drops every configuration whose label
    # moved -- the table would simply be shorter, with nothing to notice.
    production = grid_module.production_config(stage).label
    labels = sorted({
        row["label"] for record in records for row in record["rows"]
        if row["stage"] == stage
    })
    table = []
    for label in labels:
        rows = [
            row for record in eligible for row in record["rows"]
            if row["stage"] == stage and row["label"] == label
        ]
        scored = [row[objective] for row in rows if row.get(objective) == row.get(objective)]
        passed = [
            row for row in rows
            if feasible(row, alpha_budget=alpha_budget, dimension_budget=dimension_budget)
        ]
        table.append({
            "stage": stage,
            "label": label,
            "is_production": label == production,
            "n_subjects": len(rows),
            "n_feasible": len(passed),
            "n_failed": sum("failed" in row for row in rows),
            "wins": wins.get(label, 0),
            "objective_median": statistics.median(scored) if scored else float("nan"),
            "objective_min": min(scored) if scored else float("nan"),
            "objective_max": max(scored) if scored else float("nan"),
            "alpha_delta_median": statistics.median(
                [row["alpha_delta_db"] for row in rows if "alpha_delta_db" in row]
            ) if rows else float("nan"),
            "n_selected_median": statistics.median(
                [row["n_selected"] for row in rows if "n_selected" in row]
            ) if rows else float("nan"),
        })
    return sorted(table, key=lambda row: (-row["wins"], -row["objective_median"]))


def confirm(
    records: list[dict], stage: str, winner: str, *,
    alpha_budget: float, dimension_budget: int, muscle_floor: float,
) -> dict:
    """Head-to-head against production on the held-out fold, per subject."""
    objective = OBJECTIVE[stage]
    production = grid_module.production_config(stage).label
    beats, total, deltas = 0, 0, []
    for record in records:
        if record["fold"] != "confirm":
            continue
        if stage == "muscle" and record["presence"]["muscle_annotated_fraction"] < muscle_floor:
            continue
        rows = {row["label"]: row for row in record["rows"] if row["stage"] == stage}
        if winner not in rows or production not in rows:
            continue
        candidate, incumbent = rows[winner], rows[production]
        if not feasible(candidate, alpha_budget=alpha_budget, dimension_budget=dimension_budget):
            continue
        if candidate.get(objective) != candidate.get(objective):
            continue
        if incumbent.get(objective) != incumbent.get(objective):
            continue
        total += 1
        delta = candidate[objective] - incumbent[objective]
        deltas.append(delta)
        beats += delta > 0
    return {
        "stage": stage, "winner": winner, "production": production,
        "beats_production_on": beats, "of": total,
        "delta_median": statistics.median(deltas) if deltas else float("nan"),
        "delta_min": min(deltas) if deltas else float("nan"),
        "delta_max": max(deltas) if deltas else float("nan"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--alpha-budget", default=0.5, type=float,
                        help="dB of alpha-band loss a configuration may cost")
    parser.add_argument("--dimension-budget", default=20, type=int,
                        help="components of 270 a stage may remove")
    parser.add_argument("--muscle-floor", default=0.005, type=float,
                        help="annotated muscle fraction below which a recording "
                             "carries no muscle and is excluded from that stage")
    args = parser.parse_args(argv)

    records = load(args.input_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    budget = {
        "alpha_budget": args.alpha_budget,
        "dimension_budget": args.dimension_budget,
        "muscle_floor": args.muscle_floor,
    }

    folds = {record["subject"]: record["fold"] for record in records}
    print(f"{len(records)} recordings: "
          f"{sum(f == 'search' for f in folds.values())} search, "
          f"{sum(f == 'confirm' for f in folds.values())} confirm")
    carry_muscle = sum(
        record["presence"]["muscle_annotated_fraction"] >= args.muscle_floor
        for record in records
    )
    print(f"muscle present (>={args.muscle_floor:.1%} annotated) in {carry_muscle}/{len(records)}")

    lines, confirmations = [], []
    for stage in grid_module.STAGES:
        table = stage_table(records, stage, fold="search", **budget)
        expected = {config.label for config in grid_module.stage_grid(stage)}
        found = {row["label"] for row in table}
        if found != expected:
            print(f"  note: {stage} run holds {len(found)} configurations, "
                  f"the current grid defines {len(expected)}"
                  f"{'; ' + str(len(found - expected)) + ' not in the grid' if found - expected else ''}")
        lines.extend(table)
        print(f"\n=== {stage} (objective: {OBJECTIVE[stage]}, search fold) ===")
        print(f"{'wins':>5} {'feas':>6} {'med':>8} {'min':>8} {'max':>8} "
              f"{'alpha':>7} {'ncomp':>6}  config")
        for row in table:
            mark = " *production" if row["is_production"] else ""
            print(f"{row['wins']:>5} {row['n_feasible']:>3}/{row['n_subjects']:<2} "
                  f"{row['objective_median']:>8.3f} {row['objective_min']:>8.3f} "
                  f"{row['objective_max']:>8.3f} {row['alpha_delta_median']:>+7.2f} "
                  f"{row['n_selected_median']:>6.1f}  {row['label']}{mark}")
        if table and table[0]["wins"]:
            confirmations.append(confirm(records, stage, table[0]["label"], **budget))

    print("\n=== held-out confirmation ===")
    for row in confirmations:
        chosen = "production" if row["winner"] == row["production"] else row["winner"]
        print(f"{row['stage']}: {chosen} beats production on "
              f"{row['beats_production_on']}/{row['of']} confirm subjects, "
              f"delta median {row['delta_median']:+.3f} "
              f"[{row['delta_min']:+.3f}, {row['delta_max']:+.3f}]")

    table_path = args.output_root / "dss_param_search_table.tsv"
    columns = list(lines[0]) if lines else []
    table_path.write_text(
        "\t".join(columns) + "\n"
        + "\n".join("\t".join(str(row[column]) for column in columns) for row in lines) + "\n"
    )
    summary_path = args.output_root / "dss_param_search_summary.json"
    summary_path.write_text(json.dumps(
        {"budget": budget, "folds": folds, "confirmations": confirmations, "table": lines},
        indent=2, sort_keys=True,
    ))
    print(f"\nwrote {table_path}\nwrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
