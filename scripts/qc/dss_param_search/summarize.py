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


def feasible(
    row: dict, production: dict | None = None, *, alpha_slack: float, dimension_budget: int
) -> bool:
    """Whether a configuration's collateral is acceptable on this subject.

    The alpha bar is *production's own cost on the same recording*, plus slack
    -- not an absolute number. An absolute bar is the mistake B5 was criticised
    for and that a first pass here repeated: 0.5 dB read as a reasonable
    ceiling, and production itself exceeds it on 15 of 32 subjects for the
    cardiac stage. The table then ranked configurations by which subjects
    happened to fall under a round number, and eight subjects had no admissible
    configuration at all.

    Relative and per-subject, the question is the one worth asking: does this
    configuration score better without costing more than what we already run?
    """
    if "failed" in row:
        return False
    if row.get("n_selected", 0) > dimension_budget:
        return False
    alpha = row.get("alpha_delta_db")
    if alpha is None:
        return False
    if production is None or "alpha_delta_db" not in production:
        return True
    return alpha >= production["alpha_delta_db"] - alpha_slack


def load(input_root: Path) -> list[dict]:
    """Every per-subject result, in subject order."""
    records = [
        json.loads(path.read_text())
        for path in sorted(Path(input_root).glob("sub-*_dsssearch.json"))
    ]
    if not records:
        raise SystemExit(f"no sub-*_dsssearch.json under {input_root}")
    return records


def _admissible(
    records: list[dict], stage: str, *, alpha_slack: float, dimension_budget: int
) -> dict[str, list[tuple[dict, dict]]]:
    """Per subject, its recordings for one stage as ``(rows, affordable)`` pairs."""
    production_label = grid_module.production_config(stage).label
    by_subject: dict[str, list[tuple[dict, dict]]] = {}
    for record in records:
        rows = {row["label"]: row for row in record["rows"] if row["stage"] == stage}
        incumbent = rows.get(production_label)
        affordable = {
            label: feasible(row, incumbent, alpha_slack=alpha_slack,
                            dimension_budget=dimension_budget)
            for label, row in rows.items()
        }
        by_subject.setdefault(record["subject"], []).append((rows, affordable))
    return by_subject


def _eligible(records: list[dict], stage: str, *, fold: str, muscle_floor: float) -> list[dict]:
    """Recordings in one fold that can speak to one stage."""
    return [
        record for record in records
        if record["fold"] == fold
        and (stage != "muscle" or record["presence"]["muscle_annotated_fraction"] >= muscle_floor)
    ]


def stage_table(
    records: list[dict],
    stage: str,
    *,
    fold: str,
    alpha_slack: float,
    dimension_budget: int,
    muscle_floor: float,
) -> list[dict]:
    """One row per configuration, counted at two units.

    ``wins`` counts recordings and ``subject_wins`` counts subjects, and the
    table is ranked on the second. A subject's ten runs are one subject's
    physiology recorded ten times, not ten independent draws, so a recording
    count multiplies the apparent evidence by ten and lets one unusual subject
    with a full complement of runs outvote three ordinary ones. A subject is
    awarded to whichever configuration has the best median across that
    subject's own admissible recordings.
    """
    objective = OBJECTIVE[stage]
    production_label = grid_module.production_config(stage).label
    eligible = _eligible(records, stage, fold=fold, muscle_floor=muscle_floor)
    by_subject = _admissible(
        eligible, stage, alpha_slack=alpha_slack, dimension_budget=dimension_budget
    )

    def scored(rows: dict, affordable: dict) -> list[dict]:
        return [
            row for label, row in rows.items()
            if affordable[label] and row.get(objective) == row.get(objective)
        ]

    wins: dict[str, int] = {}
    subject_wins: dict[str, int] = {}
    for recordings in by_subject.values():
        for rows, affordable in recordings:
            candidates = scored(rows, affordable)
            if candidates:
                best = max(candidates, key=lambda row: row[objective])
                wins[best["label"]] = wins.get(best["label"], 0) + 1
        medians = {}
        for rows, affordable in recordings:
            for row in scored(rows, affordable):
                medians.setdefault(row["label"], []).append(row[objective])
        if medians:
            best = max(medians, key=lambda label: statistics.median(medians[label]))
            subject_wins[best] = subject_wins.get(best, 0) + 1

    # Rows come from the labels the run actually produced, not from the current
    # grid. Deriving them from the grid means editing `grid.py` and then
    # re-summarising an older run silently drops every configuration whose label
    # moved -- the table would simply be shorter, with nothing to notice.
    labels = sorted({
        row["label"] for record in records for row in record["rows"]
        if row["stage"] == stage
    })
    table = []
    for label in labels:
        rows, n_feasible = [], 0
        for recordings in by_subject.values():
            for row_map, affordable in recordings:
                if label in row_map:
                    rows.append(row_map[label])
                    n_feasible += bool(affordable[label])
        values = [row[objective] for row in rows if row.get(objective) == row.get(objective)]
        alphas = [row["alpha_delta_db"] for row in rows if "alpha_delta_db" in row]
        counts = [row["n_selected"] for row in rows if "n_selected" in row]
        table.append({
            "stage": stage,
            "label": label,
            "is_production": label == production_label,
            "n_subjects": len(by_subject),
            "n_recordings": len(rows),
            "n_feasible": n_feasible,
            "n_failed": sum("failed" in row for row in rows),
            "subject_wins": subject_wins.get(label, 0),
            "wins": wins.get(label, 0),
            "objective_median": statistics.median(values) if values else float("nan"),
            "objective_min": min(values) if values else float("nan"),
            "objective_max": max(values) if values else float("nan"),
            "alpha_delta_median": statistics.median(alphas) if alphas else float("nan"),
            "n_selected_median": statistics.median(counts) if counts else float("nan"),
        })
    return sorted(
        table,
        key=lambda row: (-row["subject_wins"], -row["wins"], -row["objective_median"]),
    )


def confirm(
    records: list[dict], stage: str, winner: str, *,
    alpha_slack: float, dimension_budget: int, muscle_floor: float,
) -> dict:
    """Head-to-head against production on the held-out fold, one point per subject.

    Paired per subject on the median across that subject's runs, so a subject
    with ten recordings does not count ten times. A winner that *is* production
    is reported as such rather than as a zero-delta contest with itself.
    """
    objective = OBJECTIVE[stage]
    production = grid_module.production_config(stage).label
    if winner == production:
        return {
            "stage": stage, "winner": winner, "production": production,
            "winner_is_production": True,
            "beats_production_on": 0, "of": 0,
            "delta_median": float("nan"),
            "delta_min": float("nan"), "delta_max": float("nan"),
        }

    eligible = _eligible(records, stage, fold="confirm", muscle_floor=muscle_floor)
    by_subject = _admissible(
        eligible, stage, alpha_slack=alpha_slack, dimension_budget=dimension_budget
    )
    beats, total, deltas = 0, 0, []
    for recordings in by_subject.values():
        paired = []
        for rows, affordable in recordings:
            if winner not in rows or production not in rows:
                continue
            candidate, incumbent = rows[winner], rows[production]
            if not affordable[winner]:
                continue
            if candidate.get(objective) != candidate.get(objective):
                continue
            if incumbent.get(objective) != incumbent.get(objective):
                continue
            paired.append(candidate[objective] - incumbent[objective])
        if not paired:
            continue
        delta = statistics.median(paired)
        total += 1
        deltas.append(delta)
        beats += delta > 0
    return {
        "stage": stage, "winner": winner, "production": production,
        "winner_is_production": False,
        "beats_production_on": beats, "of": total,
        "delta_median": statistics.median(deltas) if deltas else float("nan"),
        "delta_min": min(deltas) if deltas else float("nan"),
        "delta_max": max(deltas) if deltas else float("nan"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--alpha-slack", default=0.1, type=float,
                        help="dB of alpha-band loss a configuration may cost *beyond* "
                             "what production costs on the same recording")
    parser.add_argument("--dimension-budget", default=20, type=int,
                        help="components of 270 a stage may remove")
    parser.add_argument("--muscle-floor", default=0.005, type=float,
                        help="annotated muscle fraction below which a recording "
                             "carries no muscle and is excluded from that stage")
    args = parser.parse_args(argv)

    records = load(args.input_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    budget = {
        "alpha_slack": args.alpha_slack,
        "dimension_budget": args.dimension_budget,
        "muscle_floor": args.muscle_floor,
    }

    folds = {record["subject"]: record["fold"] for record in records}
    print(f"{len(records)} recordings from {len(folds)} subjects: "
          f"{sum(f == 'search' for f in folds.values())} search subjects, "
          f"{sum(f == 'confirm' for f in folds.values())} confirm subjects")
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
        print(f"{'subj':>4} {'rec':>5} {'feasible':>10} {'med':>8} {'min':>8} {'max':>8} "
              f"{'alpha':>7} {'ncomp':>6}  config")
        for row in table:
            mark = " *production" if row["is_production"] else ""
            print(f"{row['subject_wins']:>4} {row['wins']:>5} "
                  f"{row['n_feasible']:>4}/{row['n_recordings']:<5} "
                  f"{row['objective_median']:>8.3f} {row['objective_min']:>8.3f} "
                  f"{row['objective_max']:>8.3f} {row['alpha_delta_median']:>+7.2f} "
                  f"{row['n_selected_median']:>6.1f}  {row['label']}{mark}")
        if table and table[0]["subject_wins"]:
            confirmations.append(confirm(records, stage, table[0]["label"], **budget))

    print("\n=== held-out confirmation ===")
    for row in confirmations:
        if row["winner_is_production"]:
            print(f"{row['stage']}: production won the search fold -- nothing to confirm")
            continue
        print(f"{row['stage']}: {row['winner']}\n"
              f"    beats production on {row['beats_production_on']}/{row['of']} "
              f"confirm subjects, delta median {row['delta_median']:+.3f} "
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
