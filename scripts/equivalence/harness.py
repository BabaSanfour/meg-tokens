"""Prove a refactor left the behavioral pipeline's outputs unchanged.

The behavioral stages are cheap enough to re-run in full on the real data --
about two minutes for ingest, analyze, characterization, the five ``ssm``
aggregates and all eighteen figures -- so a refactor does not have to be
argued from tests. It can be *measured*: run the pipeline before the change,
run it after, and diff every derivative, figure and sidecar.

    python -m scripts.equivalence.harness prepare            # once
    python -m scripts.equivalence.harness run baseline       # before the change
    ...                                                      # refactor
    python -m scripts.equivalence.harness run candidate      # after it
    python -m scripts.equivalence.harness compare baseline candidate

``prepare`` copies the fixed inputs out of the real data root into a scratch
root: the Stage 0 behavioral layer and the per-subject ``ssm`` fits, which are
inputs here because fitting them takes cluster hours. Copies are real files,
never links, so nothing can write back into the source data.

What the comparison covers, and what it does not: every table Stage 1, 2 and 2b
write, every ``ssm`` group table the aggregates write, all eighteen figures
(rendering is deterministic, so PNGs are compared byte for byte) and every
sidecar. It does *not* cover the per-subject model fitting itself
(``ssm-fit``, ``ssm-evaluate``, ``ssm-recovery``, ``ssm-robustness``,
``ssm-exclusion-robustness``) or the recovery merge, whose outputs it
consumes; a change to the
likelihood, the solver or the model set has to be verified on the cluster.

Numeric columns are compared with ``rtol=1e-9``, which is far tighter than any
scientific tolerance and loose enough to absorb summation order. Rows are
matched after sorting on every column, so row order is irrelevant. The two
sidecar fields that record the source tree itself -- ``source_tree_sha256``
and ``git_dirty`` -- are excluded, because they are *supposed* to change when
the code does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from meg_tokens.core import ProjectConfig

RTOL = 1e-9
ATOL = 1e-12

# Recorded provenance of the source tree, not of the result. Comparing these
# would flag every refactor, including one that changed nothing observable.
PROVENANCE_FIELDS = frozenset({"source_tree_sha256", "git_dirty", "git_commit"})

STAGES: tuple[tuple[str, ...], ...] = (
    ("behavior", "ingest"),
    ("behavior", "analyze"),
    ("behavior", "characterization"),
    ("behavior", "ssm-aggregate"),
    ("behavior", "ssm-evaluate-aggregate"),
    ("behavior", "ssm-robustness-aggregate"),
    ("behavior", "ssm-exclusion-robustness-aggregate"),
    ("report", "behavior", "--figures", "all"),
)


def workspace() -> Path:
    """Where fixtures and snapshots live. Override with ``MEG_TOKENS_EQUIV_ROOT``."""
    return Path(
        os.environ.get("MEG_TOKENS_EQUIV_ROOT", Path.home() / ".cache" / "meg-tokens-equivalence")
    )


def source_root(config: str | None) -> Path:
    """The real data root the fixture is copied from."""
    if config:
        return Path(ProjectConfig.from_toml(config).bids_root)
    override = os.environ.get("MEG_TOKENS_DATA_ROOT")
    if override:
        return Path(override) / "BIDS"
    return Path(ProjectConfig.from_toml("tokens.toml").bids_root)


# --- prepare -----------------------------------------------------------------


def prepare(config: str | None) -> int:
    source = source_root(config)
    root = workspace() / "root"
    bids = root / "BIDS"
    group = bids / "derivatives" / "sub-group" / "beh"
    group.mkdir(parents=True, exist_ok=True)

    subjects = sorted(p.name for p in source.glob("sub-H*") if p.is_dir())
    for subject in subjects:
        target = bids / subject / "beh"
        target.mkdir(parents=True, exist_ok=True)
        for item in (source / subject / "beh").iterdir():
            shutil.copy2(item, target / item.name)

        derivative_source = source / "derivatives" / subject / "beh"
        if not derivative_source.is_dir():
            continue
        derivative_target = bids / "derivatives" / subject / "beh"
        derivative_target.mkdir(parents=True, exist_ok=True)
        for item in derivative_source.glob("*ssm*"):
            shutil.copy2(item, derivative_target / item.name)

    # Recovery runs only on the cluster, so its merged group tables are inputs
    # here in the same way the per-subject fits are.
    for item in (source / "derivatives" / "sub-group" / "beh").glob("*recovery*_beh.*"):
        shutil.copy2(item, group / item.name)

    config_path = workspace() / "tokens.toml"
    project = ProjectConfig.from_toml(config or "tokens.toml")
    ignore = "\n".join(f'    "{name}",' for name in project.behavior_ignore_files)
    config_path.write_text(
        "[project]\n"
        f'data_root = "{root}"\n'
        f'task = "{project.task}"\n'
        f"subject_exclusions = {list(project.subject_exclusions)}\n"
        f"infer_random_classes = {str(project.infer_random_classes).lower()}\n"
        f"behavior_ignore_files = [\n{ignore}\n]\n"
    )
    print(f"fixture ready: {root}\nconfig: {config_path}")
    return 0


# --- run ---------------------------------------------------------------------


def clear_outputs(root: Path) -> None:
    """Delete everything the staged commands write, and nothing they read.

    Without this a *removed* derivative is undetectable: the previous run's
    file is still on disk, the new run simply does not overwrite it, and the
    comparison reports no change. The inputs that must survive are the
    per-subject ``ssm`` fits and the repetition-scoped recovery tables, both
    produced on the cluster.
    """
    group = root / "sub-group"
    for item in (group / "beh").glob("*"):
        if "recovery" in item.name:
            continue
        item.unlink()
    for item in (group / "fig").glob("*"):
        item.unlink()
    for subject_dir in root.glob("sub-H*"):
        for item in (subject_dir / "beh").glob("*run-*_desc-*_beh.*"):
            item.unlink()


def run(label: str) -> int:
    config = workspace() / "tokens.toml"
    if not config.is_file():
        raise SystemExit("Run 'prepare' first; no fixture config at " + str(config))
    root = workspace() / "root" / "BIDS" / "derivatives"
    clear_outputs(root)

    for stage in STAGES:
        # The module entry point rather than the console script, so the harness
        # runs under whichever interpreter invoked it without needing the
        # environment's bin directory on PATH.
        command = [sys.executable, "-m", "meg_tokens.cli.main", "--config", str(config), *stage]
        print("  " + " ".join(stage))
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)

    snapshot = workspace() / label
    if snapshot.exists():
        shutil.rmtree(snapshot)
    (snapshot / "beh").mkdir(parents=True)
    (snapshot / "fig").mkdir(parents=True)
    for item in (root / "sub-group" / "beh").iterdir():
        shutil.copy2(item, snapshot / "beh" / item.name)
    for item in (root / "sub-group" / "fig").iterdir():
        shutil.copy2(item, snapshot / "fig" / item.name)
    for subject_dir in sorted(root.glob("sub-H*")):
        target = snapshot / "subject" / subject_dir.name
        target.mkdir(parents=True)
        for item in (subject_dir / "beh").glob("*run-*_desc-*_beh.tsv"):
            shutil.copy2(item, target / item.name)

    print(f"snapshot written: {snapshot}")
    return 0


# --- compare -----------------------------------------------------------------


def canonical(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.reindex(sorted(frame.columns), axis=1)
    if frame.empty:
        return frame
    keys = frame.astype(str)
    order = np.lexsort([keys[column].to_numpy() for column in reversed(keys.columns)])
    return frame.iloc[order].reset_index(drop=True)


def compare_frame(name: str, reference: pd.DataFrame, candidate: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    reference = canonical(reference)
    candidate = canonical(candidate)

    missing = [c for c in reference.columns if c not in candidate.columns]
    added = [c for c in candidate.columns if c not in reference.columns]
    if missing:
        problems.append(f"{name}: columns missing in candidate: {missing}")
    if added:
        problems.append(f"{name}: columns added in candidate: {added}")
    if len(reference) != len(candidate):
        problems.append(f"{name}: row count {len(reference)} -> {len(candidate)}")
        return problems

    for column in reference.columns:
        if column not in candidate.columns:
            continue
        left, right = reference[column], candidate[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            a = left.to_numpy(dtype=float)
            b = right.to_numpy(dtype=float)
            same = np.isclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True)
        else:
            a = left.astype(str).to_numpy()
            b = right.astype(str).to_numpy()
            same = a == b
        if not same.all():
            index = int(np.flatnonzero(~same)[0])
            problems.append(
                f"{name}.{column}: {int((~same).sum())}/{len(a)} values differ "
                f"(first at row {index}: {a[index]!r} -> {b[index]!r})"
            )
    return problems


def compare_tables(reference: Path, candidate: Path, label: str) -> list[str]:
    problems: list[str] = []
    left = {p.name: p for p in sorted(reference.glob("*_beh.tsv"))}
    right = {p.name: p for p in sorted(candidate.glob("*_beh.tsv"))}
    problems += [f"{label}: MISSING OUTPUT {n}" for n in sorted(set(left) - set(right))]
    problems += [f"{label}: NEW OUTPUT {n}" for n in sorted(set(right) - set(left))]
    for name in sorted(set(left) & set(right)):
        problems += compare_frame(
            f"{label}/{name}",
            pd.read_csv(left[name], sep="\t", low_memory=False),
            pd.read_csv(right[name], sep="\t", low_memory=False),
        )
    return problems


def normalize(value: object) -> object:
    """Drop source-tree provenance and absolute-path prefixes from a sidecar."""
    if isinstance(value, dict):
        return {
            key: normalize(item)
            for key, item in value.items()
            if key not in PROVENANCE_FIELDS
        }
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if isinstance(value, str) and "/" in value:
        return value.rsplit("/", 1)[-1]
    return value


def compare_sidecars(reference: Path, candidate: Path, label: str) -> list[str]:
    problems: list[str] = []
    left = {p.name: p for p in sorted(reference.glob("*.json"))}
    right = {p.name: p for p in sorted(candidate.glob("*.json"))}
    problems += [f"{label}: MISSING SIDECAR {n}" for n in sorted(set(left) - set(right))]
    problems += [f"{label}: NEW SIDECAR {n}" for n in sorted(set(right) - set(left))]
    for name in sorted(set(left) & set(right)):
        a = normalize(json.loads(left[name].read_text()))
        b = normalize(json.loads(right[name].read_text()))
        if a != b:
            changed = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
            problems.append(f"{label}: SIDECAR {name} differs in {changed or '(structure)'}")
    return problems


def compare(reference_label: str, candidate_label: str) -> int:
    reference = workspace() / reference_label
    candidate = workspace() / candidate_label
    problems = compare_tables(reference / "beh", candidate / "beh", "group")
    for subject_dir in sorted((reference / "subject").iterdir()):
        problems += compare_tables(
            subject_dir, candidate / "subject" / subject_dir.name, f"stage1/{subject_dir.name}"
        )

    left = {p.name: p for p in sorted((reference / "fig").glob("*.png"))}
    right = {p.name: p for p in sorted((candidate / "fig").glob("*.png"))}
    problems += [f"figure: MISSING {n}" for n in sorted(set(left) - set(right))]
    problems += [f"figure: NEW {n}" for n in sorted(set(right) - set(left))]
    for name in sorted(set(left) & set(right)):
        if hashlib.md5(left[name].read_bytes()).digest() != hashlib.md5(
            right[name].read_bytes()
        ).digest():
            problems.append(f"figure: PIXELS DIFFER {name}")

    problems += compare_sidecars(reference / "fig", candidate / "fig", "figure")
    problems += compare_sidecars(reference / "beh", candidate / "beh", "group")

    if not problems:
        print("EQUIVALENT: tables, figures and sidecars agree")
        return 0
    print(f"{len(problems)} difference(s):")
    for problem in problems:
        print("  " + problem)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scripts.equivalence.harness", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="Copy the fixed inputs into a scratch root.")
    prepare_parser.add_argument("--config", help="Project TOML to read the real data root from.")
    run_parser = commands.add_parser("run", help="Run every stage and snapshot its outputs.")
    run_parser.add_argument("label")
    compare_parser = commands.add_parser("compare", help="Diff two snapshots.")
    compare_parser.add_argument("reference")
    compare_parser.add_argument("candidate")
    args = parser.parse_args(argv)

    if args.command == "prepare":
        return prepare(args.config)
    if args.command == "run":
        return run(args.label)
    return compare(args.reference, args.candidate)


if __name__ == "__main__":
    sys.exit(main())
