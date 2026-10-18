"""The cluster launchers must stay in step with the CLI they submit.

`scripts/ssm_fit_array.sh` once passed `--model-set mechanistic` for months
after that flag was removed from `behavior ssm-fit`. Nothing caught it: the
script is only ever run through Slurm, where the failure is an argparse exit
buried in a job log. These two tests are the cheap version of running them.
"""

from __future__ import annotations

import contextlib
import io
import re
import shlex
from pathlib import Path

import pytest

from meg_tokens.cli import build_parser

SCRIPTS = Path(__file__).parents[1] / "scripts"

# Values that satisfy the arguments whose shell variable would otherwise be
# substituted with something the parser rejects.
CHOICE_VALUES = {
    "--rule": "complete_token_log_alignment",
    "--configuration": "tau_slow",
    "--stream": "dss",
    "--streams": "dss",
    "--locks": "go",
    "--fit": "pooled",
    "--measure": "log10",
    "--space": "HCPMMP1",
}


def invocations(script: Path) -> list[list[str]]:
    """Every `meg-tokens` command the script runs, as argv, variables filled in."""
    joined = re.sub(r"\\\n\s*", " ", script.read_text())
    found = []
    for line in joined.splitlines():
        line = line.strip()
        if line.startswith("echo") or "meg-tokens --config" not in line:
            continue
        command = line[line.index("meg-tokens"):].split("|")[0].split(")")[0].strip()
        command = re.sub(r'"?\$\{?[A-Za-z_][A-Za-z0-9_]*(:-[^}]*)?\}?"?', "3", command)
        argv = shlex.split(command)[1:]
        for flag, value in CHOICE_VALUES.items():
            if flag in argv:
                argv[argv.index(flag) + 1] = value
        found.append(argv)
    return found


def test_every_scripted_command_is_one_the_cli_accepts():
    for script in sorted(SCRIPTS.glob("*.sh")):
        for argv in invocations(script):
            stderr = io.StringIO()
            try:
                with contextlib.redirect_stderr(stderr):
                    build_parser().parse_args(argv)
            except SystemExit:  # pragma: no cover - only on a real mismatch
                pytest.fail(f"{script.name} runs a command the CLI rejects:\n  {stderr.getvalue()}")


def test_an_ssm_launcher_is_named_for_the_command_it_submits():
    """`ssm_recovery_aggregate.sh` submits `behavior ssm-recovery-aggregate`.

    Ten launchers differing only in which stage they call are impossible to
    keep straight by reading them, so the name carries the answer.
    """
    for script in sorted(SCRIPTS.glob("ssm_*.sh")):
        expected = script.stem.removesuffix("_array").replace("_", "-")
        commands = {argv[argv.index("behavior") + 1] for argv in invocations(script)}
        # Array launchers also ask the CLI for the subject list to index into.
        assert commands - {"subjects"} == {expected}, f"{script.name} submits {sorted(commands)}"
