"""The command tree, checked against itself rather than against a hand-list.

These replace a block of tests that each spelled out a few commands and
asserted argparse prints `--help`. Deriving the commands from the parser covers
every one of them instead of the subset someone remembered to add, and cannot
drift when a command is added.
"""

from __future__ import annotations

import argparse


from meg_tokens.cli.main import _HANDLERS, build_parser, main


def command_tree() -> list[tuple[str, str]]:
    """Every (domain, command) pair the parser accepts."""
    def choices(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                return action.choices
        return {}

    return [
        (domain, command)
        for domain, domain_parser in choices(build_parser()).items()
        for command in choices(domain_parser)
    ]


def test_every_command_the_parser_offers_has_a_handler():
    """A command wired into the parser but not the dispatch table would parse
    fine and then raise KeyError at run time, after the user waited for a
    config to load."""
    assert set(command_tree()) == set(_HANDLERS)


def test_subjects_are_normalized_once_on_entry(monkeypatch, tmp_path):
    """The boundary owns the canonical form, so nothing downstream re-checks.

    `--subjects H1` used to reach the workflows as the literal string, which is
    why every layer normalized defensively. Doing it here means a path helper
    can trust its caller.
    """
    seen = {}

    class _Result:
        outputs = ()

    monkeypatch.setitem(
        _HANDLERS,
        ("meg", "preprocess"),
        lambda project, args: (seen.update(subjects=args.subjects), _Result())[1],
    )
    config = tmp_path / "tokens.toml"
    config.write_text(f'[project]\ndata_root = "{tmp_path}"\n')
    main(["--config", str(config), "meg", "preprocess", "--subjects", "H1", "h2", "H03"])

    assert seen["subjects"] == ["H01", "H02", "H03"]
