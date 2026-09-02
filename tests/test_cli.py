"""The command tree, checked against itself rather than against a hand-list.

These replace a block of tests that each spelled out a few commands and
asserted argparse prints `--help`. Deriving the commands from the parser covers
every one of them instead of the subset someone remembered to add, and cannot
drift when a command is added.
"""

from __future__ import annotations

import argparse

import pytest

from meg_tokens.cli.main import _COMMAND_DEST, _HANDLERS, build_parser, main


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


def test_the_parser_offers_at_least_the_documented_domains():
    assert {domain for domain, _ in command_tree()} == set(_COMMAND_DEST)


def test_every_command_the_parser_offers_has_a_handler():
    """A command wired into the parser but not the dispatch table would parse
    fine and then raise KeyError at run time, after the user waited for a
    config to load."""
    assert set(command_tree()) == set(_HANDLERS)


def test_help_exits_zero_for_every_command(capsys):
    """One case per command would be forty-four collected tests to assert that
    argparse is argparse; the loop covers the same tree in one."""
    for domain, command in command_tree():
        with pytest.raises(SystemExit) as error:
            build_parser().parse_args([domain, command, "--help"])
        assert error.value.code == 0, f"{domain} {command}"
        assert capsys.readouterr().out.startswith("usage:"), f"{domain} {command}"


def test_report_behavior_takes_its_figure_selection_flags():
    args = build_parser().parse_args([
        "--config", "dummy.toml",
        "report", "behavior",
        "--figures", "headline", "core",
        "--formats", ".pdf",
        "--skip-missing",
        "--list-figures",
    ])
    assert args.figures == ["headline", "core"]
    assert args.formats == [".pdf"]
    assert args.skip_missing is True
    assert args.list_figures is True


def test_report_behavior_defaults_to_every_figure_in_both_formats():
    args = build_parser().parse_args(["--config", "dummy.toml", "report", "behavior"])
    assert args.figures == ["all"]
    assert args.formats == [".pdf", ".png"]
    assert args.skip_missing is False
    assert args.list_figures is False


def test_report_behavior_rejects_an_unknown_format():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["--config", "dummy.toml", "report", "behavior", "--formats", "bogus"]
        )


def test_subjects_are_normalized_once_on_entry(monkeypatch):
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
    main(["--config", "tokens.toml", "meg", "preprocess", "--subjects", "H1", "h2", "H03"])

    assert seen["subjects"] == ["H01", "H02", "H03"]


def test_an_unparseable_subject_fails_at_the_boundary(capsys):
    with pytest.raises(SystemExit):
        main(["--config", "tokens.toml", "meg", "preprocess", "--subjects", "nope"])
    assert "Invalid subject ID" in capsys.readouterr().err
