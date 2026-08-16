import pytest

from meg_tokens.cli.main import build_parser


def test_report_behavior_accepts_the_new_flags():
    parser = build_parser()
    args = parser.parse_args([
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


def test_report_behavior_defaults():
    parser = build_parser()
    args = parser.parse_args(["--config", "dummy.toml", "report", "behavior"])
    assert args.figures == ["all"]
    assert args.formats == [".pdf", ".png"]
    assert args.skip_missing is False
    assert args.list_figures is False


def test_report_behavior_rejects_an_unknown_format():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--config", "dummy.toml", "report", "behavior", "--formats", "bogus"]
        )


def test_report_behavior_help_lists_the_new_flags(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit) as error:
        parser.parse_args(
            ["--config", "dummy.toml", "report", "behavior", "--help"]
        )
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "--figures" in output
    assert "--list-figures" in output
    assert "--skip-missing" in output
