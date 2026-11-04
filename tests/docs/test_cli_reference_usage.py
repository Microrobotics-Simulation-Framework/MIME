"""The usage line the generated CLI reference prints for a command (``mime.docs._generated.usage``): what a reader must
give is written bare, what they may leave out is in brackets, a positional that takes one value is not written as a
list, and a mutually exclusive group is written as one.

The arms are parsers built for the test (a required and an optional flag, ``nargs`` of ``?``, ``1``, ``*`` and ``+`` on
a positional and on an option, both kinds of mutually exclusive group), and the real parser, whose every command is
held to what argparse itself refuses to omit: the oracle parses an empty command line and reads the names argparse
says are required, which is not the attribute the generator reads. The planted mismatch is the first rendering this
function had (every option in brackets, every optional positional a list), which the oracle must flag on every run.
Nothing here draws anything."""

import argparse
import re

import pytest

from mime.cli import build_parser
from mime.cli._guide import _subparsers_of  # pyright: ignore[reportPrivateUsage]
from mime.docs._generated import usage


def everything() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mime", add_help=False)
    parser.add_argument("--must", required=True)
    parser.add_argument("--may")
    parser.add_argument("--flag", action="store_true")
    parser.add_argument("--maybe", nargs="?")
    parser.add_argument("--several", nargs="+", metavar="ITEM")
    parser.add_argument("--anything", nargs="*", metavar="ITEM")
    either = parser.add_mutually_exclusive_group()
    either.add_argument("--left", action="store_true")
    either.add_argument("--right", action="store_true")
    exactly_one = parser.add_mutually_exclusive_group(required=True)
    exactly_one.add_argument("--up", action="store_true")
    exactly_one.add_argument("--down", action="store_true")
    parser.add_argument("single")
    parser.add_argument("at_most_one", nargs="?")
    parser.add_argument("one_in_a_list", nargs=1)
    parser.add_argument("at_least_one", nargs="+")
    parser.add_argument("any_number", nargs="*")
    return parser


def test_every_kind_of_argument_is_written_as_argparse_takes_it():
    assert usage(("x",), everything()) == (
        "mime x --must MUST [--may MAY] [--flag] [--maybe [MAYBE]] [--several ITEM [ITEM ...]] [--anything [ITEM ...]] "
        "[--left | --right] (--up | --down) single [at_most_one] one_in_a_list at_least_one [at_least_one ...] "
        "[any_number ...]")


def test_a_command_with_subcommands_says_so_and_a_hidden_option_is_left_out():
    parser = argparse.ArgumentParser(prog="mime", add_help=False)
    parser.add_argument("--shown", required=True)
    parser.add_argument("--hidden", help=argparse.SUPPRESS)
    parser.add_subparsers(dest="command")
    assert usage(("group",), parser) == "mime group <command> --shown SHOWN"


def test_the_commands_the_review_found_misprinted_are_printed_as_argparse_takes_them():
    held = _subparsers_of(build_parser())
    assert held is not None
    run = usage(("run",), held.choices["run"])
    assert " --out OUT " in run and "[--out" not in run and run.endswith(" folder")
    env = _subparsers_of(held.choices["env"])
    assert env is not None
    check = usage(("env", "check"), env.choices["check"])
    assert check.startswith("mime env check --variant VARIANT [--gpu TYPE]") and check.endswith(" [folder]")
    assert usage(("pull",), held.choices["pull"]).endswith(" [source]")
    push = usage(("push",), held.choices["push"])
    assert push.endswith(" [folder]") and "[--init | --setup | --defaults | --status | --discard | --publish]" in push
    credentials = _subparsers_of(held.choices["credentials"])
    assert credentials is not None
    assert usage(("credentials", "status"), credentials.choices["status"]).endswith(" [{zenodo,zenodo-sandbox}]")
    fetch = _subparsers_of(held.choices["assets"])
    assert fetch is not None
    assert usage(("assets", "fetch"), fetch.choices["fetch"]).endswith(" [--all] [name-or-folder ...]")


# -- the oracle: what argparse itself refuses to leave out ------------------------------------------------------------

def refused_to_omit(parser: argparse.ArgumentParser, capsys: pytest.CaptureFixture[str]) -> set[str]:
    """The names argparse says are required when it parses an empty command line (the first spelling of an option)."""
    capsys.readouterr()
    try:
        parser.parse_args([])
    except SystemExit:
        pass
    message = capsys.readouterr().err
    listed = re.search(r"the following arguments are required: (.+)", message)
    return {name.split("/")[0].strip() for name in listed.group(1).splitlines()[0].split(",")} if listed else set()


def written_bare(line: str, words: tuple[str, ...]) -> set[str]:
    """The names a usage line writes outside every bracket and parenthesis, apart from ``mime`` and the command's words
    (a value after an option is part of that option)."""
    flat, depth = "", 0
    for character in line:
        depth += character in "[(" and 1 or character in "])" and -1 or 0
        flat += character if depth == 0 and character not in "[])" else " "
    tokens = [t for t in flat.split() if t not in ("mime", *words, "<command>", "...")]
    names: set[str] = set()
    for at, token in enumerate(tokens):
        previous_is_option = at > 0 and tokens[at - 1].startswith("--")
        if token.startswith("--") or not previous_is_option:
            names.add(token)
    return names


def mismatch(parser: argparse.ArgumentParser, words: tuple[str, ...], line: str,
             capsys: pytest.CaptureFixture[str]) -> set[str]:
    """What the line writes as required that argparse does not, and the reverse: options by name, and positionals by
    count (argparse names a positional by its metavar or its dest, the line by its choices)."""
    refused = refused_to_omit(parser, capsys)
    written = written_bare(line, words)
    options = {name for name in written if name.startswith("--")}
    wrong = options ^ {name for name in refused if name.startswith("--")}
    positionals_written = len(written - options)
    positionals_refused = len({name for name in refused if not name.startswith("--")})
    return wrong | ({f"{positionals_written} bare positionals, argparse refuses {positionals_refused}"}
                    if positionals_written != positionals_refused else set())


def leaves(parser: argparse.ArgumentParser, words: tuple[str, ...] = ()):
    held = _subparsers_of(parser)
    if held is None:
        yield words, parser
        return
    for name, child in held.choices.items():
        yield from leaves(child, (*words, name))


def old_usage(parser: argparse.ArgumentParser) -> str:
    """The first rendering: every option in brackets, every optional positional a list."""
    parts = [f"[{o.option_strings[0]}{' ' + (o.metavar or o.dest.upper()) if o.nargs != 0 else ''}]"
             for o in parser._actions if o.option_strings and "-h" not in o.option_strings]  # pyright: ignore[reportPrivateUsage]
    parts += [(p.metavar or p.dest) if p.nargs not in ("*", "+", "?") else f"[{p.metavar or p.dest} ...]"
              for p in parser._actions if not p.option_strings]  # pyright: ignore[reportPrivateUsage]
    return " ".join(["mime", "x", *parts])


def test_the_control_a_parser_whose_required_flag_is_written_bare_has_no_mismatch(capsys):
    parser = argparse.ArgumentParser(prog="mime", add_help=False)
    parser.add_argument("--must", required=True)
    parser.add_argument("--may")
    parser.add_argument("single")
    parser.add_argument("maybe", nargs="?")
    assert mismatch(parser, ("x",), usage(("x",), parser), capsys) == set()


def test_planted_a_required_flag_written_in_brackets_is_a_mismatch_every_run(capsys):
    parser = argparse.ArgumentParser(prog="mime", add_help=False)
    parser.add_argument("--must", required=True)
    parser.add_argument("--may")
    parser.add_argument("single")
    assert mismatch(parser, ("x",), old_usage(parser), capsys) == {"--must"}


def test_every_command_writes_as_required_exactly_what_argparse_refuses_to_omit(capsys):
    parser = build_parser()
    seen = 0
    for words, leaf in leaves(parser):
        assert not any(group.required for group in leaf._mutually_exclusive_groups), words  # pyright: ignore[reportPrivateUsage]
        assert mismatch(leaf, words, usage(words, leaf), capsys) == set(), (words, usage(words, leaf))
        seen += 1
    assert seen >= 20, "the walk reached the commands"
