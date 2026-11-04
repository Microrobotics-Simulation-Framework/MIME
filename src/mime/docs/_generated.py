"""The text of a guide that is generated and never written: the command line's reference from the parser, the profiles'
definitions and rules from the code's own tables, the declared limits of MIME's entries from the registry, and the
build report's fields from ``BuildReport``. A guide page names one with ``<!-- mime-generate: <name> -->``
(``_executable.py``); the build replaces the line with the text, so the page that is only that line has no hand-written
text to drift, and ``mime docs check`` fails on a hand's addition to it.

Also the one check that a hand-written page's headings equal the fields of the dataclass it describes."""

from __future__ import annotations

import argparse
import dataclasses
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

from mime.cli._guide import _subparsers_of  # pyright: ignore[reportPrivateUsage]
from mime.docs._executable import DIRECTIVE, parse

REFERENCE_PAGE = Path("docs/guide/cli.md")
BUILD_REPORT_PAGE = Path("docs/guide/build_report.md")
#: The pages that hold a generated text and nothing else, by the name of the text.
ONLY_GENERATED: Mapping[str, str] = {"docs/guide/cli.md": "cli-reference"}


def _is_positional(parser: argparse.ArgumentParser, action: argparse.Action) -> bool:
    return not action.option_strings and action is not _subparsers_of(parser)


def _help(parser: argparse.ArgumentParser, action: argparse.Action) -> str:
    """An action's help with its ``%(...)s`` fields filled, on one line."""
    if action.help is None:
        return ""
    shown = parser._get_formatter()._expand_help(action)
    return " ".join(shown.split())


def _item(label: str, text: str) -> str:
    return f"- {label}: {text}" if text else f"- {label}"


def _values(action: argparse.Action) -> str:
    """The values an action takes, as argparse's own usage spells them: ``M``, ``[M]`` for at most one, ``[M ...]`` for
    any number and ``M [M ...]`` for at least one. An option with ``nargs`` of ``?`` or ``*`` may also be given bare."""
    name = _metavar(action)
    match action.nargs:
        case 0:
            return ""
        case "?":
            return f"[{name}]"
        case "*":
            return f"[{name} ...]"
        case "+":
            return f"{name} [{name} ...]"
        case int(count):
            return " ".join([name] * count)
        case _:
            return name


def _word(action: argparse.Action) -> str:
    """One action as the usage line spells it, without the brackets that say it may be left out."""
    if not action.option_strings:
        return _values(action)
    return " ".join(part for part in (action.option_strings[0], _values(action)) if part)


def usage(words: tuple[str, ...], parser: argparse.ArgumentParser) -> str:
    """The usage line of the command ``words`` (``parser``), read from the parser's actions: a required option is
    written bare and an optional one in brackets, a positional that may be left out in brackets (``[x]`` for at most
    one value, ``[x ...]`` for any number), and the options of a mutually exclusive group as ``[--a | --b]``, in parentheses
    when the group is required."""
    parts = ["mime", *words]
    if _subparsers_of(parser) is not None:
        parts.append("<command>")
    groups = list(parser._mutually_exclusive_groups)
    done: list[object] = []
    for option in parser._actions:
        if not option.option_strings or option.help == argparse.SUPPRESS or "-h" in option.option_strings:
            continue
        group = next((g for g in groups if option in g._group_actions), None)
        if group is None:
            parts.append(f"[{_word(option)}]" if not option.required else _word(option))
        elif not any(group is seen for seen in done):
            done.append(group)
            members = " | ".join(_word(m) for m in group._group_actions if m.help != argparse.SUPPRESS)
            parts.append(f"({members})" if group.required else f"[{members}]")
    for positional in parser._actions:
        if _is_positional(parser, positional):
            parts.append(_word(positional))
    return " ".join(parts)


def _metavar(action: argparse.Action) -> str:
    if isinstance(action.metavar, str):
        return action.metavar
    if action.choices is not None and not action.option_strings:
        return "{" + ",".join(str(c) for c in action.choices) + "}"
    return action.dest.upper() if action.option_strings else action.dest


def _section(parser: argparse.ArgumentParser, words: tuple[str, ...], summary: str, out: list[str],
             examples: Mapping[tuple[str, ...], str]) -> None:
    depth = min(len(words) + 2, 6)
    out += [f"{'#' * depth} `mime {' '.join(words)}`", ""]
    if summary:
        out += [summary, ""]
    if parser.description:
        out += [" ".join(parser.description.split()), ""]
    out += ["```", usage(words, parser), "```", ""]
    arguments = [a for a in parser._actions if _is_positional(parser, a)]
    for argument in arguments:
        out.append(_item(f"`{_metavar(argument)}`", _help(parser, argument)))
    for option in parser._actions:
        if not option.option_strings or option.help == argparse.SUPPRESS or "-h" in option.option_strings:
            continue
        spellings = ", ".join(f"`{s}`" for s in option.option_strings)
        value = f" `{_metavar(option)}`" if option.nargs != 0 else ""
        choices = f" (one of {', '.join(f'`{c}`' for c in option.choices)})" if option.choices else ""
        out.append(_item(f"{spellings}{value}", _help(parser, option) + choices))
    if arguments or any(o.option_strings and o.help != argparse.SUPPRESS and "-h" not in o.option_strings
                        for o in parser._actions):
        out.append("")
    if words in examples:
        out += ["Example:", "", "```sh", examples[words], "```", ""]
    held = _subparsers_of(parser)
    if held is not None:
        described = {a.dest: a.help or "" for a in held._get_subactions()}
        for name, child in held.choices.items():
            _section(child, (*words, name), " ".join(described.get(name, "").split()), out, examples)


def cli_reference(_root: Path) -> str:
    """The reference for every command of ``mime``: its summary, usage, arguments and flags with their help, and its
    example, walked from the parser. Built from the parser's actions and not from ``format_help``, whose layout
    differs between the Pythons MIME supports."""
    from mime.cli import build_parser
    from mime.cli._guide import EXAMPLES, GROUPS

    parser = build_parser()
    out = ["# The `mime` command line", "",
           "Generated from the command line's own parser by `mime docs build`; nothing here is written by hand. "
           "The flags of `mime` itself come first (each says whether it may come anywhere on the line), then each "
           "command, grouped as `mime --help` groups them.", "", "## Options of `mime` itself", ""]
    for option in parser._actions:
        if option.option_strings and option.help != argparse.SUPPRESS and "-h" not in option.option_strings:
            out.append(_item(", ".join(f"`{s}`" for s in option.option_strings), _help(parser, option)))
    out.append("")
    held = _subparsers_of(parser)
    assert held is not None
    described = {a.dest: a.help or "" for a in held._get_subactions()}
    for group, names in GROUPS:
        out += [f"## {group}", ""]
        for name in names:
            _section(held.choices[name], (name,), " ".join(described.get(name, "").split()), out, EXAMPLES)
    return "\n".join(out).rstrip("\n") + "\n"


def profiles(_root: Path) -> str:
    """The profiles as the code defines them: ``Profile``'s own docstring, whether each is built, and the table of
    what each does with each class of finding (``profile_rules.txt``), each class with what covers it and its codes."""
    from mime.validity import BUILT, FINDING_CLASSES, PROFILE_RULES, Profile, profile_of

    out = ["The definition, as the code states it:", "", f"> {' '.join((Profile.__doc__ or '').split())}", ""]
    out += ["| Profile | Built |", "|---|---|"]
    for profile in Profile:
        _, refusal = profile_of(profile)
        out.append(f"| `{profile.value}` | " + ("yes" if profile in BUILT else f"no: refused as `{refusal[0]}`"
                                                  if refusal else "no") + " |")
    out += ["", "What each built profile does with a class of finding (`generate` makes a waiver with a pending basis; "
            "`basis` accepts only the experiment's own waiver or source with a basis; `list` lists the finding and "
            "never refuses it; `refuse` refuses it):", ""]
    classes = sorted({name for _, name in PROFILE_RULES.rules})
    out += ["| Class of finding | Covered by | " + " | ".join(f"`{p.value}`" for p in BUILT) + " |",
            "|---|---|" + "---|" * len(BUILT)]
    for name in classes:
        cells: list[str] = []
        for profile in BUILT:
            rule = PROFILE_RULES.rules.get((profile, name))
            cells.append(f"`{rule.action.value}`" if rule is not None else "nothing")
        out.append(f"| `{name}` | `{FINDING_CLASSES.covers[name].value}` | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def limits(_root: Path) -> str:
    """The limits MIME's entries declare of themselves (their ``validity`` lines), as the entry pages print them, one
    list per entry. They are declared, not yet checked; a line quoted here is the entry page's, which is where its
    numbers are judged."""
    from mime.docs._export import export, load_namespaces
    from mime.docs._markdown import mapping, strings
    from mime.vocabulary import MIME_NAMESPACE

    data = export(load_namespaces())
    out: list[str] = []
    for record in sorted((r for r in cast(list[Mapping[str, Any]], data["entries"]) if r["namespace"] == MIME_NAMESPACE),
                         key=lambda r: (str(r["kind"]), str(r["name"]), str(r["version"]))):
        held = mapping(mapping(record.get("class_vars")).get("validity"))
        lines = strings(held["value"]) if "value" in held else []
        if not lines:
            continue
        out += [f"**{record['kind']} `{record['name']}` {record['version']}**", ""]
        out += [f"- {' '.join(line.split())}" for line in lines]
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


def build_report_fields(_root: Path) -> str:
    """``BuildReport``'s fields, in the order the class declares them, with their types: the names the page that reads
    the report must have a heading for."""
    from mime.build import BuildReport

    rows = ["| Field | Type |", "|---|---|"]
    rows += [f"| `{f.name}` | `{' '.join(str(f.type).split())}` |" for f in dataclasses.fields(BuildReport)]
    return "\n".join(rows) + "\n"


GENERATORS: Mapping[str, Callable[[Path], str]] = {
    "cli-reference": cli_reference, "profiles": profiles, "limits": limits, "build-report-fields": build_report_fields}


def build_report_problems(root: Path) -> list[str]:
    """The page that reads the build report has one level-2 heading, `` `name` ``, for each field of ``BuildReport``
    and none for anything else: a field with no heading, and a heading that is no field, fail."""
    from mime.build import BuildReport

    page = root / BUILD_REPORT_PAGE
    if not page.is_file():
        return [f"{BUILD_REPORT_PAGE}: the page that reads the build report is missing"]
    fields = [f.name for f in dataclasses.fields(BuildReport)]
    headings: dict[str, int] = {}
    problems: list[str] = []
    for number, line in parse(page.read_text(encoding="utf-8")).prose:
        held = re.match(r"^##\s+(?P<text>.+?)\s*$", line)
        if held is None:
            continue
        name = re.fullmatch(r"`([A-Za-z_][A-Za-z0-9_]*)`", held["text"])
        if name is None:
            problems.append(f"{BUILD_REPORT_PAGE}:{number}: the heading {held['text']!r} is not a field name in code "
                            f"ticks; a level-2 heading here is a BuildReport field")
        elif name[1] not in fields:
            problems.append(f"{BUILD_REPORT_PAGE}:{number}: the heading `{name[1]}` is not a field of BuildReport "
                            f"({', '.join(fields)})")
        else:
            headings[name[1]] = headings.get(name[1], 0) + 1
    problems += [f"{BUILD_REPORT_PAGE}: BuildReport's field `{name}` has no heading; say what it means and what to do"
                 for name in fields if name not in headings]
    problems += [f"{BUILD_REPORT_PAGE}: the field `{name}` has {count} headings; one" for name, count in headings.items()
                 if count > 1]
    return problems


def only_generated_problems(root: Path) -> list[str]:
    """A page that is a generated text and nothing else holds that directive and no other line: a hand-edited
    reference fails, naming the first line the hand added."""
    problems: list[str] = []
    for relative, name in ONLY_GENERATED.items():
        page = root / relative
        if not page.is_file():
            problems.append(f"{relative}: the page that is only the generated {name} is missing")
            continue
        lines = [line for line in page.read_text(encoding="utf-8").splitlines() if line.strip()]
        held = [(DIRECTIVE.match(line), line) for line in lines]
        stray = [line for match, line in held if match is None or match["kind"] != "generate"
                 or match["argument"] != name]
        if stray or len(lines) != 1:
            problems.append(f"{relative}: it is the generated {name} and nothing else, one line "
                            f"`<!-- mime-generate: {name} -->`; {'it holds ' + repr(stray[0]) if stray else 'it holds ' + str(len(lines)) + ' directives'}"
                            f". Change the parser's help, and the page follows")
    return problems


__all__ = ["BUILD_REPORT_PAGE", "GENERATORS", "ONLY_GENERATED", "REFERENCE_PAGE", "build_report_fields",
           "build_report_problems", "cli_reference", "limits", "only_generated_problems", "profiles"]
