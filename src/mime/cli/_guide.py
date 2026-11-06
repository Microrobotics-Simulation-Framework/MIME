"""How the ``mime`` command introduces itself, with no dependency: ``mime --help`` grouped by what one is doing, one worked
example under every command's ``--help``, what a bare ``mime`` says (and says more of in a saved experiment's folder or a
run's), and "did you mean" for a mistyped command or flag, on every Python MIME supports (3.14's argparse suggests a
command when asked to; 3.12's does not, and neither suggests a flag)."""

from __future__ import annotations

import argparse
import difflib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast, override

from mime.cli._shell import quoted

#: The commands, by what one is doing. Every command is in exactly one group (a test holds it).
GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Run", ("run", "serve", "verify", "pending", "show")),
    ("Share", ("pack", "card", "pull", "push", "credentials")),
    ("Environment", ("env", "modal", "cache", "assets", "visuals", "completion")),
    ("Reference", ("registry", "docs")),
)

#: One worked example for every command, by its words after ``mime``. Every leaf command has one (a test holds it).
EXAMPLES: Mapping[tuple[str, ...], str] = {
    ("run",): "mime run my_experiment --duration 10s --out my_run",
    ("serve",): "mime serve my_experiment --start",
    ("verify",): "mime verify my_experiment",
    ("pending",): "mime pending my_experiment",
    ("show",): "mime show my_experiment --graph",
    ("pack",): "mime pack my_experiment --out my_experiment.mimebundle",
    ("card",): "mime card my_experiment --notice",
    ("pull",): "mime pull 10.5281/zenodo.1234567 --into my_experiment",
    ("push",): "mime push my_experiment --to zenodo-sandbox",
    ("credentials",): "mime credentials status",
    ("credentials", "set"): "mime credentials set zenodo-sandbox",
    ("credentials", "status"): "mime credentials status zenodo",
    ("credentials", "remove"): "mime credentials remove zenodo-sandbox",
    ("env",): "mime env check my_experiment --variant cpu",
    ("env", "check"): "mime env check my_experiment --variant cpu",
    ("modal",): "mime modal my_experiment --gpu A100-40GB",
    ("cache",): "mime cache clear",
    ("cache", "clear"): "mime cache clear",
    ("assets",): "mime assets fetch my_experiment",
    ("assets", "list"): "mime assets list",
    ("assets", "fetch"): "mime assets fetch my_experiment",
    ("assets", "check"): "mime assets check",
    ("assets", "sources"): "mime assets sources my_experiment",
    ("assets", "audit"): "mime assets audit dist/mime_engine-0.3.0-py3-none-any.whl",
    ("assets", "licences"): "mime assets licences --revoke LicenseRef-Annin-Robotics-NonCommercial",
    ("visuals",): "mime visuals list",
    ("visuals", "list"): "mime visuals list",
    ("visuals", "fetch"): "mime visuals fetch examples/descriptions/ar4/",
    ("visuals", "view"): "mime visuals view my_run",
    ("visuals", "web"): "mime visuals web my_run --open",
    ("registry",): "mime registry --json",
    ("docs",): "mime docs check --base origin/release/v0.3.0",
    ("docs", "check"): "mime docs check --base origin/release/v0.3.0",
    ("docs", "build"): "mime docs build --out docs_pages",
    ("completion",): "mime completion bash > ~/.local/share/bash-completion/completions/mime",
}

#: A help text starting with this is laid out as written, one example's line per line.
_AS_WRITTEN = ("example", "install")


class _Formatter(argparse.HelpFormatter):
    """argparse's, except that the example under a command keeps its lines."""

    @override
    def _fill_text(self, text: str, width: int, indent: str) -> str:
        if text.startswith(_AS_WRITTEN):
            return "\n".join(indent + line for line in text.splitlines())
        return super()._fill_text(text, width, indent)


class MimeParser(argparse.ArgumentParser):
    """``argparse.ArgumentParser`` with ``mime``'s help for the top (grouped), an example under each command, and a
    "did you mean" under an error that has one."""

    #: The words of the command line being parsed and the top parser, set by :func:`main` before it parses.
    words: Sequence[str] = ()
    top: MimeParser | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("formatter_class", _Formatter)
        super().__init__(*args, **kwargs)

    @override
    def format_help(self) -> str:
        if self is not MimeParser.top:
            return super().format_help()
        formatter = self._get_formatter()
        formatter.add_usage("%(prog)s [-h] [-V] [--traceback] [--plain] <command> [<args>]", [], [])
        formatter.add_text(self.description)
        held = _subparsers_of(self)
        listed = {action.dest: action for action in (held._get_subactions() if held is not None else [])}
        for title, names in GROUPS:
            formatter.start_section(title)
            for name in names:
                formatter.add_argument(listed[name])
            formatter.end_section()
        formatter.start_section("Options")
        formatter.add_arguments([action for action in self._actions if action.option_strings])
        formatter.end_section()
        formatter.add_text(self.epilog)
        return formatter.format_help()

    @override
    def error(self, message: str) -> Any:
        hints = hints_for(message, self, MimeParser.words)
        super().error(message + "".join(f"\n  {hint}" for hint in hints))


def _subparsers_of(parser: argparse.ArgumentParser) -> argparse._SubParsersAction[argparse.ArgumentParser] | None:  # pyright: ignore[reportPrivateUsage]
    """The action holding ``parser``'s commands, or ``None`` for a command that has none."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):  # pyright: ignore[reportPrivateUsage]
            return cast("argparse._SubParsersAction[argparse.ArgumentParser]", action)  # pyright: ignore[reportPrivateUsage]
    return None


def _children(parser: argparse.ArgumentParser) -> Mapping[str, argparse.ArgumentParser]:
    held = _subparsers_of(parser)
    return held.choices if held is not None else {}


def attach_examples(parser: argparse.ArgumentParser, path: tuple[str, ...] = ()) -> None:
    """Put each command's example under its ``--help``, walking the commands from ``parser``."""
    for name, child in _children(parser).items():
        here = (*path, name)
        if here in EXAMPLES:
            held = f"{child.epilog}\n\n" if child.epilog else ""
            child.epilog = f"{held}example:\n  {EXAMPLES[here]}"
        attach_examples(child, here)


def leaf_paths(parser: argparse.ArgumentParser, path: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    """Every command's words, as the example table is keyed (a command with commands under it, and each of them)."""
    found: list[tuple[str, ...]] = []
    for name, child in _children(parser).items():
        found += [(*path, name), *leaf_paths(child, (*path, name))]
    return found


def _close(word: str, choices: Sequence[str]) -> str | None:
    found = difflib.get_close_matches(word, choices, n=1, cutoff=0.6)
    return found[0] if found else None


def hints_for(message: str, parser: argparse.ArgumentParser, words: Sequence[str]) -> list[str]:
    """The "did you mean" lines for an error: the command nearest a mistyped one, and the flag nearest each unknown flag
    the line has. It never repeats a word the user typed, only the word it suggests, so a token typed where it does not
    belong is not written to a screen or a log by it."""
    hints: list[str] = []
    wrong = re.search(r"invalid choice: '([^']*)'", message)
    if wrong is not None and "maybe you meant" not in message:
        choices = list(_children(parser))
        near = _close(wrong.group(1), choices)
        if near is not None:
            hints.append(f"did you mean {near}?")
    top = MimeParser.top
    if top is None:
        return hints
    here: argparse.ArgumentParser = top
    for word in words:
        child = _children(here).get(word)
        if child is not None:
            here = child
    known = sorted({option for each in (here, top) for option in each._option_string_actions
                    if option.startswith("--")})
    for word in words:
        flag = word.split("=", 1)[0]
        if flag.startswith("--") and flag not in known:
            near = _close(flag, known)
            if near is not None:
                hints.append(f"did you mean {near}?")
    return hints


def bare(folder: Path) -> str:
    """What a bare ``mime`` says: what one can do, and in a saved experiment's folder or a run's, the commands for it,
    with its name."""
    saved_name = _saved_name(folder)
    run_name = _run_name(folder) if saved_name is None else None
    lines = ["mime: run, check and share microrobot experiments.", ""]
    if saved_name is not None:
        lines += [f"This folder is the saved experiment {saved_name}. For it:",
                  f"  mime run . --duration 10s --out {quoted(f'../{saved_name}_run')}"
                  "      run 10 s of simulated time and write the result",
                  "  mime verify .                         rerun its reference steps and compare",
                  "  mime pending .                        what it leaves pending, with the lines to paste",
                  "  mime serve . --start                  a live session, for a viewer to connect to",
                  "  mime pack .                           one bundle file, to share"]
    elif run_name is not None:
        lines += [f"This folder is a run of {run_name}. For it:",
                  "  mime verify experiment                its saved form reproduces the run: check it",
                  "  mime serve experiment --start         the saved form as a live session",
                  "  python -c \"from mime.results import RunResult; print(RunResult.read('.').text())\"",
                  "                                        read the result",
                  "  outputs.json and report.json          the result and the whole report, readable without MIME"]
    else:
        lines += ["With a saved experiment (a folder `Experiment.save` wrote):",
                  "  mime run <folder> --duration 10s --out <dir>    run it and write the result",
                  "  mime verify <folder>                            check it still reproduces",
                  "  mime pending <folder>                           what it leaves pending",
                  "To get one that someone shared:",
                  "  mime pull <doi | address | file>",
                  "This folder is neither a saved experiment nor a run; `cd` into one for the commands that fit it."]
    lines += ["", "mime --help lists every command by what it is for; mime <command> --help shows one, with an example."]
    return "\n".join(lines)


def _saved_name(folder: Path) -> str | None:
    """The experiment's name where ``folder`` is a saved experiment (its document and its lock), else ``None``."""
    from mime.saved import DOCUMENT_FILE, LOCK_FILE

    document = folder / DOCUMENT_FILE
    if not (document.is_file() and (folder / LOCK_FILE).is_file()):
        return None
    try:
        found = re.search(r"^experiment:\s*(.+?)\s*$", document.read_text(encoding="utf-8"), re.MULTILINE)
    except (OSError, UnicodeDecodeError):
        found = None
    return found.group(1).strip("'\"") if found is not None else "(unnamed)"


def _run_name(folder: Path) -> str | None:
    """The experiment's name where ``folder`` is a run folder (its manifest and its outputs), else ``None``."""
    if not ((folder / "outputs.json").is_file() and (folder / "outputs.npz").is_file()):
        return None
    try:
        name = json.loads((folder / "outputs.json").read_text(encoding="utf-8")).get("experiment")
    except (OSError, ValueError):
        return "an experiment"
    return str(name) if name else "an experiment"
