"""``mime completion bash|zsh|fish``: a completion script written out from ``build_parser()``, so that every command, flag
and choice the parser holds is completed and none is listed by hand.

The script is static: pressing Tab runs shell code only, never Python, since importing MIME imports JAX, which takes
seconds. The parser is read once, when the script is written, into a tree of :class:`Node`; each shell's script is that
tree in the shell's own words. A flag whose value is a path completes as files, one with choices as its choices, and
any other as nothing."""

from __future__ import annotations

import argparse
import re
import shlex
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from mime.cli import _guide

SHELLS = ("bash", "zsh", "fish")

#: Where each shell reads a completion script from, as ``mime completion --help`` says it (``~`` is the home folder).
INSTALL: Mapping[str, str] = {
    "bash": "mime completion bash > ~/.local/share/bash-completion/completions/mime",
    "zsh": "mime completion zsh > ~/.zfunc/_mime",
    "fish": "mime completion fish > ~/.config/fish/completions/mime.fish",
}

#: Positional arguments that take a file or a folder though the parser does not type them as a path: a bundle file for
#: ``pull`` and a saved experiment's folder for ``assets fetch``, each beside what else it takes. A test holds that each
#: names an argument the parser has.
ALSO_FILES: frozenset[tuple[tuple[str, ...], str]] = frozenset({(("pull",), "source"), (("assets", "fetch"), "names")})

#: What else ``mime completion --help`` says of where a script goes.
WHERE: Mapping[str, str] = {
    "bash": "",
    "zsh": "   (a file named _mime in a folder on fpath, before compinit runs)",
    "fish": "",
}

_DESCRIPTION_LENGTH = 70


@dataclass(frozen=True)
class Option:
    """One flag, with every spelling of it."""

    strings: tuple[str, ...]
    description: str
    takes_value: bool
    choices: tuple[str, ...]
    files: bool


@dataclass(frozen=True)
class Node:
    """One command (the top, with empty ``words``, included): its flags (the top's too, since ``main`` takes them anywhere
    on the line), the commands under it, and what its positional arguments complete as."""

    words: tuple[str, ...]
    description: str
    options: tuple[Option, ...]
    children: tuple[Node, ...]
    choices: tuple[str, ...]
    files: bool

    @property
    def flags(self) -> tuple[str, ...]:
        return tuple(string for option in self.options for string in option.strings)


def describe(parser: argparse.ArgumentParser) -> Node:
    """The tree of ``parser``'s commands, flags and choices."""
    top = _options_of(parser)
    return _node(parser, (), "", top)


def _node(parser: argparse.ArgumentParser, words: tuple[str, ...], description: str, top: tuple[Option, ...]) -> Node:
    own = _options_of(parser)
    seen = {string for option in own for string in option.strings}
    inherited = tuple(option for option in top if not seen & set(option.strings))
    children: list[Node] = []
    held = _guide._subparsers_of(parser)  # pyright: ignore[reportPrivateUsage]
    if held is not None:
        described = {action.dest: action.help or "" for action in held._get_subactions()}
        children = [_node(child, (*words, name), _gist(described.get(name, "")), top)
                    for name, child in held.choices.items()]
    choices: list[str] = []
    files = False
    for action in parser._actions:
        if action.option_strings or isinstance(action, argparse._SubParsersAction):  # pyright: ignore[reportPrivateUsage]
            continue
        choices += [str(choice) for choice in action.choices or ()]
        files = files or action.type is Path or (words, action.dest) in ALSO_FILES
    return Node(words, description, (*own, *inherited), tuple(children), tuple(dict.fromkeys(choices)), files)


def _options_of(parser: argparse.ArgumentParser) -> tuple[Option, ...]:
    found: list[Option] = []
    for action in parser._actions:
        if not action.option_strings or action.help == argparse.SUPPRESS:
            continue
        found.append(Option(tuple(action.option_strings), _gist(action.help or ""), action.nargs != 0,
                            tuple(str(choice) for choice in action.choices or ()), action.type is Path))
    return tuple(found)


def _gist(help_text: str) -> str:
    """The first clause of a help text, on one line of plain ASCII and no longer than a menu shows."""
    text = " ".join(help_text.replace("%%", "%").split())
    text = re.split(r"[.;:(]( |$)", text, maxsplit=1)[0].strip().rstrip(",")
    if len(text) > _DESCRIPTION_LENGTH:
        text = text[:_DESCRIPTION_LENGTH].rsplit(" ", 1)[0]
    return text.encode("ascii", "replace").decode("ascii")


def _walk(node: Node) -> Iterable[Node]:
    yield node
    for child in node.children:
        yield from _walk(child)


def _name(node: Node) -> str:
    return " ".join(("mime", *node.words))


def script(shell: str, parser: argparse.ArgumentParser) -> str:
    """The completion script for ``shell`` (one of :data:`SHELLS`), over ``parser``'s commands."""
    writers: Mapping[str, Callable[[Node], str]] = {"bash": _bash, "zsh": _zsh, "fish": _fish}
    return writers[shell](describe(parser))


# ---------------------------------------------------------------------------------------------------------------- bash

def _bash_words(words: Iterable[str]) -> str:
    return " ".join(words)


def _bash(top: Node) -> str:
    nodes = list(_walk(top))
    lines = [
        "# bash completion for mime, written by `mime completion bash` from the command line's own parser.",
        f"# Install: {INSTALL['bash']}",
        "# Tab runs this function and nothing else: no Python starts.",
        "_mime() {",
        '    local cur="${COMP_WORDS[COMP_CWORD]}" prev="" flag="" node="mime" word i',
        '    (( COMP_CWORD > 0 )) && prev="${COMP_WORDS[COMP_CWORD-1]}"',
        '    COMPREPLY=()',
        '    if [[ $cur == "=" ]]; then',
        '        cur=""; flag="$prev"',
        '    elif [[ $prev == "=" ]] && (( COMP_CWORD > 1 )); then',
        '        flag="${COMP_WORDS[COMP_CWORD-2]}"',
        '    else',
        '        flag="$prev"',
        '    fi',
        '    for (( i = 1; i < COMP_CWORD; i++ )); do',
        '        word="${COMP_WORDS[i]}"',
        '        case "$node|$word" in',
    ]
    for node in nodes:
        for child in node.children:
            lines.append(f'            "{_name(node)}|{child.words[-1]}") node="{_name(child)}" ;;')
    lines += [
        '        esac',
        '    done',
        '    case "$node|$flag" in',
    ]
    for node in nodes:
        for option in node.options:
            if not option.takes_value:
                continue
            for string in option.strings:
                if option.choices:
                    action = f'COMPREPLY=($(compgen -W "{_bash_words(option.choices)}" -- "$cur")); return 0'
                elif option.files:
                    action = '_mime_files "$cur"; return 0'
                else:
                    action = "return 0"
                lines.append(f'        "{_name(node)}|{string}") {action} ;;')
    lines += [
        '    esac',
        '    case "$node" in',
    ]
    for node in nodes:
        positional: list[str] = []
        if node.children:
            positional.append(f'COMPREPLY=($(compgen -W "{_bash_words(child.words[-1] for child in node.children)}" '
                              '-- "$cur"))')
        if node.choices:
            positional.append(f'COMPREPLY+=($(compgen -W "{_bash_words(node.choices)}" -- "$cur"))')
        if node.files:
            positional.append('_mime_files "$cur"')
        lines += [
            f'        "{_name(node)}")',
            "            if [[ $cur == -* ]]; then",
            f'                COMPREPLY=($(compgen -W "{_bash_words(node.flags)}" -- "$cur"))',
            "            else",
            *[f"                {line}" for line in positional or [":"]],
            "            fi ;;",
        ]
    lines += [
        "    esac",
        "    return 0",
        "}",
        "",
        "# Files and folders by name, one to a line so that a name with a space stays one, with a slash after a folder.",
        "_mime_files() {",
        "    local IFS=$'\\n'",
        "    COMPREPLY+=($(compgen -f -- \"$1\"))",
        "    compopt -o filenames 2>/dev/null",
        "}",
        "",
        "complete -F _mime mime",
        "",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------------------- zsh

def _single(text: str) -> str:
    """``text`` as one single-quoted word of zsh or bash."""
    return shlex.quote(text) if text else "''"


def _zsh_item(name: str, description: str) -> str:
    return _single(f"{name.replace(':', chr(92) + ':')}:{description}" if description else name.replace(":", "\\:"))


def _zsh(top: Node) -> str:
    nodes = list(_walk(top))
    lines = [
        "#compdef mime",
        "# zsh completion for mime, written by `mime completion zsh` from the command line's own parser.",
        f"# Install: {INSTALL['zsh']}",
        "# Tab runs this function and nothing else: no Python starts.",
        "_mime() {",
        "    local node='mime' word flag cur=$words[CURRENT] prev i",
        "    local -a values",
        "    (( CURRENT > 1 )) && prev=$words[CURRENT-1]",
        "    for (( i = 2; i < CURRENT; i++ )); do",
        "        word=$words[i]",
        '        case "$node|$word" in',
    ]
    for node in nodes:
        for child in node.children:
            lines.append(f'            "{_name(node)}|{child.words[-1]}") node="{_name(child)}" ;;')
    lines += [
        "        esac",
        "    done",
        "    flag=$prev",
        "    if [[ $cur == --*=* ]]; then",
        "        flag=${cur%%=*}",
        "        compset -P '*='",
        "    fi",
        '    case "$node|$flag" in',
    ]
    for node in nodes:
        for option in node.options:
            if not option.takes_value:
                continue
            for string in option.strings:
                if option.choices:
                    action = f"compadd -- {_bash_words(option.choices)}"
                elif option.files:
                    action = "_files"
                else:
                    action = ":"
                lines.append(f'        "{_name(node)}|{string}") {action}; return ;;')
    lines += [
        "    esac",
        '    case "$node" in',
    ]
    for node in nodes:
        lines.append(f'        "{_name(node)}")')
        lines.append("            if [[ $cur == -* ]]; then")
        lines.append(f"                values=({' '.join(_zsh_flag_item(node, string) for string in node.flags)})")
        lines.append("                _describe -t options 'option' values")
        lines.append("            else")
        action_lines: list[str] = []
        if node.children:
            items = " ".join(_zsh_item(child.words[-1], child.description) for child in node.children)
            action_lines += [f"values=({items})", "_describe -t commands 'command' values"]
        if node.choices:
            action_lines.append(f"compadd -- {_bash_words(node.choices)}")
        if node.files:
            action_lines.append("_files")
        lines += [f"                {action}" for action in action_lines or [":"]]
        lines += ["            fi ;;"]
    lines += [
        "    esac",
        "}",
        "",
        "# Autoloaded from fpath, this file's body runs the function; sourced, it registers it.",
        'if [[ $funcstack[1] == _mime ]]; then',
        '    _mime "$@"',
        'else',
        '    compdef _mime mime',
        'fi',
        "",
    ]
    return "\n".join(lines)


def _zsh_flag_item(node: Node, string: str) -> str:
    option = next(each for each in node.options if string in each.strings)
    return _zsh_item(string, option.description)


# ---------------------------------------------------------------------------------------------------------------- fish

def _fish_quoted(text: str) -> str:
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _fish(top: Node) -> str:
    nodes = list(_walk(top))
    lines = [
        "# fish completion for mime, written by `mime completion fish` from the command line's own parser.",
        f"# Install: {INSTALL['fish']}",
        "# Tab runs these functions and nothing else: no Python starts.",
        "",
        "# The command the cursor is in: the words before it, walked through the commands.",
        "function __mime_node",
        "    set -l node mime",
        "    set -l tokens (commandline -opc)",
        "    set -e tokens[1]",
        "    for word in $tokens",
        '        switch "$node|$word"',
    ]
    for node in nodes:
        for child in node.children:
            lines += [f"            case {_fish_quoted(f'{_name(node)}|{child.words[-1]}')}",
                      f"                set node {_fish_quoted(_name(child))}"]
    lines += [
        "        end",
        "    end",
        "    echo $node",
        "end",
        "",
        "function __mime_at",
        "    set -l here (__mime_node)",
        '    test "$here" = "$argv[1]"',
        "end",
        "",
        "complete -c mime -f",
    ]
    top_node = nodes[0]
    for option in top_node.options:
        lines.append(_fish_option(None, option))
    for node in nodes:
        condition = f"-n '__mime_at \"{_name(node)}\"'"
        for child in node.children:
            lines.append(f"complete -c mime {condition} -a {_fish_quoted(child.words[-1])} -d {_fish_quoted(child.description)}")
        if node.choices:
            lines.append(f"complete -c mime {condition} -a {_fish_quoted(_bash_words(node.choices))}")
        if node.files:
            lines.append(f"complete -c mime {condition} -F")
        for option in node.options:
            if option not in top_node.options:
                lines.append(_fish_option(condition, option))
    lines.append("")
    return "\n".join(lines)


def _fish_option(condition: str | None, option: Option) -> str:
    parts = ["complete -c mime"]
    if condition is not None:
        parts.append(condition)
    for string in option.strings:
        if string.startswith("--"):
            parts.append(f"-l {string[2:]}")
        else:
            parts.append(f"-s {string[1:]}")
    if option.takes_value:
        if option.choices:
            parts += ["-x", f"-a {_fish_quoted(_bash_words(option.choices))}"]
        elif option.files:
            parts += ["-r", "-F"]
        else:
            parts += ["-x"]
    if option.description:
        parts.append(f"-d {_fish_quoted(option.description)}")
    return " ".join(parts)
