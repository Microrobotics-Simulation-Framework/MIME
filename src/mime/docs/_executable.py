"""The executable-docs check: what a page shows is either run, generated, included from a file a test holds, or said not
to be, with its reason.

A hand-written Markdown file under ``docs/`` carries four directives, each a one-line HTML comment (invisible when the
file is rendered, removed from the built page):

- ``<!-- mime-fragment: <reason> -->`` before a fenced block says that the block is a fragment: not run, with the
  reason it need not be. An unmarked block fails, and so does a fragment with no reason.
- ``<!-- mime-output: <script> -->`` before a fenced block says that the block is what ``<script>`` printed; the check
  runs the script and compares, so one changed character fails.
- ``<!-- mime-include: <file>[#region | ::Name] -->`` on a line of its own is replaced, in the built page, by the file
  (or a ``# region: name`` of it, or a Python class or function of it) as a fenced block. The code is never pasted. The
  file must be one a test names, or the lab fixture CI builds; the lab-author guide may include the fixture alone.
- ``<!-- mime-generate: <name> -->`` on a line of its own is replaced by text the build generates from the code (the
  command line's help, the profiles' definitions, the declared limits): a page that is only that directive holds no
  hand-written text for a hand to edit.

Two more checks read the same files: every ``mime ...`` command line, in a shell block or in an inline code span, is
walked down the CLI's own parser (its commands and flags must exist), and every relative link resolves, its anchor too,
into the sources or into the pages the build generates."""

from __future__ import annotations

import ast
import os
import re
import shlex
import subprocess
import sys
import tempfile
import textwrap
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from mime.cli._completion import Node, describe
from mime.docs._numbers import PAGE_DIRECTORIES

DOCS = Path("docs")
DIRECTIVE = re.compile(r"^[ \t]*<!--[ \t]*mime-(?P<kind>fragment|output|include|generate):[ \t]*(?P<argument>.*?)"
                       r"[ \t]*-->[ \t]*$")
_FENCE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
_REGION = re.compile(r"^\s*#\s*region:\s*(?P<name>\S+)\s*$")
_ENDREGION = re.compile(r"^\s*#\s*endregion(?:\s+(?P<name>\S+))?\s*$")
_SHELLS = frozenset({"sh", "bash", "shell", "console", "zsh"})
_LANGUAGES = {".py": "python", ".toml": "toml", ".yaml": "yaml", ".yml": "yaml", ".sh": "sh", ".txt": ""}
#: The one place each guide may include code from: the lab-author guide shows the fixture CI builds, and nothing else.
INCLUDES_ONLY_FROM: Mapping[str, str] = {"docs/guide/lab_author.md": "lab_fixture/"}
_SKIPPED = {"http", "https", "mailto", "ftp"}
#: The stack's one glossary, the page the guides link: the canonical page, not a mirror (``/glossary/`` and the copy
#: under ``/mime/`` answer 404 or go when the stack moves). ``mime docs check`` is offline, so it holds that a link to a
#: glossary is this string, not that the page answers.
GLOSSARY_URL = "https://microrobotica.org/glossary.html"

type Runner = Callable[[Path], str]


@dataclass(frozen=True)
class Directive:
    kind: str
    argument: str
    line: int


@dataclass(frozen=True)
class Block:
    """A fenced block: where it opens, its language, its text, and the directive on the line above it, if any."""

    line: int
    language: str
    text: str
    directive: Directive | None
    #: The line of the closing fence.
    end: int = 0


@dataclass
class Parsed:
    blocks: list[Block] = field(default_factory=list[Block])
    #: The ``include`` and ``generate`` directives on lines of their own.
    standalone: list[Directive] = field(default_factory=list[Directive])
    #: A ``fragment`` or ``output`` directive that no fenced block follows.
    orphans: list[Directive] = field(default_factory=list[Directive])
    #: The lines outside every fenced block, with their numbers.
    prose: list[tuple[int, str]] = field(default_factory=list[tuple[int, str]])
    unclosed: int | None = None


def parse(text: str) -> Parsed:
    """The fenced blocks, directives and prose lines of a Markdown text."""
    parsed = Parsed()
    pending: Directive | None = None
    opening: re.Match[str] | None = None
    opened = 0
    body: list[str] = []
    marker: Directive | None = None
    for number, line in enumerate(text.splitlines(), start=1):
        fence = _FENCE.match(line)
        if opening is not None:
            closes = fence is not None and not fence["info"].strip() and fence["fence"][0] == opening["fence"][0] \
                and len(fence["fence"]) >= len(opening["fence"])
            if closes:
                parsed.blocks.append(Block(opened, opening["info"].strip().split(" ")[0] if opening["info"].strip()
                                           else "", "\n".join(body), marker, number))
                opening, body = None, []
            else:
                body.append(line)
            continue
        if fence is not None:
            opening, opened, marker, body = fence, number, pending, []
            pending = None
            continue
        held = DIRECTIVE.match(line)
        if held is not None:
            directive = Directive(held["kind"], held["argument"], number)
            if pending is not None:
                parsed.orphans.append(pending)
            if directive.kind in ("fragment", "output"):
                pending = directive
            else:
                pending = None
                parsed.standalone.append(directive)
            continue
        if line.strip():
            if pending is not None:
                parsed.orphans.append(pending)
                pending = None
            parsed.prose.append((number, line))
    if pending is not None:
        parsed.orphans.append(pending)
    if opening is not None:
        parsed.unclosed = opened
    return parsed


def hand_written(root: Path) -> list[Path]:
    """Every hand-written Markdown file under ``docs/``, relative to ``root``: not a template (a name starting with
    ``_``) and not the build's output."""
    docs = root / DOCS
    return sorted(path.relative_to(root) for path in docs.rglob("*.md")
                  if not any(part.startswith("_") for part in path.relative_to(docs).parts))


# -- including ------------------------------------------------------------------------------------------------------

def _excerpt_symbol(source: str, name: str) -> str | None:
    tree = ast.parse(source)
    scope: list[ast.stmt] = tree.body
    found: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for part in name.split("."):
        found = next((node for node in scope if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                      and node.name == part), None)
        if found is None:
            return None
        scope = found.body
    if found is None:
        return None
    first = min([found.lineno, *(decorator.lineno for decorator in found.decorator_list)])
    lines = source.splitlines()[first - 1:found.end_lineno]
    return textwrap.dedent("\n".join(lines))


def _excerpt_region(source: str, name: str) -> str | None:
    lines = source.splitlines()
    start = next((i for i, line in enumerate(lines) if (m := _REGION.match(line)) is not None and m["name"] == name),
                 None)
    if start is None:
        return None
    kept: list[str] = []
    for line in lines[start + 1:]:
        end = _ENDREGION.match(line)
        if end is not None and end["name"] in (None, name):
            return textwrap.dedent("\n".join(kept))
        if _REGION.match(line) is None and _ENDREGION.match(line) is None:
            kept.append(line)
    return None


def split_include(argument: str) -> tuple[str, str, str]:
    """``(path, how, name)`` of an include's argument: how is ``""`` (the whole file), ``"#"`` (a region) or ``"::"``
    (a Python class or function)."""
    if "::" in argument:
        path, name = argument.split("::", 1)
        return path.strip(), "::", name.strip()
    if "#" in argument:
        path, name = argument.split("#", 1)
        return path.strip(), "#", name.strip()
    return argument.strip(), "", ""


def excerpt(root: Path, argument: str) -> tuple[str | None, str | None]:
    """The text an include names, or the problem: ``(text, None)`` or ``(None, why)``."""
    path, how, name = split_include(argument)
    target = (root / path).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError:
        return None, f"{path} is outside the checkout"
    if not target.is_file():
        return None, f"{path} is not a file of the checkout"
    source = target.read_text(encoding="utf-8")
    if how == "":
        return "\n".join(line for line in source.splitlines()
                         if _REGION.match(line) is None and _ENDREGION.match(line) is None), None
    found = _excerpt_region(source, name) if how == "#" else (
        _excerpt_symbol(source, name) if target.suffix == ".py" else None)
    if found is None:
        what = f"has no '# region: {name}'" if how == "#" else f"has no class or function {name}"
        return None, f"{path} {what}"
    return found, None


def fenced(path: str, text: str) -> str:
    language = _LANGUAGES.get(Path(path.split("#")[0].split("::")[0]).suffix, "")
    fence = "```" if "```" not in text else "````"
    return f"*From `{path.split('#')[0].split('::')[0]}`:*\n\n{fence}{language}\n{text.strip(chr(10))}\n{fence}"


def expand(text: str, root: Path, generate: Callable[[str], str]) -> tuple[str, list[str]]:
    """``text`` as the built page shows it: each include replaced by the file's text as a fenced block, each generate by
    the text the build generates, and the comments that mark a block removed. Also what stopped an expansion."""
    problems: list[str] = []
    parsed = parse(text)
    standalone = {directive.line: directive for directive in parsed.standalone}
    marked = {block.directive.line for block in parsed.blocks if block.directive is not None}
    out: list[str] = []
    in_fence = False
    for number, line in enumerate(text.splitlines(), start=1):
        if _FENCE.match(line) is not None:
            in_fence = not in_fence
        if in_fence or DIRECTIVE.match(line) is None:
            out.append(line)
            continue
        if number in marked:
            continue
        directive = standalone.get(number)
        if directive is None:
            continue
        if directive.kind == "include":
            found, why = excerpt(root, directive.argument)
            if why is not None:
                problems.append(f"line {number}: {why}")
            else:
                out.append(fenced(directive.argument, cast(str, found)))
        else:
            try:
                out.append(generate(directive.argument).rstrip("\n"))
            except KeyError:
                problems.append(f"line {number}: no generated text is named {directive.argument!r}")
    return "\n".join(out) + "\n", problems


# -- running ---------------------------------------------------------------------------------------------------------

def run_script(script: Path) -> str:
    """What ``script`` prints to standard output, run as a user would run it, in a process of its own with a home
    directory of its own (so that nothing it caches lands in the real one)."""
    with tempfile.TemporaryDirectory(prefix="mime-docs-home-") as home:
        env = {**os.environ, "PYTHONUTF8": "1", "HOME": home, "USERPROFILE": home, "XDG_CACHE_HOME": f"{home}/cache",
               "XDG_CONFIG_HOME": f"{home}/config", "XDG_DATA_HOME": f"{home}/data"}
        done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=False,
                              cwd=script.parent, env=env)
    if done.returncode != 0:
        raise RuntimeError(f"{script.name} exited {done.returncode}:\n{done.stderr[-2000:]}")
    return done.stdout


def _difference(shown: str, printed: str) -> str:
    a, b = shown.rstrip("\n").splitlines(), printed.rstrip("\n").splitlines()
    for index in range(max(len(a), len(b))):
        left, right = a[index] if index < len(a) else "<end of the block>", b[index] if index < len(b) else "<end>"
        if left != right:
            return f"first difference at its line {index + 1}: the page has {left!r}, the script printed {right!r}"
    return "the texts differ"


def refresh_outputs(root: Path, runner: Runner = run_script) -> list[str]:
    """Rewrite every output block of the hand-written files to what its script prints now; the files it changed. The
    way to bring a block back after a script or the code it runs changes on purpose: review the diff."""
    changed: list[str] = []
    for relative in hand_written(root):
        path = root / relative
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        for block in sorted(parse(text).blocks, key=lambda b: b.line, reverse=True):
            if block.directive is None or block.directive.kind != "output":
                continue
            printed = runner(root / block.directive.argument).rstrip("\n").splitlines()
            lines[block.line:block.end - 1] = printed
        updated = "\n".join(lines) + "\n"
        if updated != text:
            path.write_text(updated, encoding="utf-8", newline="\n")
            changed.append(relative.as_posix())
    return changed


# -- commands --------------------------------------------------------------------------------------------------------

_COMMAND_END = {"|", ">", ">>", "&&", "||", ";", "<", "2>&1", "&"}
_PLAIN_WORD = re.compile(r"[a-z][a-z0-9-]*$")
_INLINE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
#: A code span that may be wrapped over lines of one paragraph.
_SPAN = re.compile(r"(?<!`)`([^`]+)`(?!`)")


def _command_words(line: str) -> list[str] | None:
    """The words of ``line`` if it is a ``mime`` command line (after a ``$ `` or ``uv run [--no-sync]``), up to where
    a pipe, redirect or placeholder in brackets ends the command."""
    line = line.strip().removeprefix("$ ").strip()
    try:
        words = shlex.split(line)
    except ValueError:
        words = line.split()
    if words[:2] == ["uv", "run"]:
        words = words[2:]
        while words and words[0].startswith("--"):
            words = words[1:]
    if not words or words[0] != "mime" or (len(words) > 1 and words[1].endswith(":")):  # `mime serve: ...` is a message
        return None
    kept: list[str] = []
    for word in words:
        if word in _COMMAND_END or word.startswith("["):
            break
        kept.append(word)
    return kept


def command_problem(words: list[str], top: Node) -> str | None:
    """Why ``words`` (``mime`` then a command's words, flags and values) is not a command line of the parser, or
    ``None``: a command that is not one of its parent's, a flag that is not one of the command's."""
    node, skip = top, False
    for word in words[1:]:
        if skip:
            skip = False
            continue
        name = " ".join(("mime", *node.words))
        if word.startswith("-"):
            flag, equals, _ = word.partition("=")
            option = next((o for o in node.options if flag in o.strings), None)
            if option is None:
                return (f"{flag} is not a flag of `{name}` (it has: {', '.join(node.flags) or 'none'})")
            skip = option.takes_value and not equals
            continue
        child = next((c for c in node.children if c.words[-1] == word), None)
        if child is not None:
            node = child
        elif node.children and _PLAIN_WORD.match(word):
            return f"`{name}` has no command {word} (it has: {', '.join(c.words[-1] for c in node.children)})"
    return None


def command_lines(parsed: Parsed) -> list[tuple[int, str]]:
    """Every ``mime`` command line of a file: the lines of its shell blocks, and its inline code spans that start
    with ``mime ``."""
    found: list[tuple[int, str]] = []
    for block in parsed.blocks:
        if block.language in _SHELLS:
            logical = block.text.replace("\\\n", " ")
            found += [(block.line, line) for line in logical.splitlines() if _command_words(line) is not None]
    for number, paragraph in _paragraphs(parsed.prose):
        for span in _SPAN.finditer(paragraph):
            text = " ".join(span.group(1).split())
            if (text.startswith("mime ") or text.startswith("uv run")) and _command_words(text) is not None:
                found.append((number + paragraph.count("\n", 0, span.start()), text))
    return found


def _paragraphs(prose: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """The prose lines joined where they follow one another, each paragraph with the number of its first line, so that
    a code span wrapped over two lines is one span. A newline stays where the lines were joined, to count lines by."""
    joined: list[tuple[int, str]] = []
    for number, line in prose:
        if joined and joined[-1][0] + joined[-1][1].count("\n") + 1 == number:
            joined[-1] = (joined[-1][0], joined[-1][1] + "\n" + line)
        else:
            joined.append((number, line))
    return joined


# -- links -----------------------------------------------------------------------------------------------------------

_LINK = re.compile(r"!?\[[^\]]*\]\((?P<target><[^>]+>|[^)\s]+)(?:\s+\"[^\"]*\")?\)")
_REFERENCE = re.compile(r"^\s{0,3}\[[^\]@^][^\]]*\]:\s*(?P<target><[^>]+>|\S+)\s*$")
_HEADING = re.compile(r"^(#{1,6})\s+(?P<text>.+?)\s*#*\s*$")


def slug(heading: str) -> str:
    """The anchor GitHub gives a heading: its text without inline markup, lower-cased, punctuation dropped, spaces
    hyphens."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)
    text = re.sub(r"[`*_~]", lambda m: "_" if m.group() == "_" else "", text).lower()
    text = re.sub(r"[^\w\- ]", "", text, flags=re.UNICODE)
    return text.strip().replace(" ", "-")


def anchors(text: str) -> set[str]:
    """The anchors of a Markdown text's headings (a repeated heading takes ``-1``, ``-2``, ...)."""
    found: set[str] = set()
    seen: dict[str, int] = {}
    for _, line in parse(text).prose:
        held = _HEADING.match(line)
        if held is None:
            continue
        base = slug(held["text"])
        count = seen.get(base, 0)
        seen[base] = count + 1
        found.add(base if count == 0 else f"{base}-{count}")
    return found


def _in_pages(docs: Path, path: Path) -> bool:
    """Whether ``path`` is under one of the directories of page sources (entries, nodes, families, mechanisms, assets)."""
    try:
        return path.resolve().relative_to(docs).parts[0] in PAGE_DIRECTORIES
    except (ValueError, IndexError):
        return False


def link_problems(root: Path, files: list[Path], pages: Mapping[str, str]) -> list[str]:
    """Every relative link of the hand-written files: its path resolves, in the sources or in the pages the build
    generates (``pages``, by path under the output), and its anchor is a heading of what it names."""
    found: list[str] = []
    docs = (root / DOCS).resolve()
    for relative in files:
        text = (root / relative).read_text(encoding="utf-8")
        parsed = parse(text)
        own = anchors(text)
        for number, line in parsed.prose:
            plain = _INLINE.sub(" ", line)
            targets = [m["target"] for m in _LINK.finditer(plain)]
            if (reference := _REFERENCE.match(plain)) is not None:
                targets.append(reference["target"])
            for target in targets:
                target = target.strip("<>")
                scheme = re.match(r"([a-z][a-z0-9+.-]*):", target)
                if scheme is not None and (scheme.group(1) in _SKIPPED or len(scheme.group(1)) > 1):
                    if "glossary" in target.lower() and target != GLOSSARY_URL:
                        found.append(f"{relative}:{number}: the link {target!r} is a glossary, and the guides link "
                                     f"{GLOSSARY_URL} (mime.docs._executable.GLOSSARY_URL)")
                    continue
                path, _, anchor = target.partition("#")
                where = f"{relative}:{number}: the link {target!r}"
                if path == "":
                    if anchor and anchor not in own:
                        found.append(f"{where} names no heading of this file")
                    continue
                resolved = (root / relative).parent.joinpath(path).resolve()
                try:
                    resolved.relative_to(root.resolve())
                except ValueError:
                    found.append(f"{where} leaves the checkout")
                    continue
                if _in_pages(docs, root / relative) and _in_pages(docs, resolved):
                    continue  # a link between page sources is the docs walk's: it names what each resolves to
                if resolved.is_file():
                    named = resolved.read_text(encoding="utf-8") if resolved.suffix == ".md" else None
                    try:
                        # A page source is printed inside a page that adds headings of its own (an entry's settings): the
                        # page is what a reader's anchor lands on.
                        inside = resolved.relative_to(docs).as_posix()
                        named = pages.get(inside if inside == "index.md" else f"pages/{inside}", named)
                    except ValueError:
                        pass
                elif resolved.is_dir():
                    named = None
                else:
                    try:
                        inside = resolved.relative_to(docs).as_posix()
                    except ValueError:
                        found.append(f"{where} names no file of the checkout")
                        continue
                    built = pages.get(inside if inside == "index.md" else f"pages/{inside}")
                    if built is None:
                        found.append(f"{where} names no source, and no page the build generates")
                        continue
                    named = built
                if anchor and (named is None or anchor not in anchors(named)):
                    found.append(f"{where} names no heading of {path}")
    return found


# -- the check -------------------------------------------------------------------------------------------------------

@dataclass
class ExecutableReport:
    problems: list[str] = field(default_factory=list[str])
    summary: list[str] = field(default_factory=list[str])


_UNREAD_CODE = (
    (re.compile(r"^\s*>\s*(?:`{3,}|~{3,})"), "a fenced block inside a quotation is not read by this check; put it "
                                              "outside the quotation"),
    (re.compile(r"</?(?:pre|code)\b", re.IGNORECASE), "an HTML code block is not read by this check; use a fenced "
                                                       "block"),
)


def _unread_code(line: str) -> str | None:
    """Why a prose line holds code in a form the check does not read (a fence inside a quotation, an HTML ``pre`` or
    ``code`` element), or ``None``."""
    return next((why for pattern, why in _UNREAD_CODE if pattern.search(_INLINE.sub(" ", line))), None)


def _tests_text(root: Path) -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in sorted((root / "tests").rglob("*.py")))


def executable_report(root: Path, pages: Mapping[str, str], *, runner: Runner | None = run_script,
                      generated: frozenset[str] = frozenset()) -> ExecutableReport:
    """The executable-docs check over the hand-written files of ``root``. ``pages`` are the generated pages' text (for
    links into them); ``runner`` runs a script and returns what it printed, or is ``None`` to leave the output blocks
    unrun (they are still checked to name a script that exists); ``generated`` is the names ``mime-generate`` may use."""
    report = ExecutableReport()
    files = hand_written(root)
    tests = _tests_text(root)
    ci = root / ".github" / "workflows" / "ci.yml"
    from mime.cli import build_parser

    top = describe(build_parser())
    counts = {"blocks": 0, "fragments": 0, "outputs": 0, "includes": 0, "generated": 0, "commands": 0}
    scripts: dict[str, list[tuple[str, Block]]] = {}
    for relative in files:
        key = relative.as_posix()
        text = (root / relative).read_text(encoding="utf-8")
        parsed = parse(text)
        if parsed.unclosed is not None:
            report.problems.append(f"{key}:{parsed.unclosed}: a fenced block is never closed")
        report.problems += [f"{key}:{d.line}: the `{d.kind}` directive is followed by no fenced block"
                            for d in parsed.orphans]
        report.problems += [f"{key}:{number}: {why}" for number, line in parsed.prose
                            if (why := _unread_code(line)) is not None]
        for block in parsed.blocks:
            counts["blocks"] += 1
            where, directive = f"{key}:{block.line}", block.directive
            if directive is None:
                report.problems.append(f"{where}: a fenced block is neither included from a file a test runs, nor an "
                                       f"output, nor marked as a fragment; put `<!-- mime-fragment: <why it is not "
                                       f"run> -->` above it, or replace it with a `mime-include`")
            elif directive.kind == "fragment":
                counts["fragments"] += 1
                if not re.search(r"\w", directive.argument):
                    report.problems.append(f"{where}: a fragment is marked with no reason; say why the block is "
                                           f"not run")
            else:
                counts["outputs"] += 1
                script = directive.argument
                if not (root / script).resolve().is_relative_to((root / "examples").resolve()):
                    report.problems.append(f"{where}: the output block names {script}; a script whose output a guide "
                                           f"shows is one of examples/")
                elif not (root / script).is_file():
                    report.problems.append(f"{where}: the output block names {script}, which is no file of the "
                                           f"checkout")
                else:
                    scripts.setdefault(script, []).append((key, block))
        for directive in parsed.standalone:
            where = f"{key}:{directive.line}"
            if directive.kind == "generate":
                counts["generated"] += 1
                if directive.argument not in generated:
                    report.problems.append(f"{where}: no generated text is named {directive.argument!r} "
                                           f"(there is: {', '.join(sorted(generated)) or 'none'})")
                continue
            counts["includes"] += 1
            _, why = excerpt(root, directive.argument)
            if why is not None:
                report.problems.append(f"{where}: {why}")
                continue
            path = (root / split_include(directive.argument)[0]).resolve().relative_to(root.resolve()).as_posix()
            only = INCLUDES_ONLY_FROM.get(key)
            if only is not None and not path.startswith(only):
                report.problems.append(f"{where}: {key} includes only from {only}, the package CI builds and "
                                       f"installs; {path} is not there")
            elif only is None and not path.startswith("lab_fixture/") and path not in tests:
                report.problems.append(f"{where}: no test names {path}; an included file is one a test runs")
        counts["commands"] += len(command_lines(parsed))
        for number, line in command_lines(parsed):
            words = _command_words(line) or []
            why = command_problem(words, top)
            if why is not None:
                report.problems.append(f"{key}:{number}: `{' '.join(words)}`: {why}")
    if ci.is_file() and "./lab_fixture" not in ci.read_text(encoding="utf-8"):
        report.problems.append(f"{ci.relative_to(root)}: CI no longer builds and installs ./lab_fixture, the lab-author "
                               f"guide's worked example")
    printed = _run_all(root, scripts, runner, report) if runner is not None else {}
    for script, uses in sorted(scripts.items()):
        for key, block in uses:
            if script in printed and block.text.rstrip("\n") != printed[script].rstrip("\n"):
                report.problems.append(f"{key}:{block.line}: the output block is not what {script} prints; "
                                       f"{_difference(block.text, printed[script])}")
    report.problems += link_problems(root, files, pages)
    report.summary.append(
        f"executable docs: {len(files)} hand-written files; {counts['blocks']} fenced blocks "
        f"({counts['fragments']} fragments with a reason, {counts['outputs']} outputs"
        + (f", {len(printed)} scripts run" if runner is not None else ", scripts not run") + f"); "
        f"{counts['includes']} includes, {counts['generated']} generated texts; {counts['commands']} mime command "
        f"lines parsed against the CLI; links resolved")
    return report


def _run_all(root: Path, scripts: Mapping[str, list[tuple[str, Block]]], runner: Runner,
             report: ExecutableReport) -> dict[str, str]:
    """Run each distinct script once, together; the output of each, or a problem naming the script."""

    def one(script: str) -> tuple[str, str | None, str | None]:
        try:
            return script, runner(root / script), None
        except Exception as error:  # noqa: BLE001 - a script that fails is a docs problem, whatever it raised
            return script, None, f"{type(error).__name__}: {error}"

    printed: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for script, out, why in pool.map(one, sorted(scripts)):
            if why is not None:
                report.problems.append(f"{script}: could not be run to regenerate its output block: {why}")
            else:
                printed[script] = cast(str, out)
    return printed


if __name__ == "__main__":
    for name in refresh_outputs(Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()):
        print(f"rewrote the output blocks of {name}")


__all__ = ["DIRECTIVE", "INCLUDES_ONLY_FROM", "Block", "Directive", "ExecutableReport", "Parsed", "anchors",
           "command_lines", "command_problem", "excerpt", "executable_report", "expand", "fenced", "hand_written",
           "link_problems", "parse", "refresh_outputs", "run_script", "slug", "split_include"]
