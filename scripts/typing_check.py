#!/usr/bin/env python
"""The typing gate: pyright over ``src/mime``, strict, at zero errors, and a run that can be trusted.

Runs ``pyright --outputjson`` with the repository's ``pyrightconfig.json`` (or reads a stored run with ``--json``),
prints the totals, the counts per rule and the files with errors, and exits:

0
    pyright completed and reported no errors.
1
    pyright completed and reported errors: the gate fails.
2
    The run cannot be trusted, so no count is given: pyright is missing; it is not the release ``pyproject.toml``
    pins (every release changes diagnostics); it exited with a code other than 0 or 1 (3: the configuration could
    not be parsed, and it analysed the whole tree with defaults); it wrote no JSON; it analysed no file (a missing
    ``include`` path is reported on stderr only, with exit 0); or a core import did not resolve (a wrong interpreter
    turns real errors into unknowns and lowers the count).

Modelled on MADDENING's ``scripts/typing_baseline.py``, in one tier, since every package of MIME 0.3 is new code
held at zero. Arguments after ``--`` go to pyright.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = "src/mime/"

EXIT_OK, EXIT_ERRORS, EXIT_UNTRUSTED = 0, 1, 2

#: What every environment pyright runs in resolves: MIME's base dependencies. One of them unresolved means pyright
#: read another interpreter.
CORE_IMPORTS = ("jax", "numpy", "yaml", "maddening", "typing_extensions")

_MISSING_IMPORT = re.compile(r'Import "([^"]+)"')


class Untrusted(Exception):
    """The run cannot be trusted; no count is reported."""


def pinned_release(pyproject: Path = REPO_ROOT / "pyproject.toml") -> str:
    """The pyright release the dev group pins exactly, which the gate's numbers are tied to."""
    groups = tomllib.loads(pyproject.read_text())["dependency-groups"]
    for requirement in groups.get("dev", []):
        found = re.fullmatch(r"pyright==([0-9.]+)", requirement) if isinstance(requirement, str) else None
        if found:
            return found.group(1)
    raise Untrusted(f"{pyproject} pins no pyright==<release> in its dev group")


@dataclass
class Summary:
    version: str = ""
    files_analysed: int = 0
    errors: int = 0
    warnings: int = 0
    by_rule: Counter[str] = field(default_factory=Counter[str])
    by_file: Counter[str] = field(default_factory=Counter[str])
    missing_imports: set[str] = field(default_factory=set[str])


def summarise(report: dict[str, object]) -> Summary:
    summary = report.get("summary")
    if not isinstance(summary, dict) or "filesAnalyzed" not in summary:
        raise Untrusted("pyright's output has no summary: the run did not complete")
    found = Summary(version=str(report.get("version", "")), files_analysed=int(summary["filesAnalyzed"]))
    diagnostics = report.get("generalDiagnostics", [])
    for diagnostic in diagnostics if isinstance(diagnostics, list) else []:
        severity, rule = diagnostic.get("severity"), diagnostic.get("rule") or "(no rule)"
        path = str(diagnostic.get("file", "?"))
        path = path.split(PACKAGE_ROOT, 1)[-1] if PACKAGE_ROOT in path else path
        if rule == "reportMissingImports":
            missing = _MISSING_IMPORT.search(diagnostic.get("message", ""))
            found.missing_imports.add(missing.group(1) if missing else "?")
        if severity == "error":
            found.errors += 1
            found.by_rule[rule] += 1
            found.by_file[path] += 1
        elif severity == "warning":
            found.warnings += 1
    return found


def check_trust(found: Summary, release: str, min_files: int) -> None:
    if found.version != release:
        raise Untrusted(f"pyright {found.version or '(no version)'} ran, and the gate is tied to {release}, the "
                        f"release pyproject.toml pins: every release changes diagnostics")
    if found.files_analysed < min_files:
        raise Untrusted(f"pyright analysed {found.files_analysed} files (at least {min_files} expected): check "
                        f"`include` in pyrightconfig.json and the checkout")
    unresolved = sorted(name for name in found.missing_imports if name.split(".")[0] in CORE_IMPORTS)
    if unresolved:
        raise Untrusted(f"pyright could not resolve the core imports {', '.join(unresolved)}: it read another "
                        f"interpreter than the one MIME is installed in, so its count means nothing")


def run_pyright(command: str, extra: list[str]) -> dict[str, object]:
    try:
        done = subprocess.run([*command.split(), "--outputjson", *extra], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=False)
    except OSError as error:
        raise Untrusted(f"cannot run pyright as `{command}` ({error}): install the dev group (uv sync)") from error
    if done.stderr:
        sys.stderr.write(done.stderr if done.stderr.endswith("\n") else done.stderr + "\n")
    try:
        report = json.loads(done.stdout)
    except json.JSONDecodeError:
        sys.stderr.write(done.stdout)
        raise Untrusted(f"pyright wrote no JSON (exit {done.returncode})") from None
    if done.returncode not in (0, 1):
        raise Untrusted(f"pyright exited {done.returncode} (3: pyrightconfig.json could not be parsed, and it "
                        f"analysed the whole tree with its defaults)")
    if not isinstance(report, dict):
        raise Untrusted("pyright's JSON is not an object")
    return report


def render(found: Summary, markdown: bool) -> str:
    title = f"pyright {found.version}, strict over src/mime"
    lines = [f"### {title}" if markdown else title, "",
             f"files analysed: {found.files_analysed}; errors: {found.errors}; warnings: {found.warnings}"]
    for heading, counts in (("errors by rule", found.by_rule), ("errors by file", found.by_file)):
        if counts:
            lines += ["", f"#### {heading}" if markdown else heading]
            lines += [f"- `{name}`: {count}" if markdown else f"  {count:>4}  {name}"
                      for name, count in counts.most_common()]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--json", type=Path, help="summarise this stored `pyright --outputjson` run")
    parser.add_argument("--pyright", default=os.environ.get("PYRIGHT", "pyright"), help="the pyright command")
    parser.add_argument("--markdown", action="store_true", help="Markdown, for a CI step summary")
    parser.add_argument("--min-files", type=int, default=40,
                        help="fewer analysed files means the run did not see the package (default %(default)s)")
    parser.add_argument("pyright_args", nargs="*", help="arguments for pyright, after `--`")
    arguments = parser.parse_args(argv)
    try:
        release = pinned_release()
        report = (json.loads(arguments.json.read_text()) if arguments.json
                  else run_pyright(arguments.pyright, arguments.pyright_args))
        found = summarise(report)
        check_trust(found, release, arguments.min_files)
    except Untrusted as error:
        sys.stderr.write(f"typing_check: the run cannot be trusted: {error}\n")
        return EXIT_UNTRUSTED
    sys.stdout.write(render(found, arguments.markdown))
    return EXIT_ERRORS if found.errors else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
