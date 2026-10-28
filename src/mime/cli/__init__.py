"""The ``mime`` command, a thin layer over the Python API: each command is one call a script could make."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from mime._extras import VERIFY_EXTRA_MISSING
from mime.build import BuildRefused
from mime.experiment import Experiment


def _verify(folder: Path) -> int:
    try:
        experiment = Experiment.load(folder)
        report = experiment.build().verify()
    except BuildRefused as refused:
        print(refused, file=sys.stderr)
        return 2
    if experiment.load_report is not None:
        print(experiment.load_report.text())
    print(report.text())
    return 0 if report.reproduced else 1


def _registry(as_json: bool) -> int:
    from mime.docs._export import export_json

    exported = export_json()
    if as_json:
        print(exported)
        return 0
    found = json.loads(exported)
    for entry in found["entries"]:
        print(f"{entry['kind']}  {entry['name']}  {entry['version']}")
    for name, reason in found["not_loaded"].items():
        print(f"not loaded: {name} ({reason})")
    return 0


def _docs_check(root: Path, base: str | None, hash_seeds: bool) -> int:
    try:
        from mime.docs._check import run_check
    except ImportError as error:
        if str(error) != VERIFY_EXTRA_MISSING:
            raise
        print(f"mime docs check: {error}", file=sys.stderr)
        return 2
    return run_check(root, base, hash_seeds=hash_seeds)


def _docs_build(root: Path, out: Path | None) -> int:
    try:
        from mime.docs._build import run_build
    except ImportError as error:
        if str(error) != VERIFY_EXTRA_MISSING:
            raise
        print(f"mime docs build: {error}", file=sys.stderr)
        return 2
    return run_build(root, out)


def main(argv: Sequence[str] | None = None) -> int:
    """``mime verify <folder>``: load a saved experiment, rebuild it, check it against its lock, and rerun its
    reference steps. Exits 0 when reproduced, 1 when an output moved, 2 when the load or build is refused.

    ``mime registry [--json]``: every registered entry, MIME's and each installed namespace's, in a fresh process.

    ``mime docs check [--root <checkout>] [--base <revision>]``: the documentation's sources and their checks, in a
    checkout of MIME; with ``--base``, the README start list against that revision's too. Exits 0 when every check
    passes, 1 when one fails, 2 without the ``verify`` extra."""
    parser = argparse.ArgumentParser(prog="mime", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify", help="load, rebuild, check the hash, and rerun against the reference "
                                                "outputs")
    verify.add_argument("folder", type=Path, help="a folder written by Experiment.save")
    registry = commands.add_parser("registry", help="every registered entry and what it declares")
    registry.add_argument("--json", action="store_true", help="the whole export, as JSON")
    docs = commands.add_parser("docs", help="the documentation's sources")
    docs_commands = docs.add_subparsers(dest="docs_command", required=True)
    check = docs_commands.add_parser("check", help="every check of the documentation's sources")
    check.add_argument("--root", type=Path, default=Path.cwd(), help="the checkout to check (default: here)")
    check.add_argument("--base", metavar="REVISION", help="fail on a README start-list line this revision's list "
                                                         "does not hold (CI passes origin/v0.3)")
    check.add_argument("--skip-hash-seed-builds", action="store_true",
                       help="do not build the pages in fresh processes under eight hash seeds (the tests of "
                            "the other checks pass it to stay fast; CI never does)")
    build = docs_commands.add_parser("build", help="generate the pages (never committed) from the registry and the "
                                                   "hand-written sources")
    build.add_argument("--root", type=Path, default=Path.cwd(), help="the checkout to build from (default: here)")
    build.add_argument("--out", type=Path, help="the directory to write index.md and pages/ to (default: "
                                                "<root>/docs/_build)")
    arguments = parser.parse_args(argv)
    if arguments.command == "verify":
        return _verify(arguments.folder)
    if arguments.command == "registry":
        return _registry(arguments.json)
    if arguments.command == "docs":
        if arguments.docs_command == "build":
            return _docs_build(arguments.root, arguments.out)
        return _docs_check(arguments.root, arguments.base, not arguments.skip_hash_seed_builds)
    parser.error(f"unknown command {arguments.command!r}")


__all__ = ["main"]
