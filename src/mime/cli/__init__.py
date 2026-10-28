"""The ``mime`` command, a thin layer over the Python API: each command is one call a script could make."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from mime._extras import VERIFY_EXTRA_MISSING
from mime.assets import AssetRef
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


def _assets_named(arguments: Sequence[str]) -> list[AssetRef]:
    """The references ``mime assets fetch`` is asked for: an argument that is a directory is an experiment folder, and
    stands for the assets its lock references; any other is an asset's name."""
    from mime import assets

    refs: list[AssetRef] = []
    for argument in arguments:
        refs += assets.folder_assets(argument) if Path(argument).is_dir() else [assets.reference(argument)]
    return refs


def _assets(command: str, arguments: Sequence[str]) -> int:
    from mime import assets

    try:
        refs = _assets_named(arguments) if arguments else list(assets.known().values())
        if command == assets.LIST:
            for ref in refs:
                print(f"{ref.name}  {assets.size_text(ref.size)}  {'cached' if assets.is_cached(ref) else 'missing'}  "
                      f"{ref.licence}")
        elif command == assets.CHECK:
            outcomes = {ref.name: assets.check(ref) for ref in refs}
            for name, outcome in outcomes.items():
                if outcome != "not cached":
                    print(f"{name}  {outcome}")
            return 1 if "corrupt" in outcomes.values() else 0
        else:
            if not refs:
                print("nothing to fetch: the folder's lock references no asset")
            for ref in refs:
                stored = assets.fetch(ref, report=lambda line: print(line, flush=True))
                print(f"{ref.name}: hashed in full, at {stored}")
    except BuildRefused as refused:
        print(refused, file=sys.stderr)
        return 2
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
    passes, 1 when one fails, 2 without the ``verify`` extra.

    ``mime assets list | fetch <name>... | fetch --all | check``: the large files MIME references by sha256, in the
    cache. ``fetch`` stores them (and refuses with 2), ``check`` re-hashes the cached ones in full (1 when one is
    corrupt)."""
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
    from mime import assets

    held = commands.add_parser(assets.GROUP, help="large files referenced by sha256 and fetched on request")
    held_commands = held.add_subparsers(dest="assets_command", required=True)
    held_commands.add_parser(assets.LIST, help="every known asset, cached or missing, with its size")
    fetch = held_commands.add_parser(assets.FETCH, help="store assets in the cache (the cache, then the primary location, then backups), then hash "
                                   "them in full")
    fetch.add_argument("names", nargs="*", metavar="name-or-folder",
                       help="an asset's name, or an experiment folder: exactly the assets its lock references")
    fetch.add_argument("--all", action="store_true", help="every known asset")
    held_commands.add_parser(assets.CHECK, help="re-hash every cached asset in full")
    arguments = parser.parse_args(argv)
    if arguments.command == "verify":
        return _verify(arguments.folder)
    if arguments.command == "registry":
        return _registry(arguments.json)
    if arguments.command == "docs":
        if arguments.docs_command == "build":
            return _docs_build(arguments.root, arguments.out)
        return _docs_check(arguments.root, arguments.base, not arguments.skip_hash_seed_builds)
    if arguments.command == assets.GROUP:
        if arguments.assets_command == assets.FETCH and bool(arguments.names) == arguments.all:
            parser.error("fetch takes asset names, or --all")
        return _assets(arguments.assets_command, arguments.names if arguments.assets_command == assets.FETCH else ())
    parser.error(f"unknown command {arguments.command!r}")


__all__ = ["main"]
