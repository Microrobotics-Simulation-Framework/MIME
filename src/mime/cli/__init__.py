"""The ``mime`` command, a thin layer over the Python API: each command is one call a script could make."""

from __future__ import annotations

import argparse
import getpass
import hashlib
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


def _pack(folder: Path, out: Path | None) -> int:
    from mime import assets, saved

    try:
        path = saved.pack(folder, out=out)
    except BuildRefused as refused:
        print(refused, file=sys.stderr)
        return 2
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    print(f"packed {folder} into {path} ({assets.size_text(path.stat().st_size)})")
    print(f"sha256 {digest}")
    return 0


def _ask(question: str) -> bool:
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _pull(source: str, into: Path | None, no_assets: bool, yes: bool) -> int:
    from mime import saved

    try:
        saved.pull(source, into=into, fetch_assets=not no_assets, yes=yes,
                   confirm=_ask if sys.stdin.isatty() else None, say=lambda line: print(line, flush=True))
    except BuildRefused as refused:
        print(refused, file=sys.stderr)
        return 2
    return 0


def _auth(command: str, service: str | None) -> int:
    from mime import auth

    try:
        if command == "status":
            for each in [service] if service else list(auth.SERVICES):
                print(auth.describe(each))
        elif service is None:
            raise AssertionError("argparse requires a service for login and logout")
        elif command == "login":
            try:
                value = getpass.getpass(f"{service} access token (not shown): ").strip()
            except EOFError:
                value = ""
            print(f"saved the {service} token in {auth.login(service, value)}")
        else:
            print(f"deleted the {service} token from the credentials file" if auth.logout(service)
                  else f"the credentials file holds no {service} token")
            if auth.source(service) is not None:
                print(f"{service}: still logged in from {auth.source(service)}, which is the environment's to unset")
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
    corrupt).

    ``mime pack <folder> [--out <file>]``: write the saved experiment in ``folder`` as one bundle file. ``mime pull
    <doi | url | file> [--into <folder>] [--no-assets] [--yes]``: get a bundle, check it, unpack it into a new folder,
    fetch its assets and say what to install. ``mime auth login | logout | status <service>``: the token ``push``
    will use."""
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
    pack = commands.add_parser("pack", help="write a saved experiment folder as one bundle file")
    pack.add_argument("folder", type=Path, help="a folder written by Experiment.save")
    pack.add_argument("--out", type=Path, help="the bundle file (default: <folder>.mimebundle beside the folder)")
    pull = commands.add_parser("pull", help="get a bundle by DOI, address or file, unpack it into a new folder, and "
                                            "fetch its assets")
    pull.add_argument("source", help="a DOI (10.5281/zenodo.1234567), an https:// address, or a bundle file")
    pull.add_argument("--into", type=Path, help="the new folder (default: ./<the experiment's name>/)")
    pull.add_argument("--no-assets", action="store_true", help="unpack only; fetch the assets later with "
                                                               "`mime assets fetch <folder>`")
    pull.add_argument("--yes", action="store_true", help="download assets over the size MIME asks about without asking")
    from mime import auth

    credentials = commands.add_parser("auth", help="the access token mime push will use, kept where a commit cannot "
                                                   "publish it")
    credentials_commands = credentials.add_subparsers(dest="auth_command", required=True)
    for verb, what in (("login", "read a token (hidden) and store it"), ("logout", "delete the stored token"),
                       ("status", "whether there is a token, where from, and its last four characters")):
        each = credentials_commands.add_parser(verb, help=what)
        each.add_argument("service", nargs="?" if verb == "status" else None, choices=list(auth.SERVICES))
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
    if arguments.command == "pack":
        return _pack(arguments.folder, arguments.out)
    if arguments.command == "pull":
        return _pull(arguments.source, arguments.into, arguments.no_assets, arguments.yes)
    if arguments.command == "auth":
        return _auth(arguments.auth_command, arguments.service)
    parser.error(f"unknown command {arguments.command!r}")


__all__ = ["main"]
