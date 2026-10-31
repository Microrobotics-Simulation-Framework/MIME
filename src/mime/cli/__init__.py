"""The ``mime`` command, a thin layer over the Python API: each command is one call a script could make."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from mime._extras import VERIFY_EXTRA_MISSING
from mime._terminal import for_terminal
from mime.assets import AssetRef
from mime.build import BuildRefused
from mime.experiment import Experiment


def _say(line: str) -> None:
    """A line of progress or of a summary, with anything a stranger's text could do to a terminal written out."""
    print(for_terminal(line), flush=True)


def _verify(folder: Path) -> int:
    from mime.env import apply_lock_precision

    try:
        apply_lock_precision(folder)
        experiment = Experiment.load(folder)
        report = experiment.build().verify()
    except BuildRefused as refused:
        print(for_terminal(str(refused)), file=sys.stderr)
        return 2
    if experiment.load_report is not None:
        print(for_terminal(experiment.load_report.text()))
    print(for_terminal(report.text()))
    return 0 if report.reproduced else 1


def _run(folder: Path, duration: float, out: Path, history: bool, record: Sequence[str] = (),
         every: float | None = None) -> int:
    import time

    from mime.env import apply_lock_precision
    from mime.units import s

    started = time.perf_counter()
    try:
        apply_lock_precision(folder)
        experiment = Experiment.load(folder)
        built = experiment.build()
    except BuildRefused as refused:
        print(refused, file=sys.stderr)
        return 2
    if experiment.load_report is not None:
        print(experiment.load_report.text())
    try:
        written = built.run_to(out, s(duration), history=history, record=list(record),
                               every=s(every) if every is not None else None, source=folder,
                               load_seconds=time.perf_counter() - started)
    except ValueError as error:
        print(f"mime run: {error}", file=sys.stderr)
        return 2
    print(f"ran {written.steps} steps; wrote {written.outputs}, {written.manifest}, {written.record_file} and the "
          f"run's saved form, {written.experiment}")
    print(written.result.text())
    return 0


def _env_check(arguments: argparse.Namespace) -> int:
    from mime import env

    try:
        report = env.check(arguments.folder, variant=arguments.variant, gpu=arguments.gpu, base=arguments.base,
                           lab=arguments.lab, xla_cache=arguments.xla_cache, mime_version=arguments.mime_version)
    except BuildRefused as refused:
        print(refused, file=sys.stderr)
        _write_record(arguments.record, {"refused": [str(r) for r in refused.refusals]})
        return 2
    print(report.text())
    _write_record(arguments.record, report.record())
    return 0


def _write_record(path: Path | None, record: dict[str, object]) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _modal(arguments: argparse.Namespace) -> int:
    from mime import env

    try:
        written = env.modal_script(arguments.folder, gpu=arguments.gpu, deterministic=arguments.deterministic,
                                   python=arguments.python, out=arguments.out, checkout=arguments.checkout,
                                   lab_requirements=arguments.lab_requirements, xla_cache=not arguments.no_xla_cache,
                                   timeout=arguments.timeout, max_containers=arguments.max_containers,
                                   cpu=arguments.cpu, memory=arguments.memory)
    except BuildRefused as refused:
        print(refused, file=sys.stderr)
        return 2
    print(written.text())
    return 0


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
                print(for_terminal(f"{ref.name}  {assets.size_text(ref.size)}  "
                                   f"{'cached' if assets.is_cached(ref) else 'missing'}  {ref.licence}"))
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
                stored = assets.fetch(ref, report=_say)
                print(f"{ref.name}: hashed in full, at {stored}")
    except BuildRefused as refused:
        print(for_terminal(str(refused)), file=sys.stderr)
        return 2
    return 0


def _pack(folder: Path, out: Path | None) -> int:
    from mime import assets, saved

    try:
        path = saved.pack(folder, out=out)
    except BuildRefused as refused:
        print(for_terminal(str(refused)), file=sys.stderr)
        return 2
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    print(for_terminal(f"packed {folder} into {path} ({assets.size_text(path.stat().st_size)})"))
    print(f"sha256 {digest}")
    return 0


def _ask(question: str) -> bool:
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _pull(source: str, into: Path | None, no_assets: bool, yes: bool, max_bundle_bytes: int) -> int:
    from mime import saved

    try:
        saved.pull(source, into=into, fetch_assets=not no_assets, yes=yes, confirm=_ask if sys.stdin.isatty() else None,
                   say=_say, max_bundle_bytes=max_bundle_bytes)
    except BuildRefused as refused:
        print(for_terminal(str(refused)), file=sys.stderr)
        return 2
    return 0


#: The words of ``mime credentials``: a verb, then a service where the verb takes one.
_CREDENTIAL_VERBS = ("set", "status", "remove")
_CREDENTIAL_USAGE = ("usage: mime credentials set|status|remove <zenodo|zenodo-sandbox>\n"
                     "A token is never an argument: `set` asks for it, hidden, and a shell would keep an argument in "
                     "its history.")


def _credential_words_problem(after: Sequence[str]) -> str | None:
    """Why the words after ``mime credentials`` are not a verb and a service, or ``None``. Said without quoting any
    word, so that a token typed where it does not belong is not written to the screen or a log by the message."""
    from mime import credentials

    if any(word in ("-h", "--help") for word in after):
        return None
    if not after or after[0] not in _CREDENTIAL_VERBS:
        return "the first word is set, status or remove"
    if len(after) > 2:
        return "a verb takes one service and nothing else"
    if len(after) == 2 and after[1] not in credentials.SERVICES:
        return f"the service is one of {', '.join(credentials.SERVICES)}"
    return None


def _credentials(command: str, service: str | None) -> int:
    from mime import credentials

    try:
        if command == "status":
            for each in [service] if service else list(credentials.SERVICES):
                print(for_terminal(credentials.describe(each)))
        elif service is None:
            raise AssertionError("argparse requires a service for set and remove")
        elif command == "set":
            print(f"stored the {service} token in {credentials.store(service, credentials.prompt(service))}")
        else:
            print(f"removed the {service} token from the credentials file" if credentials.remove(service)
                  else f"the credentials file holds no {service} token")
            if credentials.source(service) is not None:
                print(f"{service}: a token is still set by {credentials.source(service)}, which is the environment's to unset")
    except BuildRefused as refused:
        print(for_terminal(str(refused)), file=sys.stderr)
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

    ``mime run <folder> --duration <seconds> --out <dir> [--history] [--record <output>]... [--every <seconds>]``:
    load and rebuild a saved experiment, run it, and write its outputs, their manifest (``outputs.json``), its record
    and the run's own saved form into ``<dir>``, which ``RunResult.read`` reads back. Exits 0, or 2 when refused.

    ``mime env check [<folder>] --variant cpu|cuda12 [--gpu <type>] [--base] [--lab <file>] [--xla-cache <dir>]``:
    hold this environment to MIME's pins and the GPU requested, then apply the precision the folder's lock records.
    Exits 0, or 2 with every difference.

    ``mime modal <folder> [--gpu <type>] [--deterministic] ...``: write the script that runs the saved experiment on
    Modal, beside it. Exits 0, or 2 when refused.

    ``mime pack <folder> [--out <file>]``: write the saved experiment in ``folder`` as one bundle file. ``mime pull
    <doi | url | file> [--into <folder>] [--no-assets] [--yes]``: get a bundle, check it, unpack it into a new folder,
    fetch its assets and say what to install. ``mime credentials set | status | remove <service>``: the token
    ``push`` will use, kept where a commit cannot publish it."""
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
    pull.add_argument("--max-bundle-bytes", type=int, default=assets.RETRIEVE_LIMIT_BYTES, metavar="BYTES",
                      help="the most a bundle may be when nothing states its size (default: "
                           f"{assets.size_text(assets.RETRIEVE_LIMIT_BYTES)})")
    from mime import credentials

    held_credentials = commands.add_parser("credentials", help="the access token mime push will use, kept where a "
                                                               "commit cannot publish it")
    credential_commands = held_credentials.add_subparsers(dest="credentials_command", required=True)
    for verb, what in (("set", "read a token (hidden) and store it"), ("remove", "delete the stored token"),
                       ("status", "whether there is a token, where from, and its last four characters, if 32 or more")):
        each = credential_commands.add_parser(verb, help=what)
        each.add_argument("service", nargs="?" if verb == "status" else None, choices=list(credentials.SERVICES))
    words = list(sys.argv[1:] if argv is None else argv)
    if words[:1] == ["credentials"]:
        problem = _credential_words_problem(words[1:])
        if problem is not None:
            parser.exit(2, f"mime credentials: {problem}\n{_CREDENTIAL_USAGE}\n")
    run = commands.add_parser("run", help="load a saved experiment, run it, and save the run beside its outputs")
    run.add_argument("folder", type=Path, help="a folder written by Experiment.save")
    run.add_argument("--duration", type=float, required=True, help="seconds of simulated time, a whole number of the "
                                                                   "experiment's base steps")
    run.add_argument("--out", type=Path, required=True, help="the folder to write the outputs, the record and the "
                                                             "run's saved form to")
    run.add_argument("--history", action="store_true", help="write every step's fields, not only the last")
    run.add_argument("--record", action="append", default=[], metavar="OUTPUT",
                     help="a per-step output to record as a series, <observation id>.<output> or an observation's id "
                          "for all of its own (repeat it for more)")
    run.add_argument("--every", type=float, default=None, metavar="SECONDS",
                     help="the spacing of the recorded series' samples, a whole number of base steps (default: every "
                          "step)")
    from mime.env import EXACT_GPUS, VARIANTS

    held_env = commands.add_parser("env", help="the environment a run is made in")
    env_commands = held_env.add_subparsers(dest="env_command", required=True)
    env_check = env_commands.add_parser("check", help="hold this environment to MIME's pins and the GPU requested, "
                                                      "then apply the precision the experiment's lock records")
    env_check.add_argument("folder", type=Path, nargs="?", help="a saved experiment, whose lock's precision to apply")
    env_check.add_argument("--variant", choices=VARIANTS, required=True, help="the pins to hold the environment to")
    env_check.add_argument("--gpu", metavar="TYPE", help=f"the exact GPU type requested ({', '.join(EXACT_GPUS)}); "
                                                         f"without it, the CPU")
    env_check.add_argument("--base", action="store_true", help="hold Python, the OS and the image's own "
                                                               "distributions to MIME's base image for this series")
    env_check.add_argument("--lab", type=Path, metavar="FILE", help="the lab's requirements file, pinned beside MIME's")
    env_check.add_argument("--xla-cache", type=Path, metavar="DIR", help="keep XLA's compilation cache under DIR, "
                                                                         "in its directory for this jaxlib and device")
    env_check.add_argument("--mime-version", metavar="VERSION",
                           help="the version of the MIME wheel pinned for the run; without it MIME is not held to one")
    env_check.add_argument("--record", type=Path, metavar="FILE", help="also write what the check found as JSON")
    modal = commands.add_parser("modal", help="write the script that runs a saved experiment on Modal")
    modal.add_argument("folder", type=Path, help="a folder written by Experiment.save")
    modal.add_argument("--gpu", metavar="TYPE", help=f"an exact GPU type ({', '.join(EXACT_GPUS)}); without it, a CPU")
    modal.add_argument("--deterministic", action="store_true", help="turn XLA's deterministic GPU mode on "
                                                                    "(--xla_gpu_deterministic_ops), recorded in the "
                                                                    "run's lock")
    modal.add_argument("--python", metavar="SERIES", help="the Python series of the image (default: the lock's)")
    modal.add_argument("--out", type=Path, help="the folder to write the script to (default: <folder>-modal beside "
                                                "the experiment)")
    modal.add_argument("--checkout", type=Path, help="build MIME's wheel from this checkout (default: the checkout "
                                                     "MIME is installed from, else the release's wheel on PyPI)")
    modal.add_argument("--lab-requirements", type=Path, metavar="FILE",
                       help="the lab's hashed requirements file, in place of the version pins the lock names")
    modal.add_argument("--no-xla-cache", action="store_true", help="keep no XLA compilation cache on a Volume")
    modal.add_argument("--timeout", type=int, default=7200, help="seconds a run may take (default: 7200)")
    modal.add_argument("--max-containers", type=int, default=1, help="containers a function may run in at once "
                                                                     "(default: 1)")
    modal.add_argument("--cpu", type=float, default=4.0, help="CPU cores to request (default: 4)")
    modal.add_argument("--memory", type=int, default=8192, help="MiB of memory to request (default: 8192)")
    arguments = parser.parse_args(argv)
    if arguments.command == "verify":
        return _verify(arguments.folder)
    if arguments.command == "registry":
        return _registry(arguments.json)
    if arguments.command == "run":
        return _run(arguments.folder, arguments.duration, arguments.out, arguments.history, arguments.record,
                    arguments.every)
    if arguments.command == "env":
        return _env_check(arguments)
    if arguments.command == "modal":
        return _modal(arguments)
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
        return _pull(arguments.source, arguments.into, arguments.no_assets, arguments.yes, arguments.max_bundle_bytes)
    if arguments.command == "credentials":
        return _credentials(arguments.credentials_command, arguments.service)
    parser.error(f"unknown command {arguments.command!r}")


__all__ = ["main"]
