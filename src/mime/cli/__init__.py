"""The ``mime`` command, a thin layer over the Python API: each command is one call a script could make."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import zipfile
from collections.abc import Sequence
from pathlib import Path

from mime._extras import VERIFY_EXTRA_MISSING
from mime._terminal import for_terminal
from mime.assets import AssetRef
from mime.build import BuildRefused, InternalError, Refusal, RefusalClass, RunFailed, UnrecognisedError
from mime.errors import TRACEBACK_VARIABLE, crossed, full_mode, short_text
from mime.experiment import Experiment

#: The flag that shows a refused command's full traceback, anywhere on the command line.
TRACEBACK_FLAG = "--traceback"
#: The exit code of a command refused for the user's or the environment's reason, and of one that met a MIME defect
#: or an error MIME does not recognise.
REFUSED, FAILED = 2, 3


def _say(line: str) -> None:
    """A line of progress or of a summary, with anything a stranger's text could do to a terminal written out."""
    print(for_terminal(line), flush=True)


def _verify(folder: Path) -> int:
    from mime.env import apply_lock_precision

    apply_lock_precision(folder)
    experiment = Experiment.load(folder)
    report = experiment.build().verify()
    if experiment.load_report is not None:
        print(for_terminal(experiment.load_report.text()))
    print(for_terminal(report.text()))
    return 0 if report.reproduced else 1


def _pending(folder: Path) -> int:
    """Each item a saved experiment leaves pending, with the lines to paste: 0 when nothing is pending, 1 when something
    is, 2 when the load or build is refused (with what is pending, where the profile was read), 3 when it meets a MIME
    defect or an error MIME does not recognise. ``folder`` may be a
    document instead (``experiment.mime.yaml``, edited to seal it), which is planned as it is, with no lock to check:
    the route a sealed document takes before it is built and saved again."""
    from mime.env import apply_lock_precision

    try:
        if folder.is_file():
            profile = Experiment.from_document(folder).plan().profile
        else:
            apply_lock_precision(folder)
            built = Experiment.load(folder).build()
            profile = built.report.profile
    except BuildRefused as error:
        code = refused(error, "pending")
        if error.report is not None and error.report.profile is not None:
            print(for_terminal(error.report.pending()), file=sys.stderr)
        return code
    if profile is None:
        print("no profile: the build was refused before its profile was read", file=sys.stderr)
        return 2
    print(for_terminal(profile.text()), end="")
    return 1 if profile.pending else 0


def _run(folder: Path, duration: float | None, out: Path, history: bool, record: Sequence[str] = (),
         every: float | None = None, stage_every: float | None = None, commands: Path | None = None) -> int:
    import time

    from mime.env import apply_lock_precision
    from mime.units import s

    started = time.perf_counter()
    apply_lock_precision(folder)
    experiment = Experiment.load(folder)
    built = experiment.build()
    if experiment.load_report is not None:
        print(experiment.load_report.text())
    if commands is not None:
        from mime.live import replay

        session = replay(built, commands, out)
        print(f"replayed {commands}: {session.steps - session.start_step} steps in chunks of {session.chunk_steps}; "
              f"wrote the session's run folder, {out}")
        return 0
    if duration is None:
        raise BuildRefused([Refusal("run.duration", RefusalClass.USER, "mime run needs --duration, or --commands",
                                    "give the simulated time: mime run <folder> --duration <seconds> --out <dir>")])
    try:
        written = built.run_to(out, s(duration), history=history, record=list(record),
                               every=s(every) if every is not None else None, source=folder,
                               load_seconds=time.perf_counter() - started,
                               stage_every=s(stage_every) if stage_every is not None else None)
    except RunFailed as failed:
        if "run.non-finite" not in failed.codes():
            raise
        raise _non_finite_here(failed, folder, duration, out, built.plans.dt) from failed
    print(f"ran {written.steps} steps; wrote {written.outputs}, {written.manifest}, {written.record_file} and the "
          f"run's saved form, {written.experiment}")
    print(written.result.text())
    return 0


def _non_finite_here(failed: RunFailed, folder: Path, duration: float, out: Path, dt: float) -> RunFailed:
    """``run.non-finite`` as the command line acts on it: its outputs written, and what to run or edit next."""
    from dataclasses import replace

    from mime.saved import DOCUMENT_FILE

    rerun = f"`mime run {folder} --duration {duration!r} --out <another folder> --history`"
    smaller = (f"give a smaller base step: in {folder / DOCUMENT_FILE}, set dt under run: to {dt / 10!r} s, then save "
               f"the folder again (Experiment.from_document(document).save(folder))")
    refusals: list[Refusal] = []
    for refusal in failed.refusals:
        if refusal.code != "run.non-finite":
            refusals.append(refusal)
            continue
        named = "stopped being finite numbers at step" in refusal.message
        action = f"{smaller}; and check the first step's inputs" if named else \
            f"find its first step: {rerun}, which keeps every step; or {smaller}"
        refusals.append(replace(refusal, message=f"{refusal.message}; the outputs are written in {out}, outputs.json "
                                                 f"marking the run non_finite", action=action))
    held = RunFailed(refusals, failed.result)
    held.written = failed.written
    return held


def _serve(arguments: argparse.Namespace) -> int:
    from mime.live import serve
    from mime.units import s

    serve(arguments.folder, commands=arguments.commands, frames=arguments.frames,
          chunk=s(arguments.chunk) if arguments.chunk is not None else None,
          pace=None if arguments.unpaced else arguments.pace, out=arguments.out,
          select=arguments.select or None, json_frames=arguments.json_frames, start=arguments.start,
          remote=arguments.remote, ready_file=arguments.ready_file, say=_say,
          outlive_launcher=arguments.outlive_launcher)
    return 0


def _visuals(arguments: argparse.Namespace) -> int:
    from mime import viz
    from mime.scene import visual_sets

    if arguments.visuals_command == "list":
        baked = {name for name, _ in viz.baked_layers()}
        for name, held in sorted(visual_sets().items()):
            print(for_terminal(f"{name}  {len(held.files)} files  {'baked' if name in baked else 'not fetched'}  "
                               f"{held.licence} ({held.tier}, {held.role})"))
        return 0
    if arguments.visuals_command == "fetch":
        viz.fetch(arguments.set, say=_say)
        print(f"{arguments.set}: baked into the asset cache, at {viz.visuals_dir()}; never into a saved or run folder")
        return 0
    path, found = viz.view(arguments.target)
    print(for_terminal(found.text()))
    print(path)
    return 0


def _env_check(arguments: argparse.Namespace) -> int:
    from mime import env

    try:
        report = env.check(arguments.folder, variant=arguments.variant, gpu=arguments.gpu, base=arguments.base,
                           lab=arguments.lab, xla_cache=arguments.xla_cache, mime_version=arguments.mime_version)
    except BuildRefused as refused:
        _write_record(arguments.record, {"refused": [str(r) for r in refused.refusals]})
        raise
    print(report.text())
    _write_record(arguments.record, report.record())
    return 0


def _write_record(path: Path | None, record: dict[str, object]) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _modal(arguments: argparse.Namespace) -> int:
    from mime import env

    written = env.modal_script(arguments.folder, gpu=arguments.gpu, deterministic=arguments.deterministic,
                               python=arguments.python, out=arguments.out, checkout=arguments.checkout,
                               lab_requirements=arguments.lab_requirements, xla_cache=not arguments.no_xla_cache,
                               timeout=arguments.timeout, max_containers=arguments.max_containers,
                               cpu=arguments.cpu, memory=arguments.memory)
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
        unplaced = [each for argument in arguments if Path(argument).is_dir() for each in _place(Path(argument))]
        if unplaced:
            raise BuildRefused(unplaced)
    return 0


def _place(folder: Path) -> list[Refusal]:
    """Put the folder's files that travel as references in place from the cache, saying what it did; an
    ``asset.missing`` for each it could not place, with why."""
    from mime import saved
    from mime.saved._terms import unplaced_refusals

    placed = saved.place_referenced(folder)
    for path in placed.placed:
        print(for_terminal(f"{path}: placed from the asset cache"))
    return unplaced_refusals(folder, placed)


def _pack(folder: Path, out: Path | None) -> int:
    from mime import assets, saved
    from mime.saved._bundle import read_manifest, referenced_of
    from mime.saved._document import DESCRIPTIONS, SCENE_FOLDER
    from mime.saved._folder import named_files
    from mime.scene import LAYERS

    path = saved.pack(folder, out=out)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    with zipfile.ZipFile(path) as archive:
        manifest = read_manifest(archive)
    referenced = referenced_of(manifest)
    form = saved.read_folder(folder)
    licence = form.document.get("licence")
    print(for_terminal(f"packed {folder} into {path} ({assets.size_text(path.stat().st_size)})"))
    print(f"sha256 {digest}")
    print(for_terminal(f"carried: {len(manifest['files'])} files and the NOTICE; referenced, never carried: "
                       f"{len(referenced)}" + "".join(f"\n  {each} (the asset {held['name']})"
                                                     for each, held in sorted(referenced.items()))))
    print(for_terminal(f"licence: {licence}" if licence else
                       "licence: none declared (packing never refuses that; mime push will ask for one: "
                       "exp.licence(\"CC-BY-4.0\"), then save)"))
    if form.previous and any(each.startswith((f"{SCENE_FOLDER}/{LAYERS}/", f"{DESCRIPTIONS}/"))
                             for each in named_files(form)):
        print("this folder predates terms: a third party's file in it is carried as yours; declare its terms and "
              "save again")
    return 0


def _card(folder: Path, out: Path | None, notice: bool) -> int:
    from mime import saved

    text = saved.notice(folder) if notice else saved.card_text(folder)
    if out is None:
        print(for_terminal(text), end="" if text.endswith("\n") else "\n")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    print(for_terminal(f"wrote the {'NOTICE' if notice else 'card'} of {folder} to {out}"))
    return 0


def _ask(question: str) -> bool:
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _typed(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return ""


def _pull(source: str, into: Path | None, no_assets: bool, yes: bool, max_bundle_bytes: int,
          from_service: str | None = None) -> int:
    from mime import saved

    saved.pull(source, into=into, fetch_assets=not no_assets, yes=yes, confirm=_ask if sys.stdin.isatty() else None,
               say=_say, max_bundle_bytes=max_bundle_bytes, from_service=from_service)
    return 0


def _push(arguments: argparse.Namespace) -> int:
    from mime import push
    from mime.push._sharing import DEFAULT_LICENCE

    service = push.SERVICES[arguments.to]
    folder: Path = arguments.folder
    ask = _ask if sys.stdin.isatty() else None
    if arguments.confirm is not None and not arguments.publish:
        raise BuildRefused([Refusal("push.not-confirmed", RefusalClass.USER,
                                    "--confirm names the draft a publish publishes, and this is not a publish",
                                    f"`mime push {folder} --to {service.name} --publish --confirm <DOI>`")])
    if arguments.init:
        path, written, licence = push.init(folder)
        print(for_terminal(f"wrote {path}" if written else f"kept {path}, which exists: edit it, or delete it and "
                                                          f"run --init again"))
        print(for_terminal(f"licence: {licence}, which the experiment declares" if licence else
                           f"licence: {DEFAULT_LICENCE}, pre-filled: the experiment declares none, and push asks it to "
                           f"declare the same, exp.licence(\"{DEFAULT_LICENCE}\"), then save"))
        print(for_terminal(f"next: name the creators in it, then `mime push {folder} --to {service.name}`"))
        return 0
    if arguments.status:
        push.status(folder, service, say=_say)
        return 0
    if arguments.discard:
        push.discard(folder, service, yes=arguments.yes, confirm=ask, say=_say)
        return 0
    if arguments.publish:
        push.publish(folder, service, confirm=arguments.confirm, ask=ask, typed=_typed if ask is not None else None,
                     say=_say, overwrite_site_edits=arguments.overwrite_site_edits,
                     keep_site_edits=arguments.keep_site_edits, full=arguments.full_plan)
        return 0
    push.push(folder, service, yes=arguments.yes, confirm=ask, say=_say,
              overwrite_site_edits=arguments.overwrite_site_edits, keep_site_edits=arguments.keep_site_edits,
              full=arguments.full_plan)
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
            print(f"{service}: a token is still set by {credentials.source(service)}, which is the environment's to "
                  "unset")
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


def refused(error: Exception, command: str) -> int:
    """Show why ``mime <command>`` stopped, as a MIME error with its code (the full chain first, with
    ``--traceback``), and give its exit code: :data:`REFUSED` for the user's or the environment's reason,
    :data:`FAILED` for a MIME defect or an error MIME does not recognise."""
    coded = crossed(error)
    if full_mode() and coded.chain:
        print(for_terminal(coded.chain), file=sys.stderr)
    if isinstance(coded, UnrecognisedError):
        headline = f"mime {command} met an error MIME does not recognise"
    elif isinstance(coded, (InternalError, RunFailed)):
        headline = f"mime {command} failed"
    else:
        headline = f"mime {command} is refused"
    print(short_text(coded, None, frames=False, switch=TRACEBACK_FLAG, headline=headline), end="", file=sys.stderr)
    plain = all(r.category in (RefusalClass.USER, RefusalClass.ENVIRONMENT) for r in coded.refusals)
    return REFUSED if plain else FAILED


def _survive_a_narrow_stream() -> None:
    """Python before 3.15 encodes a redirected stream on Windows in the ANSI code page, which cannot hold a minus sign or
    a Greek letter in a path, and a report that did would raise after the work was done. The stream keeps its
    encoding; a character it cannot encode is written as its escape."""
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(errors="backslashreplace")


def main(argv: Sequence[str] | None = None) -> int:
    """``mime verify <folder>``: load a saved experiment, rebuild it, check it against its lock, and rerun its
    reference steps. Exits 0 when reproduced, 1 when an output moved, 2 when the load or build is refused.

    ``mime pending <folder>``: load and rebuild a saved experiment, and print each item its profile leaves pending (a
    waiver, a reason, a numerical default), in plain words with the lines to paste in Python and in the document, and
    each it lists. ``mime pending <folder>/experiment.mime.yaml`` plans the document as it is, with no lock: the check
    of a document edited to seal it, which ``Experiment.from_document`` builds. Exits 0 when nothing is pending, 1 when
    something is, 2 when refused, 3 on a MIME defect or an error MIME does not recognise.

    ``mime registry [--json]``: every registered entry, MIME's and each installed namespace's, in a fresh process.

    ``mime docs check [--root <checkout>] [--base <revision>]``: the documentation's sources and their checks, in a
    checkout of MIME; with ``--base``, the README start list against that revision's too. Exits 0 when every check
    passes, 1 when one fails, 2 without the ``verify`` extra.

    ``mime assets list | fetch <name>... | fetch --all | check``: the large files MIME references by sha256, in the
    cache. ``fetch`` stores them (and refuses with 2), ``check`` re-hashes the cached ones in full (1 when one is
    corrupt).

    ``mime run <folder> --duration <seconds> --out <dir> [--history] [--record <output>]... [--every <seconds>]
    [--stage-every <seconds>]``: load and rebuild a saved experiment, run it, and write its outputs, their manifest
    (``outputs.json``), its record, its topology, its build report's summary and the run's own saved form into
    ``<dir>``, which ``RunResult.read`` reads back, with the recording layer (``recording.usdc``) where
    ``--stage-every`` asks. ``mime run <folder> --commands <log> --out <dir>`` replays a live session from its command
    log. Exits 0, or 2 when refused.

    ``mime serve <folder> [--commands <endpoint>] [--frames <endpoint>] [--chunk <seconds>] [--pace <factor>]
    [--out <dir>] [--start]``: the saved experiment's live session, speaking ``mime.live/1`` over ZMQ (commands on a REP,
    frames on a PUB), bound to localhost unless ``--remote``, until a ``stop`` command, a signal, or the exit of the
    process that launched it (unless ``--outlive-launcher``); it writes a run folder when it stops.

    ``mime env check [<folder>] --variant cpu|cuda12 [--gpu <type>] [--base] [--lab <file>] [--xla-cache <dir>]``:
    hold this environment to MIME's pins and the GPU requested, then apply the precision the folder's lock records.
    Exits 0, or 2 with every difference.

    ``mime modal <folder> [--gpu <type>] [--deterministic] ...``: write the script that runs the saved experiment on
    Modal, beside it. Exits 0, or 2 when refused.

    ``mime visuals list | fetch <set> | view <target>``: the descriptions' meshes MIME does not ship, fetched from
    their pinned upstreams into the asset cache only, and the layer that draws them on a stage or a recording.

    ``mime pack <folder> [--out <file>]``: write the saved experiment in ``folder`` as one bundle file, a file its terms
    do not let be redistributed travelling as a reference. ``mime card <folder> [--out <file>] [--notice]``: the
    experiment's card, or its NOTICE, offline, as ``mime push`` will publish them. ``mime pull
    <doi | url | file> [--into <folder>] [--no-assets] [--yes]``: get a bundle, check it, unpack it into a new folder,
    fetch its assets and say what to install. ``mime credentials set | status | remove <service>``: the token
    ``push`` will use, kept where a commit cannot publish it. ``mime push <folder> --to zenodo|zenodo-sandbox
    [--init | --status | --discard | --publish [--confirm <DOI>]] [--yes] [--full-plan] [--keep-site-edits |
    --overwrite-site-edits]``: the plan, then the experiment's draft record and its files; ``--publish`` publishes
    the draft after the confirmation in words, each declared permission asked and the record number typed.
    ``mime pull --from zenodo|zenodo-sandbox <record>``: a published record by its number.

    A refused command prints why, as a MIME error with its code, and exits 2; one that met a MIME defect, or an error
    MIME does not recognise, exits 3. ``--traceback``, anywhere on the line, prints the full traceback first."""
    if argv is None:
        _survive_a_narrow_stream()
    parser = argparse.ArgumentParser(prog="mime", description=__doc__)
    parser.add_argument(TRACEBACK_FLAG, action="store_true",
                        help="print a refused command's full traceback, every frame and chained error (it may come "
                             "anywhere on the line)")
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify", help="load, rebuild, check the hash, and rerun against the reference "
                                                "outputs")
    verify.add_argument("folder", type=Path, help="a folder written by Experiment.save")
    pending = commands.add_parser("pending", help="what a saved experiment leaves pending, with the lines to paste "
                                                  "to seal it")
    pending.add_argument("folder", type=Path, help="a folder written by Experiment.save, or its document "
                                                   "(experiment.mime.yaml), edited, to check it as it is")
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
    card = commands.add_parser("card", help="a saved experiment's card (Markdown), or its NOTICE, offline, exactly as "
                                            "mime push will publish them")
    card.add_argument("folder", type=Path, help="a folder written by Experiment.save")
    card.add_argument("--out", type=Path, help="write it to this file (default: print it)")
    card.add_argument("--notice", action="store_true", help="the NOTICE (each file's licence, credit, verdict and "
                                                            "declared permission) in place of the card")
    from mime import credentials, push

    pull = commands.add_parser("pull", help="get a bundle by DOI, address or file, unpack it into a new folder, and "
                                            "fetch its assets")
    pull.add_argument("source", help="a DOI (10.5281/zenodo.1234567; a concept DOI pulls the latest version), an "
                                     "https:// address, a bundle file, or with --from a record's number")
    pull.add_argument("--from", dest="from_service", choices=list(push.SERVICES), metavar="SERVICE",
                      help="read the published record of this number at the service's records API (zenodo-sandbox: "
                           "its DOIs do not resolve through doi.org)")
    pull.add_argument("--into", type=Path, help="the new folder (default: ./<the experiment's name>/)")
    pull.add_argument("--no-assets", action="store_true", help="unpack only; fetch the assets later with "
                                                               "`mime assets fetch <folder>`")
    pull.add_argument("--yes", action="store_true", help="download assets over the size MIME asks about without asking")
    pull.add_argument("--max-bundle-bytes", type=int, default=assets.RETRIEVE_LIMIT_BYTES, metavar="BYTES",
                      help="the most a bundle may be when nothing states its size (default: "
                           f"{assets.size_text(assets.RETRIEVE_LIMIT_BYTES)})")
    pushing = commands.add_parser("push", help="show the plan, then make or update the experiment's draft record on "
                                               "Zenodo and upload its files (nothing is published)")
    pushing.add_argument("folder", type=Path, help="a folder written by Experiment.save")
    pushing.add_argument("--to", required=True, choices=list(push.SERVICES),
                         help="the service: zenodo, or zenodo-sandbox to try it on Zenodo's test service (required)")
    pushing_mode = pushing.add_mutually_exclusive_group()
    pushing_mode.add_argument("--init", action="store_true", help="write sharing.yaml, the record's credit and "
                                                                  "description, from what the folder knows")
    pushing_mode.add_argument("--status", action="store_true", help="the draft as the service holds it, against what "
                                                                    "push uploaded")
    pushing_mode.add_argument("--discard", action="store_true", help="delete the draft (only a draft can be deleted)")
    pushing_mode.add_argument("--publish", action="store_true",
                              help="publish the draft, the one step that cannot be undone: the plan, the confirmation "
                                   "in words, each declared permission asked, then the record number typed")
    pushing.add_argument("--confirm", metavar="DOI", help="with --publish, from a script: the draft's reserved DOI, in "
                                                          "place of the record number typed (it must be that draft's)")
    pushing.add_argument("--yes", action="store_true", help="delete the draft with --discard without asking (it never "
                                                            "publishes, declares a permission or settles a site edit)")
    pushing.add_argument(push.FULL_FLAG, dest="full_plan", action="store_true",
                         help="the full plan: every file inside the bundle with its tier and verdict, every field sent")
    pushing.add_argument(push.KEEP_FLAG, dest="keep_site_edits", action="store_true",
                         help="keep each edit made on the service's site that sharing.yaml can hold, written into it, "
                              "without asking")
    pushing.add_argument(push.OVERWRITE_FLAG, dest="overwrite_site_edits", action="store_true",
                         help="write the draft's metadata from sharing.yaml and the folder over what was edited on the "
                              "service's site since push last wrote it (without it, such an edit stops the push)")
    held_credentials = commands.add_parser("credentials", help="the access token mime push will use, kept where a "
                                                               "commit cannot publish it")
    credential_commands = held_credentials.add_subparsers(dest="credentials_command", required=True)
    for verb, what in (("set", "read a token (hidden) and store it"), ("remove", "delete the stored token"),
                       ("status", "whether there is a token, where from, and its last four characters, if 32 or more")):
        each = credential_commands.add_parser(verb, help=what)
        each.add_argument("service", nargs="?" if verb == "status" else None, choices=list(credentials.SERVICES))
    words = list(sys.argv[1:] if argv is None else argv)
    full = TRACEBACK_FLAG in words
    words = [word for word in words if word != TRACEBACK_FLAG]
    if words[:1] == ["credentials"]:
        problem = _credential_words_problem(words[1:])
        if problem is not None:
            parser.exit(2, f"mime credentials: {problem}\n{_CREDENTIAL_USAGE}\n")
    run = commands.add_parser("run", help="load a saved experiment, run it, and save the run beside its outputs")
    run.add_argument("folder", type=Path, help="a folder written by Experiment.save")
    run.add_argument("--duration", type=float, default=None, help="seconds of simulated time, a whole number of the "
                                                                  "experiment's base steps (required, but with "
                                                                  "--commands)")
    run.add_argument("--out", type=Path, required=True, help="the folder to write the outputs, the record and the "
                                                             "run's saved form to")
    run.add_argument("--history", action="store_true", help="write every step's fields, not only the last")
    run.add_argument("--record", action="append", default=[], metavar="OUTPUT",
                     help="a per-step output to record as a series, <observation id>.<output> or an observation's id "
                          "for all of its own (repeat it for more)")
    run.add_argument("--every", type=float, default=None, metavar="SECONDS",
                     help="the spacing of the recorded series' samples, a whole number of base steps (default: every "
                          "step)")
    run.add_argument("--stage-every", type=float, default=None, metavar="SECONDS",
                     help="write the recording layer (recording.usdc, over the run's own stage) with a sample every "
                          "SECONDS, a whole number of the run's chunks")
    run.add_argument("--commands", type=Path, default=None, metavar="LOG",
                     help="replay a live session from the command log it wrote (commands.jsonl): its run folder again, "
                          "in place of --duration")
    held_visuals = commands.add_parser("visuals", help="descriptions' meshes MIME does not ship, baked into the asset "
                                                       "cache, and the overlay that draws them on a stage")
    visual_commands = held_visuals.add_subparsers(dest="visuals_command", required=True)
    visual_commands.add_parser("list", help="every visual set MIME has a recipe for, baked or not, with its licence")
    visuals_fetch = visual_commands.add_parser("fetch", help="download a visual set from its pinned upstream, check "
                                                             "each file's sha256, and bake it into the asset cache")
    visuals_fetch.add_argument("set", help="the visual set's name, from mime visuals list")
    visuals_view = visual_commands.add_parser("view", help="the layer that opens a stage, a recording or a folder's "
                                                           "with the baked meshes drawn, for any USD viewer")
    visuals_view.add_argument("target", type=Path, help="a stage or recording file, a run folder or a saved folder")
    from mime.live import DEFAULT_COMMANDS, DEFAULT_FRAMES

    serve = commands.add_parser("serve", help="serve a saved experiment's live session (mime.live/1, over ZMQ)")
    serve.add_argument("folder", type=Path, help="a folder written by Experiment.save")
    serve.add_argument("--commands", default=DEFAULT_COMMANDS, metavar="ENDPOINT",
                       help=f"where to take commands, a REP socket (default: {DEFAULT_COMMANDS}; port 0 takes any "
                            "free one)")
    serve.add_argument("--frames", default=DEFAULT_FRAMES, metavar="ENDPOINT",
                       help=f"where to send frames, a PUB socket (default: {DEFAULT_FRAMES})")
    serve.add_argument("--chunk", type=float, default=None, metavar="SECONDS",
                       help="the simulated time each chunk steps, one frame each (default: about 1/60 s, whole base "
                            "steps and control periods)")
    serve.add_argument("--pace", type=float, default=1.0, metavar="FACTOR",
                       help="run at FACTOR times real time (default: 1)")
    serve.add_argument("--unpaced", action="store_true", help="run as fast as it runs")
    serve.add_argument("--out", type=Path, default=None, metavar="DIR",
                       help="the run folder it writes when it stops (default: ./<experiment>-live-<time>/)")
    serve.add_argument("--select", action="append", default=[], metavar="FIELD",
                       help="a field the frames carry, <node>.<field> (repeat it; default: every bound pose and "
                            "vector, every per-step output and every flag)")
    serve.add_argument("--json-frames", action="store_true", help="send frames as JSON, for debugging")
    serve.add_argument("--start", action="store_true", help="start stepping at once, before a client's start")
    serve.add_argument("--remote", action="store_true",
                       help="allow an address that is not loopback (no authentication: prefer an SSH tunnel)")
    serve.add_argument("--ready-file", type=Path, default=None, metavar="FILE",
                       help="write the bound endpoints as JSON to FILE once the sockets listen")
    serve.add_argument("--outlive-launcher", action="store_true",
                       help="serve on when the process that launched it is gone (by default the session stops then, "
                            "writing its run folder, as on SIGTERM)")
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
    arguments = parser.parse_args(words)
    held = os.environ.get(TRACEBACK_VARIABLE)
    if full:
        os.environ[TRACEBACK_VARIABLE] = "full"
    try:
        return _command(parser, arguments)
    except Exception as error:  # noqa: BLE001 - every error a command meets is shown as a MIME error with its code
        return refused(error, arguments.command)
    finally:
        if full:
            if held is None:
                os.environ.pop(TRACEBACK_VARIABLE, None)
            else:
                os.environ[TRACEBACK_VARIABLE] = held


def _command(parser: argparse.ArgumentParser, arguments: argparse.Namespace) -> int:
    from mime import assets

    if arguments.command == "verify":
        return _verify(arguments.folder)
    if arguments.command == "pending":
        return _pending(arguments.folder)
    if arguments.command == "registry":
        return _registry(arguments.json)
    if arguments.command == "run":
        return _run(arguments.folder, arguments.duration, arguments.out, arguments.history, arguments.record,
                    arguments.every, arguments.stage_every, arguments.commands)
    if arguments.command == "serve":
        return _serve(arguments)
    if arguments.command == "visuals":
        return _visuals(arguments)
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
    if arguments.command == "card":
        return _card(arguments.folder, arguments.out, arguments.notice)
    if arguments.command == "pull":
        return _pull(arguments.source, arguments.into, arguments.no_assets, arguments.yes, arguments.max_bundle_bytes,
                     arguments.from_service)
    if arguments.command == "credentials":
        return _credentials(arguments.credentials_command, arguments.service)
    if arguments.command == "push":
        return _push(arguments)
    parser.error(f"unknown command {arguments.command!r}")


__all__ = ["FAILED", "REFUSED", "TRACEBACK_FLAG", "main", "refused"]
