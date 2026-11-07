"""The ``mime`` command, a thin layer over the Python API: each command is one call a script could make."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import zipfile
from collections.abc import Callable, Generator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, override

from mime import __version__, _import_guard
from mime._extras import VERIFY_EXTRA_MISSING
from mime._interrupt import INTERRUPTED, Interrupt
from mime._terminal import for_terminal
from mime.assets import AssetRef, PlainAsker
from mime.build import BuildRefused, InternalError, MimeError, Refusal, RefusalClass, RunFailed, UnrecognisedError
from mime.cli import _completion, _durations, _guide
from mime.cli._bounded import CLOSE_WAIT, BoundedWriter
from mime.cli._display import set_plain, styled
from mime.cli._durations import FORMS, seconds
from mime.cli._shell import quoted
from mime.errors import TRACEBACK_VARIABLE, crossed, full_mode, short_text
from mime.experiment import Experiment

if TYPE_CHECKING:
    from mime._transfers import Transfers
    from mime.cli._display import SaidDisplay
    from mime.credentials import Token
    from mime.env import EnvReport
    from mime.cli._display import RunDisplay
    from mime.control import Progress
    from mime.experiment import BuiltExperiment, RunWritten
    from mime.saved import LoadReport

#: The flag that shows a refused command's full traceback, anywhere on the command line.
TRACEBACK_FLAG = "--traceback"
#: The flag that draws nothing but plain text, as when output is piped, anywhere on the command line.
PLAIN_FLAG = "--plain"
#: What the top of ``mime --help`` ends with: an example of the commonest command, and where the rest are.
_ROOT_EXAMPLES = ("examples:\n  mime run my_experiment --duration 10s --out my_run\n"
                  "  mime pull 10.5281/zenodo.1234567\n"
                  "  mime <command> --help    one command's options, with an example")
#: The exit code of a command refused for the user's or the environment's reason, and of one that met a MIME defect
#: or an error MIME does not recognise.
REFUSED, FAILED = 2, 3


#: The duration options of ``mime serve`` that stop the session.
LIVE_TIMEOUTS = ("--idle-timeout", "--max-wall")


def option_seconds(option: str, text: str | None) -> float | None:
    """The value of a duration option (``--duration``, ``--every``, ``--stage-every``, and ``mime serve``'s
    ``--idle-timeout`` and ``--max-wall``) as seconds, ``None`` when it was
    not given; a value that is not a duration is refused with its option's code, naming the forms that are."""
    if text is None:
        return None
    value = seconds(text)
    if value is not None:
        return value
    message = f"{option} {text} is not a duration"
    action = f"write it with a unit, `mime run <folder> {option} 10s --out <dir>`; the forms are {FORMS}"
    why = "a bare number is read as seconds, and a unit after it as written: s, ms, us, min or h"
    if option in LIVE_TIMEOUTS:
        refusal = Refusal("live.timeout", RefusalClass.USER, message,
                          f"write it with a unit, `mime serve <folder> {option} 15min`; the forms are {FORMS}", why=why)
    elif option == "--every":
        refusal = Refusal("run.every", RefusalClass.USER, message, action, why=why)
    elif option == "--stage-every":
        refusal = Refusal("run.stage-every", RefusalClass.USER, message, action, why=why)
    else:
        refusal = Refusal("run.duration", RefusalClass.USER, message, action, why=why)
    raise BuildRefused([refusal])


def _say(line: str) -> None:
    """A line of progress or of a summary, with anything a stranger's text could do to a terminal written out."""
    print(for_terminal(line), flush=True)


def _warn(line: str) -> None:
    print(for_terminal(line), file=sys.stderr, flush=True)


def _xla_cache(arguments: argparse.Namespace) -> AbstractContextManager[object]:
    """XLA's persistent compilation cache for ``mime run`` and ``mime serve``, in the user's cache directory, unless
    ``--no-xla-cache``. ``mime verify`` does not use it: a verification compiles as a fresh process would."""
    if arguments.no_xla_cache:
        return nullcontext()
    from mime.env import user_xla_cache

    return user_xla_cache(say=_warn)


def _cache_clear() -> int:
    from mime.env import clear_user_xla_cache, user_xla_cache_root

    from mime.assets import bytes_text

    root = user_xla_cache_root()
    freed = clear_user_xla_cache(root)
    if freed and styled(sys.stdout):  # a terminal is told how much, in a size a person reads; no question: it rebuilds
        _say(f"freed {bytes_text(freed)} (XLA's compilation cache, {root}): it is made again by the next run")
    else:
        _say(f"removed {freed} bytes of XLA's compilation cache from {root}" if freed else
             f"no XLA compilation cache to remove at {root}")
    return 0


def _verify(folder: Path, run: bool = False) -> int:
    with Interrupt():
        try:
            return _verified_run(folder) if run else _verified(folder)
        except KeyboardInterrupt:
            _say("interrupted; nothing was verified")
            return INTERRUPTED


def _verified(folder: Path) -> int:
    from mime.env import apply_lock_precision

    apply_lock_precision(folder)
    experiment = Experiment.load(folder)
    report = experiment.build().verify()
    load = experiment.load_report.text() if experiment.load_report is not None else None

    def draw() -> None:
        from mime.cli._display import show_verification

        show_verification(load, report, sys.stdout)

    if not _drawn(draw):
        if load is not None:
            print(for_terminal(load))
        print(for_terminal(report.text()))
    return 0 if report.reproduced else 1


def _verified_run(folder: Path) -> int:
    from mime.study._override import verify_run

    report = verify_run(folder)
    print(for_terminal(report.text()))
    return 0 if report.reproduced else 1


def _drawn(draw: Callable[[], None]) -> bool:
    """Draw a result on standard output with ``draw`` when it is a terminal not asked to be plain; ``False`` when it is
    not (or ``rich`` cannot be imported), and the caller prints what it always printed."""
    if not styled(sys.stdout):
        return False
    try:
        draw()
    except ImportError:
        return False
    return True


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

    text = for_terminal(profile.text())

    def draw() -> None:
        from mime.cli._display import show_pending

        show_pending(text, sys.stdout)

    if not _drawn(draw):
        print(text, end="")
    return 1 if profile.pending else 0


#: ``--out`` given with no file: the default name, ``<experiment>.graph.html`` (compared by identity, never by value).
DEFAULT_PAGE = Path("<experiment>.graph.html")


def _show(folder: Path, graph: bool, tree: bool, as_json: bool, out: Path | None = None, cdn: bool = False,
          caption: str | None = None, open_page: bool = False) -> int:
    """What a saved experiment is made of, without running it: ``--tree``, its topology as declared (the default), and
    ``--graph``, the MADDENING graph it compiles to, as Mermaid text; both, the tree first and a blank line between.
    The tree is read from the document alone and builds nothing; the graph is built as ``mime pending`` builds it.
    ``folder`` may be a document instead of a folder. ``--json`` is the tree as data. ``--out [FILE]`` writes the graph as
    one page to pan and zoom, instead of printing it, as ``mime visuals web --out`` writes the 3-D view (the default name
    is ``<experiment>.graph.html``, never the 3-D page's; never inside a saved or run folder, ``visuals.web-out``)."""
    from mime.env import apply_lock_precision

    if out is not None and out is not DEFAULT_PAGE:
        from mime.experiment._mermaid import written_outside

        written_outside(out, f"mime show {folder} --graph --out graph.html")
    if out is not None:
        graph = True

    if tree or not graph:
        found = Experiment.from_document(_document_of(folder)).tree()

        def draw() -> None:
            from mime.cli._display import show_tree

            show_tree(found.lines(), sys.stdout)

        if as_json:
            sys.stdout.write(for_terminal(found.json()))
        elif not _drawn(draw):
            sys.stdout.write(for_terminal(found.text()))
    if graph:
        if tree:
            sys.stdout.write("\n")
        if folder.is_file():
            built = Experiment.from_document(folder).build()
        else:
            apply_lock_precision(folder)
            built = Experiment.load(folder).build()
        diagram = built.to_mermaid()
        if out is not None:
            written = diagram.save(None if out is DEFAULT_PAGE else out, cdn=cdn, caption=caption)
            where = "the library from jsDelivr, held to its hash" if cdn else "the library inlined: no network needed"
            print(for_terminal(f"{written}: {written.stat().st_size / 1e6:.2f} MB, the graph to pan and zoom, {where}"))
            if open_page:
                import webbrowser

                webbrowser.open(written.resolve().as_uri())
        else:
            sys.stdout.write(for_terminal(diagram))
    return 0


def _document_of(folder: Path) -> Path:
    """The document of the saved experiment in ``folder``; a folder that is not there is refused as ``Experiment.load``
    refuses it, naming the saved experiments beside it."""
    from mime.saved import DOCUMENT_FILE

    if folder.is_file():
        return folder
    if not folder.is_dir():
        Experiment.load(folder)
    return folder / DOCUMENT_FILE


def _run(folder: Path, duration: float | None, out: Path, history: bool, record: Sequence[str] = (),
         every: float | None = None, stage_every: float | None = None, commands: Path | None = None,
         full: bool = False, overrides: Sequence[str] = ()) -> int:
    """``mime run``. Ctrl+C: the first press, while the experiment loads and builds, ends the command (130, nothing
    written); once the run is under way it stops the run at its next chunk and the run folder is written as the steps
    reached, marked stopped (130); the second quits at once, writing nothing (``mime._interrupt``).

    On a terminal (``mime.cli._display``) the run shows its progress on standard error and ends with what needs
    attention first; ``full`` adds the rest. Anywhere else it prints what it always did."""
    from mime.cli._display import RunDisplay, damaged_cache_note

    display = RunDisplay(sys.stderr) if commands is None and styled(sys.stderr) else None
    interrupt = Interrupt(leave=display.leave, say=display.say) if display is not None else Interrupt()
    with damaged_cache_note(sys.stderr), interrupt:
        try:
            return _ran(folder, duration, out, history, record, every, stage_every, commands, interrupt, display, full,
                        overrides)
        except KeyboardInterrupt:
            if display is not None:
                display.close()
            _say("interrupted before the run began; nothing was written")
            return INTERRUPTED
        finally:
            if display is not None:
                display.close()


def _ran(folder: Path, duration: float | None, out: Path, history: bool, record: Sequence[str],
         every: float | None, stage_every: float | None, commands: Path | None, interrupt: Interrupt,
         display: RunDisplay | None, full: bool, overrides: Sequence[str] = ()) -> int:
    import time

    from mime.study._override import parse_override

    override = dict(parse_override(text) for text in overrides)

    from mime.env import apply_lock_precision
    from mime.units import TIME, s, si

    started = time.perf_counter()
    terminal = styled(sys.stdout)
    if display is not None:
        display.phase(f"loading {folder}: reading it, building it once and checking it against its lock")
    apply_lock_precision(folder)
    experiment = Experiment.load(folder)
    if display is not None:
        display.phase("building the graph to run it (compiling it, which can take many seconds)")
    built = experiment.build()
    if display is not None:
        display.pause()
    if experiment.load_report is not None and (not terminal or commands is not None):
        print(experiment.load_report.text())
    if commands is not None:
        from mime.live import replay

        session = replay(built, commands, out)
        print(f"replayed {commands}: {session.steps - session.start_step} steps in chunks of {session.chunk_steps}; "
              f"wrote the session's run folder, {out}")
        return 0
    held = si(built.duration, TIME, what="the experiment's duration") if built.duration is not None else None
    declared = held if isinstance(held, float) else None
    if duration is None and declared is not None:
        duration = declared
    if duration is None:
        raise BuildRefused([Refusal("run.duration", RefusalClass.USER,
                                    "mime run needs --duration, or --commands: this experiment declares no run length",
                                    "give the simulated time: mime run <folder> --duration <seconds> --out <dir>; or "
                                    "declare it where the experiment is made, Experiment(..., duration=s(3)), and "
                                    "save it again")])
    interrupt.running()
    if display is None or not terminal:
        own = declared is not None and duration == declared
        _say(f"running {duration!r} s of simulated time{' (the length the experiment declares)' if own else ''}; "
             "Ctrl+C stops it at the next chunk and writes the steps it reached, twice quits at once")
    stop = interrupt.stop
    if display is not None:
        stop = _watched(built, display, built.steps_in(s(duration)), interrupt, time.perf_counter())
    try:
        written = built.run_to(out, s(duration), history=history, record=list(record),
                               every=s(every) if every is not None else None, source=folder,
                               load_seconds=time.perf_counter() - started,
                               stage_every=s(stage_every) if stage_every is not None else None,
                               stop=stop, progress=False, override=override or None)
    except RunFailed as failed:
        if display is not None:
            display.close()
        if "run.non-finite" not in failed.codes():
            raise
        raise _non_finite_here(failed, folder, duration, out, built.plans.dt) from failed
    finally:
        if display is not None:
            display.close()
    stopped = written.stopped
    if terminal:
        _summarise(built, experiment.load_report, written, folder, out, started, full)
    else:
        print(f"ran {written.steps} steps; wrote {written.outputs}, {written.manifest}, {written.record_file} and the "
              f"run's saved form, {written.experiment}")
        print(written.result.text())
        if stopped is not None:
            print(f"{stopped.reason}: the folder holds the first {written.steps} of the {stopped.steps_asked} steps "
                  f"asked for ({written.steps * built.plans.dt:.6g} of {stopped.steps_asked * built.plans.dt:.6g} s), "
                  "and its outputs.json and run.json say it was stopped")
    return INTERRUPTED if stopped is not None else 0


def _watched(built: BuiltExperiment, display: RunDisplay, steps_asked: int, interrupt: Interrupt,
             began: float) -> Callable[[Progress], str | None]:
    """The ``stop=`` of a run that is drawn: it tells the display where each chunk ended (and, no oftener than a few
    times a second, which flags are raised there) and then asks what Ctrl+C asks."""
    import time

    from mime.cli._display import FlagPoll
    from mime.cli._progress import RunState

    state = RunState(steps_asked, built.plans.dt, began)
    poll = FlagPoll(built.raised_flags)
    display.running(state)

    def stop(progress: Progress) -> str | None:
        now = time.perf_counter()
        state.chunk(steps_asked - (progress.end - progress.steps), now, poll(now))
        return interrupt.stop(progress)

    return stop


def _summarise(built: BuiltExperiment, load: LoadReport | None, written: RunWritten, folder: Path, out: Path,
               started: float, full: bool) -> None:
    """The end of ``mime run`` on a terminal: what needs attention first, a short statement of what ran, the readings
    and the next steps; ``full`` adds the load report, the build report's summary and every reading."""
    import time

    from mime.cli import _summary
    from mime.cli._display import show_summary

    ended = time.perf_counter()
    readings = _summary.reading_lines(written)
    shown = readings if full else readings[:_summary.READINGS_SHOWN]
    if len(shown) < len(readings):
        shown = [*shown, f"... {len(readings) - len(shown)} more lines (--full prints them)"]
    summary = _summary.RunSummary(
        attention=tuple(_summary.attention_lines(built, written, load, folder)),
        ran=tuple(_summary.what_ran(built, written, started, ended)), readings=tuple(shown),
        wrote=tuple(_summary.wrote_lines(written, out)), next_steps=tuple(_summary.next_steps(folder, out)),
        more=_summary.more_line(out))
    sections: list[tuple[str, str]] = []
    if full:
        if load is not None:
            sections.append(("Load report", load.text()))
        sections.append(("Build report", built.report.text()))
        sections.append(("Result", written.result.text()))
    show_summary(summary, sys.stdout, full=sections)


def _non_finite_here(failed: RunFailed, folder: Path, duration: float, out: Path, dt: float) -> RunFailed:
    """``run.non-finite`` as the command line acts on it: its outputs written, and what to run or edit next."""
    from dataclasses import replace

    from mime.saved import DOCUMENT_FILE

    rerun = f"`mime run {quoted(str(folder))} --duration {duration!r} --out <another folder> --history`"
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
    """``mime serve``. On a terminal (``mime.cli._display``) it shows the live view on standard error; anywhere else it
    prints what it always printed, its two lines on standard output."""
    from mime.live import serve
    from mime.units import s

    idle_timeout = option_seconds("--idle-timeout", arguments.idle_timeout)
    max_wall = option_seconds("--max-wall", arguments.max_wall)
    display = None
    if styled(sys.stderr):
        from mime.cli._display import ServeDisplay

        display = ServeDisplay(sys.stderr)
    # serve's own lines never wait on a terminal: on the view's terminal they go through the view's bounded stream, and
    # anywhere else through a bounded stream of standard output's own
    lines = None if display is not None and display.shares_stdout else BoundedWriter(sys.stdout)

    def say(line: str) -> None:
        if lines is None:
            assert display is not None
            display.say(line)
        else:
            lines.write(for_terminal(line) + "\n")
            lines.flush()

    try:
        serve(arguments.folder, commands=arguments.commands, frames=arguments.frames,
              chunk=s(arguments.chunk) if arguments.chunk is not None else None,
              pace=None if arguments.unpaced else arguments.pace, out=arguments.out,
              select=arguments.select or None, json_frames=arguments.json_frames, start=arguments.start,
              remote=arguments.remote, ready_file=arguments.ready_file, say=say,
              outlive_launcher=arguments.outlive_launcher, watch=arguments.watch or None, view=display,
              idle_timeout=idle_timeout, max_wall=max_wall)
    finally:
        if display is not None:
            display.close()
        if lines is not None:
            lines.drain(CLOSE_WAIT)
    return 0


def _visuals(arguments: argparse.Namespace) -> int:
    from mime import viz
    from mime.scene import visual_sets

    if arguments.visuals_command == "list":
        baked = {name for name, _ in viz.baked_layers()}
        held_sets = visual_sets()

        def draw() -> None:
            from mime.assets import size_text
            from mime.cli._display import show_visual_sets

            show_visual_sets(
                [(name, str(len(held.files)), size_text(sum(ref.size for ref in held.assets)),
                  "baked" if name in baked else "not fetched", held.licence, held.verdict.tier, held.role)
                 for name, held in sorted(held_sets.items())]
                + [(name, "-", "-", "baked", "-", "-", "its manifest is beside the description it came with")
                   for name in sorted(baked - set(held_sets))], sys.stdout)

        if (held_sets or baked) and _drawn(draw):
            return 0
        for name, held in sorted(held_sets.items()):
            print(for_terminal(f"{name}  {len(held.files)} files  {'baked' if name in baked else 'not fetched'}  "
                               f"{held.licence} ({held.verdict.tier}, {held.role})"))
        for name in sorted(baked - set(held_sets)):
            print(for_terminal(f"{name}  baked (its manifest is beside the description it came with)"))
        if not held_sets and not baked:
            print("no set: a description's manifest is beside it (visuals.json); `mime visuals fetch <its folder>` "
                  "bakes its meshes")
        return 0
    if arguments.visuals_command == "fetch":
        manifests = viz.manifests_for(arguments.set)
        for held in manifests:
            with _gate_questions():
                viz.bake(held, say=_say, accept_licences=arguments.accept_licence, yes=arguments.yes,
                         force=arguments.force)
        names = [held.name for held in manifests]
        print(f"{', '.join(names)}: baked into the asset cache, at {viz.visuals_dir()}; never into a saved or run "
              f"folder")
        return 0
    if arguments.visuals_command == "web":
        return _visuals_web(arguments)
    with _gate_questions():
        path, found = viz.view(arguments.target, accept_licences=arguments.accept_licence)
    print(for_terminal(found.text()))
    print(path)
    return 0


def _visuals_web(arguments: argparse.Namespace) -> int:
    import webbrowser

    from mime import viz

    written, result = viz.web(arguments.target, arguments.out, cdn=arguments.cdn, caption=arguments.caption,
                              slow=arguments.slow_motion, glb=arguments.glb, scale_bar=not arguments.no_scale_bar,
                              legend=not arguments.no_legend, apparatus_shapes=not arguments.no_apparatus_shapes,
                              focus=arguments.focus, follow=arguments.follow)
    for line in result.lines():
        print(for_terminal(line))
    size = written.stat().st_size
    viewer = "the viewer from jsDelivr, held to its hash" if arguments.cdn else "the viewer inlined: no network needed"
    print(for_terminal(f"{written}: {size / 1e6:.2f} MB, {result.triangles} triangles, {viewer}"))
    if arguments.glb is not None:
        print(for_terminal(f"{arguments.glb}: the GLB alone, {len(result.glb) / 1e6:.2f} MB"))
    if arguments.open:
        webbrowser.open(written.resolve().as_uri())
    return 0


def _env_check(arguments: argparse.Namespace) -> int:
    from mime import env

    try:
        report = env.check(arguments.folder, variant=arguments.variant, gpu=arguments.gpu, base=arguments.base,
                           lab=arguments.lab, xla_cache=arguments.xla_cache, mime_version=arguments.mime_version)
    except BuildRefused as refused:
        _write_record(arguments.record, {"refused": [str(r) for r in refused.refusals]})
        raise
    if not _drawn(lambda: _show_environment(report)):
        print(report.text())
    _write_record(arguments.record, report.record())
    return 0


def _show_environment(report: EnvReport) -> None:
    from mime.cli._display import show_environment

    show_environment(report, sys.stdout)


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

    def draw() -> None:
        from mime.cli._display import show_registry

        show_registry([(e["kind"], e["name"], e["version"], ", ".join(e["settings"])) for e in found["entries"]],
                      found["not_loaded"], sys.stdout)

    if not _drawn(draw):
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


def _assets(command: str, arguments: Sequence[str], accept_licences: Sequence[str] = (), yes: bool = False,
            force: bool = False) -> int:
    from mime import assets

    refs = _assets_named(arguments) if arguments else list(assets.known().values())
    if command == assets.LIST:
        def draw() -> None:
            from mime.cli._display import show_assets

            show_assets([(ref.name, assets.size_text(ref.size), "cached" if assets.is_cached(ref) else "missing",
                          ref.licence, ref.tier) for ref in refs], sys.stdout)

        if not _drawn(draw):
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
        with _gate_questions(), assets.approved(refs, accept_licences=accept_licences, yes=yes, force=force):
            for ref in refs:
                stored = assets.fetch(ref, report=_say)
                print(f"{ref.name}: hashed in full, at {stored}")
        unplaced = [each for argument in arguments if Path(argument).is_dir()
                    for each in _place(Path(argument), accept_licences)]
        if unplaced:
            raise BuildRefused(unplaced)
    return 0


def _sources(names: Sequence[str]) -> int:
    """``mime assets sources``: each asset's terms and where it can be fetched from, each location looked at now."""
    from mime import assets

    named = [assets.assets_of(name) for name in names]

    def draw() -> None:
        from mime.cli._display import show_sources

        show_sources([view for refs in named for view in assets.sources_view(refs)], sys.stdout)

    if not _drawn(draw):
        print(for_terminal("\n".join(assets.sources_text(refs) for refs in named)))
    return 0


def _licences(revoke: str | None) -> int:
    """``mime assets licences``: the licences accepted on this machine; with ``--revoke <id>``, remove one (1 where none
    of that id was kept)."""
    from mime import assets

    if revoke is not None:
        if assets.revoke(revoke):
            print(for_terminal(f"{revoke}: no longer accepted; the next download under it asks again"))
            return 0
        print(for_terminal(f"{revoke}: no such acceptance is kept (`mime assets licences` lists them)"), file=sys.stderr)
        return 1
    held = assets.read_acceptances()

    def draw() -> None:
        from mime.cli._display import show_acceptances

        show_acceptances(held, sys.stdout)

    if not held or not _drawn(draw):
        print(for_terminal(assets.acceptances_text()))
    return 0


def _place(folder: Path, accept_licences: Sequence[str] = ()) -> list[Refusal]:
    """Put the folder's files that travel as references in place from the cache, saying what it did; an
    ``asset.missing`` for each it could not place, with why."""
    from mime import saved
    from mime.saved._terms import unplaced_refusals

    placed = saved.place_referenced(folder, accept_licences=accept_licences)
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
    if form.predates_terms and any(each.startswith((f"{SCENE_FOLDER}/{LAYERS}/", f"{DESCRIPTIONS}/"))
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


def _a_terminal() -> bool:
    """Whether a person can answer here: stdin is a terminal (Python leaves ``sys.stdin`` as ``None`` where it is
    closed)."""
    stdin = getattr(sys, "stdin", None)
    return stdin is not None and bool(stdin.isatty())


def _tty(stream: object) -> bool:
    try:
        return bool(stream.isatty())  # type: ignore[attr-defined]
    except (AttributeError, ValueError, OSError):
        return False


def _prompted(prompt: str) -> str:
    """A line typed after ``prompt``, which is put on the terminal the person answers at: standard output where that is a
    terminal, else standard error where that is one. ``input`` puts its prompt on standard output (a pipe, when the
    output is piped and only the person's input is a terminal: the question went into the pipe, unseen) or, with both
    ends terminals, on standard error (a file, when that is redirected). Where neither is a terminal the question goes
    where it always went, to ``input``."""
    stream = next((each for each in (sys.stdout, sys.stderr) if _tty(each)), None)
    if stream is None:
        return input(prompt)
    stream.write(prompt)
    stream.flush()
    return input()


def _ask(question: str) -> bool:
    try:
        return _prompted(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


class _PlainTerminalAsker(PlainAsker):
    """The download gates' questions where the display draws nothing (``--plain``, ``NO_COLOR``, ``TERM=dumb``, standard
    output piped) and a person can still answer: the library's plain lines and prompt, and no line saying what the
    download takes, which is new output a plain command line never had."""

    @override
    def line(self, text: str) -> None:
        return None


@contextmanager
def _gate_questions(voice: _Voice | None = None) -> Generator[None]:
    """Who asks the download gates' questions (``mime.assets``: a licence's conditions, the size of a large download) for
    this command: the display, on a terminal it draws on (``voice``'s own for ``pull``, so that its blocks keep their
    order); the library's plain lines where a person can answer but nothing is drawn; and nobody where there is no one to
    answer, so that the gate refuses and names the flag. The questions and the refusals are the library's; this only
    chooses the voice."""
    from mime import assets

    if not assets.can_ask():
        with assets.asking(None):
            yield
        return
    if voice is not None and voice.display is not None:
        with assets.asking(voice.display):
            yield
        return
    if styled(sys.stdout):
        from mime.cli._display import SaidDisplay

        display = SaidDisplay(sys.stdout)
        try:
            with assets.asking(display):
                yield
        finally:
            display.close()
        return
    with assets.asking(_PlainTerminalAsker()):
        yield


def _controlling_terminal() -> io.TextIOBase | None:
    """The terminal this process answers at (``/dev/tty``), open for writing, or ``None`` where it has none."""
    try:
        return open("/dev/tty", "w", encoding="utf-8")
    except OSError:
        return None


@contextmanager
def _setup_questions(voice: _Voice) -> Generator[tuple[Callable[[str], str] | None, Callable[[str], bool] | None]]:
    """Where ``--setup`` asks its questions: through the display's own question path where the person can answer (a
    terminal at standard input) and sees a terminal at standard output or standard error, as every other question; on
    the controlling terminal (``/dev/tty``) where both go to files (``> log 2>&1``), so that no question goes into the
    file to be answered blind; and nowhere (``push.setup``, which names ``--init``) without a terminal to answer at or
    one to show the questions on."""
    if not _a_terminal():
        yield None, None
        return
    if _tty(sys.stdout) or _tty(sys.stderr):
        yield voice.typed, voice.ask
        return
    terminal = _controlling_terminal()
    if terminal is None:
        yield None, None
        return

    def typed(prompt: str) -> str:
        terminal.write(prompt)
        terminal.flush()
        try:
            return input()
        except EOFError:
            return ""

    def ask(question: str) -> bool:
        return typed(f"{question} [y/N] ").strip().lower() in ("y", "yes")

    try:
        yield typed, ask
    finally:
        terminal.close()


def _typed(prompt: str) -> str:
    try:
        return _prompted(prompt)
    except EOFError:
        return ""


#: The past tense of what a progress display is doing, for its overall line (``2/2 files uploaded``).
_PAST = {"uploading": "uploaded", "downloading": "downloaded"}


class _Voice:
    """What ``mime push`` and ``mime pull`` say, ask and show progress with. On a terminal not asked to be plain, a display
    draws the plan, the publish step, the permissions and the site edits, and the files' progress is drawn on standard
    error; anywhere else each is what it was before the display existed (:func:`_say`, :func:`_ask`, :func:`_typed`, no
    progress). Each stream is judged alone, so ``mime push ... | tee`` still draws progress, and its lines are the
    pipe's.

    A fault in the display never changes what the command does: the lines fall back to :func:`_say`, and the progress
    switches itself off (:class:`mime.cli._display.TransferDisplay`)."""

    def __init__(self, verb: str) -> None:
        self.say: Callable[[str], None] = _say
        self.ask: Callable[[str], bool] = _ask
        self.typed: Callable[[str], str] = _typed
        self.progress: Transfers | None = None
        self.display: SaidDisplay | None = None
        self._open: list[Callable[[], None]] = []
        self._abandoned = False
        transfers = None
        if styled(sys.stderr):
            from mime.cli._display import TransferDisplay

            transfers = TransferDisplay(sys.stderr, verb, _PAST[verb])
            self.progress = transfers
            self._open.append(transfers.finish)
        if styled(sys.stdout):
            from mime.cli._display import SaidDisplay

            drawing = SaidDisplay(sys.stdout, transfers)
            self.display = drawing
            self.say, self.ask, self.typed = self._drawn(drawing.say), drawing.ask, drawing.typed
            self._open.insert(0, drawing.close)

    def _drawn(self, draw: Callable[[str], None]) -> Callable[[str], None]:
        def say(line: str) -> None:
            if not self._abandoned:
                try:
                    draw(line)
                    return
                except Exception:  # noqa: BLE001 - a display that cannot draw says its lines the old way, from here on
                    self._abandoned = True
            _say(line)

        return say

    def close(self) -> None:
        for close in self._open:
            try:
                close()
            except Exception:  # noqa: BLE001 - closing a display is never why a command fails
                pass


#: The variable ``mime pull --link env`` reads a record's share link from.
SHARE_LINK_VARIABLE = "MIME_SHARE_LINK"


def _pull_source(arguments: argparse.Namespace) -> str | Token:
    """What ``mime pull`` pulls: the source as given, or a share link read by ``--link`` (with no echo, or from
    :data:`SHARE_LINK_VARIABLE`) into a :class:`~mime.credentials.Token` the moment it is read. A ``--link`` that is
    neither is refused without its value: it may be the link itself."""
    from mime.credentials import Token
    from mime.push._records import SHARE_LINK

    parser: argparse.ArgumentParser = arguments.pull_parser
    if arguments.link is None:
        if arguments.source is None:
            parser.error("the following arguments are required: source")
        return arguments.source
    if arguments.source is not None:
        parser.error("--link reads a share link in place of the source: give one or the other")
    if arguments.link not in ("-", "env"):
        parser.error(f"--link takes - (the link asked for with no echo) or env (the link read from "
                     f"{SHARE_LINK_VARIABLE}); a link itself is the source, in quotes")
    if arguments.link == "env":
        held = Token(SHARE_LINK, os.environ.get(SHARE_LINK_VARIABLE, "").strip())
        what = f"{SHARE_LINK_VARIABLE} holds no share link"
    else:
        import getpass

        try:
            held = Token(SHARE_LINK, getpass.getpass("share link (not shown): ").strip())
        except EOFError:
            held = Token(SHARE_LINK, "")
        what = "no share link was typed"
    if held.length == 0:
        raise BuildRefused([Refusal("pull.source", RefusalClass.USER, f"--link {arguments.link}: {what}; nothing was "
                                                                      f"sent",
                                    f"`mime pull --link -`, then paste the link the record's page gives; or set "
                                    f"{SHARE_LINK_VARIABLE} to it, then `mime pull --link env`")])
    return held


def _pull(source: str | Token, into: Path | None, no_assets: bool, yes: bool, max_bundle_bytes: int,
          from_service: str | None = None, signed_in: bool = False, force: bool = False,
          accept_licences: Sequence[str] = ()) -> int:
    from mime import assets, saved

    voice = _Voice("downloading")
    try:
        with (assets.watching_downloads(voice.progress) if voice.progress is not None else nullcontext(),
              _gate_questions(voice)):
            saved.pull(source, into=into, fetch_assets=not no_assets, yes=yes,
                       confirm=voice.ask if _a_terminal() else None, say=voice.say, max_bundle_bytes=max_bundle_bytes,
                       from_service=from_service, signed_in=signed_in, force=force, accept_licences=accept_licences)
    finally:
        voice.close()
    return 0


def _push_arguments(arguments: argparse.Namespace) -> None:
    """What argparse cannot require of ``mime push`` once ``--defaults`` (no folder, no service) and ``--setup`` (no
    service) are among its modes: the folder and ``--to`` everywhere else, said as argparse says a missing argument."""
    from mime import push

    parser: argparse.ArgumentParser = arguments.push_parser
    if arguments.defaults or arguments.setup:
        ignored = [flag for flag, given in (("--confirm", arguments.confirm is not None), ("--yes", arguments.yes),
                                            (push.FULL_FLAG, arguments.full_plan),
                                            (push.KEEP_FLAG, arguments.keep_site_edits),
                                            (push.OVERWRITE_FLAG, arguments.overwrite_site_edits)) if given]
        if ignored:
            what = ("--defaults shows your own sharing defaults" if arguments.defaults else
                    "--setup asks for sharing.yaml's fields")
            parser.error(f"{what}, and pushes and publishes nothing: it takes no {', '.join(ignored)}")
    if arguments.defaults:
        if arguments.folder is not None or arguments.to is not None:
            parser.error("--defaults shows your own sharing defaults, and takes no folder and no --to")
        return
    missing = [name for name, held in (("folder", arguments.folder), ("--to", arguments.to))
               if held is None and not (name == "--to" and arguments.setup)]
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")


def _push(arguments: argparse.Namespace) -> int:
    from mime import push
    from mime.push._sharing import DEFAULT_LICENCE

    _push_arguments(arguments)
    if arguments.defaults:
        for line in push.defaults_lines():
            print(for_terminal(line))
        return 0
    folder: Path = arguments.folder
    if arguments.setup:
        voice = _Voice("uploading")
        with _setup_questions(voice) as (typed, ask):
            try:
                push.setup(folder, typed=typed, ask=ask, say=voice.say, service=arguments.to)
            finally:
                voice.close()
        return 0
    service = push.SERVICES[arguments.to]
    voice = _Voice("uploading")
    ask = voice.ask if _a_terminal() else None
    if arguments.confirm is not None and not arguments.publish:
        raise BuildRefused([Refusal("push.not-confirmed", RefusalClass.USER,
                                    "--confirm names the draft a publish publishes, and this is not a publish",
                                    f"`mime push {quoted(str(folder))} --to {service.name} --publish --confirm <DOI>`")])
    if arguments.init:
        path, written, licence = push.init(folder)
        print(for_terminal(f"wrote {path}" if written else f"kept {path}, which exists: edit it, or delete it and "
                                                          f"run --init again"))
        print(for_terminal(f"licence: {licence}, which the experiment declares" if licence else
                           f"licence: {DEFAULT_LICENCE}, pre-filled: the experiment declares none, and push asks it to "
                           f"declare the same, exp.licence(\"{DEFAULT_LICENCE}\"), then save"))
        filled = list(push.read_defaults()) if written else []
        if filled:
            print(for_terminal(f"filled from your sharing defaults ({push.defaults_path()}): {', '.join(filled)}"))
        print(for_terminal(f"next: check what your defaults filled in it, then `mime push {quoted(str(folder))} --to "
                           f"{service.name}`" if filled else
                           f"next: name the creators in it, then `mime push {quoted(str(folder))} --to {service.name}`"))
        return 0
    try:
        if arguments.status:
            push.status(folder, service, say=voice.say)
        elif arguments.discard:
            push.discard(folder, service, yes=arguments.yes, confirm=ask, say=voice.say)
        elif arguments.publish:
            push.publish(folder, service, confirm=arguments.confirm, ask=ask,
                         typed=voice.typed if ask is not None else None, say=voice.say,
                         overwrite_site_edits=arguments.overwrite_site_edits,
                         keep_site_edits=arguments.keep_site_edits, full=arguments.full_plan)
        else:
            push.push(folder, service, yes=arguments.yes, confirm=ask, say=voice.say,
                      overwrite_site_edits=arguments.overwrite_site_edits, keep_site_edits=arguments.keep_site_edits,
                      full=arguments.full_plan, progress=voice.progress)
    finally:
        voice.close()
    return 0


#: The words of ``mime credentials``: a verb, then a service where the verb takes one.
_CREDENTIAL_VERBS = ("set", "status", "remove")
_CREDENTIAL_USAGE = ("usage: mime credentials set|status|remove <zenodo|zenodo-sandbox|host:<host name>>\n"
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
    if len(after) == 2 and after[1] not in credentials.SERVICES and not credentials.is_host_service(after[1]):
        return f"the service is one of {', '.join(credentials.SERVICES)}, or host:<host name>"
    return None


class _Services(list[str]):
    """The services ``mime credentials`` takes, as argparse's choices: the named ones listed (for help and completion),
    and a host's own token, ``host:<host name>``, accepted too."""

    @override
    def __contains__(self, value: object) -> bool:
        from mime import credentials

        return super().__contains__(value) or (isinstance(value, str) and credentials.is_host_service(value))


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


def _docs_check(root: Path, base: str | None, hash_seeds: bool, evidence: bool = True,
                example_runs: bool = True) -> int:
    try:
        from mime.docs._check import run_check
    except ImportError as error:
        if str(error) != VERIFY_EXTRA_MISSING:
            raise
        print(f"mime docs check: {error}", file=sys.stderr)
        return 2
    return run_check(root, base, hash_seeds=hash_seeds, evidence=evidence, example_runs=example_runs)


def _docs_build(root: Path, out: Path | None, junit: Path | None = None, collection: Path | None = None,
                skip_evidence: bool = False) -> int:
    try:
        from mime.docs._build import run_build
    except ImportError as error:
        if str(error) != VERIFY_EXTRA_MISSING:
            raise
        print(f"mime docs build: {error}", file=sys.stderr)
        return 2
    return run_build(root, out, junit=junit, collection=collection, skip_evidence=skip_evidence)


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
    if not (styled(sys.stderr) and _refusal_block(coded, headline)):
        print(short_text(coded, None, frames=False, switch=TRACEBACK_FLAG, headline=headline), end="", file=sys.stderr)
    plain = all(r.category in (RefusalClass.USER, RefusalClass.ENVIRONMENT) for r in coded.refusals)
    return REFUSED if plain else FAILED


def _refusal_block(error: MimeError, headline: str) -> bool:
    """The refusal drawn as a block on a terminal: its code, what is wrong, what to do, why and where, then its docs
    address and the footer the plain text ends with (whose defect it is, and the file the full traceback is in). ``False``
    when it could not be drawn, and the plain text is printed instead."""
    from mime.cli._display import show_refusal
    from mime.errors._display import footer_lines

    try:
        show_refusal(error, headline, footer_lines(error, switch=TRACEBACK_FLAG), sys.stderr)
    except ImportError:
        return False
    return True


def _survive_a_narrow_stream() -> None:
    """Python before 3.15 encodes a redirected stream on Windows in the ANSI code page, which cannot hold a minus sign or
    a Greek letter in a path, and a report that did would raise after the work was done. The stream keeps its
    encoding; a character it cannot encode is written as its escape."""
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(errors="backslashreplace")


def _gate_arguments(command: argparse.ArgumentParser, *, yes_help: str | None = None) -> None:
    """The three options of a command that downloads (``mime.assets``' gates): ``--accept-licence <id>`` to accept the
    conditions of one licence by name (repeat it for more), ``-y/--yes`` to answer the question asked of a large
    download, and ``-f/--force`` to download though it would leave the drive almost full. Neither of the last two accepts
    a licence: no generic option does."""
    command.add_argument("--accept-licence", action="append", default=[], metavar="ID",
                         help="accept the conditions of this licence, by its id, without being asked (repeat it for "
                              "more); the only way to accept one where there is no terminal to ask at")
    command.add_argument("-y", "--yes", action="store_true",
                         help=yes_help or "answer yes to the question a large download is asked (over 1 GB, or a "
                                          "tenth of the free space); it never accepts a licence")
    command.add_argument("-f", "--force", action="store_true",
                         help="download though it would leave under 2 GB or 5%% of the drive free; it never accepts "
                              "a licence")


def build_parser() -> _guide.MimeParser:
    """The ``mime`` command's parser: every command, grouped, with its example."""
    parser = _guide.MimeParser(prog="mime", epilog=_ROOT_EXAMPLES,
                               description="The mime command, a thin layer over the Python API: each command is one "
                                           "call a script could make.")
    _guide.MimeParser.top = parser
    parser.add_argument(TRACEBACK_FLAG, action="store_true",
                        help="print a refused command's full traceback, every frame and chained error (it may come "
                             "anywhere on the line)")
    parser.add_argument(PLAIN_FLAG, action="store_true",
                        help="draw nothing but plain text on a terminal, as when output is piped (NO_COLOR and "
                             "TERM=dumb ask the same); it may come anywhere on the line")
    parser.add_argument("-V", "--version", action="version", version=f"mime {__version__}",
                        help="print MIME's version and exit")
    commands = parser.add_subparsers(dest="command", required=False)
    verify = commands.add_parser("verify", help="load, rebuild, check the hash, and rerun against the reference "
                                                "outputs; it keeps no XLA compilation cache, compiling as a fresh "
                                                "process would",
                            description="Load a saved experiment, rebuild it, check it against its lock and rerun "
                                        "its reference steps. It keeps no XLA compilation cache, whatever mime run "
                                        "does: a verification compiles as a fresh process would.")
    verify.add_argument("folder", type=Path, help="a folder written by Experiment.save, or with --run a run folder")
    verify.add_argument("--run", action="store_true",
                        help="the folder is a run folder (mime run's --out, or run_to's): make its run again from its "
                             "experiment/ with the overrides, duration and series its run.json records, and compare it "
                             "with its outputs, bit for bit in the environment it was made in")
    pending = commands.add_parser("pending", help="what a saved experiment leaves pending, with the lines to paste "
                                                  "to seal it")
    pending.add_argument("folder", type=Path, help="a folder written by Experiment.save, or its document "
                                                   "(experiment.mime.yaml), edited, to check it as it is")
    show = commands.add_parser("show", help="what a saved experiment is made of: its topology as a tree, and the "
                                             "graph it compiles to as Mermaid text, without running it")
    show.add_argument("folder", type=Path, help="a folder written by Experiment.save, or its document "
                                                "(experiment.mime.yaml)")
    show.add_argument("--tree", action="store_true", help="the topology as declared: structures, media, bodies and "
                                                          "their parts, apparatus, each law with what it binds, "
                                                          "observations and the controller (the default)")
    show.add_argument("--graph", action="store_true", help="the MADDENING graph the experiment compiles to, as "
                                                           "Mermaid text; it builds the experiment")
    show.add_argument("--json", action="store_true", help="the tree as JSON (the mime.tree format)")
    show.add_argument("--out", type=Path, nargs="?", const=DEFAULT_PAGE, default=None, metavar="FILE",
                      help="write the graph as one page to pan and zoom (Mermaid inlined, so it needs no network) "
                           "instead of printing it (default name: <experiment>.graph.html here, never the 3-D page's); "
                           "it builds the experiment, and is never written inside a saved or run folder. Name the "
                           "folder before --out: a word after it is taken as the file")
    show.add_argument("--cdn", action="store_true",
                      help="with --out, load Mermaid from jsDelivr, held to its hash, rather than inline it (3.6 MB "
                           "smaller; the page then needs the network)")
    show.add_argument("--caption", default=None, metavar="TEXT", help="with --out, a line shown above the diagram")
    show.add_argument("--open", action="store_true", help="with --out, open the page in the default browser")
    registry = commands.add_parser("registry", help="every registered entry and what it declares")
    registry.add_argument("--json", action="store_true", help="the whole export, as JSON")
    # A skip flag of the documentation's checks is only ever spelled in full: argparse would take ``--skip-ex`` for
    # ``--skip-example-runs``, and a guard that reads workflows for the full spelling would not see it.
    docs = commands.add_parser("docs", help="the documentation's sources", allow_abbrev=False)
    docs_commands = docs.add_subparsers(dest="docs_command", required=True)
    check = docs_commands.add_parser("check", help="every check of the documentation's sources", allow_abbrev=False)
    check.add_argument("--root", type=Path, default=Path.cwd(), help="the checkout to check (default: here)")
    check.add_argument("--base", metavar="REVISION", help="fail on a README start-list line this revision's list "
                                                         "does not hold (CI passes origin/release/v0.3.0)")
    check.add_argument("--skip-hash-seed-builds", action="store_true",
                       help="do not build the pages in fresh processes under eight hash seeds (the tests of "
                            "the other checks pass it to stay fast; CI never does)")
    check.add_argument("--skip-evidence", action="store_true",
                       help="do not collect the test suite to resolve the evidence the specification and the anomaly "
                            "registry cite (the tests of the other checks pass it, on copies of the checkout; CI "
                            "never does)")
    check.add_argument("--skip-example-runs", action="store_true",
                       help="do not run the scripts the guides' output blocks name to regenerate those blocks (the "
                            "platform smoke job passes it: the printed numbers are a Linux run's; CI's docs check "
                            "never does)")
    build = docs_commands.add_parser("build", help="generate the pages (never committed) from the registry and the "
                                                   "hand-written sources", allow_abbrev=False)
    build.add_argument("--root", type=Path, default=Path.cwd(), help="the checkout to build from (default: here)")
    build.add_argument("--out", type=Path, help="the directory to write index.md and pages/ to (default: "
                                                "<root>/docs/_build)")
    build.add_argument("--junit", type=Path, metavar="DIRECTORY",
                       help="the default lane's JUnit reports (every .xml under this directory): the evidence "
                            "counts page reads each cited test's outcome there; without it every row prints "
                            "'outcome not read (no lane report)' (CI passes it)")
    build.add_argument("--collection", type=Path, metavar="FILE",
                       help="a collection of the test suite, as mime docs check writes it (JSON), to resolve the "
                            "evidence in (the hash-seed builds of mime docs check pass it; without it the files the "
                            "evidence names are collected)")
    build.add_argument("--skip-evidence", action="store_true",
                       help="do not collect the tests the evidence counts page cites: every row reads 'not read' (the "
                            "tests of the other pages pass it; CI never does)")
    from mime import assets

    held = commands.add_parser(assets.GROUP, help="large files referenced by sha256 and fetched on request")
    held_commands = held.add_subparsers(dest="assets_command", required=True)
    held_commands.add_parser(assets.LIST, help="every known asset, cached or missing, with its size")
    fetch = held_commands.add_parser(assets.FETCH, help="store assets in the cache (the cache, then the primary location, then backups), then hash "
                                   "them in full")
    fetch.add_argument("names", nargs="*", metavar="name-or-folder",
                       help="an asset's name, or an experiment folder: exactly the assets its lock references")
    fetch.add_argument("--all", action="store_true", help="every known asset")
    _gate_arguments(fetch)
    held_commands.add_parser(assets.CHECK, help="re-hash every cached asset in full")
    licences = held_commands.add_parser(assets.LICENCES, help="the licences you have accepted on this machine "
                                                              "(downloads under one ask once); --revoke removes one")
    licences.add_argument("--revoke", metavar="ID", help="forget your acceptance of this licence, so that the next "
                                                          "download under it asks again")
    sources = held_commands.add_parser(assets.SOURCES, help="each asset's terms, tier and locations, each looked at "
                                                            "now (reachable or not, and why), downloading nothing")
    sources.add_argument("names", nargs="+", metavar="name-or-folder",
                         help="an asset's name, a saved folder (its lock's assets and its meshes' manifests), a "
                              "folder holding visuals.json, or a manifest file")
    audit = held_commands.add_parser(assets.AUDIT, help="find in a wheel, an sdist, an archive or a directory any "
                                                        "file whose bytes may not be redistributed, and any mesh "
                                                        "with no record (exit 1 when anything is found)")
    audit.add_argument("targets", nargs="+", metavar="wheel-sdist-or-directory",
                       help="a wheel, an sdist, a zip or tar archive, or a directory")
    audit.add_argument("--manifest", action="append", default=[], type=Path, metavar="FILE",
                       help="a manifest whose records count too (repeat it); the target's own manifests and locks, "
                            "the installed sources' and the local cache's baked meshes always count")
    pack = commands.add_parser("pack", help="write a saved experiment folder as one bundle file")
    pack.add_argument("folder", type=Path, help="a folder written by Experiment.save")
    pack.add_argument("--out", type=Path, help="the bundle file (default: <folder>.mimebundle beside the folder)")
    card = commands.add_parser("card", help="a saved experiment's card (Markdown), or its NOTICE, offline, as mime push "
                                            "will publish them (the card also names the experiment it was based on)")
    card.add_argument("folder", type=Path, help="a folder written by Experiment.save")
    card.add_argument("--out", type=Path, help="write it to this file (default: print it)")
    card.add_argument("--notice", action="store_true", help="the NOTICE (each file's licence, credit, verdict and "
                                                            "declared permission) in place of the card")
    from mime import credentials, push

    pull = commands.add_parser("pull", help="get a bundle by DOI, address or file, unpack it into a new folder, and "
                                            "fetch its assets")
    pull.set_defaults(pull_parser=pull)
    pull.add_argument("source", nargs="?",
                      help="a DOI (10.5281/zenodo.1234567; a concept DOI pulls the latest version), an https:// "
                           "address, a bundle file, a record's share link (in quotes; --link keeps it out of sight), "
                           "or with --from a record's number")
    pull.add_argument("--link", metavar="{-,env}",
                      help="a record's share link, read in place of the source: `--link -` asks for it with no echo, "
                           f"`--link env` reads it from {SHARE_LINK_VARIABLE}. A link given as the source stays in "
                           "your shell's history and, while pull runs, in its arguments, which any account on this "
                           "machine can read (ps, /proc/<pid>/cmdline) unless /proc is mounted with hidepid")
    pull.add_argument("--from", dest="from_service", choices=list(push.SERVICES), metavar="SERVICE",
                      help="read the published record of this number at the service's records API (zenodo-sandbox: "
                           "its DOIs do not resolve through doi.org)")
    pull.add_argument("--signed-in", action="store_true",
                      help="with --from, send the token `mime credentials` stored for that service, for a record or "
                           "files shared with your account (without it, pull sends no token)")
    pull.add_argument("--into", type=Path, help="the new folder (default: ./<the experiment's name>/)")
    pull.add_argument("--no-assets", action="store_true", help="unpack only; fetch the assets later with "
                                                               "`mime assets fetch <folder>`")
    _gate_arguments(pull, yes_help="download assets over the size MIME asks about without asking (it never accepts a "
                                    "licence)")
    pull.add_argument("--max-bundle-bytes", type=int, default=assets.RETRIEVE_LIMIT_BYTES, metavar="BYTES",
                      help="the most a bundle may be when nothing states its size (default: "
                           f"{assets.size_text(assets.RETRIEVE_LIMIT_BYTES)})")
    pushing = commands.add_parser(
        "push", help="show the plan, then make or update the experiment's draft record on Zenodo and upload its files "
                     "(nothing is published)",
        usage="%(prog)s [-h] --to {zenodo,zenodo-sandbox} [--init | --setup | --status | --discard | --publish]\n"
              "                 [--confirm DOI[@FINGERPRINT]] [--yes] [--full-plan]\n"
              "                 [--keep-site-edits | --overwrite-site-edits]\n"
              "                 folder\n"
              "       %(prog)s folder --setup [--to {zenodo,zenodo-sandbox}]\n"
              "       %(prog)s --defaults")
    pushing.set_defaults(push_parser=pushing)
    pushing.add_argument("folder", type=Path, nargs="?", help="a folder written by Experiment.save")
    pushing.add_argument("--to", choices=list(push.SERVICES),
                         help="the service: zenodo, or zenodo-sandbox to try it on Zenodo's test service (required, "
                              "but for --setup and --defaults)")
    pushing_mode = pushing.add_mutually_exclusive_group()
    pushing_mode.add_argument("--init", action="store_true", help="write sharing.yaml, the record's credit and "
                                                                  "description, from what the folder knows and your "
                                                                  "sharing defaults")
    pushing_mode.add_argument("--setup", action="store_true",
                              help="make or edit sharing.yaml a field at a time, in a terminal: each field explained, "
                                   "Enter skips it or keeps the value shown; then save your defaults if you like")
    pushing_mode.add_argument("--defaults", action="store_true",
                              help="show your sharing defaults (your name, ORCID iD, affiliation and the rest you "
                                   "repeat from record to record) and where they live; takes no folder")
    pushing_mode.add_argument("--status", action="store_true", help="the draft as the service holds it, against what "
                                                                    "push uploaded")
    pushing_mode.add_argument("--discard", action="store_true", help="delete the draft (only a draft can be deleted)")
    pushing_mode.add_argument("--publish", action="store_true",
                              help="publish the draft, the one step that cannot be undone: the plan, the confirmation "
                                   "in words, each declared permission asked, then the record number typed")
    pushing.add_argument("--confirm", metavar="DOI[@FINGERPRINT]",
                         help="with --publish, from a script: the draft's reserved DOI, in place of the record number "
                              "typed (it must be that draft's); once a push changed the draft after first printing its "
                              "DOI, the DOI and the fingerprint the last push printed beside it")
    pushing.add_argument("--yes", action="store_true", help="delete the draft with --discard without asking (it never "
                                                            "publishes, declares a permission or settles a site edit)")
    pushing.add_argument(push.FULL_FLAG, dest="full_plan", action="store_true",
                         help="the full plan: every file inside the bundle with its tier and verdict, every field sent")
    site_edits = pushing.add_mutually_exclusive_group()
    site_edits.add_argument(push.KEEP_FLAG, dest="keep_site_edits", action="store_true",
                         help="keep each edit made on the service's site that sharing.yaml can hold, written into it, "
                              "without asking")
    site_edits.add_argument(push.OVERWRITE_FLAG, dest="overwrite_site_edits", action="store_true",
                         help="write the draft's metadata from sharing.yaml and the folder over what was edited on the "
                              "service's site since push last wrote it (without it, such an edit stops the push)")
    held_credentials = commands.add_parser("credentials", help="the access token mime push will use, kept where a "
                                                               "commit cannot publish it")
    credential_commands = held_credentials.add_subparsers(dest="credentials_command", required=True)
    for verb, what in (("set", "read a token (hidden) and store it"), ("remove", "delete the stored token"),
                       ("status", "whether there is a token, where from, and its last four characters, if 32 or more")):
        each = credential_commands.add_parser(verb, help=what)
        each.add_argument("service", nargs="?" if verb == "status" else None, choices=_Services(credentials.SERVICES))
    held_cache = commands.add_parser("cache", help="the XLA compilation cache mime run and mime serve keep")
    cache_commands = held_cache.add_subparsers(dest="cache_command", required=True)
    cache_commands.add_parser("clear", help="delete the compilation cache in the user's cache directory")
    no_cache_help = ("keep no XLA compilation cache: compile everything afresh (by default one is kept in the user's "
                     "cache directory, capped in size, so that a second run of an experiment compiles less)")
    run = commands.add_parser("run", help="load a saved experiment, run it, and save the run beside its outputs")
    run.add_argument("--no-xla-cache", action="store_true", help=no_cache_help)
    run.add_argument("folder", type=Path, help="a folder written by Experiment.save")
    run.add_argument("--duration", default=None, metavar="DURATION",
                     help=f"the simulated time, a whole number of the experiment's base steps ({_durations.FORMS}); "
                          "the length the saved experiment declares when not given (required, with --commands "
                          "the exception, when it declares none)")
    run.add_argument("--out", type=Path, required=True, help="the folder to write the outputs, the record and the "
                                                             "run's saved form to")
    run.add_argument("--history", action="store_true", help="write every step's fields, not only the last")
    run.add_argument("--record", action="append", default=[], metavar="OUTPUT",
                     help="a per-step output to record as a series, <observation id>.<output> or an observation's id "
                          "for all of its own (repeat it for more)")
    run.add_argument("--every", default=None, metavar="DURATION",
                     help="the spacing of the recorded series' samples, a whole number of base steps (default: every "
                          "step); a unit as for --duration")
    run.add_argument("--stage-every", default=None, metavar="DURATION",
                     help="write the recording layer (recording.usdc, over the run's own stage) with a sample every "
                          "DURATION, a whole number of the run's chunks; a unit as for --duration")
    run.add_argument("--full", action="store_true",
                     help="on a terminal, also print the load report, the build report's summary and every reading "
                          "(the summary leads with what needs attention and keeps the rest back; piped output is "
                          "always whole, and so are the files)")
    run.add_argument("--override", action="append", default=[], metavar="PATH=VALUE",
                     help="a value this run declares for itself, as a study's axis names it: field.commands.frequency=20Hz "
                          "(repeat it for more); run.json records each, and every output is stamped overridden")
    run.add_argument("--commands", type=Path, default=None, metavar="LOG",
                     help="replay a live session from the command log it wrote (commands.jsonl): its run folder again, "
                          "in place of --duration")
    held_visuals = commands.add_parser("visuals", help="descriptions' meshes MIME does not ship, baked into the asset "
                                                       "cache, and the overlay that draws them on a stage")
    visual_commands = held_visuals.add_subparsers(dest="visuals_command", required=True)
    visual_commands.add_parser("list", help="every visual set MIME has a recipe for, baked or not, with its licence")
    visuals_fetch = visual_commands.add_parser("fetch", help="download a description's meshes from their pinned "
                                                             "upstream, check each file's sha256, and bake them into "
                                                             "the asset cache")
    visuals_fetch.add_argument("set", metavar="description-or-folder",
                               help="a description's folder (its visuals.json), a description or manifest file, a "
                                    "saved or run folder (the manifests its lock holds), or a set's name from mime "
                                    "visuals list")
    _gate_arguments(visuals_fetch)
    visuals_view = visual_commands.add_parser("view", help="the layer that opens a stage, a recording or a folder's "
                                                           "with the baked meshes drawn, for any USD viewer")
    visuals_view.add_argument("target", type=Path, help="a stage or recording file, a run folder or a saved folder")
    visuals_view.add_argument("--accept-licence", action="append", default=[], metavar="ID",
                              help="accept the conditions of this licence, by its id, without being asked (repeat it "
                                   "for more): a baked mesh is drawn only once its licence is accepted, and a mesh "
                                   "whose licence is not is left out where there is no terminal to ask at")
    visuals_web = visual_commands.add_parser("web", help="one web page of a stage, a recording or a folder's, to "
                                                         "open in a browser or share: the scene, its recording and "
                                                         "the meshes whose terms let them be shared")
    visuals_web.add_argument("target", type=Path, help="a stage or recording file, a run folder or a saved folder")
    visuals_web.add_argument("--out", type=Path, default=None, metavar="FILE",
                             help="the page to write (default: <the experiment's name>.html here); never inside a "
                                  "saved or run folder")
    visuals_web.add_argument("--glb", type=Path, default=None, metavar="FILE",
                             help="also write the scene alone as a glTF binary file")
    visuals_web.add_argument("--slow-motion", type=float, default=None, metavar="FACTOR",
                             help="display seconds per simulated second (default: a recording shorter than 4 s "
                                  "slowed to about 8 s)")
    visuals_web.add_argument("--caption", default=None, metavar="TEXT", help="a line shown above the page's notes")
    visuals_web.add_argument("--cdn", action="store_true",
                             help="load the viewer from jsDelivr, held to its hash, rather than inline it (1.4 MB "
                                  "smaller; the page then needs the network)")
    visuals_web.add_argument("--no-scale-bar", action="store_true",
                             help="leave out the scale bar (a round length in SI units that follows the zoom; on by "
                                  "default)")
    visuals_web.add_argument("--no-legend", action="store_true",
                             help="leave out the arrow legend (what each arrow is, its colour, its scale and its cap; "
                                  "on by default)")
    visuals_web.add_argument("--no-apparatus-shapes", action="store_true",
                             help="leave out the motor and magnet an apparatus's frames stand for, which the view "
                                  "draws for itself and never writes into the stage (on by default)")
    visuals_web.add_argument("--focus", default=None, metavar="BODY",
                             help="frame the named body at the start (off by default; an unknown name is refused "
                                  "with the nearest)")
    visuals_web.add_argument("--follow", default=None, metavar="BODY",
                             help="keep the camera on the named body through the recording (off by default)")
    visuals_web.add_argument("--open", action="store_true", help="open the page in the default browser")
    from mime.live import DEFAULT_COMMANDS, DEFAULT_FRAMES

    serve = commands.add_parser("serve", help="serve a saved experiment's live session (mime.live/1, over ZMQ)")
    serve.add_argument("folder", type=Path, help="a folder written by Experiment.save")
    serve.add_argument("--no-xla-cache", action="store_true", help=no_cache_help)
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
    serve.add_argument("--idle-timeout", default=None, metavar="DURATION",
                       help="stop, writing the run folder, when no client has been connected that long (10min, 2h; "
                            "default: never). Like --max-wall, it stops at the end of the chunk in hand")
    serve.add_argument("--max-wall", default=None, metavar="DURATION",
                       help="stop, writing the run folder, when the session has run that long, wall time from when "
                            "it opened (default: never). The chunk in hand is finished first, as for SIGTERM: the "
                            "stop comes at the end of the chunk in hand, later than the limit by what is left of it "
                            "(the first compiles)")
    serve.add_argument("--watch", action="append", default=[], metavar="PATH",
                       help="a row of the live view's table of values, <node>.<field> or <node>.<field>[0:3] (repeat "
                            "it; all, or none to hide the table; default: each scalar output, each flag and each "
                            "body's position). It chooses what is shown and adds nothing to the frames")
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
    completion = commands.add_parser("completion", help="print a completion script for bash, zsh or fish",
                                     epilog="install:\n" + "\n".join(
                                         f"  {shell}: {_completion.INSTALL[shell]}{_completion.WHERE[shell]}"
                                         for shell in _completion.SHELLS))
    completion.add_argument("shell", choices=_completion.SHELLS, help="the shell to write the script for")
    _guide.attach_examples(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """``mime verify <folder>``: load a saved experiment, rebuild it, check it against its lock, and rerun its
    reference steps. Exits 0 when reproduced, 1 when an output moved, 2 when the load or build is refused.

    ``mime pending <folder>``: load and rebuild a saved experiment, and print each item its profile leaves pending (a
    waiver, a reason, a numerical default), in plain words with the lines to paste in Python and in the document, and
    each it lists. ``mime pending <folder>/experiment.mime.yaml`` plans the document as it is, with no lock: the check
    of a document edited to seal it, which ``Experiment.from_document`` builds. Exits 0 when nothing is pending, 1 when
    something is, 2 when refused, 3 on a MIME defect or an error MIME does not recognise.

    ``mime cache clear``: delete the XLA compilation cache ``mime run`` and ``mime serve`` keep in the user's cache
    directory. They keep it unless ``--no-xla-cache``, with the oldest entries pruned at their start once it holds more
    than its cap; ``mime verify`` never uses it.

    ``mime completion bash|zsh|fish``: a completion script for the shell, written to standard output from this parser
    (every command, flag and choice in it), which runs no Python when Tab is pressed.

    ``mime registry [--json]``: every registered entry, MIME's and each installed namespace's, in a fresh process.

    ``mime docs check [--root <checkout>] [--base <revision>]``: the documentation's sources and their checks, in a
    checkout of MIME; with ``--base``, the README start list against that revision's too. Exits 0 when every check
    passes, 1 when one fails, 2 without the ``verify`` extra.

    ``mime assets list | fetch <name>... | fetch --all | check``: the large files MIME references by sha256, in the
    cache. ``fetch`` stores them (and refuses with 2), ``check`` re-hashes the cached ones in full (1 when one is
    corrupt).

    ``mime run <folder> [--duration <seconds>] --out <dir> [--history] [--record <output>]... [--every <seconds>]
    [--stage-every <seconds>]``: load and rebuild a saved experiment, run it, and write its outputs, their manifest
    (``outputs.json``), its record, its topology, its build report's summary and the run's own saved form into
    ``<dir>``, which ``RunResult.read`` reads back, with the recording layer (``recording.usdc``) where
    ``--stage-every`` asks. The run is as long as the experiment declared (``Experiment(..., duration=...)``) unless
    ``--duration`` says another. ``mime run <folder> --commands <log> --out <dir>`` replays a live session from its command
    log. Exits 0, or 2 when refused, or 130 when Ctrl+C ended it: the first press stops the run at the end of its chunk
    and writes it as the steps reached, marked stopped; the second quits at once, writing nothing.

    ``mime serve <folder> [--commands <endpoint>] [--frames <endpoint>] [--chunk <seconds>] [--pace <factor>]
    [--out <dir>] [--start]``: the saved experiment's live session, speaking ``mime.live/1`` over ZMQ (commands on a REP,
    frames on a PUB), bound to localhost unless ``--remote``, until a ``stop`` command, a signal, or the exit of the
    process that launched it (unless ``--outlive-launcher``), or ``--idle-timeout`` with no client connected, or
    ``--max-wall`` (each at the end of the chunk in hand); it writes a run folder when it stops, and says why. On a terminal it draws a live view on standard
    error (the state, the step, the speed, the clients, a table of values chosen by ``--watch``, and the commands
    clients ask); anywhere else it prints its two lines on standard output and nothing more.

    ``mime env check [<folder>] --variant cpu|cuda12 [--gpu <type>] [--base] [--lab <file>] [--xla-cache <dir>]``:
    hold this environment to MIME's pins and the GPU requested, then apply the precision the folder's lock records.
    Exits 0, or 2 with every difference.

    ``mime modal <folder> [--gpu <type>] [--deterministic] ...``: write the script that runs the saved experiment on
    Modal, beside it. Exits 0, or 2 when refused.

    ``mime visuals list | fetch <set> | view <target> | web <target> [--out <file>] [--glb <file>] [--slow-motion
    <factor>] [--caption <text>] [--cdn] [--open]``: the descriptions' meshes MIME does not ship, fetched from their
    pinned upstreams into the asset cache only; the layer that draws them on a stage or a recording; and one web page
    of it to share, holding only the meshes whose terms let them be shared.

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
    _import_guard.release()
    if argv is None:
        _survive_a_narrow_stream()
    parser = build_parser()
    words = list(sys.argv[1:] if argv is None else argv)
    full = TRACEBACK_FLAG in words
    set_plain(PLAIN_FLAG in words)
    words = [word for word in words if word not in (TRACEBACK_FLAG, PLAIN_FLAG)]
    try:
        return _parsed(parser, words, full)
    finally:
        set_plain(False)  # --plain is for this command line, not for the next call in a process that calls main twice


def _parsed(parser: argparse.ArgumentParser, words: list[str], full: bool) -> int:
    _guide.MimeParser.words = words
    if words[:1] == ["credentials"]:
        problem = _credential_words_problem(words[1:])
        if problem is not None:
            parser.exit(2, f"mime credentials: {problem}\n{_CREDENTIAL_USAGE}\n")
    arguments = parser.parse_args(words)
    if arguments.command is None:
        print(for_terminal(_guide.bare(Path.cwd())))
        return 0
    held = os.environ.get(TRACEBACK_VARIABLE)
    if full:
        os.environ[TRACEBACK_VARIABLE] = "full"
    from mime.cli._display import short_description_warnings

    try:
        with short_description_warnings(sys.stderr):
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
        return _verify(arguments.folder, arguments.run)
    if arguments.command == "pending":
        return _pending(arguments.folder)
    if arguments.command == "show":
        if arguments.json and arguments.graph:
            parser.error("--json is the tree as data; the graph is Mermaid text, so --json and --graph do not go together")
        if arguments.out is not None and (arguments.json or arguments.tree):
            parser.error("--out writes the graph as a page, so it does not go with --json or --tree")
        for flag, given in (("--cdn", arguments.cdn), ("--caption", arguments.caption is not None),
                            ("--open", arguments.open)):
            if given and arguments.out is None:
                parser.error(f"{flag} goes with the page --out writes, and --out is not given")
        return _show(arguments.folder, arguments.graph, arguments.tree, arguments.json, arguments.out, arguments.cdn,
                     arguments.caption, arguments.open)
    if arguments.command == "registry":
        return _registry(arguments.json)
    if arguments.command == "run":
        duration = option_seconds("--duration", arguments.duration)
        every = option_seconds("--every", arguments.every)
        stage_every = option_seconds("--stage-every", arguments.stage_every)
        with _xla_cache(arguments):
            return _run(arguments.folder, duration, arguments.out, arguments.history, arguments.record,
                        every, stage_every, arguments.commands, arguments.full, arguments.override)
    if arguments.command == "serve":
        with _xla_cache(arguments):
            return _serve(arguments)
    if arguments.command == "cache":
        return _cache_clear()
    if arguments.command == "visuals":
        return _visuals(arguments)
    if arguments.command == "completion":
        sys.stdout.write(_completion.script(arguments.shell, parser))
        return 0
    if arguments.command == "env":
        return _env_check(arguments)
    if arguments.command == "modal":
        return _modal(arguments)
    if arguments.command == "docs":
        if arguments.docs_command == "build":
            return _docs_build(arguments.root, arguments.out, arguments.junit, arguments.collection,
                               arguments.skip_evidence)
        return _docs_check(arguments.root, arguments.base, not arguments.skip_hash_seed_builds,
                           not arguments.skip_evidence, not arguments.skip_example_runs)
    if arguments.command == assets.GROUP:
        if arguments.assets_command == assets.SOURCES:
            return _sources(arguments.names)
        if arguments.assets_command == assets.LICENCES:
            return _licences(arguments.revoke)
        if arguments.assets_command == assets.AUDIT:
            reports = [assets.audit(target, manifests=arguments.manifest) for target in arguments.targets]
            for report in reports:
                print(for_terminal(report.text()))
            return 1 if any(report.findings for report in reports) else 0
        if arguments.assets_command == assets.FETCH and bool(arguments.names) == arguments.all:
            parser.error("fetch takes asset names, or --all")
        return _assets(arguments.assets_command, arguments.names if arguments.assets_command == assets.FETCH else (),
                       getattr(arguments, "accept_licence", ()), getattr(arguments, "yes", False),
                       getattr(arguments, "force", False))
    if arguments.command == "pack":
        return _pack(arguments.folder, arguments.out)
    if arguments.command == "card":
        return _card(arguments.folder, arguments.out, arguments.notice)
    if arguments.command == "pull":
        return _pull(_pull_source(arguments), arguments.into, arguments.no_assets, arguments.yes,
                     arguments.max_bundle_bytes, arguments.from_service, arguments.signed_in, arguments.force,
                     arguments.accept_licence)
    if arguments.command == "credentials":
        return _credentials(arguments.credentials_command, arguments.service)
    if arguments.command == "push":
        try:
            return _push(arguments)
        except KeyboardInterrupt:
            print(for_terminal("interrupted: nothing more was sent, and the same command goes on from what the service "
                               "holds"), file=sys.stderr)
            return 130
    parser.error(f"unknown command {arguments.command!r}")


__all__ = ["FAILED", "REFUSED", "TRACEBACK_FLAG", "main", "refused"]
