"""How a MIME error is shown: the script's own frames, then the error in its shape, then where its full chain is.

``MIME_TRACEBACK=full`` shows every frame and every chained error instead, as Python prints them; the CLI's
``--traceback`` sets it for one command. The full chain is written to a file when a short error is shown, under
``$MIME_TRACEBACK_DIR`` or ``tracebacks`` in MIME's cache directory, readable by its user alone, named
``mime-<stamp>-<pid>-<code>.txt``; only the newest :data:`KEPT` files of that name are kept, each write deleting the
older ones, and no other file there is touched."""

from __future__ import annotations

import os
import platform
import re
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Any, cast

from mime.build._refusals import ISSUES, MimeError, RefusalClass
from mime.errors._frames import is_library, walk
from mime.errors._translate import UNCODED

#: The variable that chooses the traceback: ``full`` for every frame and chained error, anything else for the short.
TRACEBACK_VARIABLE = "MIME_TRACEBACK"
FULL = "full"
#: The variable that names the directory the full chains are written to, in place of MIME's cache directory.
DIRECTORY_VARIABLE = "MIME_TRACEBACK_DIR"
#: How many full chains are kept: writing one deletes all but the newest of these.
KEPT = 20
#: The start of a full chain's file name, and the whole name: only files of this name are ever deleted.
PREFIX = "mime-"
NAME = re.compile(r"mime-\d{8}T\d{6}\.\d{9}-\d+-[A-Za-z0-9_.-]+\.txt")


def full_mode() -> bool:
    """Whether errors are shown with every frame and every chained error (``MIME_TRACEBACK=full``)."""
    return os.environ.get(TRACEBACK_VARIABLE, "").strip().lower() == FULL


def traceback_directory() -> Path:
    chosen = os.environ.get(DIRECTORY_VARIABLE)
    if chosen:
        return Path(chosen)
    import pooch

    return Path(str(pooch.os_cache("mime"))) / "tracebacks"


def _environment() -> list[str]:
    from importlib.metadata import PackageNotFoundError, version

    found: list[str] = []
    for name in ("mime", "maddening", "jax", "jaxlib", "numpy", "openmaterica"):
        try:
            found.append(f"{name} {version(name)}")
        except PackageNotFoundError:
            found.append(f"{name} not installed")
    return [f"Python {platform.python_version()} on {platform.platform()}", ", ".join(found)]


def report_text(error: MimeError) -> str:
    """What the traceback file holds: the error in its shape, the environment, and the full chain."""
    lines = [error.text(), "", "environment:", *(f"  {line}" for line in _environment()), "", "full traceback:"]
    lines.append(error.chain or "".join(traceback.format_exception(error)))
    return "\n".join(lines)


def write_traceback(error: MimeError) -> Path | None:
    """Write the error's full chain to a file of its own, keeping the newest :data:`KEPT`, and return its path; or
    ``None`` when the directory cannot be written, which a display does not stop for."""
    if error.traceback_file is not None:
        return Path(error.traceback_file)
    directory = traceback_directory()
    now = time.time_ns()
    # The name orders the files by when they were written, to the nanosecond: a file system's times can tie. A lab's
    # code is written with what a file name can hold on every system.
    stamp = f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime(now // 10**9))}.{now % 10**9:09d}"
    code = re.sub(r"[^A-Za-z0-9_.-]", "_", error.code)
    path = directory / f"{PREFIX}{stamp}-{os.getpid()}-{code}.txt"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        # Readable by its user alone: it may be attached to an issue unread, and holds the script's paths.
        handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with open(handle, "w", encoding="utf-8", newline="\n") as written:
            written.write(report_text(error))
        held = sorted(other for other in directory.glob(f"{PREFIX}*.txt")
                      if other != path and NAME.fullmatch(other.name))
        for old in held[:max(0, len(held) - (KEPT - 1))]:
            old.unlink(missing_ok=True)
    except OSError:
        return None
    error.traceback_file = str(path)
    return path


def _frame_lines(traceback_: TracebackType | None) -> list[str]:
    """The script's own frames, as Python writes them: each frame's file, line and function and the line's source,
    a frame repeated more than three times in a row (a recursion) written once with its count."""
    import linecache

    kept: list[traceback.FrameSummary] = []
    for frame, line in walk(traceback_):
        if is_library(frame):
            continue
        code = frame.f_code
        source = linecache.getline(code.co_filename, line, frame.f_globals).strip()
        kept.append(traceback.FrameSummary(code.co_filename, line, code.co_name, lookup_line=False, line=source))
    blocks = traceback.StackSummary.from_list(kept).format()
    return [line for block in blocks for line in block.rstrip("\n").split("\n")]


def _footer(error: MimeError, written: Path | None, switch: str) -> list[str]:
    lines: list[str] = []
    internal = [r for r in error.refusals if r.category is RefusalClass.INTERNAL]
    if any(r.mimes_defect and r.code != "error.uncoded" for r in internal):
        lines.append(f"This is a defect in MIME: report it at {ISSUES}, with the file named below.")
    elif any(r.code == "error.uncoded" for r in internal):
        lines.append(f"{UNCODED} at {ISSUES}, with the file named below.")
    else:
        owners = sorted({r.owner for r in internal if r.owner not in (None, "mime")})
        if owners:
            lines.append(f"This is a defect in {', '.join(owners)}, not in MIME: report it to "
                         f"{'that package' if len(owners) == 1 else 'those packages'}, with the file named below.")
    where = f" is in {written}" if written is not None else " was not written (its directory cannot be written)"
    lines.append(f"The full traceback{where}" + ("." if full_mode() else f"; {switch} shows it here."))
    return lines


def short_text(error: MimeError, traceback_: TracebackType | None, *, frames: bool = True,
               switch: str = f"{TRACEBACK_VARIABLE}=full", headline: str | None = None) -> str:
    """The error as a script's user is shown it: the script's own frames (``frames=False`` for the CLI, whose user
    wrote no frame), the error in its shape, and where its full chain is, written to its file here; ``switch`` is
    what shows the full chain in place of this. ``headline`` (the CLI's, naming its command) is the first line in
    place of the class and its own headline."""
    from mime._terminal import for_terminal

    written = write_traceback(error)
    lines: list[str] = []
    held = _frame_lines(traceback_) if frames else []
    if held:
        lines.append("Traceback (most recent call last; MIME's, JAX's and MADDENING's frames left out):")
        lines += held
    lines.append(for_terminal(error.text(headline)) if headline is not None else
                 f"{type(error).__name__}: {for_terminal(error.text())}")
    lines += _footer(error, written, switch)
    return "\n".join(lines) + "\n"


type ExceptHook = Callable[[type[BaseException], BaseException, TracebackType | None], object]


def _hook(previous: ExceptHook) -> ExceptHook:
    def hook(kind: type[BaseException], value: BaseException, traceback_: TracebackType | None) -> object:
        if isinstance(value, MimeError) and not full_mode():
            sys.stderr.write(short_text(value, traceback_))
            # A hook set before MIME's (a crash reporter, an application's error dialog) still sees MIME's errors,
            # with the short traceback; Python's own would print it a second time.
            if previous is not sys.__excepthook__:
                return previous(kind, value, traceback_)
            return None
        return previous(kind, value, traceback_)

    setattr(hook, "_mime_previous", previous)  # noqa: B010 - marks MIME's hook, so it is installed once
    return hook


def install() -> None:
    """Show a MIME error that no one catches in its short form: in a script, through ``sys.excepthook`` (wrapping the
    hook already there, which every other error still goes to, and which a MIME error goes to after its short form
    when it is not Python's own); in IPython and Jupyter, through the shell's handler of MIME's errors, when the shell
    has no other. Installed once, when MIME is imported."""
    if not hasattr(sys.excepthook, "_mime_previous"):
        sys.excepthook = _hook(sys.excepthook)
    _install_in_ipython()


def _install_in_ipython() -> None:
    held = sys.modules.get("IPython")
    if held is None:
        return
    shell = cast(Any, getattr(held, "get_ipython", lambda: None))()
    if shell is None or getattr(shell, "custom_exceptions", ()) not in ((), (MimeError,)):
        return

    def handler(self: Any, kind: type[BaseException], value: BaseException, traceback_: TracebackType | None,
                tb_offset: int | None = None) -> list[str]:
        # The shell shows nothing of what a handler returns: the handler shows it, as the shell shows a traceback.
        if not isinstance(value, MimeError) or full_mode():
            self.showtraceback((kind, value, traceback_), tb_offset=tb_offset)
            return []
        lines = short_text(value, traceback_).splitlines()
        show = getattr(self, "_showtraceback", None)
        if callable(show):
            show(kind, value, lines)
        else:
            sys.stderr.write("\n".join(lines) + "\n")
        return lines

    shell.set_custom_exc((MimeError,), handler)


__all__ = ["DIRECTORY_VARIABLE", "FULL", "KEPT", "NAME", "PREFIX", "TRACEBACK_VARIABLE", "full_mode", "install",
           "report_text", "short_text", "traceback_directory", "write_traceback"]
