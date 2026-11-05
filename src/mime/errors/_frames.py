"""Which frames of a traceback are the script's own, and the traceback of those alone.

A frame is the library's when its module is in one of :data:`HIDDEN_PACKAGES` (MIME, JAX, MADDENING and what they
run on), when its file is in Python's own library, or when it is one of the interpreter's frozen modules. Every other
frame is the script's: the script, its own modules, a lab's package, a notebook's cells."""

from __future__ import annotations

import inspect
import os
import sys
import sysconfig
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import FrameType, TracebackType

#: The packages whose frames a short traceback hides: MIME's, JAX's and MADDENING's, the libraries they compute with
#: and read files with, and the interactive shells a notebook runs a cell through.
HIDDEN_PACKAGES = frozenset({
    "mime", "maddening", "jax", "jaxlib", "numpy", "scipy", "materica", "equinox", "lineax", "optimistix",
    "jaxtyping", "typeguard", "opt_einsum", "ml_dtypes", "IPython", "ipykernel", "traitlets", "pxr", "yaml", "pooch",
    "platformdirs", "packaging",
})


def _normal(path: str) -> str:
    """``path`` as the file system compares it: case folded where the system ignores case (Windows' drive letters)."""
    return os.path.normcase(path)


_STANDARD_LIBRARY = tuple(_normal(str(Path(path).resolve())) for path in
                          {sysconfig.get_paths()["stdlib"], sysconfig.get_paths()["platstdlib"]})


def _in_standard_library(filename: str) -> bool:
    try:
        resolved = _normal(str(Path(filename).resolve()))
    except (OSError, ValueError):
        return False
    return resolved.startswith(_STANDARD_LIBRARY) and "site-packages" not in resolved \
        and "dist-packages" not in resolved


def is_library(frame: FrameType) -> bool:
    """Whether ``frame`` is a library's (hidden in a short traceback), not the script's own."""
    module = str(frame.f_globals.get("__name__", ""))
    filename = frame.f_code.co_filename
    if module.split(".", 1)[0] in HIDDEN_PACKAGES or filename.startswith("<frozen"):
        return True
    return _in_standard_library(filename)


@dataclass(frozen=True)
class Place:
    """A line of the script's: its file, its line number and the function it is in."""

    filename: str
    line: int
    function: str

    def text(self) -> str:
        return f"{self.filename}, line {self.line}"


def walk(traceback: TracebackType | None) -> Iterator[tuple[FrameType, int]]:
    """Every frame of ``traceback``, outermost first, with the line it was at."""
    while traceback is not None:
        yield traceback.tb_frame, traceback.tb_lineno
        traceback = traceback.tb_next


def script_places(traceback: TracebackType | None) -> list[Place]:
    """The script's own frames of ``traceback``, outermost first."""
    return [Place(frame.f_code.co_filename, line, frame.f_code.co_name)
            for frame, line in walk(traceback) if not is_library(frame)]


def innermost(traceback: TracebackType | None) -> FrameType | None:
    """The frame the error was raised in."""
    found = None
    for frame, _ in walk(traceback):
        found = frame
    return found


def raised_by_library(error: BaseException) -> bool:
    """Whether ``error`` was raised in a library's frame rather than the script's."""
    frame = innermost(error.__traceback__)
    return frame is not None and is_library(frame)


def _is_mimes(frame: FrameType) -> bool:
    return str(frame.f_globals.get("__name__", "")).split(".", 1)[0] == "mime"


def raiser(traceback: TracebackType | None) -> FrameType | None:
    """The frame whose call raised: the innermost that is MIME's or the script's, the libraries' and Python's frames
    below it being what it called (``difflib`` inside MIME's ``did_you_mean``, JAX inside a step of the script's);
    ``None`` when every frame is a library's."""
    found: FrameType | None = None
    for frame, _ in walk(traceback):
        if _is_mimes(frame) or not is_library(frame):
            found = frame
    return found


def raised_by_mime(error: BaseException) -> bool:
    """Whether ``error`` was raised by MIME: by one of its frames, or by what one of them called (a library, Python's
    own), with no frame of the script's between."""
    frame = raiser(error.__traceback__)
    return frame is not None and _is_mimes(frame)


def script_traceback(traceback: TracebackType | None) -> TracebackType | None:
    """``traceback`` with the library's frames left out: the script's own, in their order, or ``None``."""
    kept = [(tb.tb_frame, tb.tb_lasti, tb.tb_lineno) for tb in _entries(traceback) if not is_library(tb.tb_frame)]
    built: TracebackType | None = None
    for frame, lasti, line in reversed(kept):
        built = TracebackType(built, frame, lasti, line)
    return built


def _entries(traceback: TracebackType | None) -> Iterator[TracebackType]:
    while traceback is not None:
        yield traceback
        traceback = traceback.tb_next


def caller_place() -> Place | None:
    """The innermost frame of the script's on the stack as it stands: the line that called into MIME."""
    frame = inspect.currentframe()
    while frame is not None:
        if not is_library(frame):
            return Place(frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name)
        frame = frame.f_back
    return None


#: The attribute a warning instance carries when it has no line of the script's to name (see :func:`warn_at_script`):
#: the command line shows it as ``mime: warning: <text>``, as it shows a description's.
WITHOUT_SOURCE = "mime_without_source"


def _is_entry(frame: FrameType) -> bool:
    """Whether ``frame`` is the entry of a MIME command: the console script's stub (a file in the interpreter's
    scripts directory, or in the launcher MIME's Windows executable holds) or ``python -m mime``'s ``__main__``, with
    only the interpreter's own frames (``runpy``) outside it. A script of the user's that calls ``mime.cli.main`` is
    not: its file is neither."""
    outer = frame.f_back
    while outer is not None:
        if not is_library(outer):
            return False
        outer = outer.f_back
    if frame.f_globals.get("__name__") != "__main__":
        return False
    folder = Path(frame.f_code.co_filename).parent
    scripts = {Path(sysconfig.get_path("scripts")), Path(sys.executable).parent}
    here = Path(__file__).resolve().parent.parent
    try:
        folder = folder.resolve()
    except OSError:
        return False
    return folder in {path.resolve() for path in scripts} or folder.suffix == ".exe" \
        or here in (folder, *folder.parents)


def warn_at_script(message: str, category: type[Warning]) -> None:
    """Warn at the line of the script's that called into MIME, however many of MIME's own frames lie between: a public
    call that builds inside another (``save``, ``load``) shows the script's line, not its own. The script is the
    innermost frame that is not a library's. Where there is none, because a command line (``mime run``) called into
    MIME and its entry point is the outermost frame, the warning carries :data:`WITHOUT_SOURCE` and the command line
    shows it without a source line; elsewhere, at the outermost frame."""
    level, frame = 1, inspect.currentframe()
    while frame is not None and (level == 1 or is_library(frame)):
        if frame.f_back is None:
            break
        frame, level = frame.f_back, level + 1
    warning = category(message)
    if frame is not None and _is_entry(frame):
        setattr(warning, WITHOUT_SOURCE, True)
    warnings.warn(warning, stacklevel=level)


__all__ = ["HIDDEN_PACKAGES", "WITHOUT_SOURCE", "Place", "caller_place", "innermost", "is_library", "raised_by_library",
           "raised_by_mime", "raiser", "script_places", "script_traceback", "walk", "warn_at_script"]
