"""Ctrl+C while the ``mime`` command is still importing MIME (JAX and MADDENING take about a second) ends it quietly, 130.

Nothing has been loaded or written by then, so there is nothing to say, and Python's own answer is a traceback of whatever
import was under way. The guard is installed only in a process that the interpreter was asked to run as the command: the
``mime`` script it was given, or ``-m mime.cli`` among its own options (:func:`is_the_command`). A script that imports MIME,
whatever it is called and whatever arguments it is given, keeps Python's handling of Ctrl+C, and so does a child it spawns.
``mime.cli.main`` takes the guard off as it starts, handing Ctrl+C to the command's own handling (``mime._interrupt``), and
puts back exactly the handler there was: a job started with Ctrl+C ignored keeps ignoring it, and the guard is never
installed over that. This module imports nothing heavy, because it runs first."""

from __future__ import annotations

import os
import signal
import sys
from collections.abc import Sequence
from typing import Any

#: The exit code of a command ended by Ctrl+C, as a shell reports one (128 + SIGINT), the same as ``mime._interrupt``'s.
INTERRUPTED = 130

#: The suffix of the launcher Windows makes of the ``mime`` script, whose text is packed into it and cannot be read.
LAUNCHER = ".exe"

#: The interpreter's options that take a value, as a word after them or the rest of their own word (``-Wignore``).
_TAKES_VALUE = "WX"

#: What the guard took the place of, while it is installed.
_replaced: Any = None


def _asked_of_the_interpreter(held: Sequence[str]) -> tuple[str, str] | None:
    """What the interpreter was asked to run, from ``sys.orig_argv``: ``("module", name)`` for ``-m``, ``("script", path)``
    for a file, ``("code", "")`` for ``-c`` and ``None`` when it was asked for nothing (a console). Only the words before
    that one are the interpreter's options; the words after it are the program's own."""
    words = list(held[1:])
    index = 0
    while index < len(words):
        word = words[index]
        if word == "--" and index + 1 < len(words):
            return "script", words[index + 1]
        if not word.startswith("-") or word == "-":
            return "script", word
        if word.startswith("--"):
            index += 2 if word == "--check-hash-based-pycs" else 1
            continue
        for place, letter in enumerate(word[1:], start=1):
            rest = word[place + 1:]
            if letter in "cm":
                value = rest or (words[index + 1] if index + 1 < len(words) else "")
                return ("module", value) if letter == "m" else ("code", "")
            if letter in _TAKES_VALUE:
                index += 0 if rest else 1
                break
        index += 1
    return None


def is_the_command() -> bool:
    """Whether the interpreter was asked to run this process as the ``mime`` command: ``-m mime.cli`` among its own
    options, or the ``mime`` script (the file, or on Windows its launcher, ``mime`` with the ``.exe`` suffix, or the
    ``mime-script`` file the launcher runs). A file called ``mime`` that does not name ``mime.cli`` is a user's script."""
    asked = _asked_of_the_interpreter(list(sys.orig_argv))
    if asked is None:
        return False
    kind, name = asked
    if kind == "module":
        return name == "mime.cli"
    if kind != "script" or not sys.argv or name != sys.argv[0]:
        return False
    stem, suffix = os.path.splitext(os.path.basename(name).lower())
    if stem == "mime" and suffix == LAUNCHER:
        return True
    if stem not in ("mime", "mime-script") or suffix not in ("", ".py", ".pyw"):
        return False
    try:
        with open(name, "rb") as script:
            return b"mime.cli" in script.read(4096)
    except OSError:
        return False


def _leave(signum: int, frame: object) -> None:
    os._exit(INTERRUPTED)


def guard() -> None:
    """Install the guard when this process is the command, in the main thread, unless Ctrl+C is ignored (a job started
    with it ignored keeps ignoring it)."""
    global _replaced
    if not is_the_command():
        return
    try:
        held = signal.getsignal(signal.SIGINT)
        if held is signal.SIG_IGN:
            return
        signal.signal(signal.SIGINT, _leave)
        _replaced = held
    except ValueError:  # not the main thread
        pass


def release() -> None:
    """Take the guard off, putting back what it replaced, if it is the guard that is installed."""
    global _replaced
    if signal.getsignal(signal.SIGINT) is _leave:
        signal.signal(signal.SIGINT, _replaced if _replaced is not None else signal.default_int_handler)
    _replaced = None
