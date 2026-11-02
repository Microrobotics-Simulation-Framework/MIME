"""A word written into a command that is offered for pasting into a shell, so that what it holds stays one word.

A name or a path in such a command can come from outside (the experiment's name in a folder someone shared), and a
shell reads ``;``, ``$(...)``, a space or a quote as its own. POSIX shells: :func:`shlex.quote`. Windows has no one
quoting: ``cmd.exe`` reads ``%`` and ``"`` even inside double quotes, with no escape that works in every place, so there a
word holding anything but plain characters is put in double quotes and each ``%`` and ``"`` in it is written as ``_``
(the command then names another file than the one meant, and does no harm); PowerShell reads double quotes more kindly
but the same rule is safe in both."""

from __future__ import annotations

import re
import shlex
import sys

#: What needs no quoting anywhere.
_PLAIN = re.compile(r"[A-Za-z0-9_@+=:,./\\-]+")


def quoted(word: str, *, windows: bool | None = None) -> str:
    """``word`` as one shell word. ``windows`` defaults to whether this platform is Windows."""
    if not (sys.platform == "win32" if windows is None else windows):
        return shlex.quote(word)
    if _PLAIN.fullmatch(word):
        return word
    return '"' + word.replace("%", "_").replace('"', "_") + '"'
