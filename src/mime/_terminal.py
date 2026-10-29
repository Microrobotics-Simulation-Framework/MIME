"""Text from outside reaches a terminal only as plain characters.

A bundle's names, versions, licences and messages are written by a stranger. A control character in one is a terminal
command (an escape sequence moves the cursor, rewrites a line or sets the window title), and a newline forges a line of
the summary or of a refusal. Both are written out as the escape that names them (``\\x1b``, ``\\n``) before the text is
printed.
"""

from __future__ import annotations

import unicodedata

#: Unicode categories that act on a terminal or hide what a line says: controls (including the C1 range, one of which
#: opens an escape sequence on its own), format characters (the bidirectional overrides among them), line and paragraph
#: separators, and characters that are not assigned or are private.
_INVISIBLE = frozenset({"Cc", "Cf", "Cs", "Co", "Cn", "Zl", "Zp"})


def _escaped(character: str) -> str:
    return ascii(character)[1:-1]


def plain(text: str, *, keep: str = "") -> str:
    """``text`` with every control, format and separator character written as its escape, except those in ``keep``."""
    return "".join(character if character in keep or unicodedata.category(character) not in _INVISIBLE
                   else _escaped(character) for character in text)


def for_terminal(text: str) -> str:
    """A message made of lines, as it is printed: the newlines and tabs that lay it out stay, the rest is escaped."""
    return plain(text, keep="\n\t")
