"""TOML written and edited in place, with no TOML library beyond the standard library's reader.

**Writing** covers the closed shapes ``mime.toml`` and ``defaults.toml`` hold: text, integers, floats, booleans, arrays
and inline tables (an array of tables is written one table a line). Every value written is read back by ``tomllib`` and
compared before anything is kept.

**Editing in place** replaces, adds or removes a key of one table and leaves every other line as it was, byte for byte,
comments and line endings included. ``tomllib`` is the oracle for where a statement ends: a key's lines are the fewest
lines from its ``key =`` line that read as that one key, so a bracket inside a string or a comment is never taken for the
value's end. The comment that ends the key's own first line is kept on its new first line, and the key's indentation on
each new line; a comment inside its old value goes with it. The new lines take the file's line ending (CRLF where any
line of it is CRLF), and a file whose last line had no line feed has one once that line is replaced. The result is read
again and must hold the edit and every other key as it was; otherwise nothing is written (:class:`Unedited`). What an
edit cannot replace safely is refused, never guessed: a key written as dotted keys or as a table of its own
(``fields.languages = ...``, ``[record.fields]``), and a table written as an inline table, as dotted keys at the top
(``record = {...}``) or as an array of tables (``[[record]]``)."""

from __future__ import annotations

import copy
import datetime
import math
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

type Json = dict[str, object]

#: A key written bare; any other is quoted.
_BARE = re.compile(r"[A-Za-z0-9_-]+")
_ESCAPES = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\t": "\\t", "\n": "\\n", "\f": "\\f", "\r": "\\r"}
#: An array of tables is indented by this, one table a line.
INDENT = "  "


class Unedited(Exception):
    """An edit that was not made: what stops it, in words; the file is left as it was."""


#: The value an update gives to remove its key.
REMOVE = object()


def _string(text: str) -> str:
    """``text`` as a TOML basic string: every control character escaped, every other character as it is."""
    out = [_ESCAPES.get(each) or (f"\\u{ord(each):04X}" if ord(each) < 0x20 or ord(each) == 0x7F else each)
           for each in text]
    return '"' + "".join(out) + '"'


def key_text(key: str) -> str:
    return key if _BARE.fullmatch(key) else _string(key)


def value_text(value: object) -> str:
    """``value`` as TOML on one line: a table as an inline table."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return ("nan" if math.isnan(value) else "inf" if value > 0 else "-inf") if not math.isfinite(value) \
            else repr(value)
    if isinstance(value, str):
        return _string(value)
    if isinstance(value, datetime.date):
        return value.isoformat()
    if isinstance(value, list):
        return "[" + ", ".join(value_text(each) for each in cast(list[object], value)) + "]"
    if isinstance(value, dict):
        held = cast(Json, value)
        if not held:
            return "{}"
        return "{ " + ", ".join(f"{key_text(key)} = {value_text(each)}" for key, each in held.items()) + " }"
    raise TypeError(f"TOML holds no {type(value).__name__}")


def key_lines(key: str, value: object) -> list[str]:
    """``key = value`` as lines, without their endings: an array of tables one table a line, anything else on one."""
    items = cast(list[object], value) if isinstance(value, list) else None
    if items and all(isinstance(each, dict) for each in items):
        return [f"{key_text(key)} = [", *(f"{INDENT}{value_text(each)}," for each in items), "]"]
    return [f"{key_text(key)} = {value_text(cast(object, value))}"]


def reads(text: str) -> Json | None:
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return None


def lines_of(text: str) -> list[str]:
    """``text`` as its lines, each with its own ending: TOML's lines end at a line feed alone (a form feed or a U+2028
    inside a string ends none)."""
    pieces = text.split("\n")
    lines = [piece + "\n" for piece in pieces[:-1]]
    return lines + ([pieces[-1]] if pieces[-1] else [])


def _path(held: object) -> tuple[tuple[str, ...], bool]:
    """The keys down to the one leaf of ``held`` (a statement read alone), and whether it ends in an array of tables."""
    path: list[str] = []
    node = held
    while isinstance(node, dict) and len(cast(Json, node)) == 1:
        ((key, node),) = cast(Json, node).items()
        path.append(key)
        if isinstance(node, list):
            return tuple(path), True
    return tuple(path), False


@dataclass(frozen=True)
class Statement:
    """A statement of a TOML file: lines ``start`` to ``end`` (exclusive); a table's header (``header``, ``array`` for
    ``[[...]]``) or a key and its value (``key``, dotted keys as their parts), in the table ``table``."""

    start: int
    end: int
    table: tuple[str, ...]
    header: tuple[str, ...] | None = None
    key: tuple[str, ...] | None = None
    array: bool = False


def _key_of(line: str) -> tuple[str, ...]:
    """The dotted key a key/value line starts with: the text before the first ``=`` that reads as a key."""
    for found in re.finditer("=", line):
        held = reads(line[:found.start()] + "= 0\n")
        if held is not None:
            return _path(held)[0]
    raise Unedited("a line holds no key that reads")


def statements(lines: Sequence[str]) -> list[Statement]:
    """Every statement of ``lines`` (a file that reads), in order; blank and comment lines are none."""
    found: list[Statement] = []
    table: tuple[str, ...] = ()
    array = False
    at = 0
    while at < len(lines):
        bare = lines[at].strip()
        if not bare or bare.startswith("#"):
            at += 1
            continue
        if bare.startswith("["):
            held = reads(lines[at])
            if held is None:
                raise Unedited(f"line {at + 1} is a table's header that does not read alone")
            table, array = _path(held)
            found.append(Statement(at, at + 1, table, header=table, array=array))
            at += 1
            continue
        end = next((end for end in range(at + 1, len(lines) + 1) if reads("".join(lines[at:end])) is not None), None)
        if end is None:
            raise Unedited(f"the statement on line {at + 1} does not read")
        found.append(Statement(at, end, table if not array else (*table, "[]"), key=_key_of(lines[at])))
        at = end
    return found


def _first_line_comment(lines: Sequence[str], held: Statement) -> str:
    """The comment that ends the statement's first line, with the space before it, or ``""``: the first ``#`` whose cut
    leaves the statement reading as it did (a ``#`` inside a string is none)."""
    first = lines[held.start]
    body = first.rstrip("\r\n")
    ending = first[len(body):]
    alone = reads("".join(lines[held.start:held.end]))
    for found in re.finditer(r"[ \t]*#", body):
        if found.start() == 0:
            continue
        cut = [body[:found.start()] + ending, *lines[held.start + 1:held.end]]
        if reads("".join(cut)) == alone:
            return body[found.start():]
    return ""


def edited(text: str, table: str, updates: Mapping[str, object]) -> str:
    """``text`` with each key of ``updates`` set in the table ``table`` (a top-level table's name): its statement
    replaced (the comment that ends its first line kept), or added after the table's last statement (the table's
    header added at the end where it has none), or removed where its value is :data:`REMOVE`; every other line as it was.
    Refused (:class:`Unedited`) where the key or the table is written in a form an edit cannot replace safely, or where
    the result does not read back as ``text`` with the edit."""
    before = reads(text)
    if before is None:
        raise Unedited("it is not TOML that reads")
    if table in before and not isinstance(before[table], dict):
        raise Unedited(f"[{table}] is written as " + ("an array of tables, [[" + table + "]]" if isinstance(
            before[table], list) else f"a {type(before[table]).__name__}, not a table") + f", which an edit does not "
                       f"replace; write it as one table, [{table}], with one key a line")
    ending = "\r\n" if "\r\n" in text else "\n"
    lines = lines_of(text)
    for key, value in updates.items():
        lines = lines_of("".join(_edited_key(lines, table, key, value, ending)))
    after = "".join(lines)
    expected = copy.deepcopy(before)
    for key, value in updates.items():
        held = expected.get(table)
        if value is not REMOVE:
            cast(Json, expected.setdefault(table, {}))[key] = value
        elif isinstance(held, dict):
            cast(Json, held).pop(key, None)
    if reads(after) != expected:
        raise Unedited("read again, it does not hold the edit and every other key as it was")
    try:
        after.encode("utf-8")
    except UnicodeEncodeError:
        raise Unedited("a value holds a character UTF-8 cannot write (a lone surrogate)") from None
    return after


def _edited_key(lines: list[str], table: str, key: str, value: object, ending: str) -> list[str]:
    found = statements(lines)
    if any(each.table == () and each.key is not None and each.key[0] == table for each in found):
        raise Unedited(f"[{table}] is written as an inline table or as dotted keys at the top, which an edit does not "
                       f"replace; write it as a table, [{table}], with one key a line")
    own = [each for each in found if each.table == (table,) and each.key is not None and each.key[0] == key]
    apart = [each for each in found if each.header is not None and each.header[:2] == (table, key)]
    if apart or any(each.key != (key,) for each in own):
        raise Unedited(f"[{table}] {key} is written as dotted keys or as a table of its own, which an edit does not "
                       f"replace; write it as one key, {key} = ...")
    new = [line + ending for line in key_lines(key, value)] if value is not REMOVE else []
    if own:
        (held,) = own
        comment = _first_line_comment(lines, held)
        indent = lines[held.start][:len(lines[held.start]) - len(lines[held.start].lstrip(" \t"))]
        new = [indent + each for each in new]
        if new and comment:
            new[0] = new[0].removesuffix(ending) + comment + ending
        return [*lines[:held.start], *new, *lines[held.end:]]
    if value is REMOVE:
        return lines
    header = next((each for each in found if each.header == (table,) and not each.array), None)
    last = len(lines) if header is None else max(
        (each.end for each in found if each.table == (table,) and each.key is not None), default=header.end)
    if last == len(lines) and lines and not lines[-1].endswith("\n"):  # the last line ended, to write after it
        lines = [*lines[:-1], lines[-1] + ending]
    if header is None:
        new = [*([ending] if lines else []), f"[{table}]{ending}", *new]
    return [*lines[:last], *new, *lines[last:]]


__all__ = ["INDENT", "REMOVE", "Statement", "Unedited", "edited", "key_lines", "key_text", "lines_of", "reads",
           "statements", "value_text"]
