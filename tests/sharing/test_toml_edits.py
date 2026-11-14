"""``mime.toml`` edited in place, key by key (``mime.push._toml``, through ``--setup``'s own write, ``edit_in_place``),
held to one property over drawn TOML documents: comments (full lines and the ends of lines, inside arrays too), blank
lines, LF, CRLF and the two mixed, multi-line basic and literal strings, arrays of tables, inline tables, quoted keys,
dotted keys, sub-tables, a last line with no line feed. An edit of one key of ``[record]`` either leaves every other
key equal and every other line byte for byte (its comments and its line endings among them), the key's own first-line
comment kept on its new first line, or refuses (``manifest.write``) and leaves the file byte for byte; and it refuses
exactly where the key, or ``[record]`` itself, is written in a form an edit does not replace (dotted keys, a table of its
own, an inline table): anywhere else it must succeed, so a writer that refused everything would fail too.

The oracle is the drawing's own: it knows where each statement lies and what ends its first line, and judges the file
by ``tomllib`` and by its lines, never by the editor's reading of it.

Controls, measured by ``.github/probe/detection_rate.py`` with ``MIME_PLANT_TOML_WRITER``: ``comment`` plants a writer
that drops a comment line, ``neighbour`` one that also rewrites another key of ``[record]``; each must fail the property
on every run. And the editor's own read-back (the net under its splice) is shown catching a neighbour changed inside it.
"""

from __future__ import annotations

import copy
import os
import tempfile
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mime.build import BuildRefused
from mime.push import _setup, _toml

#: What a comment's text is drawn from: any character a TOML comment may hold (no control character), ``#`` and
#: quotes among them.
COMMENT = st.text(st.characters(exclude_categories=("Cs", "Cc")), max_size=12)
#: A bare key's spelling.
NAMES = st.from_regex(r"[a-z][a-z0-9_]{0,6}", fullmatch=True)
#: What a multi-line or a literal string's text is drawn from: no quote of its own kind, no backslash, no control.
SAFE = st.text(st.characters(exclude_categories=("Cs", "Cc"), exclude_characters="'\"\\"), max_size=10)
ENDINGS = {"lf": ("\n",), "crlf": ("\r\n",), "mixed": ("\n", "\r\n")}


def basic(text: str) -> str:
    """``text`` as a TOML basic string, written apart from the writer under test: JSON's escapes, and DEL's."""
    import json

    return json.dumps(text, ensure_ascii=False).replace("\x7f", "\\u007F")


@dataclass
class Drawn:
    """A drawn document: its lines (each with its ending), and for each key of ``[record]`` written as one key, where its
    statement lies and the comment that ends its first line; the keys written in a form no edit replaces; where a new
    key goes; and how ``[record]`` is written (``table``, ``inline`` or ``absent``)."""

    lines: list[str] = field(default_factory=list[str])
    spans: dict[str, tuple[int, int, str]] = field(default_factory=dict[str, tuple[int, int, str]])
    fixed: set[str] = field(default_factory=set[str])
    insert: int = 0
    record: str = "table"

    @property
    def text(self) -> str:
        return "".join(self.lines)


@st.composite
def values(draw: st.DrawFn) -> tuple[list[str], bool]:
    """A value written as TOML: its lines (no endings), and whether a comment may end its first line."""
    kind = draw(st.sampled_from(["basic", "literal", "number", "bool", "inline", "array", "multi-basic",
                                 "multi-literal", "tables"]))
    if kind == "basic":
        return [basic(draw(st.text(st.characters(exclude_categories=("Cs",)), max_size=12)))], True
    if kind == "literal":
        return [f"'{draw(SAFE)}'"], True
    if kind == "number":
        return [str(draw(st.integers(-10**6, 10**6)) if draw(st.booleans()) else
                    repr(draw(st.floats(allow_nan=False, allow_infinity=False, width=32))))], True
    if kind == "bool":
        return [draw(st.sampled_from(["true", "false"]))], True
    if kind == "inline":
        keys = draw(st.lists(NAMES, unique=True, max_size=3))
        return ["{ " + ", ".join(f"{key} = {basic(draw(SAFE))}" for key in keys) + " }" if keys else "{}"], True
    if kind == "array":
        return ["[" + ", ".join(basic(draw(SAFE)) for _ in range(draw(st.integers(0, 3)))) + "]"], True
    if kind == "multi-basic":
        return ['"""', *draw(st.lists(SAFE, max_size=3)), f'{draw(SAFE)}"""'], False
    if kind == "multi-literal":
        return ["'''", *draw(st.lists(SAFE, max_size=3)), f"{draw(SAFE)}'''"], False
    items = [f"  {{ name = {basic(draw(SAFE))} }}," for _ in range(draw(st.integers(0, 3)))]
    return ["[", *items, "]"], True


@st.composite
def documents(draw: st.DrawFn) -> Drawn:
    held = Drawn()
    ends = ENDINGS[draw(st.sampled_from(list(ENDINGS)))]

    def line(text: str) -> None:
        held.lines.append(text + draw(st.sampled_from(ends)))

    def trivia() -> None:
        for _ in range(draw(st.integers(0, 2))):
            line(f"# {draw(COMMENT)}" if draw(st.booleans()) else draw(st.sampled_from(["", "  "])))

    def tail() -> str:
        return f"  # {draw(COMMENT)}" if draw(st.booleans()) else ""

    def statement(key: str) -> tuple[int, int, str]:
        """A key and its value, comments where TOML takes them: the statement's span and its first line's comment."""
        start = len(held.lines)
        written, first_ok = draw(values())
        first = tail() if first_ok else ""
        if len(written) == 1:
            line(f"{key} = {written[0]}{first}")
            return start, len(held.lines), first
        line(f"{key} = {written[0]}{first}")
        for each in written[1:-1]:
            line(each + (tail() if first_ok else ""))
            if first_ok and draw(st.booleans()):
                line(f"  # {draw(COMMENT)}")
        line(written[-1] + tail())
        return start, len(held.lines), first

    trivia()
    line('format = "mime.toml/1"' + tail())
    held.record = draw(st.sampled_from(["table", "table", "table", "inline", "absent"]))
    if held.record == "inline":
        line('record = { title = "x" }' + tail())
    trivia()
    line("[bundle]")
    line('licence = "CC-BY-4.0"' + tail())
    trivia()
    if draw(st.booleans()):
        line("[[things]]")
        statement("name")
        trivia()
    names = draw(st.lists(NAMES, unique=True, min_size=1, max_size=6))
    forms = [draw(st.sampled_from(["plain", "plain", "plain", "quoted", "dotted", "subtable", "arraytable"]))
             for _ in names]
    if held.record == "table":
        line("[record]" + tail())
        held.insert = len(held.lines)
        trivia()
        for name, form in zip(names, forms, strict=True):
            if form in ("plain", "quoted"):
                spelled = name if form == "plain" else draw(st.sampled_from([f'"{name}"', f"'{name}'"]))
                held.spans[name] = statement(spelled)
                held.insert = held.spans[name][1]
            elif form == "dotted":
                held.fixed.add(name)
                for part in draw(st.lists(NAMES, unique=True, min_size=1, max_size=2)):
                    statement(f"{name}.{part}")
                held.insert = len(held.lines)
            trivia()
        for name, form in zip(names, forms, strict=True):
            if form in ("subtable", "arraytable"):
                held.fixed.add(name)
                line(f"[record.{name}]" if form == "subtable" else f"[[record.{name}]]")
                statement("inner")
                trivia()
    if draw(st.booleans()):
        line("[other]")
        statement("key")
        trivia()
    if held.record == "absent":
        held.insert = len(held.lines)
    if held.lines[-1].strip() and draw(st.booleans()):
        held.lines[-1] = held.lines[-1].rstrip("\r\n")
    return held


#: A new value of a key of ``[record]``: any shape ``mime.toml`` holds, or the key removed.
NEW = st.one_of(
    st.text(st.characters(exclude_categories=("Cs",)), max_size=12),
    st.integers(-10**6, 10**6), st.booleans(), st.floats(allow_nan=False, allow_infinity=False),
    st.lists(st.text(max_size=6), max_size=3),
    st.lists(st.dictionaries(st.text(max_size=4), st.text(max_size=6), max_size=3), min_size=1, max_size=3),
    st.dictionaries(st.text(max_size=4), st.one_of(st.text(max_size=6), st.lists(st.integers(), max_size=2)),
                    max_size=3),
    st.just(_toml.REMOVE))


def planted(kind: str, target: str, real: Callable[[str, str, object], str]) -> Callable[[str, str, object], str]:
    """A writer that drops the first comment line it writes (``comment``), or rewrites another key of ``[record]``
    as well (``neighbour``)."""
    def write(text: str, table: str, updates: object) -> str:
        after = real(text, table, updates)
        lines = _toml.lines_of(after)
        if kind == "comment":
            at = next((index for index, each in enumerate(lines) if each.lstrip().startswith("#")), None)
            return after if at is None else "".join(lines[:at] + lines[at + 1:])
        other = next((each for each in _toml.statements(lines) if each.table == ("record",) and each.key is not None
                      and each.key[0] != target), None)
        if other is None:
            return after
        ending = "\r\n" if lines[other.start].endswith("\r\n") else "\n"
        return "".join([*lines[:other.start], f'{_toml.key_text(other.key[0])} = "planted"{ending}', *lines[other.end:]])
    return write


def edited_on_disk(drawn: Drawn, key: str, value: object) -> tuple[bytes, BuildRefused | None]:
    """The drawn file edited through ``--setup``'s write: its bytes after, and the refusal where it refused."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "mime.toml"
        path.write_bytes(drawn.text.encode("utf-8"))
        try:
            _setup.edit_in_place(path, {"record": {key: value}}, "mime push <folder> --setup")
        except BuildRefused as refused:
            return path.read_bytes(), refused
        return path.read_bytes(), None


@settings(deadline=None, max_examples=300)
@given(drawn=documents(), choice=st.integers(0, 10), fresh=NAMES, value=NEW)
def test_an_edit_keeps_every_other_key_comment_and_line_ending_or_refuses_leaving_the_file(
        drawn: Drawn, choice: int, fresh: str, value: object):
    known = sorted([*drawn.spans, *drawn.fixed])
    key = known[choice % len(known)] if known and choice < 8 else fresh
    plant = os.environ.get("MIME_PLANT_TOML_WRITER")
    with pytest.MonkeyPatch.context() as patch:
        if plant:
            patch.setattr(_setup, "edited", planted(plant, key, _toml.edited))
        after, refused = edited_on_disk(drawn, key, value)
    before = tomllib.loads(drawn.text)
    if drawn.record == "inline" or key in drawn.fixed:
        assert refused is not None and refused.codes() == ["manifest.write"], "a form no edit replaces is refused"
        assert after == drawn.text.encode("utf-8"), "and the file is left byte for byte"
        return
    assert refused is None, refused
    text = after.decode("utf-8")
    expected = copy.deepcopy(before)
    if value is _toml.REMOVE:
        expected.get("record", {}).pop(key, None)
    else:
        expected.setdefault("record", {})[key] = value
    assert tomllib.loads(text) == expected, "every other key as it was, and the edit"
    ending = "\r\n" if "\r\n" in drawn.text else "\n"
    old, new = list(drawn.lines), _toml.lines_of(text)
    start, end, comment = drawn.spans.get(key, (drawn.insert, drawn.insert, ""))
    if value is _toml.REMOVE and key not in drawn.spans:
        assert new == old, "nothing to remove, nothing changed"
        return
    if start == len(old) and old and not old[-1].endswith("\n"):
        old[-1] += ending  # the last line ended, to write after it
    head, rest = old[:start], old[end:]
    assert new[:len(head)] == head, "every line before the key's, byte for byte"
    assert new[len(new) - len(rest):] == rest if rest else True, "every line after the key's, byte for byte"
    middle = new[len(head):len(new) - len(rest)]
    if value is _toml.REMOVE:
        assert middle == [], "a removed key leaves nothing of itself"
        return
    if drawn.record == "absent":
        assert middle[:2] == [ending, f"[record]{ending}"], "a table the file lacks is added at its end"
        middle = middle[2:]
    assert all(each.endswith(ending) for each in middle), "new lines end as the file's do"
    assert tomllib.loads("".join(middle)) == {key: value}, "the key's new lines are the key alone"
    assert middle[0].rstrip("\r\n").endswith(comment), "the comment that ended its first line, kept"


def test_the_editors_read_back_refuses_a_splice_that_changed_a_neighbour(monkeypatch: pytest.MonkeyPatch):
    """The net under the splice: an inner edit planted to rewrite another key too is caught by ``edited``'s own reading
    of what it wrote, and refused."""
    text = '[record]\ntitle = "x"\nkeywords = ["a"]\n'
    assert _toml.edited(text, "record", {"title": "y"}) == '[record]\ntitle = "y"\nkeywords = ["a"]\n', \
        "the control: the edit as written"
    real = _toml._edited_key  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(_toml, "_edited_key", lambda lines, table, key, value, ending: [
        each.replace('["a"]', '["b"]') for each in real(lines, table, key, value, ending)])
    with pytest.raises(_toml.Unedited, match="does not hold the edit and every other key as it was"):
        _toml.edited(text, "record", {"title": "y"})


@pytest.mark.parametrize("text", ["", "a = 1", "a = 1\r\n", '[record]\ntitle = "x"  # mine\n'])
def test_a_value_is_written_as_text_toml_reads_back_as_the_value(text: str):
    """The control of the writer's escapes: a string of every control character, DEL and a non-ASCII letter reads back
    as itself."""
    awkward = "".join(chr(code) for code in range(32)) + "\x7fé \"\\"
    after = _toml.edited(text, "record", {"title": awkward})
    assert tomllib.loads(after)["record"]["title"] == awkward
