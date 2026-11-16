"""The reader of unit labels that ``mime.core`` calls where a label is defined (G1: a ``ParamSpec``, a ``node_input``, a
``Units(...)``, a symbol-table row), installed by ``mime.units`` when it is imported.

``mime.core`` imports nothing above itself (``tests/test_import_boundaries.py``), and the label grammar is ``mime.units``'s:
its table of units, its dimensions and its refusal. So the dependency runs the other way, as a registration: ``mime.units``
hands this module its reader, its writer and its error class, and the core's constructors call them here. ``mime`` imports
``mime.units`` before any other of its modules, so a label made anywhere is read; a label made before that is a defect and
fails closed, never passes unread."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_reader: Callable[[str], object] | None = None
_writer: Callable[..., str] | None = None
_error: type[Exception] | None = None


def install_label_reader(reader: Callable[[str], object], writer: Callable[..., str], error: type[Exception]) -> None:
    """Called once by ``mime.units``: ``reader`` parses a label (raising ``error`` for one that is no unit), ``writer``
    writes a label from parts (markers or labels), and ``error`` is the class a bad label raises."""
    global _reader, _writer, _error
    _reader, _writer, _error = reader, writer, error


def _installed() -> tuple[Callable[[str], object], Callable[..., str], type[Exception]]:
    if _reader is None or _writer is None or _error is None:
        raise RuntimeError("a unit label was made before mime.units was imported, so it cannot be read; import mime first")
    return _reader, _writer, _error


def read_label(label: str) -> None:
    """``label`` parsed, or the unit error for one that is not a unit label."""
    _installed()[0](label)


def write_label(*parts: Any) -> str:
    """A label written from dimension markers and labels (``mime.units.unit_label``)."""
    return _installed()[1](*parts)


def label_error() -> type[Exception]:
    """The class a label that is not a unit raises (``mime.units.UnitError``)."""
    return _installed()[2]
