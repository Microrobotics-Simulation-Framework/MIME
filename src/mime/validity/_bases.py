"""A reason's basis: one type for a waiver's basis and a value's source.

A basis is a text (why, in the experimenter's words), a choice (``chosen(...)``: the value is the experimenter's own,
which overrides its setting's declared role, recorded as a declared change), or pending (``pending()``: the reason is
left for later, which exploratory records and stamps on every output and research refuses by name). a1's named bases
(``basis("name")``, a library pinned by its hash) are one more kind: the saved form keys each kind by its own name
(``reason: {pending: true}``, ``reason: {chosen: ...}``), a text staying a plain string, so a kind is added without a
new shape.

This module imports nothing of MIME's, so that :mod:`mime.units` can carry a basis on a quantity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast, override


@dataclass(frozen=True)
class Pending:
    """A reason left for later (:func:`pending`). Exploratory records it and stamps every output with it; research
    refuses it by name. It is the one placeholder MIME writes and so can tell: a text that only looks like a reason
    (``"TODO"``) passes, until a1's named bases and strict's evidence judge what a reason says."""

    @override
    def __str__(self) -> str:
        return "pending"


@dataclass(frozen=True)
class Chosen:
    """A value the experimenter chose (:func:`chosen`), with why: it is design whatever its setting's declared role,
    since a role depends on use (a tube's radius is physical in a lab's replica and design in a study of it)."""

    text: str

    @override
    def __str__(self) -> str:
        return f"chosen: {self.text}"


#: A reason's basis: a text, a choice, or pending.
type Basis = str | Pending | Chosen

PENDING = Pending()


def pending() -> Pending:
    """A reason left for later: ``declare("umr.mass", kg(8.5e-5), reason=pending())``,
    ``Acceptance(covers=(...), reason=pending())``. Exploratory builds with it and stamps every output; research refuses
    it by name, and ``built.report.pending()`` and ``mime pending <folder>`` list it with the line to change."""
    return PENDING


def chosen(text: str) -> Chosen:
    """A value the experimenter chose, with why: ``Tube(inner_radius=mm(2.0, reason=chosen("the radius this study
    sweeps")))``. It makes the setting design for this experiment, whatever its declared role."""
    return Chosen(text)


def basis_problem(value: object) -> str | None:
    """Why ``value`` is not a basis, or ``None``: a non-empty text, ``chosen(...)`` with its text, or ``pending()``."""
    if isinstance(value, Pending):
        return None
    if isinstance(value, Chosen):
        held = cast(object, value.text)  # a script is not type-checked
        return None if isinstance(held, str) and held.strip() else "chosen(...) is given no text"
    if isinstance(value, str):
        return None if value.strip() else "the reason is an empty text"
    return f"{value!r} is not a reason: write a text, chosen(\"...\"), or pending()"


def is_pending(value: object) -> bool:
    return isinstance(value, Pending)


def basis_text(value: Basis) -> str:
    """The basis as the report and the resolved record print it: a text as it is, ``pending``, ``chosen: <text>``."""
    return value if isinstance(value, str) else str(value)


__all__ = ["PENDING", "Basis", "Chosen", "Pending", "basis_problem", "basis_text", "chosen", "is_pending", "pending"]
