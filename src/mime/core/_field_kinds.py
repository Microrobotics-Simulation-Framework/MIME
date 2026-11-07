"""The kinds of field a node's typed state gives by a field's type, which ``mime verify`` reads where a field's values
mean nothing on some rows: a constraint's held points, judged only on the rows whose forces act (``mime.saved``). A
field of any other type is judged on all of it. The kind is the field's type, never its node's class: a new node whose
typed state writes a field as ``HeldPoints`` is judged so, with nothing declared beside it.

A node's typed state is the closed TypedDict that names exactly the fields it holds (``mime.nodes``'s README), found
among those its class's module and its bases' modules hold (:func:`field_kinds`)."""

from __future__ import annotations

import sys
from collections.abc import Collection, Mapping
from enum import Enum
from functools import cache

from jax import Array
from jaxtyping import Float
from typing_extensions import Annotated, NoExtraItems, NotRequired, ReadOnly, Required, get_args, get_origin, \
    get_type_hints, is_typeddict

#: The point a constraint held, a row per group, in the world (m): a row stands for a contact only where its paired
#: force acts, and elsewhere is wherever the search ended (MIME-ANO-144).
type HeldPoints = Float[Array, "groups 3"]  # noqa: F722 - a jaxtyping shape
#: The force each held point took (N), zero where it was free: the field a ``HeldPoints`` field pairs with.
type HeldForces = Float[Array, "groups"]  # noqa: F722 - a jaxtyping shape


class FieldKind(Enum):
    """How a saved field is judged: on all of it (``OWN``), or, for held points, on the rows whose paired force acts."""

    OWN = "own"
    HELD_POINTS = "held points"
    HELD_FORCES = "held forces"


_KINDS: Mapping[object, FieldKind] = {HeldPoints: FieldKind.HELD_POINTS, HeldForces: FieldKind.HELD_FORCES}
_QUALIFIERS = (ReadOnly, NotRequired, Required)


def _kind_of(hint: object) -> FieldKind:
    while True:
        origin = get_origin(hint)
        if origin in _QUALIFIERS or origin is Annotated:
            hint = get_args(hint)[0]
            continue
        return _KINDS.get(hint, FieldKind.OWN)


def _states_near(node_class: type) -> tuple[type, ...]:
    """The TypedDicts of the modules of ``node_class`` and of its bases, defined there or imported, each once."""
    found: dict[int, type] = {}
    for cls in node_class.__mro__:
        module = sys.modules.get(cls.__module__)
        for value in vars(module).values() if module is not None else ():
            if isinstance(value, type) and is_typeddict(value):
                found.setdefault(id(value), value)
    return tuple(found.values())


def _required(hint: object, total: bool) -> bool:
    """Whether a field is required: ``NotRequired`` or ``Required`` where it says, else the TypedDict's totality. Read
    from the evaluated hint, since a module that postpones its annotations leaves ``__required_keys__`` counting every
    field."""
    while True:
        origin = get_origin(hint)
        if origin is NotRequired:
            return False
        if origin is Required:
            return True
        if origin is ReadOnly or origin is Annotated:
            hint = get_args(hint)[0]
            continue
        return total


@cache
def _typed(state: type) -> tuple[Mapping[str, object], frozenset[str]] | None:
    """A TypedDict's evaluated hints and its required fields; ``None`` where its annotations do not evaluate."""
    try:
        hints = get_type_hints(state, include_extras=True)
    except Exception:  # noqa: BLE001  (a TypedDict whose annotations do not evaluate types none of its fields)
        return None
    total = bool(getattr(state, "__total__", True))
    return hints, frozenset(name for name, hint in hints.items() if _required(hint, total))


def _names_exactly(state: type, fields: frozenset[str]) -> bool:
    typed = _typed(state)
    if typed is None:
        return False
    hints, required = typed
    takes_extra = getattr(state, "__extra_items__", NoExtraItems) is not NoExtraItems
    return required <= fields and (takes_extra or fields <= set(hints))


@cache
def _kinds(node_class: type, fields: frozenset[str]) -> Mapping[str, FieldKind]:
    found: dict[str, set[FieldKind]] = {}
    for state in _states_near(node_class):
        typed = _typed(state)
        if typed is None or not _names_exactly(state, fields):
            continue
        for name, hint in typed[0].items():
            if name in fields:
                found.setdefault(name, set()).add(_kind_of(hint))
    return {name: next(iter(kinds)) if len(kinds) == 1 else FieldKind.OWN
            for name in sorted(fields) for kinds in (found.get(name, {FieldKind.OWN}),)}


def field_kinds(node: object, fields: Collection[str]) -> dict[str, FieldKind]:
    """The kind of each of ``fields``, the fields ``node`` holds, from its typed state: the closed TypedDict, among
    those its class's module and its bases' modules hold, that names exactly those fields (every field it requires is
    held, and every field held is one it names, unless it takes extra items). A field no such TypedDict types, or that
    two of them type differently, is ``OWN``."""
    return dict(_kinds(type(node), frozenset(fields)))


def paired_forces(kinds: Mapping[str, FieldKind]) -> str | None:
    """The field the held points of a typed state pair with: its one ``HeldForces`` field, ``None`` where it has none
    or several."""
    forces = [name for name, kind in kinds.items() if kind is FieldKind.HELD_FORCES]
    return forces[0] if len(forces) == 1 else None


__all__ = ["FieldKind", "HeldForces", "HeldPoints", "field_kinds", "paired_forces"]
