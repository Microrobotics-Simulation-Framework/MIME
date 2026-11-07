"""The kinds of field a node's typed state gives by a field's type, which ``mime verify`` reads where a field's values
mean nothing on some rows: a constraint's held points, judged only on the rows whose forces act (``mime.saved``). A
field of any other type is judged on all of it. The kind is the field's type, never its node's class: a new node whose
typed state writes a field as ``HeldPoints`` is judged so, with nothing declared beside it.

A node's typed state is the closed TypedDict that names exactly the fields it holds (``mime.nodes``'s README), found
among those its class's module and its bases' modules hold (:func:`field_kinds`).

The typed state also declares a field's units where nothing else does (no row of the node's symbol table, no interface
an edge carries it as): beside its type, ``stopped_pose: ReadOnly[Annotated[Float[Array, "7"], Units("m; 1")]]``
(:func:`typed_units`). ``verify`` splits a field into blocks by its units, so a field that declares none is judged
whole."""

from __future__ import annotations

import sys
from collections.abc import Collection, Mapping
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Units:
    """A state field's units, declared beside its type in its node's typed state: ``Annotated[HeldPoints,
    Units("m")]``. A composite lists its parts in order, separated by semicolons, its leading parts vectors of three
    (``"m; 1"`` for a pose), as the symbol table and the vocabulary spell them."""

    text: str


_KINDS: Mapping[object, FieldKind] = {HeldPoints: FieldKind.HELD_POINTS, HeldForces: FieldKind.HELD_FORCES}
_QUALIFIERS = (ReadOnly, NotRequired, Required)


def _kind_of(hint: object) -> FieldKind:
    while True:
        origin = get_origin(hint)
        if origin in _QUALIFIERS or origin is Annotated:
            hint = get_args(hint)[0]
            continue
        return _KINDS.get(hint, FieldKind.OWN)


def _units_of(hint: object) -> str | None:
    while True:
        origin = get_origin(hint)
        if origin in _QUALIFIERS:
            hint = get_args(hint)[0]
            continue
        if origin is Annotated:
            held = [mark.text for mark in getattr(hint, "__metadata__", ()) if isinstance(mark, Units)]
            if held:
                return held[0]
            hint = get_args(hint)[0]
            continue
        return None


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


@cache
def _units(node_class: type, fields: frozenset[str]) -> Mapping[str, str]:
    found: dict[str, set[str]] = {}
    for state in _states_near(node_class):
        typed = _typed(state)
        if typed is None or not _names_exactly(state, fields):
            continue
        for name, hint in typed[0].items():
            units = _units_of(hint)
            if name in fields and units is not None:
                found.setdefault(name, set()).add(units)
    return {name: next(iter(held)) for name, held in sorted(found.items()) if len(held) == 1}


def typed_units(node: object, fields: Collection[str]) -> dict[str, str]:
    """The units ``node``'s typed state declares beside its fields' types (:class:`Units`), by field, read from the
    TypedDict :func:`field_kinds` reads: the closed one, among those its class's module and its bases' modules hold,
    that names exactly ``fields``. A field none declares, or that two such TypedDicts declare differently, is
    absent."""
    return dict(_units(type(node), frozenset(fields)))


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


__all__ = ["FieldKind", "HeldForces", "HeldPoints", "Units", "field_kinds", "paired_forces", "typed_units"]
