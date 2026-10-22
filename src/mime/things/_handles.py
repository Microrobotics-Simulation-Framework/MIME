from __future__ import annotations

import difflib
from collections.abc import Iterable
from typing import override

from mime.things._things import ThingKind


def did_you_mean(name: str, candidates: Iterable[str]) -> str:
    """The suggestion and candidate list a refusal of an unknown name ends with."""
    pool = sorted(set(candidates))
    if not pool:
        return "There are no candidates."
    close = difflib.get_close_matches(name, pool, n=3, cutoff=0.6)
    hint = f"Did you mean {' or '.join(repr(c) for c in close)}? " if close else ""
    return f"{hint}The candidates are: {', '.join(pool)}."


class TypedPath:
    """A reference deeper than a thing, built from a handle (``source.magnetic``). It saves as its dotted string."""

    __slots__ = ("parts",)

    def __init__(self, *parts: str) -> None:
        self.parts = parts

    @override
    def __str__(self) -> str:
        return ".".join(self.parts)

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self)!r})"

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, TypedPath) and type(other) is type(self) and other.parts == self.parts

    @override
    def __hash__(self) -> int:
        return hash((type(self), self.parts))


class FieldPath(TypedPath):
    """A field of an apparatus: ``(apparatus, field)``."""

    __slots__ = ()

    @property
    def owner(self) -> str:
        return self.parts[0]

    @property
    def field(self) -> str:
        return self.parts[1]


class Handle:
    """What ``add_*`` returns: the thing's name and kind. An entry binds a handle, so a misspelt variable is a
    ``NameError`` on its line; the saved form holds the name."""

    __slots__ = ("name", "kind")
    thing_kind: ThingKind

    def __init__(self, name: str) -> None:
        self.name = name
        self.kind = self.thing_kind

    @override
    def __str__(self) -> str:
        return self.name

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


class StructureHandle(Handle):
    __slots__ = ()
    thing_kind = ThingKind.STRUCTURE


class MediumHandle(Handle):
    __slots__ = ()
    thing_kind = ThingKind.MEDIUM


class BodyHandle(Handle):
    __slots__ = ()
    thing_kind = ThingKind.BODY


class ApparatusHandle(Handle):
    """Its fields are typed paths: ``source.push`` is a :class:`FieldPath`, and an unknown field is an
    ``AttributeError`` on the line that names it."""

    __slots__ = ("_fields",)
    thing_kind = ThingKind.APPARATUS

    def __init__(self, name: str, fields: Iterable[str]) -> None:
        super().__init__(name)
        self._fields = tuple(fields)

    def __getattr__(self, field: str) -> FieldPath:
        if field.startswith("_"):
            raise AttributeError(field)
        if field in self._fields:
            return FieldPath(self.name, field)
        raise AttributeError(f"apparatus {self.name!r} has no field {field!r}. {did_you_mean(field, self._fields)}")
