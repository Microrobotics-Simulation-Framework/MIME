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


class GeometryPath(TypedPath):
    """A named part of a thing's geometry: ``(thing, name)``."""

    __slots__ = ()

    @property
    def owner(self) -> str:
        return self.parts[0]

    @property
    def name(self) -> str:
        return self.parts[-1]


class RegionPath(GeometryPath):
    """A region of a thing's geometry (``tube.lumen``), which a medium fills."""

    __slots__ = ()


class SurfacePath(GeometryPath):
    """A surface of a thing's geometry (``tube.inner_surface``), facing the region it bounds; a contact binds one."""

    __slots__ = ()


class DirectionPath(GeometryPath):
    """A direction of a thing's geometry (``tube.axis``), which a setting may refer to rather than repeat."""

    __slots__ = ()


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
    """Its geometry's regions, surfaces and directions are typed paths: ``tube.lumen``, ``tube.inner_surface`` and
    ``tube.axis``; a name its geometry does not have is an ``AttributeError`` on the line that names it."""

    __slots__ = ("_paths",)
    thing_kind = ThingKind.STRUCTURE

    def __init__(self, name: str, regions: Iterable[str] = (), surfaces: Iterable[str] = (),
                 directions: Iterable[str] = ()) -> None:
        super().__init__(name)
        paths: dict[str, type[GeometryPath]] = {}
        for kind, names in ((RegionPath, regions), (SurfacePath, surfaces), (DirectionPath, directions)):
            paths.update({part: kind for part in names})
        self._paths = paths

    def __getattr__(self, part: str) -> GeometryPath:
        if part.startswith("_"):
            raise AttributeError(part)
        if part in self._paths:
            return self._paths[part](self.name, part)
        raise AttributeError(f"structure {self.name!r} has no region, surface or direction {part!r}. "
                             f"{did_you_mean(part, self._paths)}")


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
