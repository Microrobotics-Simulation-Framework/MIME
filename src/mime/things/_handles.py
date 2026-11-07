from __future__ import annotations

import difflib
from collections.abc import Callable, Iterable
from dataclasses import fields, is_dataclass
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


class CommandPath(TypedPath):
    """A command of an apparatus, its value as a study or an override sets it: ``field.commands.frequency``,
    ``(apparatus, "commands", command)``."""

    __slots__ = ()

    @property
    def owner(self) -> str:
        return self.parts[0]

    @property
    def command(self) -> str:
        return self.parts[-1]


class InitialPath(TypedPath):
    """A value of a body's initial state: ``umr.initial.position``, ``(body, "initial", name)``."""

    __slots__ = ()

    @property
    def owner(self) -> str:
        return self.parts[0]

    @property
    def name(self) -> str:
        return self.parts[-1]


class SettingPath(TypedPath):
    """A setting of an attached law, by the id ``attach`` returned: ``drag.spacing`` is ``("drag:umr", "spacing")``,
    and a setting of a group of settings ``drag.lubrication.epsilon``. It saves as its dotted string,
    ``drag:umr.lubrication.epsilon``. ``value`` is what the law declares there, which a group's settings are read
    from."""

    __slots__ = ("value",)

    def __init__(self, *parts: str, value: object = None) -> None:
        super().__init__(*parts)
        self.value = value

    @property
    def owner(self) -> str:
        return self.parts[0]

    @property
    def setting(self) -> str:
        return ".".join(self.parts[1:])

    def __getattr__(self, name: str) -> SettingPath:
        if name.startswith("_") or name == "value":
            raise AttributeError(name)
        held = object.__getattribute__(self, "value")
        names = _settings_of(held)
        if name in names:
            return SettingPath(*self.parts, name, value=getattr(held, name))
        raise AttributeError(f"the setting {self} has no setting {name!r}. {did_you_mean(name, names)}")


def _settings_of(value: object) -> tuple[str, ...]:
    """The settings of an entry or a group of settings: its dataclass fields."""
    return tuple(held.name for held in fields(value)) if is_dataclass(value) and not isinstance(value, type) else ()


class _Names:
    """A handle's namespace of typed paths (``field.commands``, ``umr.initial``): each name it holds is an attribute,
    and any other is an ``AttributeError`` on the line that names it."""

    __slots__ = ("_make", "_names", "_what")

    def __init__(self, names: Iterable[str], make: Callable[[str], TypedPath], what: str) -> None:
        self._names, self._make, self._what = tuple(names), make, what

    def __getattr__(self, name: str) -> TypedPath:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._names:
            return self._make(name)
        raise AttributeError(f"{self._what} has no {name!r}. {did_you_mean(name, self._names)}")

    @override
    def __dir__(self) -> list[str]:
        return list(self._names)


class BodyHandle(Handle):
    """Its initial state's values are typed paths, as a study and an override set them: ``umr.initial.position``."""

    __slots__ = ("_initial",)
    thing_kind = ThingKind.BODY

    def __init__(self, name: str, initial: Iterable[str] = ()) -> None:
        super().__init__(name)
        self._initial = tuple(initial)

    @property
    def initial(self) -> _Names:
        def path(key: str) -> TypedPath:
            return InitialPath(self.name, "initial", key)

        return _Names(self._initial, path, f"body {self.name!r}'s initial state")


class ApparatusHandle(Handle):
    """Its fields are typed paths: ``source.push`` is a :class:`FieldPath`, and an unknown field is an
    ``AttributeError`` on the line that names it. Its commands are typed paths in a namespace of their own,
    ``field.commands.frequency`` (:class:`CommandPath`), so that no field's name collides with them: ``commands`` is
    reserved, and no field takes that name."""

    __slots__ = ("_fields", "_commands")
    thing_kind = ThingKind.APPARATUS

    def __init__(self, name: str, fields: Iterable[str], commands: Iterable[str] = ()) -> None:
        super().__init__(name)
        self._fields = tuple(fields)
        self._commands = tuple(commands)

    @property
    def commands(self) -> _Names:
        def path(command: str) -> TypedPath:
            return CommandPath(self.name, "commands", command)

        return _Names(self._commands, path, f"apparatus {self.name!r}'s commands")

    def __getattr__(self, field: str) -> FieldPath:
        if field.startswith("_"):
            raise AttributeError(field)
        if field in self._fields:
            return FieldPath(self.name, field)
        raise AttributeError(f"apparatus {self.name!r} has no field {field!r}. {did_you_mean(field, self._fields)}")


class LawHandle(str):
    """What ``attach`` returns: the law's id, a ``str`` (``"drag:umr"``) as it always was, carrying the law's
    settings as typed paths (``drag.spacing``, ``drag.lubrication.epsilon``: :class:`SettingPath`), which a study and
    an override set."""

    law: object

    def __new__(cls, id: str, law: object = None) -> LawHandle:
        held = super().__new__(cls, id)
        held.law = law
        return held

    def __getattr__(self, name: str) -> SettingPath:
        if name.startswith("_"):
            raise AttributeError(name)
        law = object.__getattribute__(self, "law")
        names = _settings_of(law)
        if name in names:
            return SettingPath(str(self), name, value=getattr(law, name))
        raise AttributeError(f"the law {str(self)!r} has no setting {name!r}. {did_you_mean(name, names)}")
