from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, override

from mime.things._prims import WORLD, PrimPath

if TYPE_CHECKING:
    from mime.materials import Record
    from mime.registry import ChainSegment, Drive, Dynamics, FieldProvider, Geometry
    from mime.things._parts import Part
    from mime.things._handles import TypedPath
    from mime.units import Quantity


class ThingKind(Enum):
    STRUCTURE = "structure"
    MEDIUM = "medium"
    BODY = "body"
    APPARATUS = "apparatus"


def _frozen(mapping: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(mapping or {}))


class State(Mapping[str, Any]):
    """A body's initial state, by the names its dynamics declares (``State(position=mm((0, 0, 1)))``). Phase 1 checks
    the names against the dynamics and each value's units."""

    def __init__(self, **values: Any) -> None:
        self._values = dict(values)

    @override
    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    @override
    def __iter__(self):
        return iter(self._values)

    @override
    def __len__(self) -> int:
        return len(self._values)

    @override
    def __repr__(self) -> str:
        return f"State({', '.join(f'{k}={v!r}' for k, v in self._values.items())})"


@dataclass(frozen=True, eq=False)
class Structure:
    """A passive solid: a tube wall, a phantom, a channel wall. ``geometry`` is a registered geometry kind (``Tube``),
    whose regions, surfaces and directions the structure's handle names (``tube.lumen``). Its material arrives with
    materials; with ``dynamics=None`` it is fixed and builds nothing."""

    name: str
    geometry: Geometry | None = None
    dynamics: Dynamics | None = None
    kind = ThingKind.STRUCTURE


@dataclass(frozen=True, eq=False)
class Medium:
    """What fills a region and carries fields: ``fills`` names the region (``fills=tube.lumen``), so its geometry is
    written once, by the structure. ``flow`` is the dynamics of its flow field; ``None`` is quiescent and builds
    nothing. ``material`` is the record its values are read from (``mime.materials.Record``), at the experiment's
    conditions; a value declared in the experiment takes the place of the record's, with its reason."""

    name: str
    fills: TypedPath | str | None = None
    flow: Dynamics | None = None
    material: Record | None = None
    kind = ThingKind.MEDIUM

    @property
    def dynamics(self) -> Dynamics | None:
        return self.flow


@dataclass(frozen=True)
class PermanentMagnet:
    """A part's role as a permanent magnet, a model choice the document holds. Its magnetisation direction, a unit
    vector in the part's frame, is design data: the build reads it from the part's prim (the ``mime:`` schema), and
    ``direction`` declares it here for a part whose prim carries none. Declared in both places, it is refused."""

    direction: tuple[float, float, float] | None = None
    model: ClassVar[str] = "permanent_magnet"


@dataclass(frozen=True, eq=False)
class Body:
    """A robot or particle: its motion dynamics and its initial state.

    ``prim`` binds it to a prim the scene already holds (``scene["/World/UMR"]``, from a referenced asset), and its
    name then defaults to the prim's last element; without one it is written Python-first, at ``/World/<name>``.
    ``magnets`` gives parts their role, by name: for a bound body, a part's path below its prim (``{"Magnets":
    PermanentMagnet()}``); for one written Python-first, a part it declares in ``parts``.

    ``parts`` are the body's parts (``mime.things.Part``), each a solid with a material: from them the build derives
    the body's mass, inertia, buoyant volume, centre of buoyancy and magnetic moment, and a value declared in the
    experiment takes the place of a derived one, with its reason."""

    name: str = ""
    dynamics: Dynamics = field(kw_only=True)
    initial: State = field(default_factory=State, kw_only=True)
    prim: PrimPath | str | None = field(default=None, kw_only=True)
    magnets: Mapping[str, PermanentMagnet] = field(default_factory=dict[str, PermanentMagnet], kw_only=True)
    parts: Mapping[str, Part] = field(default_factory=dict[str, "Part"], kw_only=True)
    kind = ThingKind.BODY

    def __post_init__(self) -> None:
        if not self.name and self.prim is not None:
            object.__setattr__(self, "name", PrimPath(str(self.prim)).leaf)
        object.__setattr__(self, "magnets", _frozen(self.magnets))
        object.__setattr__(self, "parts", _frozen(self.parts))


@dataclass(frozen=True, eq=False)
class Apparatus:
    """What drives the experiment. ``fields`` maps a field's name to its provider (``fields={"magnetic": ...}``), and
    ``commands`` each command it takes to the drive that writes it (``commands={"frequency": Hz(10)}``, a quantity
    being short for ``drives.Constant(Hz(10))``): its providers' commands by name, its chain segments' as
    ``<segment>.<command>`` (``"motor.velocity"``).

    ``chain`` maps each segment's name to its dynamics (``mime.registry.ChainSegment``), in chain order: the first is
    placed in the world, each later one mounts on a frame of the one before it. ``parts`` are its parts, each a solid
    fixed to a frame of a segment (``Part(..., mount="motor.rotor")``), or to the world without a ``mount``, placed by
    ``at`` in that frame and not turned in it; ``magnets`` gives parts a magnet's role by name, with the direction
    declared in the part's frame, since a part declared in the experiment has no prim to carry it. Its driver and
    calibration arrive with the first apparatus that has them."""

    name: str
    fields: Mapping[str, FieldProvider] = field(default_factory=dict[str, "FieldProvider"])
    commands: Mapping[str, Drive | Quantity] = field(default_factory=dict[str, "Drive | Quantity"])
    chain: Mapping[str, ChainSegment] = field(default_factory=dict[str, "ChainSegment"], kw_only=True)
    parts: Mapping[str, Part] = field(default_factory=dict[str, "Part"], kw_only=True)
    magnets: Mapping[str, PermanentMagnet] = field(default_factory=dict[str, PermanentMagnet], kw_only=True)
    kind = ThingKind.APPARATUS

    def __post_init__(self) -> None:
        from mime.drives import as_drive  # the drives sit above the things they drive

        object.__setattr__(self, "fields", _frozen(self.fields))
        object.__setattr__(self, "commands", _frozen({name: as_drive(value) for name, value in
                                                      dict(self.commands or {}).items()}))
        object.__setattr__(self, "chain", _frozen(self.chain))
        object.__setattr__(self, "parts", _frozen(self.parts))
        object.__setattr__(self, "magnets", _frozen(self.magnets))

    @property
    def dynamics(self) -> None:
        return None


Thing = Structure | Medium | Body | Apparatus


def prim_path_of(thing: Thing) -> str:
    """The scene path of ``thing``: the prim a body is bound to, or ``/World/<name>`` for a thing written
    Python-first."""
    if isinstance(thing, Body) and thing.prim is not None:
        return str(thing.prim)
    return f"{WORLD}/{thing.name}"
