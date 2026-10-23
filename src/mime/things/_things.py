from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, override

if TYPE_CHECKING:
    from mime.registry import Drive, Dynamics, FieldProvider, Geometry
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
    nothing."""

    name: str
    fills: TypedPath | str | None = None
    flow: Dynamics | None = None
    kind = ThingKind.MEDIUM

    @property
    def dynamics(self) -> Dynamics | None:
        return self.flow


@dataclass(frozen=True, eq=False)
class Body:
    """A robot or particle: its motion dynamics and its initial state."""

    name: str
    dynamics: Dynamics
    initial: State = field(default_factory=State)
    kind = ThingKind.BODY


@dataclass(frozen=True, eq=False)
class Apparatus:
    """What drives the experiment. ``fields`` maps a field's name to its provider (``fields={"magnetic": ...}``), and
    ``commands`` each command its providers take to the drive that writes it (``commands={"frequency": Hz(10)}``, a
    quantity being short for ``drives.Constant(Hz(10))``). Its chain, driver and calibration arrive with the first
    apparatus that has them."""

    name: str
    fields: Mapping[str, FieldProvider] = field(default_factory=dict[str, "FieldProvider"])
    commands: Mapping[str, Drive | Quantity] = field(default_factory=dict[str, "Drive | Quantity"])
    kind = ThingKind.APPARATUS

    def __post_init__(self) -> None:
        from mime.drives import as_drive  # the drives sit above the things they drive

        object.__setattr__(self, "fields", _frozen(self.fields))
        object.__setattr__(self, "commands", _frozen({name: as_drive(value) for name, value in
                                                      dict(self.commands or {}).items()}))

    @property
    def dynamics(self) -> None:
        return None


Thing = Structure | Medium | Body | Apparatus
