from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, override

if TYPE_CHECKING:
    from mime.registry import Dynamics, FieldProvider


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
    """A passive solid: a tube wall, a phantom, a channel wall. Geometry and material arrive with the scene and
    materials; until a structure dynamics is registered, ``dynamics=None`` builds nothing."""

    name: str
    dynamics: Dynamics | None = None
    kind = ThingKind.STRUCTURE


@dataclass(frozen=True, eq=False)
class Medium:
    """What fills a region and carries fields. ``flow`` is the dynamics of its flow field; ``None`` builds nothing."""

    name: str
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
    """What drives the experiment. ``fields`` maps a field's name to its provider (``fields={"magnetic": ...}``).
    Its chain, driver, calibration and commands arrive with the first apparatus that has them."""

    name: str
    fields: Mapping[str, FieldProvider] = field(default_factory=dict[str, "FieldProvider"])
    kind = ThingKind.APPARATUS

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", _frozen(self.fields))

    @property
    def dynamics(self) -> None:
        return None


Thing = Structure | Medium | Body | Apparatus
