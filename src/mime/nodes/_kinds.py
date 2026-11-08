"""A geometry kind as a node holds it: its registered name and version, and its settings in SI as node constants, from
which the node makes the kind again and evaluates its samples in its step, so that a study's derivative in a setting
flows through them (the MRS rung's node, and the body's node holding its hull's points)."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, cast

from mime.registry import Direction, EntryKind, Geometry, Registered, lookup
from mime.units import Quantity, unit_symbol


def registered_kind(kind: str, version: str) -> Registered:
    held = lookup(EntryKind.GEOMETRY, kind, version)
    if held is None or not issubclass(held.cls, Geometry):
        raise ValueError(f"no geometry kind {kind} {version} is registered; import the package that registers it")
    return held


def kind_from_settings(kind: str, version: str, settings: Mapping[str, object]) -> Geometry:
    """The registered geometry kind ``kind`` made again from its settings in SI, as the node holds them."""
    held = registered_kind(kind, version)
    made: dict[str, object] = {}
    for setting in dataclasses.fields(cast(Any, held.cls)):
        value = settings[setting.name]
        declared = held.settings.get(setting.name)
        if declared is not None and declared.dimension is not None:
            made[setting.name] = Quantity(cast(Any, value), declared.dimension)
        else:
            made[setting.name] = value
    return cast(Geometry, cast(Any, held.cls)(**made))


def traced_settings(held: Registered, settings: Mapping[str, object]) -> tuple[str, ...]:
    """The kind's settings a step takes as leaves: its numbers, not its directions or integers (a part's direction is
    its frame, and turning the body is its pose)."""
    return tuple(name for name, value in settings.items()
                 if isinstance(value, float) and setting_kind(held, name) is not Direction)


def held_as_built(held: Registered, name: str) -> str:
    """The description of a kind's setting a node holds as built rather than as a leaf its step reads (not one of
    :func:`traced_settings`): a direction is its part's frame; any other (a vector of sizes, an integer) the node
    consumes when it is built, so that a new value takes a rebuild, and a study refuses to vary it."""
    if setting_kind(held, name) is Direction:
        return "held as built: a direction of the kind is its part's frame"
    return ("held as built: consumed when the node is built, so that a new value takes a rebuild (a study refuses to "
            "vary it)")


def setting_kind(held: Registered, name: str) -> object:
    declared = held.settings.get(name)
    return None if declared is None else declared.kind


def setting_units(held: Registered, name: str) -> str:
    declared = held.settings.get(name)
    return "1" if declared is None or declared.dimension is None else unit_symbol(declared.dimension)


__all__ = ["held_as_built", "kind_from_settings", "registered_kind", "setting_kind", "setting_units",
           "traced_settings"]
