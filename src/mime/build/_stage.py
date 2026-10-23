"""What the stage writer is given, taken from phase 1's plans: each thing's prim, a structure's geometry with its
settings in SI, a body's initial pose, the referenced layers."""

from __future__ import annotations

from typing import cast

from mime.build._plan import PlanSet
from mime.registry import Direction
from mime.scene import GeometryPrim, StageSpec, ThingPrim
from mime.things import ThingKind
from mime.units import unit_symbol


def _geometry(plans: PlanSet, name: str) -> GeometryPrim | None:
    plan = plans.geometries.get(name)
    if plan is None:
        return None
    units: dict[str, str] = {}
    for setting, declared in plan.registered.settings.items():
        if declared.kind is Direction:
            units[setting] = "1"
        elif declared.dimension is not None:
            units[setting] = unit_symbol(declared.dimension)
    return GeometryPrim(plan.registered.name, plan.registered.version, dict(plan.settings), units, plan.drawing)


def stage_spec(plans: PlanSet) -> StageSpec:
    """The stage of a planned experiment. A body's initial ``position`` and ``orientation``, where its dynamics takes
    them, place its prim; a body written Python-first is shown as a sphere of its declared extent."""
    initial = {plan.thing: plan.initial for plan in plans.entries if plan.thing is not None and plan.thing_kind is
               ThingKind.BODY and plan.initial}
    extents = {line.path.partition(".")[0]: line.value for line in plans.values if line.path.endswith(".extent")}
    things: list[ThingPrim] = []
    for name, kind in plans.things.items():
        start = initial.get(name, {})
        position = start.get("position")
        orientation = start.get("orientation")
        extent = extents.get(name) if kind is ThingKind.BODY else None
        attributes = {"fills": plans.fills[name]} if name in plans.fills else {}
        things.append(ThingPrim(
            plans.scene.prims[name], name, kind.value, name in plans.scene.bound,
            cast(tuple[float, float, float], tuple(position)) if position is not None else None,
            cast(tuple[float, float, float, float], tuple(orientation)) if orientation is not None else None,
            _geometry(plans, name), extent if isinstance(extent, float) else None, attributes))
    return StageSpec(plans.frame.up, plans.scene.meters_per_unit,
                     tuple((layer.at, layer.file) for layer in plans.scene.layers), tuple(things))
