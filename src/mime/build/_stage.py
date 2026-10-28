"""What the stage writer is given, taken from phase 1's plans: each thing's prim, a structure's geometry with its
settings in SI, a body's initial pose, the referenced layers."""

from __future__ import annotations

from typing import cast

from mime.build._plan import PlanSet, composed
from mime.registry import ChainSegment, Direction
from mime.scene import GeometryPrim, MountedPrim, StageSpec, ThingPrim
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
                     tuple((layer.at, layer.file) for layer in plans.scene.layers), tuple(things), _mounted(plans))


def _pose(values: tuple[float, ...]) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    return (values[0], values[1], values[2]), (values[3], values[4], values[5], values[6])


def _mounted(plans: PlanSet) -> tuple[MountedPrim, ...]:
    """Each segment of each apparatus's chain at its base's pose at the start, each frame a part is mounted on at its
    pose in its segment's base frame, and each part by its authored transform there, below the apparatus's prim."""
    found: list[MountedPrim] = []
    segments = {plan.owner: plan for plan in plans.entries if isinstance(plan.entry, ChainSegment)}
    magnets = {(m.body, m.key): m.in_part for m in plans.scene.magnets}
    for owner, plan in segments.items():
        held = cast(ChainSegment, plan.entry)
        placement = held.placement(plan.settings)
        base = composed(plan.mount.initial_pose, placement) if plan.mount is not None else placement
        root = f"{plans.scene.prims[plan.thing or '']}/{owner.partition('.')[2]}"
        found.append(MountedPrim(root, *_pose(base), {"segment": plan.describe()}))
    for apparatus, parts in plans.apparatus_parts.items():
        prim = plans.scene.prims[apparatus]
        framed: set[str] = set()
        for key, part in parts.items():
            mount = part.mount
            if mount is None:
                path = f"{prim}/{key}"
            else:
                plan = segments[mount.segment]
                frame = f"{prim}/{mount.segment.partition('.')[2]}/{mount.frame}"
                if frame not in framed and plan.segment is not None:
                    framed.add(frame)
                    local = plan.segment.initial[mount.index] if plan.segment.initial else (0.0, 0.0, 0.0, 1.0, 0.0,
                                                                                              0.0, 0.0)
                    found.append(MountedPrim(frame, *_pose(local), {"frame": mount.frame}))
                path = f"{frame}/{key}"
            on = f"{mount.segment}.{mount.frame}" if mount is not None else "the world"
            found.append(MountedPrim(path, *_pose(part.offset), {"mount": on, "solid": part.solid},
                                     magnets.get((apparatus, key))))
    return tuple(found)
