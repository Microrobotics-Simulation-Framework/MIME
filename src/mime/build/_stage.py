"""What the stage writer is given, taken from phase 1's plans: each thing's prim, a structure's geometry kind and each
part's of a body written Python-first, each with its shape settings in SI, its transform and how it is shown (its
native form, or its drawing at the writer's resolution, with its regions' prims), a body's initial pose, the
referenced layers."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import cast

from mime.build._plan import PlanSet, composed
from mime.build._refusals import MIME_DEFECT, BuildRefused, Refusal, RefusalClass
from mime.registry import (
    ChainSegment,
    Direction,
    Drawing,
    Geometry,
    NoPrimitive,
    RegionPrimitive,
    Registered,
    implements,
)
from mime.scene import (
    DRAWING_SEGMENTS,
    KindPrim,
    MountedPrim,
    RegionPrim,
    StageSpec,
    ThingPrim,
    kind_frame,
    shape_settings,
)
from mime.things import ThingKind
from mime.units import Magnitude, unit_symbol

#: The region primitives with a USD primitive of their own, written as a prim below their kind's.
STAGED_PRIMITIVES = ("sphere", "cylinder")


def _units(record: Registered) -> dict[str, str]:
    units: dict[str, str] = {}
    for setting, declared in record.settings.items():
        if declared.kind is Direction:
            units[setting] = "1"
        elif declared.dimension is not None:
            units[setting] = unit_symbol(declared.dimension)
    return units


def _defect(record: Registered, problem: str) -> BuildRefused:
    cls = record.cls
    return BuildRefused([Refusal("entry.defect", RefusalClass.INTERNAL, f"{cls.label()} ({cls.__module__}): {problem}",
                                 f"a defect in {cls.label()}, defined in {cls.__module__}: report it to that package. "
                                 f"If that package is MIME: {MIME_DEFECT}")])


def _called[T](record: Registered, what: str, call: Callable[[], T]) -> T:
    try:
        return call()
    except Exception as error:  # noqa: BLE001 - an entry's defect becomes a refusal naming it
        raise _defect(record, f"{what} raised {error!r}") from error


def kind_prim(geometry: Geometry, record: Registered, settings: Mapping[str, Magnitude], *,
              segments: int = DRAWING_SEGMENTS, at: tuple[float, float, float] = (0.0, 0.0, 0.0),
              regions: bool = False, opacity: float | None = None) -> KindPrim:
    """A kind as the stage writes it, its origin moved by ``at`` (a part's place in its body), drawn at ``segments``
    when it has no native form; with ``regions``, each region whose primitive is a sphere or a cylinder gets its own
    prim (a kind with a native form is that prim already). A setting held as a whole number (the helix's starts) is
    written beside the quantities, as a count."""
    translation, orientation = kind_frame(geometry, settings)
    moved = (translation[0] + at[0], translation[1] + at[1], translation[2] + at[2])
    counts = {item.name: float(value) for item in dataclasses.fields(geometry) if item.name not in settings
              and isinstance(value := cast(object, getattr(geometry, item.name)), int) and not isinstance(value, bool)}
    native = type(geometry).native
    drawing: Drawing | None = None
    if native is None and implements(geometry, "drawing"):
        drawn = _called(record, "drawing()", lambda: geometry.drawing(settings, segments))
        if not isinstance(cast(object, drawn), Drawing):
            raise _defect(record, f"drawing() returned {type(drawn).__name__}, not a mime.registry.Drawing")
        drawing = drawn
    staged: list[RegionPrim] = []
    if regions and native is None and implements(geometry, "region_primitive"):
        for region in geometry.regions:
            primitive = _called(record, f"region_primitive({region!r})",
                                lambda region=region: geometry.region_primitive(region, settings))
            if isinstance(primitive, RegionPrimitive) and primitive.primitive in STAGED_PRIMITIVES:
                staged.append(RegionPrim(region, primitive))
            elif not isinstance(cast(object, primitive), (RegionPrimitive, NoPrimitive)):
                raise _defect(record, f"region_primitive({region!r}) returned {type(primitive).__name__}")
    return KindPrim(record.name, record.version, {**shape_settings(geometry, settings), **counts},
                    {**_units(record), **dict.fromkeys(counts, "1")}, moved, orientation, native, drawing, segments,
                    tuple(staged), opacity)


def _parts(plans: PlanSet, name: str, segments: int) -> dict[str, KindPrim]:
    """The parts of a body written Python-first, each at its centroid in the body's frame; a part that others lie
    inside is drawn translucent, so they show through it."""
    lines = [line for line in plans.materials.parts if line.body == name]
    holding = {line.inside for line in lines if line.inside is not None}
    return {line.name: kind_prim(line.solid, line.registered, line.settings, segments=segments, at=line.centre,
                                 opacity=0.4 if line.name in holding else None) for line in lines}


def stage_spec(plans: PlanSet, *, segments: int = DRAWING_SEGMENTS) -> StageSpec:
    """The stage of a planned experiment, its drawn kinds at ``segments`` facets a turn (the writer's resolution). A
    body's initial ``position`` and ``orientation``, where its dynamics takes them, place its prim; a body written
    Python-first is shown by its parts, or with none as a sphere of its declared extent. Raises ``BuildRefused`` for a
    kind whose drawing or region primitives fail."""
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
        planned = plans.geometries.get(name)
        geometry = kind_prim(planned.geometry, planned.registered, planned.settings, segments=segments,
                             regions=True) if planned is not None else None
        bound = name in plans.scene.bound
        parts = _parts(plans, name, segments) if kind is ThingKind.BODY and not bound else {}
        things.append(ThingPrim(
            plans.scene.prims[name], name, kind.value, bound,
            cast(tuple[float, float, float], tuple(position)) if position is not None else None,
            cast(tuple[float, float, float, float], tuple(orientation)) if orientation is not None else None,
            geometry, extent if isinstance(extent, float) else None, attributes, parts))
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
