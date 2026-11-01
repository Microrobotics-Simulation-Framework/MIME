"""What the stage writer is given, taken from phase 1's plans: each thing's prim, a structure's geometry kind and each
part's of a body written Python-first, each with its shape settings in SI, its transform and how it is shown (its
native form, or its drawing at the writer's resolution, with its regions' prims), a body's initial pose, the
referenced layers.

The stage writer has versions, as the saved form has, and a lock records the one that wrote its stage
(:data:`STAGE_WRITER`, the lock's ``stage.writer``; a lock without it was written by version 1). Version 2 (dev1
phase 7n) places every frame of a chain segment in the world, draws each description's visuals and each field a law
reads at a body, and draws a kind's proxy within :data:`PROXY_TRIANGLES`. Version 1 is the writer before it: a chain's
frames only where a part is mounted, at their pose in their segment, no visuals or vectors, and every proxy at the
writer's full resolution. A saved folder is rebuilt by the writer its lock names, so a stage version 1 wrote is held to
version 1's bytes."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from mime.build._plan import PlanSet, composed
from mime.build._refusals import BuildRefused, entry_defect
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
    STAGE_WRITER,
    KindPrim,
    MountedPrim,
    RegionPrim,
    StageSpec,
    ThingPrim,
    VectorPrim,
    VisualPrim,
    kind_frame,
    mesh_source,
    shape_settings,
)
from mime.things import ThingKind
from mime.units import Magnitude, unit_symbol
from mime.vocabulary import FRAME_POSES, POSE, lookup

#: The region primitives with a USD primitive of their own, written as a prim below their kind's.
STAGED_PRIMITIVES = ("sphere", "cylinder")


def region_prim_path(kind_path: str, geometry: Geometry, region: str, primitive: RegionPrimitive) -> str | None:
    """The prim the stage writer writes ``region`` of a kind at ``kind_path`` as: the kind's own prim where its native
    form is that primitive (a sphere's solid), a prim below it for a sphere or a cylinder of a kind with no native form
    (the tube's lumen), and none for a primitive USD has no form of (a slab)."""
    native = type(geometry).native
    if native is not None:
        return kind_path if native.primitive.lower() == primitive.primitive else None
    return f"{kind_path}/{region}" if primitive.primitive in STAGED_PRIMITIVES else None


def _units(record: Registered) -> dict[str, str]:
    units: dict[str, str] = {}
    for setting, declared in record.settings.items():
        if declared.kind is Direction:
            units[setting] = "1"
        elif declared.dimension is not None:
            units[setting] = unit_symbol(declared.dimension)
    return units


def _defect(record: Registered, problem: str, error: BaseException | None = None) -> BuildRefused:
    from mime.errors import raised_at

    return BuildRefused([entry_defect(record.cls, problem, where=raised_at(error) if error is not None else None)])


def _called[T](record: Registered, what: str, call: Callable[[], T]) -> T:
    try:
        return call()
    except Exception as error:  # noqa: BLE001 - an entry's defect becomes a refusal naming it
        raise _defect(record, f"{what} raised {error!r}", error) from error


#: The most triangles a kind's drawn proxy holds in the stage: a drawing over it is drawn again at fewer facets a turn
#: (down to :data:`FEWEST_SEGMENTS`) until it fits. A presentation budget, with its basis: the screw's proxy at the
#: writer's full resolution was nearly all of A0's saved stage, which every run folder carries, and a viewport gains
#: nothing from triangles smaller than its pixels (the 0.2 line's shimmering lesson).
PROXY_TRIANGLES = 2_500
FEWEST_SEGMENTS = 16


def _triangles(drawing: Drawing) -> int:
    return sum(max(0, count - 2) for count in drawing.face_vertex_counts)


def _resolutions(segments: int) -> list[int]:
    """The facets a turn a drawing is tried at: the writer's, then fewer, a multiple of four each, to the fewest."""
    fewer = range(segments - segments % 4 - (4 if segments % 4 == 0 else 0), FEWEST_SEGMENTS - 1, -4)
    return [segments, *(n for n in fewer if n >= FEWEST_SEGMENTS)]


def kind_prim(geometry: Geometry, record: Registered, settings: Mapping[str, Magnitude], *,
              segments: int = DRAWING_SEGMENTS, at: tuple[float, float, float] = (0.0, 0.0, 0.0),
              regions: bool = False, opacity: float | None = None, budget: int | None = PROXY_TRIANGLES) -> KindPrim:
    """A kind as the stage writes it, its origin moved by ``at`` (a part's place in its body), drawn at ``segments``
    when it has no native form, or at fewer where that drawing holds more than ``budget`` triangles (the segments it
    was drawn at are recorded on its proxy; ``None``, the first stage writer's, no budget); with ``regions``, each region whose primitive is a sphere
    or a cylinder gets its own prim (a kind with a native form is that prim already). A setting held as a whole
    number (the helix's starts) is written beside the quantities, as a count."""
    translation, orientation = kind_frame(geometry, settings)
    moved = (translation[0] + at[0], translation[1] + at[1], translation[2] + at[2])
    counts = {item.name: float(value) for item in dataclasses.fields(geometry) if item.name not in settings
              and isinstance(value := cast(object, getattr(geometry, item.name)), int) and not isinstance(value, bool)}
    native = type(geometry).native
    drawing: Drawing | None = None
    if native is None and implements(geometry, "drawing"):
        for tried in _resolutions(segments) if budget is not None else [segments]:
            drawn = _called(record, "drawing()", lambda tried=tried: geometry.drawing(settings, tried))
            if not isinstance(cast(object, drawn), Drawing):
                raise _defect(record, f"drawing() returned {type(drawn).__name__}, not a mime.registry.Drawing")
            drawing, segments = drawn, tried
            if budget is None or _triangles(drawn) <= budget:
                break
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


def _parts(plans: PlanSet, name: str, segments: int, budget: int | None) -> dict[str, KindPrim]:
    """The parts of a body written Python-first, each at its centroid in the body's frame; a part that others lie
    inside is drawn translucent, so they show through it."""
    lines = [line for line in plans.materials.parts if line.body == name]
    holding = {line.inside for line in lines if line.inside is not None}
    return {line.name: kind_prim(line.solid, line.registered, line.settings, segments=segments, at=line.centre,
                                 opacity=0.4 if line.name in holding else None, budget=budget) for line in lines}


def stage_spec(plans: PlanSet, *, segments: int = DRAWING_SEGMENTS, writer: int = STAGE_WRITER) -> StageSpec:
    """The stage of a planned experiment as stage writer ``writer`` writes it, its drawn kinds at ``segments`` facets
    a turn (the writer's resolution). A body's initial ``position`` and ``orientation``, where its dynamics takes them,
    place its prim; a body written Python-first is shown by its parts, or with none as a sphere of its declared
    extent. Raises ``BuildRefused`` for a kind whose drawing or region primitives fail."""
    budget = PROXY_TRIANGLES if writer >= 2 else None
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
                             regions=True, budget=budget) if planned is not None else None
        bound = name in plans.scene.bound
        parts = _parts(plans, name, segments, budget) if kind is ThingKind.BODY and not bound else {}
        things.append(ThingPrim(
            plans.scene.prims[name], name, kind.value, bound,
            cast(tuple[float, float, float], tuple(position)) if position is not None else None,
            cast(tuple[float, float, float, float], tuple(orientation)) if orientation is not None else None,
            geometry, extent if isinstance(extent, float) else None, attributes, parts))
    references = tuple((layer.at, layer.file) for layer in plans.scene.layers)
    if writer < 2:
        return StageSpec(plans.frame.up, plans.scene.meters_per_unit, references, tuple(things), _mounted_v1(plans))
    vectors = tuple(VectorPrim(held.prim, held.position, held.unit, held.name, held.metres_per_unit)
                    for held in vector_sources(plans))
    return StageSpec(plans.frame.up, plans.scene.meters_per_unit, references, tuple(things), _mounted(plans),
                     _visuals(plans), vectors)


def _pose(values: Sequence[float]) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    return (values[0], values[1], values[2]), (values[3], values[4], values[5], values[6])


_IDENTITY_POSE = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def frame_prim_names(frames: Sequence[str]) -> dict[str, str]:
    """Each frame's prim name below its segment's: its own name where USD takes it as one, else the name with each
    character USD does not take written ``_`` (and ``_`` before a leading digit), numbered where two would meet. The
    frame's own name is on its prim as ``mime:frame``, and the stage binding maps it to its prim."""
    names: dict[str, str] = {}
    taken: set[str] = set()
    for frame in frames:
        held = frame if _IDENTIFIER.fullmatch(frame) else re.sub(r"[^A-Za-z0-9_]", "_", frame) or "_"
        if held[0].isdigit():
            held = f"_{held}"
        unique, count = held, 2
        while unique in taken:
            unique, count = f"{held}_{count}", count + 1
        taken.add(unique)
        names[frame] = unique
    return names


@dataclass(frozen=True)
class SegmentPrims:
    """A chain segment as the stage holds it: its prim, its frames' prims by frame name, and each frame's pose in the
    world at the start."""

    owner: str
    node: str
    field: str
    thing: str
    root: str
    frames: Mapping[str, str]
    initial: Mapping[str, tuple[float, ...]]


def segment_prims(plans: PlanSet) -> dict[str, SegmentPrims]:
    """Every chain segment's prims, by segment id: its own below its apparatus's prim, its frames' below it."""
    found: dict[str, SegmentPrims] = {}
    for plan in plans.entries:
        if not isinstance(plan.entry, ChainSegment) or plan.segment is None or plan.thing is None:
            continue
        placement = plan.entry.placement(plan.settings)
        base = composed(plan.mount.initial_pose, placement) if plan.mount is not None else placement
        root = f"{plans.scene.prims[plan.thing]}/{plan.owner.partition('.')[2]}"
        names = frame_prim_names(plan.segment.frames)
        initial = {frame: composed(base, plan.segment.initial[index] if len(plan.segment.initial) > index
                                   else _IDENTITY_POSE)
                   for index, frame in enumerate(plan.segment.frames)}
        port = plan.record.publishes.get(FRAME_POSES.name) if plan.record is not None else None
        found[plan.owner] = SegmentPrims(plan.owner, plan.node_name(port.node if port is not None else ""),
                                         port.field if port is not None else "frame_poses", plan.thing, root,
                                         {frame: f"{root}/{name}" for frame, name in names.items()}, initial)
    return found


@dataclass(frozen=True)
class BodyPose:
    """A body's prim and the graph's node and field that hold its pose."""

    thing: str
    prim: str
    node: str
    field: str


def body_poses(plans: PlanSet) -> list[BodyPose]:
    """Every body whose dynamics publishes its pose, in the plans' order."""
    found: list[BodyPose] = []
    for plan in plans.entries:
        if plan.thing_kind is not ThingKind.BODY or plan.thing is None or plan.record is None:
            continue
        port = plan.record.publishes.get(POSE.name)
        if port is not None and plan.thing in plans.scene.prims:
            found.append(BodyPose(plan.thing, plans.scene.prims[plan.thing], plan.node_name(port.node), port.field))
    return found


#: An arrow's length, in metres, for a field of its interface's typical size at a body of unknown extent: a
#: presentation choice (the arrow of a field twice as strong is twice as long).
ARROW_LENGTH = 0.01


@dataclass(frozen=True)
class VectorSource:
    """A field a law reads at a body, a vector in the world's axes, as the stage draws it: its prim below the field's
    apparatus, the provider's node and the output it reads, the apparatus, its public name, its units, the body pose it
    is drawn at, the body's position at the start, and how long its arrow is per unit of the field (m per unit)."""

    prim: str
    node: str
    field: str
    thing: str
    name: str
    unit: str
    at: BodyPose
    position: tuple[float, float, float]
    metres_per_unit: float


def vector_sources(plans: PlanSet) -> list[VectorSource]:
    """Every field read at a body's pose that is a vector in the world's axes (a flux density at the body), once each:
    a law's read with a reader slot, whose points are the body's pose."""
    bodies = {(held.node, held.field): held for held in body_poses(plans)}
    initial = {plan.thing: plan.initial for plan in plans.entries if plan.thing is not None and plan.thing_kind is
               ThingKind.BODY and plan.initial}
    extents = {line.path.partition(".")[0]: line.value for line in plans.values if line.path.endswith(".extent")}
    found: list[VectorSource] = []
    seen: set[tuple[str, str]] = set()
    for plan in plans.entries:
        for read in plan.reads.values():
            if read.slot is None or read.points is None or (read.node, read.field) in seen:
                continue
            body = bodies.get((read.points.node, read.points.field))
            declared = lookup(read.interface)
            provider = next((p for p in plans.entries if read.node in p.node_names()), None)
            if body is None or declared is None or declared.shape != (3,) or declared.frame != "world" or \
                    provider is None or provider.thing is None:
                continue
            seen.add((read.node, read.field))
            extent = extents.get(body.thing)
            length = 4.0 * extent if isinstance(extent, float) and extent > 0.0 else ARROW_LENGTH
            start = cast(tuple[float, ...], initial.get(body.thing, {}).get("position", (0.0, 0.0, 0.0)))
            field_name = provider.owner.partition(".")[2] or provider.owner
            found.append(VectorSource(f"{plans.scene.prims[provider.thing]}/{field_name}/{read.slot}", read.node,
                                      read.field, provider.thing, f"{provider.owner}.{read.slot}", declared.units,
                                      body, (float(start[0]), float(start[1]), float(start[2])),
                                      length / (declared.magnitude or 1.0)))
    return found


def _mounted(plans: PlanSet) -> tuple[MountedPrim, ...]:
    """Each segment of each apparatus's chain at its base's pose at the start, below the apparatus's prim; every frame
    of it below that, placed in the world at its pose at the start; and each part by its authored transform in the
    frame it is mounted on."""
    found: list[MountedPrim] = []
    segments = {plan.owner: plan for plan in plans.entries if isinstance(plan.entry, ChainSegment)}
    prims = segment_prims(plans)
    magnets = {(m.body, m.key): m.in_part for m in plans.scene.magnets}
    for owner, plan in segments.items():
        held = cast(ChainSegment, plan.entry)
        placement = held.placement(plan.settings)
        base = composed(plan.mount.initial_pose, placement) if plan.mount is not None else placement
        root = f"{plans.scene.prims[plan.thing or '']}/{owner.partition('.')[2]}"
        found.append(MountedPrim(root, *_pose(base), {"segment": plan.describe()}))
        if owner in prims:
            framed = prims[owner]
            found += [MountedPrim(path, *_pose(framed.initial[frame]), {"frame": frame}, world=True)
                      for frame, path in framed.frames.items()]
    for apparatus, parts in plans.apparatus_parts.items():
        prim = plans.scene.prims[apparatus]
        for key, part in parts.items():
            mount = part.mount
            if mount is None:
                path = f"{prim}/{key}"
            elif mount.segment in prims and mount.frame in prims[mount.segment].frames:
                path = f"{prims[mount.segment].frames[mount.frame]}/{key}"
            else:
                path = f"{prim}/{mount.segment.partition('.')[2]}/{mount.frame}/{key}"
            on = f"{mount.segment}.{mount.frame}" if mount is not None else "the world"
            found.append(MountedPrim(path, *_pose(part.offset), {"mount": on, "solid": part.solid},
                                     magnets.get((apparatus, key))))
    return tuple(found)


def _mounted_v1(plans: PlanSet) -> tuple[MountedPrim, ...]:
    """The first stage writer's chain: each segment at its base's pose at the start, below the apparatus's prim; each
    frame a part is mounted on below its segment's, at its pose in the segment's base frame; and each part by its
    authored transform there."""
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
                    local = plan.segment.initial[mount.index] if plan.segment.initial else _IDENTITY_POSE
                    found.append(MountedPrim(frame, *_pose(local), {"frame": mount.frame}))
                path = f"{frame}/{key}"
            on = f"{mount.segment}.{mount.frame}" if mount is not None else "the world"
            found.append(MountedPrim(path, *_pose(part.offset), {"mount": on, "solid": part.solid},
                                     magnets.get((apparatus, key))))
    return tuple(found)


def _visuals(plans: PlanSet) -> tuple[VisualPrim, ...]:
    """What the stage draws on each frame of each segment: its description's visuals below ``visuals/``, a mesh named
    by its visual set and file sha256 where a set lists it, and the simple shapes that stand in for meshes below
    ``fallback/``."""
    found: list[VisualPrim] = []
    prims = segment_prims(plans)
    for plan in plans.entries:
        if plan.segment is None or plan.owner not in prims:
            continue
        frames = prims[plan.owner].frames
        counts: dict[tuple[str, str], int] = {}
        for visual in plan.segment.visuals:
            if visual.frame not in frames:
                continue
            group = "visuals" if visual.role == "visual" else "fallback"
            index = counts.get((visual.frame, group), 0)
            counts[(visual.frame, group)] = index + 1
            name = {"visual": "visual", "collision": "collision", "frames": "bone"}.get(visual.role, "shape")
            source = mesh_source(visual.mesh) if visual.kind == "mesh" and visual.mesh is not None else None
            translation, orientation = _pose(visual.origin)
            found.append(VisualPrim(
                f"{frames[visual.frame]}/{group}/{name}_{index}", visual.kind, visual.role, translation, orientation,
                visual.scale, visual.size, visual.radius, visual.length, visual.end, visual.colour, visual.mesh,
                source[0].name if source is not None else None, source[1].sha256 if source is not None else None))
    return tuple(found)
