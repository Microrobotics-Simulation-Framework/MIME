"""The stage writer: the scene as one USD layer, ``scene/scene.usda`` in the saved form, written from the declarations
alone so that the same declarations give the same bytes.

What it writes:

- the layer's ``upAxis``, the declared up (+Y or +Z, the only ups a scene declares), and ``metersPerUnit``, a
  ``/World`` root with no transform of its own, and ``defaultPrim`` set to it;
- one prim per thing, ``mime:thing`` (its kind) and ``mime:name`` (its name in the document) on each. A thing written
  Python-first sits at ``/World/<name>``; a body bound to the prim an asset is referenced at takes its pose and those
  attributes on that prim, stronger than the asset's own;
- a structure's geometry kind as the mesh its ``drawing`` gives, with the kind's settings in SI as
  ``mime:geometry:<setting>`` (each with its unit in ``customData``), so the stage records what the physics used;
- a body's initial pose as its transform, and, for a body written Python-first, a sphere of its declared extent;
- an apparatus's chain: each segment's prim below the apparatus's, at its base's pose at the start, each frame a part
  is mounted on below its segment's, and each part below its frame, by its authored transform, a magnet part with
  ``MimeMagnetAPI`` and its direction;
- the references to asset layers, by their path below the stage's folder.

USD stores points as float32: a mesh's points stay in its own frame, near the origin, and the frame is placed by
double-precision ``translate`` and ``orient`` ops. Display colours and opacities are primvars, with no material.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


from mime.registry import Drawing
from mime.scene import _testing
from mime.scene._compose import MAGNET_API, MAGNET_DIRECTION
from mime.scene._frame import UP_AXES
from mime.scene._pxr import Gf, Sdf, Usd, UsdGeom, UsdValidation, Vt
from mime.things import WORLD
from mime.units import Magnitude

_STRUCTURE_COLOUR = (0.85, 0.85, 0.9)
_STRUCTURE_OPACITY = 0.3
_BODY_COLOUR = (0.2, 0.6, 0.8)


@dataclass(frozen=True)
class GeometryPrim:
    """A structure's geometry kind as the stage records it: its registered name and version, its settings in SI with
    each one's unit, and what to draw."""

    name: str
    version: str
    settings: Mapping[str, Magnitude]
    units: Mapping[str, str]
    drawing: Drawing | None


@dataclass(frozen=True)
class ThingPrim:
    """A thing's prim: its path, its document name and kind, whether it binds a prim an asset brings in, its pose
    (a translation in metres and a unit quaternion ``(w, x, y, z)``), and what else the stage shows of it."""

    path: str
    name: str
    kind: str
    bound: bool = False
    translation: tuple[float, float, float] | None = None
    orientation: tuple[float, float, float, float] | None = None
    geometry: GeometryPrim | None = None
    extent: float | None = None
    attributes: Mapping[str, str] = field(default_factory=dict[str, str])


@dataclass(frozen=True)
class MountedPrim:
    """A prim of an apparatus's chain, below the apparatus's prim: a segment, placed at its base's pose at the start; a
    frame of a segment that a part is mounted on, placed at its pose in the segment's base frame at the start; or a
    part, placed by its authored transform in the frame it is mounted on (in the world for a part with no mount).
    ``attributes`` are written as ``mime:<name>``; ``magnet`` is a magnet part's direction in its own frame, written
    with ``MimeMagnetAPI``."""

    path: str
    translation: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    attributes: Mapping[str, str] = field(default_factory=dict[str, str])
    magnet: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class StageSpec:
    """Everything the stage is written from: the declared up (+Y or +Z, a unit vector), the stage's metres per unit,
    each reference as ``(prim path, file below the stage's folder)``, the things' prims, and the prims of the
    apparatus's chains and parts."""

    up: tuple[float, float, float]
    meters_per_unit: float
    references: tuple[tuple[str, str], ...]
    things: tuple[ThingPrim, ...]
    mounted: tuple[MountedPrim, ...] = ()


def up_axis(up: tuple[float, float, float]) -> str:
    """The token the stage declares the unit vector ``up`` with, Y or Z; ``KeyError`` for any other, which the scene
    refuses at its declaration."""
    return UP_AXES[up]


def _quat(q: tuple[float, float, float, float]) -> Any:
    return Gf.Quatd(q[0], Gf.Vec3d(q[1], q[2], q[3]))


def _place(prim: Any, translation: tuple[float, float, float] | None,
           orientation: tuple[float, float, float, float] | None, scale: float) -> None:
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    if translation is not None:
        xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(*(c / scale for c in translation)))
    if orientation is not None:
        xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(_quat(orientation))


def _display(gprim: Any, colour: tuple[float, float, float], opacity: float | None = None) -> None:
    gprim.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*colour)]))
    if opacity is not None:
        gprim.CreateDisplayOpacityAttr(Vt.FloatArray([opacity]))


def _setting(prim: Any, name: str, value: Magnitude, unit: str) -> None:
    if isinstance(value, tuple):
        if len(value) != 3:
            return
        attribute = prim.CreateAttribute(f"mime:geometry:{name}", Sdf.ValueTypeNames.Double3)
        attribute.Set(Gf.Vec3d(*value))
    else:
        attribute = prim.CreateAttribute(f"mime:geometry:{name}", Sdf.ValueTypeNames.Double)
        attribute.Set(float(value))
    attribute.SetCustomDataByKey("units", unit)


def _mesh(stage: Any, path: str, drawing: Drawing, scale: float) -> Any:
    mesh = UsdGeom.Mesh.Define(stage, path)
    points = [Gf.Vec3f(*(c / scale for c in point)) for point in drawing.points]
    mesh.CreatePointsAttr(Vt.Vec3fArray(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(list(drawing.face_vertex_counts)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(list(drawing.face_vertex_indices)))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    lows = [min(float(p[i]) for p in points) for i in range(3)]
    highs = [max(float(p[i]) for p in points) for i in range(3)]
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*lows), Gf.Vec3f(*highs)]))
    _display(mesh, _STRUCTURE_COLOUR, _STRUCTURE_OPACITY)
    _place(mesh.GetPrim(), drawing.translation, drawing.orientation, scale)
    return mesh.GetPrim()


def _thing(stage: Any, thing: ThingPrim, scale: float) -> None:
    if thing.bound:
        prim = stage.GetPrimAtPath(thing.path)
        _place(prim, thing.translation, thing.orientation, scale)
    elif thing.geometry is not None and thing.geometry.drawing is not None:
        prim = _mesh(stage, thing.path, thing.geometry.drawing, scale)
        prim.CreateAttribute("mime:geometry:representation", Sdf.ValueTypeNames.String).Set(
            thing.geometry.drawing.representation)
    elif thing.translation is not None or thing.orientation is not None:
        prim = UsdGeom.Xform.Define(stage, thing.path).GetPrim()
        _place(prim, thing.translation, thing.orientation, scale)
    else:
        prim = UsdGeom.Scope.Define(stage, thing.path).GetPrim()
    prim.CreateAttribute("mime:thing", Sdf.ValueTypeNames.Token).Set(thing.kind)
    prim.CreateAttribute("mime:name", Sdf.ValueTypeNames.String).Set(thing.name)
    for name, value in sorted(thing.attributes.items()):
        prim.CreateAttribute(f"mime:{name}", Sdf.ValueTypeNames.String).Set(value)
    if thing.geometry is not None:
        prim.CreateAttribute("mime:geometry", Sdf.ValueTypeNames.Token).Set(thing.geometry.name)
        prim.CreateAttribute("mime:geometry:version", Sdf.ValueTypeNames.String).Set(thing.geometry.version)
        for name, value in sorted(thing.geometry.settings.items()):
            _setting(prim, name, value, thing.geometry.units.get(name, ""))
    if thing.extent is not None and not thing.bound:
        sphere = UsdGeom.Sphere.Define(stage, f"{thing.path}/extent")
        radius = thing.extent / scale
        sphere.CreateRadiusAttr(radius)
        sphere.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(-radius, -radius, -radius), Gf.Vec3f(radius, radius,
                                                                                             radius)]))
        _display(sphere, _BODY_COLOUR)


def _mounted(stage: Any, mounted: MountedPrim, scale: float) -> None:
    prim = UsdGeom.Xform.Define(stage, mounted.path).GetPrim()
    _place(prim, mounted.translation, mounted.orientation, scale)
    for name, value in sorted(mounted.attributes.items()):
        prim.CreateAttribute(f"mime:{name}", Sdf.ValueTypeNames.String).Set(value)
    if mounted.magnet is not None:
        prim.AddAppliedSchema(MAGNET_API)
        prim.CreateAttribute(MAGNET_DIRECTION, Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(*mounted.magnet))


def stage_text(spec: StageSpec) -> str:
    """The stage's root layer as ``.usda`` text: byte for byte the same for the same ``spec``."""
    layer = Sdf.Layer.CreateAnonymous(".usda")
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageUpAxis(stage, up_axis(spec.up))
    UsdGeom.SetStageMetersPerUnit(stage, spec.meters_per_unit)
    world = UsdGeom.Xform.Define(stage, WORLD)
    if not _testing.omitting_default_prim():
        stage.SetDefaultPrim(world.GetPrim())
    references = {at for at, _ in spec.references}
    for at in sorted(references):
        stage.DefinePrim(at)
    for thing in sorted(spec.things, key=lambda t: t.path):
        _thing(stage, thing, spec.meters_per_unit)
    for mounted in sorted(spec.mounted, key=lambda m: m.path):
        _mounted(stage, mounted, spec.meters_per_unit)
    del stage, world  # the references are added to the layer alone: a stage would try to resolve them here
    for at, file in sorted(spec.references):
        layer.GetPrimAtPath(at).referenceList.Prepend(Sdf.Reference(f"./{file}"))
    return layer.ExportToString()


@dataclass(frozen=True)
class StageFinding:
    """One finding of usd-core's validators on a stage: ``error``, ``warning`` or ``info``, the validator, and its
    message."""

    severity: str
    validator: str
    message: str


def check_stage(path: Path) -> list[StageFinding]:
    """Every finding of every validator usd-core registers (its ``UsdValidation`` framework, which ``usdchecker``
    runs) on the stage at ``path``, its references resolved from beside it."""
    stage = Usd.Stage.Open(str(path))
    registry = UsdValidation.ValidationRegistry()
    context = UsdValidation.ValidationContext(registry.GetOrLoadAllValidators())
    found: list[StageFinding] = []
    for error in context.Validate(stage):
        kind = str(error.GetType()).rsplit(".", 1)[-1].lower()
        validator = error.GetValidator()
        name = validator.GetMetadata().name if validator else error.GetName()
        found.append(StageFinding(kind, str(name), str(error.GetMessage())))
    return found


__all__ = ["GeometryPrim", "MountedPrim", "StageFinding", "StageSpec", "ThingPrim", "check_stage", "stage_text",
           "up_axis"]
