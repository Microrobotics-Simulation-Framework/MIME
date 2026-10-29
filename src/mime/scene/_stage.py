"""The stage writer: the scene as one USD layer, ``scene/scene.usda`` in the saved form, written from the declarations
alone so that the same declarations give the same bytes.

What it writes:

- the layer's ``upAxis``, the declared up (+Y or +Z, the only ups a scene declares), and ``metersPerUnit``, a
  ``/World`` root with no transform of its own, and ``defaultPrim`` set to it;
- one prim per thing, ``mime:thing`` (its kind) and ``mime:name`` (its name in the document) on each. A thing written
  Python-first sits at ``/World/<name>``; a body bound to the prim an asset is referenced at takes its pose and those
  attributes on that prim, stronger than the asset's own;
- a geometry kind (a structure's, and each part's of a body written Python-first, below the body's prim) as a prim with
  ``MimeGeometryAPI``, ``mime:geometry`` (its name) and ``mime:geometry:version``, placed by its transform (its
  placement settings: no centre or axis is written as an attribute). A kind with a native form is that USD primitive,
  its native attributes the parameters in the stage's units; any other is an ``Xform`` with its shape settings in SI
  as ``mime:geometry:<setting>`` (each with its unit in ``customData``) and its drawing as a mesh on its ``proxy``
  child (purpose ``proxy``), derived: ``mime:derived_from`` and ``mime:derived_hash`` (``_derived.py``). A region with
  a sphere or cylinder for a primitive (the tube's lumen) has its own prim below its kind's, the primitive's USD
  form, derived the same way;
- a body's initial pose as its transform, and, for a body written Python-first with no parts, a sphere of its
  declared extent;
- an apparatus's chain: each segment's prim below the apparatus's, at its base's pose at the start, each frame a part
  is mounted on below its segment's, and each part below its frame, by its authored transform, a magnet part with
  ``MimeMagnetAPI`` and its direction;
- the references to asset layers, by their path below the stage's folder.

USD stores points as float32: a mesh's points stay in its own frame, near the origin, and the frame is placed by
double-precision ``translate`` and ``orient`` ops. Display colours and opacities are primvars, with no material.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import numpy as np


from mime.registry import Drawing, Native, RegionPrimitive
from mime.scene import _testing
from mime.scene._compose import MAGNET_API, MAGNET_DIRECTION
from mime.scene._derived import (
    DERIVED_FROM,
    DERIVED_HASH,
    GEOMETRY,
    GEOMETRY_API,
    GEOMETRY_VERSION,
    mesh_hash,
    primitive_hash,
    source,
    written,
)
from mime.scene._frame import UP_AXES
from mime.scene._kinds import IDENTITY, Quaternion, Vector, onto
from mime.scene._pxr import Gf, Sdf, Usd, UsdGeom, UsdValidation, Vt
from mime.things import WORLD
from mime.units import Magnitude

#: The writer's resolution: facets round a full turn of each curved direction of a drawn kind.
DRAWING_SEGMENTS = 64
#: A derived mesh's prim, below its kind's.
PROXY = "proxy"
_STRUCTURE_COLOUR = (0.85, 0.85, 0.9)
_STRUCTURE_OPACITY = 0.3
_BODY_COLOUR = (0.2, 0.6, 0.8)
_REGION_COLOUR = (0.55, 0.75, 1.0)
_REGION_OPACITY = 0.15


@dataclass(frozen=True)
class RegionPrim:
    """A region of a kind with a USD primitive of its own, below the kind's prim: its name and its OpenMaterica
    primitive, a sphere or a cylinder, in the kind's own frame."""

    name: str
    primitive: RegionPrimitive


@dataclass(frozen=True)
class KindPrim:
    """A geometry kind as the stage records it: its registered name and version, its shape settings in SI with each
    one's unit (its placement settings are the prim's transform: ``translation`` in metres and ``orientation``, a unit
    quaternion ``(w, x, y, z)``, in its parent's frame), its native form or its drawing at ``segments`` (neither: the
    prim alone), its regions' prims, and its display opacity (``None``: opaque)."""

    name: str
    version: str
    settings: Mapping[str, Magnitude]
    units: Mapping[str, str]
    translation: Vector = (0.0, 0.0, 0.0)
    orientation: Quaternion = IDENTITY
    native: Native | None = None
    drawing: Drawing | None = None
    segments: int = DRAWING_SEGMENTS
    regions: tuple[RegionPrim, ...] = ()
    opacity: float | None = None


@dataclass(frozen=True)
class ThingPrim:
    """A thing's prim: its path, its document name and kind, whether it binds a prim an asset brings in, its pose
    (a translation in metres and a unit quaternion ``(w, x, y, z)``), and what else the stage shows of it: a
    structure's geometry kind (its prim the thing's), a body's declared extent, and a body's parts by name, each a kind
    in the body's frame."""

    path: str
    name: str
    kind: str
    bound: bool = False
    translation: tuple[float, float, float] | None = None
    orientation: tuple[float, float, float, float] | None = None
    geometry: KindPrim | None = None
    extent: float | None = None
    attributes: Mapping[str, str] = field(default_factory=dict[str, str])
    parts: Mapping[str, KindPrim] = field(default_factory=dict[str, KindPrim])


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
        attribute = prim.CreateAttribute(f"{GEOMETRY}:{name}", Sdf.ValueTypeNames.Double3)
        attribute.Set(Gf.Vec3d(*value))
    else:
        attribute = prim.CreateAttribute(f"{GEOMETRY}:{name}", Sdf.ValueTypeNames.Double)
        attribute.Set(float(value))
    attribute.SetCustomDataByKey("units", unit)


def _extent(prim: Any) -> None:
    """The extent USD's own definition of the primitive gives, authored as the validators ask."""
    boundable = UsdGeom.Boundable(prim)
    boundable.CreateExtentAttr(UsdGeom.Boundable.ComputeExtentFromPlugins(boundable, Usd.TimeCode.Default()))


def _mesh(stage: Any, path: str, drawing: Drawing, scale: float, colour: tuple[float, float, float],
          opacity: float | None) -> Any:
    mesh = UsdGeom.Mesh.Define(stage, path)
    points = (np.asarray(drawing.points, dtype=np.float64).reshape(-1, 3) / scale).astype(np.float32)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(list(drawing.face_vertex_counts)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(list(drawing.face_vertex_indices)))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.stack([points.min(axis=0), points.max(axis=0)])))
    _display(mesh, colour, opacity)
    return mesh.GetPrim()


def _derived(prim: Any, held: KindPrim, digest: str, **detail: object) -> None:
    parameters = {name: written(value) for name, value in held.settings.items()
                  if not isinstance(value, tuple) or len(value) == 3}
    prim.CreateAttribute(DERIVED_FROM, Sdf.ValueTypeNames.String).Set(source(held.name, held.version, parameters,
                                                                             **detail))
    prim.CreateAttribute(DERIVED_HASH, Sdf.ValueTypeNames.String).Set(digest)


def _proxy(stage: Any, path: str, held: KindPrim, drawing: Drawing, scale: float, colour: tuple[float, float, float],
           opacity: float | None) -> None:
    """The kind's drawing on its ``proxy`` child, purpose ``proxy``, with what it was derived from and the hash of the
    mesh as USD holds it."""
    prim = _mesh(stage, f"{path}/{PROXY}", drawing, scale, colour, opacity)
    UsdGeom.Imageable(prim).CreatePurposeAttr(UsdGeom.Tokens.proxy)
    prim.CreateAttribute("mime:representation", Sdf.ValueTypeNames.String).Set(drawing.representation)
    points, counts, indices = (prim.GetAttribute(name).Get() for name in ("points", "faceVertexCounts",
                                                                          "faceVertexIndices"))
    _derived(prim, held, mesh_hash(points, counts, indices), segments=held.segments)


def _region(stage: Any, path: str, held: KindPrim, region: RegionPrim, scale: float) -> None:
    """A region's primitive as its own USD primitive below its kind's prim: a sphere, or a cylinder along its
    direction, in the kind's own frame; derived from the kind."""
    fields = region.primitive.fields
    if region.primitive.primitive == "cylinder":
        primitive = UsdGeom.Cylinder.Define(stage, f"{path}/{region.name}")
        primitive.CreateRadiusAttr(float(cast(float, fields["radius"])) / scale)
        primitive.CreateHeightAttr(float(cast(float, fields["length"])) / scale)
        primitive.CreateAxisAttr(UsdGeom.Tokens.z)
        at, turned = cast(Vector, fields["centre"]), onto(cast(Vector, fields["direction"]))
    elif region.primitive.primitive == "sphere":
        primitive = UsdGeom.Sphere.Define(stage, f"{path}/{region.name}")
        primitive.CreateRadiusAttr(float(cast(float, fields["radius"])) / scale)
        at, turned = cast(Vector, fields["centre"]), IDENTITY
    else:
        raise ValueError(f"region {region.name!r}: a {region.primitive.primitive} has no USD primitive of its own")
    prim = primitive.GetPrim()
    _place(prim, at, turned, scale)
    _extent(prim)
    _display(primitive, _REGION_COLOUR, _REGION_OPACITY)
    prim.CreateAttribute("mime:region", Sdf.ValueTypeNames.Token).Set(region.name)
    _derived(prim, held, primitive_hash(prim), region=region.name)


def _kind(stage: Any, path: str, held: KindPrim, scale: float, colour: tuple[float, float, float]) -> Any:
    """A geometry kind's prim at ``path``: its native primitive, or an ``Xform`` with its settings, its drawing on its
    ``proxy`` child and its regions' prims; ``MimeGeometryAPI`` and the kind's name and version on either; placed by
    its transform."""
    native = held.native
    if native is not None:
        prim = stage.DefinePrim(path, native.primitive)
        schema = getattr(UsdGeom, native.primitive)(prim)
        for attribute, setting in sorted(native.lengths.items()):
            length = float(cast(float, held.settings[setting])) / _testing.native_scale(scale)
            if _testing.writing_capsule_height_with_caps() and native.primitive == "Capsule" and attribute == "height":
                length += 2.0 * float(cast(float, held.settings["radius"])) / scale
            getattr(schema, f"Create{attribute.capitalize()}Attr")(length)
        for attribute, token in sorted(native.tokens.items()):
            getattr(schema, f"Create{attribute.capitalize()}Attr")(token)
        _extent(prim)
        _display(UsdGeom.Gprim(prim), colour, held.opacity)
    else:
        prim = UsdGeom.Xform.Define(stage, path).GetPrim()
        for name, value in sorted(held.settings.items()):
            _setting(prim, name, value, held.units.get(name, ""))
        if held.drawing is not None:
            _proxy(stage, path, held, held.drawing, scale, colour, held.opacity)
        for region in held.regions:
            _region(stage, path, held, region, scale)
    prim.AddAppliedSchema(GEOMETRY_API)
    prim.CreateAttribute(GEOMETRY, Sdf.ValueTypeNames.Token).Set(held.name)
    prim.CreateAttribute(GEOMETRY_VERSION, Sdf.ValueTypeNames.String).Set(held.version)
    _place(prim, held.translation, held.orientation, scale)
    return prim


def _thing(stage: Any, thing: ThingPrim, scale: float) -> None:
    if thing.bound:
        prim = stage.GetPrimAtPath(thing.path)
        _place(prim, thing.translation, thing.orientation, scale)
    elif thing.geometry is not None:
        held = thing.geometry
        prim = _kind(stage, thing.path, replace(held, opacity=held.opacity if held.native is not None else
                                                _STRUCTURE_OPACITY), scale, _STRUCTURE_COLOUR)
    elif thing.translation is not None or thing.orientation is not None:
        prim = UsdGeom.Xform.Define(stage, thing.path).GetPrim()
        _place(prim, thing.translation, thing.orientation, scale)
    else:
        prim = UsdGeom.Scope.Define(stage, thing.path).GetPrim()
    prim.CreateAttribute("mime:thing", Sdf.ValueTypeNames.Token).Set(thing.kind)
    prim.CreateAttribute("mime:name", Sdf.ValueTypeNames.String).Set(thing.name)
    for name, value in sorted(thing.attributes.items()):
        prim.CreateAttribute(f"mime:{name}", Sdf.ValueTypeNames.String).Set(value)
    if thing.bound:
        return
    for name, part in sorted(thing.parts.items()):
        _kind(stage, f"{thing.path}/{name}", part, scale, _BODY_COLOUR)
    if thing.extent is not None and not thing.parts:
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


__all__ = ["DRAWING_SEGMENTS", "PROXY", "KindPrim", "MountedPrim", "RegionPrim", "StageFinding", "StageSpec",
           "ThingPrim", "check_stage", "stage_text", "up_axis"]
