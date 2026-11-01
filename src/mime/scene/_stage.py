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
- an apparatus's chain: each segment's prim below the apparatus's, at its base's pose at the start; every frame of the
  segment below it, placed in the world (its ``xformOpOrder`` resets the transform stack, so its transform is its
  world pose: a state's frame pose, as it stands), at its pose at the start; each part below the frame it is mounted
  on, by its authored transform, a magnet part with ``MimeMagnetAPI`` and its direction; and below each frame what the
  stage draws of it: its description's visuals (``visuals/``: a box, cylinder or sphere as its USD primitive; a mesh
  as an ``Xform`` naming its visual set and file sha256, which the asset cache's overlay fills, ``mime.viz``), and the
  simple shapes that stand in for meshes the stage does not hold (``fallback/``: bones from the frame's origin to each
  child's, a link's collision primitives);
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
    #: Placed in the world, whatever its parent's transform: its transform stack is reset.
    world: bool = False


@dataclass(frozen=True)
class VisualPrim:
    """A shape drawn on a chain's frame, at ``path`` below the frame's prim, placed in the frame (``translation`` in
    metres, ``orientation`` a unit quaternion ``(w, x, y, z)``): ``kind`` ``box`` (``size``), ``cylinder``
    (``radius``, ``length``, along its z), ``sphere`` (``radius``), ``bone`` (a cylinder of ``radius`` from the
    frame's origin to ``end``), or ``mesh``: an ``Xform`` scaled by ``scale``, holding its description's file name
    (``file``) and, where a visual set lists it, the set (``asset``) and the file's sha256, never its bytes. ``role``
    is ``visual``, ``collision`` or ``frames``."""

    path: str
    kind: str
    role: str
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    size: tuple[float, float, float] | None = None
    radius: float | None = None
    length: float | None = None
    end: tuple[float, float, float] | None = None
    colour: tuple[float, float, float] | None = None
    file: str | None = None
    asset: str | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class VectorPrim:
    """A field a law reads at a body, as the stage draws it: an arrow below ``path``, placed in the world at the body's
    position (``translation``, m) and hidden until a recording or a client gives it a value (``mime:vector:value``, in
    ``unit``); its arrow is ``metres_per_unit`` long per unit of the field, along the vector."""

    path: str
    translation: tuple[float, float, float]
    unit: str
    name: str
    metres_per_unit: float


#: The arrow's radius, as a fraction of its length: a presentation choice.
ARROW_RADIUS_FRACTION = 0.05


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
    visuals: tuple[VisualPrim, ...] = ()
    vectors: tuple[VectorPrim, ...] = ()


def up_axis(up: tuple[float, float, float]) -> str:
    """The token the stage declares the unit vector ``up`` with, Y or Z; ``KeyError`` for any other, which the scene
    refuses at its declaration."""
    return UP_AXES[up]


def _quat(q: tuple[float, float, float, float]) -> Any:
    return Gf.Quatd(q[0], Gf.Vec3d(q[1], q[2], q[3]))


def _place(prim: Any, translation: tuple[float, float, float] | None,
           orientation: tuple[float, float, float, float] | None, scale: float, *, world: bool = False) -> None:
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    if world:
        xformable.SetResetXformStack(True)
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
    _place(prim, mounted.translation, mounted.orientation, scale, world=mounted.world)
    for name, value in sorted(mounted.attributes.items()):
        prim.CreateAttribute(f"mime:{name}", Sdf.ValueTypeNames.String).Set(value)
    if mounted.magnet is not None:
        prim.AddAppliedSchema(MAGNET_API)
        prim.CreateAttribute(MAGNET_DIRECTION, Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(*mounted.magnet))


_VISUAL_COLOUR = (0.7, 0.7, 0.72)
#: The attributes a mesh visual's prim names its source with, which the asset cache's overlay reads.
VISUAL_FILE, VISUAL_ASSET, VISUAL_SHA256, VISUAL_ROLE = ("mime:visual:file", "mime:visual:asset", "mime:visual:sha256",
                                                         "mime:visual:role")
_Z = (0.0, 0.0, 1.0)


def _towards(end: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """The unit quaternion turning +Z onto ``end``'s direction, the shorter way."""
    length = float(np.linalg.norm(end))
    d = np.asarray(end, dtype=np.float64) / length
    w = 1.0 + float(d[2])
    if w < 1e-12:
        return (0.0, 1.0, 0.0, 0.0)
    q = np.array([w, -float(d[1]), float(d[0]), 0.0])
    q /= np.linalg.norm(q)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _visual(stage: Any, held: VisualPrim, scale: float) -> None:
    """A shape on a frame: a USD primitive, or a mesh's ``Xform`` naming its source."""
    translation, orientation = held.translation, held.orientation
    if held.kind == "bone" and held.end is not None:
        translation = (translation[0] + held.end[0] / 2, translation[1] + held.end[1] / 2,
                       translation[2] + held.end[2] / 2)
        orientation = _towards(held.end)
    colour = held.colour or _VISUAL_COLOUR
    if held.kind == "mesh":
        prim = UsdGeom.Xform.Define(stage, held.path).GetPrim()
        _place(prim, translation, orientation, scale)
        if held.scale != (1.0, 1.0, 1.0):
            UsdGeom.Xformable(prim).AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*held.scale))
        prim.CreateAttribute(VISUAL_FILE, Sdf.ValueTypeNames.String).Set(held.file or "")
        if held.asset is not None and held.sha256 is not None:
            prim.CreateAttribute(VISUAL_ASSET, Sdf.ValueTypeNames.String).Set(held.asset)
            prim.CreateAttribute(VISUAL_SHA256, Sdf.ValueTypeNames.String).Set(held.sha256)
        prim.CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray).Set(
            Vt.Vec3fArray([Gf.Vec3f(*colour)]))
    elif held.kind == "box" and held.size is not None:
        cube = UsdGeom.Cube.Define(stage, held.path)
        cube.CreateSizeAttr(1.0)
        prim = cube.GetPrim()
        _place(prim, translation, orientation, scale)
        UsdGeom.Xformable(prim).AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(*(c / scale for c in held.size)))
        _extent(prim)
        _display(cube, colour)
    elif held.kind in ("cylinder", "bone") and held.radius is not None:
        length = held.length if held.kind == "cylinder" else float(np.linalg.norm(held.end or _Z))
        cylinder = UsdGeom.Cylinder.Define(stage, held.path)
        cylinder.CreateRadiusAttr(held.radius / scale)
        cylinder.CreateHeightAttr((length or 0.0) / scale)
        cylinder.CreateAxisAttr(UsdGeom.Tokens.z)
        prim = cylinder.GetPrim()
        _place(prim, translation, orientation, scale)
        _extent(prim)
        _display(cylinder, colour)
    elif held.kind == "sphere" and held.radius is not None:
        sphere = UsdGeom.Sphere.Define(stage, held.path)
        sphere.CreateRadiusAttr(held.radius / scale)
        prim = sphere.GetPrim()
        _place(prim, translation, orientation, scale)
        _extent(prim)
        _display(sphere, colour)
    else:
        return
    prim.CreateAttribute(VISUAL_ROLE, Sdf.ValueTypeNames.Token).Set(held.role)


#: The attributes a vector's prim carries: its value (time-sampled in a recording), its units and its arrow's scale.
VECTOR_VALUE, VECTOR_UNITS, VECTOR_SCALE = "mime:vector:value", "mime:vector:units", "mime:vector:metres_per_unit"
ARROW = "arrow"


def _vector(stage: Any, held: VectorPrim, scale: float) -> None:
    """A vector's prim: an ``Xform`` placed in the world at the body, hidden, holding an arrow a metre long along its
    +Z (a cylinder from its origin) that a recording orients along the vector and scales, whole, to its length."""
    prim = UsdGeom.Xform.Define(stage, held.path).GetPrim()
    _place(prim, held.translation, (1.0, 0.0, 0.0, 0.0), scale, world=True)
    UsdGeom.Xformable(prim).AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1.0, 1.0, 1.0))
    UsdGeom.Imageable(prim).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    prim.CreateAttribute(VECTOR_VALUE, Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(0.0, 0.0, 0.0))
    prim.CreateAttribute(VECTOR_UNITS, Sdf.ValueTypeNames.String).Set(held.unit)
    prim.CreateAttribute(VECTOR_SCALE, Sdf.ValueTypeNames.Double).Set(held.metres_per_unit)
    prim.CreateAttribute("mime:vector:name", Sdf.ValueTypeNames.String).Set(held.name)
    arrow = UsdGeom.Cylinder.Define(stage, f"{held.path}/{ARROW}")
    arrow.CreateRadiusAttr(ARROW_RADIUS_FRACTION / scale)
    arrow.CreateHeightAttr(1.0 / scale)
    arrow.CreateAxisAttr(UsdGeom.Tokens.z)
    _place(arrow.GetPrim(), (0.0, 0.0, 0.5), None, scale)
    _extent(arrow.GetPrim())
    _display(arrow, (0.95, 0.55, 0.1))


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
    for visual in sorted(spec.visuals, key=lambda v: v.path):
        _visual(stage, visual, spec.meters_per_unit)
    for vector in sorted(spec.vectors, key=lambda v: v.path):
        _vector(stage, vector, spec.meters_per_unit)
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


__all__ = ["ARROW", "DRAWING_SEGMENTS", "PROXY", "VECTOR_SCALE", "VECTOR_UNITS", "VECTOR_VALUE", "VISUAL_ASSET", "VISUAL_FILE", "VISUAL_ROLE", "VISUAL_SHA256", "KindPrim",
           "MountedPrim", "RegionPrim", "StageFinding", "StageSpec", "ThingPrim", "VectorPrim", "VisualPrim", "check_stage",
           "stage_text", "up_axis"]
