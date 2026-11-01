"""The USD stage: the scene declared Python-first (up, units, referenced assets, prim paths), the world frame, analytic
geometry kinds, the ``mime:`` schema for design data on part prims, and the stage writer; with viz, the only importer
of pxr.

A geometry kind is a contract a lab implements: a registered ``mime.registry.Geometry`` naming its regions, surfaces
and directions, whose surfaces are evaluated by :class:`Shape` (the protocol: signed distance, outward normal, and the
axis where one exists), registered in ``mime.core``, and whose ``drawing`` is what the stage shows of it."""

from mime.core import Shape, register_shape
from mime.scene._compose import (
    LAYERS,
    MAGNET_API,
    MAGNET_DIRECTION,
    ComposedLayer,
    Composition,
    MagnetDesign,
    compose,
    valid_part_path,
    valid_prim_path,
    within,
)
from mime.scene._derived import (
    DERIVED_FROM,
    DERIVED_HASH,
    GEOMETRY_API,
    DerivedFinding,
    StageEdits,
    derived_findings,
    stage_derived_findings,
    stage_edits,
)
from mime.scene._frame import (
    METERS_PER_UNIT,
    UP_AXES,
    WorldFrame,
    meters_per_unit,
    scene_up,
    unit_vector,
    up_axis_problem,
    world_frame,
)
from mime.scene._geometry import Axis, Tube, X, Y, Z
from mime.scene._previous import KEPT
from mime.scene._sampling import (
    QUADRATURE,
    QUASI_UNIFORM,
    body_reach,
    length_of,
    quadrature,
    quasi_uniform,
    sampled_reach,
)
from mime.scene._screw import Screw
from mime.scene._kinds import kind_frame, placed, shape_settings
from mime.scene._read import KindOnStage, geometry_schema_problems, read_kind
from mime.scene._solids import Box, Capsule, Cylinder, Ellipsoid, Sphere
from mime.scene._scene import Asset, Reference, Scene, asset, sha256_hex
from mime.scene._stage import (
    DRAWING_SEGMENTS,
    PROXY,
    KindPrim,
    MountedPrim,
    RegionPrim,
    StageFinding,
    StageSpec,
    ThingPrim,
    check_stage,
    stage_text,
    up_axis,
)

#: The stage's file below a saved experiment's folder.
STAGE_FILE = "scene/scene.usda"

__all__ = [
    "Asset", "Axis", "Box", "Capsule", "ComposedLayer", "Composition", "Cylinder", "DERIVED_FROM", "DERIVED_HASH",
    "DRAWING_SEGMENTS", "DerivedFinding", "Ellipsoid", "GEOMETRY_API", "KEPT", "KindOnStage", "KindPrim", "LAYERS",
    "MAGNET_API", "MAGNET_DIRECTION", "METERS_PER_UNIT", "MagnetDesign", "MountedPrim", "PROXY", "QUADRATURE",
    "QUASI_UNIFORM", "Reference", "RegionPrim", "STAGE_FILE", "Scene", "Screw", "Shape", "Sphere", "StageEdits",
    "StageFinding", "StageSpec", "ThingPrim", "Tube", "UP_AXES", "WorldFrame", "X", "Y", "Z", "asset", "body_reach",
    "check_stage",
    "compose", "derived_findings", "geometry_schema_problems", "kind_frame", "length_of", "meters_per_unit", "placed",
    "quadrature", "quasi_uniform", "read_kind", "register_shape", "sampled_reach", "scene_up", "sha256_hex",
    "shape_settings", "stage_derived_findings", "stage_edits", "stage_text", "unit_vector", "up_axis",
    "up_axis_problem", "valid_part_path", "valid_prim_path", "within", "world_frame",
]
