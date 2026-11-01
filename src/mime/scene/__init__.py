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
from mime.scene._screw import Screw
from mime.scene._kinds import kind_frame, placed, shape_settings
from mime.scene._read import KindOnStage, geometry_schema_problems, read_kind
from mime.scene._solids import Box, Capsule, Cylinder, Ellipsoid, Sphere
from mime.scene._scene import Asset, Reference, Scene, asset, sha256_hex
from mime.scene._stage import (
    ARROW,
    DRAWING_SEGMENTS,
    STAGE_WRITER,
    STAGE_WRITERS,
    PROXY,
    VECTOR_SCALE,
    VECTOR_UNITS,
    VECTOR_VALUE,
    VISUAL_ASSET,
    VISUAL_FILE,
    VISUAL_ROLE,
    VISUAL_SHA256,
    KindPrim,
    MountedPrim,
    RegionPrim,
    StageFinding,
    StageSpec,
    ThingPrim,
    VectorPrim,
    VisualPrim,
    check_stage,
    stage_text,
    up_axis,
)
from mime.scene._visual_sets import VisualFile, VisualSet, mesh_path, mesh_source, visual_sets

#: The stage's file below a saved experiment's folder.
STAGE_FILE = "scene/scene.usda"

__all__ = [
    "ARROW", "DERIVED_FROM", "DERIVED_HASH", "DRAWING_SEGMENTS", "GEOMETRY_API", "LAYERS", "MAGNET_API", "MAGNET_DIRECTION",
    "METERS_PER_UNIT", "PROXY", "STAGE_FILE", "STAGE_WRITER", "STAGE_WRITERS", "UP_AXES", "VECTOR_SCALE", "VECTOR_UNITS", "VECTOR_VALUE", "VISUAL_ASSET", "VISUAL_FILE", "VISUAL_ROLE", "VISUAL_SHA256",
    "Asset", "Axis", "Box", "Capsule", "ComposedLayer",
    "Composition", "Cylinder", "DerivedFinding", "Ellipsoid", "KindOnStage", "KindPrim", "MagnetDesign", "MountedPrim",
    "Reference", "RegionPrim", "Scene", "Screw", "Shape", "Sphere", "StageEdits", "StageFinding", "StageSpec",
    "ThingPrim", "Tube", "VectorPrim", "VisualFile", "VisualPrim", "VisualSet", "WorldFrame", "X", "Y", "Z", "asset", "check_stage",
    "compose", "derived_findings", "mesh_path", "mesh_source", "visual_sets",
    "geometry_schema_problems", "kind_frame", "meters_per_unit", "placed", "read_kind", "register_shape", "scene_up",
    "sha256_hex", "shape_settings", "stage_derived_findings", "stage_edits", "stage_text", "unit_vector", "up_axis",
    "up_axis_problem", "valid_part_path", "valid_prim_path", "within", "world_frame",
]
