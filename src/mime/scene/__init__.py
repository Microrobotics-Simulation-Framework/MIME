"""The USD stage: the scene declared Python-first (up, units, referenced assets, prim paths), the world frame, analytic
geometry kinds, the ``mime:`` schema for design data on part prims, and the stage writer; with viz, the only importer
of pxr.

A geometry kind is a contract a lab implements: a registered ``mime.registry.Geometry`` naming its regions, surfaces
and directions, whose surfaces are evaluated by :class:`Shape` (the protocol: signed distance, outward normal, and the
axis where one exists), registered in ``mime.core``, and whose ``drawing`` is what the stage shows of it."""

from mime.core import Cylinder, Shape, register_shape
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
from mime.scene._solids import SolidBox, SolidCylinder, SolidEllipsoid
from mime.scene._scene import Asset, Reference, Scene, asset, sha256_hex
from mime.scene._stage import (
    GeometryPrim,
    MountedPrim,
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
    "LAYERS", "MAGNET_API", "MAGNET_DIRECTION", "METERS_PER_UNIT", "STAGE_FILE", "UP_AXES", "Asset", "Axis",
    "ComposedLayer", "Composition", "Cylinder", "GeometryPrim", "MagnetDesign", "MountedPrim", "Reference", "Scene",
    "Shape", "SolidBox", "SolidCylinder", "SolidEllipsoid", "StageFinding", "StageSpec", "ThingPrim", "Tube",
    "WorldFrame", "X", "Y", "Z", "asset", "check_stage", "compose", "meters_per_unit", "register_shape", "scene_up",
    "sha256_hex", "stage_text", "unit_vector", "up_axis", "up_axis_problem", "valid_part_path", "valid_prim_path",
    "within", "world_frame",
]
