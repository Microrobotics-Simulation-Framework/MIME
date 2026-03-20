"""GeometrySource protocol — consumed by spatial nodes.

Defines the contract that fluid environment nodes (CSFFlowNode,
ConcentrationDiffusionNode, TissueDeformationNode) depend on for
spatial domain definition.

Supports parametric geometries (cylinder, sphere — needed for B0, B4-T1)
and versioned mesh references (needed for B4-T2 and beyond, fulfilled
by Neurobotika).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable
import json


@runtime_checkable
class GeometrySource(Protocol):
    """Abstract contract for spatial domain geometry.

    Must be usable without MICROBOTICA (enables standalone MIME
    testing and CI benchmarks). Parametric subtypes serialise to
    JSON for standalone use and to USD as typed prims.
    """

    @property
    def domain_bounds(self) -> tuple[tuple[float, float, float],
                                     tuple[float, float, float]]:
        """Axis-aligned bounding box: ((x_min, y_min, z_min), (x_max, y_max, z_max))."""
        ...

    @property
    def geometry_type(self) -> str:
        """Short identifier: 'cylinder', 'sphere', 'mesh', etc."""
        ...

    @property
    def version(self) -> str:
        """Version string for reproducibility tracking."""
        ...

    def to_json(self) -> str:
        """Serialise to JSON for standalone MIME testing."""
        ...


@dataclass(frozen=True)
class CylinderGeometry:
    """Parametric cylindrical channel geometry.

    Used for B0 (experimental validation), B1/B2 (physics benchmarks),
    and B4-T1 (simple navigation geometry).
    """
    diameter_m: float
    length_m: float
    center_x: float = 0.0
    center_y: float = 0.0
    center_z: float = 0.0
    axis: str = "z"  # Primary axis: 'x', 'y', or 'z'

    @property
    def domain_bounds(self) -> tuple[tuple[float, float, float],
                                     tuple[float, float, float]]:
        r = self.diameter_m / 2.0
        half_l = self.length_m / 2.0
        cx, cy, cz = self.center_x, self.center_y, self.center_z
        if self.axis == "z":
            return ((cx - r, cy - r, cz - half_l),
                    (cx + r, cy + r, cz + half_l))
        elif self.axis == "x":
            return ((cx - half_l, cy - r, cz - r),
                    (cx + half_l, cy + r, cz + r))
        else:  # y
            return ((cx - r, cy - half_l, cz - r),
                    (cx + r, cy + half_l, cz + r))

    @property
    def geometry_type(self) -> str:
        return "cylinder"

    @property
    def version(self) -> str:
        return "parametric-1.0"

    def to_json(self) -> str:
        return json.dumps({
            "type": "cylinder",
            "diameter_m": self.diameter_m,
            "length_m": self.length_m,
            "center": [self.center_x, self.center_y, self.center_z],
            "axis": self.axis,
            "version": self.version,
        })


@dataclass(frozen=True)
class MeshGeometry:
    """Reference to an external USD mesh file.

    Used for B4-T2 (Neurobotika ventricular mesh) and B4-T3
    (pathological anatomy variants).
    """
    mesh_uri: str
    mesh_version: str = ""
    segmentation_parameters: str = ""
    source_mri_metadata: str = ""
    # Bounds must be provided since we can't read the mesh without USD
    bounds_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounds_max: tuple[float, float, float] = (1.0, 1.0, 1.0)

    @property
    def domain_bounds(self) -> tuple[tuple[float, float, float],
                                     tuple[float, float, float]]:
        return (self.bounds_min, self.bounds_max)

    @property
    def geometry_type(self) -> str:
        return "mesh"

    @property
    def version(self) -> str:
        return self.mesh_version

    def to_json(self) -> str:
        return json.dumps({
            "type": "mesh",
            "mesh_uri": self.mesh_uri,
            "mesh_version": self.mesh_version,
            "segmentation_parameters": self.segmentation_parameters,
            "source_mri_metadata": self.source_mri_metadata,
            "bounds_min": list(self.bounds_min),
            "bounds_max": list(self.bounds_max),
        })
