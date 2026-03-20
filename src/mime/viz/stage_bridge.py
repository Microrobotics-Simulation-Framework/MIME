"""StageBridge — writes simulation state to a live USD stage each timestep.

The bridge is a passive observer (PolicyRunner StepObserver callback).
It converts JAX state dicts to USD prim attributes. The viewport reads
the stage and produces pixels. The simulation never imports this module.

Dynamic vs. static geometry:
- Dynamic state (robot position, field vector) is updated every timestep
  via update() inside a Sdf.ChangeBlock for batched notifications.
- Static geometry (parametric tube, Neurobotika mesh) is loaded once at
  setup time and never touched by update().

Usage:
    bridge = StageBridge()
    bridge.register_robot("body", CylinderGeometry(2e-3, 10e-3))
    bridge.add_parametric_geometry(CylinderGeometry(2e-3, 10e-3))
    runner = PolicyRunner(gm, ci, observers=[bridge.as_observer()])
"""

from __future__ import annotations

import math
import logging
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# USD imports are optional — the viz layer should not break the core package
try:
    from pxr import Usd, UsdGeom, Gf, Sdf
    _HAS_USD = True
except ImportError:
    _HAS_USD = False

from mime.core.geometry import GeometrySource, CylinderGeometry, MeshGeometry


def _require_usd() -> None:
    if not _HAS_USD:
        raise ImportError(
            "StageBridge requires OpenUSD Python bindings (pxr). "
            "Install with: pip install usd-core"
        )


def _quat_to_gf(q: np.ndarray) -> "Gf.Quatf":
    """Convert [w, x, y, z] numpy array to Gf.Quatf."""
    return Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _vec3_to_gf(v: np.ndarray) -> "Gf.Vec3d":
    """Convert (3,) numpy array to Gf.Vec3d."""
    return Gf.Vec3d(float(v[0]), float(v[1]), float(v[2]))


class _RegisteredPrim:
    """Internal bookkeeping for a dynamic prim."""

    def __init__(
        self,
        node_name: str,
        prim_path: str,
        prim_type: str,
        state_fields: dict[str, str],
    ):
        self.node_name = node_name
        self.prim_path = prim_path
        self.prim_type = prim_type
        # Maps state field name -> what to do with it
        # e.g. {"position": "translate", "orientation": "orient"}
        self.state_fields = state_fields


class StageBridge:
    """Writes simulation state to a live USD stage each timestep.

    Parameters
    ----------
    stage : Usd.Stage, optional
        If provided, writes to this stage. If None, creates an
        in-memory stage.
    up_axis : str
        Stage up axis. Default "Z".
    meters_per_unit : float
        Stage meters per unit. Default 1.0 (SI metres).
    """

    # TODO(double-buffer): Phase 2 — add double_buffered=True mode.
    # Design: two in-memory stages, atomic pointer swap after each step.
    # See RENDERING_PLAN.md "Threading and Double-Buffering" section.
    # The synchronous implementation below is forward-compatible — the
    # swap point is inside as_observer(), after update() completes.

    def __init__(
        self,
        stage: Optional[Any] = None,
        up_axis: str = "Z",
        meters_per_unit: float = 1.0,
    ):
        _require_usd()

        if stage is None:
            self._stage = Usd.Stage.CreateInMemory()
        else:
            self._stage = stage

        # Set stage metadata
        UsdGeom.SetStageUpAxis(self._stage, UsdGeom.Tokens.z if up_axis == "Z" else UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(self._stage, meters_per_unit)

        # Create /World scope
        self._world = UsdGeom.Xform.Define(self._stage, "/World")

        # Registered dynamic prims (updated each timestep)
        self._dynamic_prims: list[_RegisteredPrim] = []

        # Camera
        self._camera_path = "/World/Camera"
        self._setup_default_camera()

    @property
    def stage(self) -> Any:
        """The live USD stage."""
        return self._stage

    def _setup_default_camera(self) -> None:
        """Create a default camera looking at the origin."""
        cam = UsdGeom.Camera.Define(self._stage, self._camera_path)
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.0001, 100.0))
        cam.GetFocalLengthAttr().Set(50.0)

        # Position camera to see the workspace
        xform = UsdGeom.Xformable(cam.GetPrim())
        xform.ClearXformOpOrder()
        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(0.005, -0.01, 0.005))

    def register_robot(
        self,
        node_name: str,
        geometry: Optional[GeometrySource] = None,
        prim_path: str = "/World/Robot",
        radius: float = 100e-6,
        half_length: float = 150e-6,
    ) -> None:
        """Register a robot body for dynamic visualisation.

        Creates a UsdGeom prim at prim_path and tracks it for
        per-timestep transform updates from the named node's state.

        Parameters
        ----------
        node_name : str
            Name of the RigidBodyNode whose state drives this prim.
        geometry : GeometrySource, optional
            If provided, determines the prim shape. If None, creates
            a sphere with the given radius.
        prim_path : str
            USD prim path.
        radius : float
            Default sphere radius [m] if no geometry provided.
        half_length : float
            Half-length for capsule/ellipsoid approximation [m].
        """
        if geometry is not None and geometry.geometry_type == "cylinder":
            cyl = UsdGeom.Capsule.Define(self._stage, prim_path)
            cyl.GetRadiusAttr().Set(float(geometry.diameter_m / 2.0))
            cyl.GetHeightAttr().Set(float(geometry.length_m))
            cyl.GetAxisAttr().Set("Z")
        else:
            sphere = UsdGeom.Sphere.Define(self._stage, prim_path)
            sphere.GetRadiusAttr().Set(float(radius))

        # Set up xform ops for dynamic updates
        prim = self._stage.GetPrimAtPath(prim_path)
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp()
        xformable.AddOrientOp()

        self._dynamic_prims.append(_RegisteredPrim(
            node_name=node_name,
            prim_path=prim_path,
            prim_type="robot",
            state_fields={"position": "translate", "orientation": "orient"},
        ))

    def register_field(
        self,
        node_name: str,
        prim_path: str = "/World/FieldArrow",
        arrow_length: float = 2e-3,
    ) -> None:
        """Register a magnetic field for dynamic arrow visualisation.

        Creates a cone prim (arrow head) that is oriented and scaled
        each timestep to show the field direction and relative magnitude.

        Parameters
        ----------
        node_name : str
            Name of the ExternalMagneticFieldNode.
        prim_path : str
            USD prim path for the arrow.
        arrow_length : float
            Base arrow length [m] at unit field strength.
        """
        # Single cone as arrow head — simple but effective
        cone = UsdGeom.Cone.Define(self._stage, prim_path)
        cone.GetRadiusAttr().Set(float(arrow_length * 0.15))
        cone.GetHeightAttr().Set(float(arrow_length))
        cone.GetAxisAttr().Set("Z")

        prim = self._stage.GetPrimAtPath(prim_path)
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp()
        xformable.AddOrientOp()
        xformable.AddScaleOp()

        self._dynamic_prims.append(_RegisteredPrim(
            node_name=node_name,
            prim_path=prim_path,
            prim_type="field",
            state_fields={"field_vector": "orient_scale"},
        ))
        self._arrow_length = arrow_length

    def add_parametric_geometry(
        self,
        geometry: GeometrySource,
        prim_path: str = "/World/Channel",
    ) -> None:
        """Add static parametric geometry as inline USD prims.

        Creates UsdGeom.Cylinder / UsdGeom.Sphere directly in the stage.
        This geometry is loaded once and never modified by update().
        """
        if isinstance(geometry, CylinderGeometry):
            cyl = UsdGeom.Cylinder.Define(self._stage, prim_path)
            cyl.GetRadiusAttr().Set(float(geometry.diameter_m / 2.0))
            cyl.GetHeightAttr().Set(float(geometry.length_m))
            cyl.GetAxisAttr().Set(geometry.axis.upper())

            # Set display as wireframe for the channel (so robot is visible inside)
            cyl.GetPrim().CreateAttribute(
                "primvars:displayOpacity", Sdf.ValueTypeNames.Float
            ).Set(0.2)

            # Position the cylinder
            xformable = UsdGeom.Xformable(cyl.GetPrim())
            xformable.ClearXformOpOrder()
            translate_op = xformable.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(
                geometry.center_x, geometry.center_y, geometry.center_z
            ))
        else:
            logger.warning(
                "add_parametric_geometry: unsupported geometry type %s, skipping",
                geometry.geometry_type,
            )

    def add_reference_geometry(
        self,
        usd_path: str,
        prim_path: str = "/World/Anatomy",
        as_reference: bool = False,
    ) -> None:
        """Load a USD file as a payload (default) or hard reference.

        Default: AddPayload — deferred loading, can be unloaded at runtime.
        as_reference=True: AddReference — immediate loading.
        """
        prim = self._stage.DefinePrim(prim_path)
        if as_reference:
            prim.GetReferences().AddReference(usd_path)
        else:
            prim.GetPayloads().AddPayload(usd_path)

    def update(self, state: dict[str, dict[str, Any]]) -> None:
        """Write current dynamic state to USD stage.

        Called each timestep. Uses Sdf.ChangeBlock to batch all
        attribute writes into a single stage notification.

        Parameters
        ----------
        state : dict
            Full graph state: {node_name: {field_name: value, ...}, ...}.
            Values are JAX arrays or numpy arrays.
        """
        with Sdf.ChangeBlock():
            for reg in self._dynamic_prims:
                node_state = state.get(reg.node_name)
                if node_state is None:
                    continue

                prim = self._stage.GetPrimAtPath(reg.prim_path)
                if not prim.IsValid():
                    continue

                xformable = UsdGeom.Xformable(prim)
                ops = xformable.GetOrderedXformOps()

                if reg.prim_type == "robot":
                    self._update_robot(ops, node_state)
                elif reg.prim_type == "field":
                    self._update_field(ops, node_state)

    def _update_robot(self, ops: list, node_state: dict) -> None:
        """Update robot prim transform from state."""
        pos = node_state.get("position")
        orient = node_state.get("orientation")

        if pos is not None:
            pos_np = np.asarray(pos)
            # ops[0] is translate
            ops[0].Set(Gf.Vec3d(float(pos_np[0]), float(pos_np[1]), float(pos_np[2])))

        if orient is not None:
            q_np = np.asarray(orient)
            # ops[1] is orient
            ops[1].Set(_quat_to_gf(q_np))

    def _update_field(self, ops: list, node_state: dict) -> None:
        """Update field arrow prim from field_vector state."""
        fv = node_state.get("field_vector")
        if fv is None:
            return

        fv_np = np.asarray(fv)
        magnitude = float(np.linalg.norm(fv_np))

        if magnitude < 1e-30:
            # Zero field — hide by scaling to zero
            ops[2].Set(Gf.Vec3d(0, 0, 0))  # scale
            return

        # Direction as unit vector
        direction = fv_np / magnitude

        # Convert direction to a quaternion that rotates Z-axis to direction
        # Using the half-angle formula: q = [cos(a/2), sin(a/2) * axis]
        z_axis = np.array([0.0, 0.0, 1.0])
        dot = float(np.dot(z_axis, direction))
        dot = max(-1.0, min(1.0, dot))

        if dot > 0.9999:
            q = np.array([1.0, 0.0, 0.0, 0.0])
        elif dot < -0.9999:
            q = np.array([0.0, 1.0, 0.0, 0.0])
        else:
            axis = np.cross(z_axis, direction)
            axis = axis / np.linalg.norm(axis)
            angle = math.acos(dot)
            half = angle / 2.0
            q = np.array([
                math.cos(half),
                axis[0] * math.sin(half),
                axis[1] * math.sin(half),
                axis[2] * math.sin(half),
            ])

        # ops[0] = translate (at origin), ops[1] = orient, ops[2] = scale
        ops[1].Set(_quat_to_gf(q))

        # Scale proportional to field magnitude (normalised to mT range)
        scale_factor = magnitude * 1e3  # T -> mT, so 10mT = scale 10
        scale_factor = max(scale_factor, 0.1)  # minimum visible size
        ops[2].Set(Gf.Vec3d(scale_factor, scale_factor, scale_factor))

    def as_observer(self) -> Callable:
        """Return a PolicyRunner StepObserver callback.

        The callback writes true_state to the USD stage after each step.
        """
        def observer(t, dt, true_state, observed_state, ext_inputs, applied_inputs):
            self.update(true_state)

        return observer

    def export(self, path: str) -> None:
        """Export the current stage to a USD file on disk."""
        self._stage.GetRootLayer().Export(path)
