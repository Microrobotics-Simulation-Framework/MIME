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
    try:
        from pxr import UsdShade, UsdLux
        _HAS_USD_SHADE = True
    except ImportError:
        _HAS_USD_SHADE = False
    _HAS_USD = True
except ImportError:
    _HAS_USD = False
    _HAS_USD_SHADE = False

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
        mesh_data: Optional[tuple] = None,
        display_color: tuple = (0.2, 0.6, 0.8),
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
        mesh_data : (vertices, triangles), optional
            If provided, creates a UsdGeom.Mesh from (N,3) float32
            vertices and (M,3) int32 triangle indices. Overrides
            geometry parameter.
        display_color : (r, g, b)
            Display color for the prim.
        """
        color = Gf.Vec3f(*display_color)

        if mesh_data is not None:
            vertices, triangles = mesh_data
            mesh = UsdGeom.Mesh.Define(self._stage, prim_path)
            points = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2]))
                      for v in vertices]
            mesh.GetPointsAttr().Set(points)
            face_counts = [3] * len(triangles)
            face_indices = [int(i) for tri in triangles for i in tri]
            mesh.GetFaceVertexCountsAttr().Set(face_counts)
            mesh.GetFaceVertexIndicesAttr().Set(face_indices)
            mesh.GetDisplayColorAttr().Set([color])
        elif geometry is not None and geometry.geometry_type == "cylinder":
            cyl = UsdGeom.Capsule.Define(self._stage, prim_path)
            cyl.GetRadiusAttr().Set(float(geometry.diameter_m / 2.0))
            cyl.GetHeightAttr().Set(float(geometry.length_m))
            cyl.GetAxisAttr().Set("Z")
            cyl.GetDisplayColorAttr().Set([color])
        else:
            sphere = UsdGeom.Sphere.Define(self._stage, prim_path)
            sphere.GetRadiusAttr().Set(float(radius))
            sphere.GetDisplayColorAttr().Set([color])

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

            # Semi-transparent vessel, visible from inside (double-sided)
            cyl.GetDisplayColorAttr().Set([Gf.Vec3f(0.85, 0.85, 0.9)])
            cyl.GetDoubleSidedAttr().Set(True)
            cyl.GetPrim().CreateAttribute(
                "primvars:displayOpacity", Sdf.ValueTypeNames.Float
            ).Set([0.2])

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

    def register_flow_cross_section(
        self,
        nx: int,
        ny: int,
        plane_origin: tuple = (0, 0, 0),
        plane_normal: tuple = (0, 0, 1),
        prim_path: str = "/World/Analysis/FlowField",
        extent_m: float = 0.01,
    ) -> None:
        """Register a flow field cross-section mesh for per-frame color updates.

        Creates an NxN flat quad mesh perpendicular to ``plane_normal``
        at ``plane_origin``, with per-vertex ``primvars:displayColor``.
        The mesh geometry is static; only the colors are updated each
        frame via ``update_flow_cross_section()``.

        Parameters
        ----------
        nx, ny : int
            Grid resolution (number of vertices in each direction).
        plane_origin : tuple
            Centre of the cross-section plane [m].
        plane_normal : tuple
            Normal to the cross-section plane. Default (0,0,1) produces
            an x-y slice perpendicular to the Z vessel axis.
        prim_path : str
            USD prim path for the mesh.
        extent_m : float
            Half-extent of the mesh in each in-plane direction [m].
        """
        # Build local axes for the plane
        normal = np.array(plane_normal, dtype=np.float64)
        normal = normal / max(np.linalg.norm(normal), 1e-30)
        # Choose a perpendicular axis
        if abs(normal[2]) < 0.9:
            up = np.array([0.0, 0.0, 1.0])
        else:
            up = np.array([1.0, 0.0, 0.0])
        u = np.cross(normal, up)
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)

        origin = np.array(plane_origin, dtype=np.float64)

        # Generate vertex positions
        points = []
        for j in range(ny):
            for i in range(nx):
                s = -extent_m + 2.0 * extent_m * i / max(nx - 1, 1)
                t = -extent_m + 2.0 * extent_m * j / max(ny - 1, 1)
                p = origin + s * u + t * v
                points.append(Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])))

        # Generate face topology: (nx-1)*(ny-1) quads
        face_vertex_counts = []
        face_vertex_indices = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                idx = j * nx + i
                face_vertex_counts.append(4)
                face_vertex_indices.extend([idx, idx + 1, idx + nx + 1, idx + nx])

        # Create the mesh prim
        mesh = UsdGeom.Mesh.Define(self._stage, prim_path)
        mesh.GetPointsAttr().Set(points)
        mesh.GetFaceVertexCountsAttr().Set(face_vertex_counts)
        mesh.GetFaceVertexIndicesAttr().Set(face_vertex_indices)

        # Per-vertex displayColor (initial: blue = zero velocity)
        primvar_api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
        color_primvar = primvar_api.CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray,
            UsdGeom.Tokens.vertex,
        )
        initial_colors = [Gf.Vec3f(0.0, 0.0, 0.5)] * (nx * ny)
        color_primvar.Set(initial_colors)

        # Store metadata for per-frame updates
        self._flow_sections: dict[str, dict] = getattr(self, "_flow_sections", {})
        self._flow_sections[prim_path] = {
            "nx": nx, "ny": ny, "n_verts": nx * ny,
        }

    def update_flow_cross_section(
        self,
        velocity_magnitude: np.ndarray,
        prim_path: str = "/World/Analysis/FlowField",
        colormap: str = "viridis",
        time_code: Optional[Any] = None,
    ) -> None:
        """Update per-vertex colors on a registered flow cross-section mesh.

        Parameters
        ----------
        velocity_magnitude : (nx, ny) float32
            Velocity magnitude at each grid point. Values are normalised
            internally (0 = min, 1 = max) before applying the colormap.
        prim_path : str
            USD prim path of the mesh (must match a prior
            ``register_flow_cross_section`` call).
        colormap : str
            Matplotlib colormap name. Default "viridis".
        time_code : Usd.TimeCode, optional
            If provided, writes time-sampled colors at this time code.
        """
        sections = getattr(self, "_flow_sections", {})
        if prim_path not in sections:
            logger.warning(
                "update_flow_cross_section: no mesh at %s — call "
                "register_flow_cross_section() first", prim_path,
            )
            return

        meta = sections[prim_path]
        mag = np.asarray(velocity_magnitude, dtype=np.float32).ravel()
        if len(mag) != meta["n_verts"]:
            logger.warning(
                "update_flow_cross_section: expected %d values, got %d",
                meta["n_verts"], len(mag),
            )
            return

        # Normalise to [0, 1]
        vmin = float(mag.min())
        vmax = float(mag.max())
        if vmax - vmin > 1e-30:
            normed = (mag - vmin) / (vmax - vmin)
        else:
            normed = np.zeros_like(mag)

        # Apply colormap
        try:
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap(colormap)
        except ImportError:
            # Fallback: blue-to-red linear
            cmap = None

        colors = []
        for val in normed:
            if cmap is not None:
                rgba = cmap(float(val))
                colors.append(Gf.Vec3f(float(rgba[0]), float(rgba[1]), float(rgba[2])))
            else:
                colors.append(Gf.Vec3f(float(val), 0.0, float(1.0 - val)))

        # Write to USD
        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return
        primvar_api = UsdGeom.PrimvarsAPI(prim)
        color_primvar = primvar_api.GetPrimvar("displayColor")
        if color_primvar:
            if time_code is not None:
                color_primvar.Set(colors, time_code)
            else:
                color_primvar.Set(colors)

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

    def update(
        self,
        state: dict[str, dict[str, Any]],
        time_code: Optional[Any] = None,
    ) -> None:
        """Write current dynamic state to USD stage.

        Called each timestep. Uses Sdf.ChangeBlock to batch all
        attribute writes into a single stage notification.

        Parameters
        ----------
        state : dict
            Full graph state: {node_name: {field_name: value, ...}, ...}.
            Values are JAX arrays or numpy arrays.
        time_code : Usd.TimeCode, optional
            If provided, writes time-sampled values at this time code
            (for animation recording). If None, writes at the default
            time (current behaviour).
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
                    self._update_robot(ops, node_state, time_code)
                elif reg.prim_type == "field":
                    self._update_field(ops, node_state, time_code)

    def _update_robot(self, ops: list, node_state: dict, time_code=None) -> None:
        """Update robot prim transform from state."""
        pos = node_state.get("position")
        orient = node_state.get("orientation")

        if pos is not None:
            pos_np = np.asarray(pos)
            val = Gf.Vec3d(float(pos_np[0]), float(pos_np[1]), float(pos_np[2]))
            # ops[0] is translate
            if time_code is not None:
                ops[0].Set(val, time_code)
            else:
                ops[0].Set(val)

        if orient is not None:
            q_np = np.asarray(orient)
            val = _quat_to_gf(q_np)
            # ops[1] is orient
            if time_code is not None:
                ops[1].Set(val, time_code)
            else:
                ops[1].Set(val)

    def _update_field(self, ops: list, node_state: dict, time_code=None) -> None:
        """Update field arrow prim from field_vector state."""
        fv = node_state.get("field_vector")
        if fv is None:
            return

        fv_np = np.asarray(fv)
        magnitude = float(np.linalg.norm(fv_np))

        if magnitude < 1e-30:
            # Zero field — hide by scaling to zero
            scale_val = Gf.Vec3d(0, 0, 0)
            if time_code is not None:
                ops[2].Set(scale_val, time_code)
            else:
                ops[2].Set(scale_val)
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
        orient_val = _quat_to_gf(q)
        scale_factor = magnitude * 1e3  # T -> mT, so 10mT = scale 10
        scale_factor = max(scale_factor, 0.1)  # minimum visible size
        scale_val = Gf.Vec3d(scale_factor, scale_factor, scale_factor)

        if time_code is not None:
            ops[1].Set(orient_val, time_code)
            ops[2].Set(scale_val, time_code)
        else:
            ops[1].Set(orient_val)
            ops[2].Set(scale_val)

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

    # -- Materials and scene dressing ----------------------------------------

    def create_material(
        self,
        name: str,
        diffuse_color: tuple = (0.8, 0.8, 0.8),
        opacity: float = 1.0,
        roughness: float = 0.5,
        metallic: float = 0.0,
        ior: float = 1.5,
        specular_color: tuple = (0.5, 0.5, 0.5),
    ) -> str:
        """Create a UsdPreviewSurface material and return its prim path."""
        if not _HAS_USD_SHADE:
            return ""
        mat_path = f"/Materials/{name}"
        mat = UsdShade.Material.Define(self._stage, mat_path)
        shader = UsdShade.Shader.Define(self._stage, f"{mat_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*diffuse_color))
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(ior)
        shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*specular_color))
        mat.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface")
        return mat_path

    def bind_material(self, prim_path: str, material_path: str) -> None:
        """Bind a material to a prim."""
        if not _HAS_USD_SHADE:
            return
        prim = self._stage.GetPrimAtPath(prim_path)
        mat = UsdShade.Material(self._stage.GetPrimAtPath(material_path))
        if prim.IsValid() and mat.GetPrim().IsValid():
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)

    def add_ground_plane(
        self,
        prim_path: str = "/World/Environment/Ground",
        size: float = 0.05,
        offset: float = -0.006,
    ) -> None:
        """Add a ground plane mesh below the scene.

        The plane is in the XZ plane (parallel to the vessel Z axis),
        offset along Y (below the pipe when viewed from the side).
        """
        mesh = UsdGeom.Mesh.Define(self._stage, prim_path)
        s = size
        points = [
            Gf.Vec3f(-s, offset, -s),
            Gf.Vec3f(s, offset, -s),
            Gf.Vec3f(s, offset, s),
            Gf.Vec3f(-s, offset, s),
        ]
        mesh.GetPointsAttr().Set(points)
        mesh.GetFaceVertexCountsAttr().Set([4])
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
        mesh.GetDoubleSidedAttr().Set(True)
        mesh.GetDisplayColorAttr().Set([Gf.Vec3f(0.82, 0.79, 0.75)])

    def add_dome_light(
        self,
        prim_path: str = "/Lights/Dome",
        intensity: float = 500.0,
    ) -> None:
        """Add an ambient dome light for scene illumination."""
        if not _HAS_USD_SHADE:
            return
        light = UsdLux.DomeLight.Define(self._stage, prim_path)
        light.GetIntensityAttr().Set(intensity)
