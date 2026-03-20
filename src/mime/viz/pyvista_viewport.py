"""PyVistaViewport — offscreen USD rendering via PyVista/VTK.

Reads a live USD stage, converts UsdGeom prims to VTK geometry (once
at setup), and updates only actor transforms each frame. Renders
offscreen via OSMesa — no display server required.

This is the local development viewport. It is replaced by
HydraStormViewport for production cloud streaming.

Performance rules:
- Geometry meshes are created ONCE at setup (register_stage)
- Per-frame: only actor transforms are updated — no mesh re-creation
- Per-frame cost: O(N_dynamic_prims) attribute reads + one VTK render
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pyvista as pv
    _HAS_PYVISTA = True
except ImportError:
    _HAS_PYVISTA = False

try:
    from pxr import Usd, UsdGeom, Gf
    _HAS_USD = True
except ImportError:
    _HAS_USD = False


def _require_deps() -> None:
    if not _HAS_PYVISTA:
        raise ImportError(
            "PyVistaViewport requires PyVista. "
            "Install with: pip install pyvista"
        )
    if not _HAS_USD:
        raise ImportError(
            "PyVistaViewport requires OpenUSD Python bindings. "
            "Install with: pip install usd-core"
        )


class _TrackedActor:
    """Internal: links a USD prim path to a PyVista actor for transform updates."""

    def __init__(self, prim_path: str, actor: Any, is_dynamic: bool = True):
        self.prim_path = prim_path
        self.actor = actor
        self.is_dynamic = is_dynamic


class PyVistaViewport:
    """Offscreen USD renderer using PyVista/VTK.

    Parameters
    ----------
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.
    background : str
        Background colour name (e.g., "white", "black", "paraview").
    """

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        background: str = "white",
    ):
        _require_deps()

        self._width = width
        self._height = height
        self._background = background

        # Create the offscreen plotter
        self._plotter = pv.Plotter(
            off_screen=True,
            window_size=(width, height),
        )
        self._plotter.set_background(background)

        # Tracked actors: prim_path -> _TrackedActor
        self._actors: dict[str, _TrackedActor] = {}

        # Whether we've done initial setup from a stage
        self._setup_done = False

    def register_stage(self, stage: Any) -> None:
        """Traverse the USD stage and create VTK geometry for all prims.

        This is called ONCE at setup. Geometry is created and added to
        the plotter. Subsequent render() calls only update transforms.

        Parameters
        ----------
        stage : Usd.Stage
            The live USD stage from StageBridge.
        """
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())

            if prim.IsA(UsdGeom.Sphere):
                self._add_sphere(prim, prim_path)
            elif prim.IsA(UsdGeom.Cylinder):
                self._add_cylinder(prim, prim_path)
            elif prim.IsA(UsdGeom.Capsule):
                self._add_capsule(prim, prim_path)
            elif prim.IsA(UsdGeom.Cone):
                self._add_cone(prim, prim_path)
            elif prim.IsA(UsdGeom.Camera):
                self._setup_camera(prim)

        self._setup_done = True

    def _add_sphere(self, prim: Any, prim_path: str) -> None:
        radius = UsdGeom.Sphere(prim).GetRadiusAttr().Get()
        if radius is None:
            radius = 1.0
        mesh = pv.Sphere(radius=float(radius))
        actor = self._plotter.add_mesh(mesh, color="steelblue", name=prim_path)
        self._actors[prim_path] = _TrackedActor(prim_path, actor, is_dynamic=True)

    def _add_cylinder(self, prim: Any, prim_path: str) -> None:
        geom = UsdGeom.Cylinder(prim)
        radius = geom.GetRadiusAttr().Get() or 1.0
        height = geom.GetHeightAttr().Get() or 2.0
        mesh = pv.Cylinder(
            radius=float(radius), height=float(height),
            direction=(0, 0, 1), center=(0, 0, 0),
        )
        # Static geometry (channels) rendered as translucent wireframe
        opacity = 0.3
        actor = self._plotter.add_mesh(
            mesh, color="lightgray", opacity=opacity,
            style="wireframe", name=prim_path,
        )
        self._actors[prim_path] = _TrackedActor(prim_path, actor, is_dynamic=False)

    def _add_capsule(self, prim: Any, prim_path: str) -> None:
        geom = UsdGeom.Capsule(prim)
        radius = geom.GetRadiusAttr().Get() or 1.0
        height = geom.GetHeightAttr().Get() or 2.0
        # Approximate capsule as cylinder (see RENDERING_PLAN.md note)
        mesh = pv.Cylinder(
            radius=float(radius), height=float(height),
            direction=(0, 0, 1), center=(0, 0, 0),
        )
        actor = self._plotter.add_mesh(mesh, color="steelblue", name=prim_path)
        self._actors[prim_path] = _TrackedActor(prim_path, actor, is_dynamic=True)

    def _add_cone(self, prim: Any, prim_path: str) -> None:
        geom = UsdGeom.Cone(prim)
        radius = geom.GetRadiusAttr().Get() or 1.0
        height = geom.GetHeightAttr().Get() or 2.0
        mesh = pv.Cone(
            radius=float(radius), height=float(height),
            direction=(0, 0, 1), center=(0, 0, 0),
        )
        actor = self._plotter.add_mesh(mesh, color="red", name=prim_path)
        self._actors[prim_path] = _TrackedActor(prim_path, actor, is_dynamic=True)

    def _setup_camera(self, prim: Any) -> None:
        """Set the plotter camera from a UsdGeom.Camera prim."""
        xformable = UsdGeom.Xformable(prim)
        world_xform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        # Extract camera position from the transform
        pos = world_xform.ExtractTranslation()
        self._plotter.camera_position = [
            (float(pos[0]), float(pos[1]), float(pos[2])),
            (0, 0, 0),  # focal point
            (0, 0, 1),  # up vector
        ]

    def render(self, stage: Any, camera: str = "/World/Camera") -> np.ndarray:
        """Render the current stage state.

        If register_stage() hasn't been called yet, calls it now.
        Then updates dynamic actor transforms and renders.

        Parameters
        ----------
        stage : Usd.Stage
            The live USD stage.
        camera : str
            USD prim path of the camera (used on first call).

        Returns
        -------
        np.ndarray
            HxWx3 uint8 RGB image.
        """
        if not self._setup_done:
            self.register_stage(stage)

        # Update transforms for dynamic actors only
        for prim_path, tracked in self._actors.items():
            if not tracked.is_dynamic:
                continue

            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue

            xformable = UsdGeom.Xformable(prim)
            ops = xformable.GetOrderedXformOps()
            if not ops:
                continue

            # Build 4x4 transform matrix from xform ops
            world_xform = xformable.ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            # Convert Gf.Matrix4d to numpy 4x4
            mat = np.array(world_xform, dtype=np.float64).reshape(4, 4).T

            # Apply to the VTK actor
            if hasattr(tracked.actor, 'user_matrix'):
                tracked.actor.user_matrix = mat
            elif hasattr(tracked.actor, 'SetUserMatrix'):
                import vtk
                vtk_mat = vtk.vtkMatrix4x4()
                for i in range(4):
                    for j in range(4):
                        vtk_mat.SetElement(i, j, mat[i, j])
                tracked.actor.SetUserMatrix(vtk_mat)

        # Render and capture
        self._plotter.render()
        img = self._plotter.screenshot(return_img=True)

        if img is None:
            return np.zeros((self._height, self._width, 3), dtype=np.uint8)

        # Ensure correct shape and dtype
        if img.shape[2] == 4:
            img = img[:, :, :3]  # RGBA -> RGB

        return img.astype(np.uint8)

    def close(self) -> None:
        """Release rendering resources."""
        if self._plotter is not None:
            self._plotter.close()
            self._plotter = None
