"""Scenery for the reader's tests: two small USD assets written by the tests, a vessel (a closed mesh, its display
primvars on its root prim, so the mesh inherits them; a derived-looking proxy mesh) and a ball (a sphere part below a
root that authors the colour and opacity its part inherits; a mesh of its own skin below the body's prim), and a scene
referencing both with a body bound to the ball. MIME's own stand-ins, never a robot's meshes."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

from mime import Body, Experiment, State
from mime.dynamics import RigidInertial
from mime.observations import Camera, Detector, World
from mime.scene import Scene, Z, asset
from mime.things import Part
from mime.scene import Sphere
from mime.units import kg_per_m3, m, m_per_s, ms, rad_per_s, um

SQRT_HALF = 0.5 ** 0.5
LOOK_ACROSS = (SQRT_HALF, -SQRT_HALF, 0.0, 0.0)
DETECTOR = Detector(width=64, height=48, fx=300.0, fy=300.0, cx=31.5, cy=23.5)
#: What the vessel's root authors, which its mesh inherits, and what the ball's root authors for its part.
VESSEL_DISPLAY = ((0.8, 0.2, 0.1), 0.35)
BALL_DISPLAY = ((0.1, 0.4, 0.9), 0.5)
VESSEL_MESH = "/World/Vessel/Wall"
BOX_FACES = ((0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7),
             (2, 7, 6), (3, 0, 4), (3, 4, 7))


def _box(half: float) -> np.ndarray:
    return np.array([(-half, -half, -half), (half, -half, -half), (half, half, -half), (-half, half, -half),
                     (-half, -half, half), (half, -half, half), (half, half, half), (-half, half, half)])


def _mesh(stage: Usd.Stage, path: str, half: float, *, derived: bool = False) -> None:
    held = UsdGeom.Mesh.Define(stage, path)
    held.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(_box(half).astype(np.float32)))
    held.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(BOX_FACES)))
    held.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for face in BOX_FACES for i in face]))
    held.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    if derived:
        held.GetPrim().CreateAttribute("mime:derived_from", Sdf.ValueTypeNames.String).Set("tube 0.1.0 abc")


def _display(prim: UsdGeom.Xform, display: tuple[tuple[float, float, float], float] | None) -> None:
    if display is None:
        return
    api = UsdGeom.PrimvarsAPI(prim)
    api.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.constant).Set(
        Vt.Vec3fArray([Gf.Vec3f(*display[0])]))
    api.CreatePrimvar("displayOpacity", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.constant).Set(
        Vt.FloatArray([display[1]]))


def primvar(prim: Usd.Prim, name: str, interpolation: str, values: list, indices: list[int] | None = None) -> None:
    """``name`` (``displayColor`` or ``displayOpacity``) authored on ``prim`` with ``interpolation``, its ``values`` and
    optionally the indices into them."""
    colour = name == "displayColor"
    held = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
        name, Sdf.ValueTypeNames.Color3fArray if colour else Sdf.ValueTypeNames.FloatArray, interpolation)
    held.Set(Vt.Vec3fArray([Gf.Vec3f(*v) for v in values]) if colour else Vt.FloatArray([float(v) for v in values]))
    if indices is not None:
        held.SetIndices(Vt.IntArray(indices))


def _root(stage: Usd.Stage, name: str) -> UsdGeom.Xform:
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    return root


def write_vessel(path: Path, display: tuple[tuple[float, float, float], float] | None = VESSEL_DISPLAY) -> Path:
    """A vessel: ``/Vessel/Wall``, a closed 2 mm box mesh below a root that authors ``display``, and ``/Vessel/Proxy``,
    a mesh a kind derived (``mime:derived_from``)."""
    stage = Usd.Stage.CreateInMemory()
    root = _root(stage, "Vessel")
    _display(root, display)
    _mesh(stage, "/Vessel/Wall", 1e-3)
    _mesh(stage, "/Vessel/Proxy", 5e-4, derived=True)
    path.write_text(stage.GetRootLayer().ExportToString(), encoding="utf-8")
    return path


def write_ball(path: Path, display: tuple[tuple[float, float, float], float] | None = BALL_DISPLAY,
               shell: tuple[tuple[str, str, list, list[int] | None], ...] = ()) -> Path:
    """A ball: ``/Ball/Shell``, a sphere part below a root that authors ``display``, and ``/Ball/Skin``, a mesh of the
    body's own. ``shell`` authors primvars on the part itself, each ``(name, interpolation, values, indices)``."""
    stage = Usd.Stage.CreateInMemory()
    root = _root(stage, "Ball")
    _display(root, display)
    sphere = UsdGeom.Sphere.Define(stage, "/Ball/Shell")
    sphere.CreateRadiusAttr(20e-6)
    for each in shell:
        primvar(sphere.GetPrim(), *each)
    _mesh(stage, "/Ball/Skin", 2e-5)
    path.write_text(stage.GetRootLayer().ExportToString(), encoding="utf-8")
    return path


def write_gprims(path: Path) -> Path:
    """Unowned gprims that are not a mesh with a body's look: a cube, a capsule, a curve, and a mesh whose points are
    time-sampled only (at 1 s and 2 s, no default)."""
    stage = Usd.Stage.CreateInMemory()
    _root(stage, "Extras")
    UsdGeom.Cube.Define(stage, "/Extras/Cube").CreateSizeAttr(1e-3)
    UsdGeom.Capsule.Define(stage, "/Extras/Capsule")
    curves = UsdGeom.BasisCurves.Define(stage, "/Extras/Curves")
    curves.CreateTypeAttr(UsdGeom.Tokens.linear)
    curves.CreateCurveVertexCountsAttr([2])
    curves.CreatePointsAttr([(0.0, 0.0, 0.0), (1e-3, 0.0, 0.0)])
    animated = UsdGeom.Mesh.Define(stage, "/Extras/Animated")
    animated.CreateFaceVertexCountsAttr([3])
    animated.CreateFaceVertexIndicesAttr([0, 1, 2])
    animated.CreatePointsAttr().Set(Vt.Vec3fArray([(0, 0, 0), (1e-3, 0, 0), (0, 1e-3, 0)]), 1.0)
    animated.GetPointsAttr().Set(Vt.Vec3fArray([(0, 0, 0), (2e-3, 0, 0), (0, 2e-3, 0)]), 2.0)
    path.write_text(stage.GetRootLayer().ExportToString(), encoding="utf-8")
    return path


def write_faces(path: Path, *, colours: list | None = None, opacities: list | None = None,
                indices: list[int] | None = None, interpolation: str = "uniform") -> Path:
    """``/Faces``: a mesh of two triangles of areas 1 and 3 (in square mm), with ``displayColor`` and ``displayOpacity``
    authored on it with ``interpolation`` (and ``indices`` into them where given)."""
    stage = Usd.Stage.CreateInMemory()
    root = _root(stage, "Faces")
    mesh = UsdGeom.Mesh.Define(stage, "/Faces/Mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1e-3, 0, 0), (0, 2e-3, 0), (0, 0, 0), (3e-3, 0, 0), (0, 2e-3, 0)])
    mesh.CreateFaceVertexCountsAttr([3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 4, 5])
    if colours is not None:
        primvar(mesh.GetPrim(), "displayColor", interpolation, colours, indices)
    if opacities is not None:
        primvar(mesh.GetPrim(), "displayOpacity", interpolation, opacities, indices)
    assert root is not None
    path.write_text(stage.GetRootLayer().ExportToString(), encoding="utf-8")
    return path


_written: dict[str, Path] = {}


def assets(vessel: tuple[tuple[float, float, float], float] | None = VESSEL_DISPLAY,
           ball: tuple[tuple[float, float, float], float] | None = BALL_DISPLAY) -> tuple[Path, Path]:
    """The two assets, written once per process and set of displays, in a folder of this process's own."""
    key = repr((vessel, ball))
    if key not in _written:
        folder = Path(tempfile.mkdtemp(prefix="mime-scenery-"))
        _written[key] = folder
        write_vessel(folder / "vessel.usda", vessel)
        write_ball(folder / "ball.usda", ball)
    folder = _written[key]
    return folder / "vessel.usda", folder / "ball.usda"


def scenery_scene(*, vessel: bool = True, extras: Path | None = None,
                  shell: tuple[tuple[str, str, list, list[int] | None], ...] = (), hide: tuple[str, ...] = (), absorbers: tuple[tuple[str, float], ...] = (),
                  vessel_display: tuple[tuple[float, float, float], float] | None = VESSEL_DISPLAY,
                  ball_display: tuple[tuple[float, float, float], float] | None = BALL_DISPLAY) -> Experiment:
    """A ball moving along x in a scene that may reference a vessel, seen by a camera that sees the whole scene."""
    vessel_file, ball_file = assets(vessel_display, ball_display)
    if shell:
        ball_file = write_ball(ball_file.with_name(f"ball_{abs(hash(repr(shell)))}.usda"), ball_display, shell)
    scene = Scene.new(up=Z, units="m")
    if vessel:
        scene.reference(asset(vessel_file), at="/World/Vessel")
    if extras is not None:
        scene.reference(asset(extras), at="/World/Extras")
    scene.reference(asset(ball_file), at="/World/Ball")
    exp = Experiment("scenery", scene=scene, dt=ms(1))
    exp.add_body(Body("ball", prim=scene["/World/Ball"], dynamics=RigidInertial(), initial=State(
        position=um((-40.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((2e-3, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"Shell": Part(Sphere(radius=um(20)))}))
    exp.declare("ball.parts.Shell.density", kg_per_m3(7850.0), reason="steel")
    exp.observe(Camera(detector=DETECTOR, mount=World(position=m((0.0, -1e-3, 0.0)), orientation=LOOK_ACROSS),
                       frame_period=ms(3), hide=hide, absorbers=absorbers))
    return exp
