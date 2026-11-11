"""G10, the camera prim's four conventions (axes, tenths of a scene unit, pixel centres, aperture offsets): a camera
written to a stage in millimetres holds the hand-computed USD numbers, reads back as itself, and USD's own camera
(``GfCamera``'s frustum, which knows nothing of MIME) projects points where ``mime.sensing.project`` does; and a prim's
display primvars are read with inheritance.

**The controls:** the optical frame written without its half turn (+y down taken as USD's +y up,
``CAMERA_PLANT=y-up``) fails the frustum check; apertures written in scene units, not tenths (``CAMERA_PLANT=units``),
fail the hand numbers."""

from __future__ import annotations

import math
import os

import jax.numpy as jnp
import numpy as np
import pytest

import mime.scene._camera as camera_module
from mime.scene import display_primvars, read_camera, write_camera
from mime.scene._pxr import Gf, Sdf, Usd, UsdGeom, Vt
from mime.sensing import BrownConrady, CameraPose, Pinhole, camera_prim, display, from_camera_prim, project

PLANT = os.environ.get("CAMERA_PLANT", "")
MM = 1e-3


@pytest.fixture
def written(monkeypatch: pytest.MonkeyPatch):
    if PLANT == "y-up":
        monkeypatch.setattr(camera_module, "_FLIP", np.eye(3))
    camera = Pinhole(640, 480, 800.0, 820.0, 315.5, 243.25, BrownConrady(-0.1, 0.02, 1e-3, -2e-3, 0.0))
    s = math.sqrt(0.5)
    pose = CameraPose(jnp.asarray([0.01, -0.1, 0.002]), jnp.asarray([s, -s, 0.0, 0.0]))
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, MM)
    held = camera_prim(camera, pose, pitch=5e-6)
    prim = write_camera(stage, "/World/camera", held, meters_per_unit=MM * (10.0 if PLANT == "units" else 1.0))
    # The stage is returned with its prim: a prim of a stage nothing holds has expired.
    return camera, pose, held, prim, stage


def test_the_prim_holds_the_hand_numbers(written):
    _, _, _, prim, _ = written
    cam = UsdGeom.Camera(prim)
    # 640 px x 5 um = 3.2 mm = 32 tenths of a millimetre; 800 px x 5 um = 4 mm = 40; fy 820: 480 x 5 um x 800 / 820.
    assert cam.GetHorizontalApertureAttr().Get() == pytest.approx(32.0, rel=1e-6)
    assert cam.GetVerticalApertureAttr().Get() == pytest.approx(2.4 * 800 / 820 * 10, rel=1e-6)
    assert cam.GetFocalLengthAttr().Get() == pytest.approx(40.0, rel=1e-6)
    # The principal point 4 px left of the centre (319.5): the filmback 4 x 5 um = 0.02 mm right, 0.2 tenths; 3.75 px
    # below the centre (239.5): the filmback 3.75 x 5 um x 800 / 820 up.
    assert cam.GetHorizontalApertureOffsetAttr().Get() == pytest.approx(0.2, rel=1e-5)
    assert cam.GetVerticalApertureOffsetAttr().Get() == pytest.approx(3.75 * 0.05 * 800 / 820, rel=1e-5)
    world = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)).T
    # USD's camera looks along its -z (the world's +y here) with its +y up (the world's +z); its origin in mm.
    assert np.allclose(world[:3, 2], (0.0, -1.0, 0.0), atol=1e-12)
    assert np.allclose(world[:3, 1], (0.0, 0.0, 1.0), atol=1e-12)
    assert np.allclose(world[:3, 3], (10.0, -100.0, 2.0), atol=1e-9)


def test_the_prim_reads_back_as_the_camera(written):
    camera, pose, held, prim, _ = written
    back = read_camera(prim, meters_per_unit=MM)
    assert back.width == 640 and back.height == 480 and back.distortion == held.distortion
    for name in ("fx", "fy", "cx", "cy"):
        assert getattr(back, name) == pytest.approx(getattr(held, name), rel=1e-6, abs=1e-4)
    assert np.allclose(back.position, held.position, atol=1e-12)
    assert np.allclose(back.orientation, held.orientation, atol=1e-9)
    again, _ = from_camera_prim(back)
    assert again.distortion.k1 == camera.distortion.k1


def test_usds_own_camera_projects_where_ours_does(written):
    camera, pose, _, prim, _ = written
    frustum = UsdGeom.Camera(prim).GetCamera(0.0).frustum
    matrix = frustum.ComputeViewMatrix() * frustum.ComputeProjectionMatrix()
    rng = np.random.default_rng(3)
    points = np.stack([rng.uniform(-4e-3, 4e-3, 12) + 0.01, rng.uniform(-0.02, 0.02, 12),
                       rng.uniform(-3e-3, 3e-3, 12) + 0.002], axis=-1)
    pinhole = Pinhole(640, 480, camera.fx, camera.fy, camera.cx, camera.cy)
    ours, _ = project(pinhole, pose, jnp.asarray(points))
    for point, expected in zip(points, np.asarray(ours), strict=True):
        ndc = matrix.Transform(Gf.Vec3d(*(point / MM)))
        # USD's image window runs from the filmback's edge: a pixel's centre is half a pixel in.
        u = (ndc[0] + 1.0) / 2.0 * 640 - 0.5
        v = (1.0 - ndc[1]) / 2.0 * 480 - 0.5
        assert abs(u - expected[0]) <= 1e-3 and abs(v - expected[1]) <= 1e-3, (u, v, expected)


def test_display_primvars_are_read_with_inheritance():
    stage = Usd.Stage.CreateInMemory()
    parent = UsdGeom.Xform.Define(stage, "/World/tube").GetPrim()
    UsdGeom.PrimvarsAPI(parent).CreatePrimvar("displayOpacity", Sdf.ValueTypeNames.FloatArray,
                                              UsdGeom.Tokens.constant).Set(Vt.FloatArray([0.3]))
    UsdGeom.PrimvarsAPI(parent).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray,
                                              UsdGeom.Tokens.constant).Set(Vt.Vec3fArray([Gf.Vec3f(0.85, 0.85, 0.9)]))
    child = UsdGeom.Mesh.Define(stage, "/World/tube/proxy").GetPrim()
    colour, opacity = display_primvars(child)
    assert opacity == pytest.approx(0.3) and colour == pytest.approx((0.85, 0.85, 0.9))
    other = UsdGeom.Xform.Define(stage, "/World/other").GetPrim()
    assert display_primvars(other) == (None, None)
    assert display(child).opacity == pytest.approx(0.3)
