"""A camera on the stage: the ``UsdGeomCamera`` prim a declared camera model is written as, with MIME's attributes for
what USD's camera does not hold, and its read back; and a prim's display primvars read with inheritance.

**The four conventions** between OpenCV's pinhole (what a calibration file states) and USD's camera:

1. **Axes.** OpenCV's optical frame has +y down the image and looks along +z; USD's camera has +y up and looks along
   -z. The prim's rotation is the optical frame's turned half a turn about its +x.
2. **Units.** USD gives a camera's lens and filmback in tenths of a scene unit (millimetres in a stage in centimetres),
   its clipping range in scene units. A pixel has no USD size: the filmback is ``width x pitch`` by
   ``height x pitch`` (``pitch`` the sensor's pixel pitch, written as ``mime:camera:pixel_pitch`` in metres), the focal
   length ``fx x pitch``; ``fy`` differs from ``fx`` by the vertical aperture.
3. **Pixel centres.** OpenCV's first pixel's centre is at ``(0, 0)``; USD's image window runs from the filmback's edge,
   a pixel's centre half a pixel in. The principal point at the image's centre is ``((width - 1) / 2, (height - 1) / 2)``.
4. **Aperture offsets.** USD's offset moves the filmback's centre off the optical axis, in the aperture's units, its
   vertical offset up: ``horizontalApertureOffset = ((width - 1) / 2 - cx) pitch`` and
   ``verticalApertureOffset = (cy - (height - 1) / 2) pitch`` (tenths of a scene unit).

The distortion (OpenCV's five coefficients), the resolution and the pixel pitch are ``mime:camera:*`` attributes. USD
draws the camera's pinhole, not its distortion."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from mime.scene._pxr import Gf, Sdf, UsdGeom, Vt

#: The attributes the camera's prim carries beyond USD's camera schema.
RESOLUTION, PITCH, DISTORTION, DISTORTION_SOURCE, MODEL = (
    "mime:camera:resolution", "mime:camera:pixel_pitch", "mime:camera:distortion", "mime:camera:distortion_source",
    "mime:camera:model")
#: The model a camera's prim declares: OpenCV's pinhole with its five Brown-Conrady coefficients.
PINHOLE_MODEL = "pinhole_brown_conrady"
#: The optical frame (+y down, looking along +z) in USD's camera frame (+y up, looking along -z).
_FLIP = np.diag([1.0, -1.0, -1.0])


class CameraPrim(NamedTuple):
    """A camera's numbers as its prim holds them, in OpenCV's terms and SI: pixels, the five distortion coefficients
    ``(k1, k2, p1, p2, k3)`` and their source, the pixel pitch (m) and the pose (the optical frame's origin and its unit
    quaternion ``(w, x, y, z)`` into the world)."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, float, float, float, float]
    distortion_source: str
    pitch: float
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]


def _matrix(q: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _quaternion(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """The unit quaternion ``(w, x, y, z)`` of a rotation matrix, with ``w >= 0``."""
    m = matrix
    trace = float(np.trace(m))
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = np.array([s / 4.0, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k])
        q = np.zeros(4)
        q[0] = (m[k, j] - m[j, k]) / s
        q[1 + i] = s / 4.0
        q[1 + j] = (m[j, i] + m[i, j]) / s
        q[1 + k] = (m[k, i] + m[i, k]) / s
    q = q if q[0] >= 0.0 else -q
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def write_camera(stage: Any, path: str, camera: CameraPrim, *, meters_per_unit: float,
                 clipping: tuple[float, float] = (1e-3, 10.0)) -> Any:
    """Define ``path`` on ``stage`` as ``camera``'s ``UsdGeomCamera``, its pose its transform (in the stage's units),
    and its ``mime:camera:*`` attributes; ``clipping`` is in metres. Returns the prim."""
    tenths = 10.0 / meters_per_unit
    prim = UsdGeom.Camera.Define(stage, path)
    prim.CreateProjectionAttr(UsdGeom.Tokens.perspective)
    prim.CreateHorizontalApertureAttr(camera.width * camera.pitch * tenths)
    prim.CreateVerticalApertureAttr(camera.height * camera.pitch * camera.fx / camera.fy * tenths)
    prim.CreateFocalLengthAttr(camera.fx * camera.pitch * tenths)
    prim.CreateHorizontalApertureOffsetAttr(((camera.width - 1) / 2.0 - camera.cx) * camera.pitch * tenths)
    prim.CreateVerticalApertureOffsetAttr((camera.cy - (camera.height - 1) / 2.0) * camera.pitch * camera.fx
                                          / camera.fy * tenths)
    prim.CreateClippingRangeAttr(Gf.Vec2f(clipping[0] / meters_per_unit, clipping[1] / meters_per_unit))
    held = prim.GetPrim()
    rotation = _matrix(camera.orientation) @ _FLIP
    xformable = UsdGeom.Xformable(held)
    xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(*(float(c) / meters_per_unit for c in camera.position)))
    q = _quaternion(rotation)
    xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(q[0], Gf.Vec3d(q[1], q[2], q[3])))
    held.CreateAttribute(MODEL, Sdf.ValueTypeNames.Token).Set(PINHOLE_MODEL)
    held.CreateAttribute(RESOLUTION, Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(camera.width, camera.height))
    held.CreateAttribute(PITCH, Sdf.ValueTypeNames.Double).Set(float(camera.pitch))
    held.CreateAttribute(DISTORTION, Sdf.ValueTypeNames.DoubleArray).Set(Vt.DoubleArray(list(camera.distortion)))
    held.CreateAttribute(DISTORTION_SOURCE, Sdf.ValueTypeNames.String).Set(camera.distortion_source)
    return held


def read_camera(prim: Any, *, meters_per_unit: float) -> CameraPrim:
    """A camera prim written by :func:`write_camera` read back into OpenCV's terms and SI."""
    camera = UsdGeom.Camera(prim)
    tenths = 10.0 / meters_per_unit
    width, height = (int(c) for c in prim.GetAttribute(RESOLUTION).Get())
    pitch = float(prim.GetAttribute(PITCH).Get())
    horizontal = float(camera.GetHorizontalApertureAttr().Get()) / tenths
    vertical = float(camera.GetVerticalApertureAttr().Get()) / tenths
    focal = float(camera.GetFocalLengthAttr().Get()) / tenths
    fx, fy = focal * width / horizontal, focal * height / vertical
    cx = (width - 1) / 2.0 - float(camera.GetHorizontalApertureOffsetAttr().Get()) / tenths * width / horizontal
    cy = (height - 1) / 2.0 + float(camera.GetVerticalApertureOffsetAttr().Get()) / tenths * height / vertical
    world = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)).T
    position = tuple(float(c) * meters_per_unit for c in world[:3, 3])
    rotation = world[:3, :3] @ _FLIP
    coefficients = tuple(float(c) for c in prim.GetAttribute(DISTORTION).Get())
    return CameraPrim(width, height, fx, fy, cx, cy, (coefficients[0], coefficients[1], coefficients[2],
                                                      coefficients[3], coefficients[4]),
                      str(prim.GetAttribute(DISTORTION_SOURCE).Get()), pitch,
                      (position[0], position[1], position[2]), _quaternion(rotation))


def display_authors(prim: Any) -> tuple[str | None, str | None]:
    """The path of the prim whose ``displayColor`` and ``displayOpacity`` a prim takes (its own or the nearest
    parent's, ``FindPrimvarWithInheritance``), each ``None`` where no prim on its path authors one."""
    api = UsdGeom.PrimvarsAPI(prim)
    found: list[str | None] = []
    for name in ("displayColor", "displayOpacity"):
        primvar = api.FindPrimvarWithInheritance(name)
        authored = primvar and primvar.HasAuthoredValue() and len(primvar.Get()) > 0
        found.append(str(primvar.GetAttr().GetPrim().GetPath()) if authored else None)
    return found[0], found[1]


def display_primvars(prim: Any) -> tuple[tuple[float, float, float] | None, float | None]:
    """A prim's ``displayColor`` and ``displayOpacity``, each its own or the nearest parent's that authors it
    (``FindPrimvarWithInheritance``), as one colour and one opacity (a constant primvar's value, or the first of an
    array's); ``None`` where no prim on its path authors one."""
    api = UsdGeom.PrimvarsAPI(prim)
    found: list[Any] = []
    for name in ("displayColor", "displayOpacity"):
        primvar = api.FindPrimvarWithInheritance(name)
        value = primvar.Get() if primvar and primvar.HasAuthoredValue() else None
        found.append(None if value is None or len(value) == 0 else value[0])
    colour = None if found[0] is None else (float(found[0][0]), float(found[0][1]), float(found[0][2]))
    return colour, None if found[1] is None else float(found[1])
