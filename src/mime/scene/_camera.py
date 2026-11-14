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

from mime.scene._pxr import Gf, Sdf, Usd, UsdGeom, Vt

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


#: What a non-constant primvar's interpolation is called in a note.
_INTERPOLATIONS = {"uniform": "per-face (uniform)", "vertex": "per-vertex", "varying": "per-vertex (varying)",
                   "faceVarying": "per-corner (faceVarying)"}
#: The two display primvars, with the width of one value.
_DISPLAY = (("displayColor", 3), ("displayOpacity", 1))


class DisplayRead(NamedTuple):
    """A prim's display primvars as USD composes them: the colour and opacity (``None`` where no prim on its path
    authors one, or the nearest authors an empty array), and, for each primvar that is not constant, what reducing it to
    one value did (``reduced``, one line each, naming the primvar, its interpolation, the prim that authors it and
    whether the mean is weighted by area)."""

    colour: tuple[float, float, float] | None
    opacity: float | None
    reduced: tuple[str, ...] = ()


def _weights(prim: Any, interpolation: str, count: int) -> np.ndarray | None:
    """The area each value of a mesh's primvar stands for (a face's, spread over its corners where the primvar is per
    vertex or per corner), from the mesh's own points; ``None`` where the prim is no mesh, holds no faces or the counts
    do not match."""
    if not prim.IsA(UsdGeom.Mesh):
        return None
    mesh = UsdGeom.Mesh(prim)
    time = Usd.TimeCode.EarliestTime()
    counts, indices, points = (mesh.GetFaceVertexCountsAttr().Get(time), mesh.GetFaceVertexIndicesAttr().Get(time),
                               mesh.GetPointsAttr().Get(time))
    if counts is None or indices is None or points is None or len(counts) == 0:
        return None
    sizes, corners = np.asarray(counts, dtype=np.int64), np.asarray(indices, dtype=np.int64)
    xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if sizes.sum() != len(corners) or (len(corners) and corners.max() >= len(xyz)):
        return None
    areas = np.zeros(len(sizes))
    start = 0
    for face, size in enumerate(sizes):
        p = xyz[corners[start:start + size]]
        start += size
        if size >= 3:
            areas[face] = 0.5 * np.linalg.norm(np.cross(p[1:-1] - p[0], p[2:] - p[0]).sum(axis=0))
    if interpolation == "uniform":
        found = areas
    else:
        per_corner = np.repeat(areas / np.maximum(sizes, 1), sizes)
        if interpolation == "faceVarying":
            found = per_corner
        else:
            found = np.zeros(len(xyz))
            np.add.at(found, corners, per_corner)
    return found if len(found) == count and found.sum() > 0.0 else None


def _read_primvar(prim: Any, name: str) -> tuple[np.ndarray, str | None] | None:
    """The values of ``name`` as USD composes it on ``prim`` (its own or the nearest parent's constant one, indices
    resolved by ``ComputeFlattened``, an array of one row a value), reduced to one row, with the line that says so where
    it was not constant; ``None`` where none is authored or the array is empty."""
    primvar = UsdGeom.PrimvarsAPI(prim).FindPrimvarWithInheritance(name)
    if not primvar or not primvar.HasAuthoredValue():
        return None
    held = primvar.ComputeFlattened(Usd.TimeCode.EarliestTime())
    if held is None or len(held) == 0:
        return None
    width = dict(_DISPLAY)[name]
    values = np.asarray(held, dtype=np.float64).reshape(len(held), width)
    interpolation = str(primvar.GetInterpolation())
    if interpolation == "constant" or len(values) == 1:
        return values[0:1], None
    weights = _weights(prim, interpolation, len(values))
    where = str(primvar.GetAttr().GetPrim().GetPath())
    how = "area-weighted mean" if weights is not None else "mean"
    reduced = np.average(values, axis=0, weights=weights)[None, :]
    return reduced, (f"{name} {_INTERPOLATIONS.get(interpolation, interpolation)} on {where} reduced to its {how}")


def read_display(prim: Any) -> DisplayRead:
    """A prim's ``displayColor`` and ``displayOpacity`` as USD composes them (``FindPrimvarWithInheritance``: its own or
    the nearest parent's constant one; indices resolved; a time-sampled one at its earliest sample), each as one value.
    A primvar that is not constant (per face, per vertex, per corner) is reduced to the mean of its values, weighted by
    area on a mesh, and ``reduced`` says so: never silently."""
    found = {name: _read_primvar(prim, name) for name, _ in _DISPLAY}
    colour, opacity = found["displayColor"], found["displayOpacity"]
    return DisplayRead(None if colour is None else (float(colour[0][0, 0]), float(colour[0][0, 1]), float(colour[0][0, 2])),
                   None if opacity is None else float(opacity[0][0, 0]),
                   tuple(line for held in (colour, opacity) if held is not None and (line := held[1]) is not None))


def display_primvars(prim: Any) -> tuple[tuple[float, float, float] | None, float | None]:
    """``read_display`` without the note of what it reduced: the colour and the opacity."""
    held = read_display(prim)
    return held.colour, held.opacity


def display_authors(prim: Any) -> tuple[str | None, str | None]:
    """The path of the prim whose ``displayColor`` and ``displayOpacity`` a prim takes (its own or the nearest
    parent's, ``FindPrimvarWithInheritance``), each ``None`` where none is authored or the nearest's array is empty."""
    found: list[str | None] = []
    for name, _ in _DISPLAY:
        primvar = UsdGeom.PrimvarsAPI(prim).FindPrimvarWithInheritance(name)
        authored = primvar and primvar.HasAuthoredValue() and len(primvar.ComputeFlattened()) > 0
        found.append(str(primvar.GetAttr().GetPrim().GetPath()) if authored else None)
    return found[0], found[1]
