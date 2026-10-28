"""A geometry kind's own frame, which is its prim's: the transform its placement settings give (``origin`` a point
its origin sits at, ``z_axis`` a direction its own +Z is turned onto), the settings that transform stands for when it
is read back, and a region primitive carried from the kind's frame into the frame the kind is placed in; and the mesh
helpers its drawing is built with."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from mime.registry import Drawing, Geometry, RegionPrimitive
from mime.units import Magnitude

type Vector = tuple[float, float, float]
type Quaternion = tuple[float, float, float, float]

IDENTITY: Quaternion = (1.0, 0.0, 0.0, 0.0)
#: The placement roles a kind's ``placement`` gives its settings.
ORIGIN, Z_AXIS = "origin", "z_axis"


def onto(axis: Vector) -> Quaternion:
    """The shortest rotation ``(w, x, y, z)`` bringing +Z onto the unit vector ``axis``; about x for -Z."""
    x, y, z = axis
    if z <= -1.0 + 1e-15:
        return (0.0, 1.0, 0.0, 0.0)
    w, ax, ay = 1.0 + z, 0.0 - y, x + 0.0
    norm = math.sqrt(w * w + ax * ax + ay * ay)
    return (w / norm, ax / norm, ay / norm, 0.0)


def rotate(q: Quaternion, v: Vector) -> Vector:
    """``v`` turned by the unit quaternion ``q``."""
    w, x, y, z = q
    tx, ty, tz = 2.0 * (y * v[2] - z * v[1]), 2.0 * (z * v[0] - x * v[2]), 2.0 * (x * v[1] - y * v[0])
    return (v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx))


def _vector(value: Magnitude, name: str) -> Vector:
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError(f"setting {name!r} is three numbers, not {value!r}")
    return (float(value[0]), float(value[1]), float(value[2]))


def kind_frame(kind: Geometry | type[Geometry], settings: Mapping[str, Magnitude]) -> tuple[Vector, Quaternion]:
    """The kind's own frame in the frame its settings are written in: a translation in metres and a unit
    quaternion."""
    translation: Vector = (0.0, 0.0, 0.0)
    orientation = IDENTITY
    for name, role in kind.placement.items():
        if role == ORIGIN:
            translation = _vector(settings[name], name)
        elif role == Z_AXIS:
            orientation = onto(_vector(settings[name], name))
        else:
            raise KeyError(f"{kind.name}'s placement gives {name!r} the role {role!r}, not {ORIGIN} or {Z_AXIS}")
    return translation, orientation


def shape_settings(kind: Geometry | type[Geometry], settings: Mapping[str, Magnitude]) -> dict[str, Magnitude]:
    """The settings that shape the kind, without those that place it."""
    return {name: value for name, value in settings.items() if name not in kind.placement}


def placement_settings(kind: Geometry | type[Geometry], translation: Vector,
                       orientation: Quaternion) -> dict[str, Magnitude]:
    """The placement settings a prim's transform stands for: its translation for ``origin``, and its own +Z turned
    by its rotation for ``z_axis``."""
    found: dict[str, Magnitude] = {}
    for name, role in kind.placement.items():
        found[name] = translation if role == ORIGIN else rotate(orientation, (0.0, 0.0, 1.0))
    return found


def kind_axes(kind: Geometry | type[Geometry], settings: Mapping[str, Magnitude]) -> tuple[Vector, Vector, Vector]:
    """The kind's own axes in the frame its settings are written in: its frame's rotation, by columns, its +Z
    exactly the setting it is turned onto (a rotation of +Z by the quaternion would round it)."""
    translation, orientation = kind_frame(kind, settings)
    del translation
    axes = [rotate(orientation, unit) for unit in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))]
    for name, role in kind.placement.items():
        if role == Z_AXIS:
            axes[2] = _vector(settings[name], name)
    return axes[0], axes[1], axes[2]


def _turned(axes: tuple[Vector, Vector, Vector], v: Vector) -> Vector:
    """``v`` in the frame whose axes are ``axes``, a term dropped where its component is zero so that an axis is
    carried exactly."""
    found = [0.0, 0.0, 0.0]
    for axis, c in zip(axes, v, strict=True):
        if c != 0.0:
            found = [found[i] + c * axis[i] for i in range(3)]
    return (found[0] + 0.0, found[1] + 0.0, found[2] + 0.0)


def placed(primitive: RegionPrimitive, kind: Geometry | type[Geometry],
           settings: Mapping[str, Magnitude]) -> RegionPrimitive:
    """``primitive`` carried from the kind's own frame into the frame the kind is placed in, by its placement
    settings: its point turned and moved, its direction turned."""
    translation, _ = kind_frame(kind, settings)
    axes = kind_axes(kind, settings)
    fields: dict[str, float | Vector] = {}
    for name, value in primitive.fields.items():
        if name in ("centre", "point"):
            turned = _turned(axes, _vector(value, name))
            fields[name] = (turned[0] + translation[0], turned[1] + translation[1], turned[2] + translation[2])
        elif name in ("direction", "normal"):
            fields[name] = _turned(axes, _vector(value, name))
        else:
            fields[name] = value
    return RegionPrimitive(primitive.primitive, fields)


def closed_drawing(points: np.ndarray, faces: list[tuple[int, ...]], representation: str) -> Drawing:
    """A drawing of ``points`` (an ``(n, 3)`` array, metres) and ``faces``, each a tuple of indices into them."""
    return Drawing(tuple((float(p[0]), float(p[1]), float(p[2])) for p in points), tuple(len(f) for f in faces),
                   tuple(i for f in faces for i in f), representation)


def banded_faces(rings: int, around: int, first: int | None, last: int | None, start: int = 1) -> list[tuple[int, ...]]:
    """The faces of ``rings`` rings of ``around`` points each, numbered from ``start`` ring by ring with ``j`` round
    each, ring ``i`` turning forward and ring ``i + 1`` lying further along: a quad between each two neighbouring
    rings, and a fan from the point ``first`` to the first ring and from ``last`` to the last (none where ``None``),
    each face counter-clockwise seen from the side that the direction along the rings crossed with the direction round
    them points to."""
    def at(i: int, j: int) -> int:
        return start + i * around + j % around

    faces: list[tuple[int, ...]] = []
    if first is not None:
        faces += [(first, at(0, j), at(0, j + 1)) for j in range(around)]
    for i in range(rings - 1):
        faces += [(at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)) for j in range(around)]
    if last is not None:
        faces += [(last, at(rings - 1, j + 1), at(rings - 1, j)) for j in range(around)]
    return faces


__all__ = ["IDENTITY", "ORIGIN", "Z_AXIS", "Quaternion", "Vector", "banded_faces", "closed_drawing", "kind_axes",
           "kind_frame", "onto", "placed", "placement_settings", "rotate", "shape_settings"]
