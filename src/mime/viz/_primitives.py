"""USD's analytic shapes as triangles, for a format that holds only meshes (glTF): the sphere, the cylinder, the cube
and the capsule, each in its prim's own frame, centred on its origin along its ``axis``, with outward normals and
counter-clockwise faces seen from outside. ``SEGMENTS`` facets a turn: each shape inscribed in its own, a sphere's
volume 0.71% short of it, a cylinder's 0.29%, a capsule's between (``tests/viz/test_gltf_primitives.py``), which a
view needs and a computation never reads."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Facets a turn of a curved shape, and half as many rings from pole to pole.
SEGMENTS = 48
#: The shapes drawn, by their USD type names.
KINDS = ("Sphere", "Cylinder", "Cube", "Capsule")


@dataclass(frozen=True)
class Shape:
    """Triangles over points, each point with its unit normal, in the prim's frame."""

    points: np.ndarray
    normals: np.ndarray
    triangles: np.ndarray


def _ring(count: int) -> tuple[np.ndarray, np.ndarray]:
    turn = np.arange(count) * (2.0 * math.pi / count)
    return np.cos(turn), np.sin(turn)


def _rings(radius: float, segments: int, rings: int, *, half: float = 0.0) -> Shape:
    """A sphere of ``radius`` from pole to pole, ``rings`` bands of ``segments`` facets; with ``half`` its northern
    hemisphere moved ``half`` up the axis and its southern down, the equator drawn twice and joined by a band: a
    capsule's spine of twice ``half``."""
    cos, sin = _ring(segments)
    points: list[np.ndarray] = [np.array([[0.0, 0.0, radius + half]])]
    normals: list[np.ndarray] = [np.array([[0.0, 0.0, 1.0]])]
    levels: list[tuple[float, float]] = []
    for ring in range(1, rings):
        polar = math.pi * ring / rings
        if 2 * ring < rings:
            levels.append((polar, half))
        elif 2 * ring > rings:
            levels.append((polar, -half))
        else:
            levels += [(polar, half), (polar, -half)] if half else [(polar, 0.0)]
    for polar, moved in levels:
        across, up = math.sin(polar), math.cos(polar)
        unit = np.stack([across * cos, across * sin, np.full(segments, up)], axis=1)
        normals.append(unit)
        points.append(unit * radius + np.array([0.0, 0.0, moved]))
    points.append(np.array([[0.0, 0.0, -radius - half]]))
    normals.append(np.array([[0.0, 0.0, -1.0]]))
    faces: list[tuple[int, int, int]] = []
    following = [(i + 1) % segments for i in range(segments)]
    faces += [(0, 1 + i, 1 + following[i]) for i in range(segments)]
    for band in range(len(levels) - 1):
        upper, lower = 1 + band * segments, 1 + (band + 1) * segments
        for i in range(segments):
            j = following[i]
            faces += [(upper + i, lower + i, lower + j), (upper + i, lower + j, upper + j)]
    last, pole = 1 + (len(levels) - 1) * segments, 1 + len(levels) * segments
    faces += [(last + i, pole, last + following[i]) for i in range(segments)]
    return Shape(np.concatenate(points), np.concatenate(normals), np.asarray(faces, dtype=np.int64))


def _cylinder(radius: float, height: float, segments: int) -> Shape:
    cos, sin = _ring(segments)
    round_ = np.stack([cos, sin, np.zeros(segments)], axis=1)
    half = np.array([0.0, 0.0, height / 2.0])
    side = np.concatenate([round_ * radius - half, round_ * radius + half])
    up, down = np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])
    top = np.concatenate([half[None], round_ * radius + half])
    bottom = np.concatenate([-half[None], round_ * radius - half])
    points = np.concatenate([side, top, bottom])
    normals = np.concatenate([round_, round_, np.tile(up, (segments + 1, 1)), np.tile(down, (segments + 1, 1))])
    faces: list[tuple[int, int, int]] = []
    for i in range(segments):
        j = (i + 1) % segments
        faces += [(i, j, segments + j), (i, segments + j, segments + i)]
    t, b = 2 * segments, 3 * segments + 1
    faces += [(t, t + 1 + i, t + 1 + (i + 1) % segments) for i in range(segments)]
    faces += [(b, b + 1 + (i + 1) % segments, b + 1 + i) for i in range(segments)]
    return Shape(points, normals, np.asarray(faces, dtype=np.int64))


def _cube(size: float) -> Shape:
    points: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []
    for axis in range(3):
        for sign in (1.0, -1.0):
            normal = np.zeros(3)
            normal[axis] = sign
            u, v = np.zeros(3), np.zeros(3)
            u[(axis + 1) % 3], v[(axis + 2) % 3] = sign, 1.0
            base = len(points) * 4
            corners = [normal + a * u + b * v for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            points.append(np.asarray(corners) * (size / 2.0))
            normals.append(np.tile(normal, (4, 1)))
            faces += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    return Shape(np.concatenate(points), np.concatenate(normals), np.asarray(faces, dtype=np.int64))


def _along(shape: Shape, axis: str) -> Shape:
    """``shape``, built along Z, turned so that Z lies along ``axis`` (a cyclic permutation: a rotation)."""
    order = {"X": [2, 0, 1], "Y": [1, 2, 0]}.get(axis.upper())
    if order is None:
        return shape
    return Shape(shape.points[:, order], shape.normals[:, order], shape.triangles)


def tessellate(kind: str, *, radius: float = 1.0, height: float = 2.0, size: float = 2.0, axis: str = "Z",
               segments: int = SEGMENTS) -> Shape | None:
    """The triangles of USD's shape ``kind`` with its attributes (a sphere's ``radius``; a cylinder's and a capsule's
    ``radius``, ``height`` and ``axis``, the capsule's height its spine's, between its two half spheres; a cube's edge
    ``size``); ``None`` for a kind this does not draw."""
    if kind == "Sphere":
        return _rings(radius, segments, segments // 2)
    if kind == "Cylinder":
        return _along(_cylinder(radius, height, segments), axis)
    if kind == "Capsule":
        return _along(_rings(radius, segments, segments // 2, half=height / 2.0), axis)
    if kind == "Cube":
        return _cube(size)
    return None


def signed_volume(points: np.ndarray, triangles: np.ndarray) -> float:
    """The volume a closed mesh encloses, positive where its faces turn counter-clockwise seen from outside."""
    a, b, c = (points[triangles[:, k]] for k in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


__all__ = ["KINDS", "SEGMENTS", "Shape", "signed_volume", "tessellate"]
