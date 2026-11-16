"""Shared pieces of the static mesh's tests: closed meshes of MIME's own making (a UV sphere, a box, a tube wall), the
brute-force oracle every crossing of a ray is judged against (float64, every triangle, no tree), the scenes that put a
mesh in the in-loop camera's view, and the planted errors their controls switch on (``CAMERA_PLANT``)."""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from in_loop_cases import LOOK_ACROSS, ballistic, scene_without_camera

from mime import Experiment
from mime.observations import Camera, Detector, World
from mime.scene import Scene, Z, asset
from mime.units import m, ms
from mime.viz import Mesh, write_mesh

#: The vessel's prim path in the scenes below, its layer referenced there.
VESSEL = "/World/Vessel"


def rotation(q: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def uv_sphere(radius: float, around: int) -> tuple[np.ndarray, np.ndarray]:
    """A closed sphere of ``around`` points round each ring and half as many bands, fanned at the poles; outward."""
    bands = max(2, around // 2)
    polar = np.pi * np.arange(1, bands) / bands
    phi = 2.0 * np.pi * np.arange(around) / around
    rings = np.stack([np.outer(np.sin(polar), np.cos(phi)), np.outer(np.sin(polar), np.sin(phi)),
                      np.outer(np.cos(polar), np.ones(around))], axis=-1).reshape(-1, 3) * radius
    points = np.concatenate([[(0.0, 0.0, radius)], rings, [(0.0, 0.0, -radius)]])
    faces: list[tuple[int, int, int]] = []
    last = len(points) - 1
    for i in range(around):
        j = (i + 1) % around
        faces.append((0, 1 + i, 1 + j))
        base = 1 + (bands - 2) * around
        faces.append((last, base + j, base + i))
        for band in range(bands - 2):
            a, b = 1 + band * around, 1 + (band + 1) * around
            faces += [(a + i, b + i, b + j), (a + i, b + j, a + j)]
    return points, np.asarray(faces, dtype=np.int64)


def box(half: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """A closed box of half-sizes ``half``, outward."""
    x, y, z = half
    points = np.array([(sx * x, sy * y, sz * z) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return points, np.asarray([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))], dtype=np.int64)


def tube(inner: float, outer: float, length: float, around: int) -> tuple[np.ndarray, np.ndarray]:
    """A closed annular tube wall along z (outer side, inner side facing the lumen, two end rings), outward."""
    phi = 2.0 * np.pi * np.arange(around) / around
    ring = np.stack([np.cos(phi), np.sin(phi)], axis=-1)
    layers = [(outer, -length / 2), (outer, length / 2), (inner, -length / 2), (inner, length / 2)]
    points = np.concatenate([np.column_stack([r * ring, np.full(around, z)]) for r, z in layers])
    ol, oh, il, ih = 0, around, 2 * around, 3 * around
    faces: list[tuple[int, int, int]] = []
    for i in range(around):
        j = (i + 1) % around
        for quad in ((ol + i, ol + j, oh + j, oh + i), (il + i, ih + i, ih + j, il + j),
                     (ol + i, il + i, il + j, ol + j), (oh + i, oh + j, ih + j, ih + i)):
            a, b, c, d = quad
            faces += [(a, b, c), (a, c, d)]
    return points, np.asarray(faces, dtype=np.int64)


def placed(points: np.ndarray, centre: np.ndarray, turn: np.ndarray) -> np.ndarray:
    return points @ turn.T + centre


@dataclass(frozen=True)
class Crossings:
    """The oracle's answer for each ray: each mesh's signed chord and crossings, the nearest crossing's ``t`` and
    triangle (``-1`` none), the ray's margin (how near, in a triangle's barycentric coordinates, it passes to any
    triangle's edge in front of it: a ray within rounding of an edge may cross either side), each crossing's triangle,
    ``t`` and the scale of its rounding (``t``'s inputs' size over the cosine of incidence: a grazing crossing's ``t`` is
    ill-conditioned), and their sums by ray (``reach``) and for the nearest; and whether the ray ``tied`` (its origin lies
    within the rounding of ``t`` of a triangle it is within ``edge`` of, so that the sign of that ``t``, and with it
    whether the triangle is crossed at all, is below the run's rounding: see :func:`brute`)."""

    chord: np.ndarray
    count: np.ndarray
    nearest: np.ndarray
    triangle: np.ndarray
    margin: np.ndarray
    hits: list[list[tuple[int, float]]]
    reach: np.ndarray
    nearest_reach: np.ndarray
    tied: np.ndarray


def t_rounding(cosine: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """The bound on the watertight test's error in ``t`` for a triangle, in units of ``eps size / cosine`` (``size`` the
    farthest corner's distance from the origin, ``cosine`` the cosine of the incidence, ``shape`` the triangle's longest
    edge squared over twice its area, ``L^2 / 2A``): ``3 + 3 shape tan(theta)``.

    The test moves each corner to the origin and shears it so the ray runs along ``z``; a sheared coordinate takes three
    roundings (the difference, the shear's product, the shear's difference) of at most ``2 size`` each, so ``3 eps size``
    in all (``u = eps / 2``), and ``t`` is the mean of the three corners' ``z`` weighted by the edge functions' shares
    (one in all), which carries the first ``3 eps size``. The shares move by the corners' error over the triangle's
    sheared area, ``3 eps size L / (A cos(theta))``, and each share multiplies a ``z`` that differs across the triangle by
    at most ``L tan(theta)``: ``3 eps size shape tan(theta) / cos(theta)``. Measured on 24 000 triangles an origin lies
    on or near, drawn as G22 draws them, the error is at most 0.31 of it in float64 and 0.26 in float32."""
    return 3.0 + 3.0 * shape * np.sqrt(np.maximum(1.0 - cosine ** 2, 0.0)) / cosine


def b_rounding(cosine: np.ndarray, shape: np.ndarray, reach: np.ndarray) -> np.ndarray:
    """The bound on the watertight test's error in a triangle's barycentric coordinates, in units of ``eps``
    (``reach`` the farthest corner's distance from the origin over the triangle's longest edge, ``size / L``):
    ``4 reach shape / cosine``.

    The ray passes the triangle at a distance of its own size, but each corner is moved to the origin and sheared with an
    error of the scale of ``size``, not of ``L``: ``2 u size`` a coordinate (the difference, then the shear's product, each
    within ``u size``), ``sqrt(2) eps size`` a corner. Moving the three corners by that moves a barycentric coordinate by
    at most ``2 sqrt(2) eps size L / (2 A cos(theta))`` (an edge's weight changes by the corner's shift over the sheared
    triangle's height, ``2 A cos(theta) / L``), which is ``2 sqrt(2) eps reach shape / cos(theta)``; the six products of the
    edge functions, each rounded once (``u |product|``), add at most ``eps shape / cos(theta)``, below ``reach shape``'s
    term since ``reach`` is at least one. Measured on 70 000 triangles a ray passes within 2 % of an edge of, drawn as G22
    draws them, with the ray aimed at a point a drawn weight from an edge, the error is at most 0.51 of it in float32 and
    1.27 of ``reach shape / cos(theta)`` in float64 (not of the bound, which has 4 for them): 0.13 and 0.32 of the bound.
    The old constant, ``MARGIN``, is a declaration that fits a ``reach`` of about 50 and fails above it."""
    return 4.0 * reach * shape / cosine


def brute(triangles: np.ndarray, owners: np.ndarray, meshes: int, origin: np.ndarray,
          directions: np.ndarray, eps: float = 0.0, edge: float = 0.0) -> Crossings:
    """Every crossing of each ray with every triangle, in float64 (Moller-Trumbore's arithmetic, independent of the
    tree): the oracle the tree is judged against.

    **A tie in ``t``.** A ray whose origin is on a triangle (a camera on a wall, a mesh's vertex at the origin) crosses it
    at a ``t`` of zero, whose sign is below the rounding of the crossing test in the run's dtype (``eps``; the bound is
    :func:`t_rounding`): the triangle counts as crossed or not by that sign, and either answer is the right one. A ray
    is ``tied`` where some triangle has ``|t|`` within that bound of zero and the ray within ``edge`` (or that
    triangle's barycentric rounding, whichever is more) of its inside.

    **A tie at an edge.** A ray passing a triangle's edge within the rounding of the barycentric coordinates
    (:func:`b_rounding`, which grows with the triangle's distance over its size and with its grazing) is claimed by
    that triangle or not by rounding, and a fold of two triangles loses or keeps both crossings together: the ray is
    ``tied`` where some triangle in front of it (``t`` above minus its rounding) has its least barycentric coordinate
    within that bound of zero. With ``eps`` zero only an exact zero ties."""
    v0 = triangles[:, 0]
    e1, e2 = triangles[:, 1] - v0, triangles[:, 2] - v0
    normal = np.cross(e1, e2)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-300)
    size = np.linalg.norm(triangles - origin, axis=2).max(axis=1)
    n = len(directions)
    reach = np.zeros(n)
    nearest_reach = np.zeros(n)
    chord = np.zeros((n, meshes))
    count = np.zeros((n, meshes), dtype=np.int64)
    nearest = np.full(n, np.inf)
    which = -np.ones(n, dtype=np.int64)
    margin = np.full(n, np.inf)
    tied = np.zeros(n, dtype=bool)
    shape = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    longest = np.max(np.linalg.norm(triangles[:, [1, 2, 0]] - triangles, axis=2), axis=1)
    shape = longest ** 2 / np.maximum(shape, 1e-300)
    reach_ratio = size / np.maximum(longest, 1e-300)
    hits: list[list[tuple[int, float]]] = []
    s = origin - v0
    q = np.cross(s, e1)
    for r, d in enumerate(directions):
        p = np.cross(d, e2)
        det = np.einsum("ij,ij->i", e1, p)
        usable = np.abs(det) > 0.0
        inv = 1.0 / np.where(usable, det, 1.0)
        u = np.einsum("ij,ij->i", s, p) * inv
        v = (q @ d) * inv
        t = np.einsum("ij,ij->i", e2, q) * inv
        w = 1.0 - u - v
        ahead = usable & (t > 0.0)
        crossed = ahead & (u >= 0.0) & (v >= 0.0) & (w >= 0.0)
        if np.any(ahead):
            margin[r] = float(np.min(np.abs(np.minimum(np.minimum(u, v), w))[ahead]))
        found = np.flatnonzero(crossed)
        hits.append([(int(k), float(t[k])) for k in found])
        cosine = np.maximum(np.abs(normal @ d), 1e-12)
        scale = size / cosine
        least = np.minimum(np.minimum(u, v), w)
        c1 = np.minimum(cosine, 1.0)
        band = eps * scale * t_rounding(c1, shape)
        near = eps * b_rounding(c1, shape, reach_ratio)
        at_origin = usable & (least >= -np.maximum(edge, near)) & (np.abs(t) <= band)
        at_edge = usable & (t > -band) & (np.abs(least) <= near)
        tied[r] = bool(np.any(at_origin | at_edge))
        for k in found:
            chord[r, owners[k]] += t[k] if det[k] < 0.0 else -t[k]
            count[r, owners[k]] += 1
            reach[r] += scale[k]
        if len(found):
            k = int(found[np.argmin(t[found])])
            nearest[r], which[r], nearest_reach[r] = t[k], k, scale[k]
    return Crossings(chord, count, nearest, which, margin, hits, reach, nearest_reach, tied)


def exact_sphere_chord(origin: np.ndarray, directions: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Each unit ray's distance from the sphere's centre (the origin) and its chord through the sphere."""
    along = directions @ -origin
    rho2 = np.maximum(origin @ origin - along * along, 0.0)
    rho = np.sqrt(rho2)
    return rho, 2.0 * np.sqrt(np.maximum(radius * radius - rho2, 0.0))


_written: dict[str, Path] = {}


def vessel_layer(points: np.ndarray, faces: np.ndarray, *, decimation: float | None = None) -> Path:
    """A layer holding the mesh as one ``Mesh`` prim (MIME's own ``write_mesh``), written once per process and mesh."""
    key = repr((points.tobytes(), faces.tobytes(), decimation))
    if key not in _written:
        folder = Path(tempfile.mkdtemp(prefix="mime-mesh-"))
        _written[key] = write_mesh(folder / "vessel.usda", Mesh(points, faces), "Vessel",
                                   decimation_distance=decimation)
    return _written[key]


def with_vessel(exp: Experiment, points: np.ndarray, faces: np.ndarray, **options: object) -> Experiment:
    """``exp`` with the mesh referenced at :data:`VESSEL`, scenery no thing owns."""
    scene = exp._settings.scene  # noqa: SLF001 - the scene the experiment was declared with
    assert scene is not None
    scene.reference(asset(vessel_layer(points, faces, **options)), at=VESSEL)  # type: ignore[arg-type]
    return exp


def tube_mesh_scene(*, around: int = 64, absorbers: tuple[tuple[str, float], ...] = ((VESSEL, 3000.0),),
                    hide: tuple[str, ...] = ("tube",), **camera: object) -> Experiment:
    """The in-loop camera's ballistic sphere and closed-form tube, with a closed tube-wall mesh along x of the same
    radii (64 segments) referenced as scenery; the closed-form tube hidden unless asked."""
    points, faces = tube(60e-6, 70e-6, 480e-6, around)
    turn = rotation((math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0))  # +z onto +x
    held = () if "tube" in hide else (("tube", 3000.0),)
    exp = ballistic(hide=hide, absorbers=(*held, *absorbers), **camera)  # type: ignore[arg-type]
    return with_vessel(exp, placed(points, np.zeros(3), turn), faces)


def mesh_only_scene(points: np.ndarray, faces: np.ndarray, detector: Detector, *,
                    absorbers: tuple[tuple[str, float], ...] = (), **camera: object) -> Experiment:
    """The ballistic ball's scene (no tube) with a mesh as scenery, the camera 1 mm away looking across."""
    exp = scene_without_camera()
    with_vessel(exp, points, faces)
    exp.observe(Camera(detector=detector, mount=World(position=m((0.0, -1e-3, 0.0)), orientation=LOOK_ACROSS),
                       frame_period=ms(3), absorbers=absorbers, hide=("ball",), **camera))  # type: ignore[arg-type]
    return exp


__all__ = ["VESSEL", "Crossings", "Scene", "Z", "box", "brute", "exact_sphere_chord", "mesh_only_scene", "placed",
           "rotation", "tube", "tube_mesh_scene", "uv_sphere", "vessel_layer", "with_vessel"]
