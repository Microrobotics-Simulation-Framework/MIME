"""G26, the tree at the edges: every ray through an edge or a vertex of a mesh is claimed by exactly one of the triangles
that share it, and a closed mesh is crossed an even number of times by every ray from outside, whatever the rounding.

G22 leaves out every ray within ``MARGIN`` of an edge (1e-4 in float32), which is where a crossing is lost or counted
twice: one crossing more or less moves a ray's signed chord by its whole ``t``. This gate goes at the edges.

**The draws** (physical scales; CLAUDE.md, "Property tests"): a triangle's edge 25 um to 1 mm long, the origin 3 edge
lengths to 5 mm away within 50 degrees of the surface's normal, a surface tilted 17 degrees at most; for the closed meshes
(a UV sphere, a box, a tube wall, 50 um to 2 mm across, turned at random) an origin outside, 1.5 radii to 3 mm beyond.
Rays are aimed at points of an edge (and at its vertices), half of them exactly and half within 32 roundings of the
edge (a rounding being the unit roundoff of the run's dtype times the distance), the places a pair of triangles
disagree. In the dtype of the run.

- **A shared edge**: two triangles ``(a, b, c)`` and ``(d, c, b)`` on either side of the edge ``b c`` (the winding of a
  consistent mesh): every ray aimed at the middle of the edge, or within 1e-3 of its length of it, has exactly one
  crossing: a ray aimed at one of the edge's points lies in the union of the two, and the watertight test claims it once
  (``_crossings``: each edge's function from its two corners alone, so the two triangles compute exactly opposite
  values).
- **A shared vertex**: four to eight triangles fanned round a vertex, a ray aimed at the vertex or within 32 roundings of
  it: exactly one.
- **A closed mesh** (drawn from the three kinds, and a box alone, the kind whose edges the slab loses most often): every ray
  aimed at a point of an edge or a vertex of the mesh, from outside it: an even number of crossings. The slab test of the tree's boxes drops a crossing at an edge (a silhouette, where the box's entry and exit
  are one distance) without the widening of :func:`mime.sensing._mesh.slab_pad`.
- **An axis-aligned camera on an axis-aligned box** (``mesh_chords``, a 64 x 48 detector looking along +z): the box's
  corners are where the pixel-centre rays of two chosen pixels cross the box's front plane, so a column, a row and
  the corner pixels of rays run along the box's silhouette edges exactly (up to the rounding of the ray and the corner
  in the dtype), the case an image of a box in a camera lined up with it holds: every sub-ray an even number of crossings.
  A ray tangent at a silhouette edge is dropped or kept with both its crossings, so this case is caught by ``no-barrier``
  and **not** by ``no-pad`` (0 odd sub-rays of 1.66 million in 60 draws each dtype, measured): the slab's widening is
  the closed meshes' test's to fail.

**Controls** (``CAMERA_PLANT``), each a change to the tree's code after the fact, in a process of its own (rates measured
with ``detection_rate.py`` on a runner, in the PR):

- ``no-barrier``: the optimisation barrier of the edge function's products removed, where XLA's CPU backend contracts a
  product into the difference as a fused multiply-add, which rounds the two triangles of an edge differently (they no
  longer compute exactly opposite values: a ray through the edge is claimed twice or not at all);
- ``no-pad``: the slab test's widening removed (``slab_pad`` of 1): a ray through an edge of a box is dropped by the
  rounding of the slab (caught by the closed meshes' test, which also fails on the head, where the widening is absent)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import HealthCheck, Phase, given, settings
from hypothesis import strategies as st
from in_loop_cases import PLANT
from mesh_cases import box, rotation, tube, uv_sphere

import mime.sensing._mesh as mesh
from mime.sensing import CameraPose, Pinhole, mesh_chords, ray_table
from mime.sensing._mesh import KEYS, traced, tree, triangles_of

if PLANT == "no-barrier":
    class _WithoutBarrier:
        """``jax.lax`` with the optimisation barrier an identity."""

        optimization_barrier = staticmethod(lambda x: x)

        def __getattr__(self, name: str):
            return getattr(jax.lax, name)

    mesh.lax = _WithoutBarrier()
if PLANT == "no-pad":
    mesh.slab_pad = lambda dtype: 1.0

RAYS = 512
#: The farthest from the edge, in the edge's length, a ray is aimed in the shared-edge case.
OFFSET = 1e-3
_traced = jax.jit(traced, static_argnames=("meshes",))
#: With a plant switched on a failing example is not shrunk (the control only needs the failure, and shrinking a planted
#: failure takes minutes a run); the property's own runs, and their 40 examples, are unchanged.
SETTINGS = settings(max_examples=40, deadline=None, print_blob=True,
                    phases=tuple(p for p in Phase if not PLANT or p is not Phase.shrink),
                    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])


def unit(rng: np.random.Generator, *, perpendicular_to: np.ndarray | None = None) -> np.ndarray:
    v = rng.normal(size=3)
    if perpendicular_to is not None:
        v = v - (v @ perpendicular_to) * perpendicular_to
    return v / np.linalg.norm(v)


def crossings_of(triangles: np.ndarray, origin: np.ndarray, directions: np.ndarray, dtype: str) -> np.ndarray:
    """Each ray's crossings of the one mesh the triangles make, traced in the run's dtype."""
    run = np.float32 if dtype == "float32" else np.float64
    built = tree(triangles, np.zeros(len(triangles), dtype=np.int64))
    found = _traced(tuple(jnp.asarray(built.arrays[key], dtype=run) for key in KEYS), meshes=1,
                    origin=jnp.asarray(origin, dtype=run), directions=jnp.asarray(directions, dtype=run),
                    live=jnp.ones(len(directions), dtype=bool))
    return np.asarray(found.count)[:, 0]


def roundings(rng: np.random.Generator, dtype: str, distance: float, n: int) -> np.ndarray:
    """``n`` lateral offsets (metres): half zero, half up to 32 roundings of the dtype at ``distance``, either sign."""
    u = float(np.finfo(np.float32 if dtype == "float32" else np.float64).eps) / 2.0 * distance
    return rng.uniform(0.0, 32.0, n) * u * (rng.random(n) < 0.5) * rng.choice([-1.0, 1.0], n)


def aimed(origin: np.ndarray, targets: np.ndarray) -> np.ndarray:
    toward = targets - origin
    return toward / np.linalg.norm(toward, axis=1, keepdims=True)


@SETTINGS
@given(size=st.floats(25e-6, 1e-3), seed=st.integers(0, 2 ** 32 - 1))
def test_an_edge_two_triangles_share_is_claimed_by_exactly_one_of_them(dtype, size, seed):
    rng = np.random.default_rng(seed)
    along = unit(rng)
    normal = unit(rng, perpendicular_to=along)
    across = np.cross(normal, along)
    middle = rng.uniform(-2e-3, 2e-3, 3)
    b, c = middle - 0.5 * size * along, middle + 0.5 * size * along
    (h1, h2), (s1, s2) = (rng.uniform(lo, hi, 2) * size for lo, hi in ((0.2, 1.0), (-0.5, 0.5)))
    z1, z2 = rng.uniform(-0.3, 0.3, 2) * (h1, h2)  # each triangle tilted 17 degrees at most from the surface's plane
    a = middle + h1 * across + s1 * along + z1 * normal
    d = middle - h2 * across + s2 * along + z2 * normal
    away = unit(rng, perpendicular_to=normal)
    angle = math.radians(rng.uniform(0.0, 50.0))
    origin = middle + rng.uniform(3 * size, max(5e-3, 3.5 * size)) * (math.cos(angle) * normal + math.sin(angle) * away)
    s = rng.uniform(0.1, 0.9, RAYS)
    targets = b * (1 - s[:, None]) + c * s[:, None]
    lateral = roundings(rng, dtype, float(np.linalg.norm(middle - origin)), RAYS)
    lateral[RAYS // 2:] = rng.uniform(-OFFSET, OFFSET, RAYS - RAYS // 2) * size
    counts = crossings_of(np.array([[a, b, c], [d, c, b]]), origin, aimed(origin, targets + lateral[:, None] * across),
                          dtype)
    assert np.all(counts == 1), f"{int(np.sum(counts != 1))} of {RAYS} rays claimed {sorted(set(counts.tolist()))} times"


@SETTINGS
@given(size=st.floats(25e-6, 1e-3), around=st.integers(4, 8), seed=st.integers(0, 2 ** 32 - 1))
def test_a_vertex_the_triangles_of_a_fan_share_is_claimed_by_exactly_one_of_them(dtype, size, around, seed):
    rng = np.random.default_rng(seed)
    normal = unit(rng)
    first = unit(rng, perpendicular_to=normal)
    second = np.cross(normal, first)
    angles = 2.0 * np.pi * (np.arange(around) + rng.uniform(0.0, 0.8, around)) / around
    rim = [rng.uniform(0.3, 1.0) * size * (math.cos(t) * first + math.sin(t) * second)
           + rng.uniform(-0.1, 0.1) * size * normal for t in angles]
    centre = rng.uniform(-2e-3, 2e-3, 3)
    triangles = np.array([[centre, centre + rim[i], centre + rim[(i + 1) % around]] for i in range(around)])
    away = unit(rng, perpendicular_to=normal)
    angle = math.radians(rng.uniform(0.0, 40.0))
    origin = centre + rng.uniform(3 * size, max(5e-3, 3.5 * size)) * (math.cos(angle) * normal + math.sin(angle) * away)
    lateral = roundings(rng, dtype, float(np.linalg.norm(centre - origin)), RAYS)[:, None] \
        * (rng.choice([-1.0, 1.0], (RAYS, 1)) * first + rng.choice([-1.0, 1.0], (RAYS, 1)) * second)
    counts = crossings_of(triangles, origin, aimed(origin, centre + lateral), dtype)
    assert np.all(counts == 1), f"{int(np.sum(counts != 1))} of {RAYS} rays claimed {sorted(set(counts.tolist()))} times"


def closed_mesh(rng: np.random.Generator, kind: str, size: float, aligned: bool) -> tuple[np.ndarray, np.ndarray]:
    """A closed mesh of ``kind``, turned at random, or only about z where ``aligned`` (a box then has edges along the axes
    its leaf boxes lie on: the case where a ray through an edge runs along the face of a box of the tree)."""
    if kind == "sphere":
        points, faces = uv_sphere(size, int(rng.choice([8, 16])))
    elif kind == "box":
        points, faces = box(tuple(size * rng.uniform(0.2, 1.0, 3)))  # type: ignore[arg-type]
    else:
        points, faces = tube(size * rng.uniform(0.5, 0.9), size, 2 * size * rng.uniform(0.5, 4.0), int(rng.choice([8, 16])))
    if aligned:
        angle = rng.choice([0.0, rng.uniform(0.0, 2.0 * np.pi)])
        q = (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))
    else:
        q = tuple(rng.normal(size=4) / 1.0)  # type: ignore[assignment]
        q = tuple(np.asarray(q) / np.linalg.norm(q))  # type: ignore[assignment]
    return points @ rotation(q).T + rng.uniform(-2e-3, 2e-3, 3), faces  # type: ignore[arg-type]


def even_at_edges(dtype: str, kind: str, size: float, seed: int, aligned: bool) -> None:
    rng = np.random.default_rng(seed)
    points, faces = closed_mesh(rng, kind, size, aligned)
    reach = float(np.max(np.linalg.norm(points - points.mean(axis=0), axis=1)))
    origin = points.mean(axis=0) + unit(rng) * rng.uniform(1.5 * reach, 1.5 * reach + 3e-3)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    n = 16 * RAYS
    pick = edges[rng.integers(0, len(edges), n)]
    s = rng.random((n, 1))
    s[: n // 10] = 0.0
    s[n // 10: n // 5] = 1.0
    targets = points[pick[:, 0]] * (1 - s) + points[pick[:, 1]] * s
    away = np.stack([unit(rng) for _ in range(n)])
    targets = targets + roundings(rng, dtype, float(np.linalg.norm(targets - origin, axis=1).mean()), n)[:, None] * away
    counts = crossings_of(triangles_of(points, faces), origin, aimed(origin, targets), dtype)
    assert np.all(counts % 2 == 0), f"{int(np.sum(counts % 2 == 1))} of {n} rays crossed the closed mesh an odd number of times"


@SETTINGS
@given(kind=st.sampled_from(["sphere", "box", "tube"]), size=st.floats(25e-6, 1e-3), aligned=st.booleans(),
       seed=st.integers(0, 2 ** 32 - 1))
def test_a_ray_at_an_edge_of_a_closed_mesh_from_outside_crosses_it_an_even_number_of_times(
        dtype, kind, size, aligned, seed):
    even_at_edges(dtype, kind, size, seed, aligned)


@SETTINGS
@given(size=st.floats(25e-6, 1e-3), seed=st.integers(0, 2 ** 32 - 1))
def test_a_ray_at_an_edge_of_a_box_from_outside_crosses_it_an_even_number_of_times(dtype, size, seed):
    """A box turned only about z, whose edges lie along the axes the tree's boxes are aligned to: where the slab test loses
    a crossing most often (a ray through such an edge runs along the face of a leaf box)."""
    even_at_edges(dtype, "box", size, seed, True)


@SETTINGS
@given(focal=st.floats(300.0, 1500.0), front=st.floats(0.5e-3, 4e-3), depth=st.floats(50e-6, 1e-3),
       left=st.integers(4, 20), right=st.integers(44, 60), top=st.integers(4, 14), bottom=st.integers(34, 44),
       seed=st.integers(0, 2 ** 32 - 1))
def test_an_axis_aligned_camera_sees_an_axis_aligned_box_crossed_an_even_number_of_times(
        dtype, focal, front, depth, left, right, top, bottom, seed):
    rng = np.random.default_rng(seed)
    detector = Pinhole(64, 48, focal, focal, 31.5, 23.5)
    pose = CameraPose(rng.uniform(-2e-3, 2e-3, 3), np.array([1.0, 0.0, 0.0, 0.0]))
    corners = ray_table(detector, pose, u=jnp.asarray([float(left), float(right)]), v=jnp.asarray([float(top), float(bottom)]))
    direction = np.stack([np.asarray(each, dtype=np.float64) for each in corners.direction], axis=-1)
    origin = np.array([float(each) for each in corners.origin])
    low_corner, high_corner = (origin + d * (front / d[2]) for d in direction)
    low = np.array([low_corner[0], low_corner[1], origin[2] + front])
    high = np.array([high_corner[0], high_corner[1], origin[2] + front + depth])
    points, faces = box((1.0, 1.0, 1.0))
    points = np.where(points > 0, high, low)
    found = mesh_chords({"box": (points, faces)}, detector, pose, samples=3)
    crossings = np.asarray(found.crossings)[0]
    assert crossings.max() == 2, "the box is in view"
    assert np.all(crossings % 2 == 0), f"{int(np.sum(crossings % 2 == 1))} sub-rays crossed the closed box an odd number of times"
