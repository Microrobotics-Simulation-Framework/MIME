"""G22, the tree against brute force: every crossing the tree finds of each ray (each mesh's signed chord and crossings,
the nearest crossing's ``t`` and its triangle) is the brute-force oracle's, every triangle tested in float64 with no
tree (``mesh_cases.brute``), on drawn scenes of closed meshes at the scales a vessel and a robot have.

**The draws** (physical scales; CLAUDE.md, "Property tests"): one to three closed meshes (a UV sphere, a box, a tube
wall), each 50 um to 2 mm across, centred within 2 mm of the origin, turned at random; a ray origin within 5 mm (inside
a mesh or outside); 192 rays (three tiles), two in three aimed at a point inside a random triangle, the rest at random.
In the dtype of the run: the tree is traced in it, the oracle in float64.

**What is judged, and where.** A ray passing within rounding of a triangle's edge may be claimed by either triangle, so
rays whose barycentric margin (``Crossings.margin``) is below ``MARGIN`` in the run's dtype are left out; the rest must
have the oracle's crossings exactly, its chord within rounding on the scale it came from (each crossing's ``t``, its
inputs' size over the cosine of incidence, summed: not the chord, since the signed sum cancels, and not ``t`` alone,
since a grazing crossing's ``t`` is ill-conditioned), its nearest ``t`` within the same rounding and its triangle where
the nearest is apart from the next.

**Controls** (``CAMERA_PLANT``), each a change to the tree after it is built, aimed at the triangle the judged rays
cross most:

- ``dropped-leaf``: that triangle's leaf emptied (spike 2's control): its crossings are lost;
- ``flipped-winding``: that triangle's winding reversed, which moves the signed chord by twice its ``t`` on each ray
  through it.

Both must fail on every draw: a draw always holds judged rays aimed at triangles."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from in_loop_cases import PLANT
from mesh_cases import box, brute, placed, rotation, tube, uv_sphere

from mime.sensing._mesh import KEYS, LEAF, closed_problem, traced, tree, triangles_of

RAYS = 192
#: The barycentric margin below which a ray may be claimed by either triangle of an edge, by the run's dtype.
MARGIN = {"float32": 1e-4, "float64": 1e-10}

_traced = jax.jit(traced, static_argnames=("meshes",))


@st.composite
def scenes(draw: st.DrawFn) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, int]:
    """Closed meshes in the world, a ray origin, and a seed for the rays."""
    meshes = []
    for _ in range(draw(st.integers(1, 3))):
        kind = draw(st.sampled_from(["sphere", "box", "tube"]))
        size = draw(st.floats(25e-6, 1e-3))
        if kind == "sphere":
            points, faces = uv_sphere(size, draw(st.sampled_from([8, 16, 24])))
        elif kind == "box":
            points, faces = box(tuple(size * draw(st.floats(0.2, 1.0)) for _ in range(3)))  # type: ignore[arg-type]
        else:
            points, faces = tube(size * draw(st.floats(0.5, 0.9)), size, 2 * size * draw(st.floats(0.5, 4.0)),
                                 draw(st.sampled_from([8, 16, 32])))
        axis = np.array([draw(st.floats(-1.0, 1.0)) for _ in range(3)]) + np.array([0.0, 0.0, 1e-3])
        angle = draw(st.floats(0.0, math.pi))
        unit = axis / np.linalg.norm(axis)
        q = (math.cos(angle / 2), *(math.sin(angle / 2) * unit))
        centre = np.array([draw(st.floats(-2e-3, 2e-3)) for _ in range(3)])
        meshes.append((placed(points, centre, rotation(q)), faces))  # type: ignore[arg-type]
    origin = np.array([draw(st.floats(-5e-3, 5e-3)) for _ in range(3)])
    return meshes, origin, draw(st.integers(0, 2 ** 32 - 1))


def rays(triangles: np.ndarray, origin: np.ndarray, seed: int) -> np.ndarray:
    """Two rays in three aimed at a point inside a random triangle (barycentric weights at least 0.05), the rest at
    random."""
    rng = np.random.default_rng(seed)
    aimed = (2 * RAYS) // 3
    weights = rng.dirichlet([2.0, 2.0, 2.0], size=aimed)
    weights = 0.05 + 0.85 * weights
    weights /= weights.sum(axis=1, keepdims=True)
    targets = np.einsum("nk,nkj->nj", weights, triangles[rng.integers(0, len(triangles), size=aimed)])
    toward = targets - origin
    free = rng.normal(size=(RAYS - aimed, 3))
    directions = np.concatenate([toward, free])
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def planted(arrays: dict[str, np.ndarray], slots: np.ndarray, triangle: int) -> dict[str, np.ndarray]:
    """The tree with the control's change at ``triangle``'s slot."""
    leaf, slot = (int(i[0]) for i in np.nonzero(slots == triangle))
    held = {key: value.copy() for key, value in arrays.items()}
    if PLANT == "dropped-leaf":
        for key in ("mesh_v0", "mesh_v1", "mesh_v2"):
            held[key][leaf] = 0.0
    elif PLANT == "flipped-winding":
        held["mesh_v1"][leaf, slot], held["mesh_v2"][leaf, slot] = arrays["mesh_v2"][leaf, slot], \
            arrays["mesh_v1"][leaf, slot]
    return held


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture,
                                                                 HealthCheck.too_slow], print_blob=True)
@given(scene=scenes())
def test_the_tree_finds_every_crossing_brute_force_finds(dtype, scene):
    meshes, origin, seed = scene
    triangles = np.concatenate([triangles_of(points, faces) for points, faces in meshes])
    owners = np.concatenate([np.full(len(faces), i) for i, (_, faces) in enumerate(meshes)])
    for points, faces in meshes:
        assert closed_problem(points, faces) is None
    directions = rays(triangles, origin, seed)
    run = np.float32 if dtype == "float32" else np.float64
    # the oracle on the numbers the tree holds: the corners and the rays rounded to the run's dtype
    held_triangles = triangles.astype(run).astype(np.float64)
    held_origin = origin.astype(run).astype(np.float64)
    held_directions = directions.astype(run).astype(np.float64)
    oracle = brute(held_triangles, owners, len(meshes), held_origin, held_directions)
    judged = oracle.margin >= MARGIN[dtype]
    built = tree(triangles, owners)
    arrays = dict(built.arrays)
    if PLANT:
        hit = np.zeros(len(triangles), dtype=np.int64)
        for r in np.flatnonzero(judged):
            for k, _ in oracle.hits[r]:
                hit[k] += 1
        assert hit.max() > 0, "no judged ray crosses a triangle: the control has nothing to catch"
        arrays = planted(arrays, built.slots, int(np.argmax(hit)))
    found = _traced(tuple(jnp.asarray(arrays[key], dtype=run) for key in KEYS), meshes=len(meshes),
                    origin=jnp.asarray(origin, dtype=run), directions=jnp.asarray(directions, dtype=run),
                    live=jnp.ones(RAYS, dtype=bool))
    count, chord = np.asarray(found.count), np.asarray(found.chord, dtype=np.float64)
    nearest, slot = np.asarray(found.nearest, dtype=np.float64), np.asarray(found.slot)
    assert np.array_equal(count[judged], oracle.count[judged]), \
        f"crossings differ on {int(np.sum(np.any(count[judged] != oracle.count[judged], axis=1)))} judged rays"
    eps = float(np.finfo(run).eps)
    allowed = 64.0 * eps * oracle.reach
    error = np.abs(chord - oracle.chord).max(axis=1)
    assert np.all(error[judged] <= allowed[judged]), \
        f"a chord off by {float(np.max((error - allowed)[judged])):.3g} m beyond rounding"
    some = judged & np.isfinite(oracle.nearest)
    assert np.all(np.abs(nearest[some] - oracle.nearest[some]) <= 64.0 * eps * oracle.nearest_reach[some])
    apart = some.copy()
    for r in np.flatnonzero(some):
        ts = sorted(t for _, t in oracle.hits[r])
        apart[r] = len(ts) == 1 or ts[1] - ts[0] > 1e3 * eps * ts[0]
    triangle = np.where(slot >= 0, built.slots.reshape(-1)[np.maximum(slot, 0)], -1)
    assert np.array_equal(triangle[apart], oracle.triangle[apart])
    assert np.all(slot[~np.isfinite(oracle.nearest) & judged] == -1)


def test_the_judged_rays_are_most_of_the_draw(float64):
    """The margin leaves out few rays: the gate judges what it draws, not an empty set."""
    points, faces = tube(60e-6, 70e-6, 480e-6, 32)
    triangles = triangles_of(points, faces)
    origin = np.array([0.0, -1e-3, 1e-5])
    oracle = brute(triangles, np.zeros(len(faces), dtype=np.int64), 1, origin, rays(triangles, origin, 7))
    assert np.mean(oracle.margin >= MARGIN["float32"]) > 0.9
    assert LEAF == 4
