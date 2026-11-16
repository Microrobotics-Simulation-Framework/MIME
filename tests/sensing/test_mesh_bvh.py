"""G22, the tree against brute force: every crossing the tree finds of each ray (each mesh's signed chord and crossings,
the nearest crossing's ``t`` and its triangle) is the brute-force oracle's, every triangle tested in float64 with no
tree (``mesh_cases.brute``), on drawn scenes of closed meshes at the scales a vessel and a robot have.

**The draws** (physical scales; CLAUDE.md, "Property tests"): one to three closed meshes (a UV sphere, a box, a tube
wall), each 50 um to 2 mm across, centred within 2 mm of the origin, turned at random; a ray origin within 5 mm (inside
a mesh or outside); 192 rays (three tiles), two in three aimed at a point inside a random triangle, the rest at random.
In the dtype of the run: the tree is traced in it, the oracle in float64.

**What is judged, and where.** A ray passing within rounding of a triangle's edge may be claimed by either triangle, so
rays whose barycentric margin (``Crossings.margin``) is below ``MARGIN`` in the run's dtype are left out, and so are the
rays that ``tied``: an origin on a triangle (a camera on a wall; a draw puts a mesh's vertex at the origin) crosses it at a
``t`` of zero, whose sign is below the rounding of the crossing test, bounded in ``mesh_cases.t_rounding``. (The renderer
itself counts a crossing at ``t`` of about 2e-12 in float32 when a ray starts on a surface: a degenerate pose, the camera on
the wall, which the derived band leaves out of the judged set; it is not fixed in ``src/``.) So are the rays that pass an
edge of a triangle in front of them within the rounding of its barycentric coordinates (``mesh_cases.b_rounding``): that
rounding is of the size of the triangle's distance from the origin, not of the triangle, so a far, thin, grazing fold may
lose or keep both its crossings together, and the fixed ``MARGIN`` is only the floor. The rest must
have the oracle's crossings exactly, its chord within rounding on the scale it came from (each crossing's ``t``, its
inputs' size over the cosine of incidence, summed: not the chord, since the signed sum cancels, and not ``t`` alone,
since a grazing crossing's ``t`` is ill-conditioned), its nearest ``t`` within the same rounding and its triangle where
the nearest is apart from the next.

**Controls** (``CAMERA_PLANT``), each a change to the tree after it is built, aimed at the triangle the judged rays
cross most:

- ``dropped-leaf``: that triangle's leaf emptied (spike 2's control): its crossings are lost;
- ``flipped-winding``: that triangle's winding reversed, which moves the signed chord by twice its ``t`` on each ray
  through it;
- ``shrunk-box``: that triangle's leaf box shrunk to a tenth of its extent about its centre, far beyond the rounding of
  the slab test (the ``slab_pad``): the crossings its box no longer holds are lost.

Both must fail on every draw: a draw always holds judged rays aimed at triangles."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import HealthCheck, example, given, settings
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
    if PLANT == "shrunk-box":
        node = arrays["mesh_v0"].shape[0] + leaf
        centre = 0.5 * (arrays["mesh_lo"][node] + arrays["mesh_hi"][node])
        half = 0.5 * (arrays["mesh_hi"][node] - arrays["mesh_lo"][node])
        held["mesh_lo"][node], held["mesh_hi"][node] = centre - 0.1 * half, centre + 0.1 * half
    elif PLANT == "dropped-leaf":
        for key in ("mesh_v0", "mesh_v1", "mesh_v2"):
            held[key][leaf] = 0.0
    elif PLANT == "flipped-winding":
        held["mesh_v1"][leaf, slot], held["mesh_v2"][leaf, slot] = arrays["mesh_v2"][leaf, slot], \
            arrays["mesh_v1"][leaf, slot]
    return held


def on_the_wall() -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, int]:
    """The draw that failed in CI: a tube, its outer wall's vertex column through the ray origin, so the origin
    is on the wall, on the diagonal of two coplanar triangles, at a ``t`` of zero to rounding."""
    size = 6.788434854395295e-4
    points, faces = tube(0.5 * size, size, 2 * size, 8)
    return [(placed(points, np.array([size, 0.0, 0.0]), rotation((1.0, 0.0, 0.0, 0.0))), faces)], np.zeros(3), 0


def far_and_grazing() -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, int]:
    """The second draw that failed in CI: a thin 30 um tube 3 mm from the origin (a hundred times its own length), among two
    larger meshes, its folded triangles crossed at a grazing angle within 3e-4 of their shared edge: the float32 test, whose
    corner coordinates carry the error of the distance and not of the triangle, lost both crossings of the fold."""
    def turned(axis: tuple[float, float, float], angle: float) -> np.ndarray:
        unit = np.array(axis) + np.array([0.0, 0.0, 1e-3])
        unit = unit / np.linalg.norm(unit)
        return rotation((math.cos(angle / 2), *(math.sin(angle / 2) * unit)))

    small, faces_small = tube(3.017198714815784e-05 * 0.6461487436425816, 3.017198714815784e-05,
                              2 * 3.017198714815784e-05 * 0.99999, 32)
    large, faces_large = tube(0.5512149444461878e-3 * 0.5, 0.5512149444461878e-3, 2 * 0.5512149444461878e-3, 32)
    cube, faces_cube = box((0.10054979417291318e-3,) * 3)
    zero = np.zeros(3)
    meshes = [
        (placed(small, np.array([-0.0012107175345038683, 0.000758537218262845, 0.0019243552369448405]),
                turned((0.0, -0.9370537164963859, -0.8760815421384325), 1.7886821923264888)), faces_small),
        (placed(large, zero, turned((0.0, 0.0, 0.0), 0.0)), faces_large),
        (placed(cube, zero, turned((0.0, 0.0, 0.0), 0.0)), faces_cube)]
    return meshes, np.array([0.0, -0.00307711041268762, 0.0]), 29168


@example(scene=far_and_grazing())
@example(scene=on_the_wall())
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
    eps = float(np.finfo(run).eps)
    oracle = brute(held_triangles, owners, len(meshes), held_origin, held_directions, eps=eps, edge=MARGIN[dtype])
    judged = (oracle.margin >= MARGIN[dtype]) & ~oracle.tied
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
