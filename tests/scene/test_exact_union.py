"""The seam-aware union (``mime.scene.ExactUnion``) and the tube tree's lumen: F2a's PR 2a gates, their predictions in
``predictions_composed_walls.json`` (committed before they ran).

1. **Exact against a brute force, tilted and offset**: the Y's lumen (equal radii, as F1a's spike, and radii of 0.5,
   0.4 and 0.3 D) and the split-and-merge loop's, turned at random and on a lattice moved by a random sub-cell offset,
   within the band, under the band's contract line of 0.05 cells in the fluid and the solid, in float64 and float32;
   float32 within 1e-3 cells of float64. **Control** (``MIME_PLANT_UNION=minimum``): the plain minimum, 0.4 to 1.75
   cells off in the fluid, fails on every draw.
2. **The fillet is the closing, in closed form**: a wedge of two half-spaces and two overlapping balls, filleted, on
   both sides, to rounding; the fillet share's derivative against a central difference. **Control**
   (``MIME_PLANT_FILLET=dropped``): the sharp union against the filleted closed form.
3. **Continuous through the junctions**: on dense lines through the Y's carina and the loop's two, filleted and sharp,
   no step exceeds the spacing (an exact distance is 1-Lipschitz). **Control** (``MIME_PLANT_UNION=no-carina-seam``):
   the carina's seam left out of the candidates.
4. **Implicit derivatives**: at points whose nearest point is a seam, the distance's derivative in the point, the
   fillet share, a daughter's radius and the junction's position against central differences. **Control**
   (``MIME_PLANT_IMPLICIT=held``): the seam's point held, its implicit step left out.

Each control's rate is measured over 200 draws on Actions (``.github/probe/detection_rate.py``, a draw a seed through
``DETECTION_SEED``); CI runs the fixed seeds."""

import math
import os

import jax
import numpy as np
import pytest
from union_oracles import (
    D,
    REACH,
    brute_distance,
    lattice,
    loop_tree,
    plain_minimum,
    rotation,
    seed_list,
    two_balls_closed_form,
    wedge_closed_form,
    y_tree,
)

import mime.scene._union as union_module
from mime.core import jnp
from mime.scene import Ball, ExactUnion, PlaneSide, TubeTree

#: The band's contract line: the signed distance exact within the band to this many cells.
CONTRACT_LINE = 0.05
#: The brute force's spacing, in cells: its floor next to the wall about 0.022 cells.
BRUTE_SPACING = 1.0 / 32.0
#: float32's composed distance against float64's, in cells.
DTYPE_LINE = 1e-3
#: The fillet's closed forms in float64 (m, on a millimetre geometry) and float32 (cells of h = D/16).
CLOSED_LINE = {"float64": 1e-12, "float32": 1e-3 * D / 16.0}
#: The derivatives' agreement, relative to their scale.
DERIVATIVE_LINE = 1e-6
UNION_PLANT = os.environ.get("MIME_PLANT_UNION")
FILLET_PLANT = os.environ.get("MIME_PLANT_FILLET")
IMPLICIT_PLANT = os.environ.get("MIME_PLANT_IMPLICIT")


def parameters_of(tree: TubeTree, dtype: str) -> dict[str, jnp.ndarray]:
    return {name: jnp.asarray(value, dtype=dtype) for name, value in tree.parameters().items()}


def drawn_y(kind: str, seed: int, cells: int) -> tuple[TubeTree, np.ndarray, float]:
    """The Y turned at random about its junction and moved by a sub-cell offset of a lattice of ``D / cells``; the
    lattice's points within the gates' reach of the junction."""
    tree, points, h, _, _ = drawn(kind, seed, cells)
    return tree, points, h


def drawn(kind: str, seed: int, cells: int) -> tuple[TubeTree, np.ndarray, float, np.ndarray, float]:
    """A Y (``spike``, ``asymmetric``) or the split-and-merge ``loop``, turned at random about the world's origin and
    moved by a sub-cell offset of a lattice of ``D / cells``; the lattice's points within the gates' reach of its
    junctions (the middle of the loop's two), that middle and that reach."""
    h = D / cells
    rng = np.random.default_rng(seed + 7919)
    tree = (loop_tree() if kind == "loop" else y_tree(kind)).turned(rotation(seed), rng.uniform(0.0, h, 3))
    junctions = [np.asarray(tree.points[int(j.name[1:])]) for j in tree.junctions if len(j.ends) >= 3]
    middle = np.mean(junctions, axis=0)
    reach = REACH * D + max(float(np.linalg.norm(j - middle)) for j in junctions)
    return tree, lattice(middle, reach, h), h, middle, reach


def composed(tree: TubeTree, points: np.ndarray, dtype: str) -> tuple[np.ndarray, object]:
    lumen = tree.lumen()
    held = parameters_of(tree, dtype)
    plan = lumen.plan(held, jnp.asarray(points, dtype=dtype), margin=0.0)
    found = lumen.signed_distance(held, jnp.asarray(points, dtype=dtype), plan)
    if UNION_PLANT == "minimum":
        return np.asarray(lumen.minimum(held, jnp.asarray(points, dtype=dtype)), dtype=np.float64), found
    return np.asarray(found.distance, dtype=np.float64), found


# -- gate 1 ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("kind", ["spike", "asymmetric", "loop"])
@pytest.mark.parametrize("seed", seed_list((11,)))
def test_the_union_is_exact_against_a_brute_force(kind, seed, dtype):
    tree, points, h, middle, reach = drawn(kind, seed, 8)
    plain = plain_minimum(tree, points)
    band = np.abs(plain) < 3.0 * h
    points, plain = points[band], plain[band]
    exact = brute_distance(tree, points, BRUTE_SPACING * h, middle, reach + 0.5 * D)
    judged = (np.abs(plain) < 2.0 * h) | (np.abs(exact) < 2.0 * h)
    distance, found = composed(tree, points, dtype)
    error = np.abs(distance - exact)[judged] / h
    fluid = exact[judged] < 0.0
    print(f"\n{kind} seed {seed} {dtype}: {int(judged.sum())} points, {int(np.sum(found.failing))} failing the check; "
          f"largest error {error[fluid].max():.4f} cells in the fluid, {error[~fluid].max():.4f} in the solid; "
          f"the plain minimum {np.abs(plain - exact)[judged][fluid].max() / h:.3f}")
    assert int(np.sum(np.asarray(found.failing)[judged])) > 0, "no point failed the check: the draw cannot discriminate"
    assert int(found.unrefined) == 0 and int(found.unconverged) == 0
    assert error.max() <= CONTRACT_LINE, f"largest error {error.max():.4f} cells over the band's line {CONTRACT_LINE}"


@pytest.mark.parametrize("seed", seed_list((13,)))
def test_the_union_is_exact_at_sixteen_cells_a_diameter(seed):
    tree, points, h = drawn_y("asymmetric", seed, 16)
    plain = plain_minimum(tree, points)
    band = np.abs(plain) < 3.0 * h
    points, plain = points[band], plain[band]
    exact = brute_distance(tree, points, BRUTE_SPACING * h, np.asarray(tree.points[1]), (REACH + 0.25) * D)
    judged = (np.abs(plain) < 2.0 * h) | (np.abs(exact) < 2.0 * h)
    distance, found = composed(tree, points, "float64")
    error = np.abs(distance - exact)[judged] / h
    assert int(found.unrefined) == 0 and int(found.unconverged) == 0
    assert error.max() <= CONTRACT_LINE


@pytest.mark.parametrize("seed", seed_list((14,)))
def test_float32_agrees_with_float64(seed):
    tree, points, h = drawn_y("asymmetric", seed, 8)
    points = points[np.abs(plain_minimum(tree, points)) < 2.0 * h]
    wide = composed(tree, points, "float64")[0]
    narrow = composed(tree, points, "float32")[0]
    assert np.max(np.abs(narrow - wide)) / h <= DTYPE_LINE


# -- gate 2 ---------------------------------------------------------------------------------------------------------
def drawn_wedge(seed: int) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray, float, np.ndarray]:
    """Two half-spaces whose solid wedge opens 20 to 120 degrees, at a random orientation and place, a fillet of 20
    to 100 um, and points within three fillet radii of the widened seam (both sides)."""
    rng = np.random.default_rng(seed)
    turn = rotation(seed + 101)
    opening = math.radians(rng.uniform(20.0, 120.0))
    between = math.pi - opening
    n1 = turn @ np.asarray([1.0, 0.0, 0.0])
    n2 = turn @ np.asarray([math.cos(between), math.sin(between), 0.0])
    origin = rng.uniform(-D, D, 3)
    fillet = rng.uniform(20e-6, 100e-6)
    bisector = (n1 + n2) / np.linalg.norm(n1 + n2)
    corner = origin + bisector * fillet / math.sin(opening / 2.0)
    edge = turn @ np.asarray([0.0, 0.0, 1.0])
    offsets = rng.normal(size=(400, 3))
    offsets *= (rng.uniform(0.0, 3.0 * fillet, 400) / np.linalg.norm(offsets, axis=1))[:, None]
    points = corner + offsets + rng.uniform(-D, D, 400)[:, None] * edge
    return (n1, n2), origin, fillet, points


def drawn_balls(seed: int) -> tuple[tuple[np.ndarray, np.ndarray], tuple[float, float], float, np.ndarray]:
    """Two balls of 0.3 to 0.5 mm overlapping so that the solid's ridge opens 20 to 120 degrees, a fillet of 20 to
    100 um, and points within three fillet radii of the ridge's circle."""
    rng = np.random.default_rng(seed)
    r1, r2 = rng.uniform(0.3e-3, 0.5e-3, 2)
    opening = math.radians(rng.uniform(20.0, 120.0))
    apart = math.sqrt(r1 * r1 + r2 * r2 - 2.0 * r1 * r2 * math.cos(math.pi - opening))
    axis = rotation(seed + 202) @ np.asarray([0.0, 0.0, 1.0])
    c1 = rng.uniform(-D, D, 3)
    c2 = c1 + apart * axis
    fillet = rng.uniform(20e-6, 100e-6)
    along = (r1 * r1 - r2 * r2 + apart * apart) / (2.0 * apart)
    ring = math.sqrt(r1 * r1 - along * along)
    across = np.cross(axis, np.eye(3)[int(np.argmin(np.abs(axis)))])
    across /= np.linalg.norm(across)
    other = np.cross(axis, across)
    phi = rng.uniform(0.0, 2.0 * math.pi, 400)
    on_ring = c1 + along * axis + ring * (np.cos(phi)[:, None] * across + np.sin(phi)[:, None] * other)
    offsets = rng.normal(size=(400, 3))
    offsets *= (rng.uniform(0.0, 3.0 * fillet, 400) / np.linalg.norm(offsets, axis=1))[:, None]
    return (c1, c2), (float(r1), float(r2)), float(fillet), on_ring + offsets


def wedge_union(normals: tuple[np.ndarray, np.ndarray], origin: np.ndarray, fillet_of) -> ExactUnion:
    regions = tuple(PlaneSide(normal_of=lambda p, n=n: n, offset=lambda p, n=n: float(n @ origin)) for n in normals)
    return ExactUnion(regions, ((0, 1),), size=D, fillet=fillet_of)


def balls_union(centres, radii, fillet_of) -> ExactUnion:
    regions = tuple(Ball(centre=lambda p, c=c: c, radius=lambda p, r=r: r) for c, r in zip(centres, radii, strict=True))
    return ExactUnion(regions, ((0, 1),), size=D, fillet=fillet_of)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("shape", ["wedge", "balls"])
@pytest.mark.parametrize("seed", seed_list((21, 22, 23)))
def test_the_fillet_is_the_closing_in_closed_form(shape, seed, dtype):
    if shape == "wedge":
        normals, origin, fillet, points = drawn_wedge(seed)
        union = wedge_union(normals, origin, lambda p: p["fillet"])
        expected = wedge_closed_form(normals, origin, fillet, points)
    else:
        centres, radii, fillet, points = drawn_balls(seed)
        union = balls_union(centres, radii, lambda p: p["fillet"])
        expected = two_balls_closed_form(centres, radii, fillet, points)
    used = 0.0 if FILLET_PLANT == "dropped" else fillet
    found = union.signed_distance({"fillet": jnp.asarray(used, dtype=dtype)}, jnp.asarray(points, dtype=dtype))
    distance = np.asarray(found.distance, dtype=np.float64)
    error = np.abs(distance - expected)
    print(f"\n{shape} seed {seed} {dtype}: fillet {fillet * 1e6:.1f} um, {int(np.sum(found.failing))} of {len(points)} "
          f"failing, {int(np.sum(expected > 0))} in the solid; largest error {error.max():.3g} m")
    assert int(np.sum(np.asarray(found.failing))) > 0 and np.any(expected > 0.0) and np.any(expected < 0.0)
    assert int(found.unconverged) == 0
    assert error.max() <= CLOSED_LINE[dtype]


@pytest.mark.parametrize("seed", seed_list((24,)))
def test_the_fillet_shares_derivative_is_its_difference(seed):
    tree, points, h = drawn_y("asymmetric", seed, 8)
    tree = TubeTree(tree.points, tree.segments, tree.radii, fillet=0.1)
    points = points[np.abs(plain_minimum(tree, points)) < 2.0 * h]
    lumen = tree.lumen()
    held = parameters_of(tree, "float64")
    found = lumen.signed_distance(held, jnp.asarray(points))
    chosen = jnp.asarray(points[np.asarray(found.failing)][:40])

    def at(share):
        return lumen.distance({**held, "fillet": share}, chosen)

    exact = np.asarray(jax.jvp(at, (held["fillet"],), (jnp.asarray(1.0),))[1])
    step = 1e-5
    difference = (8.0 * (at(held["fillet"] + step) - at(held["fillet"] - step))
                  - (at(held["fillet"] + 2 * step) - at(held["fillet"] - 2 * step))) / (12.0 * step)
    scale = np.max(np.abs(np.asarray(difference)))
    assert scale > 0.0
    assert np.max(np.abs(exact - np.asarray(difference))) <= DERIVATIVE_LINE * scale


# -- gate 3 ---------------------------------------------------------------------------------------------------------
def carina(tree: TubeTree, at: int, first: int, second: int) -> np.ndarray:
    """The point of the solid's edge between two tubes leaving the junction at point ``at``: on their bisector, as far
    from the junction as their walls meet."""
    points = np.asarray(tree.points)
    start = points[at]
    u, v = (points[sum(tree.segments[k]) - at] - start for k in (first, second))
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    half = math.acos(float(np.clip(u @ v, -1.0, 1.0))) / 2.0
    radius = max(tree.radii[first][0], tree.radii[second][0])
    return start + (u + v) / np.linalg.norm(u + v) * radius / math.sin(half)


def with_carina_seam_left_out(monkeypatch, pairs):
    """The plant: the seams of ``pairs`` give no candidate, as a Newton that never converges there."""
    held = ExactUnion._seam

    def seam(self, parameters, points, i, j, radius):
        if (i, j) in pairs:
            return (jnp.full(points.shape[:-1], jnp.inf, dtype=points.dtype), jnp.zeros(points.shape[:-1], dtype=bool),
                    points)
        return held(self, parameters, points, i, j, radius)

    monkeypatch.setattr(ExactUnion, "_seam", seam)


@pytest.mark.parametrize("fillet", [0.0, 0.1])
@pytest.mark.parametrize("shape", ["y", "loop"])
@pytest.mark.parametrize("seed", seed_list((31,)))
def test_the_distance_is_continuous_through_the_junctions(shape, fillet, seed, monkeypatch):
    if shape == "y":
        tree = y_tree("asymmetric", fillet)
        tips, carina_pairs = [carina(tree, 1, 1, 2)], {(1, 2)}
    else:
        tree = loop_tree(fillet)
        tips, carina_pairs = [carina(tree, 1, 1, 2), carina(tree, 6, 5, 6)], {(1, 2), (5, 6)}
    if UNION_PLANT == "no-carina-seam":
        with_carina_seam_left_out(monkeypatch, carina_pairs)
    rng = np.random.default_rng(seed)
    turn = rotation(seed + 303)
    offset = rng.uniform(0.0, D / 16.0, 3)
    tree = tree.turned(turn, offset)
    tips = [turn @ tip + offset for tip in tips]
    lumen = tree.lumen()
    held = parameters_of(tree, "float64")
    spacing = D / 1024.0
    lines = []
    for tip in tips:
        near = lattice(tip, 0.4 * D, D / 64.0)
        failing = near[lumen.plan(held, jnp.asarray(near), margin=0.0)]
        assert len(failing) >= 6, "too few points fail the check near the carina to draw lines through"
        for through in failing[rng.choice(len(failing), size=6, replace=False)]:
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            lines.append(through + np.arange(-256, 257)[:, None] * spacing * direction)
    points = jnp.asarray(np.concatenate(lines))
    found = lumen.signed_distance(held, points, lumen.plan(held, points, margin=0.0))
    assert int(found.unrefined) == 0
    values = np.asarray(found.distance).reshape(len(lines), -1)
    worst = float(np.max(np.abs(np.diff(values, axis=1)))) / spacing
    print(f"\n{shape} fillet {fillet}: the largest step over the spacing {worst:.9f}")
    assert worst <= 1.0 + 1e-6


# -- gate 4 ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", seed_list((41,)))
def test_the_seams_derivatives_are_the_implicit_ones(seed, monkeypatch):
    if IMPLICIT_PLANT == "held":
        monkeypatch.setattr(union_module, "_implicit", lambda reached, jacobian, residual: reached)
    tree, points, h = drawn_y("asymmetric", seed, 8)
    tree = TubeTree(tree.points, tree.segments, tree.radii, fillet=0.1)
    points = points[np.abs(plain_minimum(tree, points)) < 2.0 * h]
    lumen = tree.lumen()
    held = parameters_of(tree, "float64")
    at_points = jnp.asarray(points)
    distances, normals, radius = lumen._widened(held, at_points)
    walls = np.asarray(lumen._walls(held, at_points, distances, normals, radius))
    found = lumen.signed_distance(held, at_points)
    depth = np.asarray(radius - found.distance)
    seamed = np.nonzero(np.asarray(found.failing) & (depth < walls - 1e-3 * h))[0]
    assert seamed.size >= 4, "too few points whose nearest point is a seam to discriminate"
    rng = np.random.default_rng(seed)
    chosen = jnp.asarray(points[rng.choice(seamed, size=min(24, seamed.size), replace=False)])
    direction = rng.normal(size=chosen.shape)
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    moves = {
        "the point": (lambda t: (held, chosen + t * jnp.asarray(direction)), 1e-9),
        "the fillet share": (lambda t: ({**held, "fillet": held["fillet"] + t}, chosen), 1e-6),
        "a daughter's radius": (lambda t: ({**held, "radii": held["radii"].at[1].add(t)}, chosen), 1e-9),
        "the junction": (lambda t: ({**held, "points": held["points"].at[1].add(t * jnp.asarray(direction[0]))},
                                    chosen), 1e-9),
    }
    judged = 0
    for name, (moved, step) in moves.items():
        def at(t, moved=moved):
            parameters, where = moved(t)
            return lumen.distance(parameters, where)

        exact = np.asarray(jax.jvp(at, (jnp.asarray(0.0),), (jnp.asarray(1.0),))[1])
        values = {k: np.asarray(at(jnp.asarray(k * step))) for k in (-2, -1, 1, 2)}
        centre = np.asarray(at(jnp.asarray(0.0)))
        difference = (8.0 * (values[1] - values[-1]) - (values[2] - values[-2])) / (12.0 * step)
        forward, backward = (values[1] - centre) / step, (centre - values[-1]) / step
        scale = max(1.0, float(np.max(np.abs(difference))))
        smooth = np.abs(forward - backward) <= 1e-3 * scale
        judged += int(smooth.sum())
        apart = np.abs(exact - difference)[smooth]
        print(f"\n{name}: {int(smooth.sum())} of {len(smooth)} smooth, largest |AD - FD| / scale "
              f"{(apart.max() if apart.size else 0.0) / scale:.3g}")
        assert apart.size == 0 or apart.max() <= DERIVATIVE_LINE * scale, name
    assert judged >= 40, "too few smooth draws to discriminate"
