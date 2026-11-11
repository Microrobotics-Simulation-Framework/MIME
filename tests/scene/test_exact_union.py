"""The seam-aware union (``mime.scene.ExactUnion``) and the tube tree's lumen: F2a's PR 2a gates, their predictions in
``predictions_composed_walls.json`` and ``predictions_composed_walls_fix_round.json`` (committed before they ran).

1. **Exact against a brute force, tilted and offset**: the Y's lumen (equal radii, as F1a's spike, and radii of 0.5,
   0.4 and 0.3 D) and the split-and-merge loop's, turned at random and on a lattice moved by a random sub-cell offset,
   and a hairpin aligned with the lattice (where two walls' normals are opposed), within the band, under the band's
   contract line of 0.05 cells in the fluid and the solid, in float64 and float32; float32 within 1e-3 cells of
   float64, at the world's origin and 0.1 to 0.3 m from it; a corner of the filleted loop found from just off its
   plane. **Control** (``MIME_PLANT_UNION=minimum``): the plain minimum, 0.4 to 1.75 cells off in the fluid, fails on
   every draw; each fix of the review's round has its own (``planted``).
2. **The fillet is the closing, in closed form**: a wedge of two half-spaces (``mime.core.HalfSpace``) and two
   overlapping balls, filleted, on
   both sides, to rounding; the fillet share's derivative against a central difference. **Control**
   (``MIME_PLANT_FILLET=dropped``): the sharp union against the filleted closed form.
3. **Continuous through the junctions**: on dense lines through the Y's carina and the loop's two, filleted and sharp,
   no step exceeds the spacing (an exact distance is 1-Lipschitz). **Control** (``MIME_PLANT_UNION=no-carina-seam``):
   the carina's seam left out of the candidates.
4. **Implicit derivatives**: at points whose nearest point is a seam, the distance's derivative in the point, the
   fillet share, a daughter's radius and the junction's position against central differences. **Control**
   (``MIME_PLANT_IMPLICIT=held``): the seam's point held, its implicit step left out.

Each control's rate is measured over 200 draws on Actions (``.github/probe/detection_rate.py``, a draw a seed through
``DETECTION_SEED``); CI runs the fixed seeds. Gate 1 is ``test_union_brute_force.py``'s and gate 3
``test_union_continuity.py``'s, which import their draws and plants from here: CI deals an unlisted file to a shard by
its path, and the four gates in one file were over a shard's half hour."""

import dataclasses
import math
import os
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
import pytest
from union_oracles import (
    D,
    REACH,
    aligned_rotation,
    hairpin_tree,
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
from mime.core import HalfSpace, jnp
from mime.scene import Ball, ExactUnion, TubeTree

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


def drawn(kind: str, seed: int, cells: int, far: bool = False
          ) -> tuple[TubeTree, np.ndarray, float, np.ndarray, float]:
    """A Y (``spike``, ``asymmetric``) or the split-and-merge ``loop``, turned at random about the world's origin and
    moved by a sub-cell offset of a lattice of ``D / cells``; the lattice's points within the gates' reach of its
    junctions (the middle of the loop's two), that middle and that reach. ``hairpin``: two legs overlapping by 0.05 to
    0.15 D, turned by one of the rotations that keep the lattice's axes and moved by a sub-cell offset along the plane
    of their axes, so lattice points lie on it (where the two walls' normals are opposed); its points about the legs'
    middle. ``far``: moved besides by 0.1 to 0.3 m along each axis, as a scanner's coordinates."""
    h = D / cells
    rng = np.random.default_rng(seed + 7919)
    if kind == "hairpin":
        turn = aligned_rotation(seed)
        tree = hairpin_tree(-rng.uniform(0.05, 0.15) * D).turned(turn, turn @ (rng.uniform(0.0, h, 3) * (1, 0, 1)))
        points = tree.parameters()["points"]
        middle = 0.25 * np.sum(points, axis=0)
        reach = REACH * D
    else:
        tree = (loop_tree() if kind == "loop" else y_tree(kind)).turned(rotation(seed), rng.uniform(0.0, h, 3))
        junctions = [tree.parameters()["points"][tree.segments[j.ends[0][0]][j.ends[0][1]]]
                     for j in tree.junction_names() if len(j.ends) >= 3]
        middle = np.mean(junctions, axis=0)
        reach = REACH * D + max(float(np.linalg.norm(j - middle)) for j in junctions)
    if far:
        away = rng.choice((-1.0, 1.0), 3) * rng.uniform(0.1, 0.3, 3)
        away = np.round(away / h) * h
        tree, middle = tree.turned(np.eye(3), away), middle + away
    return tree, lattice(middle, reach, h), h, middle, reach


def planted(monkeypatch) -> None:
    """The fix round's plants (``MIME_PLANT_UNION``), each a fix reverted: ``point-start`` (a seam's projection from
    the point alone, as before the starts across the wall), ``corner-starts`` (a corner's Gauss-Newton from the point
    and its walls' nearest points alone, without the starts across the wall), ``world-allowance`` (the rounding
    allowances scaled with the distance from the world's origin), ``march-<n>`` (the march along a seam cut to ``n``
    steps each way; ``march-0``, none)."""
    if UNION_PLANT == "point-start":
        monkeypatch.setattr(ExactUnion, "_seam_starts", lambda self, parameters, points, i, size: points[None])
    elif UNION_PLANT == "corner-starts":
        held = ExactUnion._corner_starts
        monkeypatch.setattr(ExactUnion, "_corner_starts", lambda self, *args: held(self, *args)[:4])
    elif UNION_PLANT == "world-allowance":
        def allowance(self, parameters, points, ulps):
            world = points + jnp.asarray(self.origin(parameters), dtype=points.dtype)
            size = self._size(parameters, points.dtype)
            return ulps * jnp.finfo(points.dtype).eps * (jnp.sqrt(jnp.sum(world * world, axis=-1)) + size)

        monkeypatch.setattr(ExactUnion, "_allowance", allowance)
    elif UNION_PLANT is not None and UNION_PLANT.startswith("march-"):
        monkeypatch.setattr(union_module, "MARCH_STEPS", int(UNION_PLANT.removeprefix("march-")))


def composed(tree: TubeTree, points: np.ndarray, dtype: str) -> tuple[np.ndarray, object]:
    lumen = tree.lumen()
    held = parameters_of(tree, dtype)
    plan = lumen.plan(held, jnp.asarray(points, dtype=dtype), margin=0.0)
    found = lumen.signed_distance(held, jnp.asarray(points, dtype=dtype), plan)
    if UNION_PLANT == "minimum":
        return np.asarray(lumen.minimum(held, jnp.asarray(points, dtype=dtype)), dtype=np.float64), found
    return np.asarray(found.distance, dtype=np.float64), found


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


@dataclass(frozen=True)
class OwnValues:
    """A ``mime.core`` shape in a union, its values (``reads``, by its parameter names) read from the union's one
    mapping: two half-spaces, each with its own normal and offset."""

    shape: Any
    reads: tuple[tuple[str, Any], ...]

    def own(self, parameters):
        return {name: read(parameters) for name, read in self.reads}

    def signed_distance(self, parameters, points):
        return self.shape.signed_distance(self.own(parameters), points)

    def outward_normal(self, parameters, points):
        return self.shape.outward_normal(self.own(parameters), points)

    def curvature(self, parameters, points):
        return self.shape.curvature(self.own(parameters), points)


def wedge_union(normals: tuple[np.ndarray, np.ndarray], origin: np.ndarray, fillet_of) -> ExactUnion:
    """The union of the half-spaces ``n_i . (x - origin) <= 0``: each ``HalfSpace`` takes the side its normal points
    into, so its normal is ``-n_i`` and its offset ``-n_i . origin``."""
    regions = tuple(OwnValues(HalfSpace(), (("normal", lambda p, n=n: -n),
                                            ("offset", lambda p, n=n: -float(n @ origin)))) for n in normals)
    return ExactUnion(regions, ((0, 1),), size=lambda p: D, fillet=fillet_of)


def balls_union(centres, radii, fillet_of) -> ExactUnion:
    regions = tuple(Ball(centre=lambda p, c=c: c, radius=lambda p, r=r: r) for c, r in zip(centres, radii, strict=True))
    return ExactUnion(regions, ((0, 1),), size=lambda p: D, fillet=fillet_of)


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
    tree = dataclasses.replace(tree, fillet=0.1)
    points = points[np.abs(plain_minimum(tree, points)) < 2.0 * h]
    lumen = tree.lumen()
    held = parameters_of(tree, "float64")
    found = lumen.signed_distance(held, jnp.asarray(points))
    chosen = jnp.asarray(points[np.asarray(found.failing)][:40])

    def at(share):
        return lumen.signed_distance({**held, "fillet": share}, chosen).distance

    exact = np.asarray(jax.jvp(at, (held["fillet"],), (jnp.asarray(1.0),))[1])
    step = 1e-5
    difference = (8.0 * (at(held["fillet"] + step) - at(held["fillet"] - step))
                  - (at(held["fillet"] + 2 * step) - at(held["fillet"] - 2 * step))) / (12.0 * step)
    scale = np.max(np.abs(np.asarray(difference)))
    assert scale > 0.0
    assert np.max(np.abs(exact - np.asarray(difference))) <= DERIVATIVE_LINE * scale


# -- gate 4 ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", seed_list((41,)))
def test_the_seams_derivatives_are_the_implicit_ones(seed, monkeypatch):
    if IMPLICIT_PLANT == "held":
        monkeypatch.setattr(union_module, "_implicit", lambda reached, jacobian, residual: reached)
    tree, points, h = drawn_y("asymmetric", seed, 8)
    tree = dataclasses.replace(tree, fillet=0.1)
    points = points[np.abs(plain_minimum(tree, points)) < 2.0 * h]
    lumen = tree.lumen()
    held = parameters_of(tree, "float64")
    at_points = jnp.asarray(points)
    local = lumen._local(held, at_points)
    distances, normals, radius = lumen._widened(held, local)
    walls = np.asarray(lumen._walls(held, local, distances, normals, radius))
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
            return lumen.signed_distance(parameters, where).distance

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
