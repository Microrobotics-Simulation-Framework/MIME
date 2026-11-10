"""Gate 3 of the seam-aware union (``mime.scene.ExactUnion``): continuous through the junctions (an exact distance is
1-Lipschitz on dense lines through points failing the check). The draws and plants are ``test_exact_union.py``'s (the
gates' module docstring); this file holds gate 3 apart from it so that CI's shards, which deal unlisted files by their
path, carry the union's gates on three shards rather than one."""

import math
import os

import numpy as np
import pytest
from test_exact_union import parameters_of, planted
from union_oracles import D, lattice, loop_tree, rotation, seed_list, y_tree

from mime.core import jnp
from mime.scene import ExactUnion, TubeTree
from mime.scene._union import Seam

UNION_PLANT = os.environ.get("MIME_PLANT_UNION")


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
            return Seam(jnp.full(points.shape[:-1], jnp.inf, dtype=points.dtype),
                        jnp.zeros(points.shape[:-1], dtype=bool), jnp.ones(points.shape[:-1], dtype=bool))
        return held(self, parameters, points, i, j, radius)

    monkeypatch.setattr(ExactUnion, "_seam", seam)


@pytest.mark.parametrize("fillet", [0.0, 0.1])
@pytest.mark.parametrize("shape", ["y", "loop"])
@pytest.mark.parametrize("seed", seed_list((31,)))
def test_the_distance_is_continuous_through_the_junctions(shape, fillet, seed, monkeypatch):
    planted(monkeypatch)
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
