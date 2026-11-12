"""Gate 7's structure: one uniform dilation per patch (decided 4 (5)). A patch between two walls (a plane below and
one above), dilated by ``l`` whole fine cells, holds exactly the walls of the same two planes each moved ``l`` cells
into its solid: one offset of the one signed distance, both contacts opened alike. The accuracy of that rule at a
hand-over gap of at most 0.1 of the local radius is the window spike's (6.4 % worst over 28 two-wall pairs); the
patch adds nothing to which contact is opened. Control ``PATCH_PLANT=one_wall``: the dilation applied to the lower
wall only (a per-contact offset)."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mime.nodes.flow.fvm._patch import RefinementPatch, placed


class Plane:
    region = "wall"

    def __init__(self, at, up):
        self.at, self.up = at, up

    def signed_distance(self, points):
        return (self.at - points[..., 2]) if self.up else (points[..., 2] - self.at)


@pytest.fixture(autouse=True)
def float64():
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", held)


@pytest.mark.parametrize("cells", [1, 2, 3])
def test_gate7_one_dilation_moves_both_walls_by_the_same_whole_cells(cells):
    h = 4e-5
    patch = RefinementPatch((14, 14, 16), h, read="point")
    lower, upper = 3.3 * h, 12.7 * h
    points = jnp.asarray([[7.1 * h, 6.9 * h, 8.0 * h]])
    origin = jnp.zeros(3, jnp.int32)
    dilated = placed(patch, [Plane(lower, True), Plane(upper, False)], origin, cells, points, 1e-12, 1e-3)
    moved_lower = lower - cells * h
    moved_upper = upper + (0.0 if os.environ.get("PATCH_PLANT") == "one_wall" else cells * h)
    if os.environ.get("PATCH_PLANT") == "one_wall":
        dilated = placed(patch, [Plane(lower, True)], origin, cells, points, 1e-12, 1e-3)
        dilated = dilated._replace(resistivity=tuple(
            a + b for a, b in zip(dilated.resistivity, placed(patch, [Plane(upper, False)], origin, 0, points, 1e-12,
                                                                     1e-3).resistivity)))
    moved = placed(patch, [Plane(moved_lower, True), Plane(upper + cells * h, False)], origin, 0, points, 1e-12, 1e-3)
    for a, b in zip(dilated.resistivity, moved.resistivity):
        scale = float(jnp.max(jnp.abs(b)))
        assert float(jnp.max(jnp.abs(a - b))) <= 1e-12 * scale
    assert int(dilated.shell_needed) == int(moved.shell_needed)
