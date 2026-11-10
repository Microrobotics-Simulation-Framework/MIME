"""Reverse mode against forward mode through the union's distance on a batch of points, as the FVM's wall evaluates it
(every lattice point near the filleted loop's wall, its candidates planned where the check fails): the two modes are one
derivative, so their directional derivatives in the radii agree to rounding.

They did not, by 4 to 50 %, while the union took its least with ``jnp.min`` and ``jnp.minimum``: their derivatives find
the least by comparing it again with its operands, and XLA computes an operand twice, fused into two programs that
round it differently, so reverse mode found no operand equal to the least, or another, on a batch of thousands of points
(a single point's derivative agreed). The union selects its least by one comparison (``_lesser``, ``_least``).
**Control** (``MIME_PLANT_UNION=reduction``): the least taken by ``jnp.min`` and ``jnp.minimum`` again."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from union_oracles import D, lattice, loop_tree, rotation, seed_list

import mime.scene._union as union_module

#: Reverse against forward, relative to forward's size and the rounding of a sum over the batch.
MODES_LINE = 1e-9


def _planted(monkeypatch) -> None:
    if os.environ.get("MIME_PLANT_UNION") != "reduction":
        return

    def least(stacked):
        return jnp.min(stacked, axis=0), jnp.argmin(stacked, axis=0)

    monkeypatch.setattr(union_module, "_lesser", jnp.minimum)
    monkeypatch.setattr(union_module, "_least", least)


@pytest.mark.parametrize("seed", seed_list((41,)))
def test_reverse_mode_is_forward_mode_on_a_batch_of_points(seed, monkeypatch):
    _planted(monkeypatch)
    rng = np.random.default_rng(seed)
    h = D / 8.0
    tree = loop_tree(0.1).turned(rotation(seed), rng.uniform(0.0, h, 3))
    union, base = tree.lumen(), tree.parameters()
    points = lattice(base["points"].mean(axis=0), 3.0 * D, h)
    near = np.abs(np.asarray(union.minimum(base, jnp.asarray(points)))) < 1.5 * h
    points = jnp.asarray(points[near])
    plan = union.plan(base, points, 0.25 * h)
    direction = np.zeros_like(base["radii"])
    direction[1:3] = 1.0
    weights = jnp.asarray(rng.normal(size=points.shape[0]))

    def read(radii):
        return jnp.sum(weights * union.evaluate({**base, "radii": radii}, points, plan).distance)

    radii = jnp.asarray(base["radii"])
    reverse = float(jnp.sum(jax.jit(jax.grad(read))(radii) * direction))
    forward = float(jax.jit(lambda r: jax.jvp(read, (r,), (jnp.asarray(direction),))[1])(radii))
    scale = float(jnp.sum(jnp.abs(weights))) * float(jnp.finfo(jnp.float64).eps) * 1e3
    print(f"\nseed {seed}: {points.shape[0]} points, {len(plan)} planned; reverse {reverse:.12e}, forward "
          f"{forward:.12e}")
    assert abs(reverse - forward) <= MODES_LINE * abs(forward) + scale
