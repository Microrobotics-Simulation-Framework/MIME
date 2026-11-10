"""Reverse mode against forward mode through the union's distance on a batch of points, as the FVM's wall evaluates it
(the faces a node holds near the filleted loop's wall, the three face sets together, the candidates planned where the
check fails): the two modes are one derivative, so their directional derivatives in the radii agree to rounding.

They did not, by 4 to 50 %, while the union took its least with ``jnp.min`` and ``jnp.minimum``: their derivatives find
the least by comparing it again with its operands, and XLA computed an operand twice, fused into two programs that
rounded it differently, so on a batch of thousands of points with exact ties (two tubes' shared end ball) reverse mode
found another operand equal to the least than forward mode did (a single point's derivative agreed). The union selects
its least by one comparison (``_lesser``, ``_least``).

**Controls.** ``MIME_PLANT_UNION=reduction`` puts ``jnp.min`` and ``jnp.minimum`` back: whether XLA duplicates the
operand depends on its fusion, so this control's rate is measured, not assumed. ``MIME_PLANT_UNION=reverse-scaled`` makes
the selection's reverse derivative 1 % larger than its forward one, which the comparison must catch on every run."""

import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from union_oracles import seed_list

import mime.scene._union as union_module

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "flow"))
from wall_cases import draw, loop, tree_node  # noqa: E402

from mime.nodes.flow.fvm import MACGrid  # noqa: E402

#: Reverse against forward, relative.
MODES_LINE = 1e-9


def _planted(monkeypatch) -> None:
    plant = os.environ.get("MIME_PLANT_UNION")
    if plant == "reduction":
        monkeypatch.setattr(union_module, "_lesser", jnp.minimum)
        monkeypatch.setattr(union_module, "_least",
                            lambda stacked: (jnp.min(stacked, axis=0), jnp.argmin(stacked, axis=0)))
    elif plant == "reverse-scaled":
        @jax.custom_vjp
        def lesser(a, b):
            return jnp.where(b < a, b, a)

        def ahead(a, b):
            return lesser(a, b), b < a

        def back(chosen, g):
            g = 1.01 * g
            return jnp.where(chosen, 0.0, g), jnp.where(chosen, g, 0.0)

        lesser.defvjp(ahead, back)
        monkeypatch.setattr(union_module, "_lesser", lesser)


@pytest.mark.parametrize("seed", seed_list((1101,)))
def test_reverse_mode_is_forward_mode_on_a_batch_of_points(seed, monkeypatch):
    _planted(monkeypatch)
    angle, offset = draw(seed)
    placed = tree_node(loop(), 8, angle=angle, offset=offset)
    grid = MACGrid(placed.layout.cells, placed.layout.spacing)
    near, plans, _ = placed.node.held_faces(grid)
    tree = placed.node.tree()
    union, base = tree.lumen(), tree.parameters()
    flat = [np.asarray(grid.face_points(k, "float64")).reshape(-1, 3)[near[k]] for k in range(3)]
    sizes = np.cumsum([0] + [len(n) for n in near])
    points = jnp.asarray(np.concatenate(flat))
    plan = np.concatenate([plans[k] + sizes[k] for k in range(3)])
    direction = np.zeros_like(base["radii"])
    direction[[2, 3, 4]] = 1.0
    weights = jnp.asarray(np.random.default_rng(seed).normal(size=points.shape[0]))

    def read(radii):
        return jnp.sum(weights * union.evaluate({**base, "radii": radii}, points, plan).distance)

    radii = jnp.asarray(base["radii"])
    reverse = float(jnp.sum(jax.jit(jax.grad(read))(radii) * direction))
    forward = float(jax.jit(lambda r: jax.jvp(read, (r,), (jnp.asarray(direction),))[1])(radii))
    print(f"\nseed {seed}: {points.shape[0]} faces, {len(plan)} planned; reverse {reverse:.12e}, forward {forward:.12e}")
    assert abs(reverse - forward) <= MODES_LINE * abs(forward)
