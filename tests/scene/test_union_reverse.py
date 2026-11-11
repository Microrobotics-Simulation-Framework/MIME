"""Reverse mode against forward mode through the union's distance on a batch of points, as the FVM's wall evaluates it
(every face of a node's box round the filleted loop, the three face sets together, the candidates chosen in the
program where the check fails): the two modes are one derivative, so their directional derivatives in the radii agree
to rounding. It is the guard of reverse mode against forward mode, not of the tie rule: that is
``test_union_tie.py``'s.

They did not, by 4 to 50 %, while the union took its least with ``jnp.min`` and ``jnp.minimum``: their derivatives find
the least by comparing it again with its operands, and XLA computed an operand twice, fused into two programs that
rounded it differently, so on a batch of thousands of points with exact ties (two tubes' shared end ball) reverse mode
found another operand equal to the least than forward mode did (a single point's derivative agreed). The union selects
its least by one comparison (``_lesser``, ``_least``).

**Control.** ``MIME_PLANT_UNION=reverse-scaled`` makes the selection's reverse derivative 1 % larger than its forward
one, which the comparison must catch on every run. ``jnp.min`` and ``jnp.minimum`` put back (``MIME_PLANT_UNION=
reduction``) are not this gate's to catch: they are wrong in both modes at once (their derivative averages tied
operands), which a comparison of the two modes cannot see; ``test_union_tie.py`` catches them on every run."""

import dataclasses
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
    node = placed.node
    grid = MACGrid(placed.layout.cells, placed.layout.spacing)
    held = node._step_constants(None)  # noqa: SLF001 - the step's own constants
    solid = node.tree_solid(held, grid)
    points = tuple(grid.face_points(k, "float64") for k in range(3))
    base = dict(solid.parameters)
    direction = np.zeros_like(np.asarray(base["radii"]))
    direction[[2, 3, 4]] = 1.0
    weights = jnp.asarray(np.random.default_rng(seed).normal(size=3 * int(np.prod(grid.cells))))

    def read(radii):
        distance, _ = dataclasses.replace(solid, parameters={**base, "radii": radii}).made(points)
        return jnp.sum(weights * distance)

    radii = jnp.asarray(base["radii"])
    reverse = float(jnp.sum(jax.jit(jax.grad(read))(radii) * direction))
    forward = float(jax.jit(lambda r: jax.jvp(read, (r,), (jnp.asarray(direction),))[1])(radii))
    print(f"\nseed {seed}: {weights.shape[0]} faces, up to {solid.planned} chosen; reverse {reverse:.12e}, forward "
          f"{forward:.12e}")
    assert abs(reverse - forward) <= MODES_LINE * abs(forward)
