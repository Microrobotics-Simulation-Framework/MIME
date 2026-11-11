"""The union's least at an exact tie, the guard of FA8's first fix whatever XLA's fusion does: two capsules of a drawn
radius sharing a point (a V at a drawn opening, turned at random), read at points beyond their shared end ball, where
both tubes' distances are that ball's, exactly. The derivative of each point's distance in the first tube's radius at
the shared point is one operand's, ``-1`` (the first's) or ``0`` (the second's), never their average, and reverse mode
equals forward mode.

``jnp.min`` and ``jnp.minimum`` average tied operands in their derivative, in both modes at once, so a comparison of
reverse mode with forward mode cannot see them put back (``test_union_reverse.py``, at its own seed, read 26.10 both
ways against a central difference's 48.57); this reads the tie rule itself. **Control:** ``MIME_PLANT_UNION=reduction``
puts them back, and every tied point reads ``-0.5``."""

import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from union_oracles import seed_list

import mime.scene._union as union_module
from mime.scene import TubeTree

#: The points read beyond the shared ball, before those not beyond both tubes' ends are dropped.
POINTS = 32
#: How near an operand's derivative a tied point's must be: float64's rounding of a unit derivative, with room.
TIE_LINE = 1e-9


@pytest.mark.parametrize("seed", seed_list((1,)))
def test_at_an_exact_tie_the_derivative_is_one_operands(seed, monkeypatch):
    if os.environ.get("MIME_PLANT_UNION") == "reduction":
        monkeypatch.setattr(union_module, "_lesser", jnp.minimum)
        monkeypatch.setattr(union_module, "_least",
                            lambda stacked: (jnp.min(stacked, axis=0), jnp.argmin(stacked, axis=0)))
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        rng = np.random.default_rng(seed)
        radius, length = rng.uniform(1e-4, 6e-4), rng.uniform(2e-3, 4e-3)
        opening = math.radians(rng.uniform(20.0, 160.0))
        first = np.array([math.sin(opening / 2), 0.0, math.cos(opening / 2)])
        second = np.array([-math.sin(opening / 2), 0.0, math.cos(opening / 2)])
        turn, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        apex = rng.uniform(-1e-3, 1e-3, 3)
        points = np.stack([apex + length * first @ turn.T, apex, apex + length * second @ turn.T])
        tree = TubeTree.si(points, ((0, 1), (1, 2)), ((radius, radius), (radius, radius)), 0.0)
        union, parameters = tree.lumen(), tree.parameters()
        away = -(first + second) / np.linalg.norm(first + second)
        directions = away[None] + 0.3 * rng.normal(size=(POINTS, 3))
        directions = directions[(directions @ first < -0.05) & (directions @ second < -0.05)]
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        at = apex + (radius * rng.uniform(1.2, 3.0, len(directions)))[:, None] * (directions @ turn.T)
        assert len(at) >= 8, "too few points beyond both tubes' ends: the draw cannot discriminate"

        def read(radii):
            return union.evaluate({**parameters, "radii": radii}, jnp.asarray(at), None).distance

        radii = jnp.asarray(parameters["radii"])
        forward = np.asarray(jax.jacfwd(read)(radii))[:, 0, 1]
        reverse = np.asarray(jax.jacrev(read)(radii))[:, 0, 1]
        print(f"\nseed {seed}: {len(at)} tied points; forward {np.unique(np.round(forward, 9))}, reverse "
              f"{np.unique(np.round(reverse, 9))}")
        one_operands = np.minimum(np.abs(forward + 1.0), np.abs(forward)) <= TIE_LINE
        assert np.all(one_operands), forward[~one_operands]
        assert np.allclose(forward, reverse, rtol=0.0, atol=TIE_LINE)
    finally:
        jax.config.update("jax_enable_x64", held)
