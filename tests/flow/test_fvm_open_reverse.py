"""The open vessel's reverse-mode derivative under IMEX Euler, through the node's own step: its projected CG's operator
is self-adjoint under the open grid's half-weighted inner product, not the plain one reverse mode transposes in, so the
transpose solve must be the weighted one, ``V (P A P)^-1 V^-1`` (``mime.nodes.flow.fvm._solve.projected_solve``).
Forward mode solves with the operator itself, right in either product, so the reverse derivative is held to it: the
plain transpose put the viscosity's derivative of this loss 0.52 of forward mode's after six steps (3.13 against 5.98)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from open_cases import open_node, tube_resistance

STEPS = 2
#: Reverse against forward, relative: both solve to the CG's 1e-12, so they agree far inside it.
LINE = 1e-6
#: The leaves, each by its place: the laws' parameters' row and column, or the viscosity.
LEAVES = {"proximal": (1, 0), "compliance": (1, 1), "distal": (1, 2), "amplitude": (0, 1), "viscosity": None}
NAMES = tuple(LEAVES)


def _node():
    dt = 0.02
    flux = 10.0 * math.pi / 4.0
    resistance = tube_resistance(2.0)
    distal = 3.0 * resistance
    parameters = [[flux, 0.3 * flux, 1.0 / (20.0 * dt), 0.0], [0.5 * resistance, dt / distal, distal, 0.0]]
    return open_node(8, parameters=parameters, dt=dt, convection=False, tolerance=1e-12)


def _loss(node, p):
    state, _ = jax.lax.scan(lambda s, _: (node.advance(p, s, {}, node.delta_t), None), dict(node.initial_state()),
                            None, length=STEPS)
    # The inlet's pressure carries the vessel's resistance (the viscosity) and the laws'; the outlet's flux is pinned by
    # the inflow's.
    return (state["opening_pressure"][0] + state["opening_pressure"][1]) / tube_resistance(2.0) + state["opening_flux"][1]


def _values(p):
    parameters = jnp.asarray(p["opening_parameters"])
    return jnp.stack([jnp.asarray(p["viscosity"]) if LEAVES[n] is None else parameters[LEAVES[n]] for n in NAMES])


def _with(p, values):
    parameters = jnp.asarray(p["opening_parameters"])
    for k, name in enumerate(NAMES):
        if LEAVES[name] is not None:
            parameters = parameters.at[LEAVES[name]].set(values[k])
    return {**p, "viscosity": values[NAMES.index("viscosity")], "opening_parameters": parameters}


def test_the_open_vessels_reverse_derivative_is_its_forward_one(float64):
    """Each leaf's reverse-mode derivative within the line of forward mode's."""
    node = _node()
    p = node._step_constants(None)  # noqa: SLF001

    def loss(values):
        return _loss(node, _with(p, values))

    values = _values(p)
    reverse = np.asarray(jax.jit(jax.grad(loss))(values))
    forward = np.asarray(jax.jit(jax.jacfwd(loss))(values))
    off = {name: (float(r), float(f)) for name, r, f in zip(NAMES, reverse, forward, strict=True)
           if not abs(r - f) <= LINE * abs(f)}
    assert not off, off
