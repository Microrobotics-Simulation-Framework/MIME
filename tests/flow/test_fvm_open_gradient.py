"""FB15, float64: gradients with ports (F2b PR 4b; predictions in ``predictions_rk3_ports.json``). The open tube's
openings' pressures and outlet flux after six steps from a cold prediction (the openings' relative pressures start at
zero, so the first step's moves are the whole pressures), differentiated in the Windkessel's R_p, C and R_d, the
inflow's amplitude and the viscosity, reverse and forward, against a central difference that rebuilds the node, at loss
scales 1e-12 and 1e6 (the scale test), on the open node of each stage kind (IMEX Euler's projected CG with its unit
fields solved in the program; RK3's fast solves). Its control: ``stop_gradient`` on IMEX Euler's unit fields (and so on
their response ``q``), which cuts the viscosity's derivative through the moves; the laws' leaves do not reach the unit
fields, so the cut cannot be seen in them (stated, not a pass).

The gate found, on its first run, IMEX Euler's open node's reverse mode off (the viscosity's derivative 3.13 against the
difference's 5.98, the inflow's amplitude off too; forward mode right): the projected CG's operator is self-adjoint
under the open grid's half-weighted inner product, not the plain one reverse mode transposes in, so its transpose solve
is the weighted one (``mime.nodes.flow.fvm._solve.projected_solve``)."""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from open_cases import open_node, tube_resistance
from rk3_open_cases import rk3_node

from mime.nodes.flow.fvm import OpenFlowNode

STEPS = 6
#: The gate's line: the derivative against the central difference, relative (the periodic gate's: the difference's
#: truncation at a relative step of 1e-4 is about 1e-8).
LINE = 1e-6
SCALES = [1e-12, 1e6]
#: The leaves, each by its place: the laws' parameters' row and column, or the viscosity.
LEAVES = {"proximal": (1, 0), "compliance": (1, 1), "distal": (1, 2), "amplitude": (0, 1), "viscosity": None}
NAMES = tuple(LEAVES)


def _parameters(dt: float) -> list[list[float]]:
    flux = 10.0 * math.pi / 4.0
    resistance = tube_resistance(2.0)
    distal = 3.0 * resistance
    return [[flux, 0.3 * flux, 1.0 / (20.0 * dt), 0.0], [0.5 * resistance, dt / distal, distal, 0.0]]


def _node(scheme: str):
    if scheme == "rk3":
        dt = 0.1 / 64.0
        return rk3_node(8, parameters=_parameters(dt), dt=dt, convection=False)
    dt = 0.02
    return open_node(8, parameters=_parameters(dt), dt=dt, convection=False, tolerance=1e-12)


def _loss(node, p):
    state = dict(node.initial_state())
    for _ in range(STEPS):
        state = node.advance(p, state, {}, node.delta_t)
    # The inlet's pressure carries the vessel's resistance (the viscosity) and the laws'; the outlet's flux, pinned by
    # the inflow, carries neither (a loss of it alone is invariant in the viscosity: measured while writing).
    return (state["opening_pressure"][0] + state["opening_pressure"][1]) / tube_resistance(2.0) + state["opening_flux"][1]


def _values(p):
    """The leaves' values at ``p``, in :data:`NAMES`' order."""
    parameters = jnp.asarray(p["opening_parameters"])
    return jnp.stack([jnp.asarray(p["viscosity"]) if LEAVES[n] is None else parameters[LEAVES[n]] for n in NAMES])


def _with(p, values):
    """The step's constants with every leaf of :data:`NAMES` replaced by ``values``."""
    parameters = jnp.asarray(p["opening_parameters"])
    for k, name in enumerate(NAMES):
        if LEAVES[name] is not None:
            parameters = parameters.at[LEAVES[name]].set(values[k])
    return {**p, "viscosity": values[NAMES.index("viscosity")], "opening_parameters": parameters}


def _rebuilt(scheme: str, name: str, factor: float) -> float:
    """The loss of a node built afresh from the base node's arguments with ``name`` times ``factor``."""
    base = _node(scheme)
    arguments = dict(base.constants())
    if LEAVES[name] is None:
        arguments["viscosity"] = float(arguments["viscosity"]) * factor
    else:
        parameters = np.asarray(arguments["opening_parameters"], dtype=np.float64).copy()
        parameters[LEAVES[name]] *= factor
        arguments["opening_parameters"] = parameters
    node = type(base)("flow", base.delta_t, **arguments)
    return float(jax.jit(lambda p: _loss(node, p))(node._step_constants(None)))  # noqa: SLF001


@functools.cache
def central(scheme: str, step: float = 1e-4) -> np.ndarray:
    """The central difference in each leaf, each side a node built afresh, per unit of the leaf."""
    values = np.asarray(_values(_node(scheme)._step_constants(None)))  # noqa: SLF001
    return np.asarray([(_rebuilt(scheme, name, 1.0 + step) - _rebuilt(scheme, name, 1.0 - step)) / (2.0 * step * value)
                       for name, value in zip(NAMES, values, strict=True)])


def reverse(scheme: str, scale: float) -> np.ndarray:
    node = _node(scheme)
    p = node._step_constants(None)  # noqa: SLF001
    gradient = jax.jit(jax.grad(lambda v, s: s * _loss(node, _with(p, v))))(_values(p), jnp.asarray(scale))
    return np.asarray(gradient) / scale


def forward(scheme: str) -> np.ndarray:
    node = _node(scheme)
    p = node._step_constants(None)  # noqa: SLF001
    return np.asarray(jax.jit(jax.jacfwd(lambda v: _loss(node, _with(p, v))))(_values(p)))


def _off(found: np.ndarray, expected: np.ndarray) -> dict[str, tuple[float, float]]:
    """Each leaf whose derivative is off the line, with its value and the difference's."""
    return {name: (float(f), float(e)) for name, f, e in zip(NAMES, found, expected, strict=True)
            if not abs(f - e) <= LINE * abs(e)}


@pytest.mark.parametrize("scheme", ["imex_euler", "rk3"])
def test_the_gradient_with_ports_is_the_rebuilt_central_difference(float64, scheme):
    """Each leaf's derivative, reverse at both loss scales and forward, within the line of the rebuilt difference."""
    expected = central(scheme)
    for scale in SCALES:
        assert not _off(reverse(scheme, scale), expected), scale
    assert not _off(forward(scheme), expected)


def test_control_stop_gradient_on_the_unit_fields_fails_the_viscosity_from_a_cold_prediction(float64, monkeypatch):
    """The unit fields' derivative cut (``stop_gradient`` on each, so on their response ``q``): from a cold prediction
    the first step's moves are the whole pressures, and the viscosity's derivative through them is off the line."""
    expected = central("imex_euler")

    def unit_field(self, solve, force, start):
        del self
        unit, report = solve(force, start)
        return tuple(jax.lax.stop_gradient(c) for c in unit), report

    monkeypatch.setattr(OpenFlowNode, "unit_field", unit_field)
    assert "viscosity" in _off(reverse("imex_euler", 1.0), expected)
