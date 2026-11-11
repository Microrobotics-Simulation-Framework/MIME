"""FB15, float64: gradients with ports (F2b PR 4b; predictions in ``predictions_rk3_ports.json``). The open tube's
openings' pressures and outlet flux after six steps from a cold prediction (the openings' relative pressures start at zero, so the
first step's moves are the whole pressures), differentiated in the Windkessel's R_p, C and R_d, the inflow's amplitude
and the viscosity, reverse and forward, against a central difference that rebuilds the node, at loss scales 1e-12 and
1e6 (the scale test), on the open node of each stage kind (IMEX Euler's projected CG with its unit fields solved in the
program; RK3's fast solves). Its control: ``stop_gradient`` on IMEX Euler's unit fields (and so on their response
``q``), which cuts the viscosity's derivative through the moves; the laws' leaves do not reach the unit fields, so the
cut cannot be seen in them (stated, not a pass)."""

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
#: The leaves, each by its place: the laws' parameters' rows and columns, or the viscosity.
LEAVES = {"proximal": (1, 0), "compliance": (1, 1), "distal": (1, 2), "amplitude": (0, 1), "viscosity": None}


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


def _with(p, name: str, value):
    if LEAVES[name] is None:
        return {**p, "viscosity": value}
    row, column = LEAVES[name]
    return {**p, "opening_parameters": jnp.asarray(p["opening_parameters"]).at[row, column].set(value)}


def _leaf(p, name: str):
    return p["viscosity"] if LEAVES[name] is None else jnp.asarray(p["opening_parameters"])[LEAVES[name]]


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
def central(scheme: str, name: str, step: float = 1e-4) -> float:
    """The central difference in ``name``, each side a node built afresh, per unit of the leaf."""
    value = float(_leaf(_node(scheme)._step_constants(None), name))  # noqa: SLF001
    return (_rebuilt(scheme, name, 1.0 + step) - _rebuilt(scheme, name, 1.0 - step)) / (2.0 * step * value)


def reverse(scheme: str, name: str, scale: float) -> float:
    node = _node(scheme)
    p = node._step_constants(None)  # noqa: SLF001
    value = jnp.asarray(_leaf(p, name), dtype=jnp.float64)
    return float(jax.jit(jax.grad(lambda v: scale * _loss(node, _with(p, name, v))))(value)) / scale


def forward(scheme: str, name: str, scale: float) -> float:
    node = _node(scheme)
    p = node._step_constants(None)  # noqa: SLF001
    value = jnp.asarray(_leaf(p, name), dtype=jnp.float64)
    _, tangent = jax.jit(lambda v, dv: jax.jvp(lambda x: _loss(node, _with(p, name, x)), (v,), (dv,)))(
        value, jnp.asarray(scale))
    return float(tangent) / scale


@pytest.mark.parametrize("scheme", ["imex_euler", "rk3"])
@pytest.mark.parametrize("name", sorted(LEAVES))
@pytest.mark.parametrize("scale", SCALES)
def test_the_reverse_gradient_with_ports_is_the_rebuilt_central_difference(float64, scheme, name, scale):
    expected = central(scheme, name)
    assert abs(reverse(scheme, name, scale) - expected) <= LINE * abs(expected), (reverse(scheme, name, scale),
                                                                                  expected)


@pytest.mark.parametrize("scheme", ["imex_euler", "rk3"])
@pytest.mark.parametrize("name", sorted(LEAVES))
def test_the_forward_tangent_with_ports_is_the_rebuilt_central_difference(float64, scheme, name):
    expected = central(scheme, name)
    found = forward(scheme, name, 1.0)
    assert abs(found - expected) <= LINE * abs(expected), (found, expected)


def test_control_stop_gradient_on_the_unit_fields_fails_the_viscosity_from_a_cold_prediction(float64, monkeypatch):
    """The unit fields' derivative cut (``stop_gradient`` on each, so on their response ``q``): from a cold prediction
    the first step's moves are the whole pressures, and the viscosity's derivative through them is off the line."""

    def unit_field(self, solve, force, start):
        del self
        unit, report = solve(force, start)
        return tuple(jax.lax.stop_gradient(c) for c in unit), report

    monkeypatch.setattr(OpenFlowNode, "unit_field", unit_field)
    expected = central("imex_euler", "viscosity")
    found = reverse("imex_euler", "viscosity", 1.0)
    assert abs(found - expected) > LINE * abs(expected), (found, expected)
