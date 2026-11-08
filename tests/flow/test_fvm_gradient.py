"""Gate 9, float64: the flow's gradient in the viscosity and the drive, reverse and forward mode, against a central
difference that rebuilds the node, at loss scales 1e-12 and 1e6 (the scale test), with the derivative solves' reports
read from the report block; and its controls: ``stop_gradient`` on the penalty's viscosity, the penalty's viscosity baked
at build, and the primal's plain warm start in the shared solve closure.

The practical-differentiability spike's case: a periodic tube of R = 4 cells (D/h 8), two cells along z, a body force 8
on the fluid, nu 1, six IMEX Euler steps of dt 0.02 from rest, the output the lumen's flux; the CG to 1e-12. The step
is far above ``tol x tau/dt`` (tau/dt about 2), so no stall's level enters the difference."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fvm_cases import Waveform, step_of, tube_node

import mime.nodes.flow.fvm._solve as solve_module
from mime.nodes.flow.fvm import ImplicitFlowNode
from mime.nodes.flow.fvm._solve import REPORT_FIELDS

STEPS, DT, G = 6, 0.02, 8.0
AT_CAP, STAGNATED = REPORT_FIELDS.index("at_cap"), REPORT_FIELDS.index("stagnated")
#: The gate's line: the derivative against the central difference, relative. The difference's own error at a relative
#: step of 1e-4 is its truncation, about 1e-8 here (the spike: 1e-12 to 1e-9 at 1e-5).
LINE = 1e-6


def _node(nu: float = 1.0, g: float = G) -> ImplicitFlowNode:
    return tube_node(8, nu=nu, g=g, dt=DT, length_cells=2, convection=False, tolerance=1e-12)


def _flux(node: ImplicitFlowNode, p, g) -> jax.Array:
    state = node.initial_state()
    step = step_of(node, p, Waveform(steady=g))

    def body(s, _):
        return step(s), None

    ended, _ = jax.lax.scan(body, dict(state), None, length=STEPS)
    return ended["lumen_flux"], ended["solve_report_sum"]


def _rebuilt_flux(nu: float, g: float) -> float:
    node = _node(nu, g)
    return float(jax.jit(lambda p: _flux(node, p, g)[0])(node._step_constants(None)))  # noqa: SLF001


def central(name: str, step: float = 1e-4) -> float:
    """The central difference in ``name`` (``viscosity`` or ``drive``), each side a node built afresh."""
    if name == "viscosity":
        return (_rebuilt_flux(1.0 + step, G) - _rebuilt_flux(1.0 - step, G)) / (2 * step)
    return (_rebuilt_flux(1.0, G * (1 + step)) - _rebuilt_flux(1.0, G * (1 - step))) / (2 * step * G)


def _leaf(p, name: str):
    return p["viscosity"] if name == "viscosity" else G


def _with(p, name: str, value):
    """The step's constants and the drive, with ``name``'s value replaced."""
    if name == "viscosity":
        return {**p, "viscosity": value}, G
    return p, value


def reverse(name: str, scale: float):
    node = _node()
    p = node._step_constants(None)  # noqa: SLF001

    def loss(value, tap):
        constants, g = _with(p, name, value)
        flux, _ = _flux(node, {**constants, "solve_report": tap}, g)
        return scale * flux

    value, tap = jnp.asarray(_leaf(p, name), dtype=jnp.float64), jnp.asarray(p["solve_report"])
    gradient, transposed = jax.jit(jax.grad(loss, argnums=(0, 1)))(value, tap)
    return float(gradient) / scale, np.asarray(transposed)[0]


def forward(name: str, scale: float):
    node = _node()
    p = node._step_constants(None)  # noqa: SLF001
    value = jnp.asarray(_leaf(p, name), dtype=jnp.float64)
    (_, _), (tangent, reports) = jax.jit(lambda v, dv: jax.jvp(
        lambda x: _flux(node, *_with(p, name, x)), (v,), (dv,)))(value, jnp.asarray(scale))
    return float(tangent) / scale, np.asarray(reports)


@functools.cache
def _central(name: str) -> float:
    return central(name)


#: The loss scales: the decided 1e-12 and 1e6, and 1e-30, where a start that does not scale with its right-hand side
#: breaks (at 1e-12 the CG recovers from the primal's warm start, at 2.5 times the iterations).
SCALES = [1e-30, 1e-12, 1e6]


@pytest.mark.parametrize("name", ["viscosity", "drive"])
@pytest.mark.parametrize("scale", SCALES)
def test_the_reverse_gradient_is_the_rebuilt_central_difference(float64, name, scale):
    gradient, transposed = reverse(name, scale)
    assert abs(gradient / _central(name) - 1.0) < LINE
    assert transposed[AT_CAP] == 0.0 and transposed[STAGNATED] == 0.0
    assert transposed[0] >= STEPS


@pytest.mark.parametrize("name", ["viscosity", "drive"])
@pytest.mark.parametrize("scale", SCALES)
def test_the_forward_tangent_is_the_rebuilt_central_difference(float64, name, scale):
    tangent, reports = forward(name, scale)
    assert abs(tangent / _central(name) - 1.0) < LINE
    assert reports[AT_CAP] == 0.0 and reports[STAGNATED] == 0.0


def test_control_stop_gradient_on_the_penalty_viscosity_fails_the_gate(float64, monkeypatch):
    original = ImplicitFlowNode.resistivity

    def stopped(self, p, fields):
        return original(self, {**p, "viscosity": jax.lax.stop_gradient(p["viscosity"])}, fields)

    monkeypatch.setattr(ImplicitFlowNode, "resistivity", stopped)
    gradient, _ = reverse("viscosity", 1.0)
    assert abs(gradient / _central("viscosity") - 1.0) >= LINE


def test_control_the_penalty_viscosity_baked_at_build_fails_the_gate(float64, monkeypatch):
    """The value the node was built with, a Python float, in place of the traced one: the rebuilding difference sees
    it, the derivative does not."""
    original = ImplicitFlowNode.resistivity

    def baked(self, p, fields):
        return original(self, {**p, "viscosity": float(self.constants()["viscosity"])}, fields)

    monkeypatch.setattr(ImplicitFlowNode, "resistivity", baked)
    gradient, _ = reverse("viscosity", 1.0)
    assert abs(gradient / _central("viscosity") - 1.0) >= LINE


@pytest.mark.parametrize("mode", ["reverse", "forward"])
def test_control_the_primals_plain_warm_start_fails_the_scale_test(float64, monkeypatch, mode):
    """The previous step's velocity as every solve's start, the tangent's and transpose's included (the shared
    closure): at a loss scale of 1e-30 their right-hand sides are thirty orders under the start's residual, so they
    stagnate, which their reports show, or end wrong. (At 1e-12 the CG recovers, at 2.5 times the iterations: the
    structural test that every start scales with its right-hand side, in ``test_fvm_solve.py``, sees it there.)"""
    nu_central = _central("viscosity")

    def plain(guess, b, operator, inner):
        return guess if guess is not None else tuple(jnp.zeros_like(c) for c in b)

    monkeypatch.setattr(solve_module, "scaling_start", plain)
    gradient, report = reverse("viscosity", 1e-30) if mode == "reverse" else forward("viscosity", 1e-30)
    flagged = report[AT_CAP] + report[STAGNATED] > 0
    assert flagged or abs(gradient / nu_central - 1.0) >= LINE
