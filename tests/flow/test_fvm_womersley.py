"""Gate 4: the low-Womersley Bessel fit with a free radius (S7 case 2(c)'s operating point), and gate 8b: the stall's
outcome on the empty tube (the stall spike's set-up), each with its control.

Gate 4: an aligned periodic tube of D = 1, nu = 1, driven by ``G0 cos(omega t)`` on the fluid, alpha 0.137 and 0.29,
100 IMEX Euler steps a period, three periods from the quasi-static state; the last period's first harmonic fitted by
Womersley's profile with the viscosity and the radius free over the clean lumen. Its line is absolute, ``|nu_fit/nu -
1| < 0.5 %``; the 1 % viscosity plant (the viscous term only) at the same dt must fail it.

Gate 8b: the steady solve's outcome after 14 tau from 0.9 of the fixed point, ``tau = R^2 / (5.783 nu)``, against a
float64 reference (the steady solve at 1e-12): float32 at ``tau/dt`` 1e2 under 1e-4, float64 at ``tau/dt`` 1e3 under
1e-6; the control, the CG allowed to accept its warm start unsolved (no minimum iteration), above each line."""

from __future__ import annotations

import functools
import math

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.optimize
import scipy.special
from fvm_cases import DRIVES, driven, in_dtype, run, tube_node

import mime.nodes.flow.fvm._node as flow_node
import mime.nodes.flow.fvm._solve as solve_module
from mime.nodes.flow.fvm import ImplicitFlowNode, Stopping

NU, R, G0 = 1.0, 0.5, 8.0
FIT_LINE = 5e-3
STEPS_A_PERIOD = 100


def womersley(r, nu, radius, omega):
    """The first harmonic's complex amplitude of the axial velocity for a drive ``G0 cos(omega t)``."""
    alpha = radius * math.sqrt(omega / nu)
    k = (1j ** 1.5) * alpha
    return (G0 / (1j * omega)) * (1.0 - scipy.special.jv(0, k * r / radius) / scipy.special.jv(0, k))


@functools.cache
def _harmonic(dtype: str, alpha: float, offset: tuple[float, float], plant: float):
    omega = (alpha / R) ** 2 * NU
    period = 2.0 * math.pi / omega
    dt = period / STEPS_A_PERIOD
    with in_dtype(dtype):
        node = tube_node(16, nu=NU, g=0.0, dt=dt, offset=offset, convection=False, amplitude=G0, omega=omega)
        c = node.constants()
        points = np.asarray(node.grid(node._step_constants(None)).face_points(2, "float64"))  # noqa: SLF001
        r = np.hypot(points[..., 0] - c["tube_centre"][0], points[..., 1] - c["tube_centre"][1])
        quasi = np.where(r < R, (G0 / (4 * NU)) * (R ** 2 - r ** 2), 0.0)
        zero = np.zeros_like(quasi)
        node = driven(ImplicitFlowNode(**{**_arguments(node), "initial_velocity_x": zero, "initial_velocity_y": zero,
                                          "initial_velocity_z": quasi}), DRIVES[node])
        if plant != 1.0:
            original = ImplicitFlowNode.operator_of
            ImplicitFlowNode.operator_of = lambda self, p, grid, sigma: original(  # type: ignore[method-assign]
                self, {**p, "viscosity": plant * p["viscosity"]}, grid, sigma)
        try:
            state = run(node, 2 * STEPS_A_PERIOD)
            _, recorded = run(node, STEPS_A_PERIOD, state, record=lambda s: jnp.mean(s["velocity_z"], axis=2))
            samples = list(np.asarray(recorded, np.float64))
        finally:
            if plant != 1.0:
                ImplicitFlowNode.operator_of = original  # type: ignore[method-assign]
    times = (2 * STEPS_A_PERIOD + 1 + np.arange(STEPS_A_PERIOD)) * dt
    amplitude = (2.0 / STEPS_A_PERIOD) * np.tensordot(np.exp(-1j * omega * times), np.stack(samples), axes=1)
    return amplitude, r[..., 0], omega, node.constants()["spacing"]


def _arguments(node: ImplicitFlowNode) -> dict:
    c = {k: v for k, v in node.constants().items() if k != "solve_report"}
    return {"name": "flow", "timestep": node.delta_t, **c}


def fitted_viscosity(dtype: str, alpha: float, offset: tuple[float, float], plant: float = 1.0) -> float:
    amplitude, r, omega, h = _harmonic(dtype, alpha, offset, plant)
    clean = r < R - 1.5 * h
    measured = amplitude[clean]

    def residual(x):
        model = womersley(r[clean], x[0], x[1], omega)
        return np.concatenate([(model - measured).real, (model - measured).imag])

    # Levenberg-Marquardt with a relative difference step of 1e-6: the trust-region default stopped at its start on
    # float32 data (its steps all rejected, the difference step at rounding's scale), an oracle's failure, not the
    # solver's.
    found = scipy.optimize.least_squares(residual, [NU, R], method="lm", diff_step=1e-6, xtol=1e-15, ftol=1e-15,
                                         gtol=1e-15)
    return float(found.x[0])


OFFSETS = ((0.0, 0.0), (0.31, 0.12))


@pytest.mark.parametrize("alpha", [0.137, 0.29])
@pytest.mark.parametrize("offset", OFFSETS)
def test_the_low_womersley_fit_gives_the_viscosity(dtype, alpha, offset):
    assert abs(fitted_viscosity(dtype, alpha, offset) / NU - 1.0) < FIT_LINE


@pytest.mark.parametrize("alpha", [0.137, 0.29])
@pytest.mark.parametrize("offset", OFFSETS)
def test_control_a_one_percent_viscosity_plant_fails_the_womersley_fit(dtype, alpha, offset):
    assert abs(fitted_viscosity(dtype, alpha, offset, 1.01) / NU - 1.0) >= FIT_LINE


# -- the stall's outcome -------------------------------------------------------------------------------------------

TAU = R ** 2 / (5.783185962946784 * NU)
G_STALL = 8 * NU * 0.1 / R ** 2


@functools.cache
def _reference():
    with in_dtype("float64"):
        node = tube_node(16, nu=NU, g=G_STALL, dt=1.0, convection=False, scheme="steady", tolerance=1e-12)
        state = run(node, 2)
    return tuple(np.asarray(state[f"velocity_{c}"], np.float64) for c in "xyz"), node


def _plain_warm_start(guess, b, operator, inner):
    """The spike's start (the plant of the gate's second control): the previous velocity as it is, not its Galerkin
    multiple."""
    del operator, inner
    return guess if guess is not None else tuple(0.0 * c for c in b)


def _stall_deficit(dtype: str, ratio: float, least: int, plain: bool = False) -> float:
    reference, reference_node = _reference()
    c = reference_node.constants()
    with in_dtype(dtype), pytest.MonkeyPatch.context() as patch:
        node = tube_node(16, nu=NU, g=G_STALL, dt=TAU / ratio, convection=False,
                         **{f"initial_velocity_{k}": 0.9 * reference[i] for i, k in enumerate("xyz")})
        patch.setattr(flow_node, "Stopping", functools.partial(Stopping, least_iterations=least))
        if plain:
            patch.setattr(solve_module, "scaling_start", _plain_warm_start)
        state = run(node, int(round(14 * ratio)))
    grid = node.grid(node._step_constants(None))  # noqa: SLF001
    numerator = denominator = 0.0
    for k, name in enumerate("xyz"):
        points = np.asarray(grid.face_points(k, "float64"))
        fluid = np.hypot(points[..., 0] - c["tube_centre"][0], points[..., 1] - c["tube_centre"][1]) < R
        found = np.asarray(state[f"velocity_{name}"], np.float64)
        numerator += float(np.sum((found - reference[k])[fluid] ** 2))
        denominator += float(np.sum(reference[k][fluid] ** 2))
    return math.sqrt(numerator / denominator)


STALL = {"float32": (1e2, 1e-4), "float64": (1e3, 1e-6)}


def test_a_near_steady_run_reaches_the_fixed_point(dtype):
    ratio, line = STALL[dtype]
    assert _stall_deficit(dtype, ratio, 1) < line


def test_control_accepting_the_warm_start_unsolved_stalls_above_the_line(dtype):
    """With no minimum iteration, the shipped (Galerkin) start's stall sits above the line, thinly: measured 2.38e-4
    against 1e-4 in float32, 2.10e-6 against 1e-6 in float64 (the Galerkin multiple itself lowers the stall)."""
    ratio, line = STALL[dtype]
    assert _stall_deficit(dtype, ratio, 0) >= line


def test_control_the_spikes_plain_warm_start_unsolved_stalls_near_tenfold_over_the_line(dtype):
    """The draft's control at its margin: the stall the record's decision removes, the stall spike's plain warm start
    accepted unsolved, measured 8.39e-4 against 1e-4 in float32 and 1.01e-5 against 1e-6 in float64. No control at the
    decided set-up can sit much further over the line: a stall is bounded by tol x tau/dt, and each line is a tenth of
    that bound (1e-5 x 1e2 and 1e-8 x 1e3); the plain start reaches 0.84 and 1.01 of it, the shipped Galerkin start 0.23
    and 0.21. So the line asserted is five times the gate's, under each measured margin."""
    ratio, line = STALL[dtype]
    assert _stall_deficit(dtype, ratio, 0, plain=True) >= 5.0 * line

