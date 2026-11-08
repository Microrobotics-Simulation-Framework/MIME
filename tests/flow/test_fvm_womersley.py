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
from fvm_cases import in_dtype, run, tube_node

import mime.nodes.flow.fvm._node as flow_node
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
        node = tube_node(16, nu=NU, g=0.0, dt=dt, offset=offset, convection=False,
                         drive_amplitude=(0.0, 0.0, G0), drive_angular_frequency=omega)
        c = node.constants()
        points = np.asarray(node.grid(node._step_constants(None)).face_points(2, "float64"))  # noqa: SLF001
        r = np.hypot(points[..., 0] - c["tube_centre"][0], points[..., 1] - c["tube_centre"][1])
        quasi = np.where(r < R, (G0 / (4 * NU)) * (R ** 2 - r ** 2), 0.0)
        zero = np.zeros_like(quasi)
        node = ImplicitFlowNode(**{**_arguments(node), "initial_velocity_x": zero, "initial_velocity_y": zero,
                                   "initial_velocity_z": quasi})
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

    found = scipy.optimize.least_squares(residual, [NU, R], x_scale=[NU, R], xtol=1e-14, ftol=1e-14)
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


def _stall_deficit(dtype: str, ratio: float, least: int) -> float:
    reference, reference_node = _reference()
    c = reference_node.constants()
    with in_dtype(dtype):
        node = tube_node(16, nu=NU, g=G_STALL, dt=TAU / ratio, convection=False,
                         **{f"initial_velocity_{k}": 0.9 * reference[i] for i, k in enumerate("xyz")})
        held = flow_node.Stopping
        flow_node.Stopping = functools.partial(Stopping, least_iterations=least)  # type: ignore[misc]
        try:
            state = run(node, int(round(14 * ratio)))
        finally:
            flow_node.Stopping = held  # type: ignore[misc]
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
    ratio, line = STALL[dtype]
    assert _stall_deficit(dtype, ratio, 0) >= line

