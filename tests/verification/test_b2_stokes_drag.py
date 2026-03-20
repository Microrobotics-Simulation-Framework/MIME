"""B2: Stokes drag in CSF — verification benchmark.

Pass criterion: Drag force < 5% relative error vs. Stokes law
F = 6*pi*eta*r*V at Re < 0.1.

Tests the CSFFlowNode analytical drag against the exact Stokes solution
for a sphere in quiescent fluid. This validates that the drag computation
is correctly implemented before IB-LBM replaces it.
"""

import math
import pytest
import jax.numpy as jnp

from maddening.core.compliance.validation import (
    verification_benchmark, BenchmarkType,
)
from mime.nodes.environment.csf_flow import CSFFlowNode


# CSF properties
MU = 8.5e-4    # Pa.s
RHO = 1002.0   # kg/m^3


def stokes_drag_analytical(mu, a, V):
    """Exact Stokes drag on a sphere: F = 6*pi*mu*a*V."""
    return 6.0 * math.pi * mu * a * V


def stokes_torque_analytical(mu, a, omega):
    """Exact Stokes rotational drag: T = 8*pi*mu*a^3*omega."""
    return 8.0 * math.pi * mu * a**3 * omega


@verification_benchmark(
    benchmark_id="MIME-VER-001",
    description="Stokes translational drag on a sphere in quiescent CSF",
    node_type="CSFFlowNode",
    benchmark_type=BenchmarkType.ANALYTICAL,
    acceptance_criteria="Relative error < 5% vs. F = 6*pi*mu*a*V",
    references=("Purcell1977",),
)
def test_b2_translational_drag():
    """Compare CSFFlowNode drag against Stokes law for translation."""
    test_cases = [
        # All cases: Re = rho * V * 2a / mu < 0.1
        {"a": 50e-6,  "V": 1e-4},   # Re = 0.012
        {"a": 100e-6, "V": 1e-4},   # Re = 0.024
        {"a": 200e-6, "V": 5e-5},   # Re = 0.024
        {"a": 500e-6, "V": 1e-5},   # Re = 0.012
    ]

    for case in test_cases:
        a = case["a"]
        V = case["V"]

        # Verify Re < 0.1
        Re = RHO * V * (2 * a) / MU
        assert Re < 0.1, f"Re = {Re} > 0.1 — outside Stokes regime"

        node = CSFFlowNode(
            "csf", 0.001,
            robot_radius_m=a,
            fluid_viscosity_pa_s=MU,
            fluid_density_kg_m3=RHO,
            pulsatile=False,
        )
        state = node.initial_state()
        bi = {
            "position": jnp.zeros(3),
            "velocity": jnp.array([V, 0.0, 0.0]),
            "angular_velocity": jnp.zeros(3),
        }
        new_state = node.update(state, bi, 0.001)

        F_computed = float(jnp.abs(new_state["drag_force"][0]))
        F_analytical = stokes_drag_analytical(MU, a, V)

        rel_error = abs(F_computed - F_analytical) / F_analytical
        assert rel_error < 0.05, (
            f"B2 FAIL: a={a*1e6:.0f}um, V={V*1e3:.1f}mm/s, "
            f"F_computed={F_computed:.2e}, F_analytical={F_analytical:.2e}, "
            f"rel_error={rel_error:.4f}"
        )


@verification_benchmark(
    benchmark_id="MIME-VER-002",
    description="Stokes rotational drag on a sphere in quiescent CSF",
    node_type="CSFFlowNode",
    benchmark_type=BenchmarkType.ANALYTICAL,
    acceptance_criteria="Relative error < 5% vs. T = 8*pi*mu*a^3*omega",
    references=("Purcell1977",),
)
def test_b2_rotational_drag():
    """Compare CSFFlowNode rotational drag against Stokes law."""
    test_cases = [
        {"a": 100e-6, "omega": 10.0},
        {"a": 100e-6, "omega": 100.0},
        {"a": 200e-6, "omega": 50.0},
    ]

    for case in test_cases:
        a = case["a"]
        omega = case["omega"]

        node = CSFFlowNode(
            "csf", 0.001,
            robot_radius_m=a,
            fluid_viscosity_pa_s=MU,
            pulsatile=False,
        )
        state = node.initial_state()
        bi = {
            "position": jnp.zeros(3),
            "velocity": jnp.zeros(3),
            "angular_velocity": jnp.array([0.0, 0.0, omega]),
        }
        new_state = node.update(state, bi, 0.001)

        T_computed = float(jnp.abs(new_state["drag_torque"][2]))
        T_analytical = stokes_torque_analytical(MU, a, omega)

        rel_error = abs(T_computed - T_analytical) / T_analytical
        assert rel_error < 0.05, (
            f"B2 FAIL: a={a*1e6:.0f}um, omega={omega}rad/s, "
            f"rel_error={rel_error:.4f}"
        )


@verification_benchmark(
    benchmark_id="MIME-VER-003",
    description="Stokes drag linearity: F proportional to V",
    node_type="CSFFlowNode",
    benchmark_type=BenchmarkType.ANALYTICAL,
    acceptance_criteria="Drag ratio = velocity ratio within 0.1%",
)
def test_b2_drag_linearity():
    """Verify drag scales linearly with velocity (Stokes regime property)."""
    a = 100e-6
    node = CSFFlowNode("csf", 0.001, robot_radius_m=a,
                       fluid_viscosity_pa_s=MU, pulsatile=False)
    state = node.initial_state()

    velocities = [1e-4, 2e-4, 5e-4, 1e-3]
    forces = []
    for V in velocities:
        bi = {"position": jnp.zeros(3),
              "velocity": jnp.array([V, 0.0, 0.0]),
              "angular_velocity": jnp.zeros(3)}
        s = node.update(state, bi, 0.001)
        forces.append(float(jnp.abs(s["drag_force"][0])))

    # Check ratios
    for i in range(1, len(velocities)):
        v_ratio = velocities[i] / velocities[0]
        f_ratio = forces[i] / forces[0]
        assert abs(f_ratio / v_ratio - 1.0) < 0.001, (
            f"Linearity fail: v_ratio={v_ratio}, f_ratio={f_ratio}"
        )


@verification_benchmark(
    benchmark_id="MIME-VER-004",
    description="Ellipsoid drag: C_1 < C_2 for prolate spheroid",
    node_type="RigidBodyNode",
    benchmark_type=BenchmarkType.ANALYTICAL,
    acceptance_criteria="Velocity along major axis > velocity perpendicular for same force",
    references=("Lighthill1976",),
)
def test_b2_ellipsoid_drag_anisotropy():
    """Verify prolate ellipsoid has less drag along its major axis."""
    from mime.nodes.robot.rigid_body import RigidBodyNode

    a = 150e-6  # semi-major
    b = 50e-6   # semi-minor
    node = RigidBodyNode(
        "robot", 0.001,
        semi_major_axis_m=a,
        semi_minor_axis_m=b,
        fluid_viscosity_pa_s=MU,
    )
    state = node.initial_state()
    F = 1e-12

    # Force along major axis
    bi_x = {"magnetic_force": jnp.array([F, 0.0, 0.0])}
    s_x = node.update(state, bi_x, 0.001)
    V_major = float(jnp.abs(s_x["velocity"][0]))

    # Force along minor axis
    bi_y = {"magnetic_force": jnp.array([0.0, F, 0.0])}
    s_y = node.update(state, bi_y, 0.001)
    V_minor = float(jnp.abs(s_y["velocity"][1]))

    assert V_major > V_minor, (
        f"Expected V_major > V_minor, got {V_major:.2e} vs {V_minor:.2e}"
    )
