"""Verification tests for MotorNode (MIME-NODE-100).

Includes:
* MIME-VER-100 — torque-mode step response vs. analytical first-order
  spin-up of a damped rotor.
* Voltage-mode steady-state torque/speed cross-check.
* Velocity-mode PI tracking with default gains.
* Pose composition closed-form check (90 deg z-rotation).
* JAX traceability under jit / grad / vmap of the ``update`` method.
* Single GraphManager step to confirm wiring.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.compliance.validation import (
    BenchmarkType, verification_benchmark,
)
from maddening.core.graph_manager import GraphManager

from mime.nodes.actuation.motor import MotorNode, compose_pose


# ---------------------------------------------------------------------------
# Reference parameters — small lab-scale brushed DC motor
# ---------------------------------------------------------------------------

REF_PARAMS = dict(
    inertia_kg_m2=1e-4,        # J  [kg.m^2]
    kt_n_m_per_a=0.05,         # k_t [N.m/A]  (k_e auto-defaults to same)
    r_ohm=2.0,                 # R  [ohm]
    l_henry=1e-3,              # L  [H]
    damping_n_m_s=0.01,        # b  [N.m.s/rad]
)


def _make_node(**overrides) -> MotorNode:
    p = dict(REF_PARAMS)
    p.update(overrides)
    return MotorNode("motor", timestep=1e-3, **p)


# ---------------------------------------------------------------------------
# MIME-VER-100 — torque-mode step response
# ---------------------------------------------------------------------------

@pytest.mark.slow
@verification_benchmark(
    benchmark_id="MIME-VER-100",
    description=(
        "Torque-mode step response of MotorNode vs. analytical first-order "
        "rotor spin-up omega(t) = (tau/b)·(1 - exp(-b·t/J))."
    ),
    node_type="MotorNode",
    benchmark_type=BenchmarkType.ANALYTICAL,
    acceptance_criteria="Relative RMS error of omega(t) over 1 s < 5%",
)
def test_torque_mode_step_response_analytical():
    dt = 1e-3
    n_steps = 1000  # 1 s
    tau_cmd = 0.02  # N.m, well below saturation

    node = _make_node()
    state = node.initial_state()

    omegas = []
    times = []
    for k in range(n_steps):
        bi = {"commanded_torque": jnp.array(tau_cmd)}
        state = node.update(state, bi, dt)
        omegas.append(float(state["angular_velocity"]))
        times.append((k + 1) * dt)

    omegas = np.asarray(omegas)
    times = np.asarray(times)

    J = REF_PARAMS["inertia_kg_m2"]
    b = REF_PARAMS["damping_n_m_s"]
    omega_analytical = (tau_cmd / b) * (1.0 - np.exp(-b * times / J))

    # Skip the very first sample (t=dt) where semi-implicit and analytical
    # diverge by O(dt) — we still include it in the RMS though, as the plan
    # requires the trajectory-wide error.
    rms_err = np.sqrt(
        np.mean(((omegas - omega_analytical) / np.maximum(omega_analytical, 1e-12)) ** 2)
    )
    assert rms_err < 0.05, (
        f"MIME-VER-100 FAIL: torque-mode RMS rel-error = {rms_err:.4f} "
        f"> 0.05. Final omega(sim)={omegas[-1]:.4f}, "
        f"omega(an)={omega_analytical[-1]:.4f}"
    )

    # Sanity: in torque-mode the current must stay zero.
    assert float(state["current"]) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Voltage-mode steady-state cross-check
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_voltage_mode_steady_state():
    """Apply step voltage; confirm ω → V·k_t / (k_t·k_e + R·b)."""
    dt = 1e-4   # finer dt — RL time constant is L/R = 5e-4 s
    n_steps = 100_000  # 10 s, well past every time constant
    V = 5.0

    node = _make_node()
    state = node.initial_state()

    for _ in range(n_steps):
        bi = {"commanded_voltage": jnp.array(V)}
        state = node.update(state, bi, dt)

    omega_ss_sim = float(state["angular_velocity"])
    i_ss_sim = float(state["current"])

    kt = REF_PARAMS["kt_n_m_per_a"]
    ke = kt
    R = REF_PARAMS["r_ohm"]
    b = REF_PARAMS["damping_n_m_s"]

    omega_ss_th = V * kt / (kt * ke + R * b)
    i_ss_th = (V - ke * omega_ss_th) / R

    assert abs(omega_ss_sim - omega_ss_th) / omega_ss_th < 0.01, (
        f"voltage-mode omega_ss: sim={omega_ss_sim:.4f}, "
        f"theory={omega_ss_th:.4f}"
    )
    assert abs(i_ss_sim - i_ss_th) / i_ss_th < 0.01, (
        f"voltage-mode i_ss: sim={i_ss_sim:.4e}, theory={i_ss_th:.4e}"
    )


# ---------------------------------------------------------------------------
# Velocity-mode PI tracking
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_velocity_mode_pi_tracking():
    """Velocity-mode reaches commanded omega with <1% steady-state error."""
    dt = 1e-3
    n_steps = 5000  # 5 s — long enough for default Ki to integrate down
    omega_des = 10.0  # rad/s

    node = _make_node()
    state = node.initial_state()

    for _ in range(n_steps):
        bi = {"commanded_velocity": jnp.array(omega_des)}
        state = node.update(state, bi, dt)

    omega_final = float(state["angular_velocity"])
    err = abs(omega_final - omega_des) / omega_des
    assert err < 0.01, (
        f"velocity-mode steady-state err = {err:.4f} (>1%). "
        f"omega_final = {omega_final:.4f}, target = {omega_des:.4f}"
    )


# ---------------------------------------------------------------------------
# Pose composition — 90 deg z-rotation
# ---------------------------------------------------------------------------

def test_pose_composition_z_rotation_90():
    parent = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # identity
    axis = jnp.array([0.0, 0.0, 1.0])
    angle = jnp.array(math.pi / 2.0)
    tool = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # identity

    pose = compose_pose(parent, axis, angle, tool)

    # Translation must be zero
    assert float(jnp.linalg.norm(pose[0:3])) < 1e-7

    # Quaternion for 90deg z-rotation: [cos45, 0, 0, sin45]
    expected_q = jnp.array([
        math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0),
    ])
    q = pose[3:7]
    err = float(jnp.linalg.norm(q - expected_q))
    assert err < 1e-6, f"pose quat = {q}, expected {expected_q}"


def test_pose_composition_with_tool_offset():
    """Tool offset along +x in rotor frame, parent at origin, rotor angle 0
    -> world position should equal the offset."""
    parent = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    axis = jnp.array([0.0, 0.0, 1.0])
    angle = jnp.array(0.0)
    tool = jnp.array([0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    pose = compose_pose(parent, axis, angle, tool)
    np.testing.assert_allclose(np.asarray(pose[0:3]), [0.1, 0.0, 0.0], atol=1e-7)


# ---------------------------------------------------------------------------
# JAX traceability — jit / grad / vmap
# ---------------------------------------------------------------------------

def test_update_is_jittable():
    node = _make_node()
    state = node.initial_state()
    dt = 1e-3

    @jax.jit
    def step(s, tau):
        return node.update(s, {"commanded_torque": tau}, dt)

    new_state = step(state, jnp.array(0.01))
    assert "angular_velocity" in new_state
    assert float(new_state["angular_velocity"]) > 0.0


def test_update_is_differentiable():
    """Gradient of final omega w.r.t. applied torque must be finite & positive."""
    node = _make_node()
    init = node.initial_state()
    dt = 1e-3
    n = 50

    def loss(tau):
        s = init
        for _ in range(n):
            s = node.update(s, {"commanded_torque": tau}, dt)
        return s["angular_velocity"]

    g = jax.grad(loss)(jnp.array(0.01))
    assert jnp.isfinite(g)
    assert float(g) > 0.0


def test_update_is_vmappable():
    node = _make_node()
    init = node.initial_state()
    dt = 1e-3

    def step(tau):
        return node.update(init, {"commanded_torque": tau}, dt)["angular_velocity"]

    taus = jnp.linspace(0.0, 0.05, 8)
    omegas = jax.vmap(step)(taus)
    assert omegas.shape == (8,)
    # omega should be monotonically non-decreasing in tau (at this fixed dt).
    diffs = jnp.diff(omegas)
    assert bool(jnp.all(diffs >= -1e-9))


# ---------------------------------------------------------------------------
# GraphManager integration — one step
# ---------------------------------------------------------------------------

def test_motor_runs_in_graph_manager():
    node = _make_node()

    gm = GraphManager()
    gm.add_node(node)
    gm.add_external_input("motor", "commanded_torque", shape=())
    gm.add_external_input("motor", "parent_pose_world", shape=(7,))
    gm.compile()

    ext = {
        "motor": {
            "commanded_torque": jnp.array(0.02, dtype=jnp.float32),
            "parent_pose_world": jnp.array(
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=jnp.float32,
            ),
        },
    }
    gm.step(external_inputs=ext)
    state = gm.get_node_state("motor")
    assert float(state["angular_velocity"]) > 0.0
    fluxes = node.compute_boundary_fluxes(state, ext["motor"], 1e-3)
    assert fluxes["rotor_pose_world"].shape == (7,)
    assert "rotor_angular_velocity" in fluxes
    assert "rotor_angle" in fluxes


# ---------------------------------------------------------------------------
# Mode precedence sanity check
# ---------------------------------------------------------------------------

def test_torque_mode_preempts_voltage_mode():
    """When both commanded_torque and commanded_voltage are non-zero,
    torque-mode wins and the armature current stays at zero."""
    dt = 1e-3
    node = _make_node()
    s = node.initial_state()
    bi = {
        "commanded_torque": jnp.array(0.01),
        "commanded_voltage": jnp.array(5.0),
    }
    s = node.update(s, bi, dt)
    assert float(s["current"]) == pytest.approx(0.0, abs=1e-12)
    assert float(s["angular_velocity"]) > 0.0


def test_metadata_consistency():
    """MIME consistency check returns no errors for MotorNode."""
    node = _make_node()
    errors = node.validate_mime_consistency()
    assert errors == [], f"validate_mime_consistency errors: {errors}"
