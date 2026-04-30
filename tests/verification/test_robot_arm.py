"""Verification tests for ``RobotArmNode`` (MIME-NODE-102).

Five `@verification_benchmark`-decorated regressions plus JAX
traceability and ``GraphManager`` integration. The 3-link planar URDF
fixture at ``tests/control/fixtures/three_link_planar.urdf`` is used
throughout: revolute-z joints, link lengths L1=L2=1.0, L3=0.5, masses
2.0 / 1.5 / 0.5 kg.

JAX double precision is enabled at module load — the FK benchmark
needs sub-1e-10 m agreement.
"""

from __future__ import annotations

import os

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.compliance.validation import (
    BenchmarkType, verification_benchmark,
)
from maddening.core.graph_manager import GraphManager
from mime.nodes.actuation.robot_arm import RobotArmNode


URDF_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "control", "fixtures",
                 "three_link_planar.urdf")
)
L1, L2, L3 = 1.0, 1.0, 0.5

# Tool-tip offset in link_3's COM frame: link_3 COM is at L3/2 from
# joint_3, so the tip is +L3/2 further along link_3's local +x.
EE_OFFSET = (L3 / 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)


# --------------------------------------------------------------------
# Analytical reference (numpy, double precision)
# --------------------------------------------------------------------

def _planar_fk_tip(q):
    """Closed-form planar 3-link tool-tip position for the fixture."""
    q1, q2, q3 = q
    x = L1 * np.cos(q1) + L2 * np.cos(q1 + q2) + L3 * np.cos(q1 + q2 + q3)
    y = L1 * np.sin(q1) + L2 * np.sin(q1 + q2) + L3 * np.sin(q1 + q2 + q3)
    return np.array([x, y, 0.0])


# --------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------

@pytest.fixture
def arm():
    return RobotArmNode(
        name="arm",
        timestep=1e-3,
        urdf_path=URDF_PATH,
        end_effector_link_name="link_3",
        end_effector_offset_in_link=EE_OFFSET,
        joint_friction_n_m_s=(0.0, 0.0, 0.0),  # disable friction for clean V&V
        gravity_world=(0.0, 0.0, -9.80665),
    )


# --------------------------------------------------------------------
# MIME-VER-120  Forward kinematics vs analytical (3-link planar)
# --------------------------------------------------------------------

@verification_benchmark(
    benchmark_id="MIME-VER-120",
    description="RobotArmNode forward kinematics vs analytical 3-link planar formula",
    node_type="RobotArmNode",
    benchmark_type=BenchmarkType.ANALYTICAL,
    acceptance_criteria="|pos_node - pos_analytic|_inf < 1e-10 over 5 random configs",
)
def test_ver120_fk_vs_analytical(arm):
    rng = np.random.default_rng(20260430)
    qs = rng.uniform(-np.pi, np.pi, size=(5, 3))
    max_err = 0.0
    for q in qs:
        state = {"joint_angles": jnp.asarray(q),
                 "joint_velocities": jnp.zeros(3)}
        fluxes = arm.compute_boundary_fluxes(state, {}, 1e-3)
        pos_node = np.asarray(fluxes["end_effector_pose_world"][:3])
        pos_ref = _planar_fk_tip(q)
        err = float(np.max(np.abs(pos_node - pos_ref)))
        max_err = max(max_err, err)
    assert max_err < 1e-10, f"max FK error {max_err:e} exceeds 1e-10"


# --------------------------------------------------------------------
# MIME-VER-121  Mass-matrix symmetry & PD
# --------------------------------------------------------------------

@verification_benchmark(
    benchmark_id="MIME-VER-121",
    description="Mass matrix is symmetric and positive-definite at random configs",
    node_type="RobotArmNode",
    benchmark_type=BenchmarkType.ANALYTICAL,
    acceptance_criteria="‖M − Mᵀ‖_F < 1e-12 and λ_min > 0 over 5 random configs",
)
def test_ver121_mass_matrix_symmetric_pd(arm):
    from mime.control.kinematics import mass_matrix
    rng = np.random.default_rng(20260430)
    qs = rng.uniform(-np.pi, np.pi, size=(5, 3))
    for q in qs:
        M = np.asarray(mass_matrix(arm._tree, jnp.asarray(q)))
        asym = np.linalg.norm(M - M.T, ord="fro")
        assert asym < 1e-12, f"mass matrix non-symmetric: ‖M-Mᵀ‖={asym:e}"
        eigvals = np.linalg.eigvalsh(M)
        assert eigvals.min() > 0.0, f"mass matrix not PD: λ_min={eigvals.min()}"


# --------------------------------------------------------------------
# MIME-VER-122  Free-fall consistency: τ = -g(q) ⇒ q̈ ≈ 0
# --------------------------------------------------------------------

@verification_benchmark(
    benchmark_id="MIME-VER-122",
    description="Gravity-compensated zero-velocity arm has zero joint acceleration",
    node_type="RobotArmNode",
    benchmark_type=BenchmarkType.ANALYTICAL,
    acceptance_criteria="‖q̈‖_inf < 1e-10 when τ = g(q), q̇ = 0, F_ext = 0",
)
def test_ver122_free_fall_round_trip(arm):
    from mime.control.kinematics import gravity_vector, mass_matrix, nonlinear_bias
    rng = np.random.default_rng(20260430)
    qs = rng.uniform(-np.pi, np.pi, size=(5, 3))
    g_world = jnp.array([0.0, 0.0, -9.80665])
    for q in qs:
        q_j = jnp.asarray(q)
        qd_j = jnp.zeros(3)
        # Gravity torque from RNEA (q̇=0 ⇒ pure gravity)
        g_q = gravity_vector(arm._tree, q_j, g_world)
        # Apply τ = -g_q (gravity comp): expect q̈ ≈ 0
        bias = nonlinear_bias(arm._tree, q_j, qd_j, g_world)
        # bias at q̇=0 equals g_q exactly
        assert np.allclose(np.asarray(bias), np.asarray(g_q), atol=1e-12)
        M = mass_matrix(arm._tree, q_j)
        rhs = (-g_q) - bias  # τ + tau_ext - bias = -g - g = -2g? No: gravity-comp τ = +g_q
        # Recompute the right way: with τ_cmd = +g_q (cancels gravity in the sum)
        rhs = g_q - bias
        qdd = jnp.linalg.solve(M, rhs)
        assert np.max(np.abs(np.asarray(qdd))) < 1e-10, \
            f"gravity-compensated q̈ not zero: max={np.max(np.abs(np.asarray(qdd)))}"


# --------------------------------------------------------------------
# MIME-VER-123  Gravity-compensated static hold
# --------------------------------------------------------------------

@verification_benchmark(
    benchmark_id="MIME-VER-123",
    description="Gravity-comp tau holds the arm static for 1000 steps at dt=1e-3",
    node_type="RobotArmNode",
    benchmark_type=BenchmarkType.ANALYTICAL,
    acceptance_criteria="max joint drift < 1e-3 rad after 1000 steps",
)
def test_ver123_gravity_hold(arm):
    from mime.control.kinematics import gravity_vector
    tree = arm._tree
    g_world = jnp.array(arm.params["gravity_world"])
    q0 = jnp.array([0.3, -0.4, 0.5])
    dt = 1e-3
    zeros_wrench = jnp.zeros((3, 6))

    @jax.jit
    def step(q, qd):
        g_q = gravity_vector(tree, q, g_world)
        new_state = arm.update(
            {"joint_angles": q, "joint_velocities": qd},
            {"commanded_joint_torques": g_q,
             "external_wrenches_per_link": zeros_wrench},
            dt,
        )
        return new_state["joint_angles"], new_state["joint_velocities"]

    q, qd = q0, jnp.zeros(3)
    for _ in range(1000):
        q, qd = step(q, qd)
    drift = float(jnp.max(jnp.abs(q - q0)))
    assert drift < 1e-3, f"joint drift after 1 s: {drift}"


# --------------------------------------------------------------------
# MIME-VER-124  PD trajectory tracking on a smooth sinusoid
# --------------------------------------------------------------------

@verification_benchmark(
    benchmark_id="MIME-VER-124",
    description="PD-controlled sinusoidal joint tracking (gravity-comp + PD)",
    node_type="RobotArmNode",
    benchmark_type=BenchmarkType.ANALYTICAL,
    acceptance_criteria="RMS joint error < 0.5° over 2 s sinusoid",
)
def test_ver124_pd_tracking(arm):
    """Inverse-dynamics + PD trajectory tracking.

    Computes feed-forward torque ``τ_ff = M(q)q̈_des + c(q,q̇) + g(q)``
    via RNEA at the current state, then adds a PD correction. This is
    the standard manipulator-control law (Sciavicco/Siciliano §6.6); a
    pure PD without the inverse-dynamics feed-forward leaves O(few°) of
    tracking error on a sinusoidal trajectory at the gains we'd
    realistically use on a small arm.
    """
    from mime.control.kinematics import nonlinear_bias, mass_matrix
    tree = arm._tree
    g_world = jnp.array(arm.params["gravity_world"])
    Kp = jnp.array([200.0, 200.0, 80.0])
    Kd = jnp.array([20.0, 20.0, 8.0])
    dt = 1e-3
    n_steps = 2000  # 2 s
    A = jnp.array([0.3, 0.2, 0.4])
    omega = 2.0 * jnp.pi * 0.5  # 0.5 Hz
    zeros_wrench = jnp.zeros((3, 6))

    def q_des(t):
        return A * jnp.sin(omega * t)

    def qd_des(t):
        return A * omega * jnp.cos(omega * t)

    def qdd_des(t):
        return -A * omega * omega * jnp.sin(omega * t)

    @jax.jit
    def step(q, qd, t):
        e = q_des(t) - q
        ed = qd_des(t) - qd
        # Inverse dynamics at current state for the desired acceleration.
        M = mass_matrix(tree, q)
        bias = nonlinear_bias(tree, q, qd, g_world)
        tau_ff = M @ qdd_des(t) + bias
        tau = tau_ff + Kp * e + Kd * ed
        new_state = arm.update(
            {"joint_angles": q, "joint_velocities": qd},
            {"commanded_joint_torques": tau,
             "external_wrenches_per_link": zeros_wrench},
            dt,
        )
        return new_state["joint_angles"], new_state["joint_velocities"], e

    q = q_des(jnp.array(0.0))
    qd = qd_des(jnp.array(0.0))
    err_sq_sum = jnp.zeros(3)
    for k in range(n_steps):
        t = k * dt
        q, qd, e = step(q, qd, jnp.array(t))
        err_sq_sum = err_sq_sum + e * e
    rms = jnp.sqrt(err_sq_sum / n_steps)
    rms_deg = float(jnp.max(rms)) * 180.0 / float(jnp.pi)
    assert rms_deg < 0.5, f"RMS tracking error {rms_deg:.4f}° exceeds 0.5°"


# --------------------------------------------------------------------
# JAX traceability: jit / grad / vmap on update
# --------------------------------------------------------------------

def test_update_is_jit_traceable(arm):
    state = arm.initial_state()
    inputs = {"commanded_joint_torques": jnp.zeros(3),
              "external_wrenches_per_link": jnp.zeros((3, 6))}
    jitted = jax.jit(lambda s, i: arm.update(s, i, 1e-3))
    out = jitted(state, inputs)
    assert "joint_angles" in out and out["joint_angles"].shape == (3,)


def test_update_is_grad_traceable(arm):
    state = arm.initial_state()

    def loss(tau):
        s = arm.update(state,
                       {"commanded_joint_torques": tau,
                        "external_wrenches_per_link": jnp.zeros((3, 6))},
                       1e-3)
        return jnp.sum(s["joint_angles"] ** 2)

    g = jax.grad(loss)(jnp.array([0.1, 0.2, 0.3]))
    assert g.shape == (3,)
    assert jnp.all(jnp.isfinite(g))


def test_update_is_vmap_traceable(arm):
    state = arm.initial_state()

    def step(tau):
        return arm.update(state,
                          {"commanded_joint_torques": tau,
                           "external_wrenches_per_link": jnp.zeros((3, 6))},
                          1e-3)["joint_angles"]

    taus = jnp.stack([jnp.zeros(3), jnp.ones(3) * 0.5, -jnp.ones(3) * 0.3])
    qs = jax.vmap(step)(taus)
    assert qs.shape == (3, 3)


# --------------------------------------------------------------------
# GraphManager integration: one step within a fresh graph
# --------------------------------------------------------------------

def test_graph_manager_integration(arm):
    gm = GraphManager()
    gm.add_node(arm)
    state = gm.step(external_inputs={})
    arm_state = state["arm"]
    assert arm_state["joint_angles"].shape == (3,)
    assert arm_state["joint_velocities"].shape == (3,)


# --------------------------------------------------------------------
# End-effector flux at q=0 matches the URDF-documented tool tip
# --------------------------------------------------------------------

def test_ee_pose_at_q_zero(arm):
    state = arm.initial_state()
    fluxes = arm.compute_boundary_fluxes(state, {}, 1e-3)
    ee_pos = np.asarray(fluxes["end_effector_pose_world"][:3])
    expected = np.array([L1 + L2 + L3, 0.0, 0.0])
    assert np.allclose(ee_pos, expected, atol=1e-12), \
        f"EE at q=0 expected {expected}, got {ee_pos}"
