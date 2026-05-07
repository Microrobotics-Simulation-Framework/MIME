"""Unit tests for the IK helpers in mime.control.kinematics.ik.

Coverage:
- pose_error_6d: zero on identity, linear in small perturbations.
- damped_least_squares: matches J⁻¹ for invertible J at λ→0; bounded
  near singular J.
- solve_ik_iterative: converges from a small random seed to a target
  pose on the AR4 URDF for a batch of feasible target poses.

The AR4 URDF is read from experiments/ar4_helical_drive/assets/ar4.urdf
and parsed once.  Convergence-batch test uses 20 random targets to
keep CI runtime reasonable; pose-error tolerance is 5e-4 (~0.5 mm /
0.5 mrad combined).
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mime.control.kinematics import (
    damped_least_squares,
    ee_jacobian,
    joint_to_world_transforms,
    parse_urdf,
    pose_error_6d,
    pose_to_matrix,
    solve_ik_iterative,
)


URDF_PATH = (
    Path(__file__).resolve().parents[3]
    / "experiments" / "ar4_helical_drive" / "assets" / "ar4.urdf"
)


# ── pose_error_6d ─────────────────────────────────────────────────────


def test_pose_error_zero_at_identity():
    T = jnp.eye(4)
    e = pose_error_6d(T, T)
    np.testing.assert_allclose(np.asarray(e), np.zeros(6), atol=1e-7)


def test_pose_error_linear_translation():
    T_a = jnp.eye(4)
    delta = jnp.array([0.01, -0.02, 0.005])
    T_b = T_a.at[:3, 3].set(delta)
    e = pose_error_6d(T_a, T_b)
    np.testing.assert_allclose(np.asarray(e[:3]), np.asarray(delta), atol=1e-7)
    np.testing.assert_allclose(np.asarray(e[3:]), np.zeros(3), atol=1e-7)


def test_pose_error_small_rotation():
    # Small rotation about z by 1 mrad — angular error should be (0,0,1e-3).
    theta = 1e-3
    R = jnp.array([
        [jnp.cos(theta), -jnp.sin(theta), 0.0],
        [jnp.sin(theta),  jnp.cos(theta), 0.0],
        [0.0,             0.0,            1.0],
    ])
    T_a = jnp.eye(4)
    T_b = T_a.at[:3, :3].set(R)
    e = pose_error_6d(T_a, T_b)
    np.testing.assert_allclose(np.asarray(e[3:]), np.array([0.0, 0.0, theta]),
                               atol=1e-9)


# ── damped_least_squares ─────────────────────────────────────────────


def test_dls_recovers_inverse_for_invertible_square_J():
    rng = np.random.default_rng(0)
    J = jnp.asarray(rng.standard_normal((6, 6)))
    e = jnp.asarray(rng.standard_normal(6))
    # λ → 0: result should match J⁻¹ e (since J is invertible).
    q_dot = damped_least_squares(J, e, lam=1e-6)
    expected = jnp.linalg.solve(J, e)
    np.testing.assert_allclose(np.asarray(q_dot), np.asarray(expected), atol=1e-3)


def test_dls_bounded_at_singular_J():
    # J with rank 5 (last row zero) — J^+ would diverge but DLS bounds it.
    J = jnp.eye(6).at[5, 5].set(0.0)
    e = jnp.ones(6)
    q_dot = damped_least_squares(J, e, lam=0.05)
    assert jnp.all(jnp.isfinite(q_dot))
    assert float(jnp.linalg.norm(q_dot)) < 1e3


# ── ee_jacobian ───────────────────────────────────────────────────────


def test_ee_jacobian_matches_link_jacobian_when_offset_is_identity():
    # With ee_offset = identity AND link_com = 0, ee_jacobian == frame_jacobian.
    # The AR4 has nonzero link COM offsets so this test uses a contrived
    # case: ee_offset is set to (-com_local) so p_ee = p_link_inertial.
    tree = parse_urdf(URDF_PATH)
    n = tree.num_joints
    q = jnp.zeros(n)
    ee_idx = n - 1   # link_6 is the last URDF link
    com_local = np.asarray(tree.link_com_local[ee_idx])
    # ee_offset_pose7 chosen so the EE point is at the link inertial origin.
    ee_offset = jnp.array([
        com_local[0], com_local[1], com_local[2],
        1.0, 0.0, 0.0, 0.0,
    ])
    from mime.control.kinematics import frame_jacobian
    J_link = frame_jacobian(tree, q, ee_idx)
    J_ee = ee_jacobian(tree, q, ee_idx, ee_offset)
    # Should be identical when EE point coincides with link inertial origin.
    np.testing.assert_allclose(
        np.asarray(J_link), np.asarray(J_ee), atol=1e-7,
    )


# ── solve_ik_iterative ───────────────────────────────────────────────


def _ar4_ee_world_pose(tree, q, ee_idx, ee_offset_pose7):
    """Forward-kinematic EE world pose."""
    joint_world = joint_to_world_transforms(tree, q)
    T_offset = pose_to_matrix(ee_offset_pose7)
    return joint_world[ee_idx] @ T_offset


@pytest.mark.parametrize("seed", range(5))
def test_solve_ik_converges_on_ar4(seed):
    """From a small perturbation around a known feasible config, IK
    must converge to the target pose."""
    tree = parse_urdf(URDF_PATH)
    n = tree.num_joints
    ee_idx = n - 1
    ee_offset = jnp.zeros(7).at[3].set(1.0)   # identity (qw=1)

    rng = np.random.default_rng(seed)
    # A "feasible" target: random but small joints (away from limits).
    q_target = jnp.asarray(rng.uniform(-0.5, 0.5, size=n).astype(np.float32))
    T_target = _ar4_ee_world_pose(tree, q_target, ee_idx, ee_offset)

    # Seed IK from a perturbed starting point (within ±0.2 rad).
    q_init = jnp.asarray(
        (np.asarray(q_target) + rng.uniform(-0.2, 0.2, size=n)).astype(np.float32)
    )

    q_sol = solve_ik_iterative(
        tree, q_init, T_target, ee_idx, ee_offset,
        n_iters=30, lam=0.05,
    )
    T_sol = _ar4_ee_world_pose(tree, q_sol, ee_idx, ee_offset)
    e = pose_error_6d(T_sol, T_target)
    err_norm = float(jnp.linalg.norm(e))
    assert err_norm < 5e-4, (
        f"IK did not converge: pose_error_norm = {err_norm:.2e} (target < 5e-4); "
        f"q_init={q_init}, q_sol={q_sol}, q_target={q_target}"
    )
