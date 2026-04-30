"""Tests for ``mime.control.kinematics`` (Wave A.3).

Covers:
  * URDF parse round-trip on the 3-link planar fixture.
  * Forward kinematics at q=0 and q=[π/2, 0, 0].
  * FK against analytical 3-link planar formula at random configs.
  * Mass matrix symmetry + positive definiteness.
  * RNEA self-consistency: τ_round_trip ≡ τ_input.
  * jit / grad / vmap traceability of FK.
  * Fixed-joint merging — a 3-link arm with a fixed joint inserted between
    revolutes should yield the same FK as the simple 3-link arm.
"""

from __future__ import annotations

import os
import textwrap

import numpy as np
import jax

# Enable double precision for round-trip dynamics tests. JAX defaults to
# float32 which gives RNEA round-trip errors around 1e-6. URDF parsing and
# downstream consumers (Wave B's RobotArmNode) should configure precision
# at simulation start; the tests here pin float64 explicitly.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from mime.control.kinematics import (
    parse_urdf,
    KinematicTree,
    ancestor_mask,
    joint_to_world_transforms,
    link_to_world_transforms,
    link_world_poses,
    frame_jacobian,
    mass_matrix,
    rnea,
    gravity_vector,
    nonlinear_bias,
    JOINT_TYPE_REVOLUTE,
)


FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "three_link_planar.urdf"
)

# Link lengths from the fixture
L1, L2, L3 = 1.0, 1.0, 0.5


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------


def test_parse_three_link_planar():
    tree = parse_urdf(FIXTURE)
    assert isinstance(tree, KinematicTree)
    assert tree.num_joints == 3
    assert tree.joint_names == ("joint_1", "joint_2", "joint_3")
    assert tree.link_names == ("link_1", "link_2", "link_3")
    assert tree.root_link_name == "base_link"
    # All revolute about z
    assert np.all(tree.joint_types == JOINT_TYPE_REVOLUTE)
    np.testing.assert_allclose(
        tree.joint_axes_local, np.tile(np.array([0, 0, 1.0]), (3, 1))
    )
    # Parent chain
    np.testing.assert_array_equal(tree.parent_idxs, np.array([-1, 0, 1]))
    # Damping read from <dynamics>
    np.testing.assert_allclose(tree.joint_friction, [0.05, 0.05, 0.02])
    # Limits
    np.testing.assert_allclose(
        tree.joint_limits[:, 0], [-3.1416, -3.1416, -3.1416]
    )


def test_ancestor_mask_three_link():
    tree = parse_urdf(FIXTURE)
    M = ancestor_mask(tree.parent_idxs)
    expected = np.array(
        [
            [True, False, False],
            [True, True, False],
            [True, True, True],
        ]
    )
    np.testing.assert_array_equal(M, expected)


# ---------------------------------------------------------------------------
# Forward kinematics — qualitative
# ---------------------------------------------------------------------------


def test_fk_q_zero():
    tree = parse_urdf(FIXTURE)
    q = jnp.zeros(3)
    joint_world = joint_to_world_transforms(tree, q)
    # Joint origins at q=0: (0,0,0), (1,0,0), (2,0,0)
    expected_pos = np.array([[0, 0, 0], [L1, 0, 0], [L1 + L2, 0, 0]])
    np.testing.assert_allclose(
        np.asarray(joint_world[:, :3, 3]), expected_pos, atol=1e-12
    )
    # All rotations identity
    for i in range(3):
        np.testing.assert_allclose(
            np.asarray(joint_world[i, :3, :3]), np.eye(3), atol=1e-12
        )

    # Link COMs at L1/2, L1+L2/2, L1+L2+L3/2 along x
    link_world = link_to_world_transforms(tree, q)
    expected_com = np.array(
        [[0.5, 0, 0], [L1 + 0.5, 0, 0], [L1 + L2 + 0.25, 0, 0]]
    )
    np.testing.assert_allclose(
        np.asarray(link_world[:, :3, 3]), expected_com, atol=1e-12
    )


def test_fk_q_first_joint_90():
    tree = parse_urdf(FIXTURE)
    q = jnp.array([jnp.pi / 2, 0.0, 0.0])
    joint_world = joint_to_world_transforms(tree, q)
    # Rotating joint_1 by π/2 swings the entire chain to +y.
    np.testing.assert_allclose(
        np.asarray(joint_world[0, :3, 3]), np.zeros(3), atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(joint_world[1, :3, 3]), np.array([0, L1, 0]), atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(joint_world[2, :3, 3]),
        np.array([0, L1 + L2, 0]),
        atol=1e-6,
    )


def _planar_fk_analytical(q):
    """End-effector (tip of link_3) position for a 3-link planar arm.

    Joints rotate about z, links along local +x with lengths L1, L2, L3.
    EE x = L1·cos(q1) + L2·cos(q1+q2) + L3·cos(q1+q2+q3),  y = sin(...).
    """
    q1, q2, q3 = q
    a1 = q1
    a12 = q1 + q2
    a123 = q1 + q2 + q3
    x = L1 * np.cos(a1) + L2 * np.cos(a12) + L3 * np.cos(a123)
    y = L1 * np.sin(a1) + L2 * np.sin(a12) + L3 * np.sin(a123)
    return np.array([x, y, 0.0])


def test_fk_against_analytical():
    tree = parse_urdf(FIXTURE)
    rng = np.random.default_rng(0)
    for _ in range(5):
        q = rng.uniform(-np.pi, np.pi, size=3)
        joint_world = np.asarray(joint_to_world_transforms(tree, jnp.array(q)))
        # End-effector position = joint_3 origin + R_3 @ [L3, 0, 0]
        ee = joint_world[2, :3, 3] + joint_world[2, :3, :3] @ np.array([L3, 0, 0])
        np.testing.assert_allclose(ee, _planar_fk_analytical(q), atol=1e-6)


def test_link_world_poses_shape_and_translations():
    tree = parse_urdf(FIXTURE)
    q = jnp.zeros(3)
    poses = link_world_poses(tree, q)
    assert poses.shape == (3, 7)
    expected_t = np.array(
        [[0.5, 0, 0], [L1 + 0.5, 0, 0], [L1 + L2 + 0.25, 0, 0]]
    )
    np.testing.assert_allclose(np.asarray(poses[:, :3]), expected_t, atol=1e-12)
    # WXYZ identity quaternion
    np.testing.assert_allclose(
        np.asarray(poses[:, 3:]),
        np.tile(np.array([1, 0, 0, 0]), (3, 1)),
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Mass matrix
# ---------------------------------------------------------------------------


def test_mass_matrix_symmetry_and_pd():
    tree = parse_urdf(FIXTURE)
    rng = np.random.default_rng(42)
    for _ in range(5):
        q = jnp.array(rng.uniform(-np.pi, np.pi, size=3))
        M = np.asarray(mass_matrix(tree, q))
        # Symmetry
        sym_err = np.linalg.norm(M - M.T, ord="fro")
        assert sym_err < 1e-10, f"M not symmetric, ‖M-Mᵀ‖_F = {sym_err}"
        # Positive definite
        eigvals = np.linalg.eigvalsh(0.5 * (M + M.T))
        assert np.all(eigvals > 0), f"M has non-positive eigenvalue: {eigvals}"


# ---------------------------------------------------------------------------
# RNEA self-consistency
# ---------------------------------------------------------------------------


def test_rnea_self_consistency():
    """τ → q̈ via M⁻¹(τ - bias) → τ_round_trip via RNEA must round-trip."""
    tree = parse_urdf(FIXTURE)
    g_world = jnp.array([0.0, 0.0, -9.81])
    rng = np.random.default_rng(7)
    for _ in range(3):
        q = jnp.array(rng.uniform(-1.0, 1.0, size=3))
        qd = jnp.array(rng.uniform(-1.0, 1.0, size=3))
        tau = jnp.array(rng.uniform(-1.0, 1.0, size=3))

        M = np.asarray(mass_matrix(tree, q))
        bias = np.asarray(nonlinear_bias(tree, q, qd, g_world))
        # Solve via numpy (avoids cuSolver init flakiness on some GPU rigs;
        # the correctness of M and bias is what's under test, not the solver).
        qdd_np = np.linalg.solve(M, np.asarray(tau) - bias)
        qdd = jnp.asarray(qdd_np)

        tau_round = rnea(tree, q, qd, qdd, g_world)
        err = float(jnp.linalg.norm(tau_round - tau))
        assert err < 1e-9, f"RNEA round-trip error {err} exceeds 1e-9"


def test_gravity_vector_consistency():
    """gravity_vector(q) ≡ rnea(q, qd=0, qdd=0, g)."""
    tree = parse_urdf(FIXTURE)
    g_world = jnp.array([0.0, 0.0, -9.81])
    q = jnp.array([0.3, -0.4, 0.1])
    g_a = gravity_vector(tree, q, g_world)
    g_b = rnea(tree, q, qd=None, qdd=None, gravity_world=g_world)
    np.testing.assert_allclose(np.asarray(g_a), np.asarray(g_b), atol=1e-12)


# ---------------------------------------------------------------------------
# Frame Jacobian sanity
# ---------------------------------------------------------------------------


def test_frame_jacobian_against_finite_difference():
    """Compare J for link_2 against a finite-difference of FK position."""
    tree = parse_urdf(FIXTURE)
    q0 = jnp.array([0.2, -0.3, 0.5])
    eps = 1e-6
    J = np.asarray(frame_jacobian(tree, q0, link_idx=2))

    def link_pos(q):
        link_world = link_to_world_transforms(tree, q)
        return link_world[2, :3, 3]

    p0 = np.asarray(link_pos(q0))
    Jv_fd = np.zeros((3, 3))
    for i in range(3):
        dq = np.zeros(3)
        dq[i] = eps
        p_plus = np.asarray(link_pos(q0 + dq))
        Jv_fd[:, i] = (p_plus - p0) / eps
    # Compare linear (top 3 rows of J) against finite diff
    np.testing.assert_allclose(J[:3, :], Jv_fd, atol=1e-4)


# ---------------------------------------------------------------------------
# JAX traceability
# ---------------------------------------------------------------------------


def test_fk_jit_traceable():
    tree = parse_urdf(FIXTURE)

    @jax.jit
    def fk_tip(q):
        joint_world = joint_to_world_transforms(tree, q)
        return joint_world[2, :3, 3] + joint_world[2, :3, :3] @ jnp.array([L3, 0, 0])

    q = jnp.array([0.1, 0.2, 0.3])
    out = fk_tip(q)
    np.testing.assert_allclose(np.asarray(out), _planar_fk_analytical(q), atol=1e-6)


def test_fk_grad_traceable():
    tree = parse_urdf(FIXTURE)

    def squared_x(q):
        joint_world = joint_to_world_transforms(tree, q)
        ee = joint_world[2, :3, 3] + joint_world[2, :3, :3] @ jnp.array([L3, 0, 0])
        return ee[0]

    g = jax.grad(squared_x)(jnp.array([0.1, 0.2, 0.3]))
    # Just check it runs and returns a finite (3,) gradient
    assert g.shape == (3,)
    assert np.all(np.isfinite(np.asarray(g)))


def test_fk_vmap_traceable():
    tree = parse_urdf(FIXTURE)

    def fk_tip(q):
        joint_world = joint_to_world_transforms(tree, q)
        return joint_world[2, :3, 3] + joint_world[2, :3, :3] @ jnp.array([L3, 0, 0])

    qs = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [jnp.pi / 2, 0.0, 0.0],
            [0.1, -0.2, 0.3],
        ]
    )
    out = jax.vmap(fk_tip)(qs)
    expected = np.stack([_planar_fk_analytical(np.asarray(q)) for q in qs])
    np.testing.assert_allclose(np.asarray(out), expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Fixed-joint merging
# ---------------------------------------------------------------------------


@pytest.fixture
def fixed_joint_urdf(tmp_path):
    """A 3-link planar arm with a *fixed* joint inserted between joint_2 and
    joint_3. The fixed joint adds a 0.5m offset along x, and joint_3 sits on
    a new intermediate link (mass 0). The merged tree must still have 3 DOFs
    and the FK at the same q must match a "shifted" 3-link arm where
    L2_effective = L2 + 0.5 = 1.5.
    """
    urdf = textwrap.dedent(
        """\
        <?xml version="1.0"?>
        <robot name="fixed_joint_arm">
          <link name="base_link"/>

          <joint name="joint_1" type="revolute">
            <parent link="base_link"/><child link="link_1"/>
            <origin xyz="0 0 0"/><axis xyz="0 0 1"/>
            <limit lower="-3.14" upper="3.14" effort="10" velocity="1"/>
          </joint>
          <link name="link_1">
            <inertial>
              <origin xyz="0.5 0 0"/>
              <mass value="2.0"/>
              <inertia ixx="0" iyy="0.1666667" izz="0.1666667"
                       ixy="0" ixz="0" iyz="0"/>
            </inertial>
          </link>

          <joint name="joint_2" type="revolute">
            <parent link="link_1"/><child link="link_2"/>
            <origin xyz="1 0 0"/><axis xyz="0 0 1"/>
            <limit lower="-3.14" upper="3.14" effort="10" velocity="1"/>
          </joint>
          <link name="link_2">
            <inertial>
              <origin xyz="0.5 0 0"/>
              <mass value="1.5"/>
              <inertia ixx="0" iyy="0.125" izz="0.125"
                       ixy="0" ixz="0" iyz="0"/>
            </inertial>
          </link>

          <joint name="fixed_extension" type="fixed">
            <parent link="link_2"/><child link="link_2_ext"/>
            <origin xyz="0.5 0 0"/>
          </joint>
          <link name="link_2_ext">
            <inertial>
              <origin xyz="0 0 0"/>
              <mass value="0.0"/>
              <inertia ixx="0" iyy="0" izz="0"
                       ixy="0" ixz="0" iyz="0"/>
            </inertial>
          </link>

          <joint name="joint_3" type="revolute">
            <parent link="link_2_ext"/><child link="link_3"/>
            <origin xyz="0.5 0 0"/><axis xyz="0 0 1"/>
            <limit lower="-3.14" upper="3.14" effort="10" velocity="1"/>
          </joint>
          <link name="link_3">
            <inertial>
              <origin xyz="0.25 0 0"/>
              <mass value="0.5"/>
              <inertia ixx="0" iyy="0.0104167" izz="0.0104167"
                       ixy="0" ixz="0" iyz="0"/>
            </inertial>
          </link>
        </robot>
        """
    )
    p = tmp_path / "fixed_joint.urdf"
    p.write_text(urdf)
    return str(p)


def test_fixed_joint_merging_dof_count(fixed_joint_urdf):
    tree = parse_urdf(fixed_joint_urdf)
    # 4 declared joints, 1 fixed → merged → 3 DOFs.
    assert tree.num_joints == 3
    assert "fixed_extension" not in tree.joint_names
    assert tree.joint_names == ("joint_1", "joint_2", "joint_3")
    # link_2_ext should have been merged into link_2.
    assert "link_2_ext" not in tree.link_names


def test_fixed_joint_merging_fk(fixed_joint_urdf):
    """FK with merged fixed joint should match a simple 3-link arm.

    In ``fixed_joint_urdf``: joint_1 origin = (0,0,0), joint_2 origin = (1,0,0),
    fixed_extension origin = (0.5,0,0), joint_3 origin = (0.5,0,0). After
    merging the fixed joint, joint_3's origin (in link_1's joint_2 frame)
    becomes (0.5,0,0) preceded by (0.5,0,0) → (1.0, 0, 0). The effective
    arm geometry is therefore L1=1.0, L2=1.0, L3=0.5 — the same as the
    plain three_link_planar fixture.
    """
    tree = parse_urdf(fixed_joint_urdf)
    rng = np.random.default_rng(11)
    L1_eff, L2_eff, L3_eff = 1.0, 1.0, 0.5
    for _ in range(3):
        q = rng.uniform(-1.0, 1.0, size=3)
        joint_world = np.asarray(joint_to_world_transforms(tree, jnp.array(q)))
        ee = joint_world[2, :3, 3] + joint_world[2, :3, :3] @ np.array([L3_eff, 0, 0])
        # Analytical
        a1 = q[0]
        a12 = q[0] + q[1]
        a123 = q[0] + q[1] + q[2]
        ee_a = np.array(
            [
                L1_eff * np.cos(a1) + L2_eff * np.cos(a12) + L3_eff * np.cos(a123),
                L1_eff * np.sin(a1) + L2_eff * np.sin(a12) + L3_eff * np.sin(a123),
                0.0,
            ]
        )
        np.testing.assert_allclose(ee, ee_a, atol=1e-6)
