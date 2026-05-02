"""Diagnose why the AR4 rotor isn't coupling to the helix dipole.

Compares the new chain (Motor → PermanentMagnetNode → response) at the
configured AR4 home pose against the legacy uniform-field (1.2 mT) that
dejongh_confined uses. Reports:

  1. Geometry: rotor world position, magnet body axis in world, helix
     onboard axis in world, and the rotor↔helix vector.
  2. Field magnitude at the helix from our PermanentMagnetNode (point-
     dipole), vs the legacy 1.2 mT uniform field.
  3. Torque magnitude on the helix from each.
  4. Spin axis vs helix axis alignment.
"""
from __future__ import annotations
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
sys.path.insert(0, "/home/nick/MSF/MIME/src")

import jax
import jax.numpy as jnp
import numpy as np

from mime.nodes.actuation.permanent_magnet import PermanentMagnetNode
from mime.nodes.actuation.motor import MotorNode

MU0_4PI = 1e-7  # μ0 / (4π)

# ── AR4 home-pose params (from physics/params.py) ───────────────────
ARM_HOME_RAD = (-0.02762, 0.23895, -0.77476, 1.55612, 1.54629, 0.00000)
BASE_POSE_WORLD = (-0.05, 0.328, -0.43, 1.0, 0.0, 0.0, 0.0)
END_EFFECTOR_OFFSET_IN_LINK = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
URDF_PATH = "/home/nick/MSF/MIME/experiments/ar4_helical_drive/assets/ar4.urdf"

MOTOR_AXIS_IN_PARENT = (0.0, 0.0, 1.0)        # EE-z is the spin axis
MAGNET_AXIS_IN_BODY = (1.0, 0.0, 0.0)         # rotor-body x = the dipole axis at θ=0
MAGNET_DIPOLE_A_M2 = 18.89                     # de Jongh N45 RPM
MAGNET_RADIUS_M = 17.5e-3
MAGNET_LENGTH_M = 20e-3

# Helix world state (matches setup.py seed).
INIT_POS = (0.0, -1e-3, 0.0)
INIT_ORIENT_WXYZ = (0.7071068, 0.0, 0.7071068, 0.0)  # +90° about world-y
HELIX_MOMENT_AXIS_BODY = (1.0, 0.0, 0.0)
N_HELIX_MAGNETS = 2
M_SINGLE_HELIX = 8.4e-4

# Legacy ExternalMagneticFieldNode comparison
LEGACY_B_T = 1.2e-3  # 1.2 mT uniform


def quat_to_R(q):
    """WXYZ quaternion → 3×3 rotation matrix."""
    w, x, y, z = q
    return jnp.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),         1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),         2*(y*z + w*x),       1 - 2*(x*x + y*y)],
    ])


def fk_ee_world():
    from mime.nodes.actuation.robot_arm import RobotArmNode
    from mime.control.kinematics.fk import joint_to_world_transforms
    from mime.control.kinematics.transform import (
        pose_to_matrix, _rotation_matrix_to_quat,
    )
    arm = RobotArmNode(
        name="arm", timestep=5e-4, urdf_path=URDF_PATH,
        end_effector_link_name="link_6",
        end_effector_offset_in_link=END_EFFECTOR_OFFSET_IN_LINK,
        base_pose_world=BASE_POSE_WORLD,
    )
    q = jnp.asarray(ARM_HOME_RAD)
    base_pose = jnp.asarray(BASE_POSE_WORLD)
    ee_offset = jnp.asarray(END_EFFECTOR_OFFSET_IN_LINK)
    link_xforms = joint_to_world_transforms(arm._tree, q)
    link_xforms = pose_to_matrix(base_pose)[None] @ link_xforms
    T_ee = link_xforms[arm._ee_idx] @ pose_to_matrix(ee_offset)
    pos = T_ee[:3, 3]
    R = T_ee[:3, :3]
    quat = _rotation_matrix_to_quat(R)
    return pos, R, quat


def main():
    # 1) EE / rotor / magnet world pose
    ee_pos, ee_R, ee_quat = fk_ee_world()
    print("=" * 70)
    print("GEOMETRY")
    print("=" * 70)
    print(f"EE world position:   {tuple(float(x) for x in ee_pos)}")
    print(f"EE-x world axis:     {tuple(float(x) for x in ee_R[:, 0])}")
    print(f"EE-y world axis:     {tuple(float(x) for x in ee_R[:, 1])}")
    print(f"EE-z world axis:     {tuple(float(x) for x in ee_R[:, 2])}  ← motor spin axis")

    # Motor: rotor pose = parent ⊕ R(axis, θ).  At θ=0 the rotor body
    # frame coincides with the parent frame.
    rotor_pos = ee_pos
    rotor_R = ee_R  # at θ=0 (snapshot)
    # Magnet body dipole at θ=0, in world:
    m_body = jnp.asarray(MAGNET_AXIS_IN_BODY)
    m_world = rotor_R @ m_body * MAGNET_DIPOLE_A_M2

    print(f"\nRotor world pos:     {tuple(float(x) for x in rotor_pos)}")
    print(f"Rotor magnet dipole world (θ=0): {tuple(float(x) for x in m_world)}")
    print(f"  |m| = {float(jnp.linalg.norm(m_world)):.3e} A·m²")

    # Helix
    helix_pos = jnp.asarray(INIT_POS)
    helix_R = quat_to_R(jnp.asarray(INIT_ORIENT_WXYZ))
    helix_moment_world = helix_R @ jnp.asarray(HELIX_MOMENT_AXIS_BODY) * (
        N_HELIX_MAGNETS * M_SINGLE_HELIX
    )
    print(f"\nHelix world position: {tuple(float(x) for x in helix_pos)}")
    print(f"Helix body-z (long) world: {tuple(float(x) for x in helix_R[:, 2])}  ← helix spin axis")
    print(f"Helix dipole world: {tuple(float(x) for x in helix_moment_world)}")

    r = helix_pos - rotor_pos
    print(f"\nrotor → helix vector: {tuple(float(x) for x in r)}")
    print(f"|r| = {float(jnp.linalg.norm(r))*1000:.2f} mm")

    # 2) Spin-axis ↔ helix-axis alignment
    align = float(jnp.dot(ee_R[:, 2], helix_R[:, 2]))
    print(f"\nspin_axis · helix_axis = {align:+.4f}  (1 = aligned, 0 = orthogonal)")

    # 3) Field at helix from rotor's permanent magnet (point dipole)
    print()
    print("=" * 70)
    print("FIELD AT HELIX")
    print("=" * 70)
    r_hat = r / jnp.linalg.norm(r)
    r3 = jnp.linalg.norm(r) ** 3
    B_dipole = MU0_4PI * (3 * jnp.dot(m_world, r_hat) * r_hat - m_world) / r3
    print(f"PermanentMagnetNode (point-dipole) B at helix:")
    print(f"  B = {tuple(float(x) for x in B_dipole)}")
    print(f"  |B| = {float(jnp.linalg.norm(B_dipole))*1e6:.3f} µT")
    print(f"  |B| = {float(jnp.linalg.norm(B_dipole))*1e3:.6f} mT")

    print(f"\nLegacy ExternalMagneticFieldNode uniform B:")
    print(f"  |B| = {LEGACY_B_T * 1e3:.3f} mT  (dejongh_confined default)")

    ratio = LEGACY_B_T / float(jnp.linalg.norm(B_dipole))
    print(f"\n>>> Legacy field is {ratio:.1f}× stronger than our point-dipole at this distance <<<")

    # 4) Peak torque magnitude on the helix
    print()
    print("=" * 70)
    print("TORQUE ON HELIX")
    print("=" * 70)
    tau_dipole = jnp.cross(helix_moment_world, B_dipole)
    print(f"Our chain  |τ| = {float(jnp.linalg.norm(tau_dipole)):.3e} N·m")
    # Legacy: τ = m × B with |B| = 1.2 mT, |m| = 1.68e-3, perpendicular case
    tau_legacy_max = float(jnp.linalg.norm(helix_moment_world)) * LEGACY_B_T
    print(f"Legacy peak |τ|= {tau_legacy_max:.3e} N·m  (when m ⊥ B)")

    # 5) Sweep an idealised rotor revolution to see the AC torque envelope
    print()
    print("=" * 70)
    print("ROTOR SWEEP (one revolution)")
    print("=" * 70)
    # Build the rotation about EE-z (motor spin axis = ee_R[:,2])
    spin_axis = ee_R[:, 2]
    thetas = jnp.linspace(0, 2 * jnp.pi, 9)
    print(f"{'theta_deg':>10}  {'|B|µT':>10}  {'|τ|_Nm':>12}  {'τ_z':>12}")
    for th in thetas:
        # Rodrigues
        K = jnp.array([
            [0, -spin_axis[2], spin_axis[1]],
            [spin_axis[2], 0, -spin_axis[0]],
            [-spin_axis[1], spin_axis[0], 0],
        ])
        Rspin = jnp.eye(3) + jnp.sin(th) * K + (1 - jnp.cos(th)) * (K @ K)
        m_w = (Rspin @ rotor_R) @ m_body * MAGNET_DIPOLE_A_M2
        Bd = MU0_4PI * (3 * jnp.dot(m_w, r_hat) * r_hat - m_w) / r3
        td = jnp.cross(helix_moment_world, Bd)
        print(f"{float(jnp.degrees(th)):>10.1f}  {float(jnp.linalg.norm(Bd))*1e6:>10.3f}  "
              f"{float(jnp.linalg.norm(td)):>12.3e}  {float(td[2]):>+12.3e}")


if __name__ == "__main__":
    main()
