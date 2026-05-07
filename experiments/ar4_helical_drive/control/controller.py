"""Live actuation control for the AR4 + helical-UMR drive experiment.

Each tick the runner calls
``get_external_inputs(params, step_count, state=prev_full_state)``
(state is optional — runner falls back to the legacy 2-arg signature
if absent).  We emit:

* ``motor.commanded_velocity`` — angular velocity the rotor PI loop
  tracks. Live-editable via ``FIELD_FREQUENCY_HZ`` in params.
* ``arm.commanded_joint_torques`` — joint-space PD that drives the
  AR4 EE to a target pose.

M3+M4: closed-loop **position + orientation** tracking.

Position: target EE pose translation = ``(body.x, 0, STANDOFF_M)``,
low-pass filtered (``CONTROL_TARGET_ALPHA``).

Orientation: target EE rotation has its z-axis aligned with the
body's z-axis (helix long axis), reading
``state["body"]["orientation"]``. Roll about the spin axis is set
by Gram-Schmidt against the home-pose EE-x, keeping the wrist roll
close to home.  Filter is a slerp toward the target quaternion at
``CONTROL_TARGET_ALPHA``.

Set ``ENABLE_ORIENTATION_FEEDBACK = False`` in params to revert to
M3 (orientation held at home) — useful for studying the underlying
elliptical-field tilt instability without controller suppression.

Architecture: a **single JIT'd dispatch per physics step** that
combines differential IK with computed-torque PD:

  q_target ← q + DLS( J_ee(q),  K_ik · pose_error( T_ee(q), T_target ) )
  τ        ← M(q) · [K_p (q_target − q) − K_d q̇] + Coriolis(q, q̇)

Properties:

* No periodic Newton-IK — the EE moves smoothly toward the target
  every step instead of teleporting in q-space every N steps. This
  removes the joint-target jitter caused by step-changes in q_target.
* One XLA dispatch per call (FK + Jacobian + DLS + CRBA + RNEA in
  the same trace). Sub-millisecond per step on GPU after warm cache.
* Robust near singularities: the DLS regulariser λ caps |q̇| even
  when the Jacobian is ill-conditioned.

Plain joint-space PD is unstable on the AR4 because the wrist-roll
joint's effective inertia is ≈ 2.4×10⁻⁶ kg·m² — the explicit-Euler
stability bound `K_p · dt² < I_min` makes any usable gain
catastrophic on that joint. The IDPD law inverts the mass-matrix
coupling so the effective dynamics are joint-decoupled
`q̈ = K_p (q_t - q) - K_d q̇`. Gravity is handled inside
RobotArmNode (auto_gravity_compensation=True), and Coriolis is the
residual of the bias term once gravity is removed.

Gains (params.py):

* ``CONTROL_K_P``, ``CONTROL_K_D`` — IDPD natural freq + damping.
* ``CONTROL_K_IK`` — task-space velocity gain on the differential
  IK step.
* ``CONTROL_IK_LAMBDA`` — DLS regulariser.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mime.control.kinematics import (
    damped_least_squares,
    ee_jacobian,
    joint_to_world_transforms,
    mass_matrix,
    nonlinear_bias,
    parse_urdf,
    pose_error_6d,
    pose_to_matrix,
)
from mime.control.kinematics.transform import (
    _quat_to_rotation_matrix,
    _rotation_matrix_to_quat,
)


# ── Controller singleton ─────────────────────────────────────────────


class _TrackingController:
    """Encapsulates the kinematic tree, gains, and per-call state."""

    def __init__(self, params: dict):
        urdf_path_rel = Path(params["URDF_PATH"])
        if urdf_path_rel.is_absolute():
            urdf_path = urdf_path_rel
        else:
            # Resolve relative to this experiment directory.
            urdf_path = Path(__file__).resolve().parents[1] / urdf_path_rel
        self.tree = parse_urdf(urdf_path)
        self.n_dof = self.tree.num_joints
        self.q_home = jnp.asarray(params["ARM_HOME_RAD"], dtype=jnp.float32)
        self.ee_link_idx = self.n_dof - 1   # link_6 is the last joint
        self.ee_offset_pose7 = jnp.asarray(
            params.get("END_EFFECTOR_OFFSET_IN_LINK",
                       (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
            dtype=jnp.float32,
        )
        self.base_pose7 = jnp.asarray(
            params.get("BASE_POSE_WORLD",
                       (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
            dtype=jnp.float32,
        )
        # Per-joint gains in 1/s² (K_p) and 1/s (K_d) — they shape
        # the effective decoupled dynamics q̈ = K_p (q_t - q) - K_d q̇.
        # 100 / 20 gives a 10-rad/s natural frequency (1.6 Hz) with
        # damping ratio 1.0 (critically damped). Same per joint via
        # the M(q)-multiplied IDPD law.
        self.K_p = float(params.get("CONTROL_K_P", 100.0))
        self.K_d = float(params.get("CONTROL_K_D", 20.0))
        self.gravity_world = jnp.asarray(
            params.get("GRAVITY_WORLD", (0.0, 0.0, -9.80665)),
            dtype=jnp.float32,
        )
        # Damped-LS regularisation for the IK Newton step.
        self.lam = float(params.get("CONTROL_IK_LAMBDA", 0.05))
        # Standoff (m) of the EE above the tube. AR4 reach maxes out
        # at ~0.20 m above the tube origin given the desk geometry.
        self.standoff_m = float(params.get("CONTROL_STANDOFF_M", 0.20))
        # World-x reach envelope (M5 safety clamp).
        self.x_min_m = float(params.get("CONTROL_X_MIN_M", -0.05))
        self.x_max_m = float(params.get("CONTROL_X_MAX_M", +0.05))
        # Low-pass filter on the target translation: alpha is the
        # blend with the new sample (1 = no filter, 0 = frozen).
        self.target_alpha = float(params.get("CONTROL_TARGET_ALPHA", 0.05))
        # Orientation tracking. Set False to keep the rotor's spin
        # axis fixed at world-x (M3 behaviour) — useful to study the
        # elliptical-field tilt instability without controller
        # suppression.
        self.enable_orientation_feedback = bool(
            params.get("ENABLE_ORIENTATION_FEEDBACK", True)
        )
        # Velocity feedforward: project body orientation forward by
        # this much (seconds) using its angular velocity, so the
        # controller's IDPD steady-state lag aligns with the
        # prediction. ≈ IDPD time constant (1/sqrt(K_p)) is the
        # right scale: τ·ω of lag becomes zero. Set to 0 to disable.
        self.lookahead_s = float(params.get("CONTROL_LOOKAHEAD_S", 0.010))

        # Cache the home EE *orientation* in base frame — that's
        # what M3 holds fixed while the position tracks the body.
        T_base = pose_to_matrix(self.base_pose7)
        T_base_inv = jnp.linalg.inv(T_base)
        self._T_base = T_base
        self._T_base_inv = T_base_inv
        T_offset = pose_to_matrix(self.ee_offset_pose7)
        joint_world = joint_to_world_transforms(self.tree, self.q_home)
        T_home_ee_world = T_base @ joint_world[self.ee_link_idx] @ T_offset
        self.R_home_world = T_home_ee_world[:3, :3]
        # Filtered target translation and orientation (world frame).
        # Initialised to the home-pose EE pose to avoid step inputs.
        self._target_pos_world = T_home_ee_world[:3, 3]
        self._target_quat_world = _rotation_matrix_to_quat(self.R_home_world)
        # Cached target in world frame and base frame, refreshed each
        # call.
        self.T_target_world = T_home_ee_world
        self.T_target_base = T_base_inv @ T_home_ee_world
        # Task-space P gain: how aggressively the differential-IK
        # step reduces the EE pose error each physics step. Kept
        # modest (10 1/s) so the resulting joint-space target step
        # is small and the IDPD law tracks it without overshoot.
        self.K_ik = float(params.get("CONTROL_K_IK", 10.0))

        # JIT-compile the entire per-step law: differential IK (one
        # Jacobian-pseudo-inverse step toward the target pose) +
        # IDPD (computed-torque). Single XLA dispatch per physics
        # step — sub-millisecond on GPU after warm cache.  The
        # static kinematic tree is closure-captured.
        from mime.control.kinematics.crba import mass_matrix as _mm
        from mime.control.kinematics.rnea import (
            nonlinear_bias as _nb, gravity_vector as _gv,
        )
        tree_static = self.tree
        ee_idx_static = self.ee_link_idx
        ee_offset_static = self.ee_offset_pose7
        T_base_static = self._T_base

        ik_iters_static = int(params.get("CONTROL_IK_INNER_ITERS", 5))

        @jax.jit
        def _compute_torque(
            q, qd, T_target_world, K_p, K_d, K_ik, lam, g_world,
        ):
            # Multi-iter Newton DLS: each iter recomputes the
            # Jacobian and pose error at the latest q_target. With 5
            # iters re-seeded from q (current state), large rotation
            # gaps (e.g. helix tumbling at 90°/100ms) are tracked
            # without lag. Single-step DLS would lag because the
            # linear approximation breaks down for big rotations.
            R_base = T_base_static[:3, :3]
            R_base_inv = R_base.T
            block = jnp.zeros((6, 6), dtype=q.dtype)
            block = block.at[:3, :3].set(R_base_inv)
            block = block.at[3:, 3:].set(R_base_inv)
            T_offset = pose_to_matrix(ee_offset_static)

            def _ik_body(_, q_iter):
                jw = joint_to_world_transforms(tree_static, q_iter)
                T_ee_w = T_base_static @ jw[ee_idx_static] @ T_offset
                e = pose_error_6d(T_ee_w, T_target_world)
                e_base = block @ e
                J_base = ee_jacobian(
                    tree_static, q_iter, ee_idx_static, ee_offset_static,
                )
                dq = damped_least_squares(J_base, K_ik * e_base, lam)
                return q_iter + dq

            q_target = jax.lax.fori_loop(0, ik_iters_static, _ik_body, q)

            # IDPD on the converged q_target.
            qdd_des = K_p * (q_target - q) - K_d * qd
            M = _mm(tree_static, q)
            bias = _nb(tree_static, q, qd, g_world)
            coriolis = bias - _gv(tree_static, q, g_world)
            tau = M @ qdd_des + coriolis

            # Forward kinematics to current EE world pose for return.
            jw0 = joint_to_world_transforms(tree_static, q)
            T_ee_world = T_base_static @ jw0[ee_idx_static] @ T_offset
            return tau, q_target, T_ee_world

        self._compute_torque = _compute_torque

    def _update_target_from_body(
        self,
        body_pos: jnp.ndarray,
        body_orient_quat: jnp.ndarray,
        body_angvel_world: jnp.ndarray,
    ) -> None:
        """Refresh the world-frame target pose from the body's current
        pose.  Translation low-pass filtered, orientation slerp-filtered.

        With ``self.lookahead_s > 0`` the body orientation is projected
        forward by that many seconds using its angular velocity. The
        IDPD's steady-state lag (τ·ω) then aligns with the prediction
        lookahead — net tracking error is zero at steady-state ω.
        """
        # ── Translation: track body.x at fixed standoff ───────────
        # Clamp target.x to the AR4's reach envelope.  Beyond this
        # range the IK can't converge without driving joints into
        # limits or singularities; the rotor saturates at the edge
        # while the helix swims past.
        x_safe = jnp.clip(body_pos[0], self.x_min_m, self.x_max_m)
        target_pos_raw = jnp.array([
            x_safe, 0.0, self.standoff_m,
        ], dtype=jnp.float32)
        a = self.target_alpha
        self._target_pos_world = (
            a * target_pos_raw + (1.0 - a) * self._target_pos_world
        )

        # ── Orientation ────────────────────────────────────────────
        if self.enable_orientation_feedback:
            # Project body orientation forward by lookahead_s using
            # its angular velocity (small-angle approx — fine since
            # ω·lookahead is at most a few degrees).
            #   R_body_future ≈ (I + skew(ω) · Δt) @ R_body_now
            R_body_now = _quat_to_rotation_matrix(body_orient_quat)
            wx, wy, wz = body_angvel_world[0], body_angvel_world[1], body_angvel_world[2]
            tau_la = jnp.float32(self.lookahead_s)
            skew_w = jnp.array([
                [   0.0, -wz,  wy],
                [   wz,   0.0, -wx],
                [  -wy,  wx,    0.0],
            ], dtype=jnp.float32) * tau_la
            R_body = (jnp.eye(3, dtype=jnp.float32) + skew_w) @ R_body_now
            target_z = R_body[:, 2]
            # Re-normalise (small-angle approx isn't a true rotation).
            target_z = target_z / jnp.maximum(jnp.linalg.norm(target_z), 1e-12)
            home_x = self.R_home_world[:, 0]
            target_x = home_x - jnp.dot(home_x, target_z) * target_z
            target_x_norm = jnp.linalg.norm(target_x)
            target_x = jnp.where(
                target_x_norm > 1e-6,
                target_x / jnp.maximum(target_x_norm, 1e-30),
                # Singularity: home-x parallel to body-z. Fall back to
                # home-y as the seed direction.
                self.R_home_world[:, 1],
            )
            target_y = jnp.cross(target_z, target_x)
            R_target_raw = jnp.stack([target_x, target_y, target_z], axis=1)
            target_quat_raw = _rotation_matrix_to_quat(R_target_raw)
            # Resolve quaternion sign ambiguity (q and -q represent
            # the same rotation) — pick the one closer to the
            # filtered target so slerp doesn't take the long way.
            sign = jnp.where(
                jnp.dot(self._target_quat_world, target_quat_raw) < 0.0,
                -1.0, 1.0,
            )
            target_quat_raw = sign * target_quat_raw
            # Slerp-LERP: linear blend + renormalise. For the small
            # angle changes we have here this is indistinguishable
            # from true slerp and avoids the trig.
            q_blend = (
                a * target_quat_raw + (1.0 - a) * self._target_quat_world
            )
            self._target_quat_world = q_blend / jnp.linalg.norm(q_blend)

        R_target = _quat_to_rotation_matrix(self._target_quat_world)
        T_target = jnp.eye(4, dtype=jnp.float32)
        T_target = T_target.at[:3, :3].set(R_target)
        T_target = T_target.at[:3, 3].set(self._target_pos_world)
        self.T_target_world = T_target
        self.T_target_base = self._T_base_inv @ T_target

    def compute(self, params, step_count, state):
        # Update the target pose from the body's most recent state.
        if state is not None and "body" in state:
            body_pos = jnp.asarray(state["body"]["position"], dtype=jnp.float32)
            body_quat = jnp.asarray(
                state["body"].get(
                    "orientation",
                    jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
                ),
                dtype=jnp.float32,
            )
            body_angvel = jnp.asarray(
                state["body"].get(
                    "angular_velocity",
                    jnp.zeros(3, dtype=jnp.float32),
                ),
                dtype=jnp.float32,
            )
            self._update_target_from_body(body_pos, body_quat, body_angvel)

        # Single-dispatch differential IK + IDPD.
        if state is not None and "arm" in state:
            q = jnp.asarray(state["arm"]["joint_angles"], dtype=jnp.float32)
            qd = jnp.asarray(state["arm"]["joint_velocities"], dtype=jnp.float32)
        else:
            q = self.q_home
            qd = jnp.zeros(self.n_dof, dtype=jnp.float32)

        tau, q_target_now, _ = self._compute_torque(
            q, qd, self.T_target_world,
            jnp.float32(self.K_p), jnp.float32(self.K_d),
            jnp.float32(self.K_ik), jnp.float32(self.lam),
            self.gravity_world,
        )
        # Cache q_target for diagnostics (M1 unit test reads it).
        self._q_target = q_target_now

        omega_rad_s = float(2.0 * np.pi * params["FIELD_FREQUENCY_HZ"])
        return {
            "motor": {
                "commanded_velocity": jnp.float32(omega_rad_s),
            },
            "arm": {
                "commanded_joint_torques": tau.astype(jnp.float32),
            },
        }


# Module-level singleton — the runner module-loads this file once,
# but the controller's pre-computation must survive across step calls.
_controller_instance: _TrackingController | None = None


def get_external_inputs(params: dict, step_count: int, state=None) -> dict:
    global _controller_instance
    if _controller_instance is None:
        _controller_instance = _TrackingController(params)
    return _controller_instance.compute(params, step_count, state)
