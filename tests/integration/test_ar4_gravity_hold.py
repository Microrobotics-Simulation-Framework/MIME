"""M0 smoke test: AR4 with auto-gravity-comp holds its IK home pose.

The AR4 helical-drive experiment IK-derives a home pose that places
the EE at world (0, 0, 0.20) with EE-z aligned with world-x. With
``auto_gravity_compensation=True`` and zero commanded joint torques,
the arm must hold this pose against gravity — joint drift after
100 steps must be ≪ 1 mrad.

This is a prerequisite for the closed-loop AR4 controller (M2+):
the controller's PD law assumes gravity is already cancelled, so it
only handles the tracking error. If gravity isn't being cancelled
the arm will sag while the controller fights it, masking real
tracking-loop bugs.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from mime.nodes.actuation.robot_arm import RobotArmNode

# IK-derived AR4 home pose at 20 cm standoff (max reach, EE-z = world-x).
# Mirrors experiments/ar4_helical_drive/physics/params.py::ARM_HOME_RAD.
ARM_HOME_RAD = (-0.02763, 0.49873, -1.51875, 1.54673, 1.55147, 0.00000)
BASE_POSE_WORLD = (-0.05, 0.328, -0.43, 1.0, 0.0, 0.0, 0.0)
URDF_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments" / "ar4_helical_drive" / "assets" / "ar4.urdf"
)


@pytest.mark.parametrize("n_steps", [100])
def test_auto_gravity_comp_holds_home_pose(n_steps):
    arm = RobotArmNode(
        name="arm",
        timestep=5e-4,
        urdf_path=str(URDF_PATH),
        end_effector_link_name="link_6",
        base_pose_world=BASE_POSE_WORLD,
        auto_gravity_compensation=True,
        gravity_world=(0.0, 0.0, -9.80665),
    )

    n = arm._num_joints
    state = arm.initial_state()
    q_home = jnp.asarray(ARM_HOME_RAD, dtype=jnp.float32)
    state["joint_angles"] = q_home
    state["joint_velocities"] = jnp.zeros(n, dtype=jnp.float32)

    zero_torques = jnp.zeros(n, dtype=jnp.float32)
    boundary_inputs = {"commanded_joint_torques": zero_torques}

    dt = 5e-4
    for _ in range(n_steps):
        state = arm.update(state, boundary_inputs, dt=dt)

    drift = np.asarray(state["joint_angles"]) - np.asarray(q_home)
    drift_inf = float(np.max(np.abs(drift)))
    qd_norm = float(np.linalg.norm(np.asarray(state["joint_velocities"])))

    assert drift_inf < 1e-3, (
        f"Joint drift {drift_inf:.2e} rad exceeds 1 mrad after {n_steps} steps "
        f"with auto_gravity_compensation=True; per-joint drift = {drift}"
    )
    assert qd_norm < 1e-3, (
        f"Joint velocities {qd_norm:.2e} rad/s nonzero after settle; "
        f"qd = {state['joint_velocities']}"
    )
