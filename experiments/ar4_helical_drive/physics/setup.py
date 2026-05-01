"""Adapter from runner params dict → AR4 + helical-UMR graph.

Mirrors ``experiments/dejongh_confined/physics/setup.py`` but builds
the *new-chain* graph: Motor + PermanentMagnetNode + RobotArmNode
(URDF-driven AR4) + the existing dejongh stack (UMR + drag + body).

The MIME runner imports this module and calls ``build_graph(params)``
once at startup, expecting a configured (but un-compiled)
``GraphManager`` back. The runner subsequently calls
``control/controller.py::get_external_inputs(params, step)`` each tick
for live-editable parameters.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

from mime.experiments.dejongh_new_chain import build_graph as _build_dejongh_new_chain
from mime.nodes.actuation.robot_arm import RobotArmNode


def _resolve_urdf(params: dict) -> Path:
    p = Path(params["URDF_PATH"])
    if p.is_absolute():
        return p
    # Resolve relative to this experiment dir:
    # MIME/experiments/ar4_helical_drive/physics/setup.py → experiment dir
    experiment_dir = Path(__file__).resolve().parents[1]
    return experiment_dir / p


def build_graph(params: dict):
    """Construct the AR4 + motor + permanent-magnet + UMR graph."""
    urdf_path = _resolve_urdf(params)

    # 1) Build the dejongh-new-chain stack (Motor + PermanentMagnet +
    # UMR + drag + lubrication). The arm is added separately below;
    # the magnet's pose comes from arm.end_effector_pose_world.
    gm = _build_dejongh_new_chain(
        design_name=params["DESIGN_NAME"],
        vessel_name=params["VESSEL_NAME"],
        mu_Pa_s=params["MU_PA_S"],
        delta_rho=params["DELTA_RHO_KG_M3"],
        dt=params["DT_PHYS"],
        use_lubrication=params["USE_LUBRICATION"],
        lubrication_epsilon_mm=params["LUB_EPSILON_MM"],
        # Magnet base pose is overridden each step by the arm's EE
        # pose (via the edge below); these are the starting values.
        magnet_base_xyz_m=(0.0, 0.0, 0.0),
        magnet_dipole_a_m2=params["MAGNET_DIPOLE_A_M2"],
        magnet_radius_m=params["MAGNET_RADIUS_M"],
        magnet_length_m=params["MAGNET_LENGTH_M"],
        field_model=params["FIELD_MODEL"],
        motor_axis_in_parent=params["MOTOR_AXIS_IN_PARENT"],
        motor_inertia_kg_m2=params["MOTOR_INERTIA_KG_M2"],
        motor_kt_n_m_per_a=params["MOTOR_KT_N_M_PER_A"],
        motor_r_ohm=params["MOTOR_R_OHM"],
        motor_l_henry=params["MOTOR_L_HENRY"],
        motor_damping_n_m_s=params["MOTOR_DAMPING_N_M_S"],
        use_coupling_group=params.get("USE_COUPLING_GROUP", True),
    )

    # 2) Add the AR4 arm and wire its EE pose into the motor.
    arm = RobotArmNode(
        name="arm",
        timestep=params["DT_PHYS"],
        urdf_path=str(urdf_path),
        end_effector_link_name=params["END_EFFECTOR_LINK_NAME"],
        end_effector_offset_in_link=params["END_EFFECTOR_OFFSET_IN_LINK"],
        base_pose_world=params["BASE_POSE_WORLD"],
        joint_friction_n_m_s=params["JOINT_FRICTION_N_M_S"],
        gravity_world=params["GRAVITY_WORLD"],
    )
    gm.add_node(arm)
    gm.add_edge("arm", "motor", "end_effector_pose_world", "parent_pose_world")

    # 3) Expose the arm's commanded torque as an external input.
    n_dof = arm._num_joints
    gm.add_external_input(
        "arm", "commanded_joint_torques", shape=(n_dof,), dtype=jnp.float32,
    )

    # 4) Seed the UMR pre-sunk so the viewer doesn't open on the
    # gravity transient (matches the dejongh_confined trick).
    init_pos = jnp.array([
        params["INIT_X_M"], params["INIT_Y_M"], params["INIT_Z_M"],
    ], dtype=jnp.float32)
    if hasattr(gm, "set_initial_state"):
        gm.set_initial_state("body", {"position": init_pos})
    else:
        body = gm._nodes["body"].node if hasattr(gm._nodes["body"], "node") else gm._nodes["body"]
        if hasattr(body, "_initial_state") and body._initial_state is not None:
            body._initial_state["position"] = init_pos

    # 5) Seed the arm in a relaxed home pose so the magnet starts
    # above the vessel rather than at the unstable zero-pose.
    home_q = jnp.asarray(params["ARM_HOME_RAD"], dtype=jnp.float32)
    if hasattr(gm, "set_initial_state"):
        gm.set_initial_state("arm", {"joint_angles": home_q})
    else:
        arm_node = gm._nodes["arm"].node if hasattr(gm._nodes["arm"], "node") else gm._nodes["arm"]
        if hasattr(arm_node, "_initial_state") and arm_node._initial_state is not None:
            arm_node._initial_state["joint_angles"] = home_q

    return gm
