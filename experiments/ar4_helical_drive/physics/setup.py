"""Adapter from runner params dict → AR4 + helical-UMR graph.

Mirrors ``experiments/dejongh_confined/physics/setup.py`` but builds
the *new-chain* graph: Motor + PermanentMagnetNode + RobotArmNode
(URDF-driven AR4) + the existing dejongh stack (UMR + drag + body).

The MIME runner imports this module and calls ``build_graph(params)``
once at startup. The runner's ``params`` namespace only forwards
JSON-roundtrippable scalars (numbers + bools) — strings, tuples, and
``None`` values get dropped on the way through ZMQ. So we use
``params.get(key, default)`` for every non-scalar parameter, with the
defaults inlined here. The defaults are the same values the
companion ``params.py`` would set if the namespace had survived
intact.
"""

from __future__ import annotations

# JAX env vars must be set before the first ``import jax`` anywhere in
# this process. The MIME runner imports JAX before we get here, so we
# *also* propagate the same flags from ``MIME/tests/conftest.py`` — if
# they were already set by the runner this is a no-op.
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

from pathlib import Path

import jax
import jax.numpy as jnp

from mime.experiments.dejongh_new_chain import build_graph as _build_dejongh_new_chain
from mime.nodes.actuation.robot_arm import RobotArmNode


# ---- Defaults for non-JSON-roundtrippable params ----------------------
# These are the same values the experiment's ``params.py`` declares.
# We re-state them here because the MIME runner strips strings, tuples,
# and ``None`` when serialising the params dict over ZMQ to the
# subprocess that actually executes ``build_graph``.

_DEFAULTS = {
    "DESIGN_NAME": "FL-9",
    "VESSEL_NAME": '1/4"',
    "URDF_PATH": "assets/ar4.urdf",
    "END_EFFECTOR_LINK_NAME": "link_6",
    "BASE_POSE_WORLD": (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    "END_EFFECTOR_OFFSET_IN_LINK": (0.05, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    "JOINT_FRICTION_N_M_S": None,
    "GRAVITY_WORLD": (0.0, 0.0, -9.80665),
    "ARM_HOME_RAD": (0.0, -0.6, 0.6, 0.0, -0.5, 0.0),
    "AUTO_GRAVITY_COMPENSATION": True,
    "MOTOR_AXIS_IN_PARENT": (1.0, 0.0, 0.0),
    "MAGNET_AXIS_IN_BODY": (1.0, 0.0, 0.0),
    "FIELD_MODEL": "point_dipole",
    "EARTH_FIELD_WORLD_T": (0.0, 0.0, 0.0),
}


# Persistent JAX compile cache. The runner's first cold compile is the
# 30-second freeze in MICROROBOTICA today; with the cache primed,
# subsequent runs start in ~5 s.
_jax_cache = Path(
    os.environ.get(
        "JAX_COMPILATION_CACHE_DIR",
        str(Path.home() / ".cache" / "jax_compilation_cache"),
    )
)
_jax_cache.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(_jax_cache))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)


def _resolve_urdf(params: dict) -> Path:
    p = Path(params.get("URDF_PATH", _DEFAULTS["URDF_PATH"]))
    if p.is_absolute():
        return p
    # Resolve relative to this experiment dir:
    # MIME/experiments/ar4_helical_drive/physics/setup.py → experiment dir
    experiment_dir = Path(__file__).resolve().parents[1]
    return experiment_dir / p


def build_graph(params: dict):
    """Construct the AR4 + motor + permanent-magnet + UMR graph."""
    urdf_path = _resolve_urdf(params)

    # 1) Build the dejongh-new-chain stack.
    gm = _build_dejongh_new_chain(
        design_name=params.get("DESIGN_NAME", _DEFAULTS["DESIGN_NAME"]),
        vessel_name=params.get("VESSEL_NAME", _DEFAULTS["VESSEL_NAME"]),
        mu_Pa_s=params["MU_PA_S"],
        delta_rho=params["DELTA_RHO_KG_M3"],
        dt=params["DT_PHYS"],
        use_lubrication=params["USE_LUBRICATION"],
        lubrication_epsilon_mm=params["LUB_EPSILON_MM"],
        magnet_base_xyz_m=(0.0, 0.0, 0.0),
        magnet_dipole_a_m2=params["MAGNET_DIPOLE_A_M2"],
        magnet_radius_m=params["MAGNET_RADIUS_M"],
        magnet_length_m=params["MAGNET_LENGTH_M"],
        field_model=params.get("FIELD_MODEL", _DEFAULTS["FIELD_MODEL"]),
        motor_axis_in_parent=params.get(
            "MOTOR_AXIS_IN_PARENT", _DEFAULTS["MOTOR_AXIS_IN_PARENT"],
        ),
        motor_inertia_kg_m2=params["MOTOR_INERTIA_KG_M2"],
        motor_kt_n_m_per_a=params["MOTOR_KT_N_M_PER_A"],
        motor_r_ohm=params["MOTOR_R_OHM"],
        motor_l_henry=params["MOTOR_L_HENRY"],
        motor_damping_n_m_s=params["MOTOR_DAMPING_N_M_S"],
        use_coupling_group=params.get("USE_COUPLING_GROUP", False),
        # AR4 lays the tube along world-x (horizontal) and uses
        # gravity-toward-floor (-z). dejongh's defaults (z-axis tube,
        # -y gravity) stay unchanged for the legacy experiment.
        vessel_axis=int(params.get("VESSEL_AXIS", 0)),
        body_gravity_direction=tuple(params.get(
            "BODY_GRAVITY_DIRECTION", (0.0, 0.0, -1.0),
        )),
        magnet_axis_in_body=tuple(params.get(
            "MAGNET_AXIS_IN_BODY", _DEFAULTS["MAGNET_AXIS_IN_BODY"],
        )),
    )

    # 2) Add the AR4 arm and wire its EE pose into the motor.
    arm = RobotArmNode(
        name="arm",
        timestep=params["DT_PHYS"],
        urdf_path=str(urdf_path),
        end_effector_link_name=params.get(
            "END_EFFECTOR_LINK_NAME", _DEFAULTS["END_EFFECTOR_LINK_NAME"],
        ),
        end_effector_offset_in_link=params.get(
            "END_EFFECTOR_OFFSET_IN_LINK",
            _DEFAULTS["END_EFFECTOR_OFFSET_IN_LINK"],
        ),
        base_pose_world=params.get(
            "BASE_POSE_WORLD", _DEFAULTS["BASE_POSE_WORLD"],
        ),
        joint_friction_n_m_s=params.get(
            "JOINT_FRICTION_N_M_S", _DEFAULTS["JOINT_FRICTION_N_M_S"],
        ),
        gravity_world=params.get(
            "GRAVITY_WORLD", _DEFAULTS["GRAVITY_WORLD"],
        ),
        auto_gravity_compensation=bool(params.get(
            "AUTO_GRAVITY_COMPENSATION", _DEFAULTS["AUTO_GRAVITY_COMPENSATION"],
        )),
    )
    gm.add_node(arm)
    gm.add_edge("arm", "motor", "end_effector_pose_world", "parent_pose_world")

    n_dof = arm._num_joints
    gm.add_external_input(
        "arm", "commanded_joint_torques", shape=(n_dof,), dtype=jnp.float32,
    )

    # 3) Seed the UMR pre-sunk + initial orientation rotated so the
    # helix's body z-axis (the FL-9 mesh's long axis) aligns with the
    # new vessel axis (world-x). Quaternion for +90° rotation about
    # world-y: (cos45, 0, sin45, 0) = (0.7071, 0, 0.7071, 0) in
    # (qw, qx, qy, qz) — this maps body-z to world-x.
    init_pos = jnp.array([
        params["INIT_X_M"], params["INIT_Y_M"], params["INIT_Z_M"],
    ], dtype=jnp.float32)
    init_orient = jnp.array(
        [0.7071068, 0.0, 0.7071068, 0.0], dtype=jnp.float32,
    )
    # GraphManager exposes ``set_node_state`` (not set_initial_state).
    # It REPLACES the entire state dict for the node, so we merge
    # our overrides on top of the existing initial state to keep
    # velocity / angular_velocity / etc. at their defaults.
    body_state = dict(gm.get_node_state("body")) if hasattr(gm, "get_node_state") else dict(gm._state["body"])
    body_state["position"] = init_pos
    body_state["orientation"] = init_orient
    if hasattr(gm, "set_node_state"):
        gm.set_node_state("body", body_state)
    else:
        gm._state["body"] = body_state

    # 4) Seed the arm in a relaxed home pose. We also pre-compute the
    # home-pose FK and write ``link_poses_world`` / ``end_effector_pose_world``
    # directly into the initial state so the very first published
    # ResultFrame carries valid poses — without this the first frame
    # has identity-7 placeholders, which the rotation-matrix-to-quat
    # path can turn into NaNs that break MICROROBOTICA's JSON parser.
    from mime.control.kinematics.fk import joint_to_world_transforms
    from mime.control.kinematics.transform import (
        pose_to_matrix, _rotation_matrix_to_quat,
    )
    import jax
    home_q = jnp.asarray(
        params.get("ARM_HOME_RAD", _DEFAULTS["ARM_HOME_RAD"]),
        dtype=jnp.float32,
    )
    home_qd = jnp.zeros_like(home_q)
    base_pose = jnp.asarray(
        params.get("BASE_POSE_WORLD", _DEFAULTS["BASE_POSE_WORLD"]),
        dtype=jnp.float32,
    )
    ee_offset = jnp.asarray(
        params.get(
            "END_EFFECTOR_OFFSET_IN_LINK", _DEFAULTS["END_EFFECTOR_OFFSET_IN_LINK"],
        ),
        dtype=jnp.float32,
    )
    link_xforms = joint_to_world_transforms(arm._tree, home_q)
    link_xforms = pose_to_matrix(base_pose)[None] @ link_xforms
    link_t = link_xforms[:, :3, 3]
    link_R = link_xforms[:, :3, :3]
    link_quats = jax.vmap(_rotation_matrix_to_quat)(link_R)
    home_link_poses = jnp.concatenate([link_t, link_quats], axis=1)
    T_ee = link_xforms[arm._ee_idx] @ pose_to_matrix(ee_offset)
    home_ee_pose = jnp.concatenate([
        T_ee[:3, 3], _rotation_matrix_to_quat(T_ee[:3, :3]),
    ])
    # Same set_node_state pattern as the body — merge home-pose
    # overrides into the existing arm state dict.
    arm_state = dict(gm.get_node_state("arm")) if hasattr(gm, "get_node_state") else dict(gm._state["arm"])
    arm_state["joint_angles"] = home_q
    arm_state["joint_velocities"] = home_qd
    arm_state["link_poses_world"] = home_link_poses
    arm_state["end_effector_pose_world"] = home_ee_pose
    if hasattr(gm, "set_node_state"):
        gm.set_node_state("arm", arm_state)
    else:
        gm._state["arm"] = arm_state

    # 5) Eagerly compile the graph step here so the ~30 s cold-cache
    # XLA compile happens during the runner's "Loading experiment"
    # phase (which the user expects to take a while) rather than on
    # the first viewport tick (which the user sees as "frozen"). The
    # compile is idempotent — subsequent ``gm.step`` calls reuse it.
    if hasattr(gm, "compile"):
        gm.compile()

    return gm
