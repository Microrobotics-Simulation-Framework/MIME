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
    )
    gm.add_node(arm)
    gm.add_edge("arm", "motor", "end_effector_pose_world", "parent_pose_world")

    n_dof = arm._num_joints
    gm.add_external_input(
        "arm", "commanded_joint_torques", shape=(n_dof,), dtype=jnp.float32,
    )

    # 3) Seed the UMR pre-sunk.
    init_pos = jnp.array([
        params["INIT_X_M"], params["INIT_Y_M"], params["INIT_Z_M"],
    ], dtype=jnp.float32)
    if hasattr(gm, "set_initial_state"):
        gm.set_initial_state("body", {"position": init_pos})
    else:
        body = gm._nodes["body"].node if hasattr(gm._nodes["body"], "node") else gm._nodes["body"]
        if hasattr(body, "_initial_state") and body._initial_state is not None:
            body._initial_state["position"] = init_pos

    # 4) Seed the arm in a relaxed home pose.
    home_q = jnp.asarray(
        params.get("ARM_HOME_RAD", _DEFAULTS["ARM_HOME_RAD"]),
        dtype=jnp.float32,
    )
    if hasattr(gm, "set_initial_state"):
        gm.set_initial_state("arm", {"joint_angles": home_q})
    else:
        arm_node = gm._nodes["arm"].node if hasattr(gm._nodes["arm"], "node") else gm._nodes["arm"]
        if hasattr(arm_node, "_initial_state") and arm_node._initial_state is not None:
            arm_node._initial_state["joint_angles"] = home_q

    # 5) Eagerly compile the graph step here so the ~30 s cold-cache
    # XLA compile happens during the runner's "Loading experiment"
    # phase (which the user expects to take a while) rather than on
    # the first viewport tick (which the user sees as "frozen"). The
    # compile is idempotent — subsequent ``gm.step`` calls reuse it.
    if hasattr(gm, "compile"):
        gm.compile()

    return gm
