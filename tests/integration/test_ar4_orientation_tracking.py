"""M4 integration test: orientation tracking aligns the controller's
target EE-z with the body's body-z axis.

Setup: feed the controller a body pose whose body-z is rotated 30°
off world-x (e.g. the helix has tilted while swimming). After the
slerp filter has converged, the controller's target rotation must
satisfy ``R_target[:, 2] ≈ R_body[:, 2]`` to within 1e-3 (numerical
slerp residue + Gram-Schmidt).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest


PARAMS_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments" / "ar4_helical_drive" / "physics" / "params.py"
)
CONTROLLER_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments" / "ar4_helical_drive" / "control" / "controller.py"
)


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_params() -> dict:
    ns: dict = {}
    with open(PARAMS_PATH) as fh:
        exec(fh.read(), ns)
    return {k: v for k, v in ns.items() if not k.startswith("_") and k.isupper()}


def _axis_angle_quat(axis, angle):
    """Build a WXYZ quaternion from axis-angle (axis assumed unit)."""
    h = 0.5 * angle
    return jnp.concatenate([
        jnp.array([jnp.cos(h)], dtype=jnp.float32),
        jnp.sin(h) * jnp.asarray(axis, dtype=jnp.float32),
    ])


def test_target_z_axis_tracks_body_z_axis():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    params = _load_params()
    params["ENABLE_ORIENTATION_FEEDBACK"] = True
    controller = _load_module_from_path("ar4_controller", CONTROLLER_PATH)
    controller._controller_instance = None

    home = jnp.asarray(params["ARM_HOME_RAD"], dtype=jnp.float32)
    # Body pose: tilted 30° about world-y from the home orientation.
    # The body's body-z direction (3rd column of R) should land at
    # roughly cos(30°)·world-x + sin(30°)·world-z.
    # Home orientation (from setup.py seed) maps body-z → world-x:
    #   q_home = (0.7071, 0, 0.7071, 0).
    # Apply a +30° about world-y on top of that.
    q_home_body = jnp.array([0.7071068, 0.0, 0.7071068, 0.0], dtype=jnp.float32)
    q_tilt = _axis_angle_quat([0.0, 1.0, 0.0], jnp.deg2rad(30.0))
    # Quaternion product (wxyz Hamilton): q_tilted = q_tilt * q_home_body
    def qmul(p, q):
        pw, px, py, pz = p[0], p[1], p[2], p[3]
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        return jnp.array([
            pw*qw - px*qx - py*qy - pz*qz,
            pw*qx + px*qw + py*qz - pz*qy,
            pw*qy - px*qz + py*qw + pz*qx,
            pw*qz + px*qy - py*qx + pz*qw,
        ])
    q_body_tilted = qmul(q_tilt, q_home_body)
    # Body-z direction expected: column 2 of rotation matrix from q_body_tilted.
    from mime.control.kinematics.transform import _quat_to_rotation_matrix
    R_body = _quat_to_rotation_matrix(q_body_tilted)
    body_z_world = R_body[:, 2]

    state = {
        "arm": {
            "joint_angles": home,
            "joint_velocities": jnp.zeros_like(home),
        },
        "body": {
            "position": jnp.array([0.0, 0.0, -1e-3], dtype=jnp.float32),
            "orientation": q_body_tilted,
        },
    }
    # Bypass the full controller and call _update_target_from_body
    # directly — the controller's IK + mass-matrix work isn't needed
    # to test the orientation-target update logic, and skipping it
    # avoids the JAX dispatch cost of 100+ Python calls.
    # First call to bring controller state up.
    controller.get_external_inputs(params, 0, state=state)
    inst = controller._controller_instance
    body_pos = state["body"]["position"]
    body_quat = state["body"]["orientation"]
    body_angvel = jnp.zeros(3, dtype=jnp.float32)   # static body
    for _ in range(200):
        inst._update_target_from_body(body_pos, body_quat, body_angvel)

    R_target = np.asarray(inst.T_target_world[:3, :3])
    target_z = R_target[:, 2]
    body_z = np.asarray(body_z_world)
    cos_err = float(np.dot(target_z, body_z))
    assert cos_err > 0.999, (
        f"Target z-axis not aligned with body z-axis: "
        f"target_z={target_z}, body_z={body_z}, cos={cos_err:.4f}"
    )


def test_orientation_feedback_can_be_disabled():
    """With ENABLE_ORIENTATION_FEEDBACK=False the target rotation
    should remain at the home-pose orientation regardless of body
    orientation."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    params = _load_params()
    params["ENABLE_ORIENTATION_FEEDBACK"] = False
    controller = _load_module_from_path("ar4_controller", CONTROLLER_PATH)
    controller._controller_instance = None

    home = jnp.asarray(params["ARM_HOME_RAD"], dtype=jnp.float32)
    # Body tilted 45° about world-y on top of home seed orientation.
    q_home_body = jnp.array([0.7071068, 0.0, 0.7071068, 0.0], dtype=jnp.float32)
    h = 0.5 * np.deg2rad(45.0)
    q_tilt = jnp.array([np.cos(h), 0.0, np.sin(h), 0.0], dtype=jnp.float32)
    def qmul(p, q):
        pw, px, py, pz = p[0], p[1], p[2], p[3]
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        return jnp.array([
            pw*qw - px*qx - py*qy - pz*qz,
            pw*qx + px*qw + py*qz - pz*qy,
            pw*qy - px*qz + py*qw + pz*qx,
            pw*qz + px*qy - py*qx + pz*qw,
        ])
    q_body_tilted = qmul(q_tilt, q_home_body)

    state = {
        "arm": {"joint_angles": home, "joint_velocities": jnp.zeros_like(home)},
        "body": {
            "position": jnp.array([0.0, 0.0, -1e-3], dtype=jnp.float32),
            "orientation": q_body_tilted,
        },
    }
    # Initialise the controller, then drive _update_target_from_body
    # directly — same shortcut as the alignment test above.
    controller.get_external_inputs(params, 0, state=state)
    inst = controller._controller_instance
    body_pos = state["body"]["position"]
    body_quat = state["body"]["orientation"]
    body_angvel = jnp.zeros(3, dtype=jnp.float32)   # static body
    for _ in range(50):
        inst._update_target_from_body(body_pos, body_quat, body_angvel)
    R_target = np.asarray(inst.T_target_world[:3, :3])
    R_home = np.asarray(inst.R_home_world)
    diff = float(np.linalg.norm(R_target - R_home))
    assert diff < 1e-5, (
        f"With orientation feedback off the target rotation should "
        f"equal the home-pose rotation; got diff = {diff:.2e}"
    )
