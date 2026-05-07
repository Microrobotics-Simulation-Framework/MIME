"""A/B test: static rotor vs. rotor that perfectly tracks the body's
x-position.  Tests the hypothesis that the AR4 needs closed-loop
control to keep the helix swimming straight.

For perfect-tracking we bypass the arm and feed
motor.parent_pose_world as an external input directly, set to
(body.position[0], 0, 0.20, 1, 0, 0, 0) each step.  This keeps the
rotor's perpendicular axis exactly above the body regardless of where
the body has swum.

For the static-rotor baseline the parent pose is fixed at
(0, 0, 0.20, 1, 0, 0, 0).

Compares: alignment over time, swim distance over time, fraction of
time the body is well-aligned (|body-z · world-x| > 0.9).
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, "/home/nick/MSF/MIME/src")
sys.path.insert(0, "/home/nick/MSF/MIME")

import jax.numpy as jnp
import numpy as np

from mime.experiments.dejongh_new_chain import build_graph as _build_chain
from mime.experiments.dejongh import default_mlp_weights_path

DT = 5e-4
N = int(15.0 / DT)   # 15 s
DRIVE_HZ = 3.0
OMEGA = jnp.asarray(2 * jnp.pi * DRIVE_HZ, dtype=jnp.float32)

def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def build_chain_no_arm():
    """Build the dejongh_new_chain without the AR4 arm.
    motor.parent_pose_world is exposed as an external input so the
    test loop can drive it with either fixed or tracking values.
    """
    gm = _build_chain(
        design_name="FL-9", vessel_name='1/4"',
        mu_Pa_s=1e-3, delta_rho=410.0, dt=DT,
        use_lubrication=True, lubrication_epsilon_mm=0.15,
        magnet_base_xyz_m=(0.0, 0.0, 0.20),
        magnet_dipole_a_m2=18.89,
        magnet_radius_m=17.5e-3,
        magnet_length_m=20e-3,
        field_model="point_dipole",
        motor_axis_in_parent=(0.0, 0.0, 1.0),
        use_coupling_group=True,
        vessel_axis=0,
        body_gravity_direction=(0.0, 0.0, -1.0),
        magnet_axis_in_body=(1.0, 0.0, 0.0),
        mlp_weights_path=default_mlp_weights_path(),
    )
    body_state = dict(gm.get_node_state("body"))
    body_state["position"] = jnp.array([0.0, 0.0, -1e-3], dtype=jnp.float32)
    body_state["orientation"] = jnp.array([0.7071068, 0.0, 0.7071068, 0.0],
                                          dtype=jnp.float32)
    gm.set_node_state("body", body_state)
    gm.compile()
    return gm


def run(label: str, tracking: bool):
    gm = build_chain_no_arm()
    print(f"\n=== {label} ===")
    print(f"{'t_s':>6} {'x_mm':>9} {'body_z·x':>10} {'mean_align':>11}")
    align_log = []
    x_log = []
    last_print = -1
    for step in range(N):
        # Build the parent pose for the motor.
        if tracking:
            # Read the body's most recent position from the node state.
            # set_node_state returns immediately — but we want last
            # observed position, so we keep a copy across iterations.
            try:
                body_x = float(x_log[-1] if x_log else 0.0)
            except Exception:
                body_x = 0.0
            parent_pose = jnp.array(
                [body_x, 0.0, 0.20, 1.0, 0.0, 0.0, 0.0],
                dtype=jnp.float32,
            )
        else:
            parent_pose = jnp.array(
                [0.0, 0.0, 0.20, 1.0, 0.0, 0.0, 0.0],
                dtype=jnp.float32,
            )

        ext = {
            "motor": {
                "commanded_velocity": OMEGA,
                "parent_pose_world": parent_pose,
            },
        }
        s = gm.step(ext)
        pos = np.asarray(s["body"]["position"])
        q = np.asarray(s["body"]["orientation"])
        R = quat_to_R(q)
        align = R[0, 2]
        align_log.append(float(align))
        x_log.append(float(pos[0]))

        sec = int((step + 1) * DT)
        if sec != last_print and (step + 1) * DT == sec:
            last_print = sec
            mean_align = float(np.mean(align_log[-int(1.0/DT):]))
            print(f"{sec:>6.1f} {pos[0]*1000:>9.3f} {align:>10.4f} {mean_align:>11.4f}")

    align_arr = np.array(align_log)
    x_arr = np.array(x_log)
    swim_speed = (x_arr[-1] - x_arr[0]) / (N * DT) * 1000  # mm/s
    well_aligned_frac = np.mean(np.abs(align_arr) > 0.9)
    print(f"\n  swim:        {swim_speed:+.2f} mm/s")
    print(f"  fraction well-aligned (|cos| > 0.9): {well_aligned_frac*100:.1f}%")
    print(f"  final x: {x_arr[-1]*1000:+.2f} mm")
    return swim_speed, well_aligned_frac


run("STATIC rotor at world (0, 0, 0.20)", tracking=False)
run("TRACKING rotor at (body.x, 0, 0.20)", tracking=True)
