"""A/B test: is the AR4's aggressive orientation-tracking helping or
hurting the helix?  Compares feedback ON vs feedback OFF.

Logs body angular velocity magnitude (should be ~18.85 rad/s = 3 Hz
if helix is properly synced; much higher if tumbling).  Also logs
the rotor pose magnitude — if the AR4 is thrashing, the rotor
position changes rapidly, perturbing the field source.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, "/home/nick/MSF/MIME/src")
sys.path.insert(0, "/home/nick/MSF/MIME")

import importlib.util
import jax.numpy as jnp
import numpy as np

PARAMS_PATH = "/home/nick/MSF/MIME/experiments/ar4_helical_drive/physics/params.py"
SETUP_PATH = "/home/nick/MSF/MIME/experiments/ar4_helical_drive/physics/setup.py"
CONTROLLER_PATH = "/home/nick/MSF/MIME/experiments/ar4_helical_drive/control/controller.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def run(label, enable_orientation_feedback):
    ns: dict = {}
    with open(PARAMS_PATH) as fh: exec(fh.read(), ns)
    params = {k: v for k, v in ns.items() if not k.startswith("_") and k.isupper()}
    params["ENABLE_ORIENTATION_FEEDBACK"] = enable_orientation_feedback

    setup = _load("ar4_setup", SETUP_PATH)
    ctrl = _load("ar4_ctrl", CONTROLLER_PATH)
    ctrl._controller_instance = None

    gm = setup.build_graph(params)
    dt = float(params["DT_PHYS"])
    N = int(0.6 / dt)

    print(f"\n{'='*78}\n{label}  (orient_fb={enable_orientation_feedback})")
    print(f"{'='*78}")
    print(f"{'t_ms':>6} {'body_z·x':>10} {'|ω_body|':>10} "
          f"{'ω_drive*':>9} {'rotor_y':>9} {'rotor_z':>9} "
          f"{'body_x_mm':>10}")
    drive_omega = 2 * np.pi * params["FIELD_FREQUENCY_HZ"]
    print(f"{'':>6} {'':>10} {'rad/s':>10} {drive_omega:>9.2f} "
          f"{'mm':>9} {'mm':>9} {'':>10}")

    prev = {n: gm.get_node_state(n) for n in gm._nodes}
    for step in range(N):
        ext = ctrl.get_external_inputs(params, step, state=prev)
        gm.step(ext)
        prev = {n: gm.get_node_state(n) for n in gm._nodes}
        if step % 100 == 0 or step in (10, 50, N-1):
            body_q = np.asarray(prev["body"]["orientation"])
            body_w = np.asarray(prev["body"]["angular_velocity"])
            R_b = quat_to_R(body_q)
            rotor_pos = np.asarray(prev["motor"]["rotor_pose_world"])[:3] \
                if "rotor_pose_world" in prev["motor"] \
                else np.asarray(prev["arm"]["end_effector_pose_world"])[:3]
            print(f"{step*dt*1000:>6.0f} {R_b[0,2]:>+10.4f} "
                  f"{float(np.linalg.norm(body_w)):>10.2f} "
                  f"{'':>9} "
                  f"{rotor_pos[1]*1000:>+9.4f} {rotor_pos[2]*1000:>+9.4f} "
                  f"{prev['body']['position'][0]*1000:>+10.4f}")


run("orientation feedback OFF — pure M3 baseline", False)
run("orientation feedback ON — M4 closed-loop", True)
