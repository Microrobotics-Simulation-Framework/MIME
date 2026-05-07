"""Trace body orientation, controller target, and EE actual orientation
each step to find where the rotor's spin axis stops tracking the
helix's long axis.

Logs every 50 ms (100 steps): body-z·world-x, target-z·world-x,
EE-z·world-x, and the angle between EE-z and body-z.
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


def _load_module(name, path):
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


ns: dict = {}
with open(PARAMS_PATH) as fh: exec(fh.read(), ns)
params = {k: v for k, v in ns.items() if not k.startswith("_") and k.isupper()}

setup = _load_module("ar4_setup", SETUP_PATH)
controller = _load_module("ar4_controller", CONTROLLER_PATH)
controller._controller_instance = None

print("Building chain...", flush=True)
gm = setup.build_graph(params)
dt = float(params["DT_PHYS"])
N_steps = int(0.6 / dt)   # 0.6 s — past the user's reported failure point
print(f"sim {N_steps} steps × {dt*1000:.2f} ms = {N_steps*dt:.2f} s\n", flush=True)

prev_state = {n: gm.get_node_state(n) for n in gm._nodes}

print(f"{'step':>5} {'t_ms':>6} "
      f"{'body_z·x':>10} {'tgt_z·x':>10} {'ee_z·x':>10} "
      f"{'ee_z·body_z':>12} {'body_x_mm':>10} {'rotor_θ':>8}")

import time
t_start = time.perf_counter()
for step in range(N_steps):
    ext = controller.get_external_inputs(params, step, state=prev_state)
    gm.step(ext)
    prev_state = {n: gm.get_node_state(n) for n in gm._nodes}

    if step % 100 == 0 or step in (10, 50, N_steps - 1):
        body_q = np.asarray(prev_state["body"]["orientation"])
        R_body = quat_to_R(body_q)
        body_z = R_body[:, 2]

        # Controller's target rotation
        inst = controller._controller_instance
        R_target = np.asarray(inst.T_target_world[:3, :3])
        target_z = R_target[:, 2]

        # Actual EE world rotation from arm state
        link_poses = np.asarray(prev_state["arm"]["link_poses_world"])
        ee_pose = np.asarray(prev_state["arm"]["end_effector_pose_world"])
        ee_q = ee_pose[3:7]
        R_ee = quat_to_R(ee_q)
        ee_z = R_ee[:, 2]

        body_x = float(prev_state["body"]["position"][0])
        rotor_theta = float(prev_state["motor"]["angle"])

        print(f"{step:>5} {step*dt*1000:>6.1f} "
              f"{body_z[0]:>+10.4f} {target_z[0]:>+10.4f} {ee_z[0]:>+10.4f} "
              f"{float(np.dot(ee_z, body_z)):>+12.4f} "
              f"{body_x*1000:>+10.4f} {rotor_theta:>+8.3f}",
              flush=True)

print(f"\ntotal sim time {time.perf_counter() - t_start:.1f}s")
