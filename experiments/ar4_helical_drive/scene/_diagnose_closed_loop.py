"""Compare closed-loop AR4 with vs. without orientation feedback.

Runs the full graph for 5 s under the controller, with and without
``ENABLE_ORIENTATION_FEEDBACK``.  Prints alignment, swim distance,
and orbital amplitude — the headline numbers for M6 visual
acceptance.

Without feedback, the elliptical-field tilt instability tumbles the
helix off-axis within seconds. With feedback, the rotor reorients to
keep the helix locked, alignment stays high, and the helix
corkscrews along the tube cleanly.
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


def run_one(label: str, enable_orientation_feedback: bool, sim_s: float = 5.0):
    ns: dict = {}
    with open(PARAMS_PATH) as fh: exec(fh.read(), ns)
    params = {k: v for k, v in ns.items() if not k.startswith("_") and k.isupper()}
    params["ENABLE_ORIENTATION_FEEDBACK"] = enable_orientation_feedback

    setup = _load_module("ar4_setup", SETUP_PATH)
    controller = _load_module("ar4_controller", CONTROLLER_PATH)
    controller._controller_instance = None
    gm = setup.build_graph(params)

    dt = float(params["DT_PHYS"])
    n_steps = int(sim_s / dt)
    prev_state = {n: gm.get_node_state(n) for n in gm._nodes}

    align_log = []
    body_x_log = []
    yz_log = []
    for step in range(n_steps):
        ext = controller.get_external_inputs(params, step, state=prev_state)
        gm.step(ext)
        prev_state = {n: gm.get_node_state(n) for n in gm._nodes}
        q = np.asarray(prev_state["body"]["orientation"])
        pos = np.asarray(prev_state["body"]["position"])
        R = quat_to_R(q)
        align_log.append(R[0, 2])  # body-z dot world-x
        body_x_log.append(pos[0])
        yz_log.append(np.sqrt(pos[1]**2 + pos[2]**2))

    align = np.array(align_log)
    body_x = np.array(body_x_log)
    yz = np.array(yz_log)
    well_aligned = float(np.mean(np.abs(align) > 0.9))
    swim_mm_s = (body_x[-1] - body_x[0]) * 1000 / sim_s
    orbit_mm = float((yz.max() - yz.min())) * 1000

    print(f"=== {label} ===")
    print(f"  fraction time |body-z·world-x| > 0.9: {well_aligned*100:.1f}%")
    print(f"  swim speed: {swim_mm_s:+.2f} mm/s")
    print(f"  body x at end: {body_x[-1]*1000:+.3f} mm")
    print(f"  orbit yz amplitude: {orbit_mm:.3f} mm")
    print(f"  alignment at start={align[0]:+.3f}, mid={align[n_steps//2]:+.3f}, end={align[-1]:+.3f}")
    print()


run_one("M3 only — orientation feedback OFF (instability expected)",
        enable_orientation_feedback=False)
run_one("M4 — orientation feedback ON (stable swim expected)",
        enable_orientation_feedback=True)
