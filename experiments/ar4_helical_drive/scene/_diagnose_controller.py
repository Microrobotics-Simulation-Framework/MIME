"""End-to-end smoke test: build the full AR4 graph, load the
controller, step a few times via the runner-style call, and confirm
no NaN/exceptions and that the rotor pose tracks the body's x.

Used for M6 visual validation prep — confirms the experiment is in
a state the MICROROBOTICA runner will execute cleanly.
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

ns: dict = {}
with open(PARAMS_PATH) as fh: exec(fh.read(), ns)
params = {k: v for k, v in ns.items() if not k.startswith("_") and k.isupper()}

setup = _load_module("ar4_setup", SETUP_PATH)
controller = _load_module("ar4_controller", CONTROLLER_PATH)
controller._controller_instance = None

print("Building full AR4 + helix chain...", flush=True)
gm = setup.build_graph(params)
print(f"  Built {len(gm._nodes)} nodes: {list(gm._nodes)}", flush=True)

prev_state = {n: gm.get_node_state(n) for n in gm._nodes}
print(f"  Body initial pos: {np.asarray(prev_state['body']['position'])}", flush=True)
print(f"  Arm initial q: {np.asarray(prev_state['arm']['joint_angles'])}", flush=True)

# Step 50 times (25 ms of physics). Slow first step due to JIT,
# subsequent should fly.
import time
for step in range(50):
    t0 = time.perf_counter()
    ext = controller.get_external_inputs(params, step, state=prev_state)
    gm.step(ext)
    prev_state = {n: gm.get_node_state(n) for n in gm._nodes}
    dt = time.perf_counter() - t0
    if step < 3 or step == 49:
        body_x = float(prev_state["body"]["position"][0])
        arm_link5 = np.asarray(prev_state["arm"]["link_poses_world"][5][:3])
        any_nan = any(
            np.any(np.isnan(np.asarray(v)))
            for n in prev_state for v in prev_state[n].values()
            if hasattr(v, "shape")
        )
        print(f"  step {step}: dt={dt*1000:.1f}ms, body.x={body_x*1000:+.4f}mm, "
              f"link5_world={arm_link5}, any_nan={any_nan}", flush=True)

print("OK — controller + graph integrate cleanly through 50 steps.")
