"""Trace the body's velocity around the moment the constraint instability fires.

Logs every step from step 500 onward (the JUMP region) so we can see
what the velocity, force, and post-constraint position do.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, "/home/nick/MSF/MIME/src")

import jax.numpy as jnp
import numpy as np

PARAMS_PATH = "/home/nick/MSF/MIME/experiments/ar4_helical_drive/physics/params.py"
ns: dict = {}
with open(PARAMS_PATH) as fh:
    exec(fh.read(), ns)
params = {k: v for k, v in ns.items() if not k.startswith("_") and k.isupper()}

sys.path.insert(0, "/home/nick/MSF/MIME")
from experiments.ar4_helical_drive.physics.setup import build_graph

gm = build_graph(params)

freq_hz = float(params.get("FIELD_FREQUENCY_HZ", 10.0))
omega = jnp.asarray(2 * jnp.pi * freq_hz, dtype=jnp.float32)
zero_t = jnp.zeros(6, dtype=jnp.float32)
ext = {
    "motor": {"commanded_velocity": omega},
    "arm":   {"commanded_joint_torques": zero_t},
}

dt = float(params["DT_PHYS"])
N = 600

print(f"{'step':>5} {'t_ms':>6} "
      f"{'x_mm':>10} {'y_mm':>10} {'z_mm':>10} "
      f"{'vx_m/s':>10} {'vy_m/s':>10} {'vz_m/s':>10} "
      f"{'|F|µN':>9} {'|τ|nNm':>9}")

verbose_from = 400
prev_pos = None
for step in range(N):
    state = gm.step(ext)
    pos = np.asarray(state["body"]["position"])
    vel = np.asarray(state["body"]["velocity"])
    F = np.asarray(state["magnet"]["magnetic_force"])
    T = np.asarray(state["magnet"]["magnetic_torque"])

    if step >= verbose_from or step in {0, 100, 200, 300, 400, 500}:
        print(f"{step:>5} {step*dt*1000:>6.1f} "
              f"{pos[0]*1000:>10.4f} {pos[1]*1000:>10.4f} {pos[2]*1000:>10.4f} "
              f"{vel[0]:>10.4f} {vel[1]:>10.4f} {vel[2]:>10.4f} "
              f"{np.linalg.norm(F)*1e6:>9.3f} {np.linalg.norm(T)*1e9:>9.3f}")
    prev_pos = pos
