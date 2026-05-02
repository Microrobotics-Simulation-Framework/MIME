"""Trace the helix's angular velocity, orientation, and torque axis to
see whether it's rotating around the right axis (world-x = vessel axis).
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, "/home/nick/MSF/MIME/src")
sys.path.insert(0, "/home/nick/MSF/MIME")

import jax.numpy as jnp
import numpy as np

PARAMS_PATH = "/home/nick/MSF/MIME/experiments/ar4_helical_drive/physics/params.py"
ns: dict = {}
with open(PARAMS_PATH) as fh:
    exec(fh.read(), ns)
params = {k: v for k, v in ns.items() if not k.startswith("_") and k.isupper()}

from experiments.ar4_helical_drive.physics.setup import build_graph

gm = build_graph(params)
freq_hz = float(params.get("FIELD_FREQUENCY_HZ", 10.0))
omega_drive = 2 * jnp.pi * freq_hz
ext = {
    "motor": {"commanded_velocity": jnp.asarray(omega_drive, dtype=jnp.float32)},
    "arm":   {"commanded_joint_torques": jnp.zeros(6, dtype=jnp.float32)},
}

dt = float(params["DT_PHYS"])
N = 600

def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ])

print(f"{'step':>5} {'t_ms':>6} "
      f"{'body_z·world_x':>16} {'ωx':>9} {'ωy':>9} {'ωz':>9} "
      f"{'|τ|nNm':>10} {'τx':>9} {'τy':>9} {'τz':>9} "
      f"{'B_y_µT':>9} {'B_z_µT':>9}")

for step in range(N):
    s = gm.step(ext)
    q = np.asarray(s["body"]["orientation"])
    omega_body = np.asarray(s["body"]["angular_velocity"])
    tau = np.asarray(s["magnet"]["magnetic_torque"])
    B = np.asarray(s["ext_magnet"]["field_vector"])
    R = quat_to_R(q)
    body_z_world = R[:, 2]   # helical long axis in world
    if step % 20 == 0 or step < 5:
        print(f"{step:>5} {step*dt*1000:>6.1f} "
              f"{body_z_world[0]:>+16.4f} "
              f"{omega_body[0]:>+9.2f} {omega_body[1]:>+9.2f} {omega_body[2]:>+9.2f} "
              f"{np.linalg.norm(tau)*1e9:>10.2f} "
              f"{tau[0]*1e9:>+9.2f} {tau[1]*1e9:>+9.2f} {tau[2]*1e9:>+9.2f} "
              f"{B[1]*1e6:>+9.2f} {B[2]*1e6:>+9.2f}")
