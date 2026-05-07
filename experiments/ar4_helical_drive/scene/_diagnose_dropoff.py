"""Trace body alignment and swim speed vs. axial position.

Hypothesis: as the helix swims along world-x away from the rotor's
on-axis position (rotor fixed at (0, 0, 0.20)), the rotating field at
the helix loses its perpendicular geometry — the rotation plane
tilts, the optimal helix orientation tilts, and eventually the helix
can't track the rotating field and loses lock.

Logs every 100 ms: x position, body-z·world-x (alignment), spin axis,
mean angular velocity, |B| at body, |F_grad|, etc.
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
freq = float(params.get("FIELD_FREQUENCY_HZ", 3.0))
omega_drive = jnp.asarray(2 * jnp.pi * freq, dtype=jnp.float32)
ext = {
    "motor": {"commanded_velocity": omega_drive},
    "arm":   {"commanded_joint_torques": jnp.zeros(6, dtype=jnp.float32)},
}
dt = float(params["DT_PHYS"])

# 30 s of physics
N = int(30.0 / dt)
print(f"dt={dt}, N={N}, sim={N*dt:.1f}s, drive freq = {freq} Hz, drive ω = {2*np.pi*freq:.2f} rad/s")
print(f"Rotor at world (0, 0, 0.20). Vessel along world-x.\n")

def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])

# Aggregate angular velocity over each 100 ms window.
window = int(0.1 / dt)
print(f"{'t_s':>5} {'x_mm':>8} {'y_mm':>8} {'z_mm':>8} {'body_z·x':>10} "
      f"{'<ωx>':>9} {'<ωy>':>9} {'<ωz>':>9} {'|B|µT':>8} {'|F_g|µN':>8}")

omega_buf = []
for step in range(N):
    s = gm.step(ext)
    pos = np.asarray(s["body"]["position"])
    q = np.asarray(s["body"]["orientation"])
    omega_b = np.asarray(s["body"]["angular_velocity"])
    B = np.asarray(s["ext_magnet"]["field_vector"])
    F_mag = np.asarray(s["magnet"]["magnetic_force"])
    omega_buf.append(omega_b)
    if (step + 1) % window == 0:
        omega_mean = np.mean(np.array(omega_buf), axis=0) / (2 * np.pi)
        omega_buf = []
        R = quat_to_R(q)
        align = R[0, 2]
        print(f"{(step+1)*dt:>5.1f} "
              f"{pos[0]*1000:>8.3f} {pos[1]*1000:>8.3f} {pos[2]*1000:>8.3f} "
              f"{align:>10.4f} "
              f"{omega_mean[0]:>+9.3f} {omega_mean[1]:>+9.3f} {omega_mean[2]:>+9.3f} "
              f"{np.linalg.norm(B)*1e6:>8.2f} "
              f"{np.linalg.norm(F_mag)*1e6:>8.3f}")
