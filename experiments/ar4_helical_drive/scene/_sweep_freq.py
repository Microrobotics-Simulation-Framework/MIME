"""Sweep drive frequency to find the regime where the helix syncs
with the rotating field instead of tilting off-axis.

At our 20 cm standoff with the de Jongh paper magnet (18.89 A·m²)
the field at the UMR is elliptical (1:2 ratio per Mahoney/Abbott
Eq 1).  Drive frequencies near step-out (~10 Hz) cause stick-slip
sync, exciting the body's tilt mode.  Below step-out the helix
locks and corkscrews along its axis, like the legacy dejongh.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, "/home/nick/MSF/MIME/src")
sys.path.insert(0, "/home/nick/MSF/MIME")

import importlib
import jax.numpy as jnp
import numpy as np

PARAMS_PATH = "/home/nick/MSF/MIME/experiments/ar4_helical_drive/physics/params.py"
ns: dict = {}
with open(PARAMS_PATH) as fh:
    exec(fh.read(), ns)
base_params = {k: v for k, v in ns.items() if not k.startswith("_") and k.isupper()}

DT = float(base_params["DT_PHYS"])
N_STEPS = 1200   # 0.6 s
ZERO_T = jnp.zeros(6, dtype=jnp.float32)

FREQS = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])

def run(freq_hz: float):
    from experiments.ar4_helical_drive.physics import setup as _setup
    importlib.reload(_setup)
    gm = _setup.build_graph(dict(base_params))
    omega = jnp.asarray(2 * jnp.pi * freq_hz, dtype=jnp.float32)
    ext = {
        "motor": {"commanded_velocity": omega},
        "arm":   {"commanded_joint_torques": ZERO_T},
    }
    pos_log = np.zeros((N_STEPS, 3))
    align_log = np.zeros(N_STEPS)
    omega_x_log = np.zeros(N_STEPS)
    for i in range(N_STEPS):
        s = gm.step(ext)
        pos_log[i] = np.asarray(s["body"]["position"])
        q = np.asarray(s["body"]["orientation"])
        R = quat_to_R(q)
        align_log[i] = R[0, 2]   # body-z · world-x
        omega_x_log[i] = float(s["body"]["angular_velocity"][0])
    transient = int(0.1 / DT)
    swim_x = (pos_log[-1, 0] - pos_log[transient, 0]) / ((N_STEPS - transient) * DT)
    return {
        "swim_mm_s": swim_x * 1000,
        "min_align": float(align_log.min()),
        "final_align": float(align_log[-1]),
        "mean_omega_x_rps": float(omega_x_log[transient:].mean()) / (2 * np.pi),
        "x_max_mm": float(np.abs(pos_log[:, 0]).max()) * 1000,
    }


print(f"{'freq_Hz':>8} {'mean_ωhelix_Hz':>16} {'swim_mm_s':>12} "
      f"{'min_align':>12} {'final_align':>12} {'x_max_mm':>10}")
for f in FREQS:
    r = run(f)
    print(f"{f:>8.1f} {r['mean_omega_x_rps']:>16.3f} {r['swim_mm_s']:>12.4f} "
          f"{r['min_align']:>12.4f} {r['final_align']:>12.4f} {r['x_max_mm']:>10.3f}")

print()
print("Interpretation:")
print("  - mean_ωhelix ≈ freq_Hz    →  locked / synced  (good)")
print("  - mean_ωhelix < freq_Hz    →  slipping / above step-out (tilts)")
print("  - min_align near 1.0       →  body-z stays along world-x  (good)")
print("  - min_align < 0.9          →  body tilts > 25° off tube axis")
