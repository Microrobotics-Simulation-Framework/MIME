"""Graph-based sweep: simulate the full AR4 chain at several
(standoff, dipole) combinations and report observed swim speed and
orbital amplitude.

Re-uses the actual params.py + setup.py path; just patches the runtime
parameters before each run.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, "/home/nick/MSF/MIME/src")

import importlib
import jax.numpy as jnp
import numpy as np

PARAMS_PATH = "/home/nick/MSF/MIME/experiments/ar4_helical_drive/physics/params.py"
ns: dict = {}
with open(PARAMS_PATH) as fh:
    exec(fh.read(), ns)
base_params = {k: v for k, v in ns.items() if not k.startswith("_") and k.isupper()}

# IK-derived home pose for each standoff (precomputed; see /tmp/ik_*.py)
# AR4 max reach above tube ~20 cm; further standoffs are infeasible.
HOME_BY_STANDOFF = {
    0.07: (-0.02755, 0.14074, -0.26752, -1.57363, -1.54282, 0.00000),
    0.15: (-0.02762, 0.23895, -0.77476,  1.55612,  1.54629, 0.00000),
    0.20: (-0.02763, 0.49873, -1.51875,  1.54673,  1.55147, 0.00000),
}

POINTS = [
    # (standoff_m, dipole_A_m2) — at 20 cm sweep down dipole
    (0.20, 18.89),
    (0.20, 10.0),
    (0.20, 5.0),
    (0.20, 2.0),
    (0.20, 1.0),
    (0.20, 0.5),
]

DT = float(base_params["DT_PHYS"])
N_STEPS = 800   # 0.4 s
FREQ_HZ = 10.0
OMEGA = jnp.asarray(2 * jnp.pi * FREQ_HZ, dtype=jnp.float32)
ZERO_TORQUES = jnp.zeros(6, dtype=jnp.float32)

def run_one(standoff_m: float, dipole: float):
    sys.path.insert(0, "/home/nick/MSF/MIME")
    from experiments.ar4_helical_drive.physics import setup as _setup
    importlib.reload(_setup)

    p = dict(base_params)
    p["MAGNET_DIPOLE_A_M2"] = float(dipole)
    p["ARM_HOME_RAD"] = HOME_BY_STANDOFF.get(round(standoff_m, 2), p["ARM_HOME_RAD"])

    gm = _setup.build_graph(p)
    ext = {
        "motor": {"commanded_velocity": OMEGA},
        "arm":   {"commanded_joint_torques": ZERO_TORQUES},
    }

    pos_log = np.zeros((N_STEPS, 3))
    F_log = np.zeros(N_STEPS)
    B_log = np.zeros(N_STEPS)
    for i in range(N_STEPS):
        st = gm.step(ext)
        pos_log[i] = np.asarray(st["body"]["position"])
        F_log[i] = float(np.linalg.norm(np.asarray(st["magnet"]["magnetic_force"])))
        B_log[i] = float(np.linalg.norm(np.asarray(st["ext_magnet"]["field_vector"])))

    # Analyse the trajectory after the first 50 ms (ramp-up).
    transient = int(0.05 / DT)
    pos = pos_log[transient:]
    swim_vel_x = (pos[-1, 0] - pos[0, 0]) / ((N_STEPS - transient) * DT)  # m/s
    yz = pos[:, 1:3]  # (N, 2)
    yz_radius = np.linalg.norm(yz, axis=1)
    orbit_amp = float(yz_radius.max() - yz_radius.min())
    return {
        "B_mean_µT": float(B_log[transient:].mean()) * 1e6,
        "F_mean_µN": float(F_log[transient:].mean()) * 1e6,
        "swim_mm_s": swim_vel_x * 1000,
        "orbit_mm":  orbit_amp * 1000,
        "x_max_mm":  float(np.abs(pos[:, 0]).max()) * 1000,
    }


print(f"{'r_cm':>5} {'m_Am²':>8} {'B_mean_µT':>11} {'F_mean_µN':>11} "
      f"{'swim_mm_s':>11} {'orbit_mm':>10} {'x_max_mm':>10}")

for (r, m) in POINTS:
    res = run_one(r, m)
    print(f"{int(r*100):>5} {m:>8.2f} {res['B_mean_µT']:>11.1f} "
          f"{res['F_mean_µN']:>11.3f} {res['swim_mm_s']:>11.3f} "
          f"{res['orbit_mm']:>10.4f} {res['x_max_mm']:>10.3f}")
