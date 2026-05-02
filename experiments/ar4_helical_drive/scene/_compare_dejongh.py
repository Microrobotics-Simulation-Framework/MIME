"""Side-by-side: legacy dejongh chain (uniform field, known stable) vs
our AR4 chain (Motor + PermanentMagnet, allegedly unstable).

Both run for 0.3 s with FL-9 in 1/4" vessel, 10 Hz drive, water.
Reports trajectory and detects instabilities.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, "/home/nick/MSF/MIME/src")
sys.path.insert(0, "/home/nick/MSF/MIME")

import jax.numpy as jnp
import numpy as np

DT = 5e-4
N_STEPS = 600

# ────────── (A) Legacy dejongh chain ──────────────────────────────────
from mime.experiments.dejongh import build_graph as build_legacy
gm_a = build_legacy(
    design_name="FL-9", vessel_name='1/4"',
    mu_Pa_s=1e-3, delta_rho=410.0, dt=DT,
    use_lubrication=True, lubrication_epsilon_mm=0.15,
)
# Seed body pre-sunk
init_pos = jnp.array([0.0, -1.0e-3, 0.0], dtype=jnp.float32)
state_a = dict(gm_a.get_node_state("body"))
state_a["position"] = init_pos
gm_a.set_node_state("body", state_a)

ext_a = {
    "field": {
        "frequency_hz": jnp.asarray(10.0, dtype=jnp.float32),
        "field_strength_mt": jnp.asarray(1.2, dtype=jnp.float32),
    },
}

# ────────── (B) Our AR4 chain ──────────────────────────────────────
PARAMS_PATH = "/home/nick/MSF/MIME/experiments/ar4_helical_drive/physics/params.py"
ns: dict = {}
with open(PARAMS_PATH) as fh:
    exec(fh.read(), ns)
params_b = {k: v for k, v in ns.items() if not k.startswith("_") and k.isupper()}
# Force lubrication on for an apples-to-apples comparison.
params_b["USE_LUBRICATION"] = True

from experiments.ar4_helical_drive.physics.setup import build_graph as build_b
gm_b = build_b(params_b)
ext_b = {
    "motor": {"commanded_velocity": jnp.asarray(2*jnp.pi*10.0, dtype=jnp.float32)},
    "arm":   {"commanded_joint_torques": jnp.zeros(6, dtype=jnp.float32)},
}

print(f"\n{'='*70}\nLegacy dejongh: uniform B field, vessel along z, gravity -y\n{'='*70}")
print(f"{'step':>5} {'t_ms':>6} {'x_mm':>10} {'y_mm':>10} {'z_mm':>10} {'|v|_m/s':>10}")
prev_pos_a = None
jumps_a = 0
for step in range(N_STEPS):
    s = gm_a.step(ext_a)
    pos = np.asarray(s["body"]["position"])
    vel = np.asarray(s["body"]["velocity"])
    if step % 50 == 0 or step == N_STEPS - 1:
        print(f"{step:>5} {step*DT*1000:>6.1f} "
              f"{pos[0]*1000:>10.4f} {pos[1]*1000:>10.4f} {pos[2]*1000:>10.4f} "
              f"{np.linalg.norm(vel):>10.4f}")
    if prev_pos_a is not None and np.linalg.norm(pos - prev_pos_a) > 5e-3:
        jumps_a += 1
    prev_pos_a = pos

print(f"  → instability jumps detected: {jumps_a}")

print(f"\n{'='*70}\nOur AR4 chain: point-dipole B, vessel along x, gravity -z\n{'='*70}")
print(f"{'step':>5} {'t_ms':>6} {'x_mm':>10} {'y_mm':>10} {'z_mm':>10} {'|v|_m/s':>10}")
prev_pos_b = None
jumps_b = 0
for step in range(N_STEPS):
    s = gm_b.step(ext_b)
    pos = np.asarray(s["body"]["position"])
    vel = np.asarray(s["body"]["velocity"])
    F_grav = np.asarray(s["gravity"]["gravity_force"])
    F_mag = np.asarray(s["magnet"]["magnetic_force"])
    F_drag = np.asarray(s["lub"]["drag_force"]) if "lub" in s else np.asarray(s["mlp_drag"]["drag_force"])
    if step % 50 == 0 or step == N_STEPS - 1:
        print(f"{step:>5} {step*DT*1000:>6.1f} "
              f"{pos[0]*1000:>10.4f} {pos[1]*1000:>10.4f} {pos[2]*1000:>10.4f} "
              f"{np.linalg.norm(vel):>10.4f}   "
              f"F_g={np.linalg.norm(F_grav)*1e6:>7.2f}µN  "
              f"F_mag={np.linalg.norm(F_mag)*1e6:>6.2f}µN  "
              f"F_drag={np.linalg.norm(F_drag)*1e6:>7.2f}µN")
    if prev_pos_b is not None and np.linalg.norm(pos - prev_pos_b) > 5e-3:
        jumps_b += 1
    prev_pos_b = pos

print(f"  → instability jumps detected: {jumps_b}")
