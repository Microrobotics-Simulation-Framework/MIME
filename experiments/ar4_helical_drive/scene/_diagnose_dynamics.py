"""Step the AR4 helical drive graph and watch what the helix does.

Reports per-step position, orientation quat norm, |B| at the body,
gradient-force magnitude, magnetic torque magnitude, and detects
discontinuous jumps that would manifest visually as teleportation.
"""
from __future__ import annotations
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys
sys.path.insert(0, "/home/nick/MSF/MIME/src")

import importlib.util
import jax
import jax.numpy as jnp
import numpy as np

# Load params.py the same way the runner does (exec into a dict).
PARAMS_PATH = "/home/nick/MSF/MIME/experiments/ar4_helical_drive/physics/params.py"
ns: dict = {}
with open(PARAMS_PATH) as fh:
    exec(fh.read(), ns)
params = {k: v for k, v in ns.items() if not k.startswith("_") and k.isupper()}

from experiments.ar4_helical_drive.physics.setup import build_graph  # type: ignore[import]

gm = build_graph(params)

# Inject a constant motor command (10 Hz commanded velocity).
freq_hz = float(params.get("FIELD_FREQUENCY_HZ", 10.0))
commanded_velocity = jnp.asarray(2 * jnp.pi * freq_hz, dtype=jnp.float32)
arm_n_dof = 6
zero_torques = jnp.zeros(arm_n_dof, dtype=jnp.float32)

dt = float(params["DT_PHYS"])
N = 600  # 0.3 s of simulation

print(f"dt = {dt}, steps = {N} (sim time {N * dt:.3f} s)")
print(f"motor commanded ω = {float(commanded_velocity):.2f} rad/s = {freq_hz:.1f} Hz")
print()
print(f"{'step':>5} {'t_ms':>6} "
      f"{'body_pos.x_mm':>14} {'body_pos.y_mm':>14} {'body_pos.z_mm':>14} "
      f"{'|q|':>8} {'rotor_θ':>8} "
      f"{'|B|_µT':>10} {'|F_mag|_nN':>12} {'|τ_mag|_Nm':>12}")

ext = {
    "motor": {"commanded_velocity": commanded_velocity},
    "arm": {"commanded_joint_torques": zero_torques},
}

prev_pos = None
for step in range(N):
    state = gm.step(ext)
    body_pos = np.asarray(state["body"]["position"])
    body_q = np.asarray(state["body"]["orientation"])
    motor_th = float(state["motor"]["angle"])
    B = np.asarray(state["ext_magnet"]["field_vector"])
    F_mag = np.asarray(state["magnet"]["magnetic_force"])
    tau_mag = np.asarray(state["magnet"]["magnetic_torque"])

    qnorm = float(np.linalg.norm(body_q))
    Bn = float(np.linalg.norm(B)) * 1e6  # µT
    Fn = float(np.linalg.norm(F_mag)) * 1e9  # nN
    tn = float(np.linalg.norm(tau_mag))

    if step % 20 == 0 or step < 5 or step in {N - 1}:
        print(f"{step:>5} {step*dt*1000:>6.1f} "
              f"{body_pos[0]*1000:>14.4f} {body_pos[1]*1000:>14.4f} {body_pos[2]*1000:>14.4f} "
              f"{qnorm:>8.5f} {motor_th:>8.3f} "
              f"{Bn:>10.2f} {Fn:>12.3f} {tn:>12.3e}")

    # Jump detection: body moves more than vessel diameter in one step
    if prev_pos is not None:
        dp = np.linalg.norm(body_pos - prev_pos)
        if dp > 5e-3:  # > 5 mm in 0.5 ms = 10 m/s, suspicious
            print(f"  ! JUMP at step {step}: dp = {dp*1000:.2f} mm")
            print(f"    prev = {prev_pos}, now = {body_pos}")
    prev_pos = body_pos
