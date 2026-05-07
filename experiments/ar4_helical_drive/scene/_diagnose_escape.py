"""Trace the body until it escapes the vessel.  Print a high-density
log near the moment of escape so we can see what triggers the jump.
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

# Run for 30 s.
N = 60000
print(f"dt = {dt}, N = {N}, sim time = {N*dt:.2f} s")

prev_pos = None
escape_step = None
trace = []
for step in range(N):
    s = gm.step(ext)
    pos = np.asarray(s["body"]["position"])
    vel = np.asarray(s["body"]["velocity"])
    omega_b = np.asarray(s["body"]["angular_velocity"])
    F_mag = np.asarray(s["magnet"]["magnetic_force"])
    F_drag_lub = np.asarray(s["lub"]["drag_force"]) if "lub" in s else None
    F_drag_mlp = np.asarray(s["mlp_drag"]["drag_force"])
    trace.append({
        "step": step,
        "pos": pos.copy(),
        "vel": vel.copy(),
        "omega": omega_b.copy(),
        "F_mag": float(np.linalg.norm(F_mag)),
        "F_drag_lub": float(np.linalg.norm(F_drag_lub)) if F_drag_lub is not None else 0.0,
        "F_drag_mlp": float(np.linalg.norm(F_drag_mlp)),
    })
    # Detect escape: |pos| > vessel half_length+epsilon, or radial > vessel radius+epsilon
    r_yz = np.sqrt(pos[1]**2 + pos[2]**2)
    if escape_step is None and (np.abs(pos[0]) > 0.5 + 1e-3 or r_yz > 1e-3 * 1.05):
        escape_step = step
        print(f"\n*** ESCAPE at step {step} (t={step*dt*1000:.1f} ms): "
              f"pos={pos*1000} mm, r_yz={r_yz*1000:.4f} mm")
    if prev_pos is not None and np.linalg.norm(pos - prev_pos) > 5e-3:
        if escape_step is None:
            print(f"  ! big jump at step {step}: dp = {np.linalg.norm(pos - prev_pos)*1000:.2f} mm")
    prev_pos = pos

# Print the trace centred on the escape (or last step if no escape)
center = escape_step if escape_step is not None else N - 1
start = max(0, center - 30)
end = min(N, center + 5)
print(f"\nTrace around step {center}:")
print(f"{'step':>5} {'t_ms':>7} "
      f"{'x_mm':>9} {'y_mm':>9} {'z_mm':>9} "
      f"{'vx':>9} {'vy':>9} {'vz':>9} "
      f"{'|F_mag|µN':>10} {'|F_lub|µN':>11} {'|F_mlp|µN':>11}")
for i in range(start, end):
    t = trace[i]
    print(f"{t['step']:>5} {t['step']*dt*1000:>7.1f} "
          f"{t['pos'][0]*1000:>9.4f} {t['pos'][1]*1000:>9.4f} {t['pos'][2]*1000:>9.4f} "
          f"{t['vel'][0]:>9.4f} {t['vel'][1]:>9.4f} {t['vel'][2]:>9.4f} "
          f"{t['F_mag']*1e6:>10.3f} {t['F_drag_lub']*1e6:>11.3f} {t['F_drag_mlp']*1e6:>11.3f}")
