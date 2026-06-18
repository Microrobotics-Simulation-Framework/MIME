#!/usr/bin/env python3
"""EP-1 leg (c) fixed — populate the RUNNING branch by ramping UP through the
saddle-node first, THEN ramp DOWN to find the homoclinic re-lock f_si↓.

(The first attempt started from rest at high f, where the body can't catch the field
at all → stuck near 0 spin, never on the running branch. Fix: up-through-saddle-node,
then down.) Produces the full hysteresis-loop figure.

Run: python scripts/mod1_rampdown_v2.py
"""
from __future__ import annotations
import os, sys, math
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import jax.numpy as jnp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mime.core.quaternion import rotate_vector_inverse
from mod1_saddle_node_ramp import build, make_node


def run(node, mB, dt, f_of_t, n):
    st = node.initial_state(); phi = 0.0; psi = 0.0
    fs = np.empty(n); sp = np.empty(n)
    for i in range(n):
        t = i * dt; f = f_of_t(t); phi += 2 * math.pi * f * dt
        q = st["orientation"]
        wz = float(rotate_vector_inverse(q, st["angular_velocity"])[2])
        psi += wz * dt
        st = node.update(st, {"magnetic_torque": jnp.array([0., 0., mB * math.sin(phi - psi)])}, dt)
        fs[i] = f; sp[i] = wz / (2 * math.pi)
        if not np.isfinite(wz): fs = fs[:i]; sp = sp[:i]; break
    return fs, sp


def main():
    R, I_a, m_eff = build()
    C = abs(R[5, 5]); mB = 2 * 8.4e-4 * 1.2e-3; dt = 1e-4
    f_n = math.sqrt(mB / I_a) / (2 * math.pi)
    f_si_an = (4 / math.pi) * f_n; f_so_an = mB / C / (2 * math.pi)
    fmax = 800.0; t_up = 6.0; t_hold = 0.3; t_dn = 10.0
    def sched(t):
        if t < t_up: return fmax * (t / t_up)              # up through saddle-node → running
        if t < t_up + t_hold: return fmax
        return fmax * max(0.0, 1.0 - (t - t_up - t_hold) / t_dn)   # down → homoclinic re-lock
    n = int((t_up + t_hold + t_dn) / dt)
    fs, sp = run(make_node(R, I_a, m_eff, dt), mB, dt, sched, n)
    i_dn = int((t_up + t_hold) / dt)
    fd, sd = fs[i_dn:], sp[i_dn:]
    # re-lock: scanning DOWN, first f where spin/f > 0.7 sustained 20 ms
    f_si = None; run_c = 0
    for i in range(len(fd)):
        if fd[i] < 2: continue
        if sd[i] / fd[i] > 0.7:
            run_c += 1
            if run_c > 200: f_si = fd[i]; break
        else: run_c = 0
    print("=" * 64)
    print("EP-1 leg (c) fixed — homoclinic re-lock from the running branch")
    print("=" * 64)
    print(f"  analytic: f_si (homoclinic) = {f_si_an:.1f} Hz, f_so = {f_so_an:.0f} Hz")
    print(f"  measured f_si↓ (ramp-down re-lock) = {f_si:.1f} Hz" if f_si
          else "  f_si↓: re-lock not detected")
    if f_si:
        print(f"  vs analytic {f_si_an:.1f} Hz  (ratio {f_si/f_si_an:.2f})")
    # figure
    fig, ax = plt.subplots(figsize=(8, 5))
    iu = int(t_up / dt)
    ax.plot(fs[:iu], sp[:iu], ",", color="tab:blue", label="ramp up → saddle-node")
    ax.plot(fd, sd, ",", color="tab:red", label="ramp down → homoclinic")
    ax.plot([0, fmax], [0, fmax], "k--", lw=0.7, alpha=0.5, label="synchronous")
    ax.axvline(f_so_an, color="tab:blue", ls=":", lw=1, label=f"f_so≈{f_so_an:.0f}")
    if f_si: ax.axvline(f_si, color="tab:red", ls=":", lw=1, label=f"f_si↓≈{f_si:.0f}")
    ax.set_xlabel("drive frequency (Hz)"); ax.set_ylabel("body spin (Hz)")
    ax.set_title("FL-9 step-out HYSTERESIS LOOP (1-DOF, ζ≈0.014)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("/tmp/ep1_hysteresis_loop_v2.png", dpi=130)
    print("  saved /tmp/ep1_hysteresis_loop_v2.png")
    print("=" * 64)


if __name__ == "__main__":
    main()
