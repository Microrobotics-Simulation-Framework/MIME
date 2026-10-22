#!/usr/bin/env python3
"""EP-1 — full step-out HYSTERESIS LOOP on the standalone 1-DOF spin model.

Real RigidBodyNode inertial integrator + confined-BEM rotational drag. Measures:
  (a) f_so↑  — slow ramp UP from rest, no perturbation  → saddle-node branch
  (c) f_si↓  — ramp DOWN from a desynchronised (running) state → homoclinic re-lock
  (d) trip    — physically-motivated perturbation at fixed freq in the window
Reports f_so↑, f_si↓, the hysteresis gap, vs analytic (mB/C ≈ 622 Hz, (4/π)ω_n ≈ 22 Hz),
and saves the loop figure. See plans/MIME_EP1_hysteresis_protocol.md.

Run: python scripts/mod1_hysteresis_loop.py
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
from mime.nodes.robot.rigid_body import RigidBodyNode
from mime.core.quaternion import rotate_vector_inverse
from mod1_saddle_node_ramp import build, make_node


def integrate(node, mB, dt, f_of_t, n, perturb_A=0.0, perturb_hz=13.0, w0=0.0):
    """Integrate the driven pendulum with field-frequency schedule f_of_t(step).
    Returns arrays (f, spin_hz). Optional spin-axis perturbation A·sin(2π fp t)."""
    st = node.initial_state()
    if w0:
        st = {**st, "angular_velocity": jnp.array([0.0, 0.0, w0])}
    phi = 0.0; psi = 0.0
    fs = np.empty(n); sp = np.empty(n)
    for i in range(n):
        t = i * dt
        f = f_of_t(t)
        phi += 2.0 * math.pi * f * dt
        q = st["orientation"]
        wz_b = float(rotate_vector_inverse(q, st["angular_velocity"])[2])
        psi += wz_b * dt
        tau = mB * math.sin(phi - psi)
        if perturb_A:
            tau += perturb_A * math.sin(2.0 * math.pi * perturb_hz * t)
        st = node.update(st, {"magnetic_torque": jnp.array([0.0, 0.0, tau])}, dt)
        fs[i] = f; sp[i] = wz_b / (2.0 * math.pi)
        if not np.isfinite(wz_b):
            fs = fs[:i]; sp = sp[:i]; break
    return fs, sp


def detect_break(fs, sp, lock=True):
    """First freq where |spin/f - 1| crosses the lock/unlock boundary (0.5) for >20ms."""
    n = len(fs); run = 0
    for i in range(n):
        if fs[i] < 3.0: continue
        locked = abs(sp[i] / fs[i] - 1.0) < 0.5
        hit = (not locked) if lock else locked
        run = run + 1 if hit else 0
        if run > 200:
            return fs[i]
    return None


def main():
    R, I_a, m_eff = build()
    C = abs(R[5, 5]); mB = 2 * 8.4e-4 * 1.2e-3; dt = 1e-4
    f_so_an = mB / C / (2 * math.pi)
    f_n = math.sqrt(mB / I_a) / (2 * math.pi)
    f_si_an = (4 / math.pi) * f_n
    print("=" * 70)
    print("EP-1 step-out HYSTERESIS LOOP (standalone 1-DOF)")
    print("=" * 70)
    print(f"  analytic: omega_n={f_n:.1f}Hz  homoclinic f_si={f_si_an:.1f}Hz  "
          f"saddle-node f_so=mB/C={f_so_an:.0f}Hz")
    fmax = 800.0

    # (a) slow ramp UP from rest, no perturbation
    t_up = 8.0
    fa, sa = integrate(make_node(R, I_a, m_eff, dt), mB, dt,
                       lambda t: fmax * min(1.0, t / t_up), int((t_up + 0.3) / dt))
    f_so_up = detect_break(fa, sa, lock=True)
    print(f"  (a) ramp-up f_so↑   = {f_so_up if f_so_up else '>'+str(fmax)} Hz")

    # (c) ramp DOWN from a running (desync) state: start from rest at fmax (→ running),
    #     hold, then ramp down to 0; detect re-lock.
    t_hold, t_dn = 0.3, 8.0
    def f_down(t):
        if t < t_hold: return fmax
        return fmax * max(0.0, 1.0 - (t - t_hold) / t_dn)
    fc, sc = integrate(make_node(R, I_a, m_eff, dt), mB, dt, f_down,
                       int((t_hold + t_dn + 0.2) / dt))
    # restrict re-lock detection to the ramp-down portion
    i0 = int(t_hold / dt)
    f_si_dn = detect_break(fc[i0:], sc[i0:], lock=False)
    print(f"  (c) ramp-down f_si↓ = {f_si_dn if f_si_dn else 'no re-lock'} Hz")

    # (d) perturbation trip: hold at fixed freqs in the window with a physical
    #     perturbation (~5° moment misalignment → 9% mB ≈ 1.8e-7 N·m).
    A = 1.8e-7
    print(f"  (d) perturbation trip (A={A:.1e} N·m ≈ {A/mB*100:.0f}% mB, ~5° misalign):")
    trip = None
    for fhold in (40, 80, 150, 250, 400):
        # come up locked to fhold via a quick ramp, then hold with perturbation
        def f_sched(t, fh=fhold):
            return fh * min(1.0, t / 0.5)
        n = int(1.5 / dt)
        fd, sd = integrate(make_node(R, I_a, m_eff, dt), mB, dt, f_sched, n,
                           perturb_A=A)
        held = sd[int(1.0/dt):]; fh_arr = fd[int(1.0/dt):]
        locked = np.mean(np.abs(held / np.maximum(fh_arr, 1e-9) - 1.0) < 0.5) > 0.5
        print(f"      f={fhold:4d} Hz: {'LOCKED' if locked else 'TRIPPED (desync)'}")
        if not locked and trip is None:
            trip = fhold
    print("-" * 70)
    print(f"  RESULT: f_so↑ = {f_so_up if f_so_up else '>800'} Hz, "
          f"f_si↓ = {f_si_dn} Hz  (analytic {f_so_an:.0f} / {f_si_an:.1f})")
    if f_so_up and f_si_dn:
        print(f"  hysteresis gap = {f_so_up - f_si_dn:.0f} Hz  "
              f"(bistable window {f_si_dn:.0f}–{f_so_up:.0f} Hz)")
    print(f"  perturbation ({A/mB*100:.0f}% mB) first trips at ~{trip} Hz")

    # figure: hysteresis loop (spin vs drive, up and down)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fa, sa, ".", ms=1, alpha=0.4, color="tab:blue", label="ramp up (a)")
    ax.plot(fc[i0:], sc[i0:], ".", ms=1, alpha=0.4, color="tab:red", label="ramp down (c)")
    ax.plot([0, fmax], [0, fmax], "k--", lw=0.8, alpha=0.6, label="synchronous (spin=drive)")
    if f_so_up: ax.axvline(f_so_up, color="tab:blue", ls=":", label=f"f_so↑≈{f_so_up:.0f}")
    if f_si_dn: ax.axvline(f_si_dn, color="tab:red", ls=":", label=f"f_si↓≈{f_si_dn:.0f}")
    ax.set_xlabel("drive frequency (Hz)"); ax.set_ylabel("body spin (Hz)")
    ax.set_title("FL-9 step-out hysteresis loop (underdamped, ζ≈0.014)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("/tmp/ep1_hysteresis_loop.png", dpi=130)
    print("  saved /tmp/ep1_hysteresis_loop.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
