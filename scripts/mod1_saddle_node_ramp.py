#!/usr/bin/env python3
"""EP-1 leg (a) — standalone 1-DOF saddle-node test.

Slow quasi-static frequency ramp UP FROM REST, NO perturbation, on the real
RigidBodyNode inertial integrator with the confined-BEM rotational drag. Tests the
central underdamped claim: an undisturbed locked branch tracks far past the homoclinic
floor (~22 Hz) toward the saddle-node f_so = mB/C ≈ 622 Hz, and the apparent step-out
RISES as the ramp slows (direct evidence of the saddle-node branch).

PASS: clean-ramp step-out > 200 Hz and increases as the ramp rate decreases.
FAIL: steps out at ~22-40 Hz with no perturbation (→ re-examine C or the integrator).

Run: python scripts/mod1_saddle_node_ramp.py
"""
from __future__ import annotations
import os, sys, math
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import jax.numpy as jnp
from mime.nodes.robot.rigid_body import RigidBodyNode
from mime.core.quaternion import rotate_vector_inverse


def build():
    """Confined-BEM rotational drag C and axial inertia (FL-9), reusing schwarz cfg."""
    from mime.experiments import schwarz_vessel_helix as S
    from mime.nodes.environment.stokeslet.cylinder_wall_table import load_wall_table
    mu = S._DEFAULTS["MU_PA_S"]
    mesh = {"N_THETA": 24, "N_ZETA": 36}
    m = S._screw_mesh(mesh)
    table = load_wall_table(S._DEFAULTS["WALL_TABLE"])
    near = S._near_node(mesh, m, table, mu)
    R = np.asarray(near.resistance_matrix_si())
    R_cyl, L, eps = 1.56e-3, 7.47e-3, 0.33
    rho_body = 1410.0
    a2 = 1.0 + eps**2 / 2.0
    a4 = 1.0 + 3.0 * eps**2 + (3.0 / 8.0) * eps**4
    V = math.pi * R_cyl**2 * a2 * L
    m_body = rho_body * V
    I_axial = rho_body * (math.pi / 2.0) * R_cyl**4 * a4 * L
    return R, I_axial, m_body


def make_node(R, I_a, m_eff, dt):
    return RigidBodyNode("body", dt, use_inertial=True, I_eff=float(I_a),
                         m_eff=float(m_eff), resistance_matrix=R,
                         locked_spin_axis_body=(0.0, 0.0, 1.0),
                         semi_major_axis_m=3.735e-3, semi_minor_axis_m=1.56e-3,
                         fluid_viscosity_pa_s=1.0e-3)


def ramp_stepout(node, mB, dt, f_max, t_ramp, hold_s=0.2):
    """Ramp drive 0→f_max over t_ramp (then hold). Return the drive freq at which the
    body spin first falls persistently below 50% of the drive (step-out)."""
    st = node.initial_state()
    n = int((t_ramp + hold_s) / dt)
    phi = 0.0      # accumulated field angle
    psi = 0.0      # accumulated body spin angle (body-z)
    f_so = None
    below = 0
    for i in range(n):
        t = i * dt
        f = f_max * min(1.0, t / t_ramp)
        phi += 2.0 * math.pi * f * dt
        q = st["orientation"]
        wz_b = float(rotate_vector_inverse(q, st["angular_velocity"])[2])
        psi += wz_b * dt
        lag = phi - psi
        T = jnp.array([0.0, 0.0, mB * math.sin(lag)])
        st = node.update(st, {"magnetic_torque": T}, dt)
        spin = wz_b / (2.0 * math.pi)
        if f > 5.0 and spin < 0.5 * f:
            below += 1
            if below > 200 and f_so is None:    # 20 ms of sustained slip
                f_so = f
                break
        else:
            below = 0
        if not np.isfinite(float(st["angular_velocity"][2])):
            return float("nan")
    return f_so if f_so is not None else f_max  # f_max = "never stepped out in range"


def main():
    R, I_a, m_eff = build()
    C = abs(R[5, 5]); mB = 2 * 8.4e-4 * 1.2e-3
    dt = 1e-4
    f_so_analytic = mB / C / (2 * math.pi)
    f_n = math.sqrt(mB / I_a) / (2 * math.pi)
    print("=" * 70)
    print("EP-1 leg (a): saddle-node ramp test (no perturbation, from rest)")
    print("=" * 70)
    print(f"  C={C:.3e}  I_axial={I_a:.3e}  mB={mB:.3e}")
    print(f"  analytic: omega_n={f_n:.1f} Hz, homoclinic={(4/math.pi)*f_n:.1f} Hz, "
          f"saddle-node mB/C={f_so_analytic:.0f} Hz")
    print("-" * 70)
    print(f"  {'ramp rate':>12} {'t_ramp':>7} {'step-out':>10}")
    f_max = 800.0
    results = []
    for t_ramp in (1.0, 2.0, 4.0, 8.0):
        fso = ramp_stepout(node := make_node(R, I_a, m_eff, dt), mB, dt, f_max, t_ramp)
        rate = f_max / t_ramp
        tag = ">800 (no step-out)" if fso >= f_max else f"{fso:.0f} Hz"
        print(f"  {rate:8.0f} Hz/s {t_ramp:7.1f}s {tag:>14}")
        results.append((t_ramp, fso))
    print("-" * 70)
    rising = all(results[i][1] <= results[i+1][1] + 5 for i in range(len(results)-1))
    hi = max(r[1] for r in results)
    print(f"  step-out rises (or saturates high) as ramp slows: {rising}")
    print(f"  max step-out observed: {hi:.0f} Hz  (analytic saddle-node {f_so_analytic:.0f} Hz)")
    verdict = "PASS" if (hi > 200 and rising) else "FAIL"
    print(f"  VERDICT: {verdict}  "
          f"(PASS = >200 Hz and rising toward {f_so_analytic:.0f}; "
          f"FAIL = steps out at 22-40 Hz)")
    print("=" * 70)


if __name__ == "__main__":
    main()
