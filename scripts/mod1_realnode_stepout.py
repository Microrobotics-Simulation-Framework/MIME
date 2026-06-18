#!/usr/bin/env python3
"""MOD-1 / M1 real-node check — drive the ACTUAL RigidBodyNode inertial integrator
with the real confined-BEM 6×6 drag and confirm (a) numerical stability of the new
inertial-drag wiring (spin-down) and (b) lock→step-out near the analytical
prediction (~20-40 Hz; overdamped ceiling 621 Hz; measured 27 Hz).

This exercises the SAME code path M2 couples into the schwarz graph (the
`use_inertial=True` + confined `resistance_matrix` branch in rigid_body.py), so it
de-risks the wiring before touching the experiment graph. The magnetic restoring
torque T_z = m·B·sin(φ_field − ψ_body) is applied in the harness (driven pendulum);
everything else (inertia, BEM drag, symplectic ω-then-orientation order) is the node.

Run: python scripts/mod1_realnode_stepout.py
"""
from __future__ import annotations
import os, sys, math
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import jax.numpy as jnp
from mime.nodes.robot.rigid_body import RigidBodyNode
from mime.core.quaternion import rotate_vector_inverse


def build():
    """FL-9 confined resistance (SI) + inertia, reusing the schwarz config."""
    from mime.experiments import schwarz_vessel_helix as S
    from mime.nodes.environment.stokeslet.cylinder_wall_table import load_wall_table
    mu = S._DEFAULTS["MU_PA_S"]
    mesh = {"N_THETA": 24, "N_ZETA": 36}
    m = S._screw_mesh(mesh)
    table = load_wall_table(S._DEFAULTS["WALL_TABLE"])
    near = S._near_node(mesh, m, table, mu)
    R = np.asarray(near.resistance_matrix_si())
    # inertia from the analytical spike
    R_cyl, L, eps = 1.56e-3, 7.47e-3, 0.33
    rho_body, rho_water = 1410.0, 1000.0
    a2 = 1.0 + eps**2 / 2.0
    a4 = 1.0 + 3.0 * eps**2 + (3.0 / 8.0) * eps**4
    V = math.pi * R_cyl**2 * a2 * L
    m_body = rho_body * V
    I_body = rho_body * (math.pi / 2.0) * R_cyl**4 * a4 * L
    return R, I_body, m_body


def make_node(R, I_eff, m_eff, dt):
    return RigidBodyNode("body", dt, use_inertial=True, I_eff=float(I_eff),
                         m_eff=float(m_eff), resistance_matrix=R,
                         locked_spin_axis_body=(0.0, 0.0, 1.0),
                         semi_major_axis_m=3.735e-3, semi_minor_axis_m=1.56e-3,
                         fluid_viscosity_pa_s=1.0e-3)


def spin_axis(q, state_axis=(0.0, 0.0, 1.0)):
    """body-z spin component of ω, read in body frame."""
    return None


def run_drive(node, mB, f_drive_hz, dt, t_total, omega0=0.0, perturb=0.0):
    """Drive at fixed field frequency; return mean body-z spin over the last 40%."""
    st = node.initial_state()
    st = {**st, "angular_velocity": jnp.array([0.0, 0.0, omega0])}
    n = int(t_total / dt)
    omega_f = 2 * math.pi * f_drive_hz
    psi = 0.0           # accumulated body spin angle about z (body frame)
    spins = []
    for i in range(n):
        phi = omega_f * (i * dt)
        q = st["orientation"]
        # body-frame spin angle: integrate body-z component of ω
        wz_b = float(rotate_vector_inverse(q, st["angular_velocity"])[2])
        psi += wz_b * dt
        lag = phi - psi + (perturb if i == int(0.1 * n) else 0.0)
        T = jnp.array([0.0, 0.0, mB * math.sin(lag)])
        bi = {"magnetic_torque": T}
        st = node.update(st, bi, dt)
        if not np.all(np.isfinite(np.asarray(st["angular_velocity"]))):
            return float("nan"), True
        if i > 0.6 * n:
            spins.append(float(rotate_vector_inverse(st["orientation"],
                                                     st["angular_velocity"])[2]))
    return float(np.mean(spins)) / (2 * math.pi), False  # mean spin freq (Hz)


def main():
    R, I_eff, m_eff = build()
    C_rot = abs(R[5, 5])
    dt = 1e-4
    print("=" * 74)
    print("MOD-1 real-node check — actual RigidBodyNode inertial + confined BEM drag")
    print("=" * 74)
    print(f"  C_rot={C_rot:.3e}  I_eff={I_eff:.3e}  dt={dt:.0e}")
    wn = math.sqrt((2 * 8.4e-4 * 1.2e-3) / I_eff)
    print(f"  ω_n={wn:.1f} rad/s  homoclinic (4/π)ω_n={(4/math.pi)*wn/(2*math.pi):.1f} Hz"
          f"  overdamped ceiling={(2*8.4e-4*1.2e-3)/(2*math.pi*C_rot):.0f} Hz")
    print("-" * 74)

    # (a) spin-down: no drive, initial spin → exponential decay, time const I/C_rot.
    node = make_node(R, I_eff, m_eff, dt)
    st = node.initial_state()
    w0 = 100.0
    st = {**st, "angular_velocity": jnp.array([0.0, 0.0, w0])}
    tau = I_eff / C_rot
    t_check = tau
    n = int(t_check / dt)
    for _ in range(n):
        st = node.update(st, {}, dt)
    wz = float(st["angular_velocity"][2])
    analytic = w0 * math.exp(-1.0)
    print(f"  (a) spin-down @ t=τ={tau:.4f}s: ω_z={wz:.3f} vs analytic {analytic:.3f} "
          f"(err {abs(wz-analytic)/analytic*100:.1f}%)  finite={np.isfinite(wz)}")
    print("-" * 74)

    # (b) lock→step-out: ramp drive frequency, with a perturbation kick.
    mB = 2 * 8.4e-4 * 1.2e-3
    print("  (b) driven pendulum (ramp + perturbation), mean body-z spin freq:")
    print(f"      {'f_drive':>8}  {'f_spin':>8}  {'locked?':>8}")
    last_locked = None
    f_so = None
    for f in [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80]:
        f_spin, blew = run_drive(node, mB, f, dt, t_total=0.6, omega0=0.0,
                                 perturb=0.5)
        locked = (not blew) and abs(f_spin - f) < 0.15 * f
        tag = "NaN" if blew else ("lock" if locked else "SLIP")
        print(f"      {f:8.0f}  {f_spin:8.2f}  {tag:>8}")
        if last_locked and not locked and f_so is None:
            f_so = f
        last_locked = locked
    print("-" * 74)
    print(f"  step-out between last lock and first slip ≈ {f_so} Hz "
          f"(expect ~20-40; NOT 621)")
    print("=" * 74)


if __name__ == "__main__":
    main()
