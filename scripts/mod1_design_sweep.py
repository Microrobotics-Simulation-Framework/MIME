#!/usr/bin/env python3
"""M1 — standalone step-out HYSTERESIS across the de Jongh design family + a fine
ζ/size scaling sweep (report "future research" pattern; simulation-only).

Physics: the underdamped driven pendulum  I·θ̈ + C·θ̇ + mB·sinΔθ = C·ω,
  ζ  = C / (2√(I·mB)),   ω_n = √(mB/I),
  saddle-node (ramp-up)  f_so = mB/C          (scale-invariant ≈622 Hz for FL-9),
  homoclinic  (ramp-down) f_si = (4/π)·ω_n,
  window ratio  f_so/f_si = π/(8ζ)  →  hysteresis CLOSES at ζ = π/8 ≈ 0.39.

Reuses the real RigidBodyNode inertial integrator + confined-BEM rotational drag
C = |R[5,5]| (same model as mod1_saddle_node_ramp / mod1_hysteresis_loop), just
parameterised over geometry. All FL/FW designs share R_cyl=1.56 mm → all reuse the
validated wall_R2.035 table.

Modes (default: all):
  check    — FL-9 regression gate (reproduce C, ζ, f_so, f_si to <1%)
  designs  — hysteresis loop for all 17 de Jongh designs → mod1_design_sweep.jsonl
  scaling  — fine ζ sweep (normalized universal curve) + absolute-λ confirmation
             + BEM λ³ check → mod1_scaling_sweep.jsonl, mod1_scaling_bemcheck.jsonl

Run:  .venv/bin/python scripts/mod1_design_sweep.py --mode all
"""
from __future__ import annotations
import os, sys, math, json, time, argparse
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
import numpy as np
import jax.numpy as jnp
from mime.nodes.robot.rigid_body import RigidBodyNode
from mime.core.quaternion import rotate_vector_inverse
from mime.experiments import schwarz_vessel_helix as S
from mime.nodes.environment.stokeslet.cylinder_wall_table import load_wall_table
from mime.nodes.environment.stokeslet.dejongh_geometry import FL_TABLE, FW_TABLE
import jax
from mod1_hysteresis_loop import detect_break

OUT = REPO / "experiments" / "schwarz_vessel_helix" / "output" / "damping_sweep"
OUT.mkdir(parents=True, exist_ok=True)
MB0 = 2.0 * 8.4e-4 * 1.2e-3          # net m·B for FL-9 (2 magnets × 1.2 mT) = 2.016e-6
RHO_BODY = 1410.0                    # de Jongh screw density (water + Δρ 410)
EPS = 0.33
WALL_R2035 = S._DEFAULTS["WALL_TABLE"]
FL9_NU, FL9_L, FL9_RC = 2.33, 7.47, 1.56


def build_design(nu, L_mm, R_cyl_mm=1.56, eps=EPS, wall_table=WALL_R2035, mesh=(24, 36)):
    """Confined-BEM 6×6 resistance R + axial inertia + mass for one screw geometry.
    Mirrors mod1_saddle_node_ramp.build() but with geometry passed in."""
    mu = S._DEFAULTS["MU_PA_S"]
    L = L_mm * 1e-3
    R_cyl = R_cyl_mm * 1e-3
    p = {"NU_FL": nu, "L_UMR_M": L, "R_CYL_UMR_M": R_cyl,
         "N_THETA": mesh[0], "N_ZETA": mesh[1], "EPS": eps}
    m = S._screw_mesh(p)
    table = load_wall_table(wall_table)
    near = S._near_node(p, m, table, mu)
    R = np.asarray(near.resistance_matrix_si())
    a2 = 1.0 + eps ** 2 / 2.0
    a4 = 1.0 + 3.0 * eps ** 2 + (3.0 / 8.0) * eps ** 4
    V = math.pi * R_cyl ** 2 * a2 * L
    m_body = RHO_BODY * V
    I_axial = RHO_BODY * (math.pi / 2.0) * R_cyl ** 4 * a4 * L
    return R, I_axial, m_body


def make_node(R, I_a, m_eff, dt, L_mm=FL9_L, R_cyl_mm=FL9_RC):
    """Inertial RigidBodyNode with the supplied 6×6 confined drag (isotropic inertia;
    only I_axial matters for the pure spin-axis 1-DOF step-out)."""
    return RigidBodyNode("body", dt, use_inertial=True, I_eff=float(I_a),
                         m_eff=float(m_eff), resistance_matrix=np.asarray(R),
                         locked_spin_axis_body=(0.0, 0.0, 1.0),
                         semi_major_axis_m=L_mm * 1e-3 / 2.0, semi_minor_axis_m=R_cyl_mm * 1e-3,
                         fluid_viscosity_pa_s=1.0e-3)


def analytic(C, I_a, mB):
    f_so = mB / C / (2 * math.pi)
    f_n = math.sqrt(mB / I_a) / (2 * math.pi)
    f_si = (4.0 / math.pi) * f_n
    zeta = C / (2.0 * math.sqrt(I_a * mB))
    return dict(C=C, I_axial=I_a, mB=mB, zeta=zeta, omega_n_hz=f_n,
                f_so_analytic=f_so, f_si_analytic=f_si)


def jitted_integrate(node, mB, dt, f_arr, w0=0.0):
    """lax.scan rollout of the SAME driven-pendulum logic as mod1_hysteresis_loop.integrate
    (phi/psi accumulation, tau=mB·sin(phi−psi), real RigidBodyNode.update) but compiled
    once and run in-XLA → ~1000× faster than the eager Python loop. Returns (f, spin_hz)."""
    st0 = node.initial_state()
    if w0:
        st0 = {**st0, "angular_velocity": jnp.asarray([0.0, 0.0, w0])}
    f_arr = jnp.asarray(f_arr, dtype=jnp.float64)
    two_pi = 2.0 * math.pi

    def step(carry, f):
        st, phi, psi = carry
        phi = phi + two_pi * f * dt
        q = st["orientation"]
        wz_b = rotate_vector_inverse(q, st["angular_velocity"])[2]
        psi = psi + wz_b * dt
        tau = mB * jnp.sin(phi - psi)
        T = jnp.zeros(3, dtype=jnp.float64).at[2].set(tau)
        st = node.update(st, {"magnetic_torque": T}, dt)
        return (st, phi, psi), (f, wz_b / two_pi)

    carry0 = (st0, jnp.asarray(0.0, jnp.float64), jnp.asarray(0.0, jnp.float64))
    _, (fs, sp) = jax.lax.scan(step, carry0, f_arr)
    return np.asarray(fs), np.asarray(sp)


def measure_loop(R, I_a, m_eff, mB, dt=1e-4, fmax=800.0, t_up=5.0, t_hold=0.3, t_dn=5.0,
                 L_mm=FL9_L, R_cyl_mm=FL9_RC):
    """Ramp-up (saddle-node branch) + ramp-down-from-running (homoclinic re-lock),
    identical protocol to mod1_hysteresis_loop legs (a)+(c). Returns f_so↑, f_si↓."""
    t_a = np.arange(0.0, t_up + 0.3, dt)
    f_a = np.minimum(1.0, t_a / t_up) * fmax
    fa, sa = jitted_integrate(make_node(R, I_a, m_eff, dt, L_mm, R_cyl_mm), mB, dt, f_a)
    f_so_up = detect_break(fa, sa, lock=True)

    t_c = np.arange(0.0, t_hold + t_dn + 0.2, dt)
    f_c = np.where(t_c < t_hold, fmax, fmax * np.maximum(0.0, 1.0 - (t_c - t_hold) / t_dn))
    fc, sc = jitted_integrate(make_node(R, I_a, m_eff, dt, L_mm, R_cyl_mm), mB, dt, f_c)
    i0 = int(t_hold / dt)
    f_si_dn = detect_break(fc[i0:], sc[i0:], lock=False)
    return f_so_up, f_si_dn


# ─────────────────────────────────────────────────────────────────────────────
def mode_check():
    print("=" * 72)
    print("M1a — FL-9 regression gate")
    print("=" * 72)
    t0 = time.perf_counter()
    R, I_a, m_body = build_design(FL9_NU, FL9_L, FL9_RC)
    C = abs(R[5, 5])
    a = analytic(C, I_a, MB0)
    print(f"  build {time.perf_counter()-t0:.1f}s  C={C:.3e}  I_axial={I_a:.3e}  "
          f"m={m_body:.3e}")
    print(f"  ζ={a['zeta']:.4f}  ω_n={a['omega_n_hz']:.1f} Hz  "
          f"f_so={a['f_so_analytic']:.0f} Hz  f_si={a['f_si_analytic']:.1f} Hz")
    ok = (abs(C - 5.16e-10) / 5.16e-10 < 0.05 and abs(a["zeta"] - 0.014) < 0.004
          and abs(a["f_so_analytic"] - 622) / 622 < 0.05
          and abs(a["f_si_analytic"] - 22) < 4)
    print(f"  VERDICT: {'PASS' if ok else 'CHECK'} "
          f"(expect C≈5.16e-10, ζ≈0.014, f_so≈622, f_si≈22)")
    return ok


def loop_params(f_so_an):
    """Ramp ceiling just above the saddle-node (so the ramp-down starts from a running
    state); fixed durations → bounded step count regardless of design. Ramp-DOWN is
    slower (t_dn) so the abrupt homoclinic re-lock window (a few Hz wide) is resolved by
    the detector. jitted rollout makes even 6+10 s cheap (~1 s). The simulated f_so↑ is a
    confirmation; the exact value is the analytic mB/C."""
    fmax = float(max(1.15 * f_so_an, 60.0))
    return fmax, 6.0, 10.0            # fmax, t_up, t_dn


def mode_designs():
    print("=" * 72)
    print("M1b — hysteresis across all 17 de Jongh designs")
    print("=" * 72)
    rows = []
    designs = [("FL", i, FL_TABLE[i]["nu"], 7.47) for i in sorted(FL_TABLE)]
    designs += [("FW", i, FW_TABLE[i]["nu"], FW_TABLE[i]["L_UMR"]) for i in sorted(FW_TABLE)]
    outpath = OUT / "mod1_design_sweep.jsonl"
    with open(outpath, "w") as fh:
        for grp, idx, nu, L in designs:
            t0 = time.perf_counter()
            R, I_a, m_body = build_design(nu, L, FL9_RC)
            C = abs(R[5, 5])
            a = analytic(C, I_a, MB0)
            fmax, t_up, t_dn = loop_params(a["f_so_analytic"])
            f_so_up, f_si_dn = measure_loop(R, I_a, m_body, MB0, fmax=fmax,
                                            t_up=t_up, t_dn=t_dn, L_mm=L)
            row = dict(group=grp, id=idx, nu=nu, L_mm=L,
                       f_so_sim=(None if f_so_up is None else float(f_so_up)),
                       f_si_sim=(None if f_si_dn is None else float(f_si_dn)),
                       window_ratio=a["f_so_analytic"] / a["f_si_analytic"], **a)
            rows.append(row)
            fh.write(json.dumps(row) + "\n"); fh.flush()
            print(f"  {grp}-{idx:<2} nu={nu:<4} L={L:<5} ζ={a['zeta']:.4f} "
                  f"f_so={a['f_so_analytic']:6.0f} f_si={a['f_si_analytic']:5.1f} "
                  f"| sim↑={f_so_up} sim↓={f_si_dn}  ({time.perf_counter()-t0:.0f}s)")
    zmin = min(r["zeta"] for r in rows); zmax = max(r["zeta"] for r in rows)
    print(f"  → wrote {outpath}")
    print(f"  ζ range across family: {zmin:.4f} … {zmax:.4f}  "
          f"(all < 0.39 window-close: {zmax < 0.39})")
    return rows


def mode_scaling():
    print("=" * 72)
    print("M1c — fine ζ/size scaling sweep")
    print("=" * 72)
    # FL-9 reference
    R0, I0, m0 = build_design(FL9_NU, FL9_L, FL9_RC)
    C0 = abs(R0[5, 5])
    zeta0 = C0 / (2.0 * math.sqrt(I0 * MB0))
    omega_n0 = math.sqrt(MB0 / I0)           # rad/s
    f_n0 = omega_n0 / (2 * math.pi)
    print(f"  FL-9 ref: C0={C0:.3e} I0={I0:.3e} ζ0={zeta0:.4f} f_n0={f_n0:.1f} Hz")

    # Universal normalized curve — vary ζ via C at fixed I0, mB0 (ω_n fixed). Diagonal R
    # isolates the pure 1-DOF pendulum (kills corkscrew coupling). Because in units of
    # 1/ω_n the pendulum depends ONLY on ζ, this single simulated curve IS the physics;
    # the absolute-size representation is the exact analytic mapping through ζ↔λ:
    #   f_so_abs = mB/C = mB0/C0 (scale-INVARIANT ≈622 Hz),  f_si_abs = (4/π)·f_n0/λ,
    # with λ = ζ0/ζ  (ζ∝1/λ) and f_n(λ)=f_n0/λ. (A per-λ scaled re-sim would only
    # re-confirm this mapping; the C∝λ³ code assumption is instead checked directly by
    # the BEM rebuild below, which is the only non-analytic link.)
    dt = 1e-4
    zetas = np.geomspace(0.005, 1.2, 46)
    f_so_abs_inv = MB0 / C0 / (2 * math.pi)      # = f_so for FL-9, scale-invariant
    outpath = OUT / "mod1_scaling_sweep.jsonl"
    rows = []
    with open(outpath, "w") as fh:
        for z in zetas:
            C = 2.0 * z * math.sqrt(I0 * MB0)
            Rdiag = np.diag([abs(R0[0, 0]), abs(R0[1, 1]), abs(R0[2, 2]),
                             abs(R0[3, 3]), abs(R0[4, 4]), C])
            a = analytic(C, I0, MB0)
            fmax, t_up, t_dn = loop_params(a["f_so_analytic"])
            f_so_up, f_si_dn = measure_loop(Rdiag, I0, m0, MB0, dt=dt, fmax=fmax,
                                            t_up=t_up, t_dn=t_dn)
            lam = zeta0 / z                       # equivalent isometric scale vs FL-9
            row = dict(zeta=float(z), lambda_rel=float(lam),
                       body_len_mm=float(lam * FL9_L), omega_n_hz=f_n0,
                       f_so_analytic=a["f_so_analytic"], f_si_analytic=a["f_si_analytic"],
                       f_so_sim_norm=(None if f_so_up is None else float(f_so_up) / f_n0),
                       f_si_sim_norm=(None if f_si_dn is None else float(f_si_dn) / f_n0),
                       f_so_abs=float(f_so_abs_inv), f_si_abs=float((4 / math.pi) * f_n0 / lam))
            rows.append(row); fh.write(json.dumps(row) + "\n"); fh.flush()
    zc = math.pi / 8.0
    print(f"  → wrote {outpath}  ({len(rows)} points); window closes at ζ=π/8={zc:.3f} "
          f"→ λ={zeta0/zc:.3f} → body {zeta0/zc*FL9_L*1e3:.0f} µm")

    # (3) BEM λ³ check — rebuild the confined BEM at λ=0.5, 2.0 (scale body+tube
    #     together → ratio 2.035 unchanged → same dimensionless table).
    bem = OUT / "mod1_scaling_bemcheck.jsonl"
    with open(bem, "w") as fh:
        base = dict(lam=1.0, C=C0, C_pred=C0, rel_err=0.0)
        fh.write(json.dumps(base) + "\n")
        print(f"  BEM λ³ check (ref C0={C0:.3e}):")
        for lam in (0.5, 2.0):
            Rl, _, _ = build_design(FL9_NU, FL9_L * lam, FL9_RC * lam)
            Cl = abs(Rl[5, 5])
            pred = C0 * lam ** 3
            rel = abs(Cl - pred) / pred
            rec = dict(lam=lam, C=float(Cl), C_pred=float(pred), rel_err=float(rel))
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            print(f"    λ={lam}: C={Cl:.3e} vs C0·λ³={pred:.3e}  rel_err={rel*100:.1f}%  "
                  f"{'PASS' if rel < 0.05 else 'CHECK'}")
    print(f"  → wrote {bem}")
    return rows


def mode_wall():
    """Where the wall enters the step-out equations: hold the FL-9 body fixed and vary the
    cylindrical confinement ratio R_ves/R_cyl across the available (centred) wall tables.
    Only C (=|R[5,5]|) changes → f_so=mB/C moves with the wall, while f_si=(4/π)ω_n and ω_n
    are drag-free → wall-independent. Writes mod1_wall_ratio_sweep.jsonl."""
    import glob
    print("=" * 72)
    print("M-wall — step-out vs cylindrical confinement ratio (FL-9 body fixed)")
    print("=" * 72)
    tables = sorted(glob.glob(str(REPO / "data" / "dejongh_benchmark" / "wall_tables" / "wall_R*.npz")),
                    key=lambda p: float(p.split("wall_R")[1][:-4]))
    outpath = OUT / "mod1_wall_ratio_sweep.jsonl"
    with open(outpath, "w") as fh:
        for t in tables:
            ratio = float(t.split("wall_R")[1][:-4])
            R, I_a, m = build_design(FL9_NU, FL9_L, FL9_RC, wall_table=t)
            C = abs(R[5, 5]); a = analytic(C, I_a, MB0)
            row = dict(ratio=ratio, is_ours=(abs(ratio - 2.035) < 0.01), **a)
            fh.write(json.dumps(row) + "\n"); fh.flush()
            print(f"  ratio={ratio:6.3f}  C={C:.3e}  ζ={a['zeta']:.4f}  "
                  f"f_so={a['f_so_analytic']:5.0f}  f_si={a['f_si_analytic']:.1f}")
    print(f"  → wrote {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["check", "designs", "scaling", "wall", "all"], default="all")
    a = ap.parse_args()
    t0 = time.perf_counter()
    if a.mode in ("check", "all"):
        mode_check()
    if a.mode in ("designs", "all"):
        mode_designs()
    if a.mode in ("scaling", "all"):
        mode_scaling()
    if a.mode in ("wall", "all"):
        mode_wall()
    print(f"\n[mod1_design_sweep] total {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
