#!/usr/bin/env python3
"""G0-c: gamma_conv × resolution stability-vs-accuracy sweep at iliac Re.

Two measurements at vessel Re≈iliac:
  (A) ACCURACY — Taylor-Green vortex decay vs the exact E(t)=E0*exp(-4νt).
      A clean, bluff-body-free numerical-diffusion probe; the over-damping
      error per (gamma_conv, N) is the headline number. Re_TG = U*L/ν = 2π/ν.
  (B) STABILITY — the actual schwarz far-node (IBM wall + Poiseuille lift, held
      body) at the target vessel Re for each (gamma_conv, CPR): does it stay
      finite over the horizon? (The lift imposes Poiseuille so this is a
      stability/robustness check, not an independent accuracy check — accuracy
      is (A).)

Goal: find the (gamma_conv, CPR) that is BOTH stable in (B) and accurate in (A).
Run:
    python scripts/g0c_gamma_cpr_sweep.py --smoke     # ~seconds, 1-2 configs
    python scripts/g0c_gamma_cpr_sweep.py             # full sweep (~0.5-2 h)
"""
from __future__ import annotations
import argparse, json, os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import jax.numpy as jnp


# ── (A) Taylor-Green accuracy ────────────────────────────────────────────
def taylor_green_error(gamma_conv: float, N: int, nu: float = 0.01,
                       t_end: float = 2.0) -> dict:
    """Return TG decay error vs exact at Re_TG=2π/ν for (gamma_conv, N)."""
    from mime.nodes.environment.fvm.mesh import make_cartesian_mesh_2d
    from mime.nodes.environment.fvm.piso import (
        PisoConfig, initial_state, run_piso_with_history)
    L = 2 * np.pi
    dx = L / N
    dt = 0.4 * dx
    n = int(np.ceil(t_end / dt)); dt = t_end / n
    mesh = make_cartesian_mesh_2d(N, N, L, L, periodic_x=True,
                                  periodic_y=True, dtype=jnp.float64)
    x = np.asarray(mesh.x[:, 0]); y = np.asarray(mesh.x[:, 1])
    u0 = np.zeros((mesh.N_cells, 2))
    u0[:, 0] = np.sin(x) * np.cos(y); u0[:, 1] = -np.cos(x) * np.sin(y)
    p0 = (np.cos(2 * x) + np.cos(2 * y)) / 4.0
    uo = u0[np.asarray(mesh.owner)]; un = u0[np.asarray(mesh.neighbour)]
    F0 = np.einsum("fd,fd->f", 0.5 * (uo + un), np.asarray(mesh.Sf))
    s0 = {**initial_state(mesh), "u": jnp.asarray(u0),
          "p": jnp.asarray(p0), "F": jnp.asarray(F0)}
    cfg = PisoConfig(nu=nu, rho=1.0, gamma_conv=gamma_conv, n_corrector=2,
                     pressure_bc=("periodic", "periodic"),
                     velocity_bc=("periodic", "periodic"))
    _, hist = run_piso_with_history(mesh, bcs={}, cfg=cfg, n_steps=n, dt=dt,
                                    initial=s0, sample_every=max(1, n // 2))
    uh = np.asarray(hist["u"]); th = np.asarray(hist["t"])
    E = 0.5 * np.mean(np.sum(uh ** 2, axis=-1), axis=-1)
    E_ana = 0.25 * np.exp(-4 * nu * th[-1])
    return {"gamma": gamma_conv, "N": N, "cell_Pe": 1.0 * dx / nu,
            "Re_TG": L / nu, "err": abs(E[-1] - E_ana) / E_ana,
            "finite": bool(np.all(np.isfinite(uh)))}


# ── (B) schwarz far-node stability at vessel Re ──────────────────────────
def farnode_stable(gamma_conv: float, cpr: int, Re: float,
                   n_steps: int = 120, dt: float | None = None) -> dict:
    """Build the held-body schwarz graph at vessel Re with (gamma_conv,cpr),
    step, and report finiteness + max sampled FVM speed. ``dt`` overrides the
    default 5e-4 (e.g. 2.5e-4 to restore cpr=4 CFL at cpr=8)."""
    from mime.experiments import schwarz_vessel_helix as S
    nu = S._DEFAULTS["MU_PA_S"] / S._DEFAULTS["RHO_FLUID"]
    D = 2 * S._DEFAULTS["R_VES_M"]
    U = Re * nu / D
    p = {"N_THETA": 12, "N_ZETA": 16, "SWIM_MODE": "held",
         "FLOW_PROFILE": "poiseuille", "INCLUDE_ARM": False,
         "U_MEAN": U, "GAMMA_CONV": gamma_conv, "CPR": cpr,
         "OFFCENTER_RESISTANCE": False}
    if dt is not None:
        p["DT"] = dt
    gm = S.build_graph(p); ref = S.screw_points(p); st = None
    ok = True; vmax = 0.0
    for i in range(n_steps):
        st = gm.step(S.default_external_inputs(p, body_points_ref=ref, state=st))
        v = np.asarray(st["fvm"].get("velocity_at_points", np.zeros((1, 3))))
        if not np.all(np.isfinite(v)):
            ok = False; break
        vmax = max(vmax, float(np.max(np.abs(v))))
    return {"gamma": gamma_conv, "cpr": cpr, "Re": Re, "U_mean": U,
            "stable": ok, "vmax": vmax, "vmax_over_2U": vmax / (2 * U),
            "step": i}


def farnode_freebody_stable(gamma_conv: float, cpr: int, Re: float,
                            n_steps: int = 150) -> dict:
    """STABILITY in the real environment: a FREE, locked-spin, OFF-CENTER body
    (off-center resistance on) seeded off-axis at vessel Re — the IBM body-force
    under a moving/rotating screw, the regime the held+lift check does NOT cover.
    gamma=1.0 (full central) is textbook-fragile at high Pe; this is the test."""
    from mime.experiments import schwarz_vessel_helix as S
    import jax.numpy as jnp
    nu = S._DEFAULTS["MU_PA_S"] / S._DEFAULTS["RHO_FLUID"]
    D = 2 * S._DEFAULTS["R_VES_M"]
    U = Re * nu / D
    p = {"N_THETA": 12, "N_ZETA": 16, "SWIM_MODE": "free", "BODY_MODEL": "locked",
         "FLOW_PROFILE": "poiseuille", "INCLUDE_ARM": False,
         "U_MEAN": U, "GAMMA_CONV": gamma_conv, "CPR": cpr,
         "OFFCENTER_RESISTANCE": True, "DELTA_RHO": 0.0}
    gm = S.build_graph(p); ref = S.screw_points(p)
    bst = dict(gm.get_node_state("body"))
    bst["position"] = jnp.array([0.0, 0.0, -0.5e-3])   # seed OFF-AXIS (moving)
    gm.set_node_state("body", bst)
    st = None; ok = True; vmax = 0.0; pos = None
    for i in range(n_steps):
        st = gm.step(S.default_external_inputs(p, body_points_ref=ref, state=st))
        pos = np.asarray(st["body"]["position"])
        v = np.asarray(st["fvm"].get("velocity_at_points", np.zeros((1, 3))))
        if not (np.all(np.isfinite(v)) and np.all(np.isfinite(pos))):
            ok = False; break
        vmax = max(vmax, float(np.max(np.abs(v))))
    return {"gamma": gamma_conv, "cpr": cpr, "Re": Re, "U_mean": U,
            "stable": ok, "vmax": vmax, "step": i,
            "x_mm": float(pos[0] * 1e3), "z_mm": float(pos[2] * 1e3)}


def _load_done(path):
    """Set of completed result keys from an existing jsonl (resume support)."""
    done = {}
    if path and os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done[r["key"]] = r
    return done


def _append(path, rec):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny fast run (1-2 configs each)")
    ap.add_argument("--re", type=float, default=628.0, help="vessel Re for (B)")
    ap.add_argument("--out", type=str, default=None,
                    help="results jsonl (append + resume); default per mode")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "..", "experiments", "schwarz_vessel_helix", "output")
    out = args.out or os.path.join(
        out_dir, "g0c_sweep_smoke.jsonl" if args.smoke else "g0c_sweep_results.jsonl")
    out = os.path.abspath(out)

    if args.smoke:
        gammas, Ns, cprs, re, nsteps = [0.5, 1.0], [16], [4], 628.0, 30
    else:
        gammas, Ns, cprs, re, nsteps = [0.5, 0.7, 0.85, 1.0], [16, 32, 64], [4, 8], args.re, 120

    done = _load_done(out)
    print(f"results → {out}  ({len(done)} already done, will skip)", flush=True)

    print("=" * 66)
    print(f"(A) Taylor-Green accuracy (Re_TG≈{2*np.pi/0.01:.0f}): err vs exact decay")
    print("=" * 66)
    for g in gammas:
        for N in Ns:
            key = f"A:g{g:.2f}:N{N}"
            if key in done:
                print(f"  [skip] {key}: err={done[key]['err']*100:.2f}%", flush=True); continue
            t0 = time.time(); r = taylor_green_error(g, N)
            r.update(key=key, phase="A", wall_s=round(time.time() - t0, 1))
            _append(out, r)
            print(f"  gamma={g:.2f} N={N:>2} cell-Pe={r['cell_Pe']:4.1f}: "
                  f"err={r['err']*100:6.2f}%  finite={r['finite']}  ({r['wall_s']:.0f}s)", flush=True)

    # Held-body stability across vessel Re including the aortic-bifurcation
    # upper bound (Re≈1100). Re=628 from the first run is skipped on resume.
    re_list = [628, 1100] if not args.smoke else [628]
    print("=" * 66)
    print(f"(B) schwarz HELD-body stability, Re={re_list}")
    print("=" * 66)
    for rev in re_list:
        for g in gammas:
            for c in cprs:
                key = f"B:g{g:.2f}:cpr{c}:Re{int(rev)}"
                if key in done:
                    print(f"  [skip] {key}: stable={done[key]['stable']}", flush=True); continue
                t0 = time.time(); r = farnode_stable(g, c, float(rev), n_steps=nsteps)
                r.update(key=key, phase="B", wall_s=round(time.time() - t0, 1))
                _append(out, r)
                print(f"  Re={rev} gamma={g:.2f} cpr={c}: stable={r['stable']}  "
                      f"vmax/2U={r['vmax_over_2U']:.3f}  ({r['wall_s']:.0f}s)", flush=True)

    # FREE, off-axis, rotating, off-center body — the real (non-benign) stability
    # environment, at both Re bounds, for the two scheme extremes.
    print("=" * 66)
    print("(C) schwarz FREE off-axis body stability (gamma extremes, Re bounds)")
    print("=" * 66)
    c_gammas = [0.5, 1.0] if not args.smoke else [1.0]
    c_res = [628, 1100] if not args.smoke else [628]
    for rev in c_res:
        for g in c_gammas:
            key = f"C:g{g:.2f}:cpr4:Re{int(rev)}"
            if key in done:
                print(f"  [skip] {key}: stable={done[key]['stable']}", flush=True); continue
            t0 = time.time(); r = farnode_freebody_stable(g, 4, float(rev), n_steps=nsteps)
            r.update(key=key, phase="C", wall_s=round(time.time() - t0, 1))
            _append(out, r)
            print(f"  Re={rev} gamma={g:.2f}: stable={r['stable']}  step={r['step']}  "
                  f"x={r['x_mm']:.3f}mm z={r['z_mm']:.3f}mm  ({r['wall_s']:.0f}s)", flush=True)

    # (D) CFL fix: the cpr=8 / Re=1100 cases blew up at DT=5e-4; re-run at
    # DT=2.5e-4 (same CFL as the stable cpr=4 / DT=5e-4). Confirms the blow-up
    # was a fixed-DT artifact, not a fundamental high-Re obstacle.
    if not args.smoke:
        print("=" * 66)
        print("(D) CFL fix — cpr=8, Re=1100, DT=2.5e-4 (cpr-4-equivalent CFL)")
        print("=" * 66)
        for g in [0.5, 0.7, 0.85, 1.0]:
            key = f"D:g{g:.2f}:cpr8:Re1100:dt2.5e-4"
            if key in done:
                print(f"  [skip] {key}: stable={done[key]['stable']}", flush=True); continue
            t0 = time.time(); r = farnode_stable(g, 8, 1100.0, n_steps=120, dt=2.5e-4)
            r.update(key=key, phase="D", dt=2.5e-4, wall_s=round(time.time() - t0, 1))
            _append(out, r)
            print(f"  gamma={g:.2f}: stable={r['stable']}  vmax/2U={r['vmax_over_2U']:.3f}  "
                  f"({r['wall_s']:.0f}s)", flush=True)

    print("done.", flush=True)


if __name__ == "__main__":
    main()
