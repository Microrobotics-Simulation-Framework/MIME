"""CPR benchmark — speed and accuracy at each cpr on RTX 2060.

For the MIME iliac millibot (λ=0.375, Re_mean=182, Wo=6.1) at
cpr ∈ {2, 3, 4, 6, 8}:
  - mesh size + dx
  - JIT compile time
  - per-step ms (mean ± std over 20 steps after warmup)
  - throughput (Mcells/s)
  - K_inertial_mean error vs cpr=8 reference, with and without GNN
  - GNN inference overhead (× the uncorrected step time)

Outputs printed table + benchmark_results.csv.
"""
from __future__ import annotations
import csv
import pickle
import time
from pathlib import Path

import numpy as np
import jax, jax.numpy as jnp

from mime.nodes.environment.fvm import (
    make_pipe_mesh, make_womersley_lift_analytical, make_poiseuille_lift,
    make_poiseuille_p_lift,
)
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.piso import PisoConfig, make_piso_step
from mime.nodes.environment.fvm.ibm import (
    IBMBody, surface_integral_force,
)
from mime.nodes.environment.fvm.sdf import sphere_sdf

from step2_train import correction_body_force


PIPE_RADIUS  = 4e-3
PIPE_LENGTH  = 33e-3                 # Fix-2 minimum
ROBOT_RADIUS = 1.5e-3
LAMBDA       = ROBOT_RADIUS / PIPE_RADIUS
NU           = 3.3e-6
RHO          = 1060.0
U_DC         = 0.075                 # halved from brief's 0.15 to keep Re_peak
U_AMP        = 0.075                 # within stable cpr=2 regime
T_CYCLE      = 1.0
OMEGA        = 2.0 * np.pi
N_CYCLES     = 1
N_BENCH_STEPS = 20

CPRS = [2, 3, 4, 6, 8]


def build(cpr, with_gnn=False, corrector=None):
    mu = RHO * NU
    mesh = make_pipe_mesh(pipe_radius=PIPE_RADIUS, pipe_length=PIPE_LENGTH,
                          robot_radius=ROBOT_RADIUS, cpr=cpr)
    dx = mesh.cartesian_spacing[0]
    Nz = mesh.cartesian_shape[2]
    L_actual = Nz * dx
    sphere_centre = jnp.array([0.0, 0.0, L_actual / 2], dtype=mesh.V.dtype)
    def pipe_wall_sdf(x):
        rxy = jnp.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2 + 1e-30)
        return PIPE_RADIUS - rxy
    def sphere_sdf_fn(x):
        return sphere_sdf(x, center=sphere_centre, radius=ROBOT_RADIUS)
    bodies = [
        IBMBody(name="pipe_wall", sdf=pipe_wall_sdf),
        IBMBody(name="sphere",    sdf=sphere_sdf_fn),
    ]
    bcs = {}
    for name in ("x_min","x_max","y_min","y_max","z_min","z_max"):
        nb = int(mesh.patch(name).owner.size)
        bcs[name] = VelocityBC(u_wall=jnp.zeros((nb,3)),
                                F_through=jnp.zeros((nb,)))
    cfg = PisoConfig(
        nu=NU, rho=RHO, gamma_conv=0.0, n_corrector=2,
        pressure_bc="neumann", velocity_bc="dirichlet",
        ibm_alpha=1e5, ibm_eps=1.0 * dx,
    )
    L = make_womersley_lift_analytical(
        mesh, R_pipe=PIPE_RADIUS, U_mean_dc=U_DC, U_mean_amp=U_AMP,
        omega=OMEGA, nu=NU, axis=2, phase_offset=-np.pi / 2,
    )
    # CFL-bounded dt (cross-section)
    u_max = 2 * (U_DC + U_AMP)
    dt = 0.4 * dx / max(u_max, 1e-30)
    return dict(mesh=mesh, bcs=bcs, cfg=cfg, bodies=bodies, lift=L,
                sphere_centre=sphere_centre, sphere_sdf_fn=sphere_sdf_fn,
                dx=dx, dt=dt, L_actual=L_actual, mu=mu)


def make_step_fn(d, with_gnn, corrector=None):
    """JIT-compiled single PISO step (with or without GNN injection)."""
    if not with_gnn:
        step = make_piso_step(d["mesh"], d["bcs"], d["cfg"],
                               body_force_fn=None,
                               ibm_bodies=d["bodies"], lifting=d["lift"])
        @jax.jit
        def step_fn(state):
            return step(state, d["dt"])
        return step_fn
    rho = d["cfg"].rho

    @jax.jit
    def step_fn(carry):
        state, u_prev = carry
        f_gnn = correction_body_force(
            corrector, state["u"], state["p"], d["mesh"], rho,
            u_prev=u_prev, dt=d["dt"], U_ref=U_DC, r_b=ROBOT_RADIUS,
        )
        body_force_fn = lambda t: f_gnn
        step = make_piso_step(
            d["mesh"], d["bcs"], d["cfg"],
            body_force_fn=body_force_fn,
            ibm_bodies=d["bodies"], lifting=d["lift"],
        )
        new_state = step(state, d["dt"])
        return (new_state, state["u"])
    return step_fn


def init_state(d):
    from mime.nodes.environment.fvm.piso import initial_state
    return initial_state(d["mesh"])


def time_step(step_fn, state, with_gnn, n_warmup=2, n_bench=N_BENCH_STEPS):
    # Warmup (JIT compile + cache fill)
    t_compile_start = time.time()
    if with_gnn:
        carry = (state, state["u"])
        carry = step_fn(carry)
        carry[0]["u"].block_until_ready()
    else:
        state = step_fn(state)
        state["u"].block_until_ready()
    t_compile = time.time() - t_compile_start
    # Few more warmups
    for _ in range(n_warmup - 1):
        if with_gnn:
            carry = step_fn(carry)
        else:
            state = step_fn(state)
    if with_gnn:
        carry[0]["u"].block_until_ready()
    else:
        state["u"].block_until_ready()
    # Benchmark
    times = []
    for _ in range(n_bench):
        t0 = time.time()
        if with_gnn:
            carry = step_fn(carry)
            carry[0]["u"].block_until_ready()
        else:
            state = step_fn(state)
            state["u"].block_until_ready()
        times.append(time.time() - t0)
    if with_gnn:
        return carry, t_compile, np.mean(times), np.std(times)
    return state, t_compile, np.mean(times), np.std(times)


def K_at_cycle_end(d, state, n_cycle_steps):
    """Run 1 full cycle and report cycle-mean K_FVM via surface_integral."""
    L = d["lift"]
    step = make_piso_step(d["mesh"], d["bcs"], d["cfg"],
                           body_force_fn=None,
                           ibm_bodies=d["bodies"], lifting=L)
    @jax.jit
    def go(s):
        return jax.lax.fori_loop(0, n_cycle_steps,
                                  lambda _, x: step(x, d["dt"]), s)
    final = go(state)
    final["u"].block_until_ready()
    u_phys = final["u"] + (L.u_lift_static
                            if L.U_re is None
                            else L.u_lift_static)  # steady part only
    # For analytical Womersley, we need the physical velocity at this t.
    # Use steady component as a snapshot for drag estimation.
    p_lift_fn = make_poiseuille_p_lift(
        mu=d["mu"], U_mean=U_DC, pipe_radius=PIPE_RADIUS,
    )
    F_vec, _ = surface_integral_force(
        u_phys, final["p"], d["mesh"], d["sphere_sdf_fn"],
        mu=d["mu"], dx=d["dx"], shell_inner=0.5, shell_outer=2.5,
        ref_point=d["sphere_centre"], p_lift_fn=p_lift_fn, pipe_axis=2,
    )
    F_z = float(F_vec[2])
    F_uncon = 6 * np.pi * d["mu"] * ROBOT_RADIUS * (2 * U_DC)
    return F_z / F_uncon


def main():
    print("=" * 78)
    print("CPR Benchmark — MIME Iliac Millibot")
    print(f"  λ={LAMBDA}, Re_mean(R)={U_DC*PIPE_RADIUS/NU:.0f}, "
          f"Wo={PIPE_RADIUS*np.sqrt(OMEGA/NU):.1f}")
    print(f"  RTX 2060, float32, dense pressure solver")
    print("=" * 78)

    # Load trained corrector (Task 1 retrained 14-feature)
    corrector_path = Path(__file__).parent.parent.parent / \
        "data/gnn_training/gnn_params_local.pkl"
    if corrector_path.exists():
        with open(corrector_path, "rb") as f:
            corrector = pickle.load(f)
        gnn_available = True
        print(f"  loaded corrector: {corrector.param_count()} params\n")
    else:
        corrector = None
        gnn_available = False
        print("  WARNING: no trained corrector — GNN columns will be skipped\n")

    rows = []
    K_ref = None     # cpr=8 K becomes the reference
    for cpr in CPRS:
        print(f"--- cpr = {cpr} ---", flush=True)
        try:
            d = build(cpr)
        except Exception as e:
            print(f"  build FAIL: {e}")
            rows.append(dict(cpr=cpr, status="build-fail",
                              N_cells=None))
            continue
        n_cycle = int(round(T_CYCLE / d["dt"]))
        print(f"  mesh {d['mesh'].cartesian_shape} = {d['mesh'].N_cells} cells, "
              f"dx={d['dx']*1e3:.3f}mm, "
              f"cells_per_diameter={2*ROBOT_RADIUS/d['dx']:.1f}, "
              f"dt={d['dt']*1e3:.3f}ms, n_cycle_steps={n_cycle}")

        # ----- A/B/C/D — uncorrected timing -----
        try:
            step_fn = make_step_fn(d, with_gnn=False)
            state = init_state(d)
            state, tc_unc, ms_unc, std_unc = time_step(
                step_fn, state, with_gnn=False,
            )
            ms_unc *= 1000; std_unc *= 1000
            mcells = d["mesh"].N_cells / 1e6 / (ms_unc / 1000)
            print(f"  uncorrected: compile {tc_unc:.1f}s, "
                  f"{ms_unc:.2f}±{std_unc:.2f} ms/step, {mcells:.1f} Mcells/s")
        except Exception as e:
            print(f"  uncorrected timing FAIL: {type(e).__name__}: {e}")
            rows.append(dict(cpr=cpr, N_cells=d["mesh"].N_cells,
                              status="uncorrected-timing-fail"))
            continue

        # ----- E — uncorrected drag accuracy (1 cardiac cycle) -----
        try:
            K_unc = K_at_cycle_end(d, init_state(d), n_cycle)
            print(f"  uncorrected K_FVM (1 cycle): {K_unc:+.3f}")
        except Exception as e:
            print(f"  uncorrected drag FAIL: {type(e).__name__}: {e}")
            K_unc = None

        if cpr == 8:
            K_ref = K_unc
        err_unc = (abs(K_unc - K_ref) / abs(K_ref) * 100
                    if (K_unc is not None and K_ref is not None) else None)

        # ----- F/G — GNN -----
        K_corr = None; ms_corr = None; std_corr = None; tc_corr = None
        overhead = None
        if gnn_available:
            try:
                step_fn_g = make_step_fn(d, with_gnn=True,
                                          corrector=corrector)
                s0 = init_state(d)
                _, tc_corr, ms_corr, std_corr = time_step(
                    step_fn_g, s0, with_gnn=True,
                )
                ms_corr *= 1000; std_corr *= 1000
                overhead = ms_corr / ms_unc
                print(f"  GNN-corrected: compile {tc_corr:.1f}s, "
                      f"{ms_corr:.2f}±{std_corr:.2f} ms/step, "
                      f"overhead {overhead:.2f}×")
            except Exception as e:
                print(f"  GNN timing FAIL: {type(e).__name__}: {e}")

        rows.append(dict(
            cpr=cpr,
            N_cells=d["mesh"].N_cells,
            dx_mm=d["dx"] * 1e3,
            cells_per_diameter=2 * ROBOT_RADIUS / d["dx"],
            compile_uncorr_s=tc_unc,
            ms_per_step_uncorr=ms_unc,
            std_ms_uncorr=std_unc,
            mcells_per_s=mcells,
            K_uncorr=K_unc,
            err_uncorr_pct=err_unc,
            K_corr=K_corr,
            ms_per_step_corr=ms_corr,
            std_ms_corr=std_corr,
            gnn_overhead=overhead,
        ))

    # ----- After loop, recompute err for non-cpr=8 rows now that K_ref known -----
    for r in rows:
        if r.get("K_uncorr") is not None and K_ref is not None:
            r["err_uncorr_pct"] = abs(r["K_uncorr"] - K_ref) / abs(K_ref) * 100

    # ----- Print table -----
    print("\n" + "=" * 78)
    print("Benchmark table")
    print("=" * 78)
    print(f"{'cpr':>3} {'N_cells':>10} {'compile':>9} {'ms/step':>14} "
          f"{'Mcells/s':>9} {'K (no GNN)':>12} {'err vs c8':>10} "
          f"{'GNN ms/step':>13} {'overhead':>10}")
    for r in rows:
        if r.get("N_cells") is None:
            print(f"{r['cpr']:>3}  build/timing FAIL")
            continue
        c = r.get("compile_uncorr_s")
        unc_ms = r.get("ms_per_step_uncorr")
        std_unc = r.get("std_ms_uncorr")
        K = r.get("K_uncorr")
        err = r.get("err_uncorr_pct")
        ms_g = r.get("ms_per_step_corr")
        std_g = r.get("std_ms_corr")
        ov = r.get("gnn_overhead")
        print(f"{r['cpr']:>3} {r['N_cells']:>10d} {c:>8.1f}s "
              f"{unc_ms:>7.2f}±{std_unc:>4.2f} ms "
              f"{r['mcells_per_s']:>9.2f} "
              f"{K if K is not None else float('nan'):>12.3f} "
              f"{err if err is not None else float('nan'):>9.2f}% "
              f"{(f'{ms_g:.2f}±{std_g:.2f}' if ms_g else 'n/a'):>13} "
              f"{ov if ov is not None else float('nan'):>9.2f}×")

    # ----- CSV -----
    csv_path = Path(__file__).parent / "benchmark_results.csv"
    with open(csv_path, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"\n  CSV written: {csv_path}")

    # ----- H100 projection -----
    cpr8 = next((r for r in rows if r["cpr"] == 8 and r.get("ms_per_step_uncorr")),
                 None)
    cpr4 = next((r for r in rows if r["cpr"] == 4 and r.get("ms_per_step_uncorr")),
                 None)
    if cpr8:
        print("\nProjected H100 performance (60× FP32 throughput vs RTX 2060):")
        if cpr4:
            print(f"  cpr=4 step time on H100: ~{cpr4['ms_per_step_uncorr']/60:.2f} ms")
        print(f"  cpr=8 step time on H100: ~{cpr8['ms_per_step_uncorr']/60:.2f} ms")
        n_3cycles = 3 * int(round(T_CYCLE / build(8)["dt"]))
        print(f"  Full 3-cycle iliac run at cpr=8 on H100: "
              f"~{n_3cycles * cpr8['ms_per_step_uncorr'] / 60 / 1000:.1f} s")

    # ----- Interpretation flags -----
    if K_ref is not None:
        for r in rows:
            if r.get("err_uncorr_pct") is not None and r["err_uncorr_pct"] > 10:
                print(f"  >>> cpr={r['cpr']} uncorrected error {r['err_uncorr_pct']:.1f}% "
                      f"exceeds 10% — minimum reliable cpr without GNN is higher")
                break


if __name__ == "__main__":
    main()
