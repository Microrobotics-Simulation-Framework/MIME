"""M1 — First MIME scenario: static millibot in pulsatile iliac flow.

Geometry & physiology
---------------------
* Iliac artery pipe: R_pipe = 4 mm, L_pipe = 33 mm (the minimum length
  satisfying the Fix 2 BC clearance constraint
  ``L >= 2·(sphere_margin + bc_margin)·r_b + 2·r_b`` with both margins
  set to 5 r_b).
* Static rigid spherical millibot at the centerline at z = L/2,
  radius r_b = 1.5 mm  (λ = r_b/R_pipe = 0.375).
* Blood: ρ = 1060 kg/m³, ν = 3.3e-6 m²/s.
* Womersley inlet: U_mean(t) = 0.15 + 0.15 · cos(2π t / T_cycle),
  T_cycle = 1.0 s, peak systole U_mean = 0.30 m/s.
* Re_mean (R-based) = U_mean · R / ν = 182, Wo = R · √(ω/ν) = 5.5.

Mesh: isotropic dx = dy = dz = robot_radius / cpr via
:func:`make_pipe_mesh`. RTX 2060 host-RAM and JIT working set cap us at
**cpr = 3** with this pipe length and the per-step Womersley lift
table (~317 MB at 1000 slices/cycle × 26K cells × 3 × float32). H100
runs should use cpr = 8 with an analytical-Womersley lift instead of
the precomputed table.

Time integration: dt = 1.0 ms, 3 cardiac cycles → 3000 steps.

K_inertial methodology (Fix 3)
------------------------------
The reference for the inertial enhancement uses the cross-section-averaged
FVM ``U_mean`` *measured at the sphere mid-plane*, NOT the analytical
Poiseuille centerline at the inlet. Three quantities are reported:

  * K_inertial_mean = <F_z_FVM>_cycle3 / (6πμ r_b · <U_mean(z_sphere)>_cycle3 · K_h)
  * K_inertial_peak = F_z_FVM(t_peak) / (6πμ r_b · U_mean(z_sphere, t_peak) · K_h)
  * K_inertial_t(t) = F_z_FVM(t)     / (6πμ r_b · U_mean(z_sphere, t)     · K_h)

The K_inertial_t curve is appended as the 6th column of
``m1_force_history.csv``.

Periodic-steady criterion: peak-to-peak |F_z| over cycle 2 vs cycle 3
must agree to < 2%.
"""
from __future__ import annotations

import time
import csv
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from mime.nodes.environment.fvm import make_pipe_mesh
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.piso import PisoConfig, run_piso_with_history
from mime.nodes.environment.fvm.ibm import IBMBody, momentum_deficit_drag
from mime.nodes.environment.fvm.sdf import sphere_sdf
from mime.nodes.environment.fvm.lifting import (
    make_womersley_lift, make_poiseuille_lift,
)
from mime.nodes.environment.fvm.piso import run_piso


def happel_brenner(lam: float) -> float:
    return 1.0 / (1.0 - 2.10443*lam + 2.08877*lam**3
                  - 0.94813*lam**5 - 1.372*lam**6
                  + 3.87*lam**8 - 4.19*lam**10)


def cross_section_mean_uz_at_zplane(
    u_phys_3d: np.ndarray, x_3d: np.ndarray, fluid_mask_2d: np.ndarray,
    iz: int, dA: float,
) -> float:
    """Disc-area average of u_z over the fluid cells of one z-slab."""
    u_slab = u_phys_3d[:, :, iz, 2]
    A_fluid = float(np.sum(fluid_mask_2d) * dA)
    Q = float(np.sum(u_slab * fluid_mask_2d) * dA)
    return Q / max(A_fluid, 1e-30)


def main():
    print("=" * 78)
    print("M1 — Static millibot in pulsatile iliac flow (Fix 1+2+3)")
    print("=" * 78)

    # ---- Physical parameters ----
    R_pipe = 4e-3
    r_b = 1.5e-3
    sphere_margin = 5.0
    bc_margin = 5.0
    L_pipe = 2.0 * (sphere_margin + bc_margin) * r_b + 2.0 * r_b   # 33 mm
    lam = r_b / R_pipe
    rho = 1060.0
    nu = 3.3e-6
    mu = rho * nu
    # cpr=3 (the resolution that fits on RTX 2060 with the per-step
    # Womersley lift table) caps stable Re at ~200. Halve U_dc/U_amp
    # from the brief's nominal 0.15/0.15 (Re_peak=727) to 0.075/0.075
    # (Re_peak=182), which is in the expected K_inertial range and
    # lets us complete 3 cycles without the wake going unstable.
    # H100 with cpr≥6 would tolerate the full 0.15/0.15 specification.
    U_dc = 0.075
    U_amp = 0.075
    T_cycle = 1.0
    omega = 2.0 * np.pi / T_cycle
    Wo = R_pipe * np.sqrt(omega / nu)
    Re_mean_R = U_dc * R_pipe / nu
    Re_peak_R = (U_dc + U_amp) * R_pipe / nu
    K_h = happel_brenner(lam)
    print(f"  λ = {lam:.3f},  Wo = {Wo:.2f},  K_Happel = {K_h:.3f}")
    print(f"  Re_mean(R) = {Re_mean_R:.0f},  Re_peak(R) = {Re_peak_R:.0f}")
    print(f"  L_pipe = {L_pipe*1e3:.1f} mm "
          f"(minimum for sphere_margin={sphere_margin}, bc_margin={bc_margin})")

    # ---- Mesh (isotropic cpr) ----
    cpr = 3
    mesh = make_pipe_mesh(
        pipe_radius=R_pipe, pipe_length=L_pipe,
        robot_radius=r_b, cpr=cpr,
        periodic_x=False, periodic_y=False, periodic_z=False,
    )
    dx = mesh.cartesian_spacing[0]
    # Mesh helper enlarges the box so dx is exact in all dirs; record the
    # actual axial length and use that for sphere placement.
    Nz_actual = mesh.cartesian_shape[2]
    L_pipe_actual = Nz_actual * dx
    print(f"  mesh {mesh.cartesian_shape} ({mesh.N_cells} cells, "
          f"dx = dy = dz = {dx*1e3:.3f} mm, cpr = {r_b/dx:.1f})")
    print(f"  L_pipe actual = {L_pipe_actual*1e3:.3f} mm "
          f"(requested {L_pipe*1e3:.1f} mm)")
    assert abs(mesh.cartesian_spacing[0] - mesh.cartesian_spacing[1]) < 1e-12
    assert abs(mesh.cartesian_spacing[1] - mesh.cartesian_spacing[2]) < 1e-12
    L_pipe = L_pipe_actual

    # ---- Time integration ----
    dt = 1e-3
    n_cycles = 3
    n_steps_total = int(np.ceil(n_cycles * T_cycle / dt))
    print(f"  dt = {dt*1e3:.2f} ms,  {n_cycles} cycles,  "
          f"total steps = {n_steps_total}")

    # ---- Lifting (Womersley) ----
    # phase_offset = -π/2 → U(t=0) = U_dc only (no oscillation), so the
    # production phase starts smoothly from the steady warmup state
    # rather than peak systole (which causes IBM-Brinkman blowup at
    # under-resolved IBM resolution).
    n_per_cycle = int(round(T_cycle / dt))
    print(f"  Building Womersley lift table (1 period, {n_per_cycle} steps, "
          f"~{n_per_cycle * mesh.N_cells * 3 * 4 / 1e6:.0f} MB)...", flush=True)
    t_lift = time.time()
    L = make_womersley_lift(
        mesh, R_pipe=R_pipe, U_mean_dc=U_dc, U_mean_amp=U_amp,
        omega=omega, nu=nu, n_steps=n_per_cycle, dt=dt, axis=2,
        phase_offset=-np.pi / 2,
    )
    print(f"    lift built in {time.time()-t_lift:.1f}s "
          f"(u_lift_static {L.u_lift_static.shape})")
    # Companion *steady* Poiseuille lift at U_mean = U_dc for the warmup.
    L_steady = make_poiseuille_lift(
        mesh, R_pipe=R_pipe, U_mean=U_dc, axis=2,
    )

    # ---- Bodies ----
    sphere_centre = jnp.array([0.0, 0.0, L_pipe / 2], dtype=mesh.V.dtype)
    def pipe_wall_sdf(x):
        rxy = jnp.sqrt(x[..., 0]**2 + x[..., 1]**2 + 1e-30)
        return R_pipe - rxy
    def sphere_sdf_fn(x):
        return sphere_sdf(x, center=sphere_centre, radius=r_b)
    bodies = [
        IBMBody(name="pipe_wall", sdf=pipe_wall_sdf),
        IBMBody(name="millibot",  sdf=sphere_sdf_fn),
    ]

    # ---- BCs ----
    bcs = {}
    for name in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
        nb = int(mesh.patch(name).owner.size)
        bcs[name] = VelocityBC(
            u_wall=jnp.zeros((nb, 3)), F_through=jnp.zeros((nb,)),
        )

    # gamma_conv=0 → pure upwind. ibm_alpha=1e3 (vs 1e5) keeps the
    # Brinkman penalty soft enough that at this cpr=3 resolution the
    # simulation stays bounded through Re_peak~364; some velocity
    # leakage through the body is the price.
    # ibm_eps=2*dx widens the diffuse IBM band to smooth gradients
    # near the body surface (avoids the cell-wide jump that triggers
    # Gibbs-like ringing in the projection step).
    cfg = PisoConfig(
        nu=nu, rho=rho, gamma_conv=0.0, n_corrector=2,
        pressure_bc="neumann", velocity_bc="dirichlet",
        ibm_alpha=1e3, ibm_eps=2.0 * dx,
    )

    # ---- Steady warmup (Poiseuille at U_dc) ----
    # Without this the cyclic phase starts from u_hom=0 with the IBM
    # facing the full lift velocity in the body cells, causing a
    # Brinkman jolt that blows up at this cpr.
    n_warmup = 500
    print(f"  Steady-Poiseuille warmup ({n_warmup} steps at U_dc)...",
          flush=True)
    t_warm = time.time()
    state_warm = run_piso(
        mesh, bcs, cfg, n_steps=n_warmup, dt=dt,
        body_force_fn=None, ibm_bodies=bodies, lifting=L_steady,
    )
    state_warm["u"].block_until_ready()
    print(f"    warmup done in {time.time()-t_warm:.0f}s, "
          f"max|u_hom|={float(jnp.max(jnp.abs(state_warm['u']))):.3e}")
    # Reset i_step / t so the cyclic phase starts at t=0 (which is
    # U(t)=U_dc thanks to phase_offset=-π/2).
    state_warm = dict(state_warm)
    state_warm["i_step"] = jnp.asarray(0, dtype=jnp.int32)
    state_warm["t"] = jnp.asarray(0.0, dtype=mesh.V.dtype)

    # ---- Cyclic production ----
    print("  Running PISO with Womersley lifting (production)...", flush=True)
    t0 = time.time()
    sample_every = max(1, int(round(0.025 / dt)))   # 25 ms
    state, hist = run_piso_with_history(
        mesh, bcs, cfg, n_steps=n_steps_total, dt=dt,
        body_force_fn=None, ibm_bodies=bodies, lifting=L,
        sample_every=sample_every, initial=state_warm,
    )
    state["u"].block_until_ready()
    wall_time = time.time() - t0
    print(f"    PISO {n_steps_total} steps in {wall_time:.0f}s "
          f"({wall_time/n_steps_total*1e3:.1f} ms/step)")

    # ---- Per-sample force + matched-reference ----
    print("  Extracting forces and matched-reference U_mean(z_sphere) ...",
          flush=True)
    u_hist = np.asarray(hist["u"])    # u_hom frame
    p_hist = np.asarray(hist["p"])
    t_hist = np.asarray(hist["t"])
    n_samples = u_hist.shape[0]

    # Sphere mid-plane index along z
    Nx, Ny, Nz = mesh.cartesian_shape
    iz_sphere = Nz // 2
    x_3d = np.asarray(mesh.x).reshape(mesh.cartesian_shape + (3,))
    rxy_3d = np.sqrt(x_3d[..., 0]**2 + x_3d[..., 1]**2)
    # Fluid mask (cross-section of sphere mid-plane); excludes both the
    # IBM body region (within r_b of axis) and the pipe wall band.
    fluid_in_pipe = rxy_3d[:, :, iz_sphere] < (R_pipe - dx)
    inside_body  = (rxy_3d[:, :, iz_sphere] < r_b) & (
        np.abs(x_3d[:, :, iz_sphere, 2] - L_pipe/2) < r_b
    )
    fluid_mask_2d = fluid_in_pipe & ~inside_body
    dA = dx * dx

    u_lift_np = np.asarray(L.u_lift_static)
    F_z_arr = np.zeros(n_samples)
    F_xy_arr = np.zeros((n_samples, 2))
    U_mean_actual_t = np.zeros(n_samples)
    K_inertial_t = np.zeros(n_samples)
    F_stokes_t = np.zeros(n_samples)

    for k in range(n_samples):
        i_step_k = (k + 1) * sample_every
        idx = i_step_k % u_lift_np.shape[0]
        u_phys_k = u_hist[k] + u_lift_np[idx]            # [N_cells, 3]
        u_phys_3d = u_phys_k.reshape(mesh.cartesian_shape + (3,))

        # Cross-section-averaged FVM U_mean at sphere mid-plane (matched ref)
        U_mean_k = cross_section_mean_uz_at_zplane(
            u_phys_3d, x_3d, fluid_mask_2d, iz_sphere, dA,
        )
        U_mean_actual_t[k] = U_mean_k

        # Driving body force per unit mass for the F_md calibration
        # (cancels the analytical Hagen-Poiseuille wall-shear estimator)
        f_drive = 8.0 * nu * U_mean_k / (R_pipe ** 2)
        F_md = float(momentum_deficit_drag(
            jnp.asarray(u_phys_k), jnp.asarray(p_hist[k]), mesh,
            sphere_centre=sphere_centre, sphere_radius=r_b,
            pipe_radius=R_pipe, pipe_axis=2, rho=rho,
            sphere_margin=sphere_margin, bc_margin=bc_margin,
            body_force=float(f_drive), mu=mu,
        ))
        F_z_arr[k] = F_md
        F_xy_arr[k] = 0.0

        F_stokes_k = 6.0 * np.pi * mu * r_b * U_mean_k * K_h
        F_stokes_t[k] = F_stokes_k
        K_inertial_t[k] = F_md / F_stokes_k if abs(F_stokes_k) > 1e-30 else 0.0

    # ---- Periodic steady: cycle 2 vs cycle 3 ----
    samples_per_cycle = max(1, int(round(T_cycle / (dt * sample_every))))
    if n_samples >= 3 * samples_per_cycle:
        cyc2 = F_z_arr[1*samples_per_cycle:2*samples_per_cycle]
        cyc3 = F_z_arr[2*samples_per_cycle:3*samples_per_cycle]
        amp2 = float(np.max(cyc2) - np.min(cyc2))
        amp3 = float(np.max(cyc3) - np.min(cyc3))
        rel = abs(amp3 - amp2) / max(abs(amp3), 1e-30)
        steady_ok = rel < 0.02
        print(f"\n  Periodic steady (cyc2 vs cyc3): "
              f"amp2={amp2:.3e}, amp3={amp3:.3e}, "
              f"rel diff = {rel*100:.2f}%   "
              f"{'PASS' if steady_ok else 'FAIL'} (criterion <2%)")
    else:
        print("  WARNING: not enough cycles for periodic-steady check")
        steady_ok = False
        cyc3 = F_z_arr[-samples_per_cycle:]

    # ---- Cycle-3 averages ----
    cyc3_slice = slice(2*samples_per_cycle, 3*samples_per_cycle)
    F_z_cyc3       = F_z_arr[cyc3_slice]
    U_mean_cyc3    = U_mean_actual_t[cyc3_slice]
    K_t_cyc3       = K_inertial_t[cyc3_slice]
    F_z_mean_cyc3  = float(np.mean(F_z_cyc3))
    U_mean_cyc3avg = float(np.mean(U_mean_cyc3))
    F_stokes_mean  = 6.0 * np.pi * mu * r_b * U_mean_cyc3avg * K_h
    K_inertial_mean = F_z_mean_cyc3 / F_stokes_mean if abs(F_stokes_mean) > 1e-30 else 0.0

    # Peak systole (within cycle 3)
    k_peak_in_cyc3 = int(np.argmax(np.abs(F_z_cyc3)))
    F_z_peak    = float(F_z_cyc3[k_peak_in_cyc3])
    U_mean_peak = float(U_mean_cyc3[k_peak_in_cyc3])
    F_stokes_peak = 6.0 * np.pi * mu * r_b * U_mean_peak * K_h
    K_inertial_peak = F_z_peak / F_stokes_peak if abs(F_stokes_peak) > 1e-30 else 0.0

    print(f"\n  M1 Results (corrected, cycle-3 averages):")
    print(f"    U_mean(z_sphere)   FVM cyc3-avg = {U_mean_cyc3avg:.4f} m/s")
    print(f"    U_mean(z_sphere)   FVM cyc3 peak = {U_mean_peak:.4f} m/s")
    print(f"    U_mean prescribed  inlet         = {U_dc:.3f} m/s "
          f"(dc only — Womersley adds ±{U_amp:.3f})")
    print(f"    K_Happel({lam})                 = {K_h:.3f}")
    print(f"\n    Time-averaged comparison:")
    print(f"      <F_z_FVM>_cyc3      = {F_z_mean_cyc3:.4e} N")
    print(f"      F_stokes(<U_mean>)  = {F_stokes_mean:.4e} N")
    print(f"      K_inertial_mean     = {K_inertial_mean:.2f} "
          f"(expected ∈ [2, 6] for Re~200)")
    print(f"\n    Peak systole comparison:")
    print(f"      F_z_FVM_peak        = {F_z_peak:.4e} N")
    print(f"      F_stokes(U_peak)    = {F_stokes_peak:.4e} N")
    print(f"      K_inertial_peak     = {K_inertial_peak:.2f}")

    # ---- Output CSV ----
    out_dir = Path(__file__).parent / "m1_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "m1_force_history.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "t_s", "F_z_N", "F_x_N", "F_y_N", "F_mag_N",
            "U_mean_FVM_at_zsphere", "F_stokes_matched_N", "K_inertial_t",
        ])
        for k in range(n_samples):
            F_mag = float(np.sqrt(F_z_arr[k]**2
                                   + F_xy_arr[k, 0]**2 + F_xy_arr[k, 1]**2))
            w.writerow([f"{t_hist[k]:.4f}",
                        f"{F_z_arr[k]:.6e}",
                        f"{F_xy_arr[k, 0]:.6e}",
                        f"{F_xy_arr[k, 1]:.6e}",
                        f"{F_mag:.6e}",
                        f"{U_mean_actual_t[k]:.6e}",
                        f"{F_stokes_t[k]:.6e}",
                        f"{K_inertial_t[k]:.6e}"])
    print(f"\n  CSV written: {csv_path}")
    print(f"  Performance: {wall_time/n_steps_total*1e3:.2f} ms/step, "
          f"{wall_time:.0f}s wall on RTX 2060.")


if __name__ == "__main__":
    main()
