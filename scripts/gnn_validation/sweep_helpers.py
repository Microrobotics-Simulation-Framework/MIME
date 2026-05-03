"""Helpers for the enhanced sweep runner: WSS, profiles, GNN norm,
   pulsatile runs with waveform capture, atomic JSON manifest."""
from __future__ import annotations
import json
import os
import time
from pathlib import Path

import numpy as np
import jax, jax.numpy as jnp

from mime.nodes.environment.fvm import (
    make_pipe_mesh, make_poiseuille_lift, make_poiseuille_p_lift,
    make_womersley_lift_analytical,
)
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.piso import (
    PisoConfig, run_piso, run_piso_with_history,
)
from mime.nodes.environment.fvm.ibm import IBMBody, surface_integral_force
from mime.nodes.environment.fvm.sdf import sphere_sdf
from mime.nodes.environment.fvm.operators import grad_green_gauss


def happel_brenner(lam):
    return 1.0/(1.0-2.10443*lam+2.08877*lam**3-0.94813*lam**5
                -1.372*lam**6+3.87*lam**8-4.19*lam**10)


# ---------------------------------------------------------------------------
# Geometry helpers (single source of truth: same setup as Step 1)
# ---------------------------------------------------------------------------

def build_setup(*, lambda_, cpr, U_dc, Wo=0.0, T_cycle=1.0):
    """Build mesh + bodies + cfg + lift consistent with step1_generate_data.

    For Wo > 0 the lift is the analytical Womersley with U_mean(t) =
    U_dc + 0.5·U_dc · cos(ωt + phase_offset). Setting U_amp = 0.5·U_dc
    keeps the peak at 1.5·U_dc — well below the 2× headroom that caused
    M1 instability without warmup.
    """
    r_b = 1e-3
    R_pipe = r_b / lambda_
    sphere_margin = 5.0; bc_margin = 5.0
    L_pipe = 2.0 * (sphere_margin + bc_margin) * r_b + 2.0 * r_b
    nu = 1e-3
    rho = 1.0
    mu = rho * nu
    mesh = make_pipe_mesh(pipe_radius=R_pipe, pipe_length=L_pipe,
                          robot_radius=r_b, cpr=cpr)
    dx = mesh.cartesian_spacing[0]
    Nz = mesh.cartesian_shape[2]
    L_actual = Nz * dx
    sphere_centre = jnp.array([0.0, 0.0, L_actual / 2], dtype=mesh.V.dtype)

    def pipe_wall_sdf(x):
        rxy = jnp.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2 + 1e-30)
        return R_pipe - rxy
    def sphere_sdf_fn(x):
        return sphere_sdf(x, center=sphere_centre, radius=r_b)
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
        nu=nu, rho=rho, gamma_conv=0.0, n_corrector=2,
        pressure_bc="neumann", velocity_bc="dirichlet",
        ibm_alpha=1e5, ibm_eps=1.0 * dx,
    )

    if Wo > 0.0:
        omega = (Wo ** 2) * nu / (R_pipe ** 2)
        U_amp = 0.5 * U_dc
        L_lift = make_womersley_lift_analytical(
            mesh, R_pipe=R_pipe, U_mean_dc=U_dc, U_mean_amp=U_amp,
            omega=omega, nu=nu, axis=2, phase_offset=-np.pi / 2,
        )
    else:
        omega = 0.0
        U_amp = 0.0
        L_lift = make_poiseuille_lift(mesh, R_pipe=R_pipe,
                                        U_mean=U_dc, axis=2)

    return dict(
        mesh=mesh, bcs=bcs, cfg=cfg, bodies=bodies, lift=L_lift,
        sphere_centre=sphere_centre, sphere_sdf_fn=sphere_sdf_fn,
        pipe_wall_sdf=pipe_wall_sdf,
        dx=dx, L_actual=L_actual, R_pipe=R_pipe, r_b=r_b,
        mu=mu, rho=rho, nu=nu,
        U_dc=U_dc, U_amp=U_amp, omega=omega, Wo=Wo, T_cycle=T_cycle,
    )


def k_FVM(d, state, u_phys=None):
    """surface_integral K at shell (0.5, 2.5) with p_lift correction."""
    if u_phys is None:
        u_phys = state["u"] + d["lift"].u_lift_static
    p_lift_fn = make_poiseuille_p_lift(
        mu=d["mu"], U_mean=d["U_dc"], pipe_radius=d["R_pipe"],
    )
    F_vec, _ = surface_integral_force(
        u_phys, state["p"], d["mesh"], d["sphere_sdf_fn"],
        mu=d["mu"], dx=d["dx"], shell_inner=0.5, shell_outer=2.5,
        ref_point=d["sphere_centre"], p_lift_fn=p_lift_fn, pipe_axis=2,
    )
    F_z = float(F_vec[2])
    F_uncon = 6.0 * np.pi * d["mu"] * d["r_b"] * (2 * d["U_dc"])
    return F_z, F_z / F_uncon


# ---------------------------------------------------------------------------
# Steady run
# ---------------------------------------------------------------------------

def run_steady(d, n_steps=400):
    dt = min(0.5, 0.5 * d["dx"] / max(2 * d["U_dc"], 1e-30))
    t0 = time.time()
    state = run_piso(d["mesh"], d["bcs"], d["cfg"], n_steps=n_steps, dt=dt,
                     body_force_fn=None, ibm_bodies=d["bodies"],
                     lifting=d["lift"])
    state["u"].block_until_ready()
    wall = time.time() - t0
    return state, dt, wall


# ---------------------------------------------------------------------------
# Pulsatile run with cycle-3 force waveform capture
# ---------------------------------------------------------------------------

def run_pulsatile(d, n_cycles=3, samples_per_cycle=40):
    """Run n_cycles of analytical Womersley with PISO history every
    sample_every steps. Returns (final_state, history, dt, wall, t_hist).

    The cycle period is the *natural* one (2π/omega), NOT the cardiac
    T_cycle stored on d. For a sweep config with Wo set explicitly,
    omega = Wo²·ν/R² is what determines the oscillation period — using
    a hard-coded T_cycle would otherwise force thousands of unused
    PISO steps when omega is large (e.g., Wo=3 gives T_natural ≈ 70 ms,
    not 1 s).
    """
    u_max = 2.0 * (d["U_dc"] + d["U_amp"])
    dt = 0.4 * d["dx"] / max(u_max, 1e-30)
    omega = d["omega"]
    if omega <= 0:
        raise ValueError("run_pulsatile requires omega > 0")
    T_natural = 2.0 * np.pi / omega
    n_per_cycle = int(round(T_natural / dt))
    n_total = n_per_cycle * n_cycles
    sample_every = max(1, n_per_cycle // samples_per_cycle)
    t0 = time.time()
    state, hist = run_piso_with_history(
        d["mesh"], d["bcs"], d["cfg"], n_steps=n_total, dt=dt,
        body_force_fn=None, ibm_bodies=d["bodies"], lifting=d["lift"],
        sample_every=sample_every,
    )
    state["u"].block_until_ready()
    wall = time.time() - t0
    return state, hist, dt, wall, sample_every, n_per_cycle


def cycle_force_waveform(d, hist, sample_every, n_per_cycle, cycle_idx=2):
    """Reconstruct u_phys at each sample and extract F_z, F_r per sample.

    Returns numpy arrays Fz_cycle, Fr_cycle, t_cycle, U_inlet_cycle.
    """
    L = d["lift"]
    u_steady_np = np.asarray(L.u_lift_static)
    U_re = np.asarray(L.U_re); U_im = np.asarray(L.U_im)
    omega_np = float(L.omega)
    p_lift_fn = make_poiseuille_p_lift(
        mu=d["mu"], U_mean=d["U_dc"], pipe_radius=d["R_pipe"],
    )

    u_hist = np.asarray(hist["u"])
    p_hist = np.asarray(hist["p"])
    t_hist = np.asarray(hist["t"])
    n_samples_per_cycle = max(1, n_per_cycle // sample_every)
    start = cycle_idx * n_samples_per_cycle
    end   = (cycle_idx + 1) * n_samples_per_cycle
    end = min(end, len(t_hist))

    Fz = np.zeros(end - start)
    Fr = np.zeros(end - start)
    U_inlet = np.zeros(end - start)
    for j, k in enumerate(range(start, end)):
        t_k = float(t_hist[k])
        cwt = np.cos(omega_np * t_k); swt = np.sin(omega_np * t_k)
        u_lift_k = u_steady_np + cwt * U_re - swt * U_im
        u_phys_k = u_hist[k] + u_lift_k
        F_vec, _ = surface_integral_force(
            jnp.asarray(u_phys_k), jnp.asarray(p_hist[k]),
            d["mesh"], d["sphere_sdf_fn"],
            mu=d["mu"], dx=d["dx"], shell_inner=0.5, shell_outer=2.5,
            ref_point=d["sphere_centre"], p_lift_fn=p_lift_fn, pipe_axis=2,
        )
        Fz[j] = float(F_vec[2])
        Fr[j] = float(np.sqrt(float(F_vec[0])**2 + float(F_vec[1])**2))
        U_inlet[j] = d["U_dc"] + d["U_amp"] * cwt   # phase_offset=-π/2 ⇒ cos·cos=cos
    return Fz, Fr, t_hist[start:end], U_inlet


# ---------------------------------------------------------------------------
# Wall shear stress
# ---------------------------------------------------------------------------

def compute_wall_shear_stress(d, state):
    """Mean / max / std WSS at the pipe wall band.

    Wall band = cells with R-pipe_wall_sdf in (0.5*dx, 1.5*dx) — one
    cell layer just inside the IBM diffuse zone. WSS = μ |∂u/∂n|.
    """
    mesh = d["mesh"]; mu = d["mu"]; dx = d["dx"]
    u_phys = state["u"] + d["lift"].u_lift_static
    grad_u = jnp.stack(
        [grad_green_gauss(u_phys[:, k], mesh) for k in range(3)],
        axis=1,
    )    # [N_cells, 3, 3]   g[i, k, j] = ∂u_k/∂x_j
    rxy = jnp.sqrt(mesh.x[:, 0] ** 2 + mesh.x[:, 1] ** 2 + 1e-30)
    band = (rxy > d["R_pipe"] - 1.5 * dx) & (rxy < d["R_pipe"] - 0.5 * dx)
    # Outward radial unit vector at each cell (in cross-section plane)
    n_x = mesh.x[:, 0] / rxy
    n_y = mesh.x[:, 1] / rxy
    # ∂u_k / ∂r = ∂u_k/∂x · n_x + ∂u_k/∂y · n_y
    du_dr = grad_u[:, :, 0] * n_x[:, None] + grad_u[:, :, 1] * n_y[:, None]
    tau = mu * jnp.linalg.norm(du_dr, axis=-1)   # [N_cells]
    band = band.astype(tau.dtype)
    n_band = float(band.sum())
    if n_band < 1:
        return 0.0, 0.0, 0.0
    tau_band = tau * band
    tau_mean = float(tau_band.sum() / n_band)
    tau_max  = float(jnp.max(tau_band))
    tau_std  = float(jnp.sqrt(((tau_band - tau_mean) ** 2 * band).sum() / n_band))
    return tau_mean, tau_max, tau_std


# ---------------------------------------------------------------------------
# Radial velocity profile at sphere mid-plane
# ---------------------------------------------------------------------------

def radial_velocity_profile(d, state, n_bins=20):
    """Cross-section-averaged u_z at the sphere mid-plane, binned in r."""
    mesh = d["mesh"]
    Nx, Ny, Nz = mesh.cartesian_shape
    iz = Nz // 2
    x = np.asarray(mesh.x).reshape(Nx, Ny, Nz, 3)
    u_phys = np.asarray(state["u"] + d["lift"].u_lift_static).reshape(
        Nx, Ny, Nz, 3,
    )
    rxy = np.sqrt(x[:, :, iz, 0] ** 2 + x[:, :, iz, 1] ** 2)
    uz  = u_phys[:, :, iz, 2]
    edges = np.linspace(0, d["R_pipe"], n_bins + 1)
    profile = np.zeros(n_bins)
    for b in range(n_bins):
        mask = (rxy >= edges[b]) & (rxy < edges[b+1])
        profile[b] = float(uz[mask].mean()) if mask.any() else 0.0
    return profile


# ---------------------------------------------------------------------------
# GNN correction diagnostics
# ---------------------------------------------------------------------------

def gnn_correction_diag(corrector, d, state, dt):
    """Norm and max of GNN delta_u_face on this state."""
    if corrector is None:
        return None, None
    u_phys = state["u"] + d["lift"].u_lift_static
    delta = corrector.apply(
        u_phys, state["p"], d["mesh"], correction_weight=1.0,
        u_prev_cell=u_phys, dt=dt, U_ref=d["U_dc"], r_b=d["r_b"],
    )
    return float(jnp.linalg.norm(delta)), float(jnp.max(jnp.abs(delta)))


# ---------------------------------------------------------------------------
# Strouhal number from steady-Re Fz history
# ---------------------------------------------------------------------------

def strouhal_from_history(Fz_series, dt, r_b, U_mean):
    """Peak-frequency Strouhal from a steady-flow Fz history (Re ≥ ~100)."""
    Fz = np.asarray(Fz_series, dtype=float)
    n = len(Fz)
    if n < 16:
        return None, None
    half = Fz[n // 2:]      # post-transient
    half = half - half.mean()
    freqs = np.fft.rfftfreq(len(half), d=float(dt))
    power = np.abs(np.fft.rfft(half)) ** 2
    if len(freqs) <= 1:
        return None, None
    idx = int(np.argmax(power[1:]) + 1)
    f_shed = float(freqs[idx])
    St = f_shed * 2 * r_b / U_mean
    return f_shed, St


# ---------------------------------------------------------------------------
# Phase lag between F_z(t) and U_inlet(t)
# ---------------------------------------------------------------------------

def phase_lag(Fz_cycle, U_cycle):
    """Position-of-peak phase lag in radians (mod 2π)."""
    n = len(Fz_cycle)
    t_F = np.argmax(Fz_cycle) / n * 2 * np.pi
    t_U = np.argmax(U_cycle) / n * 2 * np.pi
    return float((t_F - t_U) % (2 * np.pi))


# ---------------------------------------------------------------------------
# Atomic manifest append (POSIX rename)
# ---------------------------------------------------------------------------

def append_to_manifest(manifest_path: Path, entry: dict):
    if manifest_path.exists():
        with open(manifest_path) as f:
            data = json.load(f)
    else:
        data = []
    # Replace entry with same label if present (idempotent)
    data = [e for e in data if e.get("label") != entry.get("label")]
    data.append(entry)
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, manifest_path)
