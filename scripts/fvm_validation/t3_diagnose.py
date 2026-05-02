"""Diagnose why FVM-IBM drag is small at low Re.

For λ=0.3 at Re_pipe=0.01:
  * Run flow without sphere, verify Poiseuille profile.
  * Run flow with sphere, measure mean axial velocity (mass flux).
  * Compute F_sphere = (f_steady * V_pipe - 8πμU_mean L) — force balance.
  * Compare to the IBM-extracted force.
  * Print u_after_explicit values at body cells.
"""
from __future__ import annotations
import numpy as np
import jax, jax.numpy as jnp
import time

from mime.nodes.environment.fvm import make_cartesian_mesh_3d
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.piso import PisoConfig, run_piso, make_piso_step, initial_state
from mime.nodes.environment.fvm.ibm import IBMBody, compute_ibm_forces, smoothed_indicator
from mime.nodes.environment.fvm.sdf import sphere_sdf

R_pipe = 0.5; L_pipe = 1.0; nu = 1.0
N_cross, N_axial = 32, 16
margin = 1.2
Lx = Ly = 2*margin*R_pipe
mesh = make_cartesian_mesh_3d(N_cross, N_cross, N_axial, Lx, Ly, L_pipe,
                               origin=(-Lx/2, -Ly/2, 0.0), periodic_z=True)
dx = mesh.cartesian_spacing[0]
print(f"Mesh: {mesh.N_cells} cells, dx={dx:.4f}")

# Body force for U_centre = 0.02
U_centre_target = 0.02
f_steady = U_centre_target * 4 * nu / R_pipe**2
print(f"body_force = {f_steady}")

def pipe_wall(x):
    rho = jnp.sqrt(x[..., 0]**2 + x[..., 1]**2 + 1e-30)
    return R_pipe - rho
wall = IBMBody(name="wall", sdf=pipe_wall)

bcs = {}
for name in ("x_min", "x_max", "y_min", "y_max"):
    p = mesh.patch(name); nbf = int(p.owner.size)
    bcs[name] = VelocityBC(u_wall=jnp.zeros((nbf, 3)), F_through=jnp.zeros((nbf,)))

def body_force(t):
    return jnp.array([0.0, 0.0, f_steady])

cfg = PisoConfig(nu=nu, rho=1.0, gamma_conv=1.0, n_corrector=2,
                 pressure_bc=("neumann", "neumann", "periodic"),
                 velocity_bc=("dirichlet", "dirichlet", "periodic"),
                 ibm_alpha=1e5, ibm_eps=1.0*dx)
dt = 0.05

# (1) No sphere
print("\n--- No sphere ---")
state = None
for _ in range(8):
    state = run_piso(mesh, bcs, cfg, n_steps=200, dt=dt,
                     body_force_fn=body_force, ibm_bodies=[wall], initial=state)
state["u"].block_until_ready()
u = np.asarray(state["u"]).reshape(N_cross, N_cross, N_axial, 3)
U_centre_no_sphere = float(u[N_cross//2, N_cross//2, N_axial//2, 2])
# Mean axial velocity (= mass flux / pipe area). Use only fluid cells.
phi = np.asarray(pipe_wall(mesh.x))
fluid_mask = (phi >= 0).reshape(N_cross, N_cross, N_axial)  # fluid where phi>=0 (outside wall body)
U_mean_no_sphere = float(np.sum(u[..., 2] * fluid_mask) / np.sum(fluid_mask))
print(f"  U_centre numerical = {U_centre_no_sphere:.5f} (target {U_centre_target})")
print(f"  U_mean numerical   = {U_mean_no_sphere:.5f} (Poiseuille = {U_centre_target/2})")

# Hagen-Poiseuille: f * pi R² = 8πμU_mean ⇒ U_mean = f R²/(8μ) = 0.005 ✓

# (2) With sphere at λ=0.3
print("\n--- With sphere λ=0.3 ---")
r_s = 0.3 * R_pipe
sphere_centre = jnp.array([0.0, 0.0, L_pipe/2])
def sphere_sdf_fn(x):
    return sphere_sdf(x, center=sphere_centre, radius=r_s)
sphere = IBMBody(name="sphere", sdf=sphere_sdf_fn,
                 extract_force=True, ref_point=sphere_centre)

state = None
for _ in range(12):
    state = run_piso(mesh, bcs, cfg, n_steps=200, dt=dt,
                     body_force_fn=body_force, ibm_bodies=[wall, sphere], initial=state)
state["u"].block_until_ready()

u = np.asarray(state["u"]).reshape(N_cross, N_cross, N_axial, 3)
u_ae = np.asarray(state["u_after_explicit"]).reshape(N_cross, N_cross, N_axial, 3)
u_pi = np.asarray(state["u_pre_ibm"]).reshape(N_cross, N_cross, N_axial, 3)

# Mass flux: integrate u_z over a z-cross-section
iz_probe = N_axial // 4   # away from sphere
mass_flux = float(np.sum(u[:, :, iz_probe, 2]) * dx * dx)
pipe_area = np.pi * R_pipe**2
U_mean_sphere = mass_flux / pipe_area
print(f"  U_mean (mass flux/A_pipe) = {U_mean_sphere:.5f}")

# Force balance: F_sphere = f_steady * V_pipe - F_wall
# Hagen-Poiseuille predicts F_wall = 8πμU_mean L_pipe
V_pipe = np.pi * R_pipe**2 * L_pipe
F_wall_HP = 8 * np.pi * nu * U_mean_sphere * L_pipe
F_sphere_balance = f_steady * V_pipe - F_wall_HP
print(f"  body force total       = {f_steady * V_pipe:.5f}")
print(f"  Wall drag (HP est)     = {F_wall_HP:.5f}")
print(f"  Sphere drag (balance)  = {F_sphere_balance:.5f}")

# IBM-extracted force
forces = compute_ibm_forces(state["u_after_explicit"], mesh.x, mesh.V, [wall, sphere],
                             alpha=cfg.ibm_alpha, eps=cfg.ibm_eps, rho=cfg.rho, dt=dt)
F_sphere_IBM = float(forces["sphere"]["force"][2])
print(f"  Sphere drag (IBM)      = {F_sphere_IBM:.5f}")

# Try Goldstein formula too (no dt)
forces_G = compute_ibm_forces(state["u_after_explicit"], mesh.x, mesh.V, [wall, sphere],
                               alpha=cfg.ibm_alpha, eps=cfg.ibm_eps, rho=cfg.rho, dt=None)
F_sphere_Goldstein = float(forces_G["sphere"]["force"][2])
print(f"  Sphere drag (Goldstein) = {F_sphere_Goldstein:.5f}")

# Stokes reference
F_stokes = 6 * np.pi * nu * r_s * U_centre_target
print(f"\n  6πμaU_centre (Stokes unbounded) = {F_stokes:.5f}")
print(f"  K_FVM_IBM = {F_sphere_IBM/F_stokes:.4f}")
print(f"  K_FVM_balance = {F_sphere_balance/F_stokes:.4f}")
print(f"  K_HS analytical = 2.37")

# Inspect u inside body
phi_s = np.asarray(sphere_sdf_fn(mesh.x))
I_s = np.asarray(smoothed_indicator(jnp.asarray(phi_s), 1.0*dx))
inside_idx = np.argsort(phi_s)[:5]  # 5 most-inside cells
print(f"\n  u_after_explicit at 5 most-inside cells:")
for idx in inside_idx:
    print(f"    phi={phi_s[idx]:+.4f} I={I_s[idx]:.3f} u_after_z={u_ae.reshape(-1, 3)[idx, 2]:.6e} "
          f"u_pre_ibm_z={u_pi.reshape(-1, 3)[idx, 2]:.6e} u_z={u.reshape(-1, 3)[idx, 2]:.6e}")
