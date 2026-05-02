"""A/B test: MLPResistanceNode vs StokesletFluidNode for FL-9 helix.

Probe both with the same set of (velocity, angular_velocity) inputs in
body frame, compare the resulting drag force/torque. If they disagree
substantially, the MLP surrogate is mis-calibrated and is responsible
for the over-fast swim speed in the AR4 chain.

Output is a table of:
  - U/Ω input → MLP F/τ vs BEM F/τ for representative cases:
    (a) pure rotation about helical axis
    (b) pure translation along helical axis
    (c) helical-corkscrew motion (combined ω, U)
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, "/home/nick/MSF/MIME/src")

import numpy as np
import jax.numpy as jnp

from mime.nodes.environment.stokeslet.fluid_node import StokesletFluidNode
from mime.nodes.environment.stokeslet.dejongh_geometry import dejongh_fl_mesh
from mime.nodes.environment.stokeslet.mlp_resistance_node import MLPResistanceNode
from mime.experiments.dejongh import default_mlp_weights_path, FL_PARAMS, R_CYL_UMR_MM, VESSELS

DT = 5e-4
MU = 1e-3
DESIGN = 9
VESSEL = '1/4"'

print("Building FL-9 surface mesh (coarse for tractable BEM)...", flush=True)
mesh = dejongh_fl_mesh(DESIGN, n_theta=24, n_zeta=40)
print(f"  N_points = {mesh.n_points}", flush=True)

print("Building StokesletFluidNode (BEM, standalone unconfined)...", flush=True)
bem = StokesletFluidNode(
    name="bem", timestep=DT, mu=MU, body_mesh=mesh,
)
R_bem = bem._R
print(f"  Resistance matrix R: shape={R_bem.shape}")
print(f"  R[0:3,0:3] (translation block, N·s/m):\n{R_bem[0:3, 0:3]}")
print(f"  R[3:6,3:6] (rotation block, N·m·s):\n{R_bem[3:6, 3:6]}")
print(f"  R[3:6,0:3] (chirality coupling, N·s):\n{R_bem[3:6, 0:3]}")

print("\nBuilding MLPResistanceNode...", flush=True)
fl = FL_PARAMS["FL-9"]
mlp = MLPResistanceNode(
    name="mlp", timestep=DT,
    mlp_weights_path=str(default_mlp_weights_path()),
    nu=fl["nu"], L_UMR_mm=fl["L_UMR_mm"],
    R_cyl_UMR_mm=R_CYL_UMR_MM,
    R_ves_mm=VESSELS[VESSEL],
    mu_Pa_s=MU,
)

# Probe MLP at a representative pose (centred, identity orientation)
mlp_state = mlp.initial_state()
mlp_inputs = {
    "robot_position": jnp.array([0.0, 0.0, 0.0]),
    "robot_orientation": jnp.array([1.0, 0.0, 0.0, 0.0]),
    "body_velocity": jnp.zeros(3),
    "body_angular_velocity": jnp.zeros(3),
}
mlp_out = mlp.update(mlp_state, mlp_inputs, DT)
R_mlp = np.asarray(mlp_out["resistance_matrix"])
print(f"  R[0:3,0:3] (translation block):\n{R_mlp[0:3, 0:3]}")
print(f"  R[3:6,3:6] (rotation block):\n{R_mlp[3:6, 3:6]}")
print(f"  R[3:6,0:3] (chirality coupling):\n{R_mlp[3:6, 0:3]}")

# ── Drive both with the same inputs ────────────────────────────────
print("\n" + "=" * 70)
print("A/B comparison")
print("=" * 70)

# In the dejongh experiment the helix axis is body-z. After the +90°
# rotation about world-y, body-z → world-x.  For both nodes we pass
# velocities in WORLD frame, but for this comparison we drive the
# helix in its rest orientation (body-z = world-z) and compare the
# resistance directly.
omega_drive = 2 * np.pi * 10.0  # 10 Hz
test_cases = [
    ("pure rotation (ω_z = +63 rad/s)",
        np.zeros(3), np.array([0, 0, omega_drive])),
    ("pure translation (U_z = +1 mm/s)",
        np.array([0, 0, 1e-3]), np.zeros(3)),
    ("corkscrew (ω_z=63, U_z=swim)",
        np.array([0, 0, 3e-3]), np.array([0, 0, omega_drive])),
]

print(f"\n{'case':<45} {'F_mlp (µN)':>20} {'F_bem (µN)':>20} {'τ_mlp (µN·m)':>22} {'τ_bem (µN·m)':>22}")
for name, U, omega in test_cases:
    UO = np.concatenate([U, omega])
    F_mlp = -R_mlp[0:3, 0:3] @ U - R_mlp[0:3, 3:6] @ omega
    tau_mlp = -R_mlp[3:6, 0:3] @ U - R_mlp[3:6, 3:6] @ omega
    F_bem = -R_bem[0:3, 0:3] @ U - R_bem[0:3, 3:6] @ omega
    tau_bem = -R_bem[3:6, 0:3] @ U - R_bem[3:6, 3:6] @ omega

    print(f"{name:<45} "
          f"{tuple(round(float(x)*1e6, 3) for x in F_mlp)!s:>20} "
          f"{tuple(round(float(x)*1e6, 3) for x in F_bem)!s:>20} "
          f"{tuple(round(float(x)*1e6, 3) for x in tau_mlp)!s:>22} "
          f"{tuple(round(float(x)*1e6, 3) for x in tau_bem)!s:>22}")

# ── Self-propulsion equilibrium: solve for swim speed under a given torque ──
print("\n" + "=" * 70)
print("Self-propelled swim speed under applied torque (paper says ~3 mm/s for FL-9 at 10 Hz, 1.2 mT)")
print("=" * 70)
# Apply commanded torque about z, F_total = 0 (free swimmer).
# Solve: 0 = -R_TT·U - R_TR·ω
#        τ_applied = R_RT·U + R_RR·ω
# Equivalently: from the 6×6 system [F; τ] = -R [U; ω]; with F = 0,
# eliminate U and find ω, then U from constraint.
def swim_from_torque(R, tau_z):
    # Project to z-axis only: U_z, ω_z
    R_local = np.array([[R[2, 2], R[2, 5]], [R[5, 2], R[5, 5]]])
    # F_z = 0:  R[2,2] U_z + R[2,5] ω_z = 0  →  U_z = -(R[2,5]/R[2,2]) ω_z
    # τ_z = R[5,2] U_z + R[5,5] ω_z
    # With F=0:  τ_z = (R[5,5] - R[5,2]·R[2,5]/R[2,2]) ω_z
    omega_z = tau_z / (R[5, 5] - R[5, 2] * R[2, 5] / R[2, 2])
    U_z = -(R[2, 5] / R[2, 2]) * omega_z
    return U_z, omega_z

# Estimate driving torque at 10 Hz, 1.2 mT, 1.68e-3 A·m² helix dipole:
# τ_drive ≈ |m_helix| × |B| (lag-aligned in steady state)
tau_drive = 1.68e-3 * 1.2e-3  # 2 µN·m peak
print(f"Applied torque about helical axis (z): {tau_drive*1e6:.3f} µN·m")

U_mlp, w_mlp = swim_from_torque(R_mlp, tau_drive)
U_bem, w_bem = swim_from_torque(R_bem, tau_drive)
print(f"  MLP   → swim {U_mlp*1000:>8.3f} mm/s   ω {w_mlp/(2*np.pi):>6.2f} Hz")
print(f"  BEM   → swim {U_bem*1000:>8.3f} mm/s   ω {w_bem/(2*np.pi):>6.2f} Hz")
print(f"  Paper → swim ~3 mm/s at 10 Hz")
