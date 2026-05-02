"""Sweep (standoff, dipole) for the AR4 RPM driver.

For each (r, m) combination, simulate 0.3 s of dynamics and report:

- |B| at the helix on-axis (μT)
- |F_grad|/|F_gravity| ratio (so ~1 means gradient effect comparable
  to gravity, much greater means the helix gets yanked around)
- orbital amplitude in the tube cross-section (mm); want << 1 mm
- swim speed along the tube (mm/s); paper reports ~3 mm/s for FL-9

The sweep doesn't run the full graph — it directly evaluates the
PermanentMagnetNode field/gradient analytics and a closed-form drag
estimate for the helix under the resulting torque. That keeps the
sweep cheap (~5 s for a 5×5 grid) so you can iterate.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, "/home/nick/MSF/MIME/src")

import jax
import jax.numpy as jnp
import numpy as np

MU0_4PI = 1e-7

# ── Helix properties (FL-9 onboard, dejongh defaults) ───────────────
M_HELIX = 1.68e-3                  # 2 × 1mm³ N45 magnets, A·m²
HELIX_VOL = float(np.pi * (1.56e-3)**2 * 7.47e-3 * 1.0545)  # ≈ 6.1e-8 m³
DELTA_RHO = 410.0                  # kg/m³
GRAVITY = 9.81
F_GRAVITY = DELTA_RHO * HELIX_VOL * GRAVITY    # N

# Translational drag of the helix in water (Stokes-equivalent radius)
# — order-of-magnitude only; the real MLP-Cholesky drag is anisotropic.
MU_WATER = 1e-3                    # Pa·s
HELIX_RADIUS = 1.56e-3             # m
DRAG_T = 6 * np.pi * MU_WATER * HELIX_RADIUS

# Rotational drag (sphere approx, μN·m·s/rad)
DRAG_R = 8 * np.pi * MU_WATER * HELIX_RADIUS**3

# ── Sweep grid ──────────────────────────────────────────────────────
DISTANCES_CM = [15, 20, 25, 30, 35, 40, 50]
DIPOLES_AM2  = [0.5, 1.0, 3.0, 10.0, 18.89]   # 18.89 = de Jongh paper
FREQ_HZ = 10.0
SIM_S = 0.3

OMEGA_DRIVE = 2 * np.pi * FREQ_HZ

print(f"{'r_cm':>5} {'m_Am²':>8} {'B_µT':>10} {'F_grad_µN':>11} "
      f"{'F_grad/F_g':>11} {'τ_drive_µNm':>12} {'τ_drag_µNm':>11} "
      f"{'ω_helix_Hz':>11} {'orbit_mm':>10} {'swim_mm_s':>10}")

for r_cm in DISTANCES_CM:
    r = r_cm / 100
    for m_ext in DIPOLES_AM2:
        # On-axis field for a point dipole at distance r:
        # B = (μ₀/4π) · 2m/r³ along the dipole axis.
        # We're at the perpendicular position (paper's geometry), so
        # |B| = (μ₀/4π) · m/r³ when dipole ⊥ separation, sweeping a
        # mix of perpendicular (1) and along-r (2x) components.
        # Use the Mahoney/Abbott form: |B(θ)|² = (μ₀m/(4π·r³))²·(1+3cos²θ).
        # RMS over revolution: <|B|²> = K²·(1 + 3·1/2) = 2.5 K². RMS ≈ 1.58 K.
        K = MU0_4PI * m_ext / r**3
        B_rms = K * np.sqrt(2.5)
        # Peak gradient force on helix dipole ~ μ₀ · m_helix · m_ext / r⁴.
        # Coefficient ~3 from paper Eq 4.
        F_grad_peak = 3 * MU0_4PI * M_HELIX * m_ext / r**4

        # Drive torque: τ = m_helix · B
        tau_drive = M_HELIX * B_rms   # peak ~ 1.5× this
        tau_drag = DRAG_R * OMEGA_DRIVE
        # Helix angular velocity: τ_drive vs τ_drag. If τ_drive < τ_drag,
        # below step-out → ω_helix ≈ (τ_drive/DRAG_R). If above step-out,
        # ω_helix capped at OMEGA_DRIVE.
        omega_helix = min(tau_drive / DRAG_R, OMEGA_DRIVE)

        # Orbital amplitude: terminal velocity from F_grad oscillating
        # at OMEGA_DRIVE, amplitude = v_term / ω = F_grad / (DRAG_T · ω)
        orbit_amp = F_grad_peak / (DRAG_T * OMEGA_DRIVE)

        # Swim speed: helical pitch coupling. FL-9 helix has chirality
        # parameter ~ 0.5 (i.e. for each rotation it advances 0.5 ×
        # body-length along its long axis). At 7.47 mm body length,
        # one rotation translates ~0.4 mm. Speed = ω_helix/(2π) × pitch.
        swim_speed = omega_helix / (2 * np.pi) * 0.4e-3  # m/s

        print(f"{r_cm:>5} {m_ext:>8.2f} {B_rms*1e6:>10.1f} "
              f"{F_grad_peak*1e6:>11.3f} {F_grad_peak/F_GRAVITY:>11.1f} "
              f"{tau_drive*1e6:>12.4f} {tau_drag*1e6:>11.4f} "
              f"{omega_helix/(2*np.pi):>11.2f} {orbit_amp*1000:>10.3f} "
              f"{swim_speed*1000:>10.3f}")

print()
print(f"FL-9 gravity force: {F_GRAVITY*1e6:.3f} µN")
print(f"Translational drag coefficient: {DRAG_T:.2e} N·s/m")
print(f"Rotational drag coefficient: {DRAG_R:.2e} N·m·s")
print(f"Drive ω = {OMEGA_DRIVE:.2f} rad/s ({FREQ_HZ:.1f} Hz)")
print()
print("Acceptance criteria for the visualisation:")
print("  - F_grad/F_g  ≲ 5    (helix not violently yanked around)")
print("  - orbit_mm    ≲ 0.5  (smaller than tube radius 1 mm)")
print("  - ω_helix_Hz  ≥ 1    (helix actually rotates)")
print("  - swim_mm_s   ≲ 10   (does not exit 100 mm tube in <10s)")
