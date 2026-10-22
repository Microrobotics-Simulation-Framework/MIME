#!/usr/bin/env python3
"""MOD-1 / M1.0 — analytical under-damped step-out prediction (NO simulation).

The headline D1 question: the overdamped Man-Lauga step-out ceiling is
f_so,od = m·B/(2π·C_rot) ≈ 621 Hz, but de Jongh MEASURE ~27 Hz (23× lower).
This script tests, with 30 min of algebra, whether *body inertia* (the system is
strongly under-damped) collapses that gap before any FVM/coupled simulation runs.

The rotational dynamics in the field-rotating frame are a DRIVEN DAMPED PENDULUM
in the phase lag Δθ = (field angle − body angle):

    I_eff·Δθ̈ + C_rot·Δθ̇ + m·B·sin(Δθ) = C_rot·ω_field

Non-dimensionalise with ω_n = √(mB/I_eff), b = 2ζ = C_rot/(I_eff·ω_n):

    Δθ'' + 2ζ·Δθ' + sin(Δθ) = F,   F = 2ζ·(ω_field/ω_n)

Two thresholds (these are DIFFERENT and the gap between them is hysteretic):
  * Saddle-node (locked fixed point ceases to EXIST): F = 1
        → ω = ω_n/(2ζ) = m·B/C_rot  = the OVERDAMPED ceiling (621 Hz). Upper bound.
  * Homoclinic / Melnikov (running solution is BORN; locked basin collapses for
    ζ≪1): F_c = (4/π)·(2ζ)  → ω_po = (4/π)·ω_n,  INDEPENDENT of ζ.
        Melnikov: M = F·∮Δθ' dt − b·∮Δθ'² dt = 2π·F − 8b = 0 → F_c = 4b/π.
        For a strongly under-damped, perturbed/ramped robot the realistic step-out
        sits near the homoclinic threshold, NOT the saddle-node.

Also runs the Basset/unsteady-Stokes quasi-steady check β = √(ω·a²/ν) at the
operating frequency and at the predicted step-out.

Run: python scripts/mod1_stepout_spike.py
"""
from __future__ import annotations
import os, sys, math
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np


def fl9_inertia():
    """I_eff of the de Jongh FL-9 screw about its long (spin) axis.

    Cross-section ρ(θ)=R_cyl(1+ε sin Nθ). Solid (resin+magnets) density ≈ water+Δρ.
    Reports body-only and body+rotational-added-mass (the latter is an UPPER bound:
    axial rotation of a near-circular section entrains little fluid in potential
    flow; included only to bracket)."""
    R_cyl, L, eps = 1.56e-3, 7.47e-3, 0.33
    rho_body = 1000.0 + 410.0          # water + Δρ (de Jongh dense screw)
    rho_water = 1000.0
    # ⟨(1+ε s)^2⟩ = 1+ε²/2 ;  ⟨(1+ε s)^4⟩ = 1+3ε²+(3/8)ε⁴   (s=sin Nθ)
    a2 = 1.0 + eps**2 / 2.0
    a4 = 1.0 + 3.0 * eps**2 + (3.0 / 8.0) * eps**4
    V = math.pi * R_cyl**2 * a2 * L                       # solid volume
    m_body = rho_body * V
    # I_axial = ρ ∫ (1/4) ρ(θ)^4 dθ dz = ρ·(π/2)·R_cyl^4·a4·L
    I_body = rho_body * (math.pi / 2.0) * R_cyl**4 * a4 * L
    I_added = 0.5 * rho_water * math.pi * R_cyl**4 * L     # umr_ode form (upper bound)
    return {"R_cyl": R_cyl, "L": L, "V": V, "m_body": m_body,
            "I_body": I_body, "I_added": I_added,
            "I_eff_bodyonly": I_body, "I_eff_withadded": I_body + I_added}


def confined_drag():
    """C_rot (axial spin) and C_tumble (transverse) from the μ-fixed confined BEM."""
    from mime.experiments import schwarz_vessel_helix as S
    from mime.nodes.environment.stokeslet.cylinder_wall_table import load_wall_table
    mu = S._DEFAULTS["MU_PA_S"]
    m = S._screw_mesh({"N_THETA": 24, "N_ZETA": 36})
    table = load_wall_table(S._DEFAULTS["WALL_TABLE"])
    near = S._near_node({"N_THETA": 24, "N_ZETA": 36}, m, table, mu)
    R = np.asarray(near.resistance_matrix_si())
    return abs(R[5, 5]), 0.5 * (abs(R[3, 3]) + abs(R[4, 4]))


def basset_beta(f_hz, a, nu):
    return math.sqrt(2 * math.pi * f_hz * a**2 / nu)


def main():
    # --- confirmed magnetics (de Jongh paper) ---
    m_screw = 2 * 8.4e-4          # A·m²  (N=2 × S-01-01-N N45 1mm cube)
    B = 1.2e-3                    # T (measured @15cm)
    mB = m_screw * B
    f_measured = 27.0             # Hz (de Jongh 9mm UMR)

    inert = fl9_inertia()
    C_rot, C_tumble = confined_drag()

    print("=" * 74)
    print("MOD-1 / M1.0 — analytical under-damped step-out (no simulation)")
    print("=" * 74)
    print(f"  m·B = {mB:.3e} N·m   C_rot(axial) = {C_rot:.3e}   "
          f"C_tumble = {C_tumble:.3e} N·m·s")
    print(f"  I_body = {inert['I_body']:.3e}   I_added(upper) = {inert['I_added']:.3e}"
          f"   (m_body={inert['m_body']:.2e} kg, V={inert['V']:.2e} m³)")
    print("-" * 74)
    print(f"  Overdamped Man-Lauga ceiling  f_so = mB/(2π·C_rot) = "
          f"{mB/(2*math.pi*C_rot):8.1f} Hz   (= 23× too high vs 27)")
    print("-" * 74)
    for label, I in (("body-only", inert["I_eff_bodyonly"]),
                     ("body+added(upper)", inert["I_eff_withadded"])):
        wn = math.sqrt(mB / I)
        zeta = C_rot / (2.0 * math.sqrt(I * mB))
        f_n = wn / (2 * math.pi)
        f_homoclinic = (4.0 / math.pi) * wn / (2 * math.pi)   # (4/π)·ω_n
        print(f"  I_eff = {I:.3e} ({label}):")
        print(f"      ζ = {zeta:.4f}  (≪1 → strongly UNDER-damped)")
        print(f"      ω_n = √(mB/I) = {wn:6.1f} rad/s  → f_n = {f_n:5.1f} Hz")
        print(f"      homoclinic pull-out (4/π)·ω_n = {f_homoclinic:5.1f} Hz   "
              f"[vs measured {f_measured:.0f} Hz → ratio {f_homoclinic/f_measured:.2f}]")
    print("-" * 74)
    # Basset / unsteady-Stokes quasi-steady check
    print("  Basset / unsteady check  β = √(ω a²/ν):")
    f_so_pred = (4.0 / math.pi) * math.sqrt(mB / inert["I_eff_bodyonly"]) / (2 * math.pi)
    for fluid, nu in (("water (de Jongh expt)", 1.0e-6), ("blood (clinical)", 3.8e-6)):
        for aname, a in (("R_cyl=1.56mm", 1.56e-3), ("fin r≈0.4mm", 4.0e-4)):
            b10 = basset_beta(10.0, a, nu)
            bso = basset_beta(f_so_pred, a, nu)
            print(f"      {fluid:24s} {aname:13s}: β(10Hz)={b10:5.2f}  "
                  f"β(f_so≈{f_so_pred:.0f}Hz)={bso:5.2f}")
    print("=" * 74)
    print("READ: homoclinic pull-out ≈ 20-25 Hz vs measured 27 Hz (overdamped=621). "
          "Inertia is the DOMINANT mechanism. Basset β≳1 ⇒ unsteady drag NOT "
          "negligible — flag for D1.")


if __name__ == "__main__":
    main()
