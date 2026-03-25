#!/usr/bin/env python3
"""FSI-coupled step-out demonstration — d2.8 UMR, 1 magnet, ratio 0.30.

Runs the full four-node MADDENING graph (ExternalMagneticFieldNode ->
PermanentMagnetResponseNode -> RigidBodyNode <-> IBLBMFluidNode) at 64^3
with a ramping field frequency. The field frequency increases linearly
from 0.5x to 1.8x the predicted step-out frequency. The UMR tracks the
field in the synchronous regime then loses synchrony at step-out.

Produces a three-panel plot showing the step-out transition:
  Panel 1: field frequency (dimensionless, omega_field / omega_so)
  Panel 2: synchrony ratio (omega_body / omega_field)
  Panel 3: drag torque (N*m, physical units)

DIMENSIONLESS FORMULATION
-------------------------
The step-out frequency shift (%) is independent of magnet count because
f_step ~ n*m*B / C_rot — the fractional shift depends only on the drag
multiplier, which is geometry- and confinement-dependent but
magnet-count-independent. The absolute Hz shift scales proportionally
with n_magnets (and thus with unconfined f_step).

B_demo is derived forward from the LBM Mach number constraint: the
step-out angular frequency omega_so must map to a Ma-safe lattice omega
(~0.003 rad/step, Ma = 0.05 at fin tips). Given omega_so = n*m*B / C_rot,
this uniquely determines B_demo = omega_so_target * C_rot / (n*m) ~ 4.4 uT.
The dimensionless physics (Adler equation) is identical at any B — the
step-out transition at omega_field / omega_so = 1 is universal.

CONFINEMENT REFERENCE BASELINE
-------------------------------
Ratio 0.15 is used as the 'unconfined reference' in the T2.6b LBM sweep
(drag multiplier = 1.000 by construction), NOT a true infinite-domain
simulation. The physical unconfined step-out frequency (128 Hz for d2.8,
1 magnet) comes from de Boer et al. (2025). The LBM at ratio 0.15 still
has finite confinement effects — it is simply the least-confined point in
our sweep. This means our confinement shift predictions are slightly
conservative (the true shift relative to infinite domain may be slightly
larger than predicted).

SUBCYCLING
----------
RigidBodyNode is subcycled 10x via CouplingGroup subcycling — it takes
10 Euler sub-steps per LBM macro step with linearly interpolated drag
torque. 10x chosen conservatively: 4x would suffice for the truncation
error at these omega values, but 10x provides margin and has negligible
cost (RigidBody update is ~1000x cheaper than one LBM step). Validated:
0.000% torque difference vs non-subcycled at convergence (spot-check,
64^3, ratio 0.30, 100 steps).

Usage:
    python3 examples/fsi_stepout_demo.py           # --local (10k steps, ~13 min RTX 2060)
    python3 examples/fsi_stepout_demo.py --local   # explicit local mode
    python3 examples/fsi_stepout_demo.py --full    # 30k steps (~40 min RTX 2060, ~2.5 min H100)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

if "JAX_PLATFORMS" not in os.environ:
    os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")

import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (d2.8 UMR, from deboer2025_params.md + umr_ode.py)
# ---------------------------------------------------------------------------

VESSEL_DIAMETER_MM = 9.4
UMR_GEOM_MM = dict(
    body_radius=0.87, body_length=4.1, cone_length=1.9,
    cone_end_radius=0.255, fin_outer_radius=1.42,
    fin_length=2.03, fin_width=0.55, fin_thickness=0.15,
    helix_pitch=8.0,
)
N_MAG = 1
M_SINGLE = 1.07e-3       # A*m^2
FLUID_VISCOSITY = 0.69e-3 # Pa.s (water at 37C)
FLUID_DENSITY = 997.0     # kg/m^3
BODY_DENSITY = 1100.0     # kg/m^3
I_EFF = 1e-10             # kg*m^2 (added-mass estimate from umr_ode.py)
SEMI_MAJOR = 2.05e-3      # m (half body length)
SEMI_MINOR = 0.87e-3      # m (body radius)

# Step-out: omega_so = n*m*B / C_rot
F_STEP_UNCONFINED = 128.0  # Hz (de Boer et al.)
C_ROT_UNCONFINED = N_MAG * M_SINGLE * 3e-3 / (2 * math.pi * F_STEP_UNCONFINED)

RATIO = 0.30
SUBCYCLE_FACTOR = 10


def main():
    parser = argparse.ArgumentParser(description="FSI step-out demo")
    parser.add_argument("--local", action="store_true", default=True,
                        help="10k steps (default)")
    parser.add_argument("--full", action="store_true",
                        help="30k steps for full ramp")
    args = parser.parse_args()

    n_steps = 30000 if args.full else 10000

    # Load predicted step-out from validated JSON
    json_path = "docs/validation/umr_deboer2025/confined_fstep_validated.json"
    with open(json_path) as f:
        preds = json.load(f)
    pred = next(x for x in preds if x["n_mag"] == N_MAG and x["ratio"] == RATIO)
    mult_rot = pred["mult_rot"]
    shift_pct = pred["shift_pct"]
    C_rot_confined = C_ROT_UNCONFINED * mult_rot

    # LBM parameters at 64^3
    N = 64
    tau = 0.8
    cs = 1.0 / math.sqrt(3)
    dx_mm = VESSEL_DIAMETER_MM / N
    dx_physical = VESSEL_DIAMETER_MM * 1e-3 / N
    dt_physical = (tau - 0.5) / 3.0 * dx_physical**2 / (FLUID_VISCOSITY / FLUID_DENSITY)
    geom_lu = {k: v / dx_mm for k, v in UMR_GEOM_MM.items()}
    R_fin_lu = geom_lu["fin_outer_radius"]
    R_vessel_lu = geom_lu["body_radius"] / RATIO

    # Derive B_demo from the Ma constraint: omega_so must map to a lattice
    # omega that keeps Ma < 0.1 at fin tips. omega_so = n*m*B / C_rot, so
    # B = omega_so_target * C_rot / (n*m).
    omega_so_lu_target = 0.003  # Ma = 0.05 at fin tips
    omega_so_phys = omega_so_lu_target / dt_physical
    B_demo = omega_so_phys * C_rot_confined / (N_MAG * M_SINGLE)

    # Frequency ramp: 0.5x to 1.8x omega_so
    f_so = omega_so_phys / (2 * math.pi)
    ramp_ratio_start = 0.5
    ramp_ratio_end = 1.8
    f_start = f_so * ramp_ratio_start
    f_end = f_so * ramp_ratio_end

    # Ma check
    omega_end_lu = 2 * math.pi * f_end * dt_physical
    Ma_end = omega_end_lu * R_fin_lu * math.sqrt(3)
    assert Ma_end < 0.1, f"Ma at ramp end = {Ma_end:.4f} exceeds 0.1"

    omega_max_lattice = 0.05 * cs / R_fin_lu
    omega_max_physical = omega_max_lattice / dt_physical

    period_start = 1.0 / (f_start * dt_physical)

    print(f"[CONFIG] 64^3, ratio={RATIO}, ramp {ramp_ratio_start:.1f}x->{ramp_ratio_end:.1f}x omega_so")
    print(f"[CONFIG] Steps: {n_steps}")
    print(f"[CONFIG] B_demo = {B_demo*1e6:.2f} uT "
          f"(derived from Ma constraint: omega_so -> 0.003 rad/step)")
    print(f"[CONFIG] omega_so = {omega_so_phys:.4f} rad/s = {f_so:.4f} Hz")
    print(f"[CONFIG] Predicted confined f_step: {pred['f_step_confined_hz']} Hz "
          f"(shift {shift_pct}%)")
    print(f"[CONFIG] Ma at ramp end: {Ma_end:.4f}")
    print(f"[CONFIG] Estimated runtime: "
          f"~{n_steps * 0.08 / 60:.0f} min (RTX 2060) / "
          f"~{n_steps * 0.005 / 60:.1f} min (H100)")

    # --- Construct nodes ---
    from maddening.core.graph_manager import GraphManager
    from mime.nodes.environment.lbm.fluid_node import (
        IBLBMFluidNode, make_iblbm_rigid_body_edges,
    )
    from mime.nodes.robot.rigid_body import RigidBodyNode
    from mime.nodes.actuation.external_magnetic_field import ExternalMagneticFieldNode
    from mime.nodes.robot.permanent_magnet_response import PermanentMagnetResponseNode

    body_geometry_params = dict(nx=N, ny=N, nz=N, **geom_lu)
    lbm = IBLBMFluidNode(
        name="lbm_fluid", timestep=dt_physical,
        nx=N, ny=N, nz=N, tau=tau,
        vessel_radius_lu=R_vessel_lu,
        body_geometry_params=body_geometry_params,
        use_bouzidi=False, dx_physical=dx_physical,
    )

    rigid = RigidBodyNode(
        name="rigid_body", timestep=dt_physical / SUBCYCLE_FACTOR,
        semi_major_axis_m=SEMI_MAJOR,
        semi_minor_axis_m=SEMI_MINOR,
        density_kg_m3=BODY_DENSITY,
        fluid_viscosity_pa_s=FLUID_VISCOSITY,
        fluid_density_kg_m3=FLUID_DENSITY,
        use_analytical_drag=False,
        use_inertial=True,
        I_eff=I_EFF,
        omega_max=omega_max_physical,
    )

    field_node = ExternalMagneticFieldNode(
        name="ext_field", timestep=dt_physical,
    )

    magnet = PermanentMagnetResponseNode(
        name="magnet_response", timestep=dt_physical,
        n_magnets=N_MAG,
        m_single=M_SINGLE,
        moment_axis=(0.0, 1.0, 0.0),
    )

    # --- Construct graph ---
    gm = GraphManager()
    gm.add_node(field_node)
    gm.add_node(magnet)
    gm.add_node(rigid)
    gm.add_node(lbm)

    gm.add_edge("ext_field", "magnet_response", "field_vector", "field_vector")
    gm.add_edge("ext_field", "magnet_response", "field_gradient", "field_gradient")
    gm.add_edge("magnet_response", "rigid_body", "magnetic_torque",
                "magnetic_torque", additive=True)
    gm.add_edge("magnet_response", "rigid_body", "magnetic_force",
                "magnetic_force", additive=True)
    gm.add_edge("rigid_body", "magnet_response", "orientation", "orientation")

    for edge in make_iblbm_rigid_body_edges(
        "lbm_fluid", "rigid_body", dx_physical, dt_physical, FLUID_DENSITY,
    ):
        gm.add_edge(edge.source_node, edge.target_node,
                     edge.source_field, edge.target_field,
                     transform=edge.transform, additive=edge.additive)

    gm.add_external_input("ext_field", "frequency_hz", shape=())
    gm.add_external_input("ext_field", "field_strength_mt", shape=())

    gm.add_coupling_group(
        ["rigid_body", "lbm_fluid"],
        max_iterations=1,
        subcycling=True,
        boundary_interpolation="linear",
    )

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()

    # --- Run simulation ---
    B_demo_mt = float(B_demo * 1e3)
    # Unit conversion factors for torque: lattice -> N*m
    torque_factor = FLUID_DENSITY * dx_physical**5 / dt_physical**2

    print(f"\nRunning {n_steps} steps...")
    t0 = time.perf_counter()

    freq_history = []
    omega_body_history = []
    torque_history = []
    time_history = []

    for step in range(n_steps):
        # Linear ramp in dimensionless units
        ramp_frac = step / max(n_steps - 1, 1)
        f_current = f_start + (f_end - f_start) * ramp_frac
        freq_ratio = f_current / f_so  # dimensionless: omega_field / omega_so

        ext = {
            "ext_field": {
                "frequency_hz": jnp.float32(f_current),
                "field_strength_mt": jnp.float32(B_demo_mt),
            },
        }
        gm.step(external_inputs=ext)

        lbm_state = gm.get_node_state("lbm_fluid")
        rb_state = gm.get_node_state("rigid_body")

        omega_body = float(rb_state["angular_velocity"][2])
        omega_field = 2 * math.pi * f_current
        tz_lattice = float(lbm_state["drag_torque"][2])

        freq_history.append(freq_ratio)
        omega_body_hz = omega_body / (2 * math.pi)
        synchrony = omega_body / omega_field if abs(omega_field) > 1e-30 else 0.0
        omega_body_history.append(synchrony)
        torque_history.append(tz_lattice * torque_factor)
        time_history.append(step * dt_physical * 1000)  # ms

        if step % 2000 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  Step {step:>6d}/{n_steps}: "
                  f"f/f_so={freq_ratio:.3f}, "
                  f"sync={synchrony:.3f}, "
                  f"Tz={tz_lattice * torque_factor:.2e} N*m, "
                  f"{elapsed:.0f}s")

    elapsed_total = time.perf_counter() - t0
    print(f"\nCompleted in {elapsed_total:.1f}s ({elapsed_total/n_steps*1000:.1f} ms/step)")

    # --- Plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    freq_arr = np.array(freq_history)
    sync_arr = np.array(omega_body_history)
    torque_arr = np.array(torque_history)
    time_arr = np.array(time_history)

    # Rolling mean for smoothing
    window = min(200, n_steps // 10)
    if window > 1:
        kernel = np.ones(window) / window
        sync_smooth = np.convolve(sync_arr, kernel, mode="same")
        torque_smooth = np.convolve(torque_arr, kernel, mode="same")
    else:
        sync_smooth = sync_arr
        torque_smooth = torque_arr

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Panel 1: field frequency
    ax1.plot(time_arr, freq_arr, "b-", linewidth=1.5, label="Field frequency ramp")
    ax1.axhline(1.0, color="red", linestyle="--", linewidth=1,
                label=f"Predicted confined $\\omega_{{so}}$ "
                      f"(Bouzidi+FSI, ratio={RATIO})")
    f_unconfined_ratio = 1.0 / (1.0 - shift_pct / 100.0)
    ax1.axhline(f_unconfined_ratio, color="gray", linestyle=":", linewidth=1,
                label="Unconfined $\\omega_{so}$ "
                      "(de Boer et al., infinite domain)")
    ax1.set_ylabel("$\\omega_{field}$ / $\\omega_{so,confined}$")
    ax1.legend(loc="lower right", fontsize=8)  # Fix 4: moved from upper left
    ax1.set_ylim(0, 2.2)

    # Compute settling transient end: first index where sync stays below
    # 1.1 for 500 consecutive steps, capped at t=3000ms (Fix 3)
    settling_end = 0
    run_count = 0
    for i in range(len(sync_smooth)):
        if abs(sync_smooth[i]) < 1.1:
            run_count += 1
            if run_count >= 500:
                settling_end = i - 499
                break
        else:
            run_count = 0
    settling_time_ms = time_arr[settling_end] if settling_end > 0 else 0.0
    settling_time_ms = min(settling_time_ms, 3000.0)  # cap at 3000ms

    # Panel 2: synchrony ratio
    ax2.plot(time_arr, sync_arr, color="lightblue", alpha=0.3, linewidth=0.5)
    ax2.plot(time_arr, sync_smooth, "b-", linewidth=1.5,
             label="$\\Omega_{body}$ / $\\omega_{field}$")
    ax2.axhline(1.0, color="green", linestyle="--", linewidth=0.8,
                label="Perfect synchrony")
    ax2.axhline(0.5, color="orange", linestyle="--", linewidth=0.8,
                label="Step-out criterion (50%)")
    # Settling transient shading
    if settling_time_ms > 0:
        ax2.axvspan(time_arr[0], settling_time_ms, color="grey", alpha=0.1)
        ax2.text(settling_time_ms * 0.3, 1.55, "Settling\ntransient",
                 fontsize=7, color="grey", ha="center")
    # Vertical line where predicted step-out is crossed
    step_cross = None
    for i, fr in enumerate(freq_arr):
        if fr >= 1.0:
            step_cross = i
            break
    if step_cross is not None:
        ax2.axvline(time_arr[step_cross], color="red", linestyle="--",
                    linewidth=0.8, alpha=0.7)
        # Fix 2: rotated annotation offset to the right, white bbox
        ax2.annotate("Predicted step-out", xy=(time_arr[step_cross], 1.6),
                     xytext=(12, 0), textcoords="offset points",
                     fontsize=7, color="red", rotation=90, va="top",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                               edgecolor="red", alpha=0.8, linewidth=0.5))
    ax2.set_ylabel("$\\Omega_{body}$ / $\\omega_{field}$")
    ax2.set_ylim(-0.6, 1.7)  # Fix 1: expanded for transient headroom
    ax2.legend(loc="lower left", fontsize=8)

    # Find observed step-out (where smoothed sync drops below 0.5)
    observed_stepout_ratio = None
    for i in range(len(sync_smooth)):
        if freq_arr[i] > 0.8 and sync_smooth[i] < 0.5:
            observed_stepout_ratio = freq_arr[i]
            break

    # Summary text box (dimensionless, consistent with axes)
    pred_text = "Predicted step-out: 1.00x $\\omega_{so}$ (Bouzidi+FSI)"
    if observed_stepout_ratio is not None:
        obs_text = f"Observed step-out: ~{observed_stepout_ratio:.2f}x $\\omega_{{so}}$"
        agreement = abs(observed_stepout_ratio - 1.0) * 100
        agr_text = f"Agreement: {agreement:.0f}% from prediction"
    else:
        obs_text = "Observed step-out: not reached in this run"
        agr_text = "(increase steps with --full)"
    textstr = f"{pred_text}\n{obs_text}\n{agr_text}"
    # Fix 5: white background with grey border
    ax2.text(0.98, 0.02, textstr, transform=ax2.transAxes, fontsize=8,
             verticalalignment="bottom", horizontalalignment="right",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.85,
                       edgecolor="#cccccc", linewidth=0.8))

    # Panel 3: drag torque (rolling mean only — raw signal removed for clarity)
    ax3.plot(time_arr, torque_smooth, "r-", linewidth=1.5,
             label="Drag torque, rolling mean (N$\\cdot$m)")
    ax3.set_ylabel("Drag torque (N$\\cdot$m)")
    ax3.set_xlabel("Physical time (ms)")
    ax3.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "FSI-coupled step-out simulation\n"
        f"d2.8 UMR, {N_MAG} magnet, confinement ratio {RATIO} "
        f"(Newtonian fluid, simple BB, 64$^3$)",
        fontsize=12,
    )
    fig.text(0.5, 0.91,
             f"Shift (%) values are confinement-ratio-independent. "
             f"Absolute Hz shifts scale with magnet count.\n"
             f"$B_{{demo}}$ = {B_demo*1e6:.1f} $\\mu$T "
             f"(derived from Ma constraint: $\\omega_{{so}} \\to$ 0.003 rad/step — "
             f"see module docstring)",
             ha="center", fontsize=8, style="italic", color="gray")

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    out_path = "docs/validation/umr_deboer2025/figures/fsi_stepout_preliminary.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {out_path}")

    # Final summary
    final_sync = float(np.mean(sync_smooth[-500:])) if n_steps > 500 else float(sync_smooth[-1])
    print(f"Final synchrony ratio (last 500 steps): {final_sync:.3f}")
    if observed_stepout_ratio is not None:
        print(f"Step-out observed at {observed_stepout_ratio:.2f}x omega_so")
    else:
        print("Step-out not fully developed in this run (try --full)")


if __name__ == "__main__":
    main()
