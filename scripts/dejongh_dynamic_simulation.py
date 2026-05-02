#!/usr/bin/env python3
"""Dynamic 6DOF simulation of a de Jongh UMR under magnetic actuation.

Assembles the MADDENING graph:

    ExternalMagneticFieldNode → PermanentMagnetResponseNode
                                        ↓ magnetic torque/force
    GravityNode                         ↓
         ↓                              ↓
         └────────────→ MLPResistanceNode ← robot pose
                            ↓ velocity, angular_velocity
                       RigidBodyNode (kinematic mode)
                            ↓ position, orientation, velocity
                            └────↘ ┘ back to MLP/magnet

Usage:
    python scripts/dejongh_dynamic_simulation.py gate      # Run unit conversion gate
    python scripts/dejongh_dynamic_simulation.py smoke     # 5s sim, FL-9 and FL-3
    python scripts/dejongh_dynamic_simulation.py scenarioA FL-9  # Single scenario
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

# Force CPU + single-thread BLAS (avoid deadlocks in background processes)
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mime.nodes.environment.stokeslet.mlp_resistance_node import MLPResistanceNode
from mime.nodes.environment.stokeslet.lubrication_node import LubricationCorrectionNode
from mime.nodes.environment.gravity_node import GravityNode
from mime.nodes.robot.rigid_body import RigidBodyNode
from mime.nodes.robot.permanent_magnet_response import PermanentMagnetResponseNode
from mime.nodes.actuation.external_magnetic_field import ExternalMagneticFieldNode
from maddening.core.graph_manager import GraphManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "dejongh_benchmark"
DIAG_DIR = DATA_DIR / "diagnostics"
REC_DIR = DATA_DIR / "recordings"
DIAG_DIR.mkdir(parents=True, exist_ok=True)
REC_DIR.mkdir(parents=True, exist_ok=True)

MLP_WEIGHTS = DATA_DIR / "mlp_cholesky_weights_v2.npz"

# FL group from de Jongh
FL_PARAMS = {
    "FL-3": {"nu": 1.0, "L_UMR_mm": 7.47},
    "FL-5": {"nu": 1.4, "L_UMR_mm": 7.47},
    "FL-7": {"nu": 1.8, "L_UMR_mm": 7.47},
    "FL-9": {"nu": 2.33, "L_UMR_mm": 7.47},
}

# Vessel radii in mm
VESSELS = {
    '1/2"':  6.35,
    '3/8"':  4.765,
    '1/4"':  3.175,
    '3/16"': 2.38,
}

# UMR volume estimate (FL, fixed length)
# π × R_cyl² × L for the cylinder body + modulation correction
# Using (1 + ε²/2)π R_cyl² L for average modulated radius
R_CYL_UMR_MM = 1.56
EPSILON_MOD = 0.33
UMR_VOLUME_M3_FL = np.pi * (R_CYL_UMR_MM * 1e-3)**2 * (7.47e-3) * (1 + EPSILON_MOD**2 / 2)
# ≈ 6.15e-8 m³

# Magnet properties (from de Jongh Table S2 / Sec VI.H)
N_MAGNETS = 2
M_SINGLE = 8.4e-4  # A·m² per N45 1mm³ cube (Supermagnete data: Mx = 0.84 Am² per magnet)


def unit_conversion_gate(R_ves_mm=3.175):
    """Critical validation: R_nd→R_SI conversion must give correct swim speed.

    FL-9 at centered position in 1/4" vessel, zero external force, magnetic torque
    only → expect U_z ≈ 3.15 mm/s (from overnight BEM sweep).
    """
    from mime.nodes.environment.stokeslet.mlp_resistance_node import MLPResistanceNode

    log.info("=" * 70)
    log.info("UNIT CONVERSION GATE (Step 2 validation)")
    log.info("=" * 70)

    if not MLP_WEIGHTS.exists():
        log.error("MLP weights not found: %s", MLP_WEIGHTS)
        return False

    # Physical magnetic torque: T_max = |m × B|_max = |m||B| at 90°
    B_field_mT = 1.2
    m_moment = N_MAGNETS * M_SINGLE  # 2 × 8.4e-4
    T_mag_max = m_moment * (B_field_mT * 1e-3)  # 2.016e-6 N·m
    log.info("B = %.2f mT, m = %.3e A·m², T_max = %.3e N·m",
             B_field_mT, m_moment, T_mag_max)

    node = MLPResistanceNode(
        name="test_mlp", timestep=1e-3,
        mlp_weights_path=str(MLP_WEIGHTS),
        nu=2.33, L_UMR_mm=7.47,
        R_cyl_UMR_mm=R_CYL_UMR_MM,
        R_ves_mm=R_ves_mm,
        mu_Pa_s=1e-3,  # water
    )

    state = node.initial_state()
    pose_pos = jnp.zeros(3)
    pose_ori = jnp.array([1.0, 0.0, 0.0, 0.0])

    # New interface: MLP takes (pos, orient, U, Ω, u_bg) → outputs (drag_F, drag_T, R).
    # To validate, apply zero velocity/Ω, extract R, then compute swim speed analytically
    # and check against BEM baseline.
    result = node.update(state, {
        "robot_position": pose_pos,
        "robot_orientation": pose_ori,
        "body_velocity": jnp.zeros(3),
        "body_angular_velocity": jnp.zeros(3),
        "background_velocity": jnp.zeros(3),
    }, dt=1e-3)
    R_SI = np.array(result["resistance_matrix"])

    # At Ω_z = 62.83 rad/s (10 Hz), the force-free swim speed from R is:
    # U_free = -R_FU⁻¹ @ R_FΩ @ [0,0,ω_z]
    omega_target = 2 * np.pi * 10.0
    R_FU = R_SI[:3, :3]
    R_FW = R_SI[:3, 3:]
    U_free = -np.linalg.solve(R_FU, R_FW @ np.array([0, 0, omega_target]))

    # Also report T required for this Ω (force-free torque balance)
    R_TU = R_SI[3:, :3]
    R_TW = R_SI[3:, 3:]
    T_ext_z = R_TU[2, :] @ U_free + R_TW[2, 2] * omega_target
    log.info("Analytical swim: U = %s m/s, T_required = %.3e N·m",
             np.array2string(U_free, precision=4), T_ext_z)

    U = U_free
    Omega = np.array([0.0, 0.0, omega_target])

    # Expected: v_z ≈ 3.15 mm/s for FL-9 at 1/4" centered at Ω_z = 62.83 rad/s
    U_expected_mm_s = 3.15
    U_actual_mm_s = float(U[2]) * 1e3  # m/s → mm/s

    eigs = np.linalg.eigvalsh(R_SI)
    spd = np.all(eigs > 0)

    log.info("Result (at Ω_z = 62.83 rad/s):")
    log.info("  U = [%.3e, %.3e, %.3e] m/s", U[0], U[1], U[2])
    log.info("  Ω = [%.3e, %.3e, %.3e] rad/s", Omega[0], Omega[1], Omega[2])
    log.info("  U_z actual: %.4f mm/s", U_actual_mm_s)
    log.info("  U_z expected: %.4f mm/s (from BEM overnight sweep)", U_expected_mm_s)

    rel_err = abs(U_actual_mm_s - U_expected_mm_s) / U_expected_mm_s
    log.info("  Relative error: %.2f%%", rel_err * 100)
    log.info("  R_SI diagonal: %s", np.array2string(np.diag(R_SI), precision=3))
    log.info("  R_SI min eigenvalue: %.3e (SPD: %s)", eigs.min(), spd)

    # Log diagnostic for autonomous error handling
    diag = {
        "B_mT": B_field_mT, "T_mag_N_m": float(T_ext_z),
        "omega_target_rad_s": float(omega_target),
        "U_actual_mm_s": U_actual_mm_s, "U_expected_mm_s": U_expected_mm_s,
        "relative_error": float(rel_err),
        "R_SI_diagonal": np.diag(R_SI).tolist(),
        "R_SI_min_eig": float(eigs.min()),
        "SPD": bool(spd),
        "U_m_s": U.tolist(),
        "Omega_rad_s": Omega.tolist(),
    }
    with open(DIAG_DIR / "unit_gate.json", "w") as f:
        json.dump(diag, f, indent=2)

    if rel_err > 0.16:  # 2 × 8% MLP test error
        log.error("UNIT CONVERSION GATE FAILED: error %.1f%% > 16%%", rel_err * 100)
        log.error("Most likely cause: wrong exponent in R_nd→R_SI scaling")
        log.error("Check MLPResistanceNode._R_nd_to_SI function")
        return False

    log.info("✓ Gate PASSED (error %.1f%% within 16%% bound)", rel_err * 100)
    return True


from mime.experiments.dejongh import build_graph_with_meta as build_graph  # noqa: E402


def run_smoke_test(design="FL-9", vessel='1/4"', duration_s=5.0,
                    dt=5e-4, B_mT=1.2, freq_hz=10.0,
                    y_init_mm=-0.5,
                    pulsatile_U_mean_mm_s=0.0,
                    pulsatile_freq_hz=1.2,
                    pulsatile_amplitude=0.6,
                    tag=""):
    """Run the dynamic simulation and save trajectory JSON."""
    log.info("=" * 70)
    log.info("SMOKE TEST: %s × %s, %.1f s at dt=%.2e",
             design, vessel, duration_s, dt)
    log.info("=" * 70)

    gm, meta = build_graph(
        design_name=design, vessel_name=vessel,
        B_strength_mT=B_mT, field_freq_hz=freq_hz, dt=dt,
        pulsatile_U_mean_mm_s=pulsatile_U_mean_mm_s,
        pulsatile_freq_hz=pulsatile_freq_hz,
        pulsatile_amplitude=pulsatile_amplitude,
    )

    gm.compile()

    # Initial state: pre-sunk to skip the ~3 ms gravity transient.
    # Mutate the graph's internal state dict (gm._state) directly.
    gm._state["body"] = dict(gm._state["body"])
    gm._state["body"]["position"] = jnp.array([0.0, y_init_mm * 1e-3, 0.0])

    n_steps = int(duration_s / dt)
    # 20 fps recording target: 1 frame per 50 ms = 100 steps at dt=0.5ms
    target_fps = 20
    log_interval = max(1, int((1.0 / target_fps) / dt))

    trajectory = []
    warnings_list = []
    clamp_count = 0
    consecutive_clamp = 0

    t0 = time.time()
    t_sim = 0.0

    # Pulsatile flow declaration for MLP external input
    has_pulsatile = pulsatile_U_mean_mm_s > 0
    if has_pulsatile:
        gm.add_external_input("mlp_drag", "background_velocity",
                              shape=(3,), dtype=jnp.float32)
        # Need to recompile after adding external input
        gm.compile()
        # Re-set initial position
        gm._state["body"] = dict(gm._state["body"])
        gm._state["body"]["position"] = jnp.array([0.0, y_init_mm * 1e-3, 0.0])

    base_external = {
        "field": {
            "frequency_hz": jnp.float32(freq_hz),
            "field_strength_mt": jnp.float32(B_mT),
        },
    }
    U_mean_m_s = pulsatile_U_mean_mm_s * 1e-3  # mm/s → m/s

    nan_step = None

    for step in range(n_steps):
        external_inputs = dict(base_external)
        if has_pulsatile:
            # Pulsatile parabolic flow at robot's current radial position.
            # Use previous step's body position (feedback loop).
            body_pos = gm._state["body"]["position"]
            r_radial = float(jnp.sqrt(body_pos[0]**2 + body_pos[1]**2))
            r_ves_m = meta["R_ves_mm"] * 1e-3
            profile = max(1.0 - (r_radial / r_ves_m) ** 2, 0.0)
            modulation = 1.0 + pulsatile_amplitude * np.sin(
                2 * np.pi * pulsatile_freq_hz * t_sim)
            # Flow in +z direction (body swims in +z, flow pushes it back)
            u_z = 2.0 * U_mean_m_s * profile * modulation
            external_inputs["mlp_drag"] = {
                "background_velocity": jnp.array([0.0, 0.0, u_z], dtype=jnp.float32),
            }

        try:
            state = gm.step(external_inputs=external_inputs)
        except Exception as e:
            log.error("Step %d raised: %s", step, e)
            nan_step = step
            break

        # Check for NaN/inf
        body = state["body"]
        pos = np.array(body["position"])
        if not np.all(np.isfinite(pos)):
            log.error("NaN/inf at step %d, pos=%s", step, pos)
            nan_step = step
            break

        # Track clamp firing
        mlp_state = state["mlp_drag"]
        clamp = float(np.array(mlp_state["clamp_fired"]))
        if clamp > 0.5:
            clamp_count += 1
            consecutive_clamp += 1
        else:
            consecutive_clamp = 0

        if consecutive_clamp == 100:
            warnings_list.append({
                "step": step, "reason": "clamp_fired_100_consecutive",
                "position_mm": (pos * 1e3).tolist(),
            })
            log.warning("Clamp fired 100 consecutive steps at step %d — gravity dominates wall drag", step)

        # Record trajectory
        if step % log_interval == 0 or step == n_steps - 1:
            q = np.array(body["orientation"])
            v = np.array(body["velocity"])
            w = np.array(body["angular_velocity"])
            # Diagnostics: field, magnet torque, MLP state
            field_state = state.get("field", {})
            magnet_state = state.get("magnet", {})
            mlp_state = state.get("mlp_drag", {})
            B_vec = np.array(field_state.get("field_vector", [0, 0, 0]))
            m_torque = np.array(magnet_state.get("magnetic_torque", [0, 0, 0]))
            mlp_drag_F = np.array(mlp_state.get("drag_force", [0, 0, 0]))
            mlp_drag_T = np.array(mlp_state.get("drag_torque", [0, 0, 0]))
            trajectory.append({
                "step": step, "t_s": t_sim,
                "position_mm": (pos * 1e3).tolist(),
                "orientation": q.tolist(),
                "velocity_mm_s": (v * 1e3).tolist(),
                "angular_velocity_rad_s": w.tolist(),
                "field_vector_mT": (B_vec * 1e3).tolist(),
                "magnetic_torque_N_m": m_torque.tolist(),
                "clamp_fired": int(clamp > 0.5),
            })
            log.info("  step %6d (t=%.3fs): pos=[%.3f,%.3f,%.3f]mm, v_z=%.3f mm/s, ω_z=%.2f rad/s, T_mag=%.2e, T_drag=%.2e",
                     step, t_sim, *(pos * 1e3),
                     v[2] * 1e3, w[2], m_torque[2], mlp_drag_T[2])

        t_sim += dt

    wall_time = time.time() - t0
    log.info("Simulation done: %d steps in %.1f s (%.2f ms/step), sim time = %.3f s",
             step + 1, wall_time, 1e3 * wall_time / (step + 1), t_sim)

    # Summary
    final_pos = np.array(state["body"]["position"])
    final_vel = np.array(state["body"]["velocity"])
    summary = {
        "design": design,
        "vessel": vessel,
        "duration_s": t_sim,
        "n_steps": step + 1,
        "dt": dt,
        "wall_time_s": wall_time,
        "nan_step": nan_step,
        "clamp_fire_count": clamp_count,
        "clamp_fire_fraction": clamp_count / max(1, step + 1),
        "final_position_mm": (final_pos * 1e3).tolist(),
        "final_velocity_mm_s": (final_vel * 1e3).tolist(),
        "equilibrium_v_z_mm_s": float(np.mean([t["velocity_mm_s"][2]
                                                for t in trajectory[-5:]])),
        "warnings": warnings_list,
    }
    name_suffix = tag if tag else "smoke"
    vessel_safe = vessel.replace(chr(34), '').replace('/', '_')
    summary_path = REC_DIR / f"{name_suffix}_{design}_{vessel_safe}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    traj_path = REC_DIR / f"{name_suffix}_{design}_{vessel_safe}_trajectory.json"
    with open(traj_path, "w") as f:
        json.dump({"meta": meta, "trajectory": trajectory, "summary": summary}, f, indent=2)

    log.info("Summary saved: %s", summary_path)
    log.info("Trajectory saved: %s", traj_path)
    log.info("Equilibrium v_z: %.3f mm/s", summary["equilibrium_v_z_mm_s"])

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["gate", "smoke", "scenarioA", "scenarioB", "all"])
    parser.add_argument("design", nargs="?", default="FL-9")
    parser.add_argument("--vessel", default='1/4"')
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=5e-4)
    args = parser.parse_args()

    if args.mode == "gate":
        ok = unit_conversion_gate(R_ves_mm=VESSELS[args.vessel])
        sys.exit(0 if ok else 1)

    # All other modes require gate to pass first
    if not unit_conversion_gate(R_ves_mm=VESSELS[args.vessel]):
        log.error("Gate failed. Check diagnostics before running simulation.")
        # Try fallback: swap a² ↔ a³ in off-diagonal blocks
        log.warning("Attempting fallback: this requires manual code edit. Aborting.")
        sys.exit(1)

    if args.mode == "smoke":
        run_smoke_test(design=args.design, vessel=args.vessel,
                       duration_s=args.duration, dt=args.dt, tag="smoke")
    elif args.mode == "scenarioA":
        run_smoke_test(design="FL-9", vessel='1/4"',
                       duration_s=args.duration, dt=args.dt, tag="scenarioA")
        run_smoke_test(design="FL-3", vessel='1/4"',
                       duration_s=args.duration, dt=args.dt, tag="scenarioA")
    elif args.mode == "scenarioB":
        # Pulsatile flow: mean velocity ~3 mm/s (order of swim speed), 1.2 Hz cardiac
        run_smoke_test(design="FL-9", vessel='1/4"',
                       duration_s=args.duration, dt=args.dt,
                       pulsatile_U_mean_mm_s=3.0,
                       pulsatile_freq_hz=1.2,
                       pulsatile_amplitude=0.6,
                       tag="scenarioB")
    elif args.mode == "scenarioC":
        log.warning("Scenario C (vessel transition) requires z-varying R_ves — not implemented. Skipping.")
    elif args.mode == "all":
        for design in ["FL-9", "FL-3"]:
            run_smoke_test(design=design, vessel='1/4"',
                           duration_s=args.duration, dt=args.dt)


if __name__ == "__main__":
    main()
