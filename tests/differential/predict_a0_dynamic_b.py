"""The second declaration of A0 on its dynamic rungs, before 0.3 runs against it (the maintainer's option A of
2026-10-31, on the first declaration's finding and its review):

    JAX_PLATFORMS=cpu .venv/bin/python tests/differential/predict_a0_dynamic_b.py           # the emulation's record
    JAX_PLATFORMS=cpu .venv/bin/python tests/differential/predict_a0_dynamic_b.py declare   # the declaration from it

The first declaration (``a0_dynamic_expected_differences.json``, ``predict_a0_dynamic.py``) stays as it was committed.
Its gates on the rotation rates by phase, the motor's angle rate and the magnet's phase read the rotor's
single-precision angle, which 0.3 and the emulation reach the steady window a unit or two of its last place apart; at
10 Hz that unit is beyond their tolerances, and 7c1's rule, which set them, is blind to it (the reference's float32 and
mixed runs share the angle bit for bit, and the emulation reproduces the tag's). This declaration reads those
quantities against the angle of the same run (``a0_phases.py``), so that the angle cancels, and gates the absolute
phase where the angle's rounding does not reach it (the run's start and its rotor loop's integral).

It runs ``predict_a0_dynamic.emulate`` (unchanged but for a trace that changes nothing it computes) on the tag's scheme
and on 0.3's dynamic one, at every step, and reads each run as ``a0_phases`` reads the reference and 0.3. Then 0.3's
scheme once more with the arm's step in single precision, as 0.3 runs it, everything else as before (its
**single-precision twin**; the drag surrogate refuses single precision, and nothing else of 0.3's float32 is the
twin's): what the rounding of 0.3's own arm step can do to each quantity, which neither half of 7c1's rule can show (the
reference's arm is the tag's, which does not move under a load, and its pair of runs differ in the body's precision,
not the arm's). It reads only the reference, the drag surrogate's weights, the AR4 description and the rigid-body
library, as ``predict_a0_dynamic.py`` does; it imports no node and no controller of 0.3. Writes
``a0_dynamic_emulation_b.json`` and ``a0_dynamic_phases_b.npz`` (the predicted turn of the rotor frame against its
angle, sample by sample), and from them ``a0_dynamic_expected_differences_b.json``. Not collected by pytest."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import predict_a0_dynamic as D  # noqa: E402
from a0_phases import (EVERY, Phases, angle_offset_change, crossings_bound, midpoint_margin,  # noqa: E402
                       phase_against_angle, quantities, reference_phases)
from a0_runs import REFERENCE_FRAME, SCRIPT_A_FRAME, window  # noqa: E402

P = D.P
EMULATION = HERE / "a0_dynamic_emulation_b.json"
PHASES = HERE / "a0_dynamic_phases_b.npz"
OUTPUT = HERE / "a0_dynamic_expected_differences_b.json"
FIRST = HERE / "a0_dynamic_expected_differences.json"
FREQUENCIES = P.FREQUENCIES
#: The steps whose angles the motor's angle rate reads: the one before the window's first increment, and its last.
FIRST_STEP, LAST_STEP = 9999, 20000


def frame(scheme: D.Dynamic) -> Any:
    return REFERENCE_FRAME if scheme.base.body_moment == (1.0, 0.0, 0.0) else SCRIPT_A_FRAME


def emulated(scheme: D.Dynamic, frequency: float) -> tuple[Phases, dict[str, Any], np.ndarray, np.ndarray]:
    """The emulation on ``scheme`` read as ``a0_phases`` reads a run; what it shows of the arm's motion; the motor's
    angle and its increments at every step."""
    trace: dict[str, list[float]] = {}
    history, motion, poses = D.emulate(scheme, frequency, trace=trace)
    body = history.body
    integral = np.concatenate([[0.0], np.asarray(trace["integral"])])
    angle = np.asarray(history.rotor_angle)
    phases = Phases(np.asarray(body.time)[::EVERY], np.asarray(body.field)[::EVERY],
                    np.asarray(body.orientation)[::EVERY], np.asarray(body.position)[::EVERY], angle[::EVERY], poses,
                    integral[::EVERY], frame(scheme))
    return phases, motion, angle, np.asarray(trace["increment"])


def _transient(motion: dict[str, Any]) -> dict[str, float]:
    return {"joint_6_excursion_max_rad": motion["joint_6_excursion_max_rad"],
            "joint_6_speed_max_rad_s": motion["joint_speed_max_rad_s"][5]}


def single_precision_arm() -> Any:
    """``predict_a0_dynamic.dynamics``'s arm step with the tree, its inputs and every constant in single precision, as
    0.3's dynamic arm takes them in a float32 run; its outputs are single precision."""
    import jax
    import jax.numpy as jnp

    from mime.core.multibody import cholesky_solve, frame_angular_jacobians, frame_forces, frame_jacobian, \
        frame_pose, inverse_dynamics, mass_matrix, read_urdf

    single = jnp.float32
    tree = jax.tree_util.tree_map(lambda leaf: jnp.asarray(leaf, single) if hasattr(leaf, "dtype") and jnp.issubdtype(
        leaf.dtype, jnp.floating) else leaf, read_urdf(P.URDF).tree)
    flange = tree.articulation.frame_index("link_6")
    frames = len(tree.articulation.frame_names)
    gravity = jnp.asarray(D.GRAVITY_WORLD, single)

    def skew(v: Any) -> Any:
        zero = jnp.zeros((), single)
        return jnp.stack([jnp.stack([zero, -v[2], v[1]]), jnp.stack([v[2], zero, -v[0]]),
                          jnp.stack([-v[1], v[0], zero])])

    def step(q: Any, qd: Any, accelerations: Any, momentum: Any, wrench: Any, dt: float) -> Any:
        q, qd, accelerations, momentum, wrench = (jnp.asarray(v, single) for v in (q, qd, accelerations, momentum,
                                                                                   wrench))
        step_size = jnp.asarray(dt, single)
        inertia = mass_matrix(tree, q)
        across = frame_angular_jacobians(tree, q)[flange]
        half = jnp.asarray(0.5, single) * step_size * (across.T @ skew(momentum) @ across)
        turned = jnp.linalg.solve(inertia - half, (inertia + half) @ qd)
        command = inertia @ accelerations + inverse_dynamics(tree, q, qd, None, gravity)
        loads = frame_forces(tree, q, jnp.zeros((frames, 6), single).at[flange].set(wrench))
        bias = inverse_dynamics(tree, q, turned, None, gravity, loads)
        rates = cholesky_solve(jnp.linalg.cholesky(inertia), inertia @ turned + step_size * (command - bias))
        moved = q + step_size * rates
        return moved, rates, frame_pose(tree, moved, "link_6"), frame_jacobian(tree, moved, "link_6")[3:] @ rates

    return jax.jit(step, static_argnums=5)


def main() -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    record: dict[str, Any] = {"tag": {}, "new": {}, "twin": {}, "angle": {}, "transient": {}}
    saved: dict[str, np.ndarray] = {}
    runs: dict[str, dict[str, Any]] = {}
    for key, frequency in FREQUENCIES.items():
        print(f"{key}: the tag's scheme, and 0.3's dynamic ...", flush=True)
        tag, _, tag_angle, _ = emulated(D.TAG_D, frequency)
        new, motion, new_angle, increments = emulated(D.NEW_D, frequency)
        runs[key] = {"tag": tag, "new": new, "new_angle": new_angle}
        record["tag"][key], record["new"][key] = quantities(tag, frequency), quantities(new, frequency)
        inside = window(np.asarray(new.time), frequency)
        reference = reference_phases(f"f32_{key}")
        record["tag"][key]["magnet_phase_against_rotor_angle_rad"] = float(np.max(np.abs(phase_against_angle(
            tag.rotor, tag.angle, reference.rotor, reference.angle)[inside])))
        saved[f"{key}_predicted_turn"] = phase_against_angle(new.rotor, new.angle, tag.rotor, tag.angle)[inside]
        start, end = float(new_angle[FIRST_STEP]), float(new_angle[LAST_STEP])
        bound, crossed = crossings_bound(start, end)
        record["angle"][key] = {
            "steps": [FIRST_STEP, LAST_STEP], "angles": [start, end], "crossed": crossed, "bound_rad": bound,
            "turn_rad": end - start,
            "midpoint_margin_in_increment_units": midpoint_margin(new_angle[FIRST_STEP:LAST_STEP],
                                                                  increments[FIRST_STEP:LAST_STEP]),
            "the_tags": [float(tag_angle[FIRST_STEP]), float(tag_angle[LAST_STEP])]}
        record["transient"][key] = _transient(motion)
    held = list(D._dynamics)
    D._dynamics[:] = [single_precision_arm()]
    try:
        for key, frequency in FREQUENCIES.items():
            print(f"{key}: 0.3's dynamic scheme, its arm's step in single precision ...", flush=True)
            twin_run, motion, twin_angle, _ = emulated(D.NEW_D, frequency)
            new = runs[key]["new"]
            inside = window(np.asarray(new.time), frequency)
            record["twin"][key] = {
                **quantities(twin_run, frequency),
                "magnet_phase_against_rotor_angle_rad": float(np.max(np.abs(
                    phase_against_angle(twin_run.rotor, twin_run.angle, new.rotor, new.angle)[inside]))),
                "angle_offset_change_rad": angle_offset_change(twin_angle, runs[key]["new_angle"], FIRST_STEP,
                                                               LAST_STEP),
                "angles_apart_rad": [float(twin_angle[step] - runs[key]["new_angle"][step])
                                     for step in (FIRST_STEP, LAST_STEP)],
                "transient": _transient(motion)}
            assert _transient(motion) != record["transient"][key], "the twin ran the double-precision arm"
    finally:
        D._dynamics[:] = held
    EMULATION.write_text(json.dumps(record, indent=1) + "\n")
    np.savez_compressed(PHASES, **saved)
    print(f"wrote {EMULATION.name} and {PHASES.name}")


# -- the declaration -----------------------------------------------------------------------------------------------------

#: The first declaration's gates that never read the rotor's angle, carried with their predictions and tolerances.
CARRIED = ("swim_speed_axial_m_s", "swim_speed_axial_mean_velocity_m_s", "phase_lag_field_minus_moment_rad",
           "tracking_error_axial_mean_m", "tracking_error_axial_rms_m", "magnet_body_distance_minus_standoff_mean_m",
           "rotor_rate_mean_hz")
#: The first declaration's gates that read the angle, and what reads each here against the angle of the same run.
REPLACED = {"field_rotation_rate_hz": "field_turning_against_motor_angle_hz",
            "body_rotation_rate_hz": "body_turning_against_field_hz",
            "rotor_angle_rate_mean_hz": "motor_angle_offset_change_rad",
            "magnet_phase_rad": "magnet_phase_against_rotor_angle_rad"}
WHAT = {
    "field_turning_against_motor_angle_hz": "the slope over the window of the field's angle at the body about the "
        "tube's axis plus the motor's angle, in Hz: the field's turning against the motor's own output (the geometry "
        "of the dipole's turning field at the body; its periodic part leaves the fitted slope off zero)",
    "body_turning_against_field_hz": "the slope over the window of the body's moment's angle less the field's, in "
        "Hz: the body's turning against the field it is locked to",
    "field_phase_against_motor_angle_rad": "the circular mean over the window of the field's angle at the body plus "
        "the motor's angle: the field's phase at the body against where the rotor's angle puts the magnet (the "
        "reference's field_vector against its own angle)",
    "magnet_mount_phase_rad": "the circular mean over the window of the turn about the rotor's axis from the rotor "
        "frame's x to the moment of the point dipole at the rotor's centre whose field the body felt (inverted from "
        "the recorded field and the two positions at the same sample): the magnet's phase in its mount, which the "
        "first declaration's gates were all blind to; on 0.3's dynamic rungs the field reads the rotor a step late, "
        "a turn of the rotor's step, which the prediction carries",
    "rotor_loop_integral_rad": "the mean over the window of the rotor loop's integral of its speed error: the angle "
        "the loop holds the rotor behind its command (theta = theta_0 + Omega t - I in exact arithmetic), the "
        "absolute phase without the angle's rounding (the PI's steady integral is the bearing's torque over Ki, "
        "frozen in single precision once each step's addition falls below half its last place)",
}


def _rule(floor: float, resolution: float, twin: float) -> dict[str, float]:
    return {"value": max(10 * floor, 3 * resolution, 10 * twin), "rounding_floor_reference_pair": 10 * floor,
            "emulation_resolution": 3 * resolution, "rounding_floor_single_precision_arm": 10 * twin}


def declare() -> None:
    """Write ``a0_dynamic_expected_differences_b.json`` from the emulation's record and the reference."""
    from a0_phases import SCALARS, phase_against_angle_miss

    record = json.loads(EMULATION.read_text())
    first = json.loads(FIRST.read_text())
    turns = np.load(PHASES)
    findings = json.loads((HERE / "references" / "a0_dynamic_findings.json").read_text())
    reference = {key: reference_phases(f"f32_{key}") for key in FREQUENCIES}
    mixed = {key: reference_phases(f"mixed_f64_{key}") for key in FREQUENCIES}
    on_reference = {key: quantities(reference[key], f) for key, f in FREQUENCIES.items()}
    on_mixed = {key: quantities(mixed[key], f) for key, f in FREQUENCIES.items()}
    predicted = {key: {name: on_reference[key][name] + (record["new"][key][name] - record["tag"][key][name])
                       for name in SCALARS} for key in FREQUENCIES}
    tolerances: dict[str, Any] = {}
    for name in SCALARS:
        parts = [_rule(abs(on_reference[k][name] - on_mixed[k][name]),
                       abs(record["tag"][k][name] - on_reference[k][name]),
                       abs(record["twin"][k][name] - record["new"][k][name])) for k in FREQUENCIES]
        tolerances[name] = {field: max(part[field] for part in parts) for field in parts[0]}
        tolerances[name]["units"] = SCALARS[name]
    # the rotor frame's phase against its angle, sample by sample against the reference's
    zero = {key: np.zeros(int(np.count_nonzero(window(np.asarray(reference[key].time, dtype=np.float64), f))))
            for key, f in FREQUENCIES.items()}
    parts = [_rule(phase_against_angle_miss(mixed[key], reference[key], zero[key], f),
                   record["tag"][key]["magnet_phase_against_rotor_angle_rad"],
                   record["twin"][key]["magnet_phase_against_rotor_angle_rad"]) for key, f in FREQUENCIES.items()]
    against_angle: dict[str, Any] = {field: max(part[field] for part in parts) for field in parts[0]}
    against_angle["units"] = "rad"
    pose = first["magnet_pose"]["tolerances"]
    angle = record["angle"]
    twin = record["twin"]
    transient_gap = max(abs(twin[key]["transient"][name] / record["transient"][key][name] - 1)
                        for key in FREQUENCIES for name in record["transient"][key])
    transient_tolerance = 10 * transient_gap
    # the review's sensitivity of the kick to the rotor's J (its hand model, r1): the excursion moves 0.66 per unit
    j_response = 0.66
    integral_response = {key: -predicted[key]["rotor_loop_integral_rad"] for key in FREQUENCIES}
    ki_resolution = {key: tolerances["rotor_loop_integral_rad"]["value"] / abs(integral_response[key])
                     for key in FREQUENCIES}
    discriminating = [name for name in SCALARS
                      if abs(predicted["3hz"][name] - predicted["10hz"][name]) > tolerances[name]["value"]]
    first_10hz = findings["by_frequency"]["10hz"]["measured"]
    first_misses = {name: abs(first_10hz[name] - first["predicted"]["10hz"][name]) / abs(
        first["predicted"]["10hz"][name]) / first["tolerances"]["values"][name]
        for name in ("body_rotation_rate_hz", "field_rotation_rate_hz", "rotor_angle_rate_mean_hz")}
    first_misses["magnet_phase_rad"] = first_10hz["magnet_phase_rad"] / pose["values"]["magnet_phase_rad"]
    OUTPUT.write_text(json.dumps({
        "written": "2026-10-31, by predict_a0_dynamic_b.py declare, AFTER the first declaration's run of 0.3 against "
                   "it and its finding (references/a0_dynamic_findings.json) and the review of PR #77 were seen, and "
                   "BEFORE 0.3's A0 ran against this declaration. No quantity this declaration gates has been "
                   "computed on any run of 0.3: the first run's records hold the first declaration's quantities only. "
                   "The only run of 0.3 made for it was one of 10 ms, to read which fields its history holds (the "
                   "controller's integral among them), which read no quantity",
        "decided": "the maintainer's option A (2026-10-31): the angle-derived gates restructured, in a new and dated "
                   "declaration written before its run, so that the rotor's single-precision angle cancels; the "
                   "first declaration stays exactly as committed; the finding's allowance is not merged as a gate",
        "shaped_by_the_first_run": [
            "the finding: 0.3's and the emulation's single-precision angles reach the window a unit or two of its last "
            "place apart, and every gate that reads the angle moves by that unit, beyond its declared tolerance at "
            "10 Hz; so every such gate is read here against the angle of the same run",
            "the review's planted phase faults (its r3): an upstream offset of the angle at 5.2 phase tolerances, the "
            "magnet turned 1e-4 rad in its mount and a downstream rate drift all passed the first comparison's rule; "
            "so the controls below plant those three, at 3 and 10 Hz, and the absolute phase is gated where the "
            "angle's rounding does not reach it (the run's start and the rotor loop's integral)",
            "the review's sensitivity of the spin-up kick to the rotor's J (0.66 of the excursion per unit, its hand "
            "model): the transient tolerance must fail a 4 % error in J; its value below is the single-precision "
            "twin's, not fitted to the first run's misses (4.6e-6 to 3.7e-5)",
            "nothing else: every predicted value and every tolerance below is computed from the reference pair, the "
            "emulation and its single-precision twin, none from a run of 0.3"],
        "first_declaration": {
            "file": "a0_dynamic_expected_differences.json",
            "stays": "exactly as committed (9e7e5a8); its gates still run, those that never read the angle as gates "
                     "and those that do as reported quantities",
            "its_10hz_result_as_measured": {
                "units_apart_at_the_window_start_and_end": [
                    abs(findings["by_frequency"]["10hz"]["angles_apart_in_units"][end])
                    for end in ("window_start", "window_end")],
                "body_rotation_rate_hz": {"measured": first_10hz["body_rotation_rate_hz"],
                                          "miss_in_tolerances": first_misses["body_rotation_rate_hz"]},
                "field_rotation_rate_hz": {"measured": first_10hz["field_rotation_rate_hz"],
                                           "miss_in_tolerances": first_misses["field_rotation_rate_hz"]},
                "rotor_angle_rate_mean_hz": {"measured": first_10hz["rotor_angle_rate_mean_hz"],
                                             "miss_in_tolerances": first_misses["rotor_angle_rate_mean_hz"]},
                "magnet_phase_rad": {"measured": first_10hz["magnet_phase_rad"], "miss_in_tolerances": first_misses["magnet_phase_rad"]},
                "reading": "the body's and the field's rotation rates and the magnet's phase miss their declared "
                           "tolerances at 10 Hz; at 3 Hz every gate holds. Recorded as measured, not gated"},
            "the_allowance": "removed as a gate. It granted each angle-reading gate (start + end) times one unit's "
                             "measured move while the angle sat within two units of the emulation's, which at 10 Hz "
                             "widened the declared tolerances 3 to 20 times on every run; it counted its mechanism "
                             "twice (0.3's miss was one move, the rule granted two), it and its two-unit cap were "
                             "written after the 10 Hz miss was seen, and it let through an upstream phase fault of two "
                             "units, a magnet turned 1e-4 rad in its mount and a downstream drift of up to 10 to 29 "
                             "field-rate tolerances (the review of PR #77, B1, W1, W2)"},
        "how_predicted": "predict_a0_dynamic_b.py runs predict_a0_dynamic.emulate on the tag's scheme and on 0.3's "
                         "dynamic one and reads each, and the frozen reference, by a0_phases.py at every tenth step; "
                         "each scalar's prediction is the float32 reference's value moved by the emulation's 0.3 less "
                         "its tag (absolute, in Hz or rad)",
        "tolerance_rule": "7c1's rule, applied to quantities the angle no longer reaches, with one floor added: the "
                          "largest of ten times the reference's own float32-to-mixed gap (its body's rounding), three "
                          "times the emulation's miss of the reference on the tag's scheme (its resolution), and ten "
                          "times the gap between the emulation and its single-precision twin (the arm's step in single "
                          "precision, as 0.3 runs it: the rounding of the dynamic rungs' own arm, which the reference "
                          "cannot show), the larger of the two frequencies. Nothing in it looks at 0.3",
        "the_twin_reproduces_the_finding": {
            "3hz": twin["3hz"]["angles_apart_rad"], "10hz": twin["10hz"]["angles_apart_rad"],
            "reading": "the twin's angle against the double-precision arm's at steps 9999 and 20000, in rad: one unit "
                       "of the angle's last place at either end at 10 Hz (3.05e-5 and 6.10e-5), none at 3 Hz, as 0.3's "
                       "against the emulation's in the first run. The single-precision arm parts the angles in the "
                       "spin-up; computed on the emulation alone"},
        "carried": {"from": "the first declaration, unchanged: they never read the rotor's angle",
                    "predicted": {key: {name: first["predicted"][key][name] for name in CARRIED} for key in FREQUENCIES},
                    "tolerances": {name: first["tolerances"]["values"][name] for name in CARRIED},
                    "units": "relative to the predicted value, except the lag (rad)",
                    "magnet_pose": {name: pose["values"][name] for name in ("magnet_position_m",
                                                                             "magnet_spin_axis_rad")}},
        "replaced": REPLACED,
        "gates": {name: {"what": WHAT[name], "predicted": {key: predicted[key][name] for key in FREQUENCIES},
                         "on_the_reference": {key: on_reference[key][name] for key in FREQUENCIES},
                         "emulation_new_less_tag": {key: record["new"][key][name] - record["tag"][key][name]
                                                    for key in FREQUENCIES},
                         "tolerance": tolerances[name], "miss": "absolute: |0.3 - predicted| <= tolerance"}
                  for name in SCALARS},
        "magnet_phase_against_rotor_angle_rad": {
            "what": "at the reference's sampled steps in the window, the turn of the rotor frame's x about the "
                    "reference's axis from the reference's x, less the two angles' difference: how far the rotor "
                    "frame's phase is from where its own angle puts it (the flange's turn about the axis and the "
                    "motor's mount), against the reference's; the first declaration's magnet_phase_rad with the angle "
                    "taken out",
            "predicted": "a0_dynamic_phases_b.npz: the same turn of the emulation's 0.3 against its tag, sample by "
                         "sample in the window",
            "miss": "the largest over the window of |found - predicted|",
            "tolerance": against_angle},
        "motor_angle_offset_change_rad": {
            "what": "the motor's own output, its angle, against the emulation's at the same steps: how far the "
                    "offset between the two single-precision angles moves from step 9999 (before the window's first "
                    "increment) to step 20000 (its last), in exact arithmetic; the motor's angle rate's miss is it "
                    "over the angle's turn",
            "emulated_angles": {key: angle[key]["angles"] for key in FREQUENCIES},
            "tolerance": {key: angle[key]["bound_rad"] for key in FREQUENCIES},
            "derived": "two single-precision accumulators whose increments round alike keep their offset, but where "
                       "the angle crosses a power of two each is rounded onto a grid twice as coarse, and an odd "
                       "offset of the finer units moves by one of them: the bound is one unit of the last place "
                       "below each power of two crossed in the window (128 rad at 3 Hz, 512 rad at 10 Hz) and nothing "
                       "between. That the increments round alike rests on the emulation's least distance, over the "
                       "window, of each step's exact sum from a point where single precision rounds it the other "
                       "way: "
                       + ", ".join(f"{angle[key]['midpoint_margin_in_increment_units']:g} units of the increment's "
                                   f"last place at {key}" for key in FREQUENCIES)
                       + ", against increments that agree to within the rotor speed's tolerance (about one such "
                         "unit) and the swap test's 8 a step",
            "checked_on_the_twin": {key: twin[key]["angle_offset_change_rad"] for key in FREQUENCIES},
            "as_a_rate": {key: angle[key]["bound_rad"] / angle[key]["turn_rad"] for key in FREQUENCIES}},
        "motor_initial_angle_rad": {
            "what": "the motor's angle in the state at t = 0, against the reference's",
            "predicted": 0.0, "tolerance": 0.0,
            "basis": "both are the declared initial state, which no step has rounded: the absolute phase is gated "
                     "where it is set. From there the rotor's absolute phase is the command's integral less the loop's "
                     "integral (gated by rotor_loop_integral_rad and rotor_rate_mean_hz), and its rounding on top is "
                     "the angle's (gated by motor_angle_offset_change_rad)"},
        "transient": {
            "predicted": first["transient"]["predicted"], "tolerance": transient_tolerance,
            "units": "relative to the predicted value",
            "basis": "ten times the largest relative gap, over joint 6's largest excursion and speed at both "
                     "frequencies, between the emulation and its single-precision twin: the rounding a "
                     "single-precision arm step leaves in the kick, measured on the emulation (the first "
                     "declaration's ten times u times the mass matrix's condition, 2.8e-2, bounded the solve's "
                     "rounding in its worst direction, 760 to 6100 times what was measured, and passed a 4 % error "
                     "in the rotor's J)",
            "twin_gap": transient_gap,
            "a_4_percent_error_in_J": {"moves_the_excursion_by": 0.04 * j_response,
                                       "in_tolerances": 0.04 * j_response / transient_tolerance,
                                       "response_per_unit": j_response,
                                       "response_from": "the review of PR #77's hand model of the wrist (r1); the "
                                                        "controls measure it on 0.3"}},
        "controls": [
            {"plant": "the motor's angle offset at its start by 2^-14 rad (6.10e-5: two units of the angle's last "
                      "place at 10 Hz's window start, 5.2 of the first declaration's phase tolerances)",
             "at": ["3hz", "10hz"], "must_fail": ["motor_initial_angle_rad"],
             "built_not_to_see_it": "every phase gate read against the angle (a constant offset of the angle is "
                                    "carried by the field and the rotor alike), and the loop's integral"},
            {"plant": "the magnet turned 1e-4 rad in its mount, about the motor's axis", "at": ["3hz", "10hz"],
             "must_fail": ["magnet_mount_phase_rad", "field_phase_against_motor_angle_rad"]},
            {"plant": "a downstream drift: the field and the body turned by r t about the tube's axis on the "
                      "unplanted run's arrays, the motor's angle untouched, r = 2 pi x 3 field-turning tolerances",
             "at": ["3hz", "10hz"], "must_fail": ["field_turning_against_motor_angle_hz"]},
            {"plant": "the rotor's J 4 % large", "at": ["3hz", "10hz"], "must_fail": ["transient"]},
            {"plant": "the rotor loop's Ki scaled by 1 +- 3 resolutions of the integral gate (its resolution the "
                      "tolerance over the integral's response, -I per unit of Ki's relative change)",
             "at": ["3hz", "10hz"], "must_fail": ["rotor_loop_integral_rad"],
             "predicted_resolution": ki_resolution},
        ],
        "controls_rule": "each plant fails its gate on every run, over 20 runs (the stress workflow), at a size above "
                         "that gate's resolution measured on 0.3 (the gate's tolerance over its response measured at "
                         "a second size); a plant that does not is reported, not retuned",
        "frequency_control": {"held_to_the_other_frequencys_prediction_these_miss": discriminating,
                              "and_the_carried": "as the first declaration's: every carried gate but the distance"},
        "swap_test": "the first declaration's, unchanged",
        "kinematic_against_dynamic": {
            "restated": "the rung difference in the tracking error (+18.5 to +18.8 %, predicted 18.60 and 18.79 %) is "
                        "a lag: the dynamic rung's joint servo trails its ramp by Kd/Kp less about a step, 19.5 ms, "
                        "on top of the controller's own 100 ms that both rungs carry, so its share of the tracking "
                        "error is speed-invariant by construction, and the first declaration's crossing rule (a line "
                        "through zero of a relative difference against a relative threshold) could not cross even its "
                        "own prediction. Its rotation-rate rows are the single-precision accumulator's and the slope "
                        "estimator's, not a rung difference",
            "for_99": "the kinematic rung's validity line needs a lag (or bandwidth) condition (the servo's ramp lag "
                      "against the controller's) and an absolute tolerance on the end effector's pose (the magnet's "
                      "position error is v x 19.5 ms, which crosses an absolute tolerance at a speed); the watch on "
                      "commanded joint speeds and accelerations is a separate guard this data cannot earn (the "
                      "commands reach 0.92 % of the limit at most). Recorded as data for a1, not built",
            "the_wrist_kick": "a bare-wrist artefact: the AR4 description has no armature and no transmission, and "
                              "its motor links are massless, so the rotor's spin-up turns a wrist 4.1 times lighter "
                              "than the rotor about its axis (the review's two-degree-of-freedom hand model gives "
                              "0.5105 rad and 33.05 rad/s at 10 Hz, as 0.3 and the emulation do). #99 must not cite "
                              "joint 6's 2.1 to 21 times its limit as an arm's behaviour. No armature here: a "
                              "separate phase builds it"},
        "what_it_bounds": "that 0.3's A0 on its dynamic rungs is the reference plus exactly the declared differences "
                          "at these two operating points in float32, with every phase read against the angle of the "
                          "same run, the absolute phase at its start and in the loop's integral, and the angle's own "
                          "rounding to its derived bound",
        "what_it_does_not": "a phase fault that enters the motor's angle after the start and leaves the speed and the "
                            "loop's integral alone (a jump of the accumulator) is gated only by the angle's offset "
                            "change across the window, to one unit of its last place per power of two crossed; the "
                            "surrogate's physics (MIME-ANO-117); A0 with the magnet's mass",
        "if_a_quantity_misses": "it is a finding, reported with its size, and is not added to the declaration "
                                "afterwards",
    }, indent=1) + "\n")
    print(f"wrote {OUTPUT.name}")


if __name__ == "__main__":
    if sys.argv[1:] == ["declare"]:
        declare()
    else:
        main()
