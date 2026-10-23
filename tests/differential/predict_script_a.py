"""Predict how script A in 0.3 differs from its frozen 0.2 reference, before 0.3 runs it:

    .venv/bin/python tests/differential/predict_script_a.py

writes ``script_a_expected_differences.json`` beside this file. It reads only the reference (``references/script_a/``:
its parameters, its steady quantities, and its trajectory at every tenth step) and the drag surrogate's weights, which
both lines share by sha256. It imports nothing of 0.3 but the surrogate's network and the steady-quantity rule. Not
collected by pytest.

**How.** It integrates script A twice in NumPy, float64, in the reference's frame, from the update rules each line
documents, with the same drag model:

- **0.2's scheme**, read from the port (MIME-port041 at 6485516) and its graph's node order (field, magnet, gravity,
  mlp_drag, body, lub): the torque reads the body's orientation at the start of the step; the body steps explicitly
  (symplectic Euler) under the drag the lubrication node computed in the step before, ``-R [v; omega]`` from that
  step's velocity, with the surrogate evaluated at the position before that; the vessel constraint projects the body
  back and removes ``(R / r)^2`` of its outward velocity (MIME-ANO-111); the field's frequency and amplitude arrive as
  float32 external inputs, so ``2 pi f`` and ``|B|`` carry float32's rounding under x64.
- **0.3's scheme**, read from ``RigidInertialNode``'s documented step: ``(M/dt + R) V' = (M/dt) V + F`` with the
  resistance at the pose at the start of the step, then ``x' = x + dt v'``, the projection, and all of the outward
  velocity removed; commands in the run's precision.

Run with 0.2's scheme, the emulation reproduces the reference's steady quantities and trajectory, which is what makes
its 0.3 run a prediction. Each change is then made alone, to say what each one does, and together; at halved steps,
to show both schemes meet as ``dt -> 0``; and with the body's contact held inside the implicit solve, the control on
the mechanism the prediction names.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).parent
REFERENCES = HERE / "references" / "script_a"
OUTPUT = HERE / "script_a_expected_differences.json"
sys.path[:0] = [str(HERE), str(HERE.parent)]

from script_a_steady import (  # noqa: E402
    GATED,
    REFERENCE_FRAME,
    REPORTED,
    History,
    speed_between,
    steady,
)


def _read(name: str) -> dict[str, Any]:
    return json.loads((REFERENCES / name).read_text())


PROVENANCE = _read("f64_provenance.json")
PARAMETERS = PROVENANCE["parameters"]
DERIVED = PARAMETERS["derived"]
NODES = PROVENANCE["nodes"]

MASS = DERIVED["m_eff_kg"]
INERTIA = DERIVED["I_eff_kg_m2"]
GRAVITY_FORCE = np.array(NODES["gravity"]["params"]["direction"]) * (
    NODES["gravity"]["params"]["delta_rho_kg_m3"] * NODES["gravity"]["params"]["volume_m3"]
    * NODES["gravity"]["params"]["g"])
MOMENT = DERIVED["N_MAGNETS"] * DERIVED["M_SINGLE_Am2"]
FIELD = PARAMETERS["field_strength_mT"] * 1e-3
VISCOSITY = PARAMETERS["mu_Pa_s"]
NU, LENGTH = DERIVED["FL_PARAMS"]["nu"], DERIVED["FL_PARAMS"]["L_UMR_mm"] * 1e-3
RADIUS = DERIVED["R_CYL_UMR_MM"] * 1e-3
OUTER = DERIVED["R_MAX_BODY_FACTOR"]
TUBE = DERIVED["R_ves_mm"] * 1e-3
EPSILON = PARAMETERS["lubrication_epsilon_mm"] * 1e-3
MARGIN = 1e-4  # the 0.2 builder's 0.1 mm safety margin (dejongh.build_graph), in the constraint radius below
EXTENT = RADIUS * OUTER
CONSTRAINT_0_2 = DERIVED["constraint_radius_m"]
START = np.array(PARAMETERS["init_position_m"])
DT = PARAMETERS["dt"]
DURATION = PROVENANCE["duration_s"]
GAP_FLOOR = NODES["lub"]["params"]["delta_floor_mm"] * 1e-3
FEATURE_GAP_FLOOR, CLAMP_FRACTION = 1e-3, 0.95  # the 0.2 surrogate's, unchanged in 0.3
AXIS = np.array([0.0, 0.0, 1.0])
MOMENT_IN_BODY = np.array(DERIVED["body_moment_axis_in_body"])


def _network() -> Callable[[np.ndarray], np.ndarray]:
    """The surrogate's network in float64, which needs JAX's x64 mode on (``main`` turns it on; the tests run in
    it)."""
    import jax.numpy as jnp

    from mime.data import asset, asset_bytes
    from mime.surrogates import predict_resistance, read_cholesky_mlp

    network = read_cholesky_mlp(asset_bytes(asset("dejongh_mlp_v2")))

    def predict(features: np.ndarray) -> np.ndarray:
        found = np.asarray(predict_resistance(network, jnp.asarray(features, dtype=jnp.float64)))
        if found.dtype != np.float64:
            raise RuntimeError("the surrogate ran in float32: turn JAX's x64 mode on first")
        return found

    return predict


_predict: list[Callable[[np.ndarray], np.ndarray]] = []
_cache: dict[float, np.ndarray] = {}


def surrogate(offset: float) -> np.ndarray:
    """The surrogate's resistance (SI) in the canonical frame (columns: the offset's direction ``n``, ``a x n``, the
    tube's axis ``a``), at a distance ``offset`` from the axis: the 0.2 node's features and clamp."""
    if offset not in _cache:
        tube, kappa = TUBE / RADIUS, RADIUS / TUBE
        largest = max((tube - OUTER) * CLAMP_FRACTION, 0.0)
        clamped = min(offset / RADIUS, largest)
        gap = max(tube - (clamped + OUTER), FEATURE_GAP_FLOOR)
        features = np.array([NU, LENGTH / RADIUS, kappa, clamped, 0.0, math.log(gap), NU ** 2, kappa ** 2])
        scale = np.kron(np.array([[VISCOSITY * RADIUS, VISCOSITY * RADIUS ** 2],
                                  [VISCOSITY * RADIUS ** 2, VISCOSITY * RADIUS ** 3]]), np.ones((3, 3)))
        if not _predict:
            _predict.append(_network())
        _cache[offset] = _predict[0](features) * scale
    return _cache[offset]


def lubrication(offset: float) -> np.ndarray:
    """The 0.2 lubrication node's correction ``w R_lub`` (SI) in the canonical frame."""
    a = EXTENT
    gap = max(TUBE - (offset + a), GAP_FLOOR)
    weight = math.exp(-gap / EPSILON)
    logarithm = max(math.log(a / gap), 0.0)
    normal = 6 * math.pi * VISCOSITY * a ** 2 / gap
    tangential = 6 * math.pi * VISCOSITY * a * (8 / 15) * logarithm
    rotational = 8 * math.pi * VISCOSITY * a ** 3 * (2 / 5) * logarithm
    return weight * np.diag([normal, tangential, tangential, 0.0, rotational, rotational])


def _turn(position: np.ndarray) -> np.ndarray:
    radial = position - (position @ AXIS) * AXIS
    normal = radial / np.linalg.norm(radial)
    frame = np.stack([normal, np.cross(AXIS, normal), AXIS], axis=1)
    pair = np.zeros((6, 6))
    pair[:3, :3] = pair[3:, 3:] = frame
    return pair


def _offset(position: np.ndarray) -> float:
    radial = position - (position @ AXIS) * AXIS
    return float(np.linalg.norm(radial))


def in_world(canonical: np.ndarray, position: np.ndarray) -> np.ndarray:
    pair = _turn(position)
    return pair @ canonical @ pair.T


def _rotation(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _multiply(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = p
    w2, x2, y2, z2 = q
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _exponential(rotation: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotation))
    if angle == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.concatenate([[math.cos(angle / 2)], math.sin(angle / 2) * rotation / angle])


def _project_0_2(position: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``CylindricalVesselConstraint.apply``: ``r_hat`` is the projected position over the radius before projection,
    so ``(R / r)^2`` of the outward velocity is removed (MIME-ANO-111)."""
    r = math.sqrt(position[0] ** 2 + position[1] ** 2 + 1e-30)
    if r <= CONSTRAINT_0_2:
        return position, velocity
    projected = position.copy()
    projected[:2] *= CONSTRAINT_0_2 / r
    r_hat = np.array([projected[0] / r, projected[1] / r, 0.0])
    return projected, velocity - max(float(velocity @ r_hat), 0.0) * r_hat


def _project_0_3(position: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``RigidInertialNode._confined`` with the cylinder's distance and ``remove_outward``."""
    radial = position - (position @ AXIS) * AXIS
    r = float(np.linalg.norm(radial))
    beyond = (r - TUBE) + EXTENT + MARGIN
    if beyond <= 0.0:
        return position, velocity
    normal = radial / r
    return position - beyond * normal, velocity - max(float(velocity @ normal), 0.0) * normal


@dataclass(frozen=True)
class Scheme:
    """``integrator``: ``explicit`` (0.2's), ``implicit`` (0.3's), or ``implicit, contact held`` (the mechanism's
    control: in contact, the solve holds the velocity along the wall's normal at zero). ``projection``: ``0.2`` or
    ``0.3``. ``commands``: ``float32`` (the reference's external inputs) or ``run`` (0.3's drives)."""

    integrator: str
    projection: str
    commands: str


SCHEME_0_2 = Scheme("explicit", "0.2", "float32")
SCHEME_0_3 = Scheme("implicit", "0.3", "run")
CONTACT_HELD = Scheme("implicit, contact held", "0.3", "run")


@dataclass(frozen=True)
class Emulated:
    history: History
    outward_before_projection_m_s: float  # the mean outward velocity the step left for the projection, over the window


def _commands(scheme: Scheme, frequency: float) -> tuple[float, float]:
    if scheme.commands == "float32":
        omega = float(np.float32(np.float32(2 * math.pi) * np.float32(frequency)))
        amplitude = float(np.float32(np.float32(PARAMETERS["field_strength_mT"]) * np.float32(1e-3)))
        return omega, amplitude
    return 2 * math.pi * frequency, FIELD


def emulate(scheme: Scheme, *, frequency: float = 10.0, dt: float = DT, duration: float = DURATION) -> Emulated:
    omega, amplitude = _commands(scheme, frequency)
    project = _project_0_2 if scheme.projection == "0.2" else _project_0_3
    steps = round(duration / dt)
    mass_matrix = np.diag([MASS] * 3 + [INERTIA] * 3)
    position, orientation = START.copy(), np.array([1.0, 0.0, 0.0, 0.0])
    velocity, spin = np.zeros(3), np.zeros(3)
    drag = np.zeros(6)  # the lubrication node's initial drag, which 0.2's body reads at its first step
    rows = np.zeros((steps + 1, 16))
    rows[0, :3], rows[0, 3:7] = position, orientation
    outward = np.zeros(steps + 1)
    moment = MOMENT * MOMENT_IN_BODY
    for step in range(steps):
        t = (step + 1) * dt
        field = amplitude * np.array([math.cos(omega * t), math.sin(omega * t), 0.0])
        torque = np.cross(_rotation(orientation) @ moment, field)
        if scheme.integrator == "explicit":
            surrogate_here = in_world(surrogate(_offset(position)), position)
            spin = spin + dt * (torque + drag[3:]) / INERTIA
            stepped = velocity + dt * (GRAVITY_FORCE + drag[:3]) / MASS
            position, velocity = project(position + dt * stepped, stepped)
            resistance = surrogate_here + in_world(lubrication(_offset(position)), position)
            drag = -resistance @ np.concatenate([velocity, spin])
        else:
            offset = _offset(position)
            resistance = in_world(surrogate(offset) + lubrication(offset), position)
            twist = np.concatenate([velocity, spin])
            gyroscopic = -np.cross(spin, INERTIA * spin)
            system = mass_matrix / dt + resistance
            load = mass_matrix / dt @ twist + np.concatenate([GRAVITY_FORCE, torque + gyroscopic])
            moved = np.linalg.solve(system, load)
            normal = _turn(position)[:3, 0]
            in_contact = (offset - TUBE) + EXTENT + MARGIN >= -1e-15
            if scheme.integrator == "implicit, contact held" and in_contact and moved[:3] @ normal > 0:
                held = np.zeros((7, 7))
                held[:6, :6], held[:6, 6], held[6, :3] = system, -np.concatenate([normal, np.zeros(3)]), normal
                moved = np.linalg.solve(held, np.concatenate([load, [0.0]]))[:6]
            outward[step + 1] = max(float(moved[:3] @ normal), 0.0)
            position, velocity = project(position + dt * moved[:3], moved[:3])
            spin = moved[3:]
        orientation = _multiply(_exponential(dt * spin), orientation)
        orientation /= np.linalg.norm(orientation)
        rows[step + 1] = np.concatenate([position, orientation, velocity, spin, field])
    time = np.arange(steps + 1) * dt
    history = History(time, rows[:, :3], rows[:, 7:10], rows[:, 3:7], rows[:, 10:13], rows[:, 13:16])
    start = time[-1] - math.floor(time[-1] / 2 * frequency + 1e-9) / frequency
    return Emulated(history, float(outward[time >= start - 1e-12].mean()))


def quantities(emulated: Emulated, frequency: float) -> dict[str, float]:
    found = steady(emulated.history, REFERENCE_FRAME, frequency)
    found["transient_speed_0_to_0.5_s_m_s"] = speed_between(emulated.history, REFERENCE_FRAME, 0.0, 0.5)
    return found


def difference(name: str, value: float, reference: float) -> float:
    """Relative for every quantity but the lag, which is absolute (rad)."""
    return value - reference if name == "phase_lag_field_minus_moment_rad" else (value - reference) / abs(reference)


def _differences(found: dict[str, float], against: dict[str, float]) -> dict[str, float]:
    return {name: difference(name, found[name], against[name]) for name in (*GATED, *REPORTED,
                                                                              "transient_speed_0_to_0.5_s_m_s")
            if name in found and name in against}


def _trajectory_check(emulated: Emulated) -> dict[str, float]:
    reference = np.load(REFERENCES / "f64_trajectory_every_10th_step.npz")
    every = round(float(reference["time_s"][1] - reference["time_s"][0]) / DT)
    h = emulated.history
    return {
        "position_max_abs_m": float(np.abs(h.position[::every] - reference["state/body/position"]).max()),
        "velocity_max_abs_m_s": float(np.abs(h.velocity[::every] - reference["state/body/velocity"]).max()),
        "orientation_max_abs": float(np.abs(h.orientation[::every] - reference["state/body/orientation"]).max()),
        "angular_velocity_max_abs_rad_s": float(np.abs(h.angular_velocity[::every]
                                                       - reference["state/body/angular_velocity"]).max()),
        "field_max_abs_T": float(np.abs(h.field[::every] - reference["state/field/field_vector"]).max()),
    }


def _operating_point(emulated_0_2: Emulated, emulated_0_3: Emulated) -> dict[str, Any]:
    """The surrogate's resistance where A's body rests, and the lag it implies under each hypothesis about which
    orientation 0.2's torque read."""
    constraint = CONSTRAINT_0_2
    canonical = surrogate(constraint) + lubrication(constraint)
    rotational = float(canonical[5, 5])
    omega = 2 * math.pi * 10.0
    ratio = rotational * omega / (MOMENT * FIELD)
    lag = math.asin(ratio)
    h = emulated_0_2.history
    inside = h.time >= 2.5 - 1e-12
    twist_canonical = np.array([(_turn(p).T @ np.concatenate([v, w]))
                                for p, v, w in zip(h.position[inside], h.velocity[inside], h.angular_velocity[inside],
                                                   strict=True)]).mean(axis=0)
    axial_torque = float(canonical[5] @ twist_canonical)
    lag_with_coupling = math.asin(axial_torque / (MOMENT * FIELD))
    return {
        "offset_m": constraint,
        "resistance_canonical_SI": canonical.tolist(),
        "rotational_resistance_about_the_axis_N_m_s": rotational,
        "R_omega_over_m_B": ratio,
        "arcsin_R_omega_over_m_B_rad": lag,
        "mean_twist_in_window_canonical": twist_canonical.tolist(),
        "arcsin_of_the_steady_axial_drag_torque_over_m_B_rad": lag_with_coupling,
        "omega_dt_rad": omega * DT,
        "lag_if_0_2_torque_read_the_previous_orientation_rad": lag - omega * DT,
        "lag_if_0_2_torque_read_this_steps_orientation_rad": lag,
        "outward_velocity_left_for_the_projection_by_0_3_m_s": emulated_0_3.outward_before_projection_m_s,
        "dt_times_buoyant_weight_over_mass_m_s": DT * float(np.linalg.norm(GRAVITY_FORCE)) / MASS,
    }


def main() -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    reference = {"f64_10": _read("f64_steady.json"), "f32_10": _read("f32_steady.json"),
                 "f64_11": _read("f64_freq11hz_steady.json")}
    print("emulating the reference (0.2's scheme) ...", flush=True)
    old = {10.0: emulate(SCHEME_0_2), 11.0: emulate(SCHEME_0_2, frequency=11.0)}
    new = {10.0: emulate(SCHEME_0_3), 11.0: emulate(SCHEME_0_3, frequency=11.0)}
    old_q = {f: quantities(e, f) for f, e in old.items()}
    new_q = {f: quantities(e, f) for f, e in new.items()}
    validation = {
        "steady_f64_10hz": _differences(old_q[10.0], reference["f64_10"]),
        "steady_f64_11hz": _differences(old_q[11.0], reference["f64_11"]),
        "trajectory_f64_10hz_every_10th_step": _trajectory_check(old[10.0]),
    }
    print(json.dumps(validation, indent=1), flush=True)

    print("each change alone ...", flush=True)
    alone = {
        "commands in the run's precision": replace(SCHEME_0_2, commands="run"),
        "all of the outward velocity removed (MIME-ANO-111)": replace(SCHEME_0_2, projection="0.3"),
        "implicit resistance with the resistance at the start of the step": replace(SCHEME_0_2, integrator="implicit"),
    }
    attribution = {name: _differences(quantities(emulate(s), 10.0), old_q[10.0]) for name, s in alone.items()}
    f32_scheme = replace(SCHEME_0_3, commands="float32")
    f32_delta = _differences(quantities(emulate(f32_scheme), 10.0), old_q[10.0])

    print("halving dt ...", flush=True)
    convergence: dict[str, dict[str, Any]] = {}
    for dt in (DT, DT / 2, DT / 4):
        pair = {label: quantities(emulate(s, dt=dt), 10.0) for label, s in
                (("0.2", replace(SCHEME_0_2, commands="run")), ("0.3", SCHEME_0_3), ("contact held", CONTACT_HELD))}
        convergence[f"{dt:g}"] = {
            "swim_speed_axial_m_s": {k: v["swim_speed_axial_m_s"] for k, v in pair.items()},
            "phase_lag_field_minus_moment_rad": {k: v["phase_lag_field_minus_moment_rad"] for k, v in pair.items()},
            "speed_0_3_over_0_2_minus_1": pair["0.3"]["swim_speed_axial_m_s"] / pair["0.2"]["swim_speed_axial_m_s"]
            - 1,
            "speed_contact_held_over_0_2_minus_1": pair["contact held"]["swim_speed_axial_m_s"]
            / pair["0.2"]["swim_speed_axial_m_s"] - 1,
        }
        if dt == DT / 2:
            half_dt = pair["0.3"]
    gaps = [c["speed_0_3_over_0_2_minus_1"] for c in convergence.values()]
    speeds = {k: [c["swim_speed_axial_m_s"][k] for c in convergence.values()] for k in ("0.2", "0.3", "contact held")}
    limits = {k: v[-1] + (v[-1] - v[-2]) for k, v in speeds.items()}  # Richardson, first order

    predicted = {
        "f64_10hz": {name: new_q[10.0][name] for name in (*GATED, *REPORTED, "transient_speed_0_to_0.5_s_m_s")},
        "f64_11hz": {name: new_q[11.0][name] for name in (*GATED, *REPORTED)},
        "f64_10hz_dt_2.5e-4": {name: half_dt[name] for name in (*GATED, *REPORTED)},
        "f32_10hz": {name: (reference["f32_10"][name] + f32_delta[name] if name == "phase_lag_field_minus_moment_rad"
                            else reference["f32_10"][name] * (1 + f32_delta[name])) for name in (*GATED, *REPORTED)},
    }
    record = expected_differences(reference, validation, attribution, f32_delta, new_q, old_q, convergence, gaps,
                                  limits, _operating_point(old[10.0], new[10.0]), predicted)
    record["lag_structure"] = lag_structure()
    OUTPUT.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {OUTPUT}")


def expected_differences(reference: dict[str, dict[str, Any]], validation: dict[str, Any],
                         attribution: dict[str, dict[str, float]], f32_delta: dict[str, float],
                         new_q: dict[float, dict[str, float]], old_q: dict[float, dict[str, float]],
                         convergence: dict[str, Any], gaps: list[float], limits: dict[str, float],
                         operating: dict[str, Any], predicted: dict[str, dict[str, float]]) -> dict[str, Any]:
    integrator = attribution["implicit resistance with the resistance at the start of the step"]
    projection = attribution["all of the outward velocity removed (MIME-ANO-111)"]
    commands = attribution["commands in the run's precision"]
    pct = 100 * integrator["swim_speed_axial_m_s"]
    return {
        "written": "2026-10-23, before 0.3 ran script A against this reference (the only earlier 0.3 run of A was "
                   "4c1's 2 s smoke check, 3.56 mm/s over its second second); by predict_script_a.py, from the "
                   "reference alone",
        "comparison": "script A built through 0.3's public API (the reproduction arm, tests/script_a.py with the "
                      "reference's values, the body starting at the reference's position) against the frozen 0.2 "
                      "reference in references/script_a/ (MIME-port041 at 6485516, MADDENING 0.4.1, CPU), by the "
                      "reference's own window rule (script_a_steady.py)",
        "shared": [
            "the drag surrogate's weights (sha256 ac1fca5c...0a345) and its feature formula; 0.3's drag node agrees "
            "with the port's to 2.8e-15 off the axis (test_port_drag.py)",
            "the resistance about the end of the body's axis, applied about its centre of mass (MIME-ANO-112): both "
            "sides, so not a difference here",
            "the body's values as the reference declared them: m_eff, I_eff (isotropic), the buoyant volume, the "
            "moment of two magnets, the outer radius 1.33 R as the contact's extent and the lubrication's sphere",
            "the field's time as its step count times dt (the port already counted steps; the tag's float32 "
            "accumulator, 1e-4 in f32, is not in this reference)",
        ],
        "how_predicted": "predict_script_a.py integrates A in NumPy with 0.2's update rules and with 0.3's, the drag "
                         "model shared. With 0.2's rules it reproduces the reference (validation below); each of "
                         "0.3's changes is then made alone and together. The predicted 0.3 values are that "
                         "emulation's, in float64; float32's are the f32 reference moved by the emulated "
                         "difference.",
        "validation_of_the_emulation_against_the_reference": validation,
        "differences": [
            {
                "id": "lag-structure",
                "what": "which orientation the torque reads",
                "reasoning": "0.2's graph adds the magnet node before the body, on the loop body -> magnet -> body, "
                             "so body.orientation -> magnet.orientation reads the previous step, as 0.3's "
                             "umr.pose -> magnetic.torque:umr.pose does. Both measured lags therefore contain "
                             "-omega dt. Settled before the run from the surrogate's rotational resistance about the "
                             "tube's axis where the body rests (with lubrication): see operating_point. With the "
                             "previous orientation, the measured lag is arcsin(R omega / m B) - omega dt; with this "
                             "step's, arcsin(R omega / m B) alone. The reference's lag is the first to within the "
                             "steady translation's coupling, and 0.03 rad from the second. The emulation, which "
                             "reads the previous orientation, reproduces the reference's lag to rounding.",
                "predicted": {"phase_lag_field_minus_moment_rad": 0.0},
                "check": "test_script_a_lag_structure.py asserts the declared mapping of the two graphs' edges",
            },
            {
                "id": "integrator",
                "what": "drag: 0.2's explicit drag from the previous step against 0.3's implicit resistance, with the "
                        "body confined by projection",
                "reasoning": "Expected beforehand: no change in the steady quantities while locked. At A's operating "
                             "point that does not hold. The body rests on the wall, and each step 0.3's implicit "
                             "solve gives it the outward velocity of a fall, about dt F/m (F the buoyant weight and "
                             "the outward hydrodynamic load), which the projection then removes: but the solve has "
                             "already resisted that velocity, so the resistance's column along the wall's normal "
                             "adds a load the body never feels (to the axial force and, through the end-referenced "
                             "coupling MIME-ANO-112 records, to the tilting torques). The emulation leaves "
                             f"{operating['outward_velocity_left_for_the_projection_by_0_3_m_s']:.3g} m/s outward for "
                             "the projection each step (dt times the buoyant weight over the mass: "
                             f"{operating['dt_times_buoyant_weight_over_mass_m_s']:.3g} m/s). 0.2's drag reads the "
                             "velocity after the projection and never sees the fall. The error is first order in "
                             "dt: the gap halves as dt halves, and both schemes meet as dt -> 0 (convergence). "
                             f"Predicted: the swim speed {pct:+.3f}%, the lag "
                             f"{integrator['phase_lag_field_minus_moment_rad']:+.3e} rad, the body's rotation rate "
                             f"{integrator['body_rotation_rate_hz']:+.2e}. The control: with the contact held in the "
                             "solve (the velocity along the normal set to zero inside it), 0.3's scheme gives "
                             f"{100 * convergence[f'{DT:g}']['speed_contact_held_over_0_2_minus_1']:+.3f}% at this "
                             "dt instead, the remaining difference between implicit and explicit resistance.",
                "predicted": integrator,
                "transient": "differs: over the first 0.5 s, the speed by displacement changes by "
                             f"{100 * integrator['transient_speed_0_to_0.5_s_m_s']:+.2f}%",
                "first_order_in_dt": {"speed_gap_at_dt_dt2_dt4": gaps, "ratios": [gaps[0] / gaps[1],
                                                                                  gaps[1] / gaps[2]],
                                      "speed_as_dt_to_0_by_richardson_m_s": limits},
            },
            {
                "id": "MIME-ANO-111",
                "what": "the 0.2 projection removed only (R/r)^2 of the outward velocity",
                "reasoning": "On the floor the step overshoots the wall by about dt times the outward velocity u "
                             "(operating_point), so 1 - (R/r)^2 is about 2 dt u / R, and the reference's body kept "
                             "that fraction of u, outward, into the next step's explicit drag. Predicted: the swim "
                             f"speed {projection['swim_speed_axial_m_s']:+.2e}, the lag "
                             f"{projection['phase_lag_field_minus_moment_rad']:+.1e} rad, the rotation rates under "
                             "1e-10.",
                "predicted": projection,
            },
            {
                "id": "MIME-ANO-110",
                "what": "the 0.2 moment computed in float32 under x64",
                "reasoning": "Predicted: none in this reference. The rounding (7.9e-9) appears where the port's node "
                             "runs on its own parameters, as the node-level differential called it; in the "
                             "reference's graph run MADDENING 0.4.1 injects m_single as a float64 array and the "
                             "product is float64. The reference's recorded moment is 1.68e-3 A m^2 to 1e-15 at "
                             "every step (the tag's run on MADDENING 0.3.1 carries the 7.9e-9). The emulation with "
                             "the float32 moment misses the reference's lag by 1.5e-10 rad; with the float64 moment "
                             "it matches.",
                "predicted": dict.fromkeys(GATED, 0.0),
            },
            {
                "id": "commands",
                "what": "found in the mapping: the reference's frequency and amplitude were float32 external inputs",
                "reasoning": "The reference's controller built both as float32 scalars, so under x64 the field turned "
                             "at 2 pi f rounded to float32 (+4.30e-8: the reference's field rate is 10.00000043 Hz) "
                             "with |B| = 1.2000000570 mT (+4.75e-8). 0.3's drives are in the run's precision. "
                             "Float64 only: in float32 both lines round alike.",
                "predicted": commands,
            },
            {
                "id": "field-time",
                "what": "the field's timekeeping",
                "reasoning": "Predicted: none. The reference is the port, which already took the field's time as "
                             "its step count times dt, as 0.3 does; the tag's float32 accumulator (about 1e-4 in "
                             "float32) was measured by phase 0's tag control and is not in this reference.",
                "predicted": dict.fromkeys(GATED, 0.0),
            },
            {
                "id": "surrogate-fixes",
                "what": "nu's range, the axis from the geometry, the value on the axis",
                "reasoning": "Predicted: none. nu = 2.33 lies in both ranges; the tube's axis from the geometry gives "
                             "the +z tube's resistance turned (4.5e-16, test_drag.py), and A's tube along x is the "
                             "reference's along z turned; the body rests 1.0002 mm off the axis, far above the "
                             "on-axis resolution (3e-15 m in float64, 1.6 um in float32).",
                "predicted": dict.fromkeys(GATED, 0.0),
            },
            {
                "id": "frame",
                "what": "found in the mapping: the frame",
                "reasoning": "Predicted: none beyond rounding. 0.3's A runs along x with up z and the moment along "
                             "body y; the reference along z with gravity along -y and the moment along body x: the "
                             "cyclic permutation x -> y -> z -> x, a proper rotation, with the field starting along "
                             "the moment in both. Every steady quantity is measured about the tube's axis.",
                "predicted": dict.fromkeys(GATED, 0.0),
            },
            {
                "id": "lubrication-and-surrogate-reads",
                "what": "found in the mapping: when the drag reads the body's position",
                "reasoning": "0.2's surrogate read the position at the start of the step before (a back edge), its "
                             "lubrication the position after the body's step (a forward edge); 0.3's one node reads "
                             "the start of the step. On the wall the offset is the constraint's radius at every "
                             "step, and the surrogate depends on the offset alone, so only the offset's direction "
                             "moves between those reads (about 1e-4 rad a step). Inside the integrator's emulation, "
                             "not separable from it.",
                "predicted": "within 'integrator'",
            },
        ],
        "operating_point": operating,
        "predicted": predicted,
        "emulated_0_2_at_10_and_11_hz": {f"{f:g}": {name: q[name] for name in (*GATED, *REPORTED)}
                                         for f, q in old_q.items()},
        "float32": {"difference_from_the_f32_reference": f32_delta,
                    "how": "0.3's scheme against 0.2's, both with float32 commands, emulated in float64, applied to "
                           "the f32 reference"},
        "tolerances": {
            "f64": {"swim_speed_axial_m_s": 1e-7, "swim_speed_axial_mean_velocity_m_s": 1e-7,
                    "body_rotation_rate_hz": 1e-9, "field_rotation_rate_hz": 1e-9,
                    "phase_lag_field_minus_moment_rad": 1e-9, "transient_speed_0_to_0.5_s_m_s": 1e-6},
            "f32": {"swim_speed_axial_m_s": 2e-4, "swim_speed_axial_mean_velocity_m_s": 2e-4,
                    "body_rotation_rate_hz": 1e-6, "field_rotation_rate_hz": 1e-6,
                    "phase_lag_field_minus_moment_rad": 1e-7},
            "units": "relative to the predicted value, except the lag (rad)",
            "basis": "f64: the emulation reproduces the reference's steady quantities to about 1e-15 and its "
                     "trajectory to rounding, so the tolerance is far above a second implementation's rounding and "
                     "below every declared difference that is not none, except the commands' on the speeds (4e-8) "
                     "and MIME-ANO-111's on the rotation rates (1e-11). f32: at least ten times the reference's own "
                     "float32-float64 "
                     "gap in each kind of quantity (speeds 1.3e-5, rotation rates 5.3e-8, lag 8.5e-9 rad), since "
                     "each line rounds on its own.",
        },
        "gated": list(GATED),
        "reported_not_gated": REPORTED,
        "controls": {
            "frequency": "0.3 at 11 Hz matches f64_freq11hz plus the 11 Hz prediction within the f64 tolerances; 0.3 "
                         "at 10 Hz must miss that 11 Hz prediction on every gated quantity",
            "dt": "0.3 at dt = 2.5e-4 s matches the prediction at that dt (predicted['f64_10hz_dt_2.5e-4']) within "
                  "the f64 tolerances, and its speed's gap to 0.2's scheme at that dt (emulated: convergence) is "
                  "half its gap to the reference at 5e-4: the ratio of the gaps lies between 1.9 and 2.1 (emulated: "
                  f"{gaps[0] / gaps[1]:.3f})",
            "mechanism": "the contact held inside the implicit solve removes the integrator's difference to "
                         "within the implicit-explicit difference (convergence['contact held'])",
        },
        "convergence": convergence,
        "if_a_quantity_misses": "the tolerance stays; the cause is found and reported; a new expected difference is "
                                "added only with its own prediction, committed before re-running, and labelled post "
                                "hoc",
    }


#: Each 0.2 edge's counterpart in 0.3's graph of script A (``None``: no counterpart), and how the two relate. Declared
#: by reading both graphs' builders; the lags beside them are computed by MIME's own rule.
EDGE_MAP: dict[str, tuple[str | None, str]] = {
    "field.field_vector -> magnet.field_vector":
        ("field.magnetic.flux_density_magnetic_torque_umr -> magnetic.torque:umr.flux_density", "same"),
    "field.field_gradient -> magnet.field_gradient":
        (None, "dropped: 0.3's torque law reads no gradient (a gradient's force is another law's; the uniform "
               "field's is zero)"),
    "body.orientation -> magnet.orientation":
        ("umr.pose -> magnetic.torque:umr.pose", "same: the torque reads the orientation at the start of the step"),
    "body.position -> mlp_drag.robot_position": ("umr.pose -> drag:umr.pose", "same"),
    "body.orientation -> mlp_drag.robot_orientation": ("umr.pose -> drag:umr.pose", "same (one edge carries the pose)"),
    "body.velocity -> mlp_drag.body_velocity":
        (None, "dropped: it fed the surrogate's own drag, which the lubrication node's replaced; 0.3 delivers the "
               "resistance, which the body applies to the velocity it solves for"),
    "body.angular_velocity -> mlp_drag.body_angular_velocity": (None, "dropped, as the velocity's"),
    "gravity.gravity_force -> body.external_force": ("gravity:umr.wrench -> umr.applied_wrench", "same"),
    "magnet.magnetic_force -> body.magnetic_force":
        ("magnetic.torque:umr.wrench -> umr.applied_wrench", "same (one wrench carries force and torque)"),
    "magnet.magnetic_torque -> body.magnetic_torque": ("magnetic.torque:umr.wrench -> umr.applied_wrench", "same"),
    "mlp_drag.resistance_matrix -> lub.resistance_matrix":
        (None, "merged: the surrogate and the lubrication are one node in 0.3"),
    "body.position -> lub.robot_position":
        ("umr.pose -> drag:umr.pose", "differs: 0.2's lubrication read the position after the body's step, 0.3's "
                                      "the start of the step"),
    "body.velocity -> lub.body_velocity":
        (None, "differs: 0.2's drag took the velocity after the step and the body applied it in the next; 0.3's "
               "resistance is applied to the velocity the body's implicit solve gives"),
    "body.angular_velocity -> lub.body_angular_velocity": (None, "differs, as the velocity's"),
    "lub.drag_force -> body.drag_force":
        ("drag:umr.resistance -> umr.resistance",
         "differs: a delivered load read a step late (explicit drag) becomes a resistance read this step and applied "
         "implicitly"),
    "lub.drag_torque -> body.drag_torque": ("drag:umr.resistance -> umr.resistance", "differs, as the force's"),
}
#: 0.3's edges with no 0.2 counterpart.
NEW_IN_0_3 = {
    "drive:field.amplitude.value -> field.magnetic.amplitude":
        "0.2's commands were external inputs, built as float32 by its controller; 0.3's are constant drives",
    "drive:field.frequency.value -> field.magnetic.frequency": "as the amplitude's",
    "drag:umr.background_load -> umr.background_load": "the load of the ambient flow on the held body: zero in "
                                                       "quiescent water",
    "umr.twist -> swim_speed:umr.twist": "an observation: 0.2's speed was computed from its history afterwards",
}


def edges_0_2() -> list[tuple[str, str, str, str]]:
    """The reference graph's edges, in the order its builder added them, from its provenance."""
    found = []
    for spec in PROVENANCE["edges"]:
        source, target = spec.removeprefix("EdgeSpec(").removesuffix(")").split(" -> ")
        found.append((*source.split("."), *target.split(".")))
    return found


def lag_structure() -> dict[str, Any]:
    """Both graphs' edges classified by MIME's lag rule, and the declared mapping between them."""
    from script_a import reproduction_arm

    from mime.build._builders import DELIVERY, FIELD, STATE, EdgeDecl
    from mime.build._lags import classify, unplanned_refusals

    carries = {"body": STATE, "field": FIELD}
    declared = [EdgeDecl("reference", (a, b), (c, d), False, "", i, carries.get(a, DELIVERY))
                for i, (a, b, c, d) in enumerate(edges_0_2())]
    old = classify(list(NODES), declared, [])
    new = reproduction_arm().build().report.lags
    new_lags = {edge.text: edge.lag.value for edge in new}
    refused = unplanned_refusals(old, {})
    return {
        "rule": "inside a loop, an edge whose source is scheduled at or after its target reads the previous step; a "
                "previous-step read is planned when it carries a thing's published state (mime.build._lags)",
        "node_order_0_2": list(NODES),
        "edges_0_2": {edge.text: edge.lag.value for edge in old},
        "edges_0_3": new_lags,
        "mapping": {text: {"lag_0_2": edge.lag.value, "counterpart_0_3": EDGE_MAP[text][0],
                           "lag_0_3": new_lags.get(EDGE_MAP[text][0] or "", None), "relation": EDGE_MAP[text][1]}
                    for text, edge in zip((e.text for e in old), old, strict=True)},
        "new_in_0_3": {text: {"lag_0_3": new_lags[text], "why": why} for text, why in NEW_IN_0_3.items()},
        "previous_step_0_2": [edge.text for edge in old if edge.lag.value == "previous step"],
        "previous_step_0_3": [edge.text for edge in new if edge.lag.value == "previous step"],
        "refused_by_0_3s_rule_in_0_2": [r.message.split(", built by")[0].removeprefix("the edge ") for r in refused],
        "inference": "it holds: the torque reads the orientation at the start of the step in both graphs, so both "
                     "measured lags contain -omega dt and agree (operating_point: the 'previous' hypothesis lies "
                     "1.7e-4 rad from the reference's lag, the 'this step' one 0.031 rad)",
    }


if __name__ == "__main__":
    main()
