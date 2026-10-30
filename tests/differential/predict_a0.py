"""Predict how A0 in 0.3 differs from its frozen reference, before 0.3 runs it:

    JAX_PLATFORMS=cpu .venv/bin/python tests/differential/predict_a0.py

The reference (``references/a0/``) is the pre-0.3 tag's A0 (MADDENING 0.3.1) with one declared patch giving its drag
surrogate the tube's axis, in float32. This integrates A0 in float64 twice, from the update rules each line documents,
with the same drag model, and writes ``a0_expected_differences.json`` beside this file. It reads only the reference
(its parameters, steady quantities and trajectory at every tenth step), the drag surrogate's weights, the AR4
description and the rigid-body library's forward kinematics and Jacobian (``mime.core.multibody``, held to the 0.2
line's by ``test_port_multibody.py``). It imports no node and no controller of 0.3. Not collected by pytest.

**The tag's scheme**, read from its sources and its graph's provenance. Each step: the controller (computed from the
state after the step before) writes joint torques that make each joint accelerate as ``Kp (q_t - q) - Kd qdot`` for
the joint target ``q_t`` of five damped-least-squares steps from the present joints towards the filtered target pose;
the arm integrates them (semi-implicit Euler), the motor's PI velocity loop turns the rotor, and then the coupling
group runs six Gauss-Seidel passes of field, magnet response, drag, body and lubrication from the start-of-step state,
each node reading the latest value of the others, the body's position and orientation as the previous pass left them.
The body steps explicitly under the drag the lubrication node computed from the body's new velocity, so the passes
converge to a backward-Euler step; the projection removes ``(R / r)^2`` of the outward velocity (MIME-ANO-111).

**0.3's scheme**: the arm is kinematic (its joint positions are the controller's last command), the rotor's angle is
the integral of its prescribed speed, the field is read at the start-of-step body, and the body steps as script A's
does (``predict_script_a.py``: ``(M/dt + R) dV = F - R V`` with the resistance at the start-of-step pose, the contact
in the solve, the surrogate about the body's centre).

With the tag's scheme the emulation reproduces the reference's steady quantities and trajectory, which is what makes
its 0.3 run a prediction."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).parent
REFERENCES = HERE / "references" / "a0"
OUTPUT = HERE / "a0_expected_differences.json"
URDF = HERE.parent / "descriptions" / "ar4.urdf"
sys.path[:0] = [str(HERE), str(HERE.parent)]

import predict_script_a as A  # noqa: E402
from a0_runs import REFERENCE_FRAME, SCRIPT_A_FRAME, A0History, quantities, reference_history  # noqa: E402
from script_a_steady import History  # noqa: E402

PROVENANCE = json.loads((REFERENCES / "f32_3hz_provenance.json").read_text())
PARAMETERS = PROVENANCE["parameters"]["params_py_namespace_as_runner_passes_it"]
NODES = PROVENANCE["nodes"]

DT = PARAMETERS["DT_PHYS"]
MASS, INERTIA = NODES["body"]["params"]["m_eff"], NODES["body"]["params"]["I_eff"]
EXTENT = NODES["body"]["params"]["semi_major_axis_m"]
TUBE = 3.175e-3  # the 1/4 inch tube's inner radius
MARGIN = 1e-4
CONSTRAINT = TUBE - EXTENT - MARGIN  # the 0.2 builder's: R_ves - R_max_body - 0.1 mm
#: The buoyant weight: (rho_body - rho_water) V g with V = m_eff / rho_body, g = 9.81, which the tag held as a float32.
GRAVITY_FORCE = np.array([0.0, 0.0, -PARAMETERS["DELTA_RHO_KG_M3"] * NODES["body"]["params"]["m_eff"]
                          / NODES["body"]["params"]["density_kg_m3"] * 9.81])
BODY_MOMENT = NODES["magnet"]["params"]["n_magnets"] * NODES["magnet"]["params"]["m_single"]
ROTOR_MOMENT = NODES["ext_magnet"]["params"]["dipole_moment_a_m2"]
TAG_MU0 = 4.0 * math.pi * 1e-7  # the tag's; 0.3's is CODATA 2022's, 1.25663706127e-6 (units.VACUUM_PERMEABILITY)
MU0_0_3 = 1.25663706127e-6
CONTACT_ULPS = 64
AXIS = np.array([1.0, 0.0, 0.0])

BASE = np.array(PARAMETERS["BASE_POSE_WORLD"][:3])
HOME = np.array(PARAMETERS["ARM_HOME_RAD"])
STANDOFF, REACH = PARAMETERS["CONTROL_STANDOFF_M"], (PARAMETERS["CONTROL_X_MIN_M"], PARAMETERS["CONTROL_X_MAX_M"])
ALPHA, K_IK, LAMBDA = PARAMETERS["CONTROL_TARGET_ALPHA"], PARAMETERS["CONTROL_K_IK"], PARAMETERS["CONTROL_IK_LAMBDA"]
ITERATIONS, K_P, K_D = PARAMETERS["CONTROL_IK_INNER_ITERS"], PARAMETERS["CONTROL_K_P"], PARAMETERS["CONTROL_K_D"]
LIMITS = np.array(NODES["arm"]["params"]["joint_limits"])
MOTOR = NODES["motor"]["params"]
TAG_PASSES = PROVENANCE["coupling_groups"][0].split("max_iterations=")[1].split(",")[0]


# -- the arm and the controller's inverse kinematics -------------------------------------------------------------------

_arm: list[Any] = []


def arm() -> Any:
    """The AR4's tree, and the jitted five-step inverse kinematics from a joint vector towards a world position with the
    home pose's orientation held."""
    if _arm:
        return _arm[0]
    import jax
    import jax.numpy as jnp

    from mime.core.multibody import damped_least_squares, frame_jacobian, frame_pose, read_urdf

    tree = read_urdf(URDF).tree
    home = np.asarray(frame_pose(tree, jnp.asarray(HOME), "link_6"))
    r_home = _rotation(home[3:])

    def rotation_vector(matrix: Any) -> Any:
        vee = 0.5 * jnp.array([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]])
        sine = jnp.linalg.norm(vee)
        angle = jnp.arctan2(sine, 0.5 * (jnp.trace(matrix) - 1.0))
        return vee * jnp.where(sine > 1e-12, angle / jnp.maximum(sine, 1e-300), 1.0)

    def rotation(q: Any) -> Any:
        w, x, y, z = q
        return jnp.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                          [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                          [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

    def target_joints(q: Any, position: Any) -> Any:
        goal = position - jnp.asarray(BASE)
        for _ in range(ITERATIONS):
            reached = frame_pose(tree, q, "link_6")
            error = jnp.concatenate([goal - reached[:3], rotation_vector(jnp.asarray(r_home) @ rotation(reached[3:]).T)])
            q = q + damped_least_squares(frame_jacobian(tree, q, "link_6"), K_IK * error, LAMBDA)
        return q

    def flange(q: Any) -> Any:
        return frame_pose(tree, q, "link_6")

    _arm.append((jax.jit(target_joints), jax.jit(flange), np.asarray(BASE + home[:3]), _rotation(home[3:])))
    return _arm[0]


def _rotation(q: np.ndarray) -> np.ndarray:
    return A._rotation(np.asarray(q, dtype=np.float64))


@dataclass
class Controller:
    """The controller's filtered target position (its target orientation is held at the home pose's)."""

    target: np.ndarray

    def tick(self, body_x: float, joints: np.ndarray) -> np.ndarray:
        """The joint target of this tick, from the body's position along the tube and the present joints."""
        ik, _, _, _ = arm()
        f32 = np.float32
        raw = np.array([min(max(body_x, REACH[0]), REACH[1]), 0.0, STANDOFF], dtype=f32)
        # In single precision, as both controllers compute it, XLA fusing the product by 1 - ALPHA into the sum: the
        # filter stops closing the gap to the standoff once a step of it is below half a unit in the last place of
        # the target (1.3e-6 m short of it, not the 2.9e-7 m of unfused arithmetic).
        scaled = (f32(ALPHA) * raw).astype(np.float64)
        self.target = (float(f32(1.0 - ALPHA)) * self.target.astype(f32).astype(np.float64) + scaled).astype(f32
                                                                                                          ).astype(np.float64)
        return np.asarray(ik(joints, self.target))


# -- the field -------------------------------------------------------------------------------------------------------


def dipole(moment: np.ndarray, centre: np.ndarray, point: np.ndarray, mu0: float) -> tuple[np.ndarray, np.ndarray]:
    """``B`` and ``G_ij = dB_i/dx_j`` of a point dipole (the same closed forms as both lines')."""
    k = mu0 / (4.0 * math.pi)
    r = point - centre
    d = float(np.linalg.norm(r))
    along = float(moment @ r)
    field = k * (3.0 * along * r / d ** 5 - moment / d ** 3)
    gradient = (3.0 * k / d ** 5) * (along * np.eye(3) + np.outer(moment, r) + np.outer(r, moment)
                                     - 5.0 * along * np.outer(r, r) / d ** 2)
    return field, gradient


# -- the body --------------------------------------------------------------------------------------------------------


def turn(position: np.ndarray, end: float = 1.0) -> np.ndarray:
    """The canonical frame as a [translation; rotation] pair: ``[n, s a x n, s a]`` for the tube's axis ``a``."""
    radial = position - (position @ AXIS) * AXIS
    normal = radial / np.linalg.norm(radial)
    frame = np.stack([normal, end * np.cross(AXIS, normal), end * AXIS], axis=1)
    pair = np.zeros((6, 6))
    pair[:3, :3] = pair[3:, 3:] = frame
    return pair


def offset_of(position: np.ndarray) -> float:
    return float(np.hypot(position[1], position[2]))


def in_world(canonical: np.ndarray, position: np.ndarray) -> np.ndarray:
    pair = turn(position)
    return pair @ canonical @ pair.T


def project_tag(position: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The tag's ``CylindricalVesselConstraint``: ``(R / r)^2`` of the outward velocity removed (MIME-ANO-111)."""
    r = math.sqrt(position[1] ** 2 + position[2] ** 2 + 1e-30)
    if r <= CONSTRAINT:
        return position, velocity
    projected = position.copy()
    projected[1:] *= CONSTRAINT / r
    hat = np.array([0.0, projected[1] / r, projected[2] / r])
    return projected, velocity - max(float(velocity @ hat), 0.0) * hat


def project_0_3(position: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radial = position - (position @ AXIS) * AXIS
    r = float(np.linalg.norm(radial))
    beyond = (r - TUBE) + EXTENT + MARGIN
    if beyond <= 0.0:
        return position, velocity
    normal = radial / r
    return position - beyond * normal, velocity - max(float(velocity @ normal), 0.0) * normal


@dataclass(frozen=True)
class Scheme:
    """``line``: ``tag`` or ``0.3``. ``arm``: ``dynamic`` (the tag's computed-torque joints) or ``kinematic``.
    ``motor``: ``pi`` (the tag's velocity loop) or ``prescribed``. ``reference_point``: where the surrogate's
    resistance is referred, ``end`` (the tag) or ``centre`` (0.3). ``mu0``: the dipole constant."""

    line: str
    arm: str
    motor: str
    reference_point: str
    mu0: float
    passes: int = 6
    body_orientation: tuple[float, ...] = (0.7071068, 0.0, 0.7071068, 0.0)
    body_moment: tuple[float, ...] = (1.0, 0.0, 0.0)


TAG = Scheme("tag", "dynamic", "pi", "end", TAG_MU0)
NEW = Scheme("0.3", "kinematic", "prescribed", "centre", MU0_0_3, body_orientation=(1.0, 0.0, 0.0, 0.0),
             body_moment=(0.0, 1.0, 0.0))


def command(frequency: float) -> float:
    """The rotor's commanded speed, rounded to single precision as the tag's controller (a float32 external input) and
    0.3's float32 run both hold it."""
    return float(np.float32(2.0 * math.pi * frequency))


def emulate(scheme: Scheme, frequency: float, *, dt: float = DT, duration: float = 10.0,
            altered: dict[str, float] | None = None) -> A0History:
    """A0 for ``duration`` seconds in float64, its history as the reference's. ``altered`` plants a fault in the
    emulation (the controls' checks of the emulator itself)."""
    altered = altered or {}
    steps = round(duration / dt)
    ik_joints, flange, home_position, _ = arm()
    omega_command = command(frequency)
    surrogate_end = {}
    moment_body = BODY_MOMENT * np.asarray(scheme.body_moment) / np.linalg.norm(scheme.body_moment)

    def resistance(position: np.ndarray, with_lubrication: bool = False) -> np.ndarray:
        offset = offset_of(position)
        if offset not in surrogate_end:
            surrogate_end[offset] = A.surrogate(offset, scheme.reference_point)
        found = in_world(surrogate_end[offset], position)
        return found + in_world(A.lubrication(offset), position) if with_lubrication else found

    position = np.array([0.0, 0.0, -1e-3])
    orientation = np.array(scheme.body_orientation, dtype=np.float64)
    orientation /= np.linalg.norm(orientation)
    velocity, spin = np.zeros(3), np.zeros(3)
    joints, joint_rates = HOME.copy(), np.zeros(6)
    controller = Controller(home_position.copy())
    angle32, rotor_rate32, integral = np.float32(0.0), np.float32(0.0), np.float32(0.0)
    rotor_rate = 0.0
    drag = np.zeros(6)
    rows = np.zeros((steps + 1, 16 + 3 + 2 + 6))
    rows[0, :3], rows[0, 3:7] = position, orientation
    rows[0, 16:19] = np.asarray(flange(joints))[:3] + BASE
    rows[0, 21:] = joints
    mass_matrix = np.diag([MASS] * 3 + [INERTIA] * 3)
    for step in range(steps):
        # the controller reads the state after the step before
        target_joints = controller.tick(position[0], joints)
        if scheme.arm == "dynamic":
            acceleration = K_P * (target_joints - joints) - K_D * joint_rates
            joint_rates = joint_rates + acceleration * dt
            joints = np.clip(joints + joint_rates * dt, LIMITS[:, 0], LIMITS[:, 1])
        else:
            joints = np.clip(target_joints, LIMITS[:, 0], LIMITS[:, 1])
        pose = np.asarray(flange(joints))
        parent_position, parent_rotation = pose[:3] + BASE, _rotation(pose[3:])
        # The rotor's angle is a single-precision accumulator in both lines (the tag's motor state, 0.3's kinematic
        # motor in a float32 run): at 100 rad one unit in its last place is 7.6e-6 rad to 1.5e-5 rad, against a step
        # of 9.4e-3, so each step adds the step rounded to a whole number of units (a rate off by up to 6e-4).
        if scheme.motor == "pi":
            f32 = np.float32
            error = f32(omega_command) - rotor_rate32
            integral = f32(integral + error * f32(dt))
            torque = f32(f32(MOTOR["velocity_kp"]) * error + f32(MOTOR["velocity_ki"]) * integral)
            rotor_rate32 = f32(f32(f32(MOTOR["inertia_kg_m2"]) * rotor_rate32 + f32(dt) * torque) / f32(
                f32(MOTOR["inertia_kg_m2"]) + f32(dt) * f32(MOTOR["damping_n_m_s"]) + f32(1e-30)))
            rotor_rate = float(rotor_rate32)
            angle32 = f32(angle32 + rotor_rate32 * f32(dt))
        else:
            rotor_rate = omega_command
            angle32 = np.float32(angle32 + np.float32(omega_command) * np.float32(dt))
        angle = float(angle32)
        rotor_moment = ROTOR_MOMENT * altered.get("moment", 1.0) * parent_rotation @ np.array(
            [math.cos(angle), math.sin(angle), 0.0])
        magnet = parent_position + np.array(altered.get("magnet_at", (0.0, 0.0, 0.0)))
        if scheme.line == "tag":
            latest_p, latest_q, latest_v, latest_w = position, orientation, velocity, spin
            for _ in range(scheme.passes):
                field, gradient = dipole(rotor_moment, magnet, latest_p, scheme.mu0)
                moment_world = A._rotation(latest_q) @ moment_body
                magnetic_torque, magnetic_force = np.cross(moment_world, field), gradient @ moment_world
                resistance_here = resistance(latest_p)
                new_spin = spin + dt * (magnetic_torque + drag[3:]) / INERTIA
                new_velocity = velocity + dt * (GRAVITY_FORCE + magnetic_force + drag[:3]) / MASS
                new_position, new_velocity = project_tag(position + dt * new_velocity, new_velocity)
                new_orientation = A._multiply(A._exponential(dt * new_spin), orientation)
                new_orientation /= np.linalg.norm(new_orientation)
                corrected = resistance_here + in_world(A.lubrication(offset_of(new_position)), new_position)
                drag = -corrected @ np.concatenate([new_velocity, new_spin])
                latest_p, latest_q, latest_v, latest_w = new_position, new_orientation, new_velocity, new_spin
            position, orientation, velocity, spin = latest_p, latest_q, latest_v, latest_w
        else:
            field, gradient = dipole(rotor_moment, magnet, position, scheme.mu0)
            moment_world = A._rotation(orientation) @ moment_body
            torque, force = np.cross(moment_world, field), gradient @ moment_world
            here = resistance(position, with_lubrication=True)
            twist = np.concatenate([velocity, spin])
            system = mass_matrix / dt + here
            applied = np.concatenate([GRAVITY_FORCE + force, torque])
            moved = twist + np.linalg.solve(system, applied - here @ twist)
            normal = turn(position)[:3, 0]
            offset = offset_of(position)
            on_the_wall = (offset - TUBE) + EXTENT + MARGIN >= -CONTACT_ULPS * np.finfo(np.float64).eps * (
                float(np.linalg.norm(position)) + EXTENT + MARGIN)
            if on_the_wall and moved[:3] @ normal > 0:
                direction = np.concatenate([normal, np.zeros(3)])
                response = np.linalg.solve(system, direction)
                moved = moved - (direction @ moved) / (direction @ response) * response
            position, velocity = project_0_3(position + dt * moved[:3], moved[:3])
            spin = moved[3:]
            orientation = A._multiply(A._exponential(dt * spin), orientation)
            orientation /= np.linalg.norm(orientation)
        rows[step + 1] = np.concatenate([position, orientation, velocity, spin, field, parent_position, [angle,
                                                                                                          rotor_rate],
                                         joints])
    time = np.arange(steps + 1) * dt
    history = History(time, rows[:, :3], rows[:, 7:10], rows[:, 3:7], rows[:, 10:13], rows[:, 13:16])
    return A0History(history, rows[:, 16:19], rows[:, 20])


# -- the prediction ------------------------------------------------------------------------------------------------------

GATED = (*("swim_speed_axial_m_s", "swim_speed_axial_mean_velocity_m_s", "body_rotation_rate_hz",
          "field_rotation_rate_hz", "phase_lag_field_minus_moment_rad"),
         "tracking_error_axial_mean_m", "tracking_error_axial_rms_m", "magnet_body_distance_minus_standoff_mean_m",
         "rotor_rate_mean_hz")
LAG = "phase_lag_field_minus_moment_rad"
FREQUENCIES = {"3hz": 3.0, "10hz": 10.0}
#: Each of 0.3's differences from the tag, made alone to the tag's scheme, by the name the declaration gives it.
CHANGES = {
    "arm-dynamics": dict(arm="kinematic"),
    "rotor-motor": dict(motor="prescribed"),
    "surrogate-about-the-centre": dict(reference_point="centre"),
    "body-step": dict(line="0.3"),
    "vacuum-permeability": dict(mu0=MU0_0_3),
}


def difference(name: str, value: float, reference: float) -> float:
    """Relative, except the lag's (rad)."""
    return value - reference if name == LAG else (value - reference) / abs(reference)


def steady(history: A0History, frequency: float, scheme: Scheme) -> dict[str, float]:
    return quantities(history, frequency, REFERENCE_FRAME if scheme.body_moment == (1.0, 0.0, 0.0) else SCRIPT_A_FRAME)


def main() -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    tag_all: dict[str, dict[str, float]] = {}
    new_all: dict[str, dict[str, float]] = {}
    alone: dict[str, dict[str, dict[str, float]]] = {name: {} for name in CHANGES}
    other_convention: dict[str, dict[str, float]] = {}
    for key, frequency in FREQUENCIES.items():
        print(f"{key}: the tag's scheme, and 0.3's ...", flush=True)
        tag_all[key] = steady(emulate(TAG, frequency), frequency, TAG)
        new_all[key] = steady(emulate(NEW, frequency), frequency, NEW)
        in_tags_convention = replace(NEW, body_orientation=TAG.body_orientation, body_moment=TAG.body_moment)
        other_convention[key] = steady(emulate(in_tags_convention, frequency), frequency, in_tags_convention)
        for name, change in CHANGES.items():
            print(f"{key}: {name} alone ...", flush=True)
            changed = replace(TAG, **change)
            alone[name][key] = steady(emulate(changed, frequency), frequency, changed)
    reference = {f"{precision}_{key}": json.loads((REFERENCES / f"{precision}_{key}_steady.json").read_text())
                 for precision in ("f32", "mixed_f64") for key in FREQUENCIES}
    record = {
        "tag": tag_all, "new": new_all, "alone": alone, "other_convention": other_convention,
        "reference": {label: {name: value for name, value in found.items() if isinstance(value, float)}
                      for label, found in reference.items()},
    }
    (HERE / "a0_emulation.json").write_text(json.dumps(record, indent=1) + "\n")
    print("wrote a0_emulation.json")


EMULATION = HERE / "a0_emulation.json"
#: Quantities reported and not gated, and why.
NOT_GATED = {
    "radial_offset_mean_m": "the body rests on the projection's constraint (its minimum equals its maximum in the "
                            "reference): it measures the constraint, not the dynamics",
    "rotation_ratio_body_over_field": "locked, the body turns with the field whatever the frequency",
    "body_axial_angular_velocity_mean_hz": "the mean of the body's angular velocity, which carries the lock's ripple "
                                           "at the window's ends: the rotation rate by phase gates the same thing",
    "field_inplane_magnitude_mean_T": "a function of the standoff, which the distance gates",
    "tracking_error_lateral_y_mean_m": "about a micrometre, at the rounding of the controller's single precision "
                                       "(the emulation misses it by 4% of itself)",
    "tracking_error_axial_max_abs_m": "an extreme of one window, at the controller's rounding",
}
DECLARED_CHANGES = [
    {
        "id": "arm-dynamics",
        "alone": "arm-dynamics",
        "what": "the arm: the tag's computed-torque joints against 0.3's kinematic arm",
        "cause": "the tag's controller writes joint torques that make each joint accelerate as Kp (q_t - q) - Kd qdot "
                 "(Kp = 1e4, Kd = 200: critically damped at 100 rad/s), so the flange trails the joint target by "
                 "2 zeta / omega_n = 20 ms of its velocity; 0.3's kinematic arm takes the controller's joint "
                 "positions as they are, with no lag (the dynamic arm is phase 7a3's). The controller's filter "
                 "(0.1 s) is common to both.",
        "expected": "the magnet leads the tag's by 20 ms times the swim speed: the tracking error smaller by "
                    "v * 0.02 s / (v * (0.1 + 0.02) s) = 1/6, a fall of 15.6% (3 Hz) and 15.8% (10 Hz); the swim "
                    "speed down by the gradient force's share, 0.81% and 0.57%; the distance from the body (its "
                    "excess over the standoff) down by 1.2e-5 and 1.6e-4 of itself; the rotation rates and the lag "
                    "unmoved (under 1e-9 and 3e-6 rad)",
        "check": "test_a0.py gates the tracking error, the speed and the distance against the prediction",
    },
    {
        "id": "rotor-motor",
        "alone": "rotor-motor",
        "what": "the rotor: the tag's PI velocity loop against 0.3's prescribed speed",
        "cause": "the tag's motor turns the rotor by a PI loop on the commanded speed (J = 1e-5, Kp = 1e-3, Ki = 1e-2, "
                 "b = 1e-4), in single precision: it starts from rest, so its angle trails the integral of the "
                 "command by b omega_d / Ki (0.19 rad at 3 Hz, 0.63 rad at 10 Hz), and its speed carries the "
                 "loop's rounding, a mean 5.06e-7 (3 Hz) and 7.9e-7 (10 Hz) of itself below the command; 0.3's "
                 "kinematic motor turns at the command, from an angle of zero. Both accumulate the angle in "
                 "single precision (a rate off by up to 6e-4 from the step's rounding to units in the last "
                 "place, common to both).",
        "expected": "the rotor's mean speed above the tag's by 5.06e-7 (3 Hz) and 7.9e-7 (10 Hz) of itself; the "
                    "field's rotation rate by phase moves by -6.9e-5 (3 Hz) and -2.6e-5 (10 Hz) of itself, and the "
                    "body's by -4.8e-5 and +5.5e-5, for no change in physics: the window is nominal periods of an "
                    "elliptical field whose actual rate is 3.0009 Hz, so the slope of its angle depends on the "
                    "rotor's phase at the window's ends, which the start-up deficit moved; the lag by -5.1e-5 rad "
                    "and +7.3e-6 rad; the speed by +1.8e-5 and -1.4e-6, the tracking error by 3e-6 and 4e-6",
        "check": "test_a0.py gates the rotor speed and the two rotation rates against the prediction",
    },
    {
        "id": "surrogate-about-the-centre",
        "alone": "surrogate-about-the-centre",
        "what": "where the drag surrogate's resistance is referred (MIME-ANO-112, declared by phase 4c3)",
        "cause": "the surrogate's resistance is about the end of the body's axis, as trained; the tag applies it "
                 "about the body's centre; 0.3 shifts it to the centre (T^T R T). The shift couples the body's "
                 "spin to its sideways motion through the lever of half its length, and changes the resistance "
                 "to its swimming. It is the same change that script A declared and measured, here at A0's "
                 "operating point.",
        "expected": "the swim speed up by 10.2% (3 Hz) and 9.2% (10 Hz), and with it the tracking error (the lag "
                    "grows with the speed) down by 10.2% and 9.2% (the filter's lag is in the speed, the speed "
                    "is not in the filter); the phase lag by -5.4e-5 rad (3 Hz) and -5.3e-4 rad (10 Hz); the "
                    "distance's excess over the standoff by +3.5e-4 and -3.4e-4 of itself",
        "check": "test_a0.py gates the speed, the lag and the tracking error against the prediction; "
                 "test_script_a.py bounds the same change on script A",
    },
    {
        "id": "body-step",
        "alone": "body-step",
        "what": "the body's step: the tag's coupling group of six Gauss-Seidel passes against 0.3's implicit solve",
        "cause": "the tag's six passes converge to a backward-Euler step in which the field, the magnet response "
                 "and the resistance read the body's position and orientation at the END of the step, the drag "
                 "uses the new velocity, and the projection removes (R / r)^2 of the outward velocity "
                 "(MIME-ANO-111); 0.3 solves (M/dt + R) dV = F - R V with the field, the torque and the resistance "
                 "at the START of the step, the contact held inside the solve, and all the outward velocity "
                 "removed. The torque therefore reads the orientation one step older in 0.3 (the 'lag "
                 "structure' script A declared and checked by the edges of its graphs).",
        "expected": "the phase lag larger by 0.86 omega dt (8.1e-3 rad at 3 Hz) and 1.98 omega dt (6.2e-2 rad at "
                    "10 Hz); the swim speed -0.15% (3 Hz) and +0.32% (10 Hz), the tracking error by +0.15% and "
                    "-0.32%; the body's rotation rate by -1.9e-5 and -5.4e-6 of itself, the field's by under "
                    "1e-8",
        "check": "test_a0.py gates the lag and the speed against the prediction",
    },
    {
        "id": "vacuum-permeability",
        "alone": "vacuum-permeability",
        "what": "mu_0: the tag's 4 pi 1e-7 against 0.3's CODATA 2022 value",
        "cause": "1.25663706127e-6 against 1.2566370614e-6",
        "expected": "none: a relative 1.35e-10 in the dipole's field; the emulation moves no quantity by more than "
                    "1.2e-12",
        "check": "below every tolerance; the emulation's alone-run is the check",
    },
]
NONE_EXPECTED = [
    {"id": "controller-timing",
     "what": "the controller's tick and read: every step, from the state after the step before, its joint command "
             "acting in that step",
     "reasoning": "the tag's runner called the controller each step with the previous step's state and fed its "
                  "torques to the same step; 0.3's controller inside the compiled step runs first, reads the "
                  "previous-step edges, and its write acts this step (no latency declared)",
     "check": "the build report's lags (test_a0.py asserts the controller reads the body and the joints at the "
              "previous step and writes the arm in this step)"},
    {"id": "controller-law",
     "what": "the target filter (0.005 per tick), the reach clamp, the five damped-least-squares iterations "
             "(damping 0.05, gain 1) towards the held home orientation",
     "reasoning": "ported by 7b1 and held to the 0.2 line's controller by test_port_controller.py; the seed of the "
                  "iterations differs (the tag's actual joints, 0.3's last command), which the arm-dynamics entry "
                  "carries",
     "check": "the tracking error and the distance gate it"},
    {"id": "commands",
     "what": "the rotor's commanded speed, 2 pi f in single precision",
     "reasoning": "the tag's controller built it as a float32 external input; 0.3's float32 run holds the drive's "
                  "value in float32; the same number",
     "check": "the rotor's speed is gated"},
    {"id": "single-precision-numerics",
     "what": "the single-precision accumulators both lines share: the rotor's angle, and the controller's target "
             "filter (which stops 1.3e-6 m short of the standoff)",
     "reasoning": "common to both; the emulation reproduces the reference's with the same rounding (the filter's "
                  "product by 1 - alpha fused into the sum, as XLA does). Were 0.3's XLA to fuse otherwise, the "
                  "distance from the body would move by about 1e-6 m (1e-3 of its excess over the standoff): a "
                  "finding if it does",
     "check": "the distance minus the standoff and the rotation rates are gated"},
    {"id": "frame",
     "what": "the body's axis along x with its moment along y at the identity (0.3's), against the tag's rotated "
             "quaternion and moment along body x",
     "reasoning": "the same physical start; the emulation in both conventions differs by under 5e-6 in every "
                  "quantity",
     "check": "the emulation's other_convention record"},
    {"id": "surrogate-weights",
     "what": "the drag surrogate's weights",
     "reasoning": "pinned to dejongh_mlp_v2 on both sides (sha256 ac1fca5c...0a345). This comparison does not "
                  "validate the surrogate's physics: MIME-ANO-117 records that v2 inherits the old wall tables' "
                  "defect",
     "check": "test_a0.py asserts the asset the drag names"},
]


def declare() -> None:
    """Write ``a0_expected_differences.json`` from the emulation's record."""
    record = json.loads(EMULATION.read_text())
    reference = {key: {name: value for name, value in found.items()} for key, found in record["reference"].items()}
    gaps, misfits, predicted, validation = {}, {}, {}, {}
    for key in FREQUENCIES:
        f32, mixed, tag, new = reference[f"f32_{key}"], reference[f"mixed_f64_{key}"], record["tag"][key], record["new"][key]
        validation[key] = {name: difference(name, tag[name], f32[name]) for name in (*GATED, *NOT_GATED) if name in tag}
        for name in GATED:
            gaps.setdefault(name, []).append(abs(difference(name, f32[name], mixed[name])))
            misfits.setdefault(name, []).append(abs(difference(name, tag[name], f32[name])))
        predicted[key] = {
            name: (f32[name] + (new[name] - tag[name]) if name == LAG else f32[name] * (new[name] / tag[name]))
            for name in GATED}
    floor = {name: 10 * max(found) for name, found in gaps.items()}
    resolution = {name: 3 * max(found) for name, found in misfits.items()}
    tolerances = {name: max(floor[name], resolution[name]) for name in GATED}
    tolerances["rotor_rate_mean_hz"] = 1e-7  # one float32 epsilon: neither the floor nor the misfit measures a rate that is a constant
    changes = []
    for entry in DECLARED_CHANGES:
        shown = {key: {name: difference(name, record["alone"][entry["alone"]][key][name], record["tag"][key][name])
                       for name in GATED} for key in FREQUENCIES}
        changes.append({**{k: v for k, v in entry.items() if k != "alone"}, "predicted_alone": shown})
    together = {key: {name: difference(name, record["new"][key][name], record["tag"][key][name]) for name in GATED}
                for key in FREQUENCIES}
    OUTPUT.write_text(json.dumps({
        "written": "2026-10-30, by predict_a0.py declare, before 0.3 ran A0 against this reference: the only earlier "
                   "runs of A0 in 0.3 were builds and runs of 1 to 2 s that read no steady quantity",
        "comparison": "A0 built through 0.3's public API (tests/script_a0.py: the AR4 carrying a motor and the "
                      "rotor magnet, closed by arm_tracking in the compiled step, the body script A's) in float32, "
                      "against the frozen float32 reference in references/a0/ (the pre-0.3 tag's A0 with its "
                      "drag surrogate given the tube's axis, MADDENING 0.3.1, CPU) at 3 Hz and 10 Hz, by the "
                      "reference's own window rule",
        "how_predicted": "predict_a0.py integrates A0 in float64 (the rotor's angle and the controller's filter in "
                         "the single precision both lines hold them in) with the tag's scheme and with 0.3's, the "
                         "drag model shared. With the tag's scheme it reproduces the reference (validation below); "
                         "each of 0.3's changes is then made alone to the tag's scheme, and all together. The "
                         "predicted values are the float32 reference moved by the difference between the two "
                         "schemes' emulated runs: relative for each quantity, absolute for the lag (rad).",
        "validation_of_the_emulation_against_the_reference": validation,
        "differences": changes,
        "none_expected": NONE_EXPECTED,
        "all_changes_together": together,
        "predicted": predicted,
        "tolerances": {
            "values": tolerances,
            "units": "relative to the predicted value, except the lag (rad)",
            "basis": "each is the larger of two measures that do not look at 0.3: ten times the reference's own "
                     "float32-to-mixed-float64 gap in that quantity (the rounding floor, the larger of the two "
                     "frequencies), and three times the emulation's miss of the reference with the tag's scheme "
                     "(its resolution); the rotor's speed, a constant, at one float32 epsilon. Neither is fitted "
                     "to 0.3's residual.",
            "rounding_floor": floor,
            "emulation_resolution": resolution,
        },
        "reported_not_gated": NOT_GATED,
        "if_a_quantity_misses": "it is a finding, reported with its size, and is not added to the declared "
                                "differences afterwards",
    }, indent=1) + "\n")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    declare() if sys.argv[1:] == ["declare"] else main()
