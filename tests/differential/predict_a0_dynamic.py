"""Predict how A0 on 0.3's dynamic rungs differs from its frozen reference, before 0.3 runs it there:

    JAX_PLATFORMS=cpu .venv/bin/python tests/differential/predict_a0_dynamic.py           # the emulation's record
    JAX_PLATFORMS=cpu .venv/bin/python tests/differential/predict_a0_dynamic.py declare   # the declaration from it

``predict_a0.py`` (phase 7c1's, unchanged) emulates the tag's A0 and 0.3's on its kinematic rungs. This extends its
loop to 0.3's dynamic rungs as their nodes document them and A0's build orders them, and writes
``a0_dynamic_emulation.json`` and, from that, ``a0_dynamic_expected_differences.json``. Like 7c1's it reads only the
reference, the drag surrogate's weights, the AR4 description and the rigid-body library (``mime.core.multibody``, its
forward kinematics, Jacobians, mass matrix and inverse dynamics, held to the 0.2 line's by ``test_port_multibody.py``);
it imports no node and no controller of 0.3. Not collected by pytest.

**0.3's dynamic scheme**, each step in the order A0's dynamic build runs it (its build report: the controller; the
field, the laws and their reactions; the body; the arm; the motor):

- the controller reads the state after the step before. Its joint loop writes ``tau = M (Kp (q_t - q) - Kd q_dot) +
  h(q, q_dot; g)``; its rotor loop reads the rotor's speed relative to its stator (through the motor's frames' rates,
  in single precision) and writes ``tau_d = Kp e + Ki I'`` with ``I' = I + e dt``, in single precision;
- the field is read from the rotor's pose **after the step before** (the reactions' order: each law's reaction is
  delivered on the rotor within the step, so the field, the laws and their reactions run before the motor, and the
  field reads the motor's previous pose; on the kinematic rungs, which take no reaction, the field reads this step's)
  at the body's start-of-step position, and the body steps as 0.3's does (``predict_a0``'s 0.3 line);
- each law's reaction on the rotor, ``[-F; -tau - (c - o) x F]`` about the rotor's origin, ``c`` the body's centre and
  ``o`` the rotor's origin as the step began;
- the arm (semi-implicit Euler): the rotor's momentum on the flange ``h = J Omega e`` (the motor's, a step late) turns
  the joint rates by the Cayley step on ``G = S^T [h]x S``; then ``q_dot' = M^-1 (M q_dot_turned + dt (tau -
  h(q, q_dot_turned; g, W)))``, ``W`` the stator's reaction on the flange (a step late); ``q' = q + dt q_dot'``;
- the motor on the flange's new pose and angular velocity ``omega``: its absolute spin
  ``J Omega' = J Omega + dt (tau_d + e . T_r + b e . omega - b Omega')`` in single precision, its angle turned by
  ``Omega' - e . omega``, its stator's reaction ``[F_r; T_r - (e . T_r) e - (tau_d - b (Omega' - e . omega)) e]``.

**The tag's scheme** is ``predict_a0``'s: the controller's computed-torque joints make each joint accelerate as
``Kp (q_t - q) - Kd q_dot`` exactly (the arm takes no load), and the motor's PI loop runs inside its own step on its
own speed, with no load. With every change below off, this loop is ``predict_a0.emulate(TAG)`` to the bit (checked by
``test_a0_dynamic.py``)."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import predict_a0 as P  # noqa: E402
from a0_runs import A0History  # noqa: E402
from script_a_steady import History  # noqa: E402

A = P.A
EMULATION = HERE / "a0_dynamic_emulation.json"
OUTPUT = HERE / "a0_dynamic_expected_differences.json"
f32 = np.float32
GRAVITY_WORLD = np.array([0.0, 0.0, -9.81])  # the experiment's, which 0.3's arm and its controller both read


@dataclass(frozen=True)
class Dynamic:
    """The tag's scheme (``base``, ``predict_a0``'s ``TAG``) or 0.3's kinematic line (``NEW``) for the body, with
    each of 0.3's dynamic rungs' changes on or off:

    - ``spinner``: the rotor carried by its absolute spin, its stator's drive and bearing reaction on the flange a step
      late, and its momentum's gyroscopic torque on the arm (the rotor's axial inertia loading the arm);
    - ``magnet_reaction``: each law's reaction on the rotor, its part about the axis on the spin and the rest passed
      by the stator to the flange;
    - ``read_through_frames``: the rotor loop reads the speed as the rotor's frame rate less the stator's along the
      axis, in single precision, where the tag's motor read its own state;
    - ``field_late``: the field reads the rotor's pose after the step before;
    - ``dynamic_arm``: the arm by 0.3's dynamic arm step under the joint loop's torque (with no load and no momentum
      it is the tag's ``q_ddot = Kp (q_t - q) - Kd q_dot`` to rounding)."""

    base: P.Scheme
    spinner: bool = False
    magnet_reaction: bool = False
    read_through_frames: bool = False
    field_late: bool = False
    dynamic_arm: bool = False


TAG_D = Dynamic(P.TAG)
#: 0.3's A0 on its dynamic rungs: 0.3's body line and surrogate, and every change of the dynamic rungs.
NEW_D = Dynamic(replace(P.NEW, arm="dynamic", motor="pi"), spinner=True, magnet_reaction=True,
                read_through_frames=True, field_late=True, dynamic_arm=True)

_dynamics: list[Any] = []


def dynamics() -> Any:
    """The jitted arm step of 0.3's dynamic rung under the joint loop's command, in float64, and the flange's pose and
    angular velocity."""
    if _dynamics:
        return _dynamics[0]
    import jax
    import jax.numpy as jnp

    from mime.core.multibody import cholesky_solve, frame_angular_jacobians, frame_forces, frame_jacobian, \
        frame_pose, inverse_dynamics, mass_matrix, read_urdf

    tree = read_urdf(P.URDF).tree
    flange = tree.articulation.frame_index("link_6")
    frames = len(tree.articulation.frame_names)
    gravity = jnp.asarray(GRAVITY_WORLD)

    def skew(v: Any) -> Any:
        return jnp.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])

    def step(q: Any, qd: Any, accelerations: Any, momentum: Any, wrench: Any, dt: float) -> Any:
        inertia = mass_matrix(tree, q)
        across = frame_angular_jacobians(tree, q)[flange]
        half = 0.5 * dt * (across.T @ skew(momentum) @ across)
        turned = jnp.linalg.solve(inertia - half, (inertia + half) @ qd)
        command = inertia @ accelerations + inverse_dynamics(tree, q, qd, None, gravity)
        loads = frame_forces(tree, q, jnp.zeros((frames, 6)).at[flange].set(wrench))
        bias = inverse_dynamics(tree, q, turned, None, gravity, loads)
        rates = cholesky_solve(jnp.linalg.cholesky(inertia), inertia @ turned + dt * (command - bias))
        moved = q + dt * rates
        return moved, rates, frame_pose(tree, moved, "link_6"), frame_jacobian(tree, moved, "link_6")[3:] @ rates

    _dynamics.append(jax.jit(step, static_argnums=5))
    return _dynamics[0]


def _rotation(q: np.ndarray) -> np.ndarray:
    return A._rotation(np.asarray(q, dtype=np.float64))


def read_rate(spin: Any, relative: Any, axis: np.ndarray, omega: np.ndarray) -> Any:
    """What the rotor loop reads in single precision: the rotor's frame rate (the flange's turning and the relative
    speed along the axis, as the motor publishes it) less the stator's, along the axis."""
    axis32, omega32 = axis.astype(f32), omega.astype(f32)
    rotor = (omega32 + f32(relative) * axis32).astype(f32)
    difference = (rotor - omega32).astype(f32)
    return f32(f32(f32(difference[0] * axis32[0]) + f32(difference[1] * axis32[1])) + f32(difference[2] * axis32[2]))


def emulate(scheme: Dynamic, frequency: float, *, dt: float = P.DT, duration: float = 10.0,
            angle_at: tuple[int, float] | None = None,
            trace: dict[str, list[float]] | None = None) -> tuple[A0History, dict[str, Any], np.ndarray]:
    """A0 for ``duration`` seconds in float64 (the rotor's state, its loop and the controller's filter in single
    precision): its history as the reference's, what the run shows of the arm's motion (the joints' largest speeds,
    their least margin to their limits, joint 6's largest excursion), and the rotor's pose at every tenth step.
    ``angle_at`` (a step and an angle) sets the rotor's single-precision angle at the end of that step, for the
    finding's control (``measure_a0_dynamic_findings.py``); the declaration's runs set nothing. ``trace``, when given,
    receives at every step the rotor loop's integral and the single-precision increment its angle took (the second
    declaration's, ``predict_a0_dynamic_b.py``); it changes nothing the run computes."""
    base = scheme.base
    steps = round(duration / dt)
    ik_joints, flange, home_position, _ = P.arm()
    arm_step = dynamics() if scheme.dynamic_arm else None
    omega_command = P.command(frequency)
    surrogate_end: dict[float, np.ndarray] = {}
    moment_body = P.BODY_MOMENT * np.asarray(base.body_moment) / np.linalg.norm(base.body_moment)
    motor = P.MOTOR
    inertia32, damping32 = f32(motor["inertia_kg_m2"]), f32(motor["damping_n_m_s"])
    kp32, ki32, dt32 = f32(motor["velocity_kp"]), f32(motor["velocity_ki"]), f32(dt)

    def resistance(position: np.ndarray, with_lubrication: bool = False) -> np.ndarray:
        offset = P.offset_of(position)
        if offset not in surrogate_end:
            surrogate_end[offset] = A.surrogate(offset, base.reference_point)
        found = P.in_world(surrogate_end[offset], position)
        return found + P.in_world(A.lubrication(offset), position) if with_lubrication else found

    position = np.array([0.0, 0.0, -1e-3])
    orientation = np.array(base.body_orientation, dtype=np.float64)
    orientation /= np.linalg.norm(orientation)
    velocity, spin = np.zeros(3), np.zeros(3)
    joints, joint_rates = P.HOME.copy(), np.zeros(6)
    controller = P.Controller(home_position.copy())
    angle32, rotor_rate32, integral = f32(0.0), f32(0.0), f32(0.0)
    spin32, read32 = f32(0.0), f32(0.0)
    rotor_rate = 0.0
    drag = np.zeros(6)
    on_flange, momentum, on_rotor = np.zeros(6), np.zeros(3), np.zeros(6)
    pose = np.asarray(flange(joints))
    parent_position, parent_rotation = pose[:3] + P.BASE, _rotation(pose[3:])
    flange_rate = np.zeros(3)
    rows = np.zeros((steps + 1, 16 + 3 + 2 + 6))
    rows[0, :3], rows[0, 3:7] = position, orientation
    rows[0, 16:19] = parent_position
    rows[0, 21:] = joints
    fastest, margin, excursion = np.zeros(6), np.inf, 0.0
    mass_matrix = np.diag([P.MASS] * 3 + [P.INERTIA] * 3)

    def magnet_now(rotation: np.ndarray, angle: float) -> np.ndarray:
        return P.ROTOR_MOMENT * rotation @ np.array([math.cos(angle), math.sin(angle), 0.0])

    def body_step(rotor_moment: np.ndarray, magnet: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The body's step, on the tag's line or 0.3's; the field at the body, and the force and torque on it."""
        nonlocal position, orientation, velocity, spin, drag
        if base.line == "tag":
            latest_p, latest_q, latest_v, latest_w = position, orientation, velocity, spin
            for _ in range(base.passes):
                field, gradient = P.dipole(rotor_moment, magnet, latest_p, base.mu0)
                moment_world = A._rotation(latest_q) @ moment_body
                magnetic_torque, magnetic_force = np.cross(moment_world, field), gradient @ moment_world
                resistance_here = resistance(latest_p)
                new_spin = spin + dt * (magnetic_torque + drag[3:]) / P.INERTIA
                new_velocity = velocity + dt * (P.GRAVITY_FORCE + magnetic_force + drag[:3]) / P.MASS
                new_position, new_velocity = P.project_tag(position + dt * new_velocity, new_velocity)
                new_orientation = A._multiply(A._exponential(dt * new_spin), orientation)
                new_orientation /= np.linalg.norm(new_orientation)
                corrected = resistance_here + P.in_world(A.lubrication(P.offset_of(new_position)), new_position)
                drag = -corrected @ np.concatenate([new_velocity, new_spin])
                latest_p, latest_q, latest_v, latest_w = new_position, new_orientation, new_velocity, new_spin
            position, orientation, velocity, spin = latest_p, latest_q, latest_v, latest_w
            return field, magnetic_force, magnetic_torque
        field, gradient = P.dipole(rotor_moment, magnet, position, base.mu0)
        moment_world = A._rotation(orientation) @ moment_body
        torque, force = np.cross(moment_world, field), gradient @ moment_world
        here = resistance(position, with_lubrication=True)
        twist = np.concatenate([velocity, spin])
        system = mass_matrix / dt + here
        applied = np.concatenate([P.GRAVITY_FORCE + force, torque])
        moved = twist + np.linalg.solve(system, applied - here @ twist)
        normal = P.turn(position)[:3, 0]
        offset = P.offset_of(position)
        on_the_wall = (offset - P.TUBE) + P.EXTENT + P.MARGIN >= -P.CONTACT_ULPS * np.finfo(np.float64).eps * (
            float(np.linalg.norm(position)) + P.EXTENT + P.MARGIN)
        if on_the_wall and moved[:3] @ normal > 0:
            direction = np.concatenate([normal, np.zeros(3)])
            response = np.linalg.solve(system, direction)
            moved = moved - (direction @ moved) / (direction @ response) * response
        position, velocity = P.project_0_3(position + dt * moved[:3], moved[:3])
        spin = moved[3:]
        orientation = A._multiply(A._exponential(dt * spin), orientation)
        orientation /= np.linalg.norm(orientation)
        return field, force, torque

    def reaction(force: np.ndarray, torque: np.ndarray, centre: np.ndarray, origin: np.ndarray) -> np.ndarray:
        return np.concatenate([-force, -torque - np.cross(centre - origin, force)])

    field = np.zeros(3)
    for step in range(steps):
        # the controller reads the state after the step before
        target_joints = controller.tick(position[0], joints)
        accelerations = P.K_P * (target_joints - joints) - P.K_D * joint_rates
        drive = f32(0.0)
        if scheme.spinner:
            error = f32(omega_command) - (read32 if scheme.read_through_frames else rotor_rate32)
            integral = f32(integral + error * dt32)
            drive = f32(kp32 * error + ki32 * integral)
        started_at, angle_before = parent_position, float(angle32)
        if scheme.field_late:
            centre = position.copy()
            field, force, torque = body_step(magnet_now(parent_rotation, angle_before), parent_position)
            on_rotor = reaction(force, torque, centre, started_at) if scheme.magnet_reaction else np.zeros(6)
        # the arm
        if arm_step is not None:
            moved, rates, pose_now, flange_rate = (np.asarray(found) for found in arm_step(
                joints, joint_rates, accelerations, momentum if scheme.spinner else np.zeros(3),
                on_flange, dt))
            joints, joint_rates = moved, rates
        else:
            joint_rates = joint_rates + accelerations * dt
            joints = np.clip(joints + joint_rates * dt, P.LIMITS[:, 0], P.LIMITS[:, 1])
            if scheme.magnet_reaction and not scheme.spinner:
                raise ValueError("the magnet's reaction reaches the arm through 0.3's dynamic arm only")
            pose_now = np.asarray(flange(joints))
        fastest = np.maximum(fastest, np.abs(joint_rates))
        margin = min(margin, float(np.min(np.minimum(joints - P.LIMITS[:, 0], P.LIMITS[:, 1] - joints))))
        excursion = max(excursion, abs(float(joints[5] - P.HOME[5])))
        parent_position, parent_rotation = pose_now[:3] + P.BASE, _rotation(pose_now[3:])
        # the motor
        if scheme.spinner:
            axis = parent_rotation @ np.array([0.0, 0.0, 1.0])
            along = f32(axis @ flange_rate)
            loads = f32(axis @ on_rotor[3:]) if scheme.magnet_reaction else f32(0.0)
            torque_sum = f32(drive + loads)
            spin32 = f32(f32(f32(inertia32 * spin32) + f32(dt32 * f32(torque_sum + f32(damping32 * along))))
                         / f32(inertia32 + f32(dt32 * damping32)))
            relative32 = f32(spin32 - along)
            increment = f32(dt32 * relative32)
            angle32 = f32(angle32 + increment)
            # the motor's relative speed as the comparison reads it from 0.3's state: its spin less the flange's
            # turning along the axis, in double precision from the single-precision state
            rotor_rate = float(spin32) - float(axis @ flange_rate)
            rotor_rate32 = relative32
            read32 = read_rate(spin32, relative32, axis, flange_rate)
            held = float(drive) - float(motor["damping_n_m_s"]) * float(relative32)
            across = on_rotor[3:] - (axis @ on_rotor[3:]) * axis
            on_flange = np.concatenate([on_rotor[:3], across - held * axis])
            momentum = float(motor["inertia_kg_m2"]) * float(spin32) * axis
        else:
            error = f32(omega_command) - rotor_rate32
            integral = f32(integral + error * dt32)
            torque32 = f32(f32(motor["velocity_kp"]) * error + f32(motor["velocity_ki"]) * integral)
            rotor_rate32 = f32(f32(f32(motor["inertia_kg_m2"]) * rotor_rate32 + dt32 * torque32) / f32(
                f32(motor["inertia_kg_m2"]) + dt32 * f32(motor["damping_n_m_s"]) + f32(1e-30)))
            rotor_rate = float(rotor_rate32)
            increment = f32(rotor_rate32 * dt32)
            angle32 = f32(angle32 + increment)
        if trace is not None:
            trace.setdefault("integral", []).append(float(integral))
            trace.setdefault("increment", []).append(float(increment))
        if angle_at is not None and step + 1 == angle_at[0]:
            angle32 = f32(angle_at[1])
        angle = float(angle32)
        if not scheme.field_late:
            centre = position.copy()
            field, force, torque = body_step(magnet_now(parent_rotation, angle), parent_position)
            if scheme.magnet_reaction:  # on the tag's order, the reaction reaches the rotor on the next step
                on_rotor = reaction(force, torque, centre, parent_position)
        rows[step + 1] = np.concatenate([position, orientation, velocity, spin, field, parent_position,
                                         [angle, rotor_rate], joints])
    time = np.arange(steps + 1) * dt
    history = History(time, rows[:, :3], rows[:, 7:10], rows[:, 3:7], rows[:, 10:13], rows[:, 13:16])
    motion = {"joint_speed_max_rad_s": [float(v) for v in fastest], "joint_limit_margin_min_rad": margin,
              "joint_6_excursion_max_rad": excursion}
    return A0History(history, rows[:, 16:19], rows[:, 20], rows[:, 19]), motion, rotor_poses(rows[::EVERY])


#: The reference's arrays are kept at every tenth step; so are the emulation's rotor poses.
EVERY = 10


def rotor_poses(rows: np.ndarray) -> np.ndarray:
    """The rotor's world pose at each row of an emulation's history: the flange's pose at the row's joints, turned by
    the rotor's angle about the flange's z (the motor sits on the flange's origin, its axis the flange's z)."""
    _, flange, _, _ = P.arm()
    found = np.zeros((len(rows), 7))
    for index, row in enumerate(rows):
        pose = np.asarray(flange(row[21:27]))
        half = 0.5 * float(row[19])
        found[index, :3] = pose[:3] + P.BASE
        found[index, 3:] = A._multiply(pose[3:], np.array([math.cos(half), 0.0, 0.0, math.sin(half)]))
    return found


def steady(history: A0History, frequency: float, scheme: Dynamic) -> dict[str, float]:
    return P.steady(history, frequency, scheme.base)


GATED = P.GATED
MOTOR_OUTPUT = "rotor_angle_rate_mean_hz"
LAG = P.LAG
FREQUENCIES = P.FREQUENCIES
#: Each of the dynamic rungs' changes made alone to the tag's scheme, by the name the declaration gives it.
CHANGES = {
    "rotor-axial-inertia": dict(spinner=True, dynamic_arm=True),
    "magnet-reaction": dict(spinner=True, dynamic_arm=True, magnet_reaction=True),
    "rotor-loop-read": dict(spinner=True, dynamic_arm=True, read_through_frames=True),
    "field-reads-the-rotor-late": dict(field_late=True),
    "dynamic-arm-step": dict(dynamic_arm=True),
}


POSES = HERE / "a0_dynamic_poses.npz"


def main() -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    record: dict[str, Any] = {"tag": {}, "new": {}, "alone": {name: {} for name in CHANGES}, "motion": {}}
    poses: dict[str, np.ndarray] = {}
    for key, frequency in FREQUENCIES.items():
        print(f"{key}: the tag's scheme, and 0.3's dynamic ...", flush=True)
        tag, motion, poses[f"{key}_tag"] = emulate(TAG_D, frequency)
        record["tag"][key] = steady(tag, frequency, TAG_D)
        record["motion"][key] = {"tag": motion}
        new, motion, poses[f"{key}_new"] = emulate(NEW_D, frequency)
        record["new"][key] = steady(new, frequency, NEW_D)
        record["motion"][key]["new"] = motion
        for name, change in CHANGES.items():
            print(f"{key}: {name} alone ...", flush=True)
            changed = replace(TAG_D, **change)
            found, motion, _ = emulate(changed, frequency)
            record["alone"][name][key] = steady(found, frequency, changed)
            record["motion"][key][name] = motion
    EMULATION.write_text(json.dumps(record, indent=1) + "\n")
    np.savez_compressed(POSES, **poses)
    print(f"wrote {EMULATION.name} and {POSES.name}")



# -- the declaration -----------------------------------------------------------------------------------------------------

DECLARED_7C1 = HERE / "a0_expected_differences.json"
SHORT = {"swim_speed_axial_m_s": "speed", "swim_speed_axial_mean_velocity_m_s": "speed (mean velocity)",
         "body_rotation_rate_hz": "body's rotation rate", "field_rotation_rate_hz": "field's rotation rate",
         "phase_lag_field_minus_moment_rad": "lag (rad)", "tracking_error_axial_mean_m": "tracking error",
         "tracking_error_axial_rms_m": "tracking error (rms)",
         "magnet_body_distance_minus_standoff_mean_m": "distance's excess over the standoff",
         "rotor_rate_mean_hz": "rotor's speed", MOTOR_OUTPUT: "motor's angle rate"}
GATED_D = (*GATED, MOTOR_OUTPUT)
#: 7c1's differences that still apply on the dynamic rungs, by their id in its declaration.
STILL = ("surrogate-about-the-centre", "body-step", "vacuum-permeability")
#: 7c1's differences the dynamic rungs close, and what is left of each, by the ids below.
CLOSED = {
    "arm-dynamics": "closed: the dynamic arm under the tag's computed-torque joint loop (Kp 1e4, Kd 200) trails its "
                    "joint targets by the tag's 20 ms, and arm_tracking now seeds its inverse kinematics from the "
                    "arm's joints as the tag's did; what is left is the arm's step (dynamic-arm-step) and what the "
                    "rotor puts on the arm (rotor-axial-inertia, magnet-reaction)",
    "rotor-motor": "closed: the dynamic motor under the tag's PI velocity loop (J 1e-5, Kp 1e-3, Ki 1e-2, b 1e-4) "
                   "starts from rest and trails the command's integral by b omega / Ki, as the tag's did; what is left "
                   "is the loop's read of the speed (rotor-loop-read) and the rotor's spin carried absolutely "
                   "(rotor-axial-inertia)",
}


#: The changes made alone on another, which they need (the reactions and the loop's read need the rotor's spin carried
#: absolutely): each one's predicted moves are what it adds to that.
RIDES_ON = {"magnet-reaction": "rotor-axial-inertia", "rotor-loop-read": "rotor-axial-inertia"}
#: What 7c1's entries declared none say on the dynamic rungs, where it differs.
CARRIED = {
    "controller-timing": "both loops tick every step, read the state after the step before (the joints, their rates, "
                         "the motor's frames) and write the arm's torques and the rotor's torque in this step, as the "
                         "tag's controller and its motor's PI did",
    "controller-law": "arm_tracking's inverse kinematics now starts from the arm's joints as it reads them, the tag's "
                      "seed too, so the seed no longer differs; the joint loop writes M (Kp e - Kd q_dot) + h by one "
                      "inverse-dynamics pass where the tag wrote M qdd + (bias - gravity) and its arm added the "
                      "gravity: the same torque to single-precision rounding",
    "single-precision-numerics": "add the rotor loop's integral, a single-precision sum in both lines whose step falls "
                                 "below half a unit in its last place near the command (its dead band, which leaves "
                                 "the tag's rotor 5e-7 of its speed slow), emulated in single precision",
}


def _differences(found: dict[str, float], against: dict[str, float]) -> dict[str, float]:
    return {name: P.difference(name, found[name], against[name]) for name in GATED_D
            if name in found and name in against}


def _sizes(shown: dict[str, dict[str, float]], tolerances: dict[str, float], least: float = 0.0) -> str:
    """The predicted moves, at each frequency, of the quantities moved by more than ``least`` of their tolerance."""
    parts = []
    for key in FREQUENCIES:
        moved = [f"{SHORT[name]} {value:+.2g}" for name, value in shown[key].items()
                 if abs(value) > least * tolerances[name]]
        parts.append(f"{key.replace('hz', ' Hz')}: " + (", ".join(moved) if moved else "nothing above "
                                                         f"{least:g} of its tolerance"))
    return "; ".join(parts)


def _tolerance_rows(record: dict[str, Any], declared: dict[str, Any]) -> dict[str, Any]:
    values = dict(declared["tolerances"]["values"])
    read = max(abs(record["alone"]["rotor-loop-read"][key]["rotor_rate_mean_hz"]
                   - record["alone"]["rotor-axial-inertia"][key]["rotor_rate_mean_hz"])
               / abs(record["alone"]["rotor-axial-inertia"][key]["rotor_rate_mean_hz"]) for key in FREQUENCIES)
    values["rotor_rate_mean_hz"] = max(1e-7, 3 * read)
    values[MOTOR_OUTPUT] = 1e-7
    return {"values": values, "read_rounding_on_the_rotor_speed": read}


def _pose_tolerances(poses: Any) -> dict[str, Any]:
    from a0_poses import MISSES, difference, misses

    zero = None
    floors: dict[str, list[float]] = {name: [] for name in MISSES}
    resolutions: dict[str, list[float]] = {name: [] for name in MISSES}
    for key, frequency in FREQUENCIES.items():
        found = np.load(HERE / "references" / "a0" / f"f32_{key}_trajectory_every_10th_step.npz")
        mixed = np.load(HERE / "references" / "a0" / f"mixed_f64_{key}_rotor_every_10th_step.npz")
        time, reference = found["time_s"], found["state__motor__rotor_pose_world"].astype(np.float64)
        zero = {name: np.zeros_like(value) for name, value in difference(reference, reference).items()}
        for name, value in misses(mixed["state__motor__rotor_pose_world"], reference, zero, time,
                                  frequency).items():
            floors[name].append(10 * value)
        for name, value in misses(poses[f"{key}_tag"], reference, zero, time, frequency).items():
            resolutions[name].append(3 * value)
    return {"values": {name: max(max(floors[name]), max(resolutions[name])) for name in MISSES},
            "rounding_floor": {name: max(found) for name, found in floors.items()},
            "emulation_resolution": {name: max(found) for name, found in resolutions.items()},
            "units": dict(MISSES)}


def _transient_tolerance() -> float:
    """Ten times float32's unit roundoff times the condition number of the arm's mass matrix at its home pose: the
    relative rounding a single-precision solve of the arm's step can leave in joint 6's motion, which this emulation in
    float64 does not carry."""
    import jax.numpy as jnp

    from mime.core.multibody import mass_matrix, read_urdf

    inertia = np.asarray(mass_matrix(read_urdf(P.URDF).tree, jnp.asarray(P.HOME)))
    return float(10 * np.finfo(np.float32).eps / 2 * np.linalg.cond(inertia))


def declare() -> None:
    """Write ``a0_dynamic_expected_differences.json`` from the emulation's record."""
    import jax

    jax.config.update("jax_enable_x64", True)
    record = json.loads(EMULATION.read_text())
    seven = json.loads(DECLARED_7C1.read_text())
    poses = np.load(POSES)
    reference = {key: json.loads((P.REFERENCES / f"f32_{key}_steady.json").read_text()) for key in FREQUENCIES}
    tolerances = _tolerance_rows(record, seven)
    values = tolerances["values"]
    alone = {name: {key: _differences(record["alone"][name][key], record["tag"][key]) for key in FREQUENCIES}
             for name in CHANGES}
    together = {key: _differences(record["new"][key], record["tag"][key]) for key in FREQUENCIES}
    predicted = {key: {name: (reference[key][name] + together[key][name] if name == LAG else
                              reference[key][name] * (record["new"][key][name] / record["tag"][key][name]))
                       for name in GATED} for key in FREQUENCIES}
    for key in FREQUENCIES:
        predicted[key][MOTOR_OUTPUT] = record["new"][key][MOTOR_OUTPUT]
    motion = record["motion"]
    kick = {key: {"joint_6_excursion_max_rad": motion[key]["new"]["joint_6_excursion_max_rad"],
                  "joint_6_speed_max_rad_s": motion[key]["new"]["joint_speed_max_rad_s"][5]} for key in FREQUENCIES}
    transient_tolerance = _transient_tolerance()
    margin = min(motion[key]["new"]["joint_limit_margin_min_rad"] for key in FREQUENCIES)
    by_id = {entry["id"]: entry for entry in seven["differences"]}
    changes = [{**{k: v for k, v in by_id[name].items() if k != "predicted_alone"},
                "predicted_alone": {key: {n: v for n, v in by_id[name]["predicted_alone"][key].items()}
                                    for key in FREQUENCIES},
                "from": "phase 7c1's declaration, unchanged: the same change made alone to the same tag's scheme"}
               for name in STILL]
    new_entries = [
        {"id": "rotor-axial-inertia",
         "what": "the rotor's inertia about its axis on the arm (7a3's review, Q-5, changed by 7a4): 0.3's dynamic motor "
                 "carries the rotor's absolute spin, its stator passes the drive's and the bearing's reaction to the "
                 "flange a step late, and the arm applies the rotor's momentum's gyroscopic torque by its Cayley step; "
                 "0.2's motor turned its rotor relative to its stator and loaded nothing",
         "cause": "the rotor's J = 1e-5 kg m^2 is 4.1 times link 6's own inertia about joint 6's axis (izz 2.43e-6 kg "
                  "m^2), and the motor's axis is joint 6's: as the PI loop spins the rotor up from rest, the drive's "
                  "reaction turns the wrist backwards, and the joint loop, whose model of the arm (the built tree, "
                  "the bare description here: nothing is grafted on A0 as declared) does not hold the rotor, pulls it "
                  "back with joint 6's stiffness M_66 Kp. Once the rotor turns steadily, its drive balances its "
                  "bearing and the reaction falls to the magnet's load about the axis; the gyroscopic torque is the "
                  "rotor's momentum (6.3e-4 N m s at 10 Hz) across the flange's turning, which the tracking keeps "
                  "small",
         "expected": f"in the steady window, {_sizes(alone['rotor-axial-inertia'], values)}; at the start, joint 6 "
                     f"turns by up to {kick['3hz']['joint_6_excursion_max_rad']:.4g} rad (3 Hz) and "
                     f"{kick['10hz']['joint_6_excursion_max_rad']:.4g} rad (10 Hz) from its home and back within "
                     f"about 0.2 s, at up to {kick['3hz']['joint_6_speed_max_rad_s']:.4g} and "
                     f"{kick['10hz']['joint_6_speed_max_rad_s']:.4g} rad/s, where the description's velocity limit "
                     f"is 1.5708 rad/s (the dynamic arm enforces no speed limit)",
         "check": "test_a0_dynamic.py gates the steady quantities against the prediction, and joint 6's largest "
                  "excursion and speed against the prediction's within the transient tolerance",
         "alone": "rotor-axial-inertia"},
        {"id": "magnet-reaction",
         "what": "the reactions now delivered: each magnetic law's reaction on the rotor, equal and opposite and moved "
                 "to its origin, its part about the axis on the spin (which the PI loop then carries) and the rest "
                 "passed by the stator to the flange; 0.2 delivered none (A0's 0.2 graph fed its motor's load_torque "
                 "nothing)",
         "cause": "the magnetic laws' torque and force on the body, reacting on the rotor 0.15 m above it",
         "expected": _sizes({key: _differences(record["alone"]["magnet-reaction"][key],
                                               record["alone"]["rotor-axial-inertia"][key]) for key in FREQUENCIES},
                            values) + " (against the rotor's axial inertia alone, which it rides on)",
         "check": "the gates that see it (seen_by); the reactions' own oracles are 7a3's (equal and opposite, the "
                  "two dipoles' closed form)",
         "alone": "magnet-reaction"},
        {"id": "rotor-loop-read",
         "what": "the motor's PI as a controller: 0.2's ran inside the motor's step on its own speed state; 0.3's "
                 "rotor loop reads the rotor's speed relative to its stator through the motor's frames (the rotor's "
                 "rate less the stator's, along the axis from the stator's pose) in single precision",
         "cause": "the read rounds where the motor's own state did not: the rotor's frame rate is the flange's "
                  "turning plus the speed along the axis, summed and differenced in float32, and taken along an axis "
                  "of float32 length; the loop then holds what it reads at the command within its own single-"
                  "precision dead band (the integral's step falls below half a unit in its last place), as the tag's "
                  "did",
         "expected": _sizes({key: _differences(record["alone"]["rotor-loop-read"][key],
                                               record["alone"]["rotor-axial-inertia"][key]) for key in FREQUENCIES},
                            values) + " (against the rotor's axial inertia alone, which it rides on)",
         "check": "the rotor's speed is gated, at a tolerance of three times this rounding's effect where that "
                  "exceeds one float32 epsilon (the emulation's read is float32 as 0.3's is, but its rounding "
                  "pattern is not 0.3's bit for bit). The emulation rounds each operation of the motor's spin "
                  "update once, as written; were 0.3's XLA to fuse its multiply into its add, the loop's dead band "
                  "would settle elsewhere within its width (about 5e-7 of the speed, the tag's): a finding if it "
                  "does",
         "alone": "rotor-loop-read"},
        {"id": "field-reads-the-rotor-late",
         "what": "the field reads the rotor's pose after the step before on the dynamic rungs, where the tag's and "
                 "0.3's kinematic A0 read this step's",
         "cause": "found by reading A0's dynamic build report before any run: each magnetic law's reaction reaches "
                  "the rotor within the step (the reaction nodes' links into rpm.motor are this step's), so the "
                  "field, the laws and their reactions are ordered before the motor, and the field's edge from "
                  "rpm.motor's frame poses is the previous step's; the kinematic rungs take no reaction and keep the "
                  "motor before the field. Not on 7a3's list",
         "expected": _sizes(alone["field-reads-the-rotor-late"], values),
         "check": "test_a0_dynamic.py asserts the edge's lag from the build report, and the gates see what it moves",
         "alone": "field-reads-the-rotor-late"},
        {"id": "dynamic-arm-step",
         "what": "the arm's step: 0.3's dynamic arm (the mass matrix by CRBA, the bias by RNEA, a Cholesky solve, the "
                 "stops' sweeps) under the joint loop's command, against the tag's arm, whose joints the command made "
                 "accelerate as Kp (q_t - q) - Kd q_dot",
         "cause": "the command cancels the arm's own dynamics in both, so with no load and no momentum the two steps "
                  "are one to rounding (jnp.linalg.solve in 0.2, Cholesky in 0.3; the AR4 declares no damping and "
                  "no friction, its base is unturned so gravity's frame is the world's in both, and no joint reaches "
                  f"a stop: the least margin to a limit over either run is {margin:.3g} rad)",
         "expected": _sizes(alone["dynamic-arm-step"], values),
         "check": "below every gate's tolerance; the joints' least margin to their limits is checked on 0.3's runs",
         "alone": "dynamic-arm-step"},
    ]
    for entry in new_entries:
        shown = alone[entry["alone"]]
        extra: dict[str, Any] = {}
        if entry["alone"] in RIDES_ON:
            # made alone on the rotor's spin carried absolutely, which it needs: what it adds to that
            shown = {key: _differences(record["alone"][entry["alone"]][key], record["alone"][RIDES_ON[entry["alone"]]][key])
                     for key in FREQUENCIES}
            extra = {"predicted_alone_with_rotor_axial_inertia": alone[entry["alone"]]}
        seen = sorted({name for key in FREQUENCIES for name, value in shown[key].items() if abs(value) > values[name]})
        changes.append({**{k: v for k, v in entry.items() if k != "alone"}, "predicted_alone": shown, **extra,
                        "seen_by": seen or "no gate: every move it predicts is below its quantity's tolerance"})
    none_expected = [
        {"id": "gravity-compensation-timing",
         "what": "the weight's share of the joint torque (7a3's review, Q-6): the 0.2 arm added it inside its own step "
                 "at that step's q, every physics step; 0.3's joint loop writes it in the command, at the "
                 "controller's rate, with its latency",
         "reasoning": "A0's controller ticks every step with no latency and reads the joints after the step before, "
                      "which are the q the arm's step starts from: the same q and the same step, so the same torque "
                      "to rounding (the emulation's arm step computes the command and the bias at one q). At another "
                      "rate or with a latency the two would differ by the weight's change over the held steps",
         "check": "the build's controller rate, latency and lags (test_a0_dynamic.py)"},
        {"id": "the-arm's-gravity",
         "what": "the 0.2 arm's gravity was 9.80665 m/s^2 and its body's 9.81; 0.3's experiment has one, 9.81",
         "reasoning": "the joint loop reads the experiment's gravity (ctx.gravity) and the arm's built tree, so the "
                      "weight it compensates is the weight the arm feels, whatever its value: it cancels in both "
                      "lines",
         "check": "the joint loop's gravity term has its own control (its share planted)"},
        {"id": "rotor-loop-timing-and-form",
         "what": "the PI loop's order (7a3's review, Q-6): the 0.2 motor read its speed before its step, advanced its "
                 "integral before using it (I' = I + e dt, tau = Kp e + Ki I'), froze it outside velocity mode, "
                 "coasted on a command of exactly zero, and added 1e-30 to its denominator with max(b, 0)",
         "reasoning": "the rotor loop reads the speed at the start of the step and advances its integral before using "
                      "it, every step; the command is never zero and there is one mode, so the freeze and the coast "
                      "are never reached; 1e-30 is below float32's resolution of J + dt b (1.005e-5) and b > 0",
         "check": "the emulation of 0.3's loop runs the same order; the rotor's speed and angle are gated"},
        {"id": "damping-and-friction",
         "what": "joint damping (0.2 explicit, 0.3 implicit) and Coulomb friction (0.2 dropped, 0.3 impulses)",
         "reasoning": "the AR4 declares neither: both are zero in both lines",
         "check": "test_a0_dynamic.py asserts the built arm's damping and friction are empty"},
        {"id": "stops",
         "what": "0.2 clipped q and kept q_dot; 0.3's stops are inelastic impulses",
         "reasoning": f"no joint reaches a limit in either run (the emulation's least margin {margin:.3g} rad)",
         "check": "test_a0_dynamic.py asserts 0.3's joints stay inside their limits"},
        {"id": "link-loads",
         "what": "0.2's link loads were world wrenches at each link's inertial origin, of which A0 fed none; 0.3's are "
                 "per frame about its origin",
         "reasoning": "A0's only loads on the arm are the motor's reaction and the magnet's through it, declared as "
                      "rotor-axial-inertia and magnet-reaction",
         "check": "those entries'"},
        {"id": "motor-mount",
         "what": "where the rotor sits: the tag's motor on the flange's origin with no tool offset, its axis the "
                 "flange's z; 0.3's on link_6 at its origin, axis Z",
         "reasoning": "the same pose",
         "check": "the magnet's pose through its mount, compared with the reference's recorded pose"},
        *[{**entry, "from": "phase 7c1's declaration", **({"on_the_dynamic_rungs": CARRIED[entry["id"]]}
                                                           if entry["id"] in CARRIED else {})}
          for entry in seven["none_expected"]],
    ]
    swap = {
        "configurations": [{"rate": "every step (0.5 ms)", "latency_steps": 0},
                           {"rate": "every step (0.5 ms)", "latency_steps": 1}],
        "run": "A0 on its dynamic rungs as declared, 3 Hz, float32, 1000 steps (0.5 s: the rotor's spin-up and the "
               "arm's kick within it), the same controller (A0Loops) placed inside the compiled step (mode 1) and as a "
               "callback (mode 2)",
        "compared": "every field of every node the two graphs share",
        "bound": "8 single-precision units in the last place per step run, of each field's largest magnitude over the "
                 "callback's run: steps x 8 x 2^-23 x max |field|",
        "basis": "what mode 2 changes, measured by 7b2: the schedule is not one of them (the same writes act at the "
                 "same steps, bit for bit for a law with integer writes, over 28 periods and latencies), and the "
                 "command is held per control period in both (the multi-rate schedule's hold, the chunk); what "
                 "changes is where the step is compiled, inside the scan or on its own on the host, so XLA may fuse "
                 "its arithmetic otherwise (a multiply into an add), a few units in the last place a step, carried "
                 "linearly through a stable loop: 7b2's bound, which A0's kinematic rig met bit for bit but the "
                 "motor's angle (1.9e-9 rad against 2.3e-4)",
        "lag": "mode 1 adds no base step of lag: 7b2 measured none, and the order that would add one is refused "
               "(post-build.controller-order). The brief's 'one base step of lag in mode 1' is the code "
               "architecture's text that 7b2 superseded; the tolerance therefore carries no lag, and one step of it "
               "must fail: the controls plant the latency removed (1 to 0 steps) and doubled (1 to 2) in mode 1",
    }
    rungs = {
        "what": "A0 on its kinematic rungs (7c1's: arm_tracking writing the kinematic arm's joint positions, the "
                "motor's speed prescribed) against A0 on its dynamic rungs (A0Loops), both 0.3, float32, 10 s, at "
                "1, 2, 3, 5, 7 and 10 Hz",
        "per_quantity": "the difference between the rungs in each gated quantity (relative, the lag absolute), and the "
                        "joint 6 transient",
        "against": "the commanded joint speeds and accelerations: the joint targets the kinematic rung is written "
                   "(arm_tracking's), their largest rate and second difference per step over the steady window and "
                   "over the whole run, beside the description's velocity limits (1.0472 rad/s for joints 1 to 5, "
                   "1.5708 for joint 6; it gives no acceleration limit)",
        "threshold": "each quantity's tolerance above: a rung difference beyond it is one A0's comparison can see",
        "crossing": "the commanded joint speed at which a line through zero, fitted to the rung difference against "
                    "the window's largest commanded joint speed, crosses the threshold",
        "conditions": ["the kinematic rung carries neither the rotor's axial reaction nor its gyroscopic torque, and "
                       "has no spin-up kick: those are the dynamic rung's alone",
                       "A0 as declared: the magnet massless, the rotor's J 1e-5 kg m^2; the controller's filter 0.005 "
                       "a tick, the tag's joint loop (Kp 1e4, Kd 200: a 20 ms trail) and rotor loop",
                       "the drag surrogate dejongh_mlp_v2 about the body's centre, 0.3's body step, dt 0.5 ms, "
                       "float32"],
        "predicted": {key: _differences(record["new"][key], json.loads(P.EMULATION.read_text())["new"][key])
                      for key in FREQUENCIES},
        "machinery": "the validity line's form is a1's: this records the measurement and its conditions "
                     "(references/a0_rungs_measured.json), so that the line can cite them",
    }
    massive = {
        "what": "A0 with the magnet's mass: density 7.5 g/cm^3 on motor.rotor (0.144 kg), J 2.59e-5 kg m^2 about the "
                "axis in all, top_speed 2 pi x 10 rad/s, the magnet's mass and mean inertia across the axis welded on "
                "link_6, its half difference across the axis (3.12e-6 kg m^2) carried as it turns; the joint loop "
                "reads the built tree with the magnet grafted",
        "compared": "never with the 0.2 reference, which carried none of it: reported as data "
                    "(references/a0_rungs_measured.json)",
        "inferences": "both the magnet's spin about a diameter (its cylinder's axis along its moment, the rotor's x, "
                      "spun about z) and its density are inferences (7a4's N1). If the physical magnet is "
                      "diametrically magnetised and spun about its own axis, the rotor is symmetric about its axis "
                      "(no half difference; J about the axis m r^2 / 2 = 2.2e-5 kg m^2 of the magnet), which is run "
                      "beside it (magnet_axis along z)",
    }
    pose_tolerances = _pose_tolerances(poses)
    OUTPUT.write_text(json.dumps({
        "written": "2026-10-31, by predict_a0_dynamic.py declare, before 0.3's A0 ran on its dynamic rungs against this "
                   "reference: the only earlier runs on the dynamic rungs were builds (whose reports were read for the "
                   "lags) and one run of 0.1 s in each mode checked for finite fields, which read no steady "
                   "quantity",
        "comparison": "A0 on 0.3's dynamic rungs through the public API (tests/script_a0_dynamic.py: Arm.Dynamic and "
                      "Motor.Dynamic, closed by A0Loops, tests/a0_loops.py, inside the compiled step, every step, no "
                      "latency; the body script A's) in float32, against the frozen float32 reference in "
                      "references/a0/ at 3 Hz and 10 Hz, by the reference's own window rule",
        "how_predicted": "predict_a0_dynamic.py extends predict_a0.py's emulation (unchanged) to the dynamic rungs as "
                         "their nodes document them and A0's dynamic build orders them; with every change off it is "
                         "predict_a0's tag scheme to the bit. Each change is made alone to the tag's scheme, and all "
                         "together; the predicted values are the float32 reference moved by the difference between "
                         "the tag's scheme and 0.3's, relative for each quantity and absolute for the lag; the motor's "
                         "angle rate is the emulation's own single-precision accumulator's",
        "validation_of_the_emulation_against_the_reference": seven["validation_of_the_emulation_against_the_reference"],
        "closed": CLOSED,
        "differences": changes,
        "none_expected": none_expected,
        "all_changes_together": together,
        "predicted": predicted,
        "tolerances": {
            "values": values,
            "units": "relative to the predicted value, except the lag (rad)",
            "basis": "phase 7c1's tolerances, by its rule, which does not look at 0.3 (the larger of ten times the "
                     "reference's own float32-to-mixed gap and three times the emulation's miss of the reference): "
                     "unchanged for the body's and the tracking quantities. The rotor's speed: the larger of one "
                     "float32 epsilon (7c1's) and three times the effect of the rotor loop's read rounding "
                     f"({tolerances['read_rounding_on_the_rotor_speed']:.3g}), which the emulation carries in float32 "
                     "but not with 0.3's rounding pattern. The motor's angle rate: one float32 epsilon, its "
                     "prediction the emulated accumulator's (each step's increment rounded to whole units in the last "
                     "place, which a change of the speed below 1e-3 of itself does not move)",
            "from_7c1": seven["tolerances"],
        },
        "transient": {"predicted": kick, "tolerance": transient_tolerance,
                      "units": "relative to the predicted value",
                      "basis": "ten times float32's unit roundoff times the condition number of the arm's mass matrix "
                               "at its home pose: the relative rounding a single-precision solve of the arm's step "
                               "can leave in joint 6's motion, which the float64 emulation does not carry"},
        "swap_test": swap,
        "kinematic_against_dynamic": rungs,
        "magnet_pose": {
            "what": "the rotor's recorded pose through its mount (the arm's flange, the motor on it, the rotor turned "
                    "about its axis) against the reference's rotor_pose_world, at every tenth step in the window",
            "misses": "the largest over the window of: the position's distance from the reference's moved by the "
                      "predicted difference; the spin axis's tilt from the reference's less the predicted tilt; the "
                      "phase's turn from the reference's about its axis less the predicted turn",
            "predicted": "a0_dynamic_poses.npz: the emulation's rotor poses on the tag's scheme and on 0.3's",
            "tolerances": pose_tolerances,
            "basis": "7c1's rule on these misses: the larger of ten times the same miss between the reference's "
                     "float32 and mixed runs, and three times the emulation's miss of the reference with the tag's "
                     "scheme"},
        "with_the_magnets_mass": massive,
        "reported_not_gated": seven["reported_not_gated"],
        "if_a_quantity_misses": "it is a finding, reported with its size, and is not added to the declared "
                                "differences afterwards",
    }, indent=1) + "\n")
    print(f"wrote {OUTPUT.name}")


if __name__ == "__main__":
    declare() if sys.argv[1:] == ["declare"] else main()
