"""The dynamic motor and its reaction on the arm (MIME-ANO-103), each oracle with a control that fails on every draw
(float64, per module), on A0's apparatus as the build makes it (``chain_cases.dynamic_rpm``: the AR4 on its dynamic
rung carrying the dynamic motor on its flange, 5 cm out and turned, the magnet on the rotor).

- **The motor's bookkeeping, equal and opposite**: over drawn loads on its stator and rotor, drive torques, rotor
  spins, mount poses and the mount's angular velocities, what the stator passes to the frame it is mounted on, moved
  to the stator's origin, plus the rate of the rotor's angular momentum over the step (``J (Omega_{n+1} - Omega_n) /
  dt`` about the axis, and ``J Omega_{n+1} (omega x e)`` as its axis turns) is the whole of what acted on the motor
  from outside: the force on both its frames, and their torques about its origin. The drive's torque on the rotor and
  its reaction on the stator cancel inside it. This is an identity of one step of the motor's own algebra, a guard
  against a term dropped, flipped or misplaced; the arm's and motor's momentum and energy together are
  ``test_spinner.py``'s (MIME-ANO-122). The motor sits off its mount's origin, so the stator's lever on the mount is
  exercised. The control: the motor that passes nothing on fails on every draw.
- **The reaction reaches the arm a step late, at its mount's frame**: over a run of A0's dynamic apparatus, the arm the
  graph steps is the arm stepped by hand with the motor's reaction of the step before as its loads, and the motor the
  graph steps is the motor stepped by hand with the magnet's reactions of the same step on its rotor: the laws'
  reactions reach the rotor, and the rotor's the arm, by the edges the build derives. The control: the same build with
  the motor's reaction edge left out fails, its arm the arm stepped with no reaction.
- **The rotor**: a constant torque against its bearing's damping spins it up as ``tau / b + (omega_0 - tau / b)
  exp(-b t / J)``, to first order in time; and the node passes MADDENING's battery.
"""

import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from chain_cases import ROTOR_DAMPING, ROTOR_INERTIA, ROTOR_TORQUE, dynamic_rpm, holding_torques
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from maddening.testing.mms import RefinementAxis

import mime.build._builders as builders
from mime.core import quaternion
from mime.nodes.actuation import DynamicMotorNode
from mime.testing import OrderLadder, first_steps, node_results

PROPERTY = settings(deadline=None, max_examples=100)
EPS = float(np.finfo(np.float64).eps)
ROUNDINGS = 64
DT = 1e-3


@pytest.fixture(scope="module")
def built():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dynamic_rpm().build()


@pytest.fixture(scope="module")
def motor(built):
    return built.graph.get_node("rpm.motor")


def unit_rows(poses):
    rows = np.asarray(poses, dtype=np.float64).reshape(-1, 7).copy()
    lengths = np.linalg.norm(rows[:, 3:], axis=1, keepdims=True)
    rows[:, 3:] = np.where(lengths > 1e-3, rows[:, 3:] / np.where(lengths > 1e-3, lengths, 1.0), [1.0, 0.0, 0.0, 0.0])
    return rows.reshape(np.asarray(poses).shape)


@st.composite
def steps(draw, count: int):
    """A step of the motor: loads on its two frames of a magnet's and a motor's scale (forces to 0.1 mN, torques to 1
    mN m), a drive torque of A0's scale, a rotor spin to 100 rad/s, the mount's frame poses within a metre, and their
    angular velocities to an arm's joint speeds (2 rad/s each way)."""
    forces = arrays(np.float64, (2, 3), elements=st.floats(-1e-4, 1e-4, allow_nan=False))
    torques = arrays(np.float64, (2, 3), elements=st.floats(-1e-3, 1e-3, allow_nan=False))
    poses = arrays(np.float64, (count, 7), elements=st.floats(-1.0, 1.0, allow_nan=False)).map(unit_rows)
    rates = arrays(np.float64, (count, 3), elements=st.floats(-2.0, 2.0, allow_nan=False))
    drive = draw(st.floats(0.1, 1.0)) * ROTOR_TORQUE * draw(st.sampled_from([-1.0, 1.0]))
    return (np.concatenate([draw(forces), draw(torques)], axis=1), drive, draw(st.floats(-100.0, 100.0)), draw(poses),
            draw(rates))


def balance_misfit(motor, wrenches, drive, spin, poses, rates) -> tuple[float, float]:
    """How far what the stator passes plus what the rotor keeps is from what acted on the motor, and the bound: the
    rounding of the largest term, the rotor's momentum over the step among them (its gain is a difference of two)."""
    held = motor.constants()
    state = {**motor.initial_state(), "spin": jnp.asarray(spin)}
    out = motor.update(state, {"torque": jnp.asarray(drive), "link_wrench": jnp.asarray(wrenches),
                               "mount_poses": jnp.asarray(poses), "mount_rates": jnp.asarray(rates)}, DT)
    index = held["mount_index"]
    frame = np.asarray(poses)[index]
    base = np.asarray(out["frame_poses"])[0]
    axis = np.asarray(quaternion.rotate(jnp.asarray(base[3:]), jnp.asarray(held["axis"])))
    passed = np.asarray(out["reaction"])[index]
    force, about_base = passed[:3], passed[3:] - np.cross(base[:3] - frame[:3], passed[:3])
    turned = ROTOR_INERTIA * float(out["spin"]) * np.cross(np.asarray(rates)[index], axis)
    kept = ROTOR_INERTIA * (float(out["spin"]) - spin) / DT * axis + turned
    acted = wrenches[0] + wrenches[1]
    misfit = max(float(np.max(np.abs(force - acted[:3]))), float(np.max(np.abs(about_base + kept - acted[3:]))))
    scale = max(float(np.max(np.abs(acted))), abs(drive), ROTOR_INERTIA * abs(spin) / DT,
                float(np.max(np.abs(passed))), float(np.max(np.abs(turned))))
    return misfit, ROUNDINGS * EPS * scale


@PROPERTY
@given(steps(27))
def test_what_the_stator_passes_and_the_rotor_keeps_is_what_acted_on_the_motor(motor, drawn):
    misfit, bound = balance_misfit(motor, *drawn)
    assert misfit <= bound, f"{misfit:.3g} > {bound:.3g}"


class SilentMotor(DynamicMotorNode):
    """The planted motor: its stator passes nothing to the frame it is mounted on."""

    def passed(self, given, base, axis, held, wrenches, turning):
        return jnp.zeros((max(self.constants()["mount_count"], 1), 6), dtype=base.dtype)


@pytest.fixture(scope="module")
def silent(motor):
    return SilentMotor(motor.name, DT, **motor.constants())


@PROPERTY
@given(steps(27))
def test_control_a_motor_that_passes_nothing_on_fails_on_every_draw(silent, drawn):
    misfit, bound = balance_misfit(silent, *drawn)
    assert misfit > bound


# -- the reaction through the build -------------------------------------------------------------------------------

RUN = 12


def by_hand(nodes, history):
    """The arm and the motor stepped by hand from their initial states over ``RUN`` steps: the arm under the holding
    torques with the motor's reaction of the step before as its loads, the motor under its torque with the sum of the
    laws' reactions of the same step on its frames and the arm's frame poses of the same step as its mount."""
    arm, motor = nodes
    arm_state, motor_state = arm.initial_state(), motor.initial_state()
    torques = jnp.asarray(holding_torques())
    arms, motors = [], []
    for n in range(RUN):
        reaction = jnp.zeros_like(motor_state["reaction"]) if n == 0 else history["rpm.motor"]["reaction"][n - 1]
        arm_state = arm.update(arm_state, {"joint_torques": torques, "link_wrench": reaction}, DT)
        loads = sum(history[f"magnetic.{law}:umr.reaction"]["link_wrench"][n] for law in ("force", "torque"))
        motor_state = motor.update(motor_state, {"torque": jnp.asarray(ROTOR_TORQUE), "link_wrench": loads,
                                                 "mount_poses": history["rpm.arm"]["frame_poses"][n],
                                                 "mount_rates": history["rpm.arm"]["frame_angular_velocities"][n]},
                                   DT)
        arms.append(np.asarray(arm_state["joint_positions"]))
        motors.append(np.asarray(motor_state["spin"]))
    return np.asarray(arms), np.asarray(motors)


def run_from_the_start(built):
    built.graph.reset_state()
    return first_steps(built, RUN)


def segments(built):
    """The arm's and the motor's nodes of ``built``, which a planted build's nodes equal, compiled once."""
    return built.graph.get_node("rpm.arm"), built.graph.get_node("rpm.motor")


def reaction_misfits(built, nodes) -> list[str]:
    """How the run of ``built`` differs from the arm and the motor ``nodes`` stepped by hand on its history."""
    history = run_from_the_start(built)
    arms, motors = by_hand(nodes, history)
    found = []
    graph_arm = np.asarray(history["rpm.arm"]["joint_positions"])
    if np.max(np.abs(graph_arm - arms)) > ROUNDINGS * EPS * RUN:
        found.append(f"the arm differs from the arm loaded by the motor's reaction by "
                     f"{np.max(np.abs(graph_arm - arms)):.3g}")
    graph_motor = np.asarray(history["rpm.motor"]["spin"])
    if np.max(np.abs(graph_motor - motors)) > ROUNDINGS * EPS * RUN * float(np.max(np.abs(motors))):
        found.append("the motor differs from the motor loaded by the magnet's reactions")
    return found


def test_the_magnets_reactions_reach_the_rotor_and_the_rotors_the_arm_a_step_late(built):
    history = run_from_the_start(built)
    assert float(np.max(np.abs(np.asarray(history["rpm.motor"]["reaction"])))) > 0.0
    assert float(np.max(np.abs(np.asarray(history["magnetic.torque:umr.reaction"]["link_wrench"])))) > 0.0
    assert reaction_misfits(built, segments(built)) == []


def test_control_the_build_with_the_motors_reaction_left_out_fails(built, monkeypatch):
    monkeypatch.setattr(builders.DynamicsBuilder, "reaction", lambda self, output: False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        planted = dynamic_rpm("unreacted_rpm").build()
    assert not [edge for edge in planted.report.lags if edge.carries == builders.REACTION]
    found = reaction_misfits(planted, segments(built))
    print(f"\n{found}")
    assert any("the arm" in line for line in found)


# -- the rotor's own law, its order and MADDENING's battery -------------------------------------------------------

HORIZON = 0.05


def spun_up(motor, steps: int) -> float:
    """The rotor's speed after ``HORIZON`` under A0's torque from rest, in ``steps`` steps."""
    p = motor._step_constants(None)
    dt = HORIZON / steps
    held = motor.constants()
    mount = jnp.tile(jnp.asarray(held["initial_mount_pose"]), (held["mount_count"], 1))
    still = jnp.zeros((held["mount_count"], 3))

    def step(state, _):
        return motor.advance(p, state, {"torque": jnp.asarray(ROTOR_TORQUE), "mount_poses": mount,
                                        "mount_rates": still}, dt), None

    final, _ = jax.jit(lambda s: jax.lax.scan(step, s, None, length=steps))(motor.initial_state())
    return float(final["spin"])


def exact_speed() -> float:
    terminal = ROTOR_TORQUE / ROTOR_DAMPING
    return terminal * (1.0 - math.exp(-ROTOR_DAMPING * HORIZON / ROTOR_INERTIA))


def test_the_rotor_spins_up_to_its_closed_form_first_order_and_passes_maddenings_battery(motor):
    ladder = OrderLadder(RefinementAxis.TIME, lambda steps: abs(spun_up(motor, steps) - exact_speed()) / exact_speed(),
                         (25, 50, 100, 200))
    assert abs(spun_up(motor, 200) - exact_speed()) <= 0.05 * exact_speed()
    options = {"bounds": {"angle": (-7.0, 7.0), "spin": (-100.0, 100.0), "frame_poses": (-1.0, 1.0),
                          "frame_angular_velocities": (-100.0, 100.0), "reaction": (-1e-3, 1e-3)},
               "boundary_bounds": {"torque": (-1e-3, 1e-3), "link_wrench": (-1e-3, 1e-3), "mount_poses": (-1.0, 1.0),
                                   "mount_rates": (-2.0, 2.0)},
               "constrain_state": lambda state: {**state, "frame_poses": unit_rows(state["frame_poses"]).astype(
                   np.asarray(state["frame_poses"]).dtype)},
               "constrain_boundary": lambda given: {**given, "mount_poses": unit_rows(given["mount_poses"]).astype(
                   np.asarray(given["mount_poses"]).dtype)}}
    results = node_results(motor, options, (ladder,))
    print("\n" + "\n".join(str(r) for r in results.values()))
    assert all(r.passed for r in results.values())
