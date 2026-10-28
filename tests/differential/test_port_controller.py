"""A0's controller against the 0.2 line's, on one recorded input sequence: ``mime.control.ArmTracking`` with the 0.2
rig's settings, stepped tick by tick on the inputs ``generate_port_controller_references.py`` drew and fed to the 0.2
controller in the port's own environment, its outputs compared with what that controller computed. Both arms: the
orientation feedback off (A0's setting) and on.

**The differences, declared before they were measured:**

1. **What it writes.** The 0.2 controller wrote joint torques, by computed-torque control of its target joint
   positions on a dynamic arm, and the motor's speed. The port drives a kinematic arm, so it writes the target joint
   positions themselves, and the motor's speed is the apparatus's own drive. The torques and the speed are not
   compared; the target joint positions (the 0.2 ``q_target``) and the filtered target are.
2. **The kinematics.** The 0.2 controller composed 4 x 4 matrices, took the rotation error as the logarithm of a
   rotation matrix and the Jacobian at the link's centre of mass shifted to the end effector; the port uses
   ``mime.core.multibody``'s quaternion poses, rotation vector and frame Jacobian. Both compute in float32 (the 0.2
   controller casts every input to it), so they differ by float32 rounding. The bounds are that rounding, derived
   below from the machine epsilon, the size of the quantity and the operations it passes through, and not from any
   difference measured between the two; they are worst-case sums, so an honest run lies well inside them.
3. **The float32 casts.** In a run with ``jax_enable_x64`` the 0.2 controller's matrices were float64 (``jnp.eye``'s
   default) while its inputs were cast to float32: mixed precision. The port computes in float32 throughout, as
   recorded in its docstring, and is not compared with the mixed arm beyond the same bounds.
4. **The first tick.** The 0.2 controller fell back to defaults when it had no state; on a recorded sequence both are
   given the state at every tick, so this makes no difference here.

**Found when measured, not declared:** 5. **The target orientation's sign.** The port's target orientation is the
negative of the 0.2 controller's quaternion at every tick (largest component difference 1.364 = 2 x 0.682, its
largest component): the home pose's orientation comes out as ``-q`` from ``mime.core.multibody``'s composition and as
``q`` from the 0.2 controller's matrix-to-quaternion conversion, and each tick's sign follows the one before (both
controllers flip the new target to the side of the held one). ``q`` and ``-q`` are one rotation, and the target joint
positions, which the inverse kinematics computes from that rotation, agree within declared difference 2 in both arms;
so the orientations are compared by the angle of the rotation between them, which a sign does not change.

**The bounds** (``bounds``), each ``EPS`` (single precision's machine epsilon, two unit roundoffs: each of the two
implementations rounds) times the quantity's size times the operations on its longest path:

- *The target's position (m):* the size is the arm's reach plus the base's distance, the chain's largest partial sum;
  the path is the pose of the end effector through the base, six joints and the flange, 8 frames of about 4 roundings
  each (a composition rotates a vector, 3, and adds it, 1).
- *The target's orientation (rad):* a unit quaternion, size 1; the same chain, and with the orientation feedback 16
  more roundings (the rotation matrices, their product, two normalisations, the cross product, the conversion back to
  a quaternion and the blend).
- *The target joint positions (rad):* each of the iterations takes a step of at most ``1 / (2 damping)`` times the
  pose error (the norm of damped least squares' pseudo-inverse is at most that, at singular value = damping), and the
  pose error is the position and orientation roundings above taken together; the iterations' errors add, and each
  iteration rounds the joint positions (size: the largest target joint position) once.

They are worst-case sums, about 2e-4 rad, 3e-6 m and 4e-6 rad on this sequence, six to sixty times above what the two
controllers differ by (each rounding's sign is free, and the sums assume it never cancels); that is what rounding alone
supports, and a tighter bound would be a claim about these two implementations that was not derived.

**What they resolve,** measured by planting each change in the port at several sizes (the smallest caught, in the arm
where the setting acts): a gain or a damping off by 0.3% (feedback off) to 0.4% (on), the joints' bound against their
shift being 2.3e-4 rad against 1.9e-4 at 0.2%; a smoothing, a standoff, a lookahead (with the orientation feedback
on) or a reach off by 0.05% or less; one iteration more or fewer; the reach clip removed. The sequence's joint
positions are scattered 0.3 rad about home for this: near home the iterations converge and a gain or a damping moves
the joints by less than the rounding does. **Below the bounds' resolution:** a gain or a damping off by less than
that, and the float32 casts dropped, which moves the joints by about 1e-6 rad, 200 times below their bound;
``test_the_controller_computes_in_single_precision...`` checks the casts by the dtype of what the controller writes
instead.
"""

import json
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from chain_cases import tracked_rpm, tracking

from mime.build import Mode
from mime.control import _arm_tracking, control_context
from mime.core import jnp
from mime.registry import BuildContext
from mime.units import m

REFERENCES = Path(__file__).parent / "references"
#: Single precision's machine epsilon, two unit roundoffs: the port and the 0.2 controller each round.
EPS = float(np.finfo(np.float32).eps)
#: The frames of the arm's chain (the base, six joints and the flange) at about 4 roundings each.
CHAIN_ROUNDINGS = 8 * 4
#: What the orientation feedback adds along the orientation's path.
FEEDBACK_ROUNDINGS = 16

#: The 0.2 rig's settings in SI, as a build hands them to the controller (``test_the_ports_context...``).
SETTINGS = {"at": (-0.05, 0.328, -0.43), "home": (-0.02761, 0.23896, -0.77475, 1.55611, 1.54628, 0.0),
            "standoff": 0.15, "reach": (-0.05, 0.05), "lookahead": 0.010}


#: The controller's own parameters, as the 0.2 rig set them.
PARAMETERS = {"smoothing": 0.005, "gain": 1.0, "damping": 0.05, "iterations": 5}


def bounds(held: dict[str, np.ndarray], feedback: bool) -> dict[str, float]:
    """The rounding bounds of declared difference 2 for one reference: position (m), orientation angle (rad) and joint
    positions (rad)."""
    arm = "feedback_on" if feedback else "feedback_off"
    farthest = float(np.max(np.linalg.norm(held[f"{arm}/target_position"], axis=1)))
    reach = float(np.linalg.norm(SETTINGS["at"])) + farthest
    position = EPS * reach * CHAIN_ROUNDINGS
    orientation = EPS * (CHAIN_ROUNDINGS + FEEDBACK_ROUNDINGS * feedback)
    step = np.hypot(position, orientation) / (2 * PARAMETERS["damping"])
    joints = PARAMETERS["iterations"] * (EPS * float(np.max(np.abs(held[f"{arm}/q_target"]))) + step)
    return {"q_target": joints, "target_position": position, "target_orientation": orientation}


def rotation_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The angle (rad) of the rotation between two unit quaternions per row, ``2 acos(|a . b|)``: a quaternion and its
    negative are one rotation, and a conjugate or a flipped axis is a different one. Computed as ``2 atan2`` of the
    relative quaternion's vector part against its scalar part, which is the same angle and is not ill-conditioned
    near zero, where ``acos`` of a dot product rounded to 1 loses half its digits."""
    a, b = (x / np.linalg.norm(x, axis=1, keepdims=True) for x in (a, b))
    scalar = np.sum(a * b, axis=1)
    vector = a[:, :1] * b[:, 1:] - b[:, :1] * a[:, 1:] - np.cross(a[:, 1:], b[:, 1:])
    return 2 * np.arctan2(np.linalg.norm(vector, axis=1), np.abs(scalar))


def port(feedback: bool, parameters: Mapping[str, float] | None = None, settings: Mapping[str, Any] | None = None):
    """A0's controller with the 0.2 rig's settings, ready to step; ``parameters`` replaces some of its parameters and
    ``settings`` some of the settings a build hands it (the standoff, the reach, the lookahead)."""
    controller = tracking(orientation_feedback=feedback, standoff=m(0.15), reach=m((-0.05, 0.05)),
                          **(PARAMETERS | dict(parameters or {})))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return controller.law(BuildContext("controller", 5e-4, 5e-4, "controller",
                                           settings=SETTINGS | dict(settings or {})))


def stepped(feedback: bool, inputs: dict[str, np.ndarray], parameters: Mapping[str, float] | None = None,
            settings: Mapping[str, Any] | None = None) -> dict[str, np.ndarray]:
    law = port(feedback, parameters, settings)
    state = law.state
    found: dict[str, list[np.ndarray]] = {"q_target": [], "target_position": [], "target_orientation": []}
    for tick in range(len(inputs["position"])):
        reads = {"body_pose": np.concatenate([inputs["position"][tick], inputs["orientation"][tick]]),
                 "body_twist": np.concatenate([np.zeros(3), inputs["angular_velocity"][tick]]),
                 "joint_positions": inputs["joint_positions"][tick]}
        state, writes = law.step(state, reads)
        found["q_target"].append(np.asarray(writes["joint_positions"]))
        found["target_position"].append(np.asarray(state["target_position"]))
        found["target_orientation"].append(np.asarray(state["target_orientation"]))
    return {name: np.asarray(rows) for name, rows in found.items()}


def reference(label: str) -> dict[str, np.ndarray]:
    with np.load(REFERENCES / f"port_controller_{label}.npz", allow_pickle=False) as held:
        return {key: np.asarray(held[key]) for key in held.files}


def worst_differences(found: dict[str, np.ndarray], held: dict[str, np.ndarray], feedback: bool) -> dict[str, float]:
    """The largest difference of each compared quantity from the reference's: the orientation's by its angle."""
    arm = "feedback_on" if feedback else "feedback_off"
    worst = {name: float(np.max(np.abs(found[name] - held[f"{arm}/{name}"]))) for name in found}
    worst["target_orientation"] = float(np.max(rotation_angle(found["target_orientation"],
                                                              held[f"{arm}/target_orientation"])))
    return worst


def exceeded(found: dict[str, np.ndarray], held: dict[str, np.ndarray], feedback: bool) -> list[str]:
    """The quantities whose largest difference is beyond its bound."""
    worst, limit = worst_differences(found, held, feedback), bounds(held, feedback)
    return [name for name in limit if worst[name] > limit[name]]


def inputs_of(held: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key.removeprefix("input/"): value for key, value in held.items() if key.startswith("input/")}


@pytest.mark.parametrize("label", ["f32", "mixed_f64"])
@pytest.mark.parametrize("feedback", [False, True], ids=["feedback_off", "feedback_on"])
def test_the_port_computes_what_the_0_2_controller_did_within_its_declared_differences(label, feedback):
    held = reference(label)
    arm = "feedback_on" if feedback else "feedback_off"
    worst = worst_differences(stepped(feedback, inputs_of(held)), held, feedback)
    limit = bounds(held, feedback)
    print(f"\n{label}, {arm}: largest differences {json.dumps(worst)}, bounds {json.dumps(limit)}")
    for name in limit:
        assert worst[name] <= limit[name], name


@pytest.mark.parametrize("label", ["f32", "mixed_f64"])
def test_the_recorded_sequence_passes_the_reach_envelope_and_lies_on_it(label):
    """The comparison exercises the reach clip: at least a quarter of the ticks lie beyond the reach, on each side of
    it, and at least four exactly on it, the generator's ``SWING`` and ``ON_IT``."""
    x = reference(label)["input/position"][:, 0]
    low, high = SETTINGS["reach"]
    assert np.mean(x > high) >= 0.1 and np.mean(x < low) >= 0.1 and np.mean((x > high) | (x < low)) >= 0.25
    assert np.sum(x == high) >= 2 and np.sum(x == low) >= 2


#: What a misread constant or a deleted line would change, each as (arm's orientation feedback, parameters, settings).
#: The lookahead acts only with the orientation feedback on. "Reach removed" is the clip's bounds pushed out of the
#: sequence's range, which is the clip deleted.
PLANTS = {
    "gain 0.9": (False, {"gain": 0.9}, {}),
    "gain 0.99": (False, {"gain": 0.99}, {}),
    "damping +1%": (False, {"damping": 0.0505}, {}),
    "iterations 6": (False, {"iterations": 6}, {}),
    "iterations 4": (False, {"iterations": 4}, {}),
    "smoothing +1%": (True, {"smoothing": 0.00505}, {}),
    "standoff +0.1 mm": (False, {}, {"standoff": 0.1501}),
    "lookahead +1%": (True, {}, {"lookahead": 0.0101}),
    "reach removed": (False, {}, {"reach": (-1.0, 1.0)}),
    "reach 51 mm": (False, {}, {"reach": (-0.051, 0.051)}),
}


@pytest.mark.parametrize("plant", PLANTS)
def test_control_a_planted_change_of_the_law_exceeds_the_declared_bounds(plant):
    """The comparison can fail: the port with one parameter or setting changed by an amount a misread constant would
    change it, or the reach clip removed, is outside the bounds on the recorded sequence in at least one quantity."""
    feedback, parameters, settings = PLANTS[plant]
    held = reference("f32")
    assert exceeded(stepped(feedback, inputs_of(held), parameters, settings), held, feedback)


@pytest.mark.parametrize("plant", ["conjugate", "x flipped"])
def test_control_b_a_wrong_handed_orientation_is_outside_the_bound(plant):
    """The orientation is compared by the rotation's angle: a conjugate (a quaternion's vector part reversed) and a
    reversed x axis are rotations other than the reference's, by up to tens and hundreds of degrees, which a
    comparison of components up to sign did not see."""
    held = reference("f32")["feedback_on/target_orientation"]
    planted = held * (np.array([1.0, -1.0, -1.0, -1.0]) if plant == "conjugate" else np.array([1.0, -1.0, 1.0, 1.0]))
    assert np.max(rotation_angle(planted, held)) > bounds(reference("f32"), True)["target_orientation"]


def test_a_quaternion_and_its_negative_are_one_rotation():
    """The sign the port's orientation takes is the reference's negative at every tick, or flips from one tick on:
    the angle between them is zero to rounding either way."""
    held = reference("f32")["feedback_on/target_orientation"]
    flipped = np.where((np.arange(len(held)) >= len(held) // 2)[:, None], -held, held)
    assert np.max(rotation_angle(-held, held)) < 1e-6
    assert np.max(rotation_angle(flipped, held)) < 1e-6


@pytest.mark.parametrize("angle", [1e-6, 1e-3, 0.3, 3.0])
def test_the_rotation_angle_is_the_angle_to_float32_resolution(angle):
    """``rotation_angle`` of the identity and a turn of ``angle`` about a skew axis returns ``angle``, and a turn
    of 1e-6 rad, where ``2 acos(|a . b|)`` in float64 would return its square root's order, is read to 1e-9."""
    axis = np.array([1.0, 2.0, 2.0]) / 3.0
    turned = np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])[None, :]
    assert abs(float(rotation_angle(turned, np.array([[1.0, 0.0, 0.0, 0.0]]))[0]) - angle) < 1e-9


def single_precision_problems(feedback: bool) -> list[str]:
    """The names, among the controller's writes and state, that are not float32 after a step."""
    held = reference("f32")
    law = port(feedback)
    inputs = inputs_of(held)
    reads = {"body_pose": np.concatenate([inputs["position"][0], inputs["orientation"][0]]),
             "body_twist": np.concatenate([np.zeros(3), inputs["angular_velocity"][0]]),
             "joint_positions": inputs["joint_positions"][0]}
    state, writes = law.step(law.state, reads)
    found = {f"state.{name}": value for name, value in state.items()} | \
        {f"writes.{name}": value for name, value in writes.items()}
    return [name for name, value in found.items() if np.asarray(value).dtype != np.float32]


@pytest.mark.parametrize("feedback", [False, True], ids=["feedback_off", "feedback_on"])
def test_the_controller_computes_in_single_precision_in_a_double_precision_run(feedback):
    """Declared difference 3: the controller's casts are kept. Dropping them changes the target joint positions by
    about 1e-6 rad, below the bounds' resolution, so this checks the dtype of what it writes and keeps under x64."""
    assert single_precision_problems(feedback) == []


def test_control_c_dropped_float32_casts_are_seen_by_the_dtype(monkeypatch):
    monkeypatch.setattr(_arm_tracking, "_F32", jnp.float64)
    assert single_precision_problems(True) != []


def test_the_ports_context_is_the_one_a_build_gives_it():
    """The settings the comparison steps the port with are those a build hands it: built from the A0 rig with no rate
    and no latency, the controller runs every step, and its context holds the same values in SI."""
    plans = tracked_rpm(rate=None, latency=None).plan()
    held = plans.controller
    assert held is not None and (held.mode, held.period, held.latency) == (Mode.CALLBACK, 1, 0)
    ctx = control_context(plans)
    assert {name: ctx.vector(name) if name in ("at", "home", "reach") else ctx.scalar(name) for name in SETTINGS} == \
        SETTINGS
    assert ctx.timestep == plans.dt
