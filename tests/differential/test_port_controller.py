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
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from chain_cases import tracked_rpm, tracking

from mime.build import Mode
from mime.control import control_context
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


def port(feedback: bool, **changed: float):
    """A0's controller with the 0.2 rig's settings, ready to step; ``changed`` replaces some of its parameters."""
    controller = tracking(orientation_feedback=feedback, standoff=m(0.15), reach=m((-0.05, 0.05)),
                          **(PARAMETERS | changed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return controller.law(BuildContext("controller", 5e-4, 5e-4, "controller", settings=SETTINGS))


def stepped(feedback: bool, inputs: dict[str, np.ndarray], **changed: float) -> dict[str, np.ndarray]:
    law = port(feedback, **changed)
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


@pytest.mark.parametrize("label", ["f32", "mixed_f64"])
@pytest.mark.parametrize("feedback", [False, True], ids=["feedback_off", "feedback_on"])
def test_the_port_computes_what_the_0_2_controller_did_within_its_declared_differences(label, feedback):
    held = reference(label)
    inputs = {key.removeprefix("input/"): value for key, value in held.items() if key.startswith("input/")}
    arm = "feedback_on" if feedback else "feedback_off"
    found = stepped(feedback, inputs)
    worst = {name: float(np.max(np.abs(found[name] - held[f"{arm}/{name}"]))) for name in found}
    worst["target_orientation"] = float(np.max(rotation_angle(found["target_orientation"],
                                                              held[f"{arm}/target_orientation"])))
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


def test_control_a_planted_change_of_the_law_exceeds_the_declared_bounds():
    """The comparison can fail: the port with its gain at 0.9 instead of the 0.2 rig's 1.0, a change the size of a
    misread setting, leaves the target joint positions outside their bound on the recorded sequence."""
    held = reference("f32")
    inputs = {key.removeprefix("input/"): value for key, value in held.items() if key.startswith("input/")}
    found = stepped(False, inputs, gain=0.9)
    assert float(np.max(np.abs(found["q_target"] - held["feedback_off/q_target"]))) > bounds(held, False)["q_target"]


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
