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
   controller casts every input to it), so they differ by float32 rounding, which the five damped-least-squares
   iterations amplify by their conditioning (about (s_max^2 + damping^2) / damping^2, near 100 at damping 0.05):
   at most 1e-4 rad in a target joint position, 1e-6 m in the target's position, 1e-5 in its orientation's
   components.
3. **The float32 casts.** In a run with ``jax_enable_x64`` the 0.2 controller's matrices were float64 (``jnp.eye``'s
   default) while its inputs were cast to float32: mixed precision. The port computes in float32 throughout, as
   recorded in its docstring, and is not compared with the mixed arm beyond the same bounds.
4. **The first tick.** The 0.2 controller fell back to defaults when it had no state; on a recorded sequence both are
   given the state at every tick, so this makes no difference here.
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
#: The bounds of declared difference 2.
JOINTS, POSITION, ORIENTATION = 1e-4, 1e-6, 1e-5


#: The 0.2 rig's settings in SI, as a build hands them to the controller (``test_the_ports_context...``).
SETTINGS = {"at": (-0.05, 0.328, -0.43), "home": (-0.02761, 0.23896, -0.77475, 1.55611, 1.54628, 0.0),
            "standoff": 0.15, "reach": (-0.05, 0.05), "lookahead": 0.010}


def port(feedback: bool, gain: float = 1.0):
    """A0's controller with the 0.2 rig's settings, ready to step."""
    controller = tracking(orientation_feedback=feedback, standoff=m(0.15), reach=m((-0.05, 0.05)), smoothing=0.005,
                          gain=gain, damping=0.05, iterations=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return controller.law(BuildContext("controller", 5e-4, 5e-4, "controller", settings=SETTINGS))


def stepped(feedback: bool, inputs: dict[str, np.ndarray], gain: float = 1.0) -> dict[str, np.ndarray]:
    law = port(feedback, gain)
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
    print(f"\n{label}, {arm}: largest differences {json.dumps(worst)}")
    assert worst["q_target"] <= JOINTS
    assert worst["target_position"] <= POSITION
    assert worst["target_orientation"] <= ORIENTATION


def test_control_a_planted_change_of_the_law_exceeds_the_declared_bounds():
    """The comparison can fail: the port with its gain at 0.9 instead of the 0.2 rig's 1.0, a change the size of a
    misread setting, leaves the target joint positions outside their bound on the recorded sequence."""
    held = reference("f32")
    inputs = {key.removeprefix("input/"): value for key, value in held.items() if key.startswith("input/")}
    found = stepped(False, inputs, gain=0.9)
    assert float(np.max(np.abs(found["q_target"] - held["feedback_off/q_target"]))) > JOINTS


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
