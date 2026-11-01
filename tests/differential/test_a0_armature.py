"""A0 on its dynamic rungs and its wrist's armature (phase 7a5, design #193).

**As declared**, A0 gives its arm's joints no armature and says why (``script_a0_dynamic.NO_ARMATURE``: the 0.2 line's
arm carried none, and the comparison holds the reference's description). The build's validity line names joint_6, the
bare wrist the rotor's spin-up kicks: its ratio 4.11 (``J / I6`` with the motor on the flange's axis), keyed on the
ratio since A0 declares no top speed for its motor, and the profile records the convention's source.

**Two examples, not preregistered and never compared with the reference**, each with a wrist armature that is an
inference, stated as one, not the AR4's documented or measured drive (the AR4's description gives its drive neither,
and nothing here fits it):

- **A geared NEMA 17 wrist**: the registered actuator's rotor (StepperOnline's 17HS19-2004S1, 82 g cm^2 from its
  drawing) behind an assumed reduction of 10, a value of a geared wrist's order chosen for the example: 100 x 8.2e-6 =
  8.2e-4 kg m^2. A0's own controller, whose computed-torque loop reads the built tree with the armature on it (the
  tree it reads holds the node's armature, checked), holds the wrist inside its speed limit at 10 Hz, the run's flag
  stays down, and the build gives no line. The control: the bare wrist, in the same run, passes its limit twenty times
  over and raises the flag. Here the free kick the check predicts (0.76 rad/s at 10 Hz) is inside the limit already,
  so this example cannot tell a controller that sees the armature from one that does not.
- **7c2's stand-in**, 1e-4 kg m^2 on the wrist: the free kick the check predicts at 10 Hz, 6.1 rad/s, passes the limit
  nearly four times, so only a controller that sees the armature holds the wrist inside it. A0's controller on the
  built tree does, the flag down; the same loops on a stale tree (every joint's armature zeroed, a law written against
  the bare description) pass the limit and raise the flag."""

import math
import warnings

import numpy as np
from a0_loops import planted_stale_model
from script_a0_dynamic import NO_ARMATURE, a0_dynamic

from mime.dynamics import actuator
from mime.units import Hz, kg_m2, ms

#: The AR4's wrist speed limit, from its description (rad/s).
WRIST_LIMIT = 1.5708
#: The assumed reduction of the geared wrist's drive: an inference for the example, not the AR4's gearing.
REDUCTION = 10.0
GEARED = REDUCTION ** 2 * actuator("stepperonline.17hs19-2004s1").inertia
#: 7c2's stand-in for a geared wrist (kg m^2): a value of the order, not a fit.
STAND_IN = 1e-4
#: The run: the spin-up's kick is over within a tenth of a second at 10 Hz (A0's own test reads it); the rotor's
#: speed there (Hz), which the free kick is judged at.
RUN, FREQUENCY = ms(150), 10.0


def wrist(armature, *, stale: bool = False):
    """The build's plans and report, the wrist's largest speed over the run, whether the flag was raised, and whether
    the tree the controller reads holds the armature the node holds; ``stale``, the controller's loops on the tree with
    every armature zeroed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        experiment = a0_dynamic("a0_wrist", frequency=Hz(FREQUENCY), armature=armature)
        plans = experiment.plan()
        if stale:
            with planted_stale_model():
                built = experiment.build()
        else:
            built = experiment.build()
    held = np.asarray(plans.models["rpm.arm"].articulation.armatures, dtype=np.float64)
    node = np.asarray(built.graph.get_node("rpm.arm").constants()["armatures"], dtype=np.float64)
    result = built.run(RUN, history=True)
    speeds = np.abs(np.asarray(result.history["rpm.arm"]["joint_velocities"])[:, 5])
    flags = [read for read in result.flags if read.path == "rpm.arm.beyond_speed_limit"]
    return plans, built.report, float(speeds.max()), bool(flags and flags[0].was_raised), np.array_equal(held, node)


def free_kick(plans) -> float:
    """The wrist's speed the build's check predicts with the joints held by nothing, at the rotor's speed (rad/s)."""
    (judged,) = plans.kicks
    return {kick.joint: kick.ratio for kick in judged.kicks}["joint_6"] * 2.0 * math.pi * FREQUENCY


def test_a0_as_declared_states_its_wrists_armature_none_and_the_build_names_its_kick():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plans = a0_dynamic().plan()
    assert plans.profile.sources["rpm.arm.armature"] == NO_ARMATURE
    assert "rpm.arm.armature" not in [item.target for item in plans.profile.listed]
    (line,) = plans.checked["rpm.arm"]
    assert line.startswith("joint_6: ") and "a ratio of 4.11, above 0.1: the rotor's top speed is not declared" in line


def test_a0s_controller_holds_a_geared_wrist_inside_its_limit_and_the_bare_wrist_passes_it():
    reason = "an inference: a 17HS19-2004S1's rotor (its drawing) behind an assumed reduction of 10"
    plans, report, peak, raised, same = wrist({"joint_6": kg_m2(GEARED, reason=reason)})
    print(f"\nthe wrist's largest speed at 10 Hz: geared {peak:.4g} rad/s (free kick {free_kick(plans):.4g})", end="")
    assert same, "the tree the controller reads holds the node's armature"
    assert peak < WRIST_LIMIT and not raised, peak
    assert "rpm.arm" not in report.checked
    _, _, bare, bare_raised, _ = wrist(None)
    print(f", bare {bare:.4g} rad/s (its limit {WRIST_LIMIT})")
    assert bare > 20 * WRIST_LIMIT and bare_raised, bare


def test_at_the_stand_in_only_a_controller_that_sees_the_armature_holds_a_kick_past_the_limit():
    armature = {"joint_6": kg_m2(STAND_IN, reason="7c2's stand-in for a geared wrist, a value of the order")}
    plans, _, peak, raised, same = wrist(armature)
    kick = free_kick(plans)
    _, _, stale, stale_raised, _ = wrist(armature, stale=True)
    print(f"\nat {STAND_IN:g} kg m^2: the free kick {kick:.4g} rad/s, A0's controller {peak:.4g}, on a stale tree "
          f"{stale:.4g} (the limit {WRIST_LIMIT})")
    assert kick > 3 * WRIST_LIMIT, "the free kick alone passes the limit, so holding it is the controller's doing"
    assert same and peak < WRIST_LIMIT and not raised, peak
    assert stale > 2 * WRIST_LIMIT and stale_raised, stale
