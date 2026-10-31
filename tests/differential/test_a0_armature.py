"""A0 on its dynamic rungs and its wrist's armature (phase 7a5, design #193).

**As declared**, A0 gives its arm's joints no armature and says why (``script_a0_dynamic.NO_ARMATURE``: the 0.2 line's
arm carried none, and the comparison holds the reference's description). The build's validity line names joint_6, the
bare wrist the rotor's spin-up kicks: its ratio 4.11 (``J / I6`` with the motor on the flange's axis), and the profile
records the convention's source.

**An example, not preregistered and never compared with the reference**: A0 with an AR4-like armature on its wrist.
Its value is an inference, stated as one: a NEMA 17 stepper's rotor (the registered actuator, StepperOnline's
17HS19-2004S1, 82 g cm^2 from its drawing) behind a reduction of 10, an assumed value of a geared wrist's order (the
AR4's description gives its drive neither, and nothing here fits it): 100 x 8.2e-6 = 8.2e-4 kg m^2. A0's own
controller, whose computed-torque loop reads the built tree with the armature on it, holds the wrist inside its speed
limit at 10 Hz, the run's flag stays down, and the build gives no line. The control: the bare wrist, in the same run,
passes its limit twenty times over and raises the flag."""

import warnings

import numpy as np
from script_a0_dynamic import NO_ARMATURE, a0_dynamic

from mime.dynamics import actuator
from mime.units import Hz, kg_m2, ms

#: The AR4's wrist speed limit, from its description (rad/s).
WRIST_LIMIT = 1.5708
#: The assumed reduction of the wrist's drive: an inference, not the AR4's documented or measured gearing.
REDUCTION = 10.0
GEARED = REDUCTION ** 2 * actuator("stepperonline.17hs19-2004s1").inertia
#: The run: the spin-up's kick is over within a tenth of a second at 10 Hz (A0's own test reads it).
RUN = ms(150)


def wrist(armature):
    """The build's report and profile, the wrist's largest speed over the run, and whether the flag was raised."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = a0_dynamic("a0_wrist", frequency=Hz(10.0), armature=armature).build()
    result = built.run(RUN, history=True)
    speeds = np.abs(np.asarray(result.history["rpm.arm"]["joint_velocities"])[:, 5])
    flags = [read for read in result.flags if read.path == "rpm.arm.beyond_speed_limit"]
    return built.report, float(speeds.max()), bool(flags and flags[0].was_raised)


def test_a0_as_declared_states_its_wrists_armature_none_and_the_build_names_its_kick():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plans = a0_dynamic().plan()
    assert plans.profile.sources["rpm.arm.armature"] == NO_ARMATURE
    assert "rpm.arm.armature" not in [item.target for item in plans.profile.listed]
    (line,) = plans.checked["rpm.arm"]
    assert line.startswith("joint_6: ") and "a ratio of 4.11" in line


def test_a0s_controller_holds_a_geared_wrist_inside_its_limit_and_the_bare_wrist_passes_it():
    reason = "an inference: a 17HS19-2004S1's rotor (its drawing) behind an assumed reduction of 10"
    report, peak, raised = wrist({"joint_6": kg_m2(GEARED, reason=reason)})
    print(f"\nthe wrist's largest speed at 10 Hz: geared {peak:.4g} rad/s", end="")
    assert peak < WRIST_LIMIT and not raised, peak
    assert "rpm.arm" not in report.checked
    _, bare, bare_raised = wrist(None)
    print(f", bare {bare:.4g} rad/s (its limit {WRIST_LIMIT})")
    assert bare > 20 * WRIST_LIMIT and bare_raised, bare
