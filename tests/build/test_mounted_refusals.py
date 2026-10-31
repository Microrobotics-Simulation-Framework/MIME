"""What a dynamic segment refuses of the mass mounted on it (phase 7a4, design #185), each refused by name with its
message checked and each taken by the kinematic rung, whose motion no mounted mass changes:

- **a tilted rotor** (``chain.spinner-tilted``): A0's magnet turned 45 degrees off the spin axis, its products of
  inertia with the axis not zero;
- **an unbalanced rotor** (``chain.spinner-unbalanced``): the magnet, symmetric about the axis, 5 mm off it;
- **an articulated payload** (``chain.articulated-payload``): the magnet with its mass on a kinematic motor's rotor
  (its prescribed joint moves it on the dynamic arm), and a kinematic arm whose links have mass mounted on the dynamic
  arm's flange;
- **a rotor too unequal across its axis** (``chain.spinner-asymmetric``): a rod spun end over end, the term not
  carried at its worst 0.95 of the gyroscopic torque carried, above the threshold of a half; A0's magnet, spun about a
  diameter, is accepted at 0.27 with its worst case in the report;
- **the worst case unknown** (``chain.spinner-speed``): A0's magnet with no ``top_speed``, and on a hub whose joint
  declares no speed limit;
- **a spinner on a segment that publishes no angular velocities** (``chain.mount-rates``): the dynamic motor on the lab
  fixture's gantry; the kinematic motor there builds.

And the mounts: a segment on a frame of an earlier segment than the one before it (``mount="arm.link_3"``) builds,
and a name no segment has is refused; a controller is given the built tree of each segment that carries mass.
"""

import importlib
import warnings

import pytest
from chain_cases import AR4, AR4_BASE, AR4_HOME, DESCRIPTIONS, MAGNET_DENSITY, SPIN_JOINT, add_body, experiment, \
    kinematic_rpm, magnet_part, massive_rpm

from mime.build import BuildRefused
from mime.drives import Constant
from mime.dynamics import Arm, Motor
from mime.scene import Cylinder, X, Z, asset
from mime.things import Apparatus
from mime.units import N_m, N_m_s, kg_m2, kg_per_m3, m, mm, ms, rad, rad_per_s

KINEMATIC_TAKES = "the kinematic rung of 'rpm.arm', whose motion no mounted mass changes"


def refusals(experiment_):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(BuildRefused) as caught:
            experiment_.plan()
    return caught.value.refusals


def only(experiment_, code: str):
    found = refusals(experiment_)
    held = [r for r in found if r.code == code]
    assert len(held) == 1, [(r.code, r.message) for r in found]
    return held[0]


def plans(experiment_):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return experiment_.plan()


def test_a_tilted_rotor_is_refused_and_the_kinematic_arm_takes_it():
    refusal = only(massive_rpm("tilted", magnet_axis=(1.0, 0.0, 1.0)), "chain.spinner-tilted")
    assert "a tilted rotor" in refusal.message and "products of inertia with the axis" in refusal.message
    assert KINEMATIC_TAKES in refusal.action
    taken = plans(massive_rpm("tilted_kinematic", magnet_axis=(1.0, 0.0, 1.0), arm="kinematic"))
    assert any("on a prescribed motion: its mass is not carried" in line for line in taken.masses)


def test_an_unbalanced_rotor_is_refused_and_the_kinematic_arm_takes_it():
    refusal = only(massive_rpm("unbalanced", magnet_axis=(0.0, 0.0, 1.0), magnet_at=mm((5.0, 0.0, 0.0))),
                   "chain.spinner-unbalanced")
    assert "0.005 m off its axis: an unbalanced rotor" in refusal.message and KINEMATIC_TAKES in refusal.action
    plans(massive_rpm("unbalanced_kinematic", magnet_axis=(0.0, 0.0, 1.0), magnet_at=mm((5.0, 0.0, 0.0)),
                      arm="kinematic"))


def test_control_the_same_rotor_balanced_and_aligned_builds_with_its_mass_welded_on():
    held = plans(massive_rpm("balanced", magnet_axis=(0.0, 0.0, 1.0)))
    assert any("welded on rpm.arm.link_6" in line for line in held.masses)


def test_mass_on_a_kinematic_motors_rotor_on_the_dynamic_arm_is_an_articulated_payload():
    refusal = only(massive_rpm("articulated", motor="kinematic", magnet_axis=(0.0, 0.0, 1.0)),
                   "chain.articulated-payload")
    assert "part 'rpm.rotor_magnet'" in refusal.message and "an articulated payload" in refusal.message
    assert "moved by the joints of 'rpm.motor'" in refusal.message and KINEMATIC_TAKES in refusal.action
    plans(massive_rpm("articulated_kinematic", motor="kinematic", magnet_axis=(0.0, 0.0, 1.0), arm="kinematic"))


def mounted_arm(base: str):
    exp = experiment(f"arm_on_arm_{base}")
    first = Arm.Dynamic(description=asset(AR4), initial_positions=rad(AR4_HOME), at=m(AR4_BASE)) if base == "dynamic" \
        else Arm.Kinematic(description=asset(AR4), initial_positions=rad(AR4_HOME), at=m(AR4_BASE))
    commands = {"arm.joint_torques": Constant(N_m((0.0,) * 6))} if base == "dynamic" else \
        {"arm.joint_positions": Constant(rad(AR4_HOME))}
    commands["aimer.joint_positions"] = Constant(rad((0.0, 0.0, 0.0)))
    exp.add_apparatus(Apparatus("rpm", chain={"arm": first, "aimer": Arm.Kinematic(
        description=asset(DESCRIPTIONS / "three_link_planar.urdf"), initial_positions=rad((0.0, 0.0, 0.0)),
        mount="link_6")}, commands=commands))
    add_body(exp)
    return exp


def test_an_arm_whose_links_have_mass_on_the_dynamic_arm_is_an_articulated_payload():
    refusal = only(mounted_arm("dynamic"), "chain.articulated-payload")
    assert "the links of 'rpm.aimer', which its description gives mass" in refusal.message
    assert KINEMATIC_TAKES in refusal.action
    plans(mounted_arm("kinematic"))


def rod(length: float = 0.2):
    """A rod 4 mm across and ``length`` long along the rotor's x, spun about z: end over end."""
    return Cylinder(radius=mm(2.0), length=m(length), axis=(1.0, 0.0, 0.0))


def test_a_rotor_too_unequal_across_its_axis_is_refused_and_a0s_magnet_is_accepted_with_its_worst_case():
    refusal = only(massive_rpm("rod", solid=rod()), "chain.spinner-asymmetric")
    assert "unequal across its axis" in refusal.message and "above 0.5" in refusal.message
    assert KINEMATIC_TAKES in refusal.action
    plans(massive_rpm("rod_kinematic", solid=rod(), arm="kinematic"))
    accepted = plans(massive_rpm("a0_magnet"))
    worst = [line for line in accepted.masses if "the term not carried at worst" in line]
    assert len(worst) == 1 and "a share of 0.267, accepted up to 0.5" in worst[0], accepted.masses
    assert "from rpm.arm.link_6's joint speed limits and the top speed" in worst[0]


def test_the_worst_case_needs_the_top_speed_and_every_joints_speed_limit():
    refusal = only(massive_rpm("no_top_speed", top_speed=None), "chain.spinner-speed")
    assert "needs its top_speed" in refusal.message and KINEMATIC_TAKES in refusal.action
    exp = experiment("unlimited_hub")
    exp.add_apparatus(Apparatus("hub", chain={
        "arm": Arm.Dynamic(description=asset(SPIN_JOINT), initial_positions=rad((0.0,))),
        "motor": Motor.Dynamic(axis=Z, inertia=kg_m2(1e-5), damping=N_m_s(0.0), initial_angle=rad(0.0), mount="hub",
                               top_speed=rad_per_s(60.0))},
        parts={"magnet": magnet_part("motor.rotor")},
        commands={"arm.joint_torques": Constant(N_m((0.0,))), "motor.torque": Constant(N_m(0.0))}))
    exp.declare("hub.parts.magnet.density", kg_per_m3(MAGNET_DENSITY), reason="sintered NdFeB, its typical value")
    add_body(exp)
    refusal = only(exp, "chain.spinner-speed")
    assert "a speed limit on every joint that turns hub.arm.hub" in refusal.message


def gantry_spinner(rung: str):
    gantry = importlib.import_module("mime_lab_fixture").Gantry
    exp = experiment(f"gantry_{rung}")
    motor = Motor.Dynamic(axis=Z, inertia=kg_m2(1e-5), damping=N_m_s(1e-4), initial_angle=rad(0.0),
                          mount="carriage") if rung == "dynamic" else \
        Motor.Kinematic(axis=Z, prescribed="velocity", initial_angle=rad(0.0), mount="carriage")
    exp.add_apparatus(Apparatus("stage", chain={"gantry": gantry(travel=X, at=m((0.0, 0.0, 0.1))), "motor": motor},
                                commands={"gantry.position": Constant(m(0.02)),
                                          **({"motor.torque": Constant(N_m(0.0))} if rung == "dynamic" else
                                             {"motor.velocity": Constant(rad_per_s(1.0))})}))
    add_body(exp)
    return exp


def test_a_spinner_on_a_segment_that_publishes_no_angular_velocities_is_refused():
    refusal = only(gantry_spinner("dynamic"), "chain.mount-rates")
    assert "'stage.gantry' publishes none (apparatus.frame_angular_velocities)" in refusal.message
    assert "the kinematic rung of 'stage.motor'" in refusal.action
    plans(gantry_spinner("kinematic"))


def test_a_segment_mounts_on_a_frame_of_an_earlier_segment_and_an_unknown_one_is_refused():
    exp = experiment("mid_arm")
    exp.add_apparatus(Apparatus("rig", chain={
        "arm": Arm.Kinematic(description=asset(AR4), initial_positions=rad(AR4_HOME), at=m(AR4_BASE)),
        "motor": Motor.Kinematic(axis=Z, prescribed="velocity", initial_angle=rad(0.0), mount="link_6"),
        "aimer": Motor.Kinematic(axis=X, prescribed="angle", initial_angle=rad(0.0), mount="arm.link_3")},
        commands={"arm.joint_positions": Constant(rad(AR4_HOME)), "motor.velocity": Constant(rad_per_s(6.0)),
                  "aimer.angle": Constant(rad(0.2))}))
    add_body(exp)
    held = plans(exp)
    aimer = next(plan for plan in held.entries if plan.owner == "rig.aimer")
    assert (aimer.mount.segment, aimer.mount.frame, aimer.mount.rates) == ("rig.arm", "link_3",
                                                                           ("rig.arm", "frame_angular_velocities"))
    wrong = experiment("nowhere")
    wrong.add_apparatus(Apparatus("rig", chain={
        "arm": Arm.Kinematic(description=asset(AR4), initial_positions=rad(AR4_HOME)),
        "motor": Motor.Kinematic(axis=Z, prescribed="velocity", initial_angle=rad(0.0), mount="link_6"),
        "aimer": Motor.Kinematic(axis=X, prescribed="angle", initial_angle=rad(0.0), mount="arm.link_9")}))
    refusal = [r for r in refusals(wrong) if r.code == "chain.mount"]
    assert refusal and "<segment>.<frame> of an earlier segment" in refusal[0].message


def test_a_controller_is_given_the_built_tree_of_each_segment_that_carries_mass():
    held = plans(massive_rpm("models", magnet_axis=(0.0, 0.0, 1.0), extra=True))
    tree = held.models["rpm.arm"]
    assert {"mounted:rpm.camera", "mounted:rpm.stator", "mounted:rpm.motor.rotor"} <= {b.name for b in tree.bodies}
    assert "rpm.motor" not in held.models
    assert not plans(kinematic_rpm("bare_models")).models



def test_a_controllers_read_of_angular_velocities_no_segment_knows_is_refused():
    """The kinematic motor on the gantry builds, and publishes its frames' angular velocities relative to a mount that
    publishes none: a controller's read of them is refused. The control: the same read of the motor on the arm, which
    knows them, passes that check and is refused only for its units (the toy controller reads a pose)."""
    from control_cases import Ticker

    for exp, apparatus, known in ((gantry_spinner("kinematic"), "stage", False),
                                  (kinematic_rpm("read_on_the_arm"), "rpm", True)):
        exp.set_controller(Ticker(), reads={"pose": f"{apparatus}.motor.frame_angular_velocities"},
                           writes={"angle": f"{apparatus}.motor.angle"}, mode="callback", rate=ms(2), latency=ms(1))
        found = {r.code: r for r in refusals(exp)}
        if known:
            assert "chain.mount-rates" not in found and "controller.signal" in found, sorted(found)
        else:
            assert "does not know its frames' angular velocities" in found["chain.mount-rates"].message, sorted(found)
