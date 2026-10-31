"""What a dynamic segment refuses of the mass mounted on it (phase 7a4, design #185), each refused by name with its
message checked and each taken by the kinematic rung, whose motion no mounted mass changes: each refused case is built
and run on the kinematic rung, its first steps finite.

- **a tilted rotor** (``chain.spinner-tilted``): A0's magnet turned 45 degrees off the spin axis, its rotating torque
  at the top speed a share of its weight on its radius of gyration far above the residue a spinner accepts;
- **an unbalanced rotor** (``chain.spinner-unbalanced``): the magnet, symmetric about the axis, 5 mm off it, and 0.1
  mm off it, each refused with the size of its rotating force at the top speed; 1e-9 m off it (a CAD placement's
  residue) and turned by a microradian are accepted, each residue in the build report;
- **a residue with no top speed to judge it** (``chain.spinner-speed``): the magnet 1e-6 m off its axis, with no
  ``top_speed``;
- **an articulated payload** (``chain.articulated-payload``): the magnet with its mass on a kinematic motor's rotor
  (its prescribed joint moves it on the dynamic arm), and a kinematic arm whose links have mass mounted on the dynamic
  arm's flange;
- **a spinner on a segment that publishes no angular velocities** (``chain.mount-rates``): the dynamic motor on the lab
  fixture's gantry; the kinematic motor there, and a kinematic arm, build and run.

And what the dynamic arm accepts: a rotor unequal across its axis, carried (A0's magnet spun about a diameter, and a rod
spun end over end, which the threshold of a half refused before the rotor's inequality was carried); a rotor on a
frame no joint turns (the arm's base). The mounts: a segment on a frame of an earlier segment than the one before it
(``mount="arm.link_3"``) builds, and a name no segment has is refused; a controller is given the built tree of each
segment that carries mass. And the frames' angular velocities, computed only where something reads them.
"""

import importlib
import warnings

import numpy as np
import pytest
from chain_cases import AR4, AR4_BASE, AR4_HOME, DESCRIPTIONS, MAGNET_DENSITY, MOTOR_OFFSET, MOTOR_TURN, \
    ROTOR_DAMPING, ROTOR_INERTIA, SPIN_JOINT, add_body, experiment, kinematic_rpm, magnet_part, massive_rpm

from mime.build import BuildRefused
from mime.drives import Constant
from mime.dynamics import Arm, Motor
from mime.scene import Cylinder, X, Z, asset
from mime.testing import first_steps
from mime.things import Apparatus, Part
from mime.units import N_m, N_m_s, kg_m2, kg_per_m3, m, mm, ms, rad, rad_per_s

KINEMATIC_TAKES = "the kinematic rung of 'rpm.arm', whose motion no mounted mass changes"
RUN_STEPS = 5


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


def runs(experiment_):
    """``experiment_`` built and run ``RUN_STEPS`` steps, every field of every node finite: its build report and its
    history."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = experiment_.build()
    history = first_steps(built, RUN_STEPS)
    for node, fields in history.items():
        for name, values in fields.items():
            assert np.all(np.isfinite(np.asarray(values))), (node, name)
    return built.report, history


def test_a_tilted_rotor_is_refused_with_its_size_and_the_kinematic_arm_takes_it():
    refusal = only(massive_rpm("tilted", magnet_axis=(1.0, 0.0, 1.0)), "chain.spinner-tilted")
    assert "a tilted rotor" in refusal.message and "products of inertia with the axis" in refusal.message
    assert "rotating torque at its top speed 62.8319 rad/s" in refusal.message, refusal.message
    assert KINEMATIC_TAKES in refusal.action
    report, _ = runs(massive_rpm("tilted_kinematic", magnet_axis=(1.0, 0.0, 1.0), arm="kinematic"))
    assert any("on a prescribed motion: its mass is not carried" in line for line in report.masses)


def test_an_unbalanced_rotor_is_refused_with_its_size_and_the_kinematic_arm_takes_it():
    refusal = only(massive_rpm("unbalanced", magnet_axis=(0.0, 0.0, 1.0), magnet_at=mm((5.0, 0.0, 0.0))),
                   "chain.spinner-unbalanced")
    assert "0.005 m off its axis: an unbalanced rotor" in refusal.message and KINEMATIC_TAKES in refusal.action
    runs(massive_rpm("unbalanced_kinematic", magnet_axis=(0.0, 0.0, 1.0), magnet_at=mm((5.0, 0.0, 0.0)),
                     arm="kinematic"))


def test_a_real_offset_is_refused_by_its_rotating_force_and_a_cad_residue_is_accepted():
    """0.1 mm off A0's axis at its top speed is a rotating force of 0.04 of the rotor's weight, refused with its size;
    1e-9 m off, and a microradian of turn, are residues, accepted with their sizes in the report."""
    refusal = only(massive_rpm("offset", magnet_axis=(0.0, 0.0, 1.0), magnet_at=mm((0.1, 0.0, 0.0))),
                   "chain.spinner-unbalanced")
    assert "0.0001 m off its axis" in refusal.message and "0.0403 of its weight" in refusal.message, refusal.message
    report, _ = runs(massive_rpm("residue", magnet_axis=(0.0, 0.0, 1.0), magnet_at=m((1e-9, 0.0, 0.0))))
    residue = [line for line in report.masses if "residues at its top speed" in line]
    assert len(residue) == 1 and "1e-09 m off its axis" in residue[0], report.masses
    turned = (0.0, float(np.sin(1e-6)), float(np.cos(1e-6)))
    report, _ = runs(massive_rpm("turned_residue", magnet_axis=turned))
    assert any("residues at its top speed" in line and "of tilt" in line for line in report.masses), report.masses


def test_a_residue_needs_the_top_speed_to_be_judged_and_the_kinematic_arm_takes_it():
    refusal = only(massive_rpm("no_top_speed", magnet_axis=(0.0, 0.0, 1.0), magnet_at=m((1e-6, 0.0, 0.0)),
                               top_speed=None), "chain.spinner-speed")
    assert "needs its top_speed" in refusal.message and KINEMATIC_TAKES in refusal.action
    runs(massive_rpm("no_top_speed_kinematic", magnet_axis=(0.0, 0.0, 1.0), magnet_at=m((1e-6, 0.0, 0.0)),
                     top_speed=None, arm="kinematic"))
    plans(massive_rpm("no_top_speed_balanced", magnet_axis=(0.0, 0.0, 1.0), top_speed=None))


def test_control_the_same_rotor_balanced_and_aligned_builds_with_its_mass_welded_on():
    held = plans(massive_rpm("balanced", magnet_axis=(0.0, 0.0, 1.0)))
    assert any("welded on rpm.arm.link_6" in line for line in held.masses)


def test_mass_on_a_kinematic_motors_rotor_on_the_dynamic_arm_is_an_articulated_payload():
    refusal = only(massive_rpm("articulated", motor="kinematic", magnet_axis=(0.0, 0.0, 1.0)),
                   "chain.articulated-payload")
    assert "part 'rpm.rotor_magnet'" in refusal.message and "an articulated payload" in refusal.message
    assert "moved by the joints of 'rpm.motor'" in refusal.message and KINEMATIC_TAKES in refusal.action
    runs(massive_rpm("articulated_kinematic", motor="kinematic", magnet_axis=(0.0, 0.0, 1.0), arm="kinematic"))


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
    runs(mounted_arm("kinematic"))


def rod(length: float = 0.2):
    """A rod 4 mm across and ``length`` long along the rotor's x, spun about z: end over end."""
    return Cylinder(radius=mm(2.0), length=m(length), axis=(1.0, 0.0, 0.0))


def test_a_rotor_unequal_across_its_axis_is_carried_on_the_dynamic_arm():
    """A0's magnet spun about a diameter, and a rod spun end over end (refused before its inequality was carried):
    each builds and runs, its half difference across the axis turned with its spin and handed to the arm."""
    for name, solid in (("a0_magnet", None), ("rod", rod())):
        report, history = runs(massive_rpm(name, solid=solid))
        lines = [line for line in report.masses if "their half difference" in line]
        assert len(lines) == 1 and "turned with the spin" in lines[0], report.masses
        assert np.any(np.asarray(history["rpm.motor"]["inertia"]) != 0.0)
    held = [line for line in plans(massive_rpm("a0_magnet_report")).masses if "across its axis: moments" in line]
    assert "moments 2.20985e-05 and 1.58598e-05" in held[0] and "half difference 3.11935e-06" in held[0], held


def test_a_rotor_on_a_frame_no_joint_turns_is_accepted():
    """A0's magnet spun about a diameter on the dynamic arm's base: no joint turns the frame, so nothing of the rotor
    moves the arm, and it is accepted (0 / 0 was read as infinity before)."""
    exp = experiment("on_the_base")
    exp.add_apparatus(Apparatus("rpm", chain={
        "arm": Arm.Dynamic(description=asset(AR4), initial_positions=rad(AR4_HOME), at=m(AR4_BASE)),
        "motor": Motor.Dynamic(axis=Z, inertia=kg_m2(ROTOR_INERTIA), damping=N_m_s(ROTOR_DAMPING),
                               initial_angle=rad(0.0), mount="base_link", at=m(MOTOR_OFFSET), orientation=MOTOR_TURN,
                               top_speed=rad_per_s(60.0))},
        parts={"magnet": magnet_part("motor.rotor")},
        commands={"arm.joint_torques": Constant(N_m((0.0,) * 6)), "motor.torque": Constant(N_m(1e-3))}))
    exp.declare("rpm.parts.magnet.density", kg_per_m3(MAGNET_DENSITY), reason="sintered NdFeB, its typical value")
    add_body(exp)
    report, history = runs(exp)
    assert any("welded on rpm.arm.base_link" in line for line in report.masses), report.masses
    assert float(np.max(np.abs(np.asarray(history["rpm.motor"]["spin"])))) > 0.0


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
    _, history = runs(gantry_spinner("kinematic"))
    assert float(np.max(np.abs(np.diff(np.asarray(history["stage.motor"]["angle"]))))) > 0.0


def test_a_kinematic_arm_on_a_segment_that_publishes_no_angular_velocities_builds_and_runs():
    """The lab fixture's gantry carrying a kinematic arm, which reads no rates where the gantry publishes none (no
    edge feeds them, and nothing reads the arm's own): it builds and runs, its frames carried by the carriage."""
    gantry = importlib.import_module("mime_lab_fixture").Gantry
    exp = experiment("gantry_arm")
    exp.add_apparatus(Apparatus("stage", chain={
        "gantry": gantry(travel=X, at=m((0.0, 0.0, 0.1))),
        "arm": Arm.Kinematic(description=asset(DESCRIPTIONS / "three_link_planar.urdf"),
                             initial_positions=rad((0.0, 0.0, 0.0)), mount="carriage"),
        "motor": Motor.Kinematic(axis=Z, prescribed="velocity", initial_angle=rad(0.0), mount="arm.link_3")},
        commands={"gantry.position": Constant(m(0.02)), "arm.joint_positions": Constant(rad((0.1, 0.2, 0.3))),
                  "motor.velocity": Constant(rad_per_s(1.0))}))
    add_body(exp)
    _, history = runs(exp)
    assert "frame_angular_velocities" not in history["stage.arm"]


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


def test_the_kinematic_rungs_compute_their_frames_rates_only_where_something_reads_them():
    """A0's kinematic apparatus, which nothing reads the rates of, steps its arm and motor as the base did, their frame
    poses alone; a dynamic motor on the kinematic arm reads the arm's; a controller reading the motor's reads the
    motor's and, through it, the arm's."""
    from control_cases import RateWatcher

    _, unread = runs(kinematic_rpm("unread_rates"))
    assert not {"frame_angular_velocities"} & ({*unread["rpm.arm"]} | {*unread["rpm.motor"]})
    _, spun = runs(massive_rpm("read_by_a_spinner", arm="kinematic"))
    assert "frame_angular_velocities" in spun["rpm.arm"]
    exp = kinematic_rpm("read_by_a_controller", prescribed="angle", controlled=("motor.angle",))
    exp.set_controller(RateWatcher(), reads={"rates": "rpm.motor.frame_angular_velocities"},
                       writes={"angle": "rpm.motor.angle"}, mode="callback", rate=ms(2), latency=ms(1))
    _, read = runs(exp)
    assert "frame_angular_velocities" in read["rpm.motor"] and "frame_angular_velocities" in read["rpm.arm"]


def test_the_build_report_names_the_built_trees_a_controllers_law_read():
    """A controller whose law reads the dynamic arm's built tree is reported as reading it; A0's tracking controller,
    which reads a description of its own, as reading none."""
    from chain_cases import tracked_rpm
    from control_cases import ModelReader

    exp = massive_rpm("model_reader", magnet_axis=(0.0, 0.0, 1.0), controlled=("motor.torque",))
    exp.set_controller(ModelReader(), reads={"joints": "rpm.arm.joint_positions"}, writes={"torque": "rpm.motor.torque"},
                       mode="callback", rate=ms(2), latency=ms(1))
    report, _ = runs(exp)
    assert any("its law read the built tree of rpm.arm" in line for line in report.controller), report.controller
    report, _ = runs(tracked_rpm("tracked_models"))
    assert any("its law read none of the built trees (ctx.models offers none)" in line for line in report.controller)
