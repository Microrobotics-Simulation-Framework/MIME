"""What an apparatus chain refuses, each with a message saying what to do, and each beside its control, the same
declaration without the defect, which builds or plans:

- **a load on the kinematic arm**: a law delivering into the chain's link wrench is refused, naming why and the
  dynamic rung that takes loads; the same law on an apparatus with no chain is refused for another reason, and the
  chain without the law builds; on the dynamic arm, a law naming the apparatus and no link is refused, a load reaching
  a link as the reaction on a field's source;
- **a closed loop**: an arm whose tree holds a loop closure is refused, naming the closure, the invitation to ask and
  the way to register one's own segment; the same tree without it builds;
- **a refused URDF element**: a mimic joint is refused, naming it and how to acknowledge it; acknowledged with its
  reason it builds, and the saved form and the report record it;
- **a reader inside the magnet**: a body that starts within the sphere that holds a point dipole's magnet is refused,
  naming the distance and the radius; the same body outside it plans;
- **several magnets on an apparatus**, and **an initial position beyond a joint's limit**, each refused with the
  limits or the magnets named; the control, one magnet and positions inside the limits, plans;
- the mounts: a segment after the first that names no frame, or one its predecessor does not have, a first segment
  that mounts on something, a part on a frame no segment has, a provider whose part is no magnet;
- **the dynamic rungs**: an arm whose moving body has geometry and no inertial (``chain.inertia-gap``), one whose
  moving joint moves a body with neither geometry nor inertial and so no mass at all (``chain.massless-joint``), a
  dynamic arm mounted on another segment (``chain.fixed-base``), a servo on it, initial velocities on the kinematic
  arm, a dynamic motor with no inertia or with a prescription, each refused with its message and what to do; the
  control: the same arms on their kinematic rung, and the dynamic arm first in its chain carrying the dynamic motor,
  build.
"""

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import pytest
from chain_cases import AR4, AR4_HOME, DESCRIPTIONS, dynamic_rpm, fixed_magnet, kinematic_rpm, magnet_part, rpm
from chain_cases import experiment as new_experiment

from mime.build import BuildRefused, InteractionBuilder, ThingDeclaration
from mime.core.multibody import ClosureKind, DescriptionReading, LoopClosure
from mime.dynamics import Arm, Motor, RigidInertial
from mime.fields import PointDipole
from mime.gradients import GradientStatus
from mime.registry import BuildContext, Delivery, Interaction, Problem, RoleKind, register_dynamics, \
    register_interaction
from mime.interactions import Magnetic
from mime.scene import Cylinder, Z, asset
from mime.things import Apparatus, Body, Part, PermanentMagnet, State
from mime.units import A_m2, N_m, N_m_s, kg, kg_m2, m, m_per_s, mm, ms, rad, rad_per_s
from mime.vocabulary import LINK_WRENCH, register_namespace

register_namespace("chainrefusals", "0.0.1")


@register_interaction
@dataclass(frozen=True, kw_only=True)
class PushOnChain(Interaction):
    """A law that loads an apparatus's chain."""

    name = "chainrefusals.push"
    version = "0.1.0"
    differentiability = GradientStatus.EXACT
    roles = {"apparatus": RoleKind.APPARATUS}
    delivers = {"push": Delivery("apparatus", LINK_WRENCH.name)}

    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        return None


def refused(experiment) -> list:
    with pytest.raises(BuildRefused) as caught:
        experiment.plan()
    return caught.value.refusals


def codes(experiment) -> set[str]:
    return {r.code for r in refused(experiment)}


def test_a_load_on_the_kinematic_arm_is_refused_saying_why_and_which_rung_takes_one():
    exp = kinematic_rpm("loaded_rpm")
    exp.attach(PushOnChain(), apparatus="rpm")
    (refusal,) = [r for r in refused(exp) if r.code == "target.kinematic-chain"]
    assert "prescribed" in refusal.message and "the dynamic rung takes loads" in refusal.message
    assert "dynamic rung" in refusal.action


def test_a_law_naming_a_dynamic_chain_and_no_link_is_refused_saying_how_a_load_reaches_one():
    exp = dynamic_rpm("pushed_rpm")
    exp.attach(PushOnChain(), apparatus="rpm")
    (refusal,) = [r for r in refused(exp) if r.code == "target.chain-link"]
    assert "naming no link" in refusal.message and "reaction on a field's source" in refusal.message


def test_control_the_same_law_on_an_apparatus_with_no_chain_is_refused_otherwise_and_the_chain_alone_plans():
    exp = new_experiment("chainless")
    exp.add_apparatus(Apparatus("stand"))
    exp.attach(PushOnChain(), apparatus="stand")
    assert "target.kinematic-chain" not in codes(exp)
    kinematic_rpm("unloaded_rpm").plan()


@register_dynamics
@dataclass(frozen=True, kw_only=True)
class ClosedArm(Arm):
    """The AR4 with a loop closure planted between its flange and its base, as a tree built in code could hold."""

    name = "chainrefusals.closed_arm"
    closed: bool = True

    def read(self) -> DescriptionReading | Problem:
        reading = super().read()
        if isinstance(reading, Problem) or not self.closed:
            return reading
        closure = LoopClosure("flange_to_base", ClosureKind.WELD, "link_6", "base_link")
        return DescriptionReading(dataclasses.replace(reading.tree, closures=(closure,)), reading.classification)


def arm_alone(arm) -> object:
    exp = new_experiment("arm_alone")
    exp.add_apparatus(Apparatus("rpm", chain={"arm": arm},
                                commands={"arm.joint_positions": rad((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))}))
    return exp


def test_a_closed_chain_is_refused_with_the_invitation():
    arm = ClosedArm(rung="kinematic", description=asset(AR4), initial_positions=rad((0.0,) * 6))
    (refusal,) = [r for r in refused(arm_alone(arm)) if r.code == "chain.closed-loop"]
    assert "flange_to_base" in refusal.message and "issues" in refusal.message
    assert "register your own chain segment" in refusal.action


def test_control_the_same_tree_without_its_closure_plans():
    arm_alone(ClosedArm(rung="kinematic", description=asset(AR4), initial_positions=rad((0.0,) * 6),
                        closed=False)).plan()


def mimic_file(tmp_path: Path) -> Path:
    """The three-link planar arm with a mimic on its last joint."""
    text = (DESCRIPTIONS / "three_link_planar.urdf").read_text()
    head, tail = text.rsplit("</joint>", 1)
    path = tmp_path / "planar_mimic.urdf"
    path.write_text(head + '  <mimic joint="joint_1" multiplier="1.0"/>\n  </joint>' + tail)
    return path


def planar(description: Path, acknowledged=()) -> object:
    exp = new_experiment("planar_mimic")
    exp.add_apparatus(Apparatus("bench", chain={"arm": Arm.Kinematic(
        description=asset(description), initial_positions=rad((0.0, 0.0, 0.0)), acknowledged=acknowledged)},
        commands={"arm.joint_positions": rad((0.1, 0.2, 0.3))}))
    return exp


def test_a_refused_urdf_element_is_refused_naming_it_and_how_to_acknowledge_it(tmp_path):
    path = mimic_file(tmp_path)
    (refusal,) = [r for r in refused(planar(path)) if r.code == "description.refused"]
    assert "joint[joint_3]/mimic" in refusal.message and "acknowledged" in refusal.message


def test_control_acknowledged_with_its_reason_it_builds_and_is_recorded(tmp_path):
    path = mimic_file(tmp_path)
    built = planar(path, acknowledged=(("joint[joint_3]/mimic", "read as independent: the test moves it alone"),)
                   ).build()
    assert any("read past as acknowledged" in line for line in built.report.descriptions)
    record = built.record.resolved["descriptions"]["bench.arm"]
    assert record["acknowledged"] == {"joint[joint_3]/mimic": "read as independent: the test moves it alone"}
    planar(DESCRIPTIONS / "three_link_planar.urdf").plan()


def test_each_mount_problem_is_refused_with_what_to_write():
    bad = new_experiment("mounts")
    bad.add_apparatus(Apparatus("rpm", chain={
        "arm": Arm.Kinematic(description=asset(AR4), initial_positions=rad((0.0,) * 6), mount="nowhere"),
        "motor": Motor.Kinematic(axis=Z, prescribed="velocity", initial_angle=rad(0.0))}))
    assert "chain.mount-first" in codes(bad)
    for mount, code in ((None, "chain.mount"), ("link_9", "chain.mount")):
        exp = new_experiment("mounts")
        exp.add_apparatus(Apparatus("rpm", chain={
            "arm": Arm.Kinematic(description=asset(AR4), initial_positions=rad((0.0,) * 6)),
            "motor": Motor.Kinematic(axis=Z, prescribed="velocity", initial_angle=rad(0.0), mount=mount)}))
        found = [r for r in refused(exp) if r.code == code]
        assert found and "link_6" in found[0].action, found


def test_a_part_on_no_frame_and_a_provider_on_no_magnet_are_refused():
    held = rpm()
    exp = new_experiment("parts")
    exp.add_apparatus(dataclasses.replace(held, parts={"rotor_magnet": magnet_part("motor.stator")}))
    assert "part.mount" in codes(exp)
    exp = new_experiment("parts")
    exp.add_apparatus(dataclasses.replace(held, parts={**held.parts, "bracket": magnet_part("arm.link_6")},
                                          fields={"magnetic": PointDipole(part="bracket")}))
    assert "provider.part" in codes(exp)


def dynamic_alone(description, joints: int, **settings) -> object:
    exp = new_experiment("dynamic_alone")
    exp.add_apparatus(Apparatus("rig", chain={"arm": Arm.Dynamic(description=asset(description),
                                                                  initial_positions=rad((0.0,) * joints), **settings)},
                                commands={"arm.joint_torques": N_m((0.0,) * joints)}))
    return exp


def test_a_dynamic_arm_with_a_massless_moving_body_is_refused_naming_it_and_the_kinematic_arm_takes_it():
    massless = DESCRIPTIONS / "massless_link.urdf"
    (refusal,) = [r for r in refused(dynamic_alone(massless, 2)) if r.code == "chain.inertia-gap"]
    assert "fore" in refusal.message and "<inertial>" in refusal.action and "kinematic rung" in refusal.action
    exp = new_experiment("kinematic_alone")
    exp.add_apparatus(Apparatus("rig", chain={"arm": Arm.Kinematic(description=asset(massless),
                                                                    initial_positions=rad((0.0, 0.0)))},
                                commands={"arm.joint_positions": rad((0.0, 0.0))}))
    exp.plan()


def test_a_dynamic_arm_whose_joint_moves_no_mass_is_refused_naming_it_and_the_kinematic_arm_takes_it():
    """A link with neither geometry nor inertial records no inertia gap; its joint has no inertia about its axis, so
    the mass matrix would have a zero row and the step would run to NaN."""
    empty = DESCRIPTIONS / "empty_link.urdf"
    (refusal,) = [r for r in refused(dynamic_alone(empty, 2)) if r.code == "chain.massless-joint"]
    assert "(joint_2)" in refusal.message and "joint_1" not in refusal.message
    assert "no inertia about their axes" in refusal.message and "neither geometry nor an <inertial>" in refusal.message
    assert "<inertial>" in refusal.action and "armature" in refusal.action and "kinematic rung" in refusal.action
    exp = new_experiment("kinematic_empty")
    exp.add_apparatus(Apparatus("rig", chain={"arm": Arm.Kinematic(description=asset(empty),
                                                                    initial_positions=rad((0.0, 0.0)))},
                                commands={"arm.joint_positions": rad((0.0, 0.0))}))
    exp.plan()


def only(experiment, code: str):
    """The one refusal of ``experiment`` with ``code``."""
    (found,) = [r for r in refused(experiment) if r.code == code]
    return found


def test_the_dynamic_rungs_refuse_what_they_cannot_take_and_build_what_they_can():
    planar = DESCRIPTIONS / "three_link_planar.urdf"
    servo = only(dynamic_alone(planar, 3, servo=ms(5)), "setting.value")
    assert "servo is the kinematic rung's: the dynamic arm follows the torques it is given" in servo.message
    exp = new_experiment("kinematic_rates")
    exp.add_apparatus(Apparatus("rig", chain={"arm": Arm.Kinematic(description=asset(planar),
                                                                    initial_positions=rad((0.0,) * 3),
                                                                    initial_velocities=rad_per_s((0.0,) * 3))},
                                commands={"arm.joint_positions": rad((0.0,) * 3)}))
    rates = only(exp, "setting.value")
    assert "initial_velocities is the dynamic rung's: the kinematic arm's joints move as commanded" in rates.message
    for motor, said in ((Motor.Dynamic(axis=Z, initial_angle=rad(0.0), mount="link_6"),
                         "the dynamic motor needs its rotor's moment of inertia about its axis, above zero"),
                        (Motor.Dynamic(axis=Z, inertia=kg_m2(1e-5), prescribed="velocity", initial_angle=rad(0.0),
                                       mount="link_6"),
                         "prescribed is the kinematic rung's: the dynamic motor is turned by the torque commanded")):
        exp = new_experiment("dynamic_motor")
        exp.add_apparatus(Apparatus("rpm", chain={
            "arm": Arm.Dynamic(description=asset(AR4), initial_positions=rad(AR4_HOME)), "motor": motor},
            commands={"arm.joint_torques": N_m((0.0,) * 6), "motor.torque": N_m(0.0)}))
        assert said in only(exp, "setting.value").message
    exp = new_experiment("mounted_dynamic")
    exp.add_apparatus(Apparatus("rpm", chain={
        "arm": Arm.Kinematic(description=asset(AR4), initial_positions=rad(AR4_HOME)),
        "second": Arm.Dynamic(description=asset(planar), initial_positions=rad((0.0,) * 3), mount="link_6")},
        commands={"arm.joint_positions": rad(AR4_HOME), "second.joint_torques": N_m((0.0,) * 3)}))
    fixed = only(exp, "chain.fixed-base")
    assert "the dynamic arm's base is fixed in the world, and it is mounted on 'link_6'" in fixed.message
    assert "its chain's first segment" in fixed.action
    dynamic_rpm("dynamic_controls").build()
    exp = new_experiment("damped_motor")
    exp.add_apparatus(Apparatus("rpm", chain={
        "arm": Arm.Dynamic(description=asset(AR4), initial_positions=rad(AR4_HOME)),
        "motor": Motor.Dynamic(axis=Z, inertia=kg_m2(1e-5), damping=N_m_s(1e-4), initial_angle=rad(0.0),
                               mount="link_6")},
        commands={"arm.joint_torques": N_m((0.0,) * 6), "motor.torque": N_m(1e-3)}))
    exp.plan()


def test_a_magnet_in_a_chain_declares_its_direction():
    held = rpm()
    exp = new_experiment("direction")
    exp.add_apparatus(dataclasses.replace(held, magnets={"rotor_magnet": PermanentMagnet()}))
    assert "magnet.direction-missing" in codes(exp)


def test_a_chain_names_its_segments_holds_segments_and_turns_them_by_unit_quaternions():
    arm = Arm.Kinematic(description=asset(AR4), initial_positions=rad((0.0,) * 6))
    for chain, code in (({"Arm": arm}, "chain.segment-name"), ({"arm": RigidInertial()}, "chain.not-a-segment"),
                        ({"arm": dataclasses.replace(arm, orientation=(1.0, 1.0, 0.0, 0.0))}, "chain.orientation"),
                        ({"arm": dataclasses.replace(arm, initial_positions=rad((0.0,) * 5))}, "chain.initial")):
        exp = new_experiment("segments")
        exp.add_apparatus(Apparatus("rpm", chain=chain))
        assert code in codes(exp), code


def test_a_description_missing_changed_or_not_urdf_is_refused(tmp_path):
    broken = tmp_path / "broken.urdf"
    broken.write_text("<robot name='x'><link name='a'>")
    for description, code in ((asset(tmp_path / "missing.urdf"), "description.missing"),
                              (asset(AR4, sha256="0" * 64), "description.changed"),
                              (asset(broken), "description.invalid")):
        exp = new_experiment("descriptions")
        exp.add_apparatus(Apparatus("rpm", chain={"arm": Arm.Kinematic(description=description,
                                                                             initial_positions=rad((0.0,) * 6))}))
        assert code in codes(exp), code


def test_an_apparatus_part_lies_inside_nothing():
    held = rpm()
    exp = new_experiment("inside")
    exp.add_apparatus(dataclasses.replace(held, parts={"rotor_magnet": dataclasses.replace(
        magnet_part("motor.rotor"), inside="elsewhere")}))
    assert "part.apparatus-material" in codes(exp)


def test_a_force_on_a_body_with_two_magnets_asks_for_its_lever():
    """The force acts where the magnet sits; a body with two magnet parts has no one centroid to give it."""
    exp = new_experiment("two_magnets")
    held = kinematic_rpm("two_magnets_source")
    for declaration in held._declarations:
        if isinstance(declaration, ThingDeclaration) and declaration.thing.name == "rpm":
            exp.add_apparatus(declaration.thing)
    exp.declare("rpm.magnetic_moment", A_m2(18.89), reason="the source")
    exp.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=mm((0.0, 0.0, -1.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))),
        parts={name: Part(Cylinder(radius=mm(0.5), length=mm(1.0), axis=(1.0, 0.0, 0.0)), at=mm((side, 0.0, 0.0)))
               for name, side in (("a", 1.0), ("b", -1.0))},
        magnets={"a": PermanentMagnet(direction=(1.0, 0.0, 0.0)), "b": PermanentMagnet(direction=(1.0, 0.0, 0.0))}))
    for name, value in (("mass", kg(8.5e-5)), ("inertia", kg_m2((1e-10, 1e-10, 1e-10))),
                        ("magnetic_moment", A_m2((1e-3, 0.0, 0.0)))):
        exp.declare(f"umr.{name}", value, reason="the body")
    exp.attach(Magnetic.Force(), body="umr", field="rpm.magnetic")
    assert "parts.several-magnets" in codes(exp)


def test_a_body_that_starts_inside_the_magnet_is_refused_naming_the_distance_and_the_radius():
    """The magnet's sphere is 20.2 mm in radius (a cylinder 35 mm across and 20 mm long): a body 5 mm from its centroid
    is inside it, and so is one at the centroid, where the field is singular."""
    for position in ((0.0, 0.0, 0.1), (0.005, 0.0, 0.1)):
        (refusal,) = [r for r in refused(fixed_magnet("inside_magnet", body_position=m(position)))
                      if r.code == "provider.reader-inside"]
        assert "'umr'" in refusal.message and "0.0201556 m sphere" in refusal.message, position
        assert "outside the magnet" in refusal.action


def test_control_the_same_body_outside_the_magnets_sphere_plans():
    fixed_magnet("outside_magnet", body_position=m((0.0, 0.0, 0.1 - 0.0202))).plan()
    fixed_magnet("far_from_magnet").plan()


def test_an_apparatus_with_two_magnets_is_refused():
    held = rpm()
    exp = new_experiment("two_magnets_on_an_apparatus")
    exp.add_apparatus(dataclasses.replace(
        held, parts={**held.parts, "second": magnet_part("motor.rotor")},
        magnets={**held.magnets, "second": PermanentMagnet(direction=(0.0, 1.0, 0.0))}))
    (refusal,) = [r for r in refused(exp) if r.code == "magnet.several"]
    assert "rotor_magnet" in refusal.message and "second" in refusal.message


def test_an_initial_position_beyond_a_joints_limit_is_refused_naming_the_joint_and_its_limits():
    """The AR4's first joint stops at 2.79 rad; starting it at 10 would jump to the limit at the first step."""
    beyond = (10.0, *AR4_HOME[1:])
    exp = new_experiment("beyond")
    exp.add_apparatus(Apparatus("rpm", chain={"arm": Arm.Kinematic(description=asset(AR4),
                                                                         initial_positions=rad(beyond))},
                                commands={"arm.joint_positions": rad(AR4_HOME)}))
    (refusal,) = [r for r in refused(exp) if r.code == "chain.initial"]
    assert "joint_1 = 10" in refusal.message and "limits -2.79253 to 2.79253" in refusal.message
    assert refusal.action == "start every joint within its limits"
    kinematic_rpm("within").plan()
