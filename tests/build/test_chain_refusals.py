"""What an apparatus chain refuses, each with a message saying what to do, and each beside its control, the same
declaration without the defect, which builds or plans:

- **a load on the kinematic arm**: a law delivering into the chain's link wrench is refused, naming why and the
  dynamic rung that takes loads; the same law on an apparatus with no chain is refused for another reason, and the
  chain without the law builds;
- **a closed loop**: an arm whose tree holds a loop closure is refused, naming the closure, the invitation to ask and
  the way to register one's own segment; the same tree without it builds;
- **a refused URDF element**: a mimic joint is refused, naming it and how to acknowledge it; acknowledged with its
  reason it builds, and the saved form and the report record it;
- the mounts: a segment after the first that names no frame, or one its predecessor does not have, a first segment
  that mounts on something, a part on a frame no segment has, a provider whose part is no magnet, and the reserved
  dynamic rung.
"""

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import pytest
from chain_cases import AR4, DESCRIPTIONS, kinematic_rpm, magnet_part, rpm
from chain_cases import experiment as new_experiment

from mime.build import BuildRefused, InteractionBuilder, ThingDeclaration
from mime.core.multibody import ClosureKind, DescriptionReading, LoopClosure
from mime.dynamics import Motor, RigidInertial, RobotArm
from mime.fields import PointDipoleField
from mime.gradients import GradientStatus
from mime.registry import BuildContext, Delivery, Interaction, Problem, RoleKind, register_dynamics, \
    register_interaction
from mime.interactions import Magnetic
from mime.scene import SolidCylinder, Z, asset
from mime.things import Apparatus, Body, Part, PermanentMagnet, State
from mime.units import A_m2, kg, kg_m2, m_per_s, mm, rad, rad_per_s
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
    assert "dynamic rung" in refusal.action and "reserved" in refusal.action


def test_control_the_same_law_on_an_apparatus_with_no_chain_is_refused_otherwise_and_the_chain_alone_plans():
    exp = new_experiment("chainless")
    exp.add_apparatus(Apparatus("stand"))
    exp.attach(PushOnChain(), apparatus="stand")
    assert "target.kinematic-chain" not in codes(exp)
    kinematic_rpm("unloaded_rpm").plan()


@register_dynamics
@dataclass(frozen=True, kw_only=True)
class ClosedArm(RobotArm):
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
    exp.add_apparatus(Apparatus("bench", chain={"arm": RobotArm.Kinematic(
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
        "arm": RobotArm.Kinematic(description=asset(AR4), initial_positions=rad((0.0,) * 6), mount="nowhere"),
        "motor": Motor.Kinematic(axis=Z, prescribed="velocity", initial_angle=rad(0.0))}))
    assert "chain.mount-first" in codes(bad)
    for mount, code in ((None, "chain.mount"), ("link_9", "chain.mount")):
        exp = new_experiment("mounts")
        exp.add_apparatus(Apparatus("rpm", chain={
            "arm": RobotArm.Kinematic(description=asset(AR4), initial_positions=rad((0.0,) * 6)),
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
                                          fields={"magnetic": PointDipoleField(part="bracket")}))
    assert "provider.part" in codes(exp)


def test_the_dynamic_rung_is_reserved():
    exp = arm_alone(RobotArm(rung="dynamic", description=asset(AR4), initial_positions=rad((0.0,) * 6)))
    (refusal,) = [r for r in refused(exp) if r.code == "rung.reserved"]
    assert "reserved, not built" in refusal.message and "'kinematic'" in refusal.action


def test_a_magnet_in_a_chain_declares_its_direction():
    held = rpm()
    exp = new_experiment("direction")
    exp.add_apparatus(dataclasses.replace(held, magnets={"rotor_magnet": PermanentMagnet()}))
    assert "magnet.direction-missing" in codes(exp)


def test_a_chain_names_its_segments_holds_segments_and_turns_them_by_unit_quaternions():
    arm = RobotArm.Kinematic(description=asset(AR4), initial_positions=rad((0.0,) * 6))
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
        exp.add_apparatus(Apparatus("rpm", chain={"arm": RobotArm.Kinematic(description=description,
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
        parts={name: Part(SolidCylinder(radius=mm(0.5), length=mm(1.0), axis=(1.0, 0.0, 0.0)), at=mm((side, 0.0, 0.0)))
               for name, side in (("a", 1.0), ("b", -1.0))},
        magnets={"a": PermanentMagnet(direction=(1.0, 0.0, 0.0)), "b": PermanentMagnet(direction=(1.0, 0.0, 0.0))}))
    for name, value in (("mass", kg(8.5e-5)), ("inertia", kg_m2((1e-10, 1e-10, 1e-10))),
                        ("magnetic_moment", A_m2((1e-3, 0.0, 0.0)))):
        exp.declare(f"umr.{name}", value, reason="the body")
    exp.attach(Magnetic.Force(), body="umr", field="rpm.magnetic")
    assert "parts.several-magnets" in codes(exp)
