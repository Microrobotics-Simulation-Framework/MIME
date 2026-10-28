"""Phase 1 on the body side's new declarations: declared values, the tube and its typed paths, a contact's binding to
a surface, an apparatus's commands and their drives, directions given by reference. Each refusal names what to write.
"""

from dataclasses import dataclass, replace
from typing import Any

import pytest
from script_a import script_a

from mime import Apparatus, Body, Experiment, Medium, State, Structure
from mime.build import BuildRefused
from mime.drives import Constant
from mime.dynamics import RigidInertial
from mime.fields import UniformRotatingField
from mime.gradients import GradientStatus
from mime.interactions import Contact, Gravity, Magnetic
from mime.registry import Dynamics, Plan, PlanContext, Value, register_dynamics, register_interaction
from mime.scene import Scene, Tube, X, Z
from mime.testing.toys import ToyField
from mime.things import DirectionPath, RegionPath, SurfacePath, ThingKind
from mime.units import Hz, kg, m, m_per_s, mm, mT, rad_per_s, s, standard_gravity
from mime.vocabulary import register_namespace


def codes(experiment: Experiment) -> list[str]:
    with pytest.raises(BuildRefused) as refused:
        experiment.plan()
    print(f"\n{refused.value}")
    return refused.value.codes()


def message(experiment: Experiment, code: str) -> str:
    with pytest.raises(BuildRefused) as refused:
        experiment.plan()
    (found,) = [r for r in refused.value.refusals if r.code == code]
    return f"{found.message} | {found.action}"


def test_script_a_plans_and_builds():
    built = script_a().build()
    assert [line.node for line in built.report.nodes] == [
        "field.magnetic", "drag:umr", "gravity:umr", "magnetic.torque:umr", "umr", "swim_speed:umr",
        "drive:field.amplitude", "drive:field.frequency"]


# -- the tube --------------------------------------------------------------------------------------------------------

def test_a_tubes_regions_surfaces_and_axis_are_typed_paths_from_its_handle():
    experiment = Experiment("tube", scene=Scene.new(up=Z, units="m"), gravity=standard_gravity(), dt=s(1e-3))
    tube = experiment.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=mm(3), wall=mm(1),
                                                                    length=mm(50))))
    assert (type(tube.lumen), type(tube.inner_surface), type(tube.axis)) == (RegionPath, SurfacePath, DirectionPath)
    assert str(tube.inner_surface) == "tube.inner_surface" and tube.axis.owner == "tube"
    with pytest.raises(AttributeError, match="Did you mean 'inner_surface'"):
        _ = tube.inner_surfac
    assert -X == (-1.0, 0.0, 0.0) and +Z == (0.0, 0.0, 1.0)


def test_a_tube_that_is_not_one_is_refused_naming_the_setting():
    experiment = script_a()
    experiment._declarations[0] = replace(experiment._declarations[0], thing=Structure(
        "tube", geometry=Tube(axis=(0, 0, 0), inner_radius=mm(-3), wall=mm(1.6), length=mm(100))))
    found = codes(experiment)
    assert "setting.direction" in found and "setting.value" not in found  # an incomplete tube is not judged further
    experiment._declarations[0] = replace(experiment._declarations[0], thing=Structure(
        "tube", geometry=Tube(axis=+X, inner_radius=mm(-3), wall=mm(1.6), length=mm(100))))
    assert "inner_radius is -0.003 m, not a length above zero" in message(experiment, "setting.value")


def test_a_geometrys_own_axis_is_written_as_numbers():
    experiment = Experiment("tubes", scene=Scene.new(up=Z, units="m"), gravity=standard_gravity(), dt=s(1e-3))
    first = experiment.add_structure(Structure("first", geometry=Tube(axis=+X, inner_radius=mm(3), wall=mm(1),
                                                                      length=mm(50))))
    experiment.add_structure(Structure("second", geometry=Tube(axis=first.axis, inner_radius=mm(3), wall=mm(1),
                                                               length=mm(50))))
    assert "are written as three numbers" in message(experiment, "setting.direction")


def test_a_medium_fills_a_region_and_a_misspelt_or_wrong_one_is_refused():
    experiment = script_a()
    assert experiment.plan().fills == {"water": "tube.lumen"}
    experiment._declarations[1] = replace(experiment._declarations[1], thing=Medium("water", fills="tube.lumen_"))
    assert "Did you mean 'lumen'" in message(experiment, "binding.unknown")
    experiment._declarations[1] = replace(experiment._declarations[1], thing=Medium(
        "water", fills=SurfacePath("tube", "inner_surface")))
    assert "takes a region" in message(experiment, "binding.wrong-kind")


def test_a_direction_by_reference_is_the_referred_unit_vector():
    plans = script_a().plan()
    (provider,) = [p for p in plans.entries if p.owner == "field.magnetic"]
    assert provider.settings["rotation_axis"] == (1.0, 0.0, 0.0)
    (observer,) = [p for p in plans.entries if p.owner == "swim_speed:umr"]
    assert observer.settings["along"] == (1.0, 0.0, 0.0)


# -- declared values -------------------------------------------------------------------------------------------------

def test_a_read_nothing_provides_is_refused_with_the_line_to_declare_it():
    experiment = script_a()
    experiment._declarations = [d for d in experiment._declarations if getattr(d, "path", None) != "water.density"]
    text = message(experiment, "value.undeclared")
    assert "gravity:umr reads water.density, the medium's mass density" in text
    assert "exp.declare('water.density', <a quantity of density (kg/m^3)>, reason=\"...\")" in text


@pytest.mark.parametrize(("path", "value", "reason", "code"), [
    ("probe.mass", kg(1.0), "x", "value.unknown-thing"),
    ("umr.mas", kg(1.0), "x", "value.unknown"),
    ("umr.mass", m(1.0), "x", "value.units"),
    ("umr.inertia", kg(1.0), "x", "value.units"),
    ("umr.mass", kg(1.0), "", "value.reason"),
    ("umr.mass", 1.0, "x", "value.units"),
])
def test_a_declared_value_that_is_not_one_is_refused(path, value, reason, code):
    experiment = Experiment("values", scene=Scene.new(up=Z, units="m"), gravity=standard_gravity(), dt=s(1e-3))
    experiment.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=m((0, 0, 0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0, 0, 0)),
        angular_velocity=rad_per_s((0, 0, 0)))))
    experiment.declare(path, value, reason=reason)
    assert code in codes(experiment)


def test_a_value_declared_twice_is_refused():
    experiment = script_a()
    experiment.declare("umr.mass", kg(1.0), reason="again")
    assert "value.duplicate" in codes(experiment)


def test_the_report_labels_each_value_declared_in_the_experiment_with_its_reason_and_readers():
    report = script_a().build().report
    (mass,) = [v for v in report.values if v.path == "umr.mass"]
    in_place = ("declared in the experiment, in place of the value its parts would give (none: body 'umr''s mass and "
                "inertia are derived from its parts, and 'shell' has no density)")
    assert (mass.source, mass.read_by) == (in_place, ("gravity:umr", "umr"))
    assert f"umr.mass = 8.491116020725203e-05 kg  [{in_place}: A's reference provenance" in report.text()


def test_gravity_reads_the_pose_only_with_its_centre_of_buoyancy_off_the_centre_of_mass():
    def reads(centre):
        experiment = script_a()
        experiment._declarations = [d for d in experiment._declarations
                                    if getattr(d, "path", None) != "umr.centre_of_buoyancy"]
        experiment.declare("umr.centre_of_buoyancy", m(centre), reason="test")
        (plan,) = [p for p in experiment.plan().entries if p.owner == "gravity:umr"]
        return sorted(plan.reads)

    assert reads((0.0, 0.0, 0.0)) == []
    assert reads((0.0, 0.0, 1e-4)) == ["pose"]


# -- contact ---------------------------------------------------------------------------------------------------------

def test_the_penalty_rung_is_reserved_and_no_rung_is_refused():
    experiment = script_a()
    experiment.attach(Contact(rung="penalty", margin=mm(0.1)), body="umr", surface="tube.inner_surface")
    assert "the rung 'penalty' is reserved, not built" in message(experiment, "rung.reserved")
    experiment = script_a()
    experiment._declarations = [d for d in experiment._declarations if not isinstance(getattr(d, "law", None),
                                                                                         Contact)]
    experiment.attach(Contact(margin=mm(0.1)), body="umr", surface="tube.inner_surface")
    assert "rung.missing" in codes(experiment)


register_namespace("bodyside", "0.0.1")


@register_interaction
@dataclass(frozen=True, kw_only=True)
class MovingContact(Contact):
    """A projection that declares it follows moving geometry, so the geometry rule lets it bind a moving thing's
    surface; what it binds is then refused as reserved, not built."""

    name = "bodyside.moving_contact"
    moving = True


def without_contact(experiment: Experiment) -> Experiment:
    experiment._declarations = [d for d in experiment._declarations if not isinstance(getattr(d, "law", None),
                                                                                         Contact)]
    return experiment


def test_projection_against_a_bodys_surface_is_refused_as_reserved():
    experiment = without_contact(script_a())
    experiment.attach(Contact.Projection(margin=mm(0.1)), body="umr", surface="umr.surface")
    assert "reads the geometry of body 'umr', which moves" in message(experiment, "geometry.moving-read")
    experiment = without_contact(script_a())
    experiment.attach(MovingContact(rung="projection", margin=mm(0.1)), body="umr", surface="umr.surface")
    assert "projection between two moving things is a constraint across several bodies" in message(
        experiment, "binding.moving-surface")


@register_dynamics
@dataclass(frozen=True, kw_only=True)
class Swaying(Dynamics):
    """A structure that moves: it builds nothing, and is here only to be refused as a confining surface."""

    name = "bodyside.swaying"
    version = "0.0.1"
    differentiability = GradientStatus.EXACT
    applies_to = frozenset({ThingKind.STRUCTURE})
    nodes = ()

    def build(self, ctx: Any, b: Any, registrations: Any) -> None:
        return None


def swaying_tube(experiment: Experiment) -> Experiment:
    experiment._declarations[0] = replace(experiment._declarations[0], thing=Structure(
        "tube", geometry=Tube(axis=+X, inner_radius=mm(3.175), wall=mm(1.6), length=mm(100)), dynamics=Swaying()))
    return experiment


def test_projection_against_a_moving_structures_surface_is_refused_as_reserved():
    with pytest.raises(BuildRefused) as refused:
        swaying_tube(script_a()).plan()
    readers = [r.message for r in refused.value.refusals if r.code == "geometry.moving-read"]
    assert len(readers) == 3 and all("it has the dynamics bodyside.swaying 0.0.1" in m for m in readers)
    for reader in ("field 'magnetic' of apparatus 'field'", "interaction contact:umr", "observation swim_speed:umr"):
        assert any(m.startswith(reader) for m in readers), reader
    experiment = swaying_tube(without_contact(script_a()))
    experiment.attach(MovingContact(rung="projection", margin=mm(0.1)), body="umr", surface="tube.inner_surface")
    assert "projection between two moving things is reserved" in message(experiment, "confinement.moving-surface")


@register_interaction
@dataclass(frozen=True, kw_only=True)
class SecondConfinement(Contact):
    name = "bodyside.second_confinement"


def test_a_second_confinement_of_one_body_is_refused():
    experiment = script_a()
    experiment.attach(SecondConfinement(rung="projection", margin=mm(0.2)), body="umr", surface="tube.inner_surface")
    text = message(experiment, "confinement.more-than-one")
    assert "contact:umr" in text and "bodyside.second_confinement:umr" in text


def test_a_velocity_rule_that_is_not_built_is_refused():
    experiment = script_a()
    experiment._declarations = [d for d in experiment._declarations if not isinstance(getattr(d, "law", None),
                                                                                         Contact)]
    experiment.attach(Contact.Projection(margin=mm(0.1), rule="remove_outwards"), body="umr",
                      surface="tube.inner_surface")
    assert "Did you mean 'remove_outward'" in message(experiment, "confinement.rule")


def test_a_negative_margin_is_refused():
    experiment = script_a()
    experiment._declarations = [d for d in experiment._declarations if not isinstance(getattr(d, "law", None),
                                                                                         Contact)]
    experiment.attach(Contact.Projection(margin=mm(-0.1)), body="umr", surface="tube.inner_surface")
    assert "margin is -0.0001 m" in message(experiment, "setting.value")


# -- commands and drives ---------------------------------------------------------------------------------------------

def field_with(commands: dict) -> Experiment:
    experiment = script_a()
    (index,) = [i for i, d in enumerate(experiment._declarations) if getattr(getattr(d, "thing", None), "name",
                                                                               None) == "field"]
    held = experiment._declarations[index].thing
    experiment._declarations[index] = replace(experiment._declarations[index],
                                              thing=Apparatus("field", fields=held.fields, commands=commands))
    return experiment


def test_each_command_needs_a_drive_and_each_drive_a_command():
    assert "command.undriven" in codes(field_with({"frequency": Hz(10)}))
    assert "Did you mean 'frequency'" in message(field_with({"frequency": Hz(10), "amplitude": mT(1.2),
                                                             "frequencies": Hz(1)}), "command.unknown")


def test_a_drives_quantity_is_the_dimension_of_its_command():
    text = message(field_with({"frequency": Hz(10), "amplitude": Hz(1.2)}), "setting.units")
    assert "magnetic flux density (T)" in text and "frequency (Hz)" in text


def test_a_quantity_written_for_a_command_is_the_constant_drive():
    plans = script_a().plan()
    (drive,) = [p for p in plans.entries if p.owner == "drive:field.frequency"]
    assert isinstance(drive.entry, Constant) and drive.settings["value"] == 10.0
    assert drive.command is not None and (drive.command.node, drive.command.field, drive.command.units) == (
        "field.magnetic", "frequency", "Hz")


def test_a_field_along_its_rotation_axis_at_t_zero_is_refused():
    experiment = script_a()
    (index,) = [i for i, d in enumerate(experiment._declarations) if getattr(getattr(d, "thing", None), "name",
                                                                               None) == "field"]
    held = experiment._declarations[index].thing
    experiment._declarations[index] = replace(experiment._declarations[index], thing=Apparatus(
        "field", fields={"magnetic": UniformRotatingField(rotation_axis="tube.axis", initial_direction=(2, 0, 0))},
        commands=held.commands))
    assert "lies along the rotation axis" in message(experiment, "setting.value")


def test_the_torque_binds_a_uniform_field_only():
    experiment = script_a()
    experiment.add_apparatus(Apparatus("source", fields={"push": ToyField()}))
    experiment.attach(Magnetic.Torque(), body="umr", field="source.push")
    assert "flag uniform" in message(experiment, "support.unsupported")


# -- the body --------------------------------------------------------------------------------------------------------

def test_an_initial_orientation_that_is_not_a_unit_quaternion_is_refused():
    experiment = Experiment("body", scene=Scene.new(up=Z, units="m"), gravity=standard_gravity(), dt=s(1e-3))
    experiment.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=m((0, 0, 0)), orientation=(1.0, 0.1, 0.0, 0.0), velocity=m_per_s((0, 0, 0)),
        angular_velocity=rad_per_s((0, 0, 0)))))
    experiment.declare("umr.mass", kg(1e-4), reason="t")
    experiment.declare("umr.inertia", kg((1e-10,) * 3) * m(1.0) * m(1.0), reason="t")
    assert "is not a unit quaternion" in message(experiment, "state.value")


@register_interaction
@dataclass(frozen=True, kw_only=True)
class ReadsAMediumsMass(Gravity):
    name = "bodyside.reads_a_mediums_mass"

    def plan(self, ctx: PlanContext) -> Plan:
        return Plan(values={"mass": Value("medium", "mass")})


def test_a_value_a_thing_does_not_have_is_an_entry_defect():
    experiment = script_a()
    experiment.attach(ReadsAMediumsMass(), body="umr", medium="water")
    assert "which a medium does not have" in message(experiment, "entry.defect")
