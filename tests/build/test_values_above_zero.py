"""The values of things that are above zero by what they are, refused where they are declared or read (UX-1's F1): a
body's mass where a dynamics divides by it (``rigid_inertial``'s step does), a part's density, an apparatus part's mass
and density, a medium's density; and a buoyant volume below zero. Each is a ``value.not-positive`` refusal naming the
value and the line to write, in every profile, where the base ran (a negative mass at 4.13 m/s, a medium of density
zero, a water that fell up) or failed with a ``ZeroDivisionError`` that had no code (a part of density zero).

Where each rule applies is read from the code: the mass's zero is refused only for a dynamics that reads the value
(``Dynamics.values``), so a body no dynamics divides by may be massless; a buoyant volume may be zero (nothing
displaced); a density never. The controls are the same experiments with the value in its range, which build, and a
planted check whose zero is wrongly refused for every reader, which the body-level control below catches."""

from __future__ import annotations

import pytest

from mime import Body, Experiment, Medium, State
from mime.build import BuildRefused
from mime.dynamics import RigidInertial
from mime.interactions import Gravity
from mime.scene import Scene, Sphere, Z
from mime.things import Part, ThingKind
from mime.units import kg, kg_per_m3, m, m3, m_per_s, mPa_s, ms, rad_per_s, um
from chain_cases import massive_rpm


def refused(build) -> list:
    try:
        build()
    except BuildRefused as error:
        return list(error.refusals)
    return []


def coded(refusals, code: str) -> list:
    return [refusal for refusal in refusals if refusal.code == code]


def ball(*, mass=None, volume=None, part_density=7850.0, medium_density=998.0, profile="exploratory") -> Experiment:
    """A steel ball in water under gravity, with the values given declared over what its parts and the medium give."""
    exp = Experiment("ball", scene=Scene.new(up=Z, units="m"), dt=ms(0.05), profile=profile)
    exp.add_medium(Medium("water"))
    exp.declare("water.density", kg_per_m3(medium_density), reason="water at 20 degC")
    exp.declare("water.viscosity", mPa_s(1.0), reason="water at 20 degC")
    exp.add_body(Body("ball", dynamics=RigidInertial(), initial=State(
        position=m((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"sphere": Part(Sphere(radius=um(50.0)))}))
    exp.declare("ball.parts.sphere.density", kg_per_m3(part_density), reason="steel")
    if mass is not None:
        exp.declare("ball.mass", kg(mass), reason="a declared mass")
    if volume is not None:
        exp.declare("ball.buoyant_volume", m3(volume), reason="a declared volume")
    exp.attach(Gravity(), body="ball", medium="water")
    return exp


@pytest.mark.parametrize("profile", ["exploratory", "research"])
def test_a_negative_mass_is_refused_with_its_magnitude_to_write(profile):
    (held,) = coded(refused(lambda: ball(mass=-1.3e-10, profile=profile).build()), "value.not-positive")
    assert "ball.mass is -1.3e-10 kg" in held.message and "below zero" in held.message
    assert "if the minus sign was the slip" in held.action and "kg(1.3e-10)" in held.action


def test_a_mass_of_zero_is_refused_where_the_dynamics_divides_by_it_and_names_it():
    (held,) = coded(refused(lambda: ball(mass=0.0).build()), "value.not-positive")
    assert "ball.mass is 0.0 kg" in held.message and "rigid_inertial divides by it" in held.message
    assert "declare it above zero" in held.action


def test_control_a_positive_mass_builds():
    assert refused(lambda: ball(mass=3.3e-11).build()) == []
    assert refused(lambda: ball().build()) == [], "the mass its parts give"


def test_a_mass_of_zero_is_a_value_where_no_dynamics_reads_it_and_below_zero_is_never():
    """The rule is read from the kind and from who reads the value, not listed: the same zero is refused for a reader
    that divides by it and taken for one that does not. The control for each half is the other."""
    from mime.materials import VALUE_KINDS, ZeroIs, not_above_zero

    kind = VALUE_KINDS[ThingKind.BODY]["mass"]
    assert kind.above_zero is not None and kind.above_zero.zero is ZeroIs.UNREAD
    assert not_above_zero("ball.mass", kind, 0.0, "declared") is None
    assert not_above_zero("ball.mass", kind, 0.0, "declared", divided_by="rigid_inertial") is not None
    assert not_above_zero("ball.mass", kind, -1e-10, "declared") is not None
    assert not_above_zero("ball.mass", kind, 1e-10, "declared", divided_by="rigid_inertial") is None


@pytest.mark.parametrize("profile", ["exploratory", "research"])
@pytest.mark.parametrize("density", [-7850.0, 0.0])
def test_a_part_density_of_zero_or_less_is_refused_naming_the_density(density, profile):
    """The base failed with a ``ZeroDivisionError`` for zero (``error.uncoded``) and with 'the inertia tensor is not
    positive definite' for a negative one, whose line sent the user to declare an inertia: the density was never named."""
    refusals = refused(lambda: ball(part_density=density, profile=profile).build())
    (held,) = coded(refusals, "value.not-positive")
    assert f"ball.parts.sphere.density is {density!r} kg/m^3" in held.message and "zero or less" in held.message
    assert "kg_per_m3(7850.0)" in held.action
    assert [refusal.code for refusal in refusals] == ["value.not-positive"], "and nothing else is blamed on it"


@pytest.mark.parametrize("density", [-998.0, 0.0])
def test_a_medium_density_of_zero_or_less_is_refused(density):
    (held,) = coded(refused(lambda: ball(medium_density=density).build()), "value.not-positive")
    assert f"water.density is {density!r} kg/m^3" in held.message and "kg_per_m3(998.0)" in held.action


def test_a_buoyant_volume_below_zero_is_refused_and_zero_is_a_volume():
    (held,) = coded(refused(lambda: ball(volume=-1e-15).build()), "value.not-positive")
    assert "ball.buoyant_volume is -1e-15 m^3" in held.message and "below zero" in held.message
    assert "declare it zero or more" in held.action
    assert refused(lambda: ball(volume=0.0).build()) == [], "nothing displaced is a volume"
    assert refused(lambda: ball(volume=4.2e-13).build()) == []


@pytest.mark.parametrize("path", ["rpm.parts.rotor_magnet.density", "rpm.parts.camera.mass", "rpm.parts.stator.mass"])
@pytest.mark.parametrize("value", [0.0, -1.0])
def test_an_apparatus_part_mass_or_density_of_zero_or_less_is_refused(path, value):
    def build():
        exp = massive_rpm(extra=True)
        # A second declaration of one path is refused as a duplicate: replace the one the rig made.
        exp._declarations = [d for d in exp._declarations if getattr(d, "path", None) != path]
        exp.declare(path, (kg if path.endswith("mass") else kg_per_m3)(value), reason="a probe")
        return exp.build()

    refusals = coded(refused(build), "value.not-positive")
    assert any(f"{path} is {value!r}" in held.message for held in refusals), [held.message for held in refusals]


def test_control_the_apparatus_with_its_masses_in_range_builds():
    assert coded(refused(lambda: massive_rpm(extra=True).build()), "value.not-positive") == []
