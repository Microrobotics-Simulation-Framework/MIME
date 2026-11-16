"""``mime.scene.rotation(axis, angle)``: the unit quaternion, in MIME's order ``(w, x, y, z)``, that turns by an angle
about an axis (F16). The order is the point: ``(x, y, z, w)`` from ROS or SciPy is accepted as a different rotation, so
the constructor is the way to write one without remembering which.

Each value is checked against ``mime.scene._kinds.rotate``, which MIME turns vectors with, so the tests hold the
constructor to what a body's orientation does, not to a formula written twice. The controls are the pitfall itself (the
same four numbers read in the other order turn the other way) and the refusals."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, strategies as st

from mime.build import BuildRefused
from mime.build._refusals import ArgumentRefused
from mime.scene import X, Y, Z, rotation
from mime.scene._kinds import rotate
from mime.units import deg, mm, rad

CLOSE = {"abs": 1e-12}


def turned(q, v):
    return rotate(q, v)


def test_a_quarter_turn_about_z_takes_x_to_y_and_y_to_minus_x_right_handed():
    q = rotation(Z, deg(90.0))
    assert q == pytest.approx((math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)), abs=1e-15)
    assert turned(q, (1.0, 0.0, 0.0)) == pytest.approx((0.0, 1.0, 0.0), **CLOSE)
    assert turned(q, (0.0, 1.0, 0.0)) == pytest.approx((-1.0, 0.0, 0.0), **CLOSE)


def test_no_turn_is_the_identity_with_the_real_part_first_and_no_negative_zero():
    q = rotation(-X, deg(0.0))
    assert q == (1.0, 0.0, 0.0, 0.0) and all(math.copysign(1.0, c) > 0 for c in q), q


def test_the_pitfall_the_same_four_numbers_in_ros_order_are_a_half_turn_about_z():
    """``(0, 0, 0, 1)`` is the identity in ``(x, y, z, w)`` and a half turn about z in MIME's order: the order is stated
    on the function, and ``rotation`` is how to avoid writing it."""
    assert turned((0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0)) == pytest.approx((-1.0, 0.0, 0.0), **CLOSE)
    assert turned(rotation(Z, deg(0.0)), (1.0, 0.0, 0.0)) == pytest.approx((1.0, 0.0, 0.0), **CLOSE)
    doc = rotation.__doc__ or ""
    assert "(w, x, y, z)" in doc and "(x, y, z, w)" in doc


@given(axis=st.tuples(*[st.floats(-5.0, 5.0)] * 3).filter(lambda a: math.hypot(*a) > 1e-3),
       angle=st.floats(-6.0, 6.0), scale=st.floats(0.1, 10.0))
def test_it_is_a_unit_quaternion_for_any_axis_length_and_turns_the_axis_into_itself(axis, angle, scale):
    q = rotation(axis, rad(angle))
    assert math.sqrt(sum(c * c for c in q)) == pytest.approx(1.0, abs=1e-12)
    along = tuple(scale * c for c in axis)
    assert rotation(along, rad(angle)) == pytest.approx(q, abs=1e-12)
    unit = tuple(c / math.hypot(*axis) for c in axis)
    assert turned(q, unit) == pytest.approx(unit, abs=1e-9)
    assert turned(rotation(axis, rad(-angle)), turned(q, (0.3, -0.5, 0.8))) == pytest.approx((0.3, -0.5, 0.8), abs=1e-9)


@pytest.mark.parametrize(("axis", "angle", "said"), [
    ((0.0, 0.0, 0.0), deg(10.0), "not a direction"),
    ((1.0, 0.0), deg(10.0), "not a direction"),
    ((math.nan, 0.0, 1.0), deg(10.0), "not a direction"),
    (Z, 0.5, "a bare number"),
    (Z, deg(math.nan), "not one finite angle"),
    (Z, mm(1.0), "needs units of"),
], ids=["zero axis", "two components", "nan axis", "bare angle", "nan angle", "a length"])
def test_an_axis_or_an_angle_it_cannot_take_is_refused_with_its_code_and_the_call_to_write(axis, angle, said):
    with pytest.raises(ArgumentRefused) as refused:
        rotation(axis, angle)
    (held,) = refused.value.refusals
    assert held.code == "call.rotation" and said in held.message and "rotation(" in held.action, held


def codes_of(orientation) -> list[str]:
    """The refusals of a body's initial state with ``orientation`` (the mass and inertia it leaves undeclared aside)."""
    from mime import Body, Experiment, State
    from mime.dynamics import RigidInertial
    from mime.scene import Scene
    from mime.units import m, m_per_s, ms, rad_per_s

    exp = Experiment("turned", scene=Scene.new(up=Z, units="m"), dt=ms(1.0))
    exp.add_body(Body("b", dynamics=RigidInertial(), initial=State(
        position=m((0.0, 0.0, 0.0)), orientation=orientation, velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    with pytest.raises(BuildRefused) as refused:
        exp.plan()
    return [r.code for r in refused.value.refusals if r.code == "state.value"]


def test_the_quaternion_it_gives_is_one_the_build_takes_as_an_orientation():
    assert codes_of(rotation(Y, deg(30.0))) == []


def test_control_a_quaternion_that_is_not_a_unit_one_is_what_the_build_refuses_in_that_slot():
    assert codes_of((0.0, 0.0, 0.0, 2.0)) == ["state.value"]
