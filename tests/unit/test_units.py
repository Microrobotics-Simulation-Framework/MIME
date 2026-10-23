import math

import pytest

from mime.units import (
    ACCELERATION,
    DIMENSIONLESS,
    FREQUENCY,
    LENGTH,
    TIME,
    Hz,
    Quantity,
    UnitError,
    deg,
    m,
    m_per_s2,
    mm,
    ms,
    mT,
    s,
    si,
    standard_gravity,
    um,
)


def test_constructors_hold_si_values():
    assert mm(3.175) == Quantity(3.175e-3, LENGTH)
    assert ms(20).value == pytest.approx(0.02) and ms(20).dimension == TIME
    assert Hz(10) == Quantity(10.0, FREQUENCY)
    assert mT(1.2).value == pytest.approx(1.2e-3)
    assert deg(180).value == pytest.approx(math.pi) and deg(180).dimension == DIMENSIONLESS
    assert um((1, 2, 3)).value == pytest.approx((1e-6, 2e-6, 3e-6))


def test_arithmetic_combines_dimensions():
    assert (m(2) / s(4)).dimension == m(1).dimension / TIME
    assert (1 / s(0.5)) == Hz(2)
    assert (m(1) / s(1) / s(1)).dimension == ACCELERATION
    assert m(1) + mm(500) == m(1.5)
    assert 2 * ms(10) == ms(20)
    assert ms(10) < s(1)


def test_mixing_dimensions_is_refused():
    with pytest.raises(UnitError, match="cannot add length"):
        m(1) + s(1)
    with pytest.raises(UnitError, match="bare number"):
        m(1) + 1.0


def test_si_checks_the_dimension_and_refuses_bare_numbers():
    assert si(ms(20), TIME, what="dt") == pytest.approx(0.02)
    assert si(standard_gravity(), ACCELERATION, what="gravity") == 9.80665
    with pytest.raises(UnitError, match=r"dt needs units of time \(s\); it was given length"):
        si(m(1), TIME, what="dt")
    with pytest.raises(UnitError, match="the bare value 0.02"):
        si(0.02, TIME, what="dt")
    assert si(0.5, DIMENSIONLESS, what="a ratio") == 0.5


def test_a_quantity_prints_in_si():
    assert str(ms(20)).endswith(" s")
    assert str(m_per_s2(9.81)) == "9.81 m/s^2"


def test_the_markers_are_the_si_dimensions_and_combine():
    from mime.units import FORCE, LENGTH, MASS, VELOCITY, Force, Length, Mass, Time, Velocity

    assert (Length, Mass, Velocity, Force) == (LENGTH, MASS, VELOCITY, FORCE)
    assert Force / Length == Mass / Time**2 and str(Mass / Time) == "kg s^-1"


def test_an_entry_reads_its_settings_in_si_by_shape():
    from mime.registry import BuildContext, PlanContext

    context = BuildContext("e", 0.01, 0.01, "e", settings={"k": 1.7, "x0": (0.05, -0.02, 0.01)})
    assert context.scalar("k") == 1.7 and context.vector("x0") == (0.05, -0.02, 0.01)
    with pytest.raises(TypeError, match="'x0' is a vector of 3 components, not a number"):
        context.scalar("x0")
    with pytest.raises(TypeError, match="'k' is a number, not a vector"):
        context.vector("k")
    with pytest.raises(KeyError, match="no dimensioned setting 'c' is set"):
        context.scalar("c")
    planning = PlanContext(0.01, (0.0, 0.0, 1.0), (0.0, 0.0, -9.8), {}, {}, {"k": 1.7})
    assert planning.scalar("k") == 1.7
