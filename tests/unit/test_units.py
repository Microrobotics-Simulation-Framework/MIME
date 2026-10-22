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
