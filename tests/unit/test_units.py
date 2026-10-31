import math

import numpy as np
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


def test_an_array_quantity_holds_si_floats_read_only_and_reads_in_a_unit_of_its_dimension():
    from mime.units import LENGTH, VELOCITY, mm_per_s, um_per_s

    speeds = Quantity.of([1e-3, 2.5e-3], VELOCITY)
    assert speeds.value.dtype == np.float64 and not speeds.value.flags.writeable
    assert speeds.to(mm_per_s) == pytest.approx([1.0, 2.5]) and speeds.to(um_per_s) == pytest.approx([1e3, 2.5e3])
    assert (speeds * s(2)).dimension == LENGTH and (speeds * s(2)).to(mm) == pytest.approx([2.0, 5.0])
    assert (speeds + Quantity.of([1e-3, 0.0], VELOCITY)).to(mm_per_s) == pytest.approx([2.0, 2.5])
    assert speeds == Quantity.of([1e-3, 2.5e-3], VELOCITY) and speeds != Quantity.of([1e-3, 2.5e-3], LENGTH)
    assert mm(3.175).to(mm) == pytest.approx(3.175) and um((1, 2)).to(mm) == pytest.approx((1e-3, 2e-3))
    with pytest.raises(TypeError, match="not hashable"):
        hash(speeds)
    with pytest.raises(TypeError, match="not ordered"):
        _ = speeds < speeds


def test_to_refuses_a_unit_of_another_dimension_naming_both_and_reads_the_same_values_in_its_own():
    from mime.units import VELOCITY, mm_per_s

    speeds = Quantity.of([1e-3, 2.5e-3], VELOCITY)
    with pytest.raises(UnitError, match=r"cannot read a quantity of velocity \(m/s\) .* in mm, a unit of length"):
        speeds.to(mm)
    with pytest.raises(UnitError, match=r"of time \(s\) .* in mm/s, a unit of velocity"):
        ms(20).to(mm_per_s)
    with pytest.raises(TypeError, match="takes a unit of mime.units"):
        speeds.to("mm/s")  # type: ignore[arg-type]
    assert ms(20).to(ms) == pytest.approx(20.0) and ms(20).to(s) == pytest.approx(0.02)


def test_an_array_quantity_is_a_result_and_never_a_declaration():
    from mime.units import VELOCITY

    with pytest.raises(UnitError, match="an array is a run's result"):
        si(Quantity.of([1.0, 2.0], VELOCITY), VELOCITY, what="a speed")
    assert si(Quantity((1.0, 2.0), VELOCITY), VELOCITY, what="a speed") == (1.0, 2.0)
