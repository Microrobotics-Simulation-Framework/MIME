"""The unit label grammar (``mime.units.parse_unit``): every label MIME writes, and the spellings of one dimension.

The grammar: names (the SI base symbols and ``N Pa J W T Hz rad``), the 1, products by spaces, one ``/`` and a single
factor after it (more than one term is written in parentheses), integer powers by ``^``, parentheses. ``unit_symbol``,
the writer, is unchanged; ``parse_unit`` reads what it writes and more.

Controls. Three planted grammar bugs, each a function of the reader replaced by a faulty one: a quotient that divides by
only the first term of its denominator (the bug the table of ``m/(s Pa)`` guards), a power that reaches only the first
term of a group, and a quotient that multiplies. Each is applied to the table of cases, which fails on every run
because every row is there, and to the two properties, whose rate is measured with ``.github/probe/detection_rate.py``:
``MIME_PLANT_UNIT_GRAMMAR`` names a plant, and a run of this file with it set applies that plant to the properties, which
then fail. ``tests/unit/predictions_bundle2.json`` holds the predicted rate."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import pytest
from hypothesis import given
from hypothesis import strategies as st

import mime.units as units
from mime.units import (
    ACCELERATION,
    ANGULAR_VELOCITY,
    CURRENT,
    DIMENSIONLESS,
    ENERGY,
    FORCE,
    FREQUENCY,
    HYDRAULIC_RESISTANCE,
    LENGTH,
    MAGNETIC_FLUX_DENSITY,
    MASS,
    POWER,
    PRESSURE,
    TIME,
    Dimension,
    UnitError,
    parse_label,
    parse_unit,
    rate_symbol,
    unit_symbol,
)

BASES = ("m", "kg", "s", "A", "K", "mol", "cd")

#: Labels that must read as the dimension beside them: named derived units, products, quotients, parentheses, powers.
EQUAL = [
    ("N", FORCE), ("kg m/s^2", FORCE), ("kg m s^-2", FORCE), ("m kg/s^2", FORCE), ("Pa", PRESSURE), ("N/m^2", PRESSURE),
    ("J", ENERGY), ("N m", ENERGY), ("W", POWER), ("J/s", POWER), ("N m/s", POWER), ("T", MAGNETIC_FLUX_DENSITY),
    ("kg/(A s^2)", MAGNETIC_FLUX_DENSITY), ("m/s^2", ACCELERATION), ("m s^-2", ACCELERATION),
    ("Pa s/m^3", HYDRAULIC_RESISTANCE), ("N s/m", MASS / TIME), ("m/(s Pa)", LENGTH / (TIME * PRESSURE)),
    ("m^2/s^2", LENGTH**2 / TIME**2), ("N m^2", FORCE * LENGTH**2), ("N A^-2", FORCE / CURRENT**2),
    ("T m^3/A", MAGNETIC_FLUX_DENSITY * LENGTH**3 / CURRENT), ("kg s^-3 A^-1", MASS / (TIME**3 * CURRENT)),
    ("m^-1 kg s^-2 A^-1", MASS / (LENGTH * TIME**2 * CURRENT)), ("1", DIMENSIONLESS), ("rad", DIMENSIONLESS),
    ("1/s", FREQUENCY), ("Hz", FREQUENCY), ("rad/s", FREQUENCY), ("(m s)^2", LENGTH**2 * TIME**2),
    ("(m/s)^2", LENGTH**2 / TIME**2), ("(m s^-1)^-1", TIME / LENGTH), ("m / s", LENGTH / TIME), ("m  s", LENGTH * TIME),
    ("1/(m s)", DIMENSIONLESS / (LENGTH * TIME)), ("kg m^2/s^2", ENERGY), ("(kg m^2)/(s^2)", ENERGY),
    ("m^2 kg s^-1", LENGTH**2 * MASS / TIME), ("mol/cd", Dimension.of((0, 0, 0, 0, 0, 1, -1))),
]

#: Labels that must be refused, each with the code, and a part of what is said.
REFUSED = [
    ("", "empty"), ("(", "missing"), (")", "missing"), ("m)", "closing"), ("(m", "not closed"), ("m^", "power"),
    ("m^0", "power"), ("m^01", "power"), ("m^+2", "power"), ("m ^2", "not a unit symbol"),
    ("m^1.5", "not a unit symbol"),
    ("mm", "no unit symbol"), ("kPa", "no unit symbol"), ("Nm", "no unit symbol"), ("10", "only number"),
    ("1m", "only number"), ("/s", "missing"), ("m/", "missing"), ("m/s/s", "more than one term"),
    ("J/mol K", "more than one term"), ("m/s s", "more than one term"), ("N,m", "not a unit symbol"),
    ("rad Hz", "turns (Hz) and radians"), ("Hz/rad", "turns (Hz) and radians"), ("m; s", "not a unit symbol"),
    ("deg/s", "no unit symbol"), ("the command's own", "no unit symbol"),
]


def raised(label: str) -> UnitError:
    with pytest.raises(UnitError) as caught:
        parse_unit(label)
    return caught.value


def test_every_label_of_the_table_reads_as_its_dimension():
    assert [(label, parse_unit(label)) for label, want in EQUAL if parse_unit(label) != want] == []


def test_each_refused_label_is_refused_with_the_labels_code_and_what_to_write():
    for label, said in REFUSED:
        error = raised(label)
        assert said in str(error), (label, str(error))
        assert error.codes() == ["units.label"] and error.refusals[0].action, label
        assert isinstance(error, ValueError)


def test_a_name_that_is_nearly_a_unit_symbol_is_offered_the_one_it_is_near():
    assert "did you mean 'kg'" in str(raised("kgs"))


def test_a_turn_and_a_radian_stay_apart_in_what_a_rate_counts():
    assert parse_unit("Hz") == FREQUENCY and parse_unit("Hz").angle == "turn"
    assert parse_unit("rad/s") == ANGULAR_VELOCITY and parse_unit("rad/s").angle == "radian"
    assert parse_unit("s^-1").angle is None  # a rate that says neither, as DIMENSIONLESS / TIME is
    assert parse_unit("rad").angle is None and parse_unit("rad") == DIMENSIONLESS  # as the unit rad holds it
    assert [d.angle for d in parse_label("m/s; rad/s")] == [None, "radian"]
    assert parse_unit("rad/s^2").angle is None and parse_unit("Hz s") == DIMENSIONLESS


def test_rate_symbol_and_the_parser_round_trip_the_two_rates():
    for rate in (FREQUENCY, ANGULAR_VELOCITY):
        assert parse_unit(rate_symbol(rate)).angle == rate.angle
    assert unit_symbol(ANGULAR_VELOCITY) == "Hz"  # the writer is unchanged: the saved form writes it as Hz


def test_a_composite_label_is_read_part_by_part():
    assert parse_label("N s/m; N s; N m s") == (MASS / TIME, FORCE * TIME, FORCE * LENGTH * TIME)
    assert parse_label("m; 1") == (LENGTH, DIMENSIONLESS)
    with pytest.raises(UnitError, match="no unit symbol 'foo'"):
        parse_label("m; foo")


def test_the_writer_writes_what_it_always_wrote():
    written = {FORCE: "N", ACCELERATION: "m/s^2", HYDRAULIC_RESISTANCE: "Pa s/m^3", DIMENSIONLESS: "1",
               FREQUENCY: "Hz", ENERGY: "N m", MASS**2 / LENGTH: "m^-1 kg^2"}
    assert {d: unit_symbol(d) for d in written} == written


# -- the properties --------------------------------------------------------------------------------------------------

#: Exponents beyond any MIME label's largest (3), drawn whole.
exponents = st.tuples(*[st.integers(-6, 6)] * 7)


def term(symbol: str, power: int) -> str:
    return symbol if power == 1 else f"{symbol}^{power}"


def product(pairs: list[tuple[str, int]], space: str) -> str:
    return space.join(term(symbol, power) for symbol, power in pairs) or "1"


@st.composite
def spelled(draw: st.DrawFn) -> tuple[Dimension, str]:
    """A dimension and one spelling of it: a flat product, a quotient with its negative exponents in a denominator (in
    parentheses where more than one), or a parenthesised group raised to a whole power, in a drawn order and spacing.
    A group is built, not found: its inside and its power are drawn and the dimension is their product, so a draw
    reaches a group raised to 2, 3 or a negative power (a random exponent vector is rarely a whole multiple) in every
    third draw."""
    style = draw(st.sampled_from(["flat", "quotient", "grouped"]))
    space = draw(st.sampled_from([" ", "  "]))
    if style == "grouped":
        inside = draw(st.tuples(*[st.integers(-2, 2)] * 7))
        k = draw(st.sampled_from([-3, -2, -1, 1, 2, 3, 2, 3, -2, -3]))
        held = tuple(e * k for e in inside)
        pairs = draw(st.permutations([(symbol, e) for symbol, e in zip(BASES, inside, strict=True) if e]))
        return Dimension.of(held), f"({product(pairs, space)})" + (f"^{k}" if k != 1 else "")
    held = draw(exponents)
    pairs = draw(st.permutations([(symbol, e) for symbol, e in zip(BASES, held, strict=True) if e]))
    if style == "flat":
        return Dimension.of(held), product(pairs, space)
    top = product([(s, e) for s, e in pairs if e > 0], space)
    under = [(s, -e) for s, e in pairs if e < 0]
    slash = draw(st.sampled_from(["/", " / "]))
    return Dimension.of(held), top + (slash + (term(*under[0]) if len(under) == 1 else f"({product(under, space)})")
                                       if under else "")


@given(held=exponents)
def test_the_parser_reads_what_the_writer_writes(held):
    written = Dimension.of(held)
    assert parse_unit(unit_symbol(written)) == written


@given(case=spelled())
def test_every_spelling_of_a_dimension_reads_as_it(case):
    held, text = case
    assert parse_unit(text) == held, text


# -- the controls ----------------------------------------------------------------------------------------------------


def _over_by_one_term(a, b, label):
    first = next((i for i, e in enumerate(b.exponents) if e), None)
    exponents_ = tuple(-e if i == first else 0 for i, e in enumerate(b.exponents))
    return units._times(a, units._Term(exponents_, b.angle), label)  # noqa: SLF001


def _over_adding(a, b, label):
    return units._times(a, b, label)  # noqa: SLF001


def _raised_first_term_only(term_, power):
    first = next((i for i, e in enumerate(term_.exponents) if e), None)
    return units._Term(tuple(e * power if i == first else e for i, e in enumerate(term_.exponents)),  # noqa: SLF001
                       term_.angle)


#: The planted bugs: the reader's function each replaces, and the faulty one.
PLANTS: dict[str, tuple[str, Callable[..., object]]] = {
    "quotient-one-term": ("_over", _over_by_one_term),
    "quotient-multiplies": ("_over", _over_adding),
    "power-first-term-only": ("_raised", _raised_first_term_only),
}


@contextmanager
def planted(name: str) -> Iterator[None]:
    function, faulty = PLANTS[name]
    held = getattr(units, function)
    setattr(units, function, faulty)
    try:
        yield
    finally:
        setattr(units, function, held)


@pytest.fixture(autouse=True)
def the_plant_a_run_names() -> Iterator[None]:
    """``MIME_PLANT_UNIT_GRAMMAR=<plant>`` plants that bug for the run, which the tests of this file then fail: how
    the control's detection rate is measured (``detection_rate.py --plant-var MIME_PLANT_UNIT_GRAMMAR``)."""
    name = os.environ.get("MIME_PLANT_UNIT_GRAMMAR")
    if name is None:
        yield
        return
    with planted(name):
        yield


def table_fails() -> bool:
    try:
        return any(parse_unit(label) != want for label, want in EQUAL)
    except UnitError:
        return True


@pytest.mark.parametrize("name", sorted(PLANTS))
def test_the_table_of_cases_fails_on_every_run_under_each_planted_bug(name):
    if os.environ.get("MIME_PLANT_UNIT_GRAMMAR"):
        pytest.skip("a run with a plant set measures the properties' rate")
    assert not table_fails(), "the arm before the plant"
    with planted(name):
        assert table_fails()


@pytest.mark.parametrize("name", sorted(PLANTS))
def test_the_spellings_property_fails_under_each_planted_bug(name):
    if os.environ.get("MIME_PLANT_UNIT_GRAMMAR"):
        pytest.skip("a run with a plant set measures the properties' rate")
    with planted(name), pytest.raises(AssertionError):
        test_every_spelling_of_a_dimension_reads_as_it()  # type: ignore[call-arg]  # a @given test: it draws its own
