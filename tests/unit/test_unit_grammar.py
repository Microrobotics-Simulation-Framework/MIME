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
from hypothesis import example, given, settings
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
    ("1/s", DIMENSIONLESS / TIME), ("Hz", FREQUENCY), ("rad/s", ANGULAR_VELOCITY), ("(m s)^2", LENGTH**2 * TIME**2),
    ("(m/s)^2", LENGTH**2 / TIME**2), ("(m s^-1)^-1", TIME / LENGTH), ("m / s", LENGTH / TIME), ("m  s", LENGTH * TIME),
    ("1/(m s)", DIMENSIONLESS / (LENGTH * TIME)), ("kg m^2/s^2", ENERGY), ("(kg m^2)/(s^2)", ENERGY),
    ("m^2 kg s^-1", LENGTH**2 * MASS / TIME), ("mol/cd", Dimension.of((0, 0, 0, 0, 0, 1, -1))),
    ("m\ts", LENGTH * TIME), ("N\tm", ENERGY), ("m\t/\ts", LENGTH / TIME),
]

#: The rates, each with what it counts: the full dimension, which equality of exponents alone does not compare.
RATES = [
    ("Hz", FREQUENCY), ("(Hz)", FREQUENCY), ("rad/s", ANGULAR_VELOCITY), ("rad s^-1", ANGULAR_VELOCITY),
    ("s^-1 rad", ANGULAR_VELOCITY), ("rad / s", ANGULAR_VELOCITY), ("rad/(s)", ANGULAR_VELOCITY),
    ("rad m/(m s)", ANGULAR_VELOCITY), ("Hz m s/(s m)", FREQUENCY), ("(rad/s)^1", ANGULAR_VELOCITY),
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
    # a factor never runs into the next one: 'm1' is not 'm'
    ("m1", "separated by spaces"), ("kg1", "separated by spaces"), ("m(s)", "separated by spaces"),
    ("m^2s", "separated by spaces"),
    # a rate per second counts its angle once: these were read as rad/s or Hz
    ("1/(rad s)", "counts -1 of radians"), ("rad^2/s", "counts 2 of radians"), ("s^-1/rad", "counts -1 of radians"),
    ("1/(Hz s^2)", "counts -1 of turns"), ("Hz^2 s", "counts 2 of turns"), ("rad^-1 s^-1", "counts -1 of radians"),
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


def same(a: Dimension, b: Dimension) -> bool:
    """Whether two dimensions are one in full: their exponents and what a rate per second counts (turns, radians, or
    neither). ``Dimension`` equality compares the exponents alone, so ``FREQUENCY == ANGULAR_VELOCITY``; a property that
    used it would not see a Hz and a rad/s swapped."""
    return a == b and a.angle == b.angle and type(a) is type(b)


#: Exponents beyond any MIME label's largest (3), drawn whole.
exponents = st.tuples(*[st.integers(-6, 6)] * 7)

#: The dimensions the writer has a symbol for (``m/s^2``, ``Pa s/m^3``, ``m^3/Pa``: its quotients), read from its own
#: table, never a list kept here. A drawn exponent vector is a writer's symbol about once in 10^7 draws, so these are
#: drawn from the table, with real weight.
WRITTEN = sorted(units._SYMBOLS, key=lambda d: d.exponents)  # noqa: SLF001
RATE_DIMENSIONS = [FREQUENCY, ANGULAR_VELOCITY]


def term(symbol: str, power: int) -> str:
    return symbol if power == 1 else f"{symbol}^{power}"


def product(pairs: list[tuple[str, int]], space: str) -> str:
    return space.join(term(symbol, power) for symbol, power in pairs) or "1"


@st.composite
def rate_spellings(draw: st.DrawFn, space: str, form: str | None = None) -> tuple[Dimension, str]:
    """A rate per second and one spelling of it, built from its parts: a turn counted once (``Hz``) or a radian counted
    once over a second (``rad`` and ``s^-1``), with up to two base units that cancel (``m m^-1``), as a flat product in a
    drawn order, as a quotient (the angle and the cancelled units over the seconds and their partners, in parentheses
    where more than one), or either in parentheses; in a drawn spacing, with or without spaces about the slash. ``form``
    fixes the choice: ``"product"`` or ``"quotient"`` (a turn then holds a cancelled unit, so that it has a slash)."""
    held = draw(st.sampled_from(RATE_DIMENSIONS))
    pairs = [("Hz" if held.angle == "turn" else "rad", 1)] + ([("s", -1)] if held.angle == "radian" else [])
    smallest = 1 if form == "quotient" and held.angle == "turn" else 0
    for base in draw(st.lists(st.sampled_from(BASES), unique=True, min_size=smallest, max_size=2)):
        pairs += [(base, 1), (base, -1)]
    pairs = draw(st.permutations(pairs))
    if form == "product" or (form is None and draw(st.booleans())):
        text = product(pairs, space)
    else:
        top = [(s, e) for s, e in pairs if e > 0]
        under = [(s, -e) for s, e in pairs if e < 0]
        slash = draw(st.sampled_from(["/", " / ", " /", "/ "]))
        text = product(top, space) + (slash + (term(*under[0]) if len(under) == 1 else f"({product(under, space)})")
                                      if under else "")
    return held, f"({text})" if draw(st.booleans()) else text


@st.composite
def spelled(draw: st.DrawFn) -> tuple[Dimension, str]:
    """A dimension and one spelling of it: a flat product, a quotient with its negative exponents in a denominator (in
    parentheses where more than one), or a parenthesised group raised to a whole power, in a drawn order and spacing; or
    a rate per second, which counts turns or radians (:func:`rate_spellings`), in one draw of three in expectation, which
    Hypothesis does not promise (a seed drew one rate in a hundred): the properties over rates alone below draw nothing else. A group is built, not
    found: its inside and its power are drawn and the dimension is their product, so a draw reaches a group raised to 2, 3
    or a negative power (a random exponent vector is rarely a whole multiple) in two draws of nine."""
    style = draw(st.sampled_from(["flat", "flat", "quotient", "quotient", "grouped", "grouped", "rate", "rate", "rate"]))
    space = draw(st.sampled_from([" ", "  ", "\t"]))
    if style == "rate":
        return draw(rate_spellings(space))
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


#: A dimension the writer writes: one of its own symbols (a quarter of the draws), a rate, a product of two symbols, or
#: any exponents.
written_dimensions = st.one_of(
    st.sampled_from(WRITTEN), st.sampled_from(RATE_DIMENSIONS),
    st.builds(lambda a, b: a * b, st.sampled_from(WRITTEN), st.sampled_from(WRITTEN)),
    exponents.map(Dimension.of))


def as_the_writer_has_it(written: Dimension) -> Dimension:
    """The dimension the writer's text stands for. A rate per second that says neither what it counts (a plain ``1/s``,
    which a product of two symbols reaches: ``Hz * 1``) is written ``Hz``, as the saved form always wrote it, and reads
    back as a frequency: a convention of the writer, held by its own test below, not a round trip that loses something."""
    return FREQUENCY if written.exponents == FREQUENCY.exponents and written.angle is None else written


@given(written=written_dimensions)
def test_the_parser_reads_what_the_writer_writes(written):
    assert same(parse_unit(rate_symbol(written)), as_the_writer_has_it(written)), rate_symbol(written)


def test_a_rate_that_says_neither_is_written_hz_and_read_as_a_frequency():
    plain = Dimension.of((0, 0, -1, 0, 0, 0, 0))
    assert plain.angle is None and rate_symbol(plain) == unit_symbol(plain) == "Hz"
    assert same(parse_unit("Hz"), FREQUENCY) and not same(parse_unit("Hz"), plain)
    assert parse_unit("s^-1").angle is None, "written another way, it is read as it says"


def test_every_spelling_of_a_dimension_reads_as_it(case):
    held, text = case
    assert same(parse_unit(text), held), text


#: The table's rows are the property's own examples, run on every call before any draw: the draws reach a rate with a slash
#: in about one example in eight and a group raised to a power in two in nine, which Hypothesis does not promise (a miss
#: of a plant of a quotient that forgets its angle, release run 43886742202, drew none). A planted bug the table
#: catches (``test_the_table_of_cases_fails_on_every_run_under_each_planted_bug``) is then caught by this property in
#: every run, not in nearly every run, and the draws still look beyond the table.
for _label, _held in [*EQUAL, *RATES]:
    test_every_spelling_of_a_dimension_reads_as_it = example(case=(_held, _label))(
        test_every_spelling_of_a_dimension_reads_as_it)
test_every_spelling_of_a_dimension_reads_as_it = given(case=spelled())(test_every_spelling_of_a_dimension_reads_as_it)


@st.composite
def rate_products(draw: st.DrawFn) -> tuple[Dimension, str]:
    return draw(rate_spellings(draw(st.sampled_from([" ", "  ", "\t"])), "product"))


@st.composite
def rate_quotients(draw: st.DrawFn) -> tuple[Dimension, str]:
    return draw(rate_spellings(draw(st.sampled_from([" ", "  ", "\t"])), "quotient"))


@given(case=rate_products())
def test_every_rate_spelled_as_a_product_reads_as_it(case):
    """Every draw is a rate, because a property over everything drew one in a hundred under a seed: a quotient that drops the
    angle was then caught 29 runs of 30."""
    held, text = case
    assert same(parse_unit(text), held), text


@given(case=rate_quotients())
def test_every_rate_spelled_as_a_quotient_reads_as_it(case):
    """Every draw is a rate with a slash in it, for the same reason: the plant of a quotient that forgets its angle."""
    held, text = case
    assert "/" in text and same(parse_unit(text), held), text


def test_the_property_compares_what_a_rate_counts():
    assert FREQUENCY == ANGULAR_VELOCITY and not same(FREQUENCY, ANGULAR_VELOCITY)
    assert same(parse_unit("Hz"), FREQUENCY) and same(parse_unit("rad/s"), ANGULAR_VELOCITY)
    assert not same(parse_unit("Hz"), ANGULAR_VELOCITY)


def test_the_writers_own_symbols_are_most_of_what_the_round_trip_draws():
    """The table the round trip draws from holds the writer's quotients, so that a dimension written ``m/s^2`` is drawn
    at the weight of its branch and not once in 10^7."""
    assert {"m/s^2", "Pa s/m^3", "m^3/Pa"} <= {units._SYMBOLS[d] for d in WRITTEN}  # noqa: SLF001


# -- the controls ----------------------------------------------------------------------------------------------------


def _over_by_one_term(a, b, label):
    first = next((i for i, e in enumerate(b.exponents) if e), None)
    exponents_ = tuple(-e if i == first else 0 for i, e in enumerate(b.exponents))
    return units._times(a, units._Term(exponents_, b.angle, -b.counted), label)  # noqa: SLF001


def _over_adding(a, b, label):
    return units._times(a, b, label)  # noqa: SLF001


def _raised_first_term_only(term_, power):
    first = next((i for i, e in enumerate(term_.exponents) if e), None)
    return units._Term(tuple(e * power if i == first else e for i, e in enumerate(term_.exponents)),  # noqa: SLF001
                       term_.angle, term_.counted * power)


def _over_forgets_angle(a, b, label):
    held = units._times(a, units._Term(tuple(-e for e in b.exponents), b.angle, -b.counted), label)  # noqa: SLF001
    return units._Term(held.exponents)  # noqa: SLF001 - the quotient counts no angle: rad/s is s^-1, Hz/1 is 1


def _rate_swapped(angle):
    return ANGULAR_VELOCITY if angle == "turn" else FREQUENCY


#: The planted bugs: the reader's function each replaces, and the faulty one.
PLANTS: dict[str, tuple[str, Callable[..., object]]] = {
    "quotient-one-term": ("_over", _over_by_one_term),
    "quotient-multiplies": ("_over", _over_adding),
    "power-first-term-only": ("_raised", _raised_first_term_only),
    "rate-swapped": ("_rate_counting", _rate_swapped),
    "quotient-forgets-angle": ("_over", _over_forgets_angle),
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
        return any(not same(parse_unit(label), want) for label, want in [*EQUAL, *RATES])
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


@pytest.mark.parametrize("name", ["rate-swapped", "quotient-forgets-angle"])
def test_the_rate_quotient_property_fails_under_each_plant_that_misreads_a_rate(name):
    if os.environ.get("MIME_PLANT_UNIT_GRAMMAR"):
        pytest.skip("a run with a plant set measures the properties' rate")
    with planted(name), pytest.raises(AssertionError):
        test_every_rate_spelled_as_a_quotient_reads_as_it()  # type: ignore[call-arg]


def test_the_round_trip_property_fails_under_the_swapped_rates():
    if os.environ.get("MIME_PLANT_UNIT_GRAMMAR"):
        pytest.skip("a run with a plant set measures the properties' rate")
    with planted("rate-swapped"), pytest.raises(AssertionError):
        test_the_parser_reads_what_the_writer_writes()  # type: ignore[call-arg]
