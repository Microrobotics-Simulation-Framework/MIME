"""The review of usability bundle 2: what each finding's fix holds.

- **A unit error's notes are the exception's own** (``add_note`` and Hypothesis's failing-example note survive).
- **A value's unit is read with its spaces kept**: ``3 m s`` is metre seconds, never 3 ms; every form the docs and tests write
  still reads as it did (the base's own reader is the oracle, copied below).
- **A label is checked where it is defined** in ``node_input``, ``ParamSpec`` and a symbol row too, and a value that is
  not text is a coded refusal.
- **``0 + quantity``** takes a Python ``int`` 0 and nothing else, and an array plus a quantity is no array of quantities.
- The ``mmHg`` comment's claim about the torr.

Where a unit error is raised decides its refusal: ``tests/saved/test_unit_error_where_raised.py``."""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import mime
from mime.core import ParamSpec, Symbol, SymbolKind, Units, node_input
from mime.study._override import WRITTEN_FORMS, parse_value
from mime.units import (
    ANGULAR_VELOCITY,
    DIMENSIONLESS,
    FREQUENCY,
    LENGTH,
    MAGNETIC_FLUX_DENSITY,
    TIME,
    Dimension,
    Hz,
    Quantity,
    UnitError,
    mm,
    mmHg,
    ms,
    parse_unit,
    rad_per_s,
    s,
    si,
    unit_symbol,
)
from mime import units

# -- the notes are the exception's own ---------------------------------------------------------------------------------


def test_add_note_survives_on_a_unit_error_beside_the_refusals_lines():
    error = UnitError("a message")
    error.add_note("a note of the caller's")
    assert "a note of the caller's" in error.__notes__
    assert any("call.units" in line for line in error.__notes__)
    assert "__notes__" not in vars(UnitError), "BaseException's own notes, not a property that builds a list a read"


def failing_notes(error_class: type[Exception]) -> list[str]:
    # No example database for these two meta-test functions only (never for a property under test): both are named
    # ``failing`` in this function, so they share a database key, and the second would replay the first's saved example
    # and report ``Failing test case`` against a ``Falsifying example`` in the first, which no note of an error class
    # explains. Each run of each is then its own search.
    @settings(database=None)
    @given(st.integers())
    def failing(n: int) -> None:
        raise error_class("boom")

    with pytest.raises(error_class) as caught:
        failing()
    return list(caught.value.__notes__)


def the_examples_note(notes: list[str]) -> list[str]:
    """Hypothesis's note of the failing example (``Falsifying example: ...`` where it shrank, ``Failing test case: ...``
    where every example fails, as here)."""
    return [note for note in notes if re.match(r"(Falsifying example|Failing test case): failing\(", note)]


def test_a_failing_hypothesis_test_keeps_its_failing_example_note_on_a_unit_error_as_on_a_value_error():
    control = failing_notes(ValueError)
    assert the_examples_note(control), "the control carries Hypothesis's note"
    assert the_examples_note(failing_notes(UnitError)) == the_examples_note(control)


# -- a value's unit with its spaces kept (the review's third major) -----------------------------------------------------


@pytest.mark.parametrize(("text", "dimension"), [("3 m s", LENGTH * TIME), ("3 m m", LENGTH**2),
                                                 ("2 m T", MAGNETIC_FLUX_DENSITY * LENGTH)])
def test_a_unit_written_with_spaces_is_its_grammars_product_never_the_prefixed_unit(text, dimension):
    held = parse_value(text)
    assert held is not None and held.dimension == dimension
    assert held.value == float(text.split()[0]), "no prefix scales it: 3 m s is not 3 ms"
    wanted = {"3 m s": TIME, "3 m m": LENGTH, "2 m T": MAGNETIC_FLUX_DENSITY}[text]
    with pytest.raises(UnitError) as caught:
        si(held, wanted, what="the setting")
    assert caught.value.codes() == ["call.units"], "a coded refusal against the setting's dimension"


def test_the_spaceless_spellings_read_as_the_prefixed_units_they_are():
    assert parse_value("3ms") == s(0.003) and parse_value("3mm") == mm(3.0)
    assert parse_value("2mT").dimension == MAGNETIC_FLUX_DENSITY and parse_value("2mT").value == 2e-3


def test_any_whitespace_between_units_is_read():
    assert parse_value("3 m\ts") == parse_value("3 m s") == parse_value("3  m   s")
    assert parse_unit("N\tm") == parse_unit("N m")


def base_parse_unit(symbol: str) -> Dimension:
    """The base's ``parse_unit`` (release/v0.3.0 before bundle 2), as it was: a symbol the writer wrote, or SI base units
    with integer powers, split at spaces: no named unit but those the writer has a symbol for, no ``/``, no parenthesis. The
    oracle's reader must be the base's own: a copy that called this PR's ``parse_unit`` would read ``rad s^-1`` as the
    grammar does and be wrong in the same way."""
    if symbol == "1":
        return DIMENSIONLESS
    for dimension, text in units._SYMBOLS.items():  # noqa: SLF001
        if text and text == symbol:
            return dimension
    exponents = [0] * 7
    terms = symbol.split()
    for term in terms:
        base, _, power = term.partition("^")
        if base not in units._BASE_SYMBOLS or (power and not re.fullmatch(r"-?[1-9][0-9]*", power)):  # noqa: SLF001
            raise ValueError(symbol)
        exponents[units._BASE_SYMBOLS.index(base)] += int(power) if power else 1  # noqa: SLF001
    if not terms:
        raise ValueError("empty")
    return Dimension.of(exponents)


def base_parse_value(text: str) -> Quantity | None:
    """The base's ``parse_value`` (release/v0.3.0 before bundle 2), as it was: the oracle for the forms the docs and tests
    wrote. Its unit lookup removed the spaces from the unit's text."""
    from mime.units import Unit

    found = re.fullmatch(r"\s*\(?\s*((?:[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)(?:\s*,\s*[+-]?(?:\d+\.?\d*|\.\d+)"
                         r"(?:[eE][+-]?\d+)?)*)\s*\)?\s*([A-Za-z][A-Za-z0-9/^ -]*)?\s*", text)
    if found is None:
        return None
    numbers = [float(n) for n in found[1].split(",")]
    magnitude = numbers[0] if len(numbers) == 1 else tuple(numbers)
    symbol = (found[2] or "1").strip()
    held = {unit.symbol.replace(" ", ""): unit for unit in vars(units).values() if isinstance(unit, Unit)}
    unit = held.get(symbol.replace(" ", ""))
    if unit is not None:
        return unit(magnitude)
    try:
        return Quantity(magnitude, base_parse_unit(symbol))
    except ValueError:
        return None


ROOT = Path(mime.__file__).resolve().parents[2]
_OVERRIDE = re.compile(r"--override[ =]+[\w.\[\]-]+=(\"[^\"]*\"|[^\s\"`)]+)")
_LIST = re.compile(r"values:\s*\[([^\]#]*)\]")
_LINE = re.compile(r"^\s*(?:duration|every|stage_every):\s*([^#\n]+?)\s*$", re.MULTILINE)


def the_forms_the_docs_and_tests_write() -> set[str]:
    """Each value text the docs, the READMEs and the tests write for ``--override`` or a spec file's values, collected
    from them where they are written."""
    found: set[str] = set(WRITTEN_FORMS)
    paths = [*ROOT.joinpath("docs").rglob("*.md"), *ROOT.joinpath("src/mime").rglob("*.md"),
             *ROOT.joinpath("tests").rglob("*.py")]
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        found.update(held.strip('"') for held in _OVERRIDE.findall(text))
        for listed in _LIST.findall(text):
            found.update(part.strip() for part in re.split(r",\s+(?=[-+\d.])", listed))
        found.update(held for held in _LINE.findall(text))
    return found | the_spaced_forms()


def the_spaced_forms() -> set[str]:
    """The forms a value can be written in with its slash spaced, built from the registry's own units: each unit's symbol
    (the rates among them, ``Hz``, ``rad/s``, ``rpm``, ``deg/s``) with spaces about its slash in every way, and the SI-base
    spellings of the two rates (``rad s^-1``, ``s^-1 rad``, ``rad/(s)``). The docs and the tests write few of them, so this
    is where a spaced rate is collected, which the base read as the unit's symbol and a reader that drops the angle reads
    as a rate that counts none."""
    found: set[str] = set()
    for unit in units.units_by_name().values():
        head, slash, tail = unit.symbol.partition("/")
        spellings = [unit.symbol] + ([f"{head} / {tail}", f"{head} /{tail}", f"{head}/ {tail}", f"{head}/({tail})"]
                                     if slash else [])
        found.update(f"{number} {spelling}" for number in ("2", "0.5", "1e3") for spelling in spellings)
    found.update(f"{number} {spelling}" for number in ("2", "10") for spelling in
                 ("rad s^-1", "s^-1 rad", "rad/(s)", "rad  /  s", "rad\t/\ts", "Hz s s^-1", "Hz m/m"))
    return found


def read(text: str, reader) -> tuple | None:
    held = reader(text)
    return None if held is None else (held.value, held.dimension.exponents, held.angle)


def the_bug_class(text: str) -> bool:
    """A form the fix changes on purpose, and only that class: letters separated by spaces that the base joined into one
    prefixed unit's symbol (``m s`` read as ``ms``, ``m T`` as ``mT``). Its unit is two or more words of letters alone, no
    slash, power or digit, whose spaceless join is the symbol of a unit of ``mime.units`` that the spaced words are not.
    A rate (``rad / s``), a quotient or a power is not of it: each is read as it was, or is a finding."""
    unit = (re.fullmatch(r"\s*\(?[^A-Za-z]*([A-Za-z].*?)\s*", text) or [None, ""])[1]
    words = unit.split()
    symbols = {each.symbol for each in units.units_by_name().values()}
    return (len(words) > 1 and all(word.isalpha() for word in words) and "".join(words) in symbols
            and " ".join(words) not in symbols)


def the_angle_the_text_names(text: str) -> str | None:
    """What a rate written as ``text`` counts, from its words alone (a model of the reading that is not the reader): a
    radian where the unit has the word ``rad``, a turn where it has ``Hz``."""
    unit = (re.fullmatch(r"\s*\(?[^A-Za-z]*([A-Za-z].*?)\s*", text) or [None, ""])[1]
    words = set(re.findall(r"[A-Za-z]+", unit))
    return "radian" if "rad" in words else "turn" if "Hz" in words else None


def test_every_value_form_the_docs_and_tests_write_reads_as_it_did_at_the_base():
    """The base's own reader is the oracle. The forms that read differently are all, and only, of the class the fix is for:
    a unit with spaces that spells another unit without them, which the docs themselves give (``3 m s``)."""
    forms = the_forms_the_docs_and_tests_write()
    read_before = sorted(text for text in forms if base_parse_value(text) is not None)
    changed = [text for text in read_before if read(text, parse_value) != read(text, base_parse_value)]
    print(f"\n{len(forms)} value forms collected, {len(read_before)} read at the base, {len(changed)} read otherwise now: "
          f"{changed}")
    assert len(read_before) >= 10, "a collector that stopped finding the forms would pass this test empty"
    assert [text for text in changed if not the_bug_class(text)] == []
    assert {"3 m s", "3 m m", "2 m T"} <= set(changed), "the docs' own three, which the fix reads as products"
    assert {"2 rad / s", "2 rad /s", "2 rad/ s", "2 deg / s", "2 mm / s"} <= set(read_before), \
        "the spaced rate forms are collected, and the base read each as its unit"
    # The base's reader cannot judge the forms it refused (``10 rad s^-1``): a model that reads the unit's words does. A form
    # with the word ``rad`` counts radians and one with ``Hz`` counts turns, in whichever spelling reaches the reader.
    named = [text for text in forms if the_angle_the_text_names(text) is not None and parse_value(text) is not None]
    assert len(named) >= 30, "the collection holds the rate forms: a collector that stopped finding them would pass empty"
    wrong = {text: (parse_value(text).angle, the_angle_the_text_names(text)) for text in named  # type: ignore[union-attr]
             if parse_value(text).angle != the_angle_the_text_names(text)}  # type: ignore[union-attr]
    assert wrong == {}, "a value counts the angle its unit names"


def test_every_value_a_spec_file_writes_reads_back_as_the_quantity_it_wrote():
    """A spec writes ``<number> <unit_symbol(dimension)>``: its unit is the SI symbol of the dimension, spaces and all."""
    for dimension, symbol in sorted(units._SYMBOLS.items(), key=lambda row: row[1]):  # noqa: SLF001
        held = parse_value(f"1.5 {symbol}")
        assert held is not None and held.value == 1.5 and same_unit(held, dimension), symbol


def same_unit(held: Quantity, dimension: Dimension) -> bool:
    """Whether a value read back is the unit that was written, in full: its exponents, what its dimension counts (turns,
    radians or neither) and what the quantity counts. ``Dimension`` equality is the exponents alone, so ``FREQUENCY ==
    ANGULAR_VELOCITY``, and a comparison by it could not see a rate read as the other."""
    return (held.dimension == dimension and held.dimension.angle == dimension.angle
            and type(held.dimension) is type(dimension) and held.angle == dimension.angle)


@pytest.mark.parametrize("rate", [Hz(1.5), rad_per_s(1.5), rad_per_s((0.0, 0.0, 1.5))], ids=["Hz", "rad/s", "rad/s vector"])
def test_a_rate_a_spec_writes_reads_back_as_the_rate_it_counts(rate):
    """Through the spec's own writer and reader (``value_text``, ``value_of``), not the unit's symbol function: a spec
    that wrote a rate of radians as ``Hz`` read it back as turns, a 2 pi misread."""
    from mime.study._spec import value_of, value_text

    back = value_of(value_text(rate, "axis"), "axis")
    assert isinstance(back, Quantity) and back.value == rate.value and back.angle == rate.angle
    written = ANGULAR_VELOCITY if rate.angle == "radian" else FREQUENCY
    assert same_unit(back, written), back
    other = ANGULAR_VELOCITY if rate.angle == "turn" else FREQUENCY
    assert not same_unit(back, other), "the comparison tells the two rates apart"


@given(exponents=st.tuples(*[st.integers(-3, 3)] * 7), number=st.floats(0.001, 1000.0))
def test_a_spec_written_value_of_any_dimension_reads_back_as_written(exponents, number):
    dimension = Dimension.of(exponents)
    unit = unit_symbol(dimension)
    held = parse_value(f"{number!r}" if unit == "1" else f"{number!r} {unit}")  # a spec writes a pure number bare
    assert held is not None and same_unit(held, dimension) and held.value == number


# -- a rate written in the other angle kind (the re-check's major) ----------------------------------------------------------

#: Each way the re-check wrote 10 radians per second: the unit's symbol, with the slash spaced as a person spaces it, and the
#: SI-base spellings the docs give. Each is a rate of radians, which a site that takes Hz refuses.
RADIANS_PER_SECOND = ["10 rad/s", "10 rad / s", "10 rad /s", "10 rad/ s", "10 rad s^-1", "10 s^-1 rad", "10 rad/(s)"]
#: The same for turns: Hz, spelled so that the grammar reads it too (a unit's name is the shortest way, and a product that
#: cancels is the way the grammar reaches it).
TURNS_PER_SECOND = ["10 Hz", "10 Hz m/m", "10 Hz s s^-1", "10 Hz^1"]


@pytest.fixture(autouse=True)
def the_plant_a_run_names(monkeypatch):
    """``MIME_PLANT_PARSE_VALUE=drops-the-angle`` plants the regression the re-check found for the run: the value reader's
    fallback builds ``Quantity(magnitude, parse_unit(label))``, which counts no angle. Its detection rate is measured with
    ``.github/probe/detection_rate.py --plant-var MIME_PLANT_PARSE_VALUE``."""
    if os.environ.get("MIME_PLANT_PARSE_VALUE") == "drops-the-angle":
        monkeypatch.setattr("mime.study._override.parse_value_unit", lambda label: (parse_unit(label), None))


@pytest.mark.parametrize("text", RADIANS_PER_SECOND)
def test_a_rate_in_radians_carries_its_angle_to_the_site_and_a_site_in_hz_refuses_it(text):
    held = parse_value(text)
    assert held is not None and held.angle == "radian" and held.dimension == FREQUENCY
    assert si(held, ANGULAR_VELOCITY, what="the site") == 10.0
    with pytest.raises(UnitError) as caught:
        si(held, FREQUENCY, what="the site")
    assert caught.value.codes() == ["call.units"]
    assert any("Hz(10.0 / (2 * pi))" in quantity for quantity, _ in caught.value.alternatives), "the conversion to write"


@pytest.mark.parametrize("text", TURNS_PER_SECOND)
def test_a_rate_in_turns_carries_its_angle_to_the_site_and_a_site_in_radians_refuses_it(text):
    held = parse_value(text)
    assert held is not None and held.angle == "turn" and held.dimension == FREQUENCY
    assert si(held, FREQUENCY, what="the site") == 10.0
    with pytest.raises(UnitError) as caught:
        si(held, ANGULAR_VELOCITY, what="the site")
    assert caught.value.codes() == ["call.units"]
    assert any("rad_per_s" in quantity for quantity, _ in caught.value.alternatives), "the conversion to write"


def test_a_number_of_turns_written_as_a_product_is_refused_where_a_number_goes():
    """``2 Hz s`` is two turns, which a site that takes radians (a bare number) refuses as ``Hz(2.0) * s(1.0)`` is."""
    held = parse_value("2 Hz s")
    assert held is not None and held.dimension == DIMENSIONLESS and held.angle == "turn"
    with pytest.raises(UnitError):
        si(held, DIMENSIONLESS, what="the site")
    assert si(parse_value("2 rad"), DIMENSIONLESS, what="the site") == 2.0


@pytest.mark.parametrize("text", ["2 rad^2", "2 Hz^2 s", "2 rad^-1", "2 rad/(rad rad)"])
def test_a_value_counts_an_angle_once_or_not_at_all(text):
    assert parse_value(text) is None


@pytest.fixture(scope="module")
def script_a_spec():
    pytest.importorskip("script_a")
    from script_a import reproduction_arm

    return reproduction_arm


def refused_codes_of(run) -> list[str]:
    from mime.build import BuildRefused

    with pytest.raises(BuildRefused) as caught:
        run()
    return caught.value.codes()


@pytest.mark.parametrize("text", RADIANS_PER_SECOND)
def test_a_rate_in_radians_is_refused_at_a_run_overridden_in_hz_and_is_never_run_as_the_number(script_a_spec, text):
    """The re-check's run: ``--override field.commands.frequency="10 rad / s"`` ran bit for bit as 10 Hz and said ``= 10.0
    Hz``. It is the base's refusal, ``setting.units``, with both conversions to write."""
    from mime.study._override import parse_override

    built = script_a_spec().build()
    path, value = parse_override(f"field.commands.frequency={text}")
    from mime.build import BuildRefused

    with pytest.raises(BuildRefused) as caught:
        built.run(ms(1.0), override={path: value})
    assert caught.value.codes() == ["setting.units"]
    action = caught.value.refusals[0].action
    assert "Hz(10.0)" in action and "Hz(10.0 / (2 * pi))" in action, action


def finals_equal(a, b) -> bool:
    return all(np.array_equal(a.final[node][name], b.final[node][name]) for node in b.final for name in b.final[node])


@pytest.fixture(scope="module")
def runs_with_the_rate_as_a_quantity(script_a_spec):
    """The runs the text forms must equal, made once: 12 Hz at the frequency, 1 rad/s about z as the initial spin."""
    from mime.units import Hz, rad_per_s

    return {"frequency": script_a_spec().build().run(ms(1.0), override={"field.commands.frequency": Hz(12.0)}),
            "spin": script_a_spec().build().run(ms(1.0), override={
                "umr.initial.angular_velocity": rad_per_s((0.0, 0.0, 1.0))})}


@pytest.mark.parametrize("text", TURNS_PER_SECOND)
def test_the_same_rate_in_turns_is_run_at_that_site_as_hz(script_a_spec, runs_with_the_rate_as_a_quantity, text):
    from mime.study._override import parse_override

    path, value = parse_override(f"field.commands.frequency={text.replace('10', '12')}")
    ran = script_a_spec().build().run(ms(1.0), override={path: value})
    assert finals_equal(ran, runs_with_the_rate_as_a_quantity["frequency"])


@pytest.mark.parametrize("text", TURNS_PER_SECOND)
def test_a_rate_in_turns_is_refused_at_an_angular_velocity_site(script_a_spec, text):
    from mime.study._override import parse_override

    path, value = parse_override(f"umr.initial.angular_velocity=0,0,{text}")
    assert refused_codes_of(lambda: script_a_spec().build().run(ms(1.0), override={path: value})) == ["state.units"]


@pytest.mark.parametrize("text", RADIANS_PER_SECOND)
def test_the_same_rate_in_radians_is_run_at_that_site_as_rad_per_s(script_a_spec, runs_with_the_rate_as_a_quantity, text):
    from mime.study._override import parse_override

    path, value = parse_override(f"umr.initial.angular_velocity=0,0,{text.replace('10', '1', 1)}")
    ran = script_a_spec().build().run(ms(1.0), override={path: value})
    assert finals_equal(ran, runs_with_the_rate_as_a_quantity["spin"])


# -- a value that is not text, a label checked where it is defined ----------------------------------------------------


# -- a label checked where it is defined ----------------------------------------------------------------------------


@pytest.mark.parametrize("make", [lambda: Units(3), lambda: Units("kPa"), lambda: Units("m/"), lambda: Units(None),
                                  lambda: node_input("kPa", (3,)), lambda: node_input(3, (3,)),  # type: ignore[arg-type]
                                  lambda: ParamSpec(units="kPa"), lambda: ParamSpec(units=3),  # type: ignore[arg-type]
                                  lambda: Symbol("x", SymbolKind.STATE, "n", "kPa", "mime.units.si"),
                                  lambda: Symbol("x", SymbolKind.STATE, "n", 3, "mime.units.si"),  # type: ignore[arg-type]
                                  lambda: parse_unit(3)])  # type: ignore[arg-type]
def test_a_label_that_is_wrong_or_is_not_text_is_the_labels_refusal_where_it_is_written(make):
    with pytest.raises(UnitError) as caught:
        make()
    assert caught.value.codes() == ["units.label"] and not isinstance(caught.value, AttributeError)


def test_a_right_label_is_made_in_each_place_and_a_prose_row_says_it_is_prose():
    assert Units("m; 1").text == "m; 1"
    assert node_input("m; 1", (7,)).expected_units == "m; 1"
    assert ParamSpec(units="Pa s").units == "Pa s" and ParamSpec().units == ""
    assert Symbol("x", SymbolKind.STATE, "n", "N m", "mime.units.si").units == "N m"
    assert Symbol("x", SymbolKind.STATE, "n", "the command's own", "mime.units.si", prose_units=True).prose_units


def test_a_param_spec_read_back_from_a_dict_is_checked_as_one_made_and_is_maddenings_own_class():
    from maddening.core.params import ParamSpec as MaddeningParamSpec

    held = ParamSpec(units="Pa s", bounds=(0.0, None))
    assert isinstance(held, MaddeningParamSpec) and ParamSpec.from_dict(held.to_dict()) == held
    with pytest.raises(UnitError) as caught:
        ParamSpec.from_dict({"units": "kPa"})
    assert caught.value.codes() == ["units.label"]


def test_every_node_of_mime_declares_its_parameters_with_the_checked_param_spec():
    """Every ``ParamSpec(units=...)`` label of ``src/mime`` is read where each is made, because no module under it
    builds a spec from MADDENING's class: a node file that imported that one would write its labels unread."""
    unchecked = [str(path.relative_to(ROOT)) for path in ROOT.joinpath("src/mime").rglob("*.py")
                 if "from maddening.core.params import ParamSpec" in path.read_text(encoding="utf-8")]
    assert sorted(unchecked) == ["src/mime/build/_maddening.py", "src/mime/core/_node.py", "src/mime/testing/_saved.py"], \
        "the one reading a graph's specs by type, the subclass's own base, and the one reading a default's trainable flag"


# -- a ufunc of NumPy on a quantity ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("call", [np.abs, np.sqrt, np.exp, np.negative, lambda q: np.add(q, q), lambda q: np.add(q, 1.0),
                                  lambda q: np.subtract(1.0, q), lambda q: np.multiply(q, 2.0)])
def test_a_ufunc_of_numpy_on_a_quantity_is_a_coded_refusal_that_says_what_to_write(call):
    """``__array_ufunc__ = None`` made these NumPy's own ``TypeError``, which has no code and says nothing to do."""
    with pytest.raises(UnitError) as caught:
        call(mm(2.0))
    error = caught.value
    assert error.codes() == ["call.units"] and not isinstance(error, TypeError)
    assert "abs(q)" in error.refusals[0].action and "q.to(unit)" in error.refusals[0].action


def test_every_operator_numpy_defers_to_a_quantity_is_still_the_quantitys_own():
    """The control: what ``None`` left to Python's operators is still left to them (a NumPy scalar on the left), with the
    same answers, and an array on the left is the bare-number refusal it was, never an array of quantities."""
    q = mm(2.0)
    assert np.float64(2.0) * q == mm(4.0) and np.int64(2) * q == mm(4.0) and np.float32(2.0) * q == mm(4.0)
    assert np.float64(2.0) / s(4.0) == units.Hz(0.5) and q + sum([mm(1.0)]) == mm(3.0)
    assert (np.float64(0.0) == q) is False and (np.float64(0.0) != q) is True
    for refused in (lambda: np.float64(3.0) < q, lambda: np.float64(3.0) >= q, lambda: np.ones(3) + q,
                    lambda: np.ones(3) * q, lambda: np.float64(3.0) + q):
        with pytest.raises(UnitError) as caught:
            refused()
        assert caught.value.codes() == ["call.units"]


# -- 0 + quantity ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("zero", [0.0, False, np.int64(0), np.float64(0.0), np.array(0), np.zeros(3), np.zeros(3, int),
                                  np.zeros(3, bool)])
def test_only_a_python_int_zero_is_added_to_a_quantity_and_no_array_makes_an_array_of_quantities(zero):
    with pytest.raises(UnitError) as caught:
        zero + mm(1.0)  # type: ignore[operator]
    assert caught.value.codes() == ["call.units"]


def test_the_python_int_zero_is_the_start_of_a_sum_and_nothing_else():
    assert 0 + mm(1.0) == mm(1.0) and sum([mm(1.0), mm(2.0)]) == mm(3.0)
    with pytest.raises(UnitError):
        mm(1.0) + 0  # type: ignore[operator]
    with pytest.raises(UnitError):
        sum([mm(1.0)], 0.0)  # type: ignore[call-overload]
    assert sum([mm(1.0)], mm(2.0)) == mm(3.0)


def test_a_number_times_a_quantity_still_scales_it_with_numpy_in_the_left_hand_scalar():
    assert np.float64(2.0) * mm(1.5) == mm(3.0) and mm(1.5) * np.float64(2.0) == mm(3.0)
    with pytest.raises(UnitError):
        np.ones(3) * mm(1.0)  # type: ignore[operator]


# -- the mmHg comment ------------------------------------------------------------------------------------------------


def test_the_torr_differs_from_the_conventional_mmhg_in_the_eighth_significant_figure():
    torr = 101325.0 / 760.0
    assert f"{mmHg.factor:.6e}" == f"{torr:.6e}", "seven significant figures agree"
    assert f"{mmHg.factor:.7e}" != f"{torr:.7e}", "the eighth does not"


# -- a spec's rate reads back as the rate it counts (the second re-check's minor) -------------------------------------------


def test_a_rate_a_spec_writes_is_read_back_by_a_study_re_read_from_its_spec_as_the_rate_it_counts(script_a_spec, tmp_path):
    """The spec writer wrote ``rad_per_s((0, 0, 1))`` as ``0.0, 0.0, 1.0 Hz``, which reads back as turns, and a study re-read
    from its spec refused every point (``state.units``) where the study it was written from ran. The real writer
    (``spec_text``) and the real reader (``Study.from_spec``), a rad/s value at a rad/s site and a Hz value at a Hz site."""
    from mime.study import Study, grid
    from mime.study._spec import spec_text
    experiment = script_a_spec()
    folder = tmp_path / "experiment"
    lock = experiment.save(folder)
    points = grid({"umr.initial.angular_velocity": [rad_per_s((0.0, 0.0, 1.0))],
                   "field.commands.frequency": [Hz(10.0)]})
    spec = tmp_path / "study.mime.yaml"
    spec.write_text(spec_text(points, duration=ms(1.0), record=[], every=None, batch=None, drop_refused=False, reasons={},
                              base="experiment", input_hash=lock["hashes"]["input"]))
    written = Study(experiment).sweep(points, duration=ms(1.0)).plan()
    read = Study.from_spec(spec).plan()
    refused = [r.code for p in read.points for r in (p.refusals or ())]
    assert refused == [] and [r.code for p in written.points for r in (p.refusals or ())] == []
    for axis, value in zip(read.axes, read.points[0].values.values(), strict=True):
        wanted = {"umr.initial.angular_velocity": "radian", "field.commands.frequency": "turn"}[axis.path]
        assert value.angle == wanted, (axis.path, value)


# -- a node author's label typo is the node's defect (the second re-check's minor) --------------------------------------------


@pytest.fixture
def a_torque_node_whose_param_specs_has_a_typo(monkeypatch):
    """The re-check's plant: ``ParamSpec(units="kPa")`` in ``MagneticTorqueNode.param_specs`` (kPa is no unit of MIME's
    vocabulary). Script A builds cleanly with it: a label is read when the specs are, not at import or at build."""
    import mime.nodes.magnetic._torque as torque

    cls = next(v for v in vars(torque).values() if isinstance(v, type) and "param_specs" in vars(v))
    real = cls.param_specs
    monkeypatch.setattr(cls, "param_specs", lambda self: {**real(self), "typo": ParamSpec(units="kPa")})
    return cls


def an_internal_defect_naming(error: BaseException, cls: type) -> None:
    from mime.errors._boundary import crossed

    refusals = crossed(error).refusals
    assert [r.code for r in refusals] == ["entry.defect"], [r.code for r in refusals]
    assert [r.category.name for r in refusals] == ["INTERNAL"]
    assert cls.__name__ in refusals[0].message and "kPa" in refusals[0].message, refusals[0].message


def test_a_label_typo_in_a_nodes_param_specs_is_the_nodes_defect_where_a_study_reads_the_specs(
        script_a_spec, a_torque_node_whose_param_specs_has_a_typo):
    from mime.build._maddening import parameter_specs

    built = script_a_spec().build()
    with pytest.raises(Exception) as caught:
        parameter_specs(built.graph)
    an_internal_defect_naming(caught.value, a_torque_node_whose_param_specs_has_a_typo)


def test_a_label_typo_in_a_nodes_param_specs_is_the_nodes_defect_where_the_docs_and_the_trainable_helper_read_a_nodes_specs(
        script_a_spec, a_torque_node_whose_param_specs_has_a_typo):
    from mime.core._node import node_param_specs

    held = script_a_spec().build().graph._nodes  # pyright: ignore[reportPrivateUsage]
    node = next(h.node for h in held.values() if type(h.node) is a_torque_node_whose_param_specs_has_a_typo)
    with pytest.raises(Exception) as caught:
        node_param_specs(node)
    an_internal_defect_naming(caught.value, a_torque_node_whose_param_specs_has_a_typo)


def test_a_param_specs_that_reads_clean_is_read_as_it_is(script_a_spec):
    from mime.build._maddening import parameter_specs

    assert parameter_specs(script_a_spec().build().graph), "script A's nodes declare parameters"
