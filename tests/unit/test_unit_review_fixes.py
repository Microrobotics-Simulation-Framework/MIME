"""The review of usability bundle 2: what each finding's fix holds.

- **A unit error's notes are the exception's own** (``add_note`` and Hypothesis's failing-example note survive).
- **A value's unit is read with its spaces kept**: ``3 m s`` is metre seconds, never 3 ms; every form the docs and tests write
  still reads as it did (the base's own reader is the oracle, copied below).
- **A label is checked where it is defined** in ``node_input``, ``param_spec`` and a symbol row too, and a value that is
  not text is a coded refusal.
- **``0 + quantity``** takes a Python ``int`` 0 and nothing else, and an array plus a quantity is no array of quantities.
- The ``mmHg`` comment's claim about the torr.

Where a unit error is raised decides its refusal: ``tests/saved/test_unit_error_where_raised.py``."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import mime
from mime.core import Symbol, SymbolKind, Units, node_input, param_spec
from mime.study._override import WRITTEN_FORMS, parse_value
from mime.units import (
    LENGTH,
    MAGNETIC_FLUX_DENSITY,
    TIME,
    Dimension,
    Quantity,
    UnitError,
    mm,
    mmHg,
    parse_unit,
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
        return Quantity(magnitude, parse_unit(symbol))
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
    return found


def read(text: str, reader) -> tuple | None:
    held = reader(text)
    return None if held is None else (held.value, held.dimension.exponents, held.angle)


def the_bug_class(text: str) -> bool:
    """A form the fix changes on purpose: a unit written with spaces that is no unit's symbol as printed, which the base
    read by removing the spaces (``m s`` as ``ms``)."""
    written = (re.fullmatch(r"\s*\(?[^A-Za-z]*([A-Za-z].*?)\s*", text) or [None, ""])[1].split()
    return len(written) > 1 and " ".join(written) not in {unit.symbol for unit in units.units_by_name().values()}


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


def test_every_value_a_spec_file_writes_reads_back_as_the_quantity_it_wrote():
    """A spec writes ``<number> <unit_symbol(dimension)>``: its unit is the SI symbol of the dimension, spaces and all."""
    for dimension, symbol in sorted(units._SYMBOLS.items(), key=lambda row: row[1]):  # noqa: SLF001
        held = parse_value(f"1.5 {symbol}")
        assert held is not None and held.value == 1.5 and held.dimension == dimension, symbol


@given(exponents=st.tuples(*[st.integers(-3, 3)] * 7), number=st.floats(0.001, 1000.0))
def test_a_spec_written_value_of_any_dimension_reads_back_as_written(exponents, number):
    dimension = Dimension.of(exponents)
    unit = unit_symbol(dimension)
    held = parse_value(f"{number!r}" if unit == "1" else f"{number!r} {unit}")  # a spec writes a pure number bare
    assert held is not None and held.dimension == dimension and held.value == number


# -- a label checked where it is defined ----------------------------------------------------------------------------


@pytest.mark.parametrize("make", [lambda: Units(3), lambda: Units("kPa"), lambda: Units("m/"), lambda: Units(None),
                                  lambda: node_input("kPa", (3,)), lambda: node_input(3, (3,)),  # type: ignore[arg-type]
                                  lambda: param_spec(units="kPa"), lambda: param_spec(units=3),  # type: ignore[arg-type]
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
    assert param_spec(units="Pa s").units == "Pa s"
    assert Symbol("x", SymbolKind.STATE, "n", "N m", "mime.units.si").units == "N m"
    assert Symbol("x", SymbolKind.STATE, "n", "the command's own", "mime.units.si", prose_units=True).prose_units


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
