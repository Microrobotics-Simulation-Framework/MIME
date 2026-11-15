"""Usability bundle 2 on the unit system: a unit error that is MIME's (F7a), a label checked where it is defined (G1) and
written from the dimension markers (G2), ``**``, ``abs`` and ``sum`` on quantities (F7c), seven more units (F17), and the
units ``--override`` reads (F13). The label grammar itself is ``test_unit_grammar.py``'s, and the collection of every label
MIME writes ``test_unit_labels.py``'s.

Equality with a bare number is documented, not changed (F7b is dropped): ``m(1) == 1.0`` stays ``False``, and the
last test here holds that to what the units page says."""

from __future__ import annotations

import copy
import math
import pickle
import sys
import traceback
from types import ModuleType

import numpy as np
import pytest

import mime.units as units
from mime.build import MimeError
from mime.core import Units
from mime.core._field_kinds import _typed  # noqa: PLC2701 - the typed state's own reader
from mime.study._override import WRITTEN_FORMS, parse_override, parse_value
from mime.units import (
    ANGULAR_VELOCITY,
    DIMENSIONLESS,
    FREQUENCY,
    LENGTH,
    PRESSURE,
    VOLUME,
    Dimension,
    Length,
    Quantity,
    UnitError,
    cP,
    deg_per_s,
    kPa,
    m,
    m2,
    m3,
    mL,
    mm,
    mmHg,
    mPa_s,
    MPa,
    Pa,
    rad_per_s,
    rpm,
    s,
    si,
    unit_label,
    unit_symbol,
    Hz,
)
from mime.vocabulary import Interface, TargetKind

# -- F7a: a UnitError is a MimeError and a ValueError ----------------------------------------------------------------


def wrong_unit() -> UnitError:
    with pytest.raises(UnitError) as caught:
        mm(3.0).to(s)
    return caught.value


def test_a_unit_error_is_a_mime_error_a_value_error_and_a_refusal_with_a_code_and_what_to_do():
    error = wrong_unit()
    assert isinstance(error, MimeError) and isinstance(error, ValueError)
    assert error.codes() == ["call.units"] and "to(m)" in error.refusals[0].action
    text = error.text()
    assert "[user] call.units:" in text and "do: read it in a unit of length" in text
    for kind in (ValueError, UnitError, MimeError):
        with pytest.raises(kind):
            mm(3.0).to(s)


def test_the_message_of_a_unit_error_is_its_own_so_the_sites_that_quote_it_read_what_they_always_did():
    error = wrong_unit()
    assert str(error) == "cannot read a quantity of length (m) (0.003 m) in s, a unit of time (s)"
    assert "call.units" not in str(error)


def test_a_traceback_prints_the_code_and_what_to_do_beneath_the_message():
    error = wrong_unit()
    shown = "".join(traceback.format_exception(error))
    assert "UnitError: cannot read a quantity of length" in shown
    assert "[user] call.units:" in shown and "    do: read it in a unit of length" in shown


def test_a_unit_error_pickles_and_copies_with_its_message_its_dimension_and_its_refusal():
    error = UnitError("cannot add", dimension=LENGTH, given=2.0, alternatives=(("Hz(1.0)", "if turns"),))
    for held in (pickle.loads(pickle.dumps(error)), copy.deepcopy(error)):  # noqa: S301 - the test's own object
        assert type(held) is UnitError and str(held) == "cannot add" and held.dimension == LENGTH and held.given == 2.0
        assert held.alternatives == (("Hz(1.0)", "if turns"),) and held.codes() == ["call.units"]
        assert held.instead(lambda q: q) == ": Hz(1.0) if turns"
    seen = wrong_unit()
    seen.refusals  # noqa: B018 - made, so that the pickle carries it
    assert pickle.loads(pickle.dumps(seen)).text() == seen.text()  # noqa: S301


def test_an_error_a_site_catches_as_a_unit_error_is_the_same_object_it_would_have_got():
    with pytest.raises(UnitError) as caught:
        si(3.0, LENGTH, what="the radius")
    assert caught.value.dimension == LENGTH and caught.value.given == 3.0
    assert caught.value.refusals[0].action.startswith("write it with its unit: m(3.0)")


# -- G1: a label is read where it is defined ------------------------------------------------------------------------


def test_an_interface_with_a_label_that_is_not_one_is_refused_where_it_is_defined_naming_it():
    with pytest.raises(UnitError) as caught:
        Interface("test.bad", TargetKind.LOAD, "m/s^", "world", (3,), magnitude=1.0)
    assert caught.value.codes() == ["units.label"] and "'test.bad'" in str(caught.value)
    good = Interface("test.good", TargetKind.LOAD, "N; N m", "world", (6,), magnitude=1.0)
    assert good.units == "N; N m"


def test_a_registration_carries_no_values_so_its_units_may_be_empty_and_no_other_kind_may():
    assert Interface("test.registered", TargetKind.REGISTRATION, "", None, ()).units == ""
    with pytest.raises(UnitError):
        Interface("test.unitless", TargetKind.LOAD, "", "world", (3,), magnitude=1.0)
    with pytest.raises(UnitError):
        Interface("test.registered", TargetKind.REGISTRATION, "mm", None, ())


def test_a_units_label_that_is_not_one_is_refused_where_it_is_written():
    for typo in ("m/s^", "kPa", "m; ", "N m/s s"):
        with pytest.raises(UnitError) as caught:
            Units(typo)
        assert caught.value.codes() == ["units.label"], typo
    assert Units("m; 1").text == "m; 1"


def test_a_typo_in_an_annotation_a_module_postponed_is_refused_when_the_state_is_read():
    """Every module that declares ``Units`` postpones its annotations, so the ``Units(...)`` is made when the hints are
    read, not at import: the typed state refuses, and does not read as a state with no types."""
    module = ModuleType("planted_units_module")
    source = (
        "from __future__ import annotations\n"
        "from typing_extensions import TypedDict\n"
        "from typing_extensions import Annotated, ReadOnly\n"
        "from mime.core import Units\n"
        "class Planted(TypedDict, closed=True):\n"
        "    speed: ReadOnly[Annotated[float, Units('m/s^')]]\n")
    sys.modules[module.__name__] = module
    try:
        exec(compile(source, "planted_units_module.py", "exec"), module.__dict__)  # noqa: S102
        with pytest.raises(UnitError) as caught:
            _typed(module.Planted)
        assert caught.value.codes() == ["units.label"]
    finally:
        del sys.modules[module.__name__]


# -- G2: a label from the dimension markers -------------------------------------------------------------------------


MARKERS = {name: held for name, held in vars(units).items() if isinstance(held, Dimension)
           and name[0].isupper() and not name.isupper() and name not in {"Dimension", "Marker"}}


def test_a_marker_writes_the_canonical_string_a_label_would_and_it_reads_back():
    assert Units(Length).text == "m" == Units("m").text
    assert Units(Length, DIMENSIONLESS).text == "m; 1" == Units("m; 1").text
    assert Units(ANGULAR_VELOCITY).text == "rad/s" and Units(FREQUENCY).text == "Hz"
    assert Units(Length, "1").text == "m; 1"  # a part may be a label, beside a marker
    assert Units(Length) == Units("m") and hash(Units(Length)) == hash(Units("m"))
    assert unit_label(Length, DIMENSIONLESS) == "m; 1"
    assert len(MARKERS) >= 20, sorted(MARKERS)
    for name, marker in MARKERS.items():
        written = unit_label(marker)
        assert written == (units.rate_symbol(marker)), name
        assert units.parse_unit(written) == marker and units.parse_unit(written).angle == marker.angle, name


def test_a_label_of_markers_goes_into_an_interface_as_the_string_it_is():
    held = Interface("test.pose", TargetKind.LOAD, unit_label(Length, DIMENSIONLESS), "world", (7,), magnitude=1.0)
    assert held.units == "m; 1"
    with pytest.raises(UnitError):
        unit_label()
    with pytest.raises(UnitError):
        unit_label(Length, "kPa")


# -- F7c: **, abs and sum --------------------------------------------------------------------------------------------


def test_an_integer_power_raises_the_dimension_with_the_value():
    assert m(2.0) ** 2 == m2(4.0) and (m(2.0) ** 3).dimension == VOLUME and m(2.0) ** 3 == m3(8.0)
    assert m(2.0) ** 1 == m(2.0) and (m(2.0) ** 0).dimension == DIMENSIONLESS and m(2.0) ** 0 == units.dimensionless(1.0)
    assert (m(2.0) ** -1).dimension == DIMENSIONLESS / LENGTH and (m(2.0) ** -1).value == 0.5
    assert m(2.0) ** 2.0 == m2(4.0)  # a whole number written as a float
    assert (mm((1.0, 2.0)) ** 2).value == (1e-6, 4e-6)
    assert (Quantity.of(np.array([1.0, 3.0]), LENGTH) ** 2).value.tolist() == [1.0, 9.0]
    assert (m(3.0) ** 2).reason is None


@pytest.mark.parametrize("power", [0.5, 1.5, True, "2", None, float("nan")])
def test_a_power_that_is_not_a_whole_number_is_refused_and_says_what_to_do_instead(power):
    with pytest.raises(UnitError) as caught:
        m2(4.0) ** power
    assert caught.value.codes() == ["call.units"] and ".to(unit)" in caught.value.refusals[0].action


def test_a_rate_that_counts_an_angle_takes_the_powers_it_can_count_and_refuses_the_rest():
    assert (Hz(2.0) ** 1) == Hz(2.0) and (Hz(2.0) ** 0).angle is None and (Hz(2.0) ** -1).angle == "turn"
    for power in (2, -2, 3):
        with pytest.raises(UnitError, match="counts an angle"):
            Hz(2.0) ** power


def test_abs_is_the_magnitudes_in_the_same_dimension_and_refuses_a_vector():
    assert abs(m(-2.0)) == m(2.0) and abs(m(2.0)) == m(2.0) and abs(Hz(-3.0)).angle == "turn"
    assert abs(Quantity.of(np.array([-1.0, 2.0]), LENGTH)).value.tolist() == [1.0, 2.0]
    with pytest.raises(UnitError) as caught:
        abs(mm((1.0, -2.0)))
    assert caught.value.codes() == ["call.units"] and "each component" in caught.value.refusals[0].action


def test_sum_starts_at_the_int_zero_and_nothing_else_is_added_to_a_quantity():
    assert sum([m(1.0), m(2.0), m(3.5)]) == m(6.5) and sum([m(2.0)]) == m(2.0)
    assert sum([mm((1.0, 2.0)), mm((3.0, 4.0))]).value == (4e-3, 6e-3)
    assert sum([Quantity.of(np.array([1.0, 2.0]), LENGTH)] * 2).value.tolist() == [2.0, 4.0]
    assert sum([m(1.0), mm(500.0)]) == m(1.5)
    for start in (0.0, False, True, 1, np.float64(0.0)):
        with pytest.raises(UnitError, match="bare number"):
            start + m(1.0)
    for other in (0, 0.0, 1.0):
        with pytest.raises(UnitError, match="bare number"):
            m(1.0) + other
    with pytest.raises(UnitError):
        sum([m(1.0), s(1.0)])
    assert sum([Hz(1.0), Hz(2.0)]) == Hz(3.0)
    with pytest.raises(UnitError, match="turns and the other"):
        sum([Hz(1.0), rad_per_s(2.0)])


def test_a_quantity_times_or_over_an_array_is_refused_by_what_an_array_quantity_is():
    for operation in (lambda: m(1.0) * np.array([1.0, 2.0]), lambda: m(1.0) / np.array([1.0, 2.0])):
        with pytest.raises(UnitError) as caught:
            operation()
        assert "run's result" in str(caught.value) and caught.value.codes() == ["call.units"]
    assert m(2.0) * np.float64(3.0) == m(6.0) and m(2.0) * np.array(3.0) == m(6.0)


def test_equality_with_a_bare_number_stays_false_and_is_documented_not_changed():
    """F7b is dropped: making ``==`` raise would break dataclass equality and containers. ``<`` refuses a bare number;
    ``==`` says no. The units page says so, and says what to write instead."""
    assert (m(1.0) == 1.0) is False and (m(0.0) == 0) is False and (m(1.0) != 1.0) is True
    with pytest.raises(UnitError):
        m(1.0) < 1.0  # noqa: B015
    from pathlib import Path

    page = (Path(units.__file__).parents[2] / "docs" / "units.md").read_text(encoding="utf-8")
    assert "`==` with a bare number" in page and ".to(" in page


# -- F17: the units --------------------------------------------------------------------------------------------------

#: Each new unit, its SI value (a cited exact value where it has one), and its dimension. The mercury column's value is
#: 13 595.1 kg/m^3 (conventional density) x 9.80665 m/s^2 (standard gravity) x 1 mm, which is 133.322387415 Pa.
NEW_UNITS = [
    (kPa, 1e3, PRESSURE), (MPa, 1e6, PRESSURE), (mmHg, 133.322387415, PRESSURE), (mL, 1e-6, VOLUME),
    (cP, 1e-3, units.DYNAMIC_VISCOSITY), (rpm, 1.0 / 60.0, FREQUENCY), (deg_per_s, math.pi / 180.0, ANGULAR_VELOCITY),
]


@pytest.mark.parametrize(("unit", "factor", "dimension"), NEW_UNITS, ids=[each[0].symbol for each in NEW_UNITS])
def test_each_new_unit_holds_its_value_in_si_and_its_dimension_reads_back_and_is_exported_once(unit, factor, dimension):
    quantity = unit(2.5)
    assert quantity.value == 2.5 * factor and quantity.dimension == dimension
    assert math.isclose(quantity.to(unit), 2.5, rel_tol=1e-15)
    names = [name for name, held in units.units_by_name().items() if held is unit]
    assert len(names) == 1 and units.__all__.count(names[0]) == 1


def test_the_conversions_of_the_new_units_against_each_other_and_their_definitions():
    assert kPa(1.0) == Pa(1000.0) and MPa(1.0) == kPa(1000.0) and MPa(1.0).to(Pa) == 1e6
    assert math.isclose(mmHg(1.0).value, 13595.1 * 9.80665 * 1e-3, rel_tol=1e-15) and mmHg(1.0).value == 133.322387415
    assert math.isclose(mmHg(120.0).to(kPa), 15.99868648, rel_tol=1e-9)  # 120 mmHg, a systolic pressure
    assert cP(1.0) == mPa_s(1.0) and cP(0.89).to(mPa_s) == 0.89
    assert mL(1.0) == m3(1e-6) and math.isclose(mL(5.0).to(units.mm3), 5000.0, rel_tol=1e-15)
    assert math.isclose(rpm(60.0).to(Hz), 1.0, rel_tol=1e-15) and math.isclose(rpm(600.0).to(Hz), 10.0, rel_tol=1e-15)
    assert math.isclose(deg_per_s(180.0).to(rad_per_s), math.pi, rel_tol=1e-15)
    assert math.isclose(deg_per_s(360.0).value, 2.0 * math.pi, rel_tol=1e-15)


def test_rpm_counts_turns_like_hz_and_deg_per_s_counts_radians_like_rad_per_s_and_the_two_stay_apart():
    assert rpm(60.0).angle == "turn" and deg_per_s(1.0).angle == "radian"
    with pytest.raises(UnitError, match="differ by 2 pi"):
        rpm(60.0).to(rad_per_s)
    with pytest.raises(UnitError, match="differ by 2 pi"):
        deg_per_s(57.0).to(Hz)
    with pytest.raises(UnitError, match="differ by 2 pi"):
        si(rpm(60.0), ANGULAR_VELOCITY, what="the spin")
    with pytest.raises(UnitError, match="differ by 2 pi"):
        si(deg_per_s(90.0), FREQUENCY, what="the field's frequency")
    assert si(rpm(60.0), FREQUENCY, what="the field's frequency") == 1.0
    assert rpm(60.0) == Hz(1.0) and rpm(60.0) != rad_per_s(1.0)
    with pytest.raises(UnitError, match="turns and the other"):
        rpm(60.0) + deg_per_s(10.0)


def test_the_units_saved_forms_write_are_the_dimensions_and_the_new_units_save_in_si():
    assert unit_symbol(kPa(1.0).dimension) == "Pa" and unit_symbol(rpm(1.0).dimension) == "Hz"
    assert unit_symbol(deg_per_s(1.0).dimension) == "Hz"  # the writer is unchanged: a rate in rad/s is written Hz
    assert unit_symbol(mL(1.0).dimension) == "m^3" and unit_symbol(cP(1.0).dimension) == "Pa s"


def test_no_name_or_symbol_a_value_may_be_written_with_names_two_units():
    by_key: dict[str, set[int]] = {}
    for name, held in units.units_by_name().items():
        for key in (name, held.symbol.replace(" ", "")):
            by_key.setdefault(key, set()).add(id(held))
    assert {key: held for key, held in by_key.items() if len(held) > 1} == {}


# -- F13: --override reads the unit names mime.units exports ---------------------------------------------------------


FORMS = [
    ("2mPa_s", mPa_s(2.0)), ("1.5mm_per_s", units.mm_per_s(1.5)), ("3N_m", units.N_m(3.0)), ("2m_per_s", units.m_per_s(2.0)),
    ("20Hz", Hz(20.0)), ("3rpm", rpm(3.0)), ("90deg_per_s", deg_per_s(90.0)), ("5kPa", kPa(5.0)), ("2cP", cP(2.0)),
    ("2 mPa s", mPa_s(2.0)), ("1.5mm/s", units.mm_per_s(1.5)), ("3 N m", units.N_m(3.0)), ("1.2 mT", units.mT(1.2)),
    ("0,0,-1 mm", mm((0.0, 0.0, -1.0))), ("0,0,-1mm", mm((0.0, 0.0, -1.0))), ("8.5e-5kg", units.kg(8.5e-5)),
    ("2 mL", mL(2.0)), ("120mmHg", mmHg(120.0)), ("1e-3 kg m^-1 s^-1", Quantity(1e-3, units.DYNAMIC_VISCOSITY)),
    ("4 m^2 kg s^-1", Quantity(4.0, LENGTH**2 * units.MASS / units.TIME)),
]


@pytest.mark.parametrize(("text", "want"), FORMS, ids=lambda value: str(value))
def test_a_value_is_read_with_the_units_name_its_symbol_or_si_base_units(text, want):
    held = parse_value(text)
    assert held == want and held is not None and held.angle == want.angle


def test_a_name_with_an_underscore_was_refused_before_and_is_read_now_by_the_same_parser_that_reads_a_spec_file():
    path, value = parse_override("water.viscosity=2mPa_s")
    assert path == "water.viscosity" and value == mPa_s(2.0)
    assert parse_value("2mPa_s") == parse_value("2 mPa s") == parse_value("2 mPa_s")
    for refused in ("2millis", "mm", "2 notaunit", "1,,2 mm", "2 mm_per"):
        assert parse_value(refused) is None, refused


def test_the_message_of_a_value_that_is_refused_shows_forms_and_each_of_them_is_read():
    from mime.build import BuildRefused

    with pytest.raises(BuildRefused) as caught:
        parse_override("water.viscosity=2millis")
    refusal = caught.value.refusals[0]
    assert refusal.code == "override.written"
    for form in WRITTEN_FORMS:
        assert form in refusal.message, form
        assert parse_value(form) is not None, form
    assert "water.viscosity=2mPa_s" in refusal.action and parse_override("water.viscosity=2mPa_s")


def test_a_bare_number_is_read_as_one_with_no_unit_and_a_dimensioned_setting_refuses_it():
    bare = parse_value("2")
    assert bare is not None and bare.dimension == DIMENSIONLESS
    from control_cases import spun_magnet

    built = spun_magnet("bare").build()
    with pytest.raises(MimeError) as caught:
        built.run(units.ms(2), override={"field.frequency": bare})
    assert caught.value.codes()[0] in {"setting.units", "call.units", "override.units"}, caught.value.codes()
