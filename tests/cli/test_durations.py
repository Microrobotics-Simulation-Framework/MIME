"""A duration on the command line (``--duration``, ``--every``, ``--stage-every``): ``10s``, ``500ms``, ``250us``,
``2min``, ``1h``, or a bare number, which is seconds as it was before units. The units are ``mime.units``' own, so
``500ms`` is what a script writes ``ms(500)``; a refusal names the forms."""

import math
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mime import units
from mime.build import BuildRefused
from mime.cli import option_seconds
from mime.cli._durations import FORMS, seconds

#: A magnitude a run is asked for: from a microsecond to a day, written the way a person writes it.
MAGNITUDES = st.floats(min_value=1e-3, max_value=1e4, allow_nan=False, allow_infinity=False)


@pytest.mark.parametrize(("text", "expected"), [
    ("10s", 10.0), ("10 s", 10.0), ("500ms", 0.5), ("250us", 250e-6), ("2min", 120.0), ("1h", 3600.0),
    ("1.5min", 90.0), ("1e-3s", 1e-3), (".5s", 0.5), ("+2s", 2.0),
])
def test_each_form_is_read_in_seconds(text, expected):
    assert seconds(text) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("text", ["0.0165", "10", "1e-3", "  2 ", "0.02", "inf", "-1"])
def test_a_bare_number_is_the_seconds_it_always_was(text):
    assert seconds(text) == float(text)


@given(MAGNITUDES, st.sampled_from(["s", "ms", "us"]))
def test_a_unit_of_mime_units_reads_as_the_quantity_a_script_writes_to_within_a_rounding(number, suffix):
    """``500ms`` is ``ms(500)``: the units' own factors. A script's ``ms(number)`` rounds twice (the number, then its
    product with 1e-3) and the command once, so they agree to a rounding, not always to the bit."""
    assert seconds(f"{number!r}{suffix}") == pytest.approx(getattr(units, suffix)(number).value, rel=4e-16)


@given(MAGNITUDES)
def test_a_minute_and_an_hour_are_sixty_and_thirty_six_hundred_seconds(number):
    assert seconds(f"{number!r}min") == pytest.approx(units.s(number * 60.0).value, rel=4e-16)
    assert seconds(f"{number!r}h") == pytest.approx(units.s(number * 3600.0).value, rel=4e-16)


#: A decimal as a person writes it: up to six significant digits, with the point anywhere from six places left to two right.
DECIMALS = st.builds(lambda digits, places: Decimal(digits).scaleb(-places), st.integers(min_value=1, max_value=999_999),
                     st.integers(min_value=-2, max_value=6))


@given(DECIMALS)
def test_every_way_of_writing_the_same_seconds_is_the_same_float(written):
    """``700ms``, ``0.7``, ``0.7s`` and ``700000us`` are one duration, so one float and one ``run.json``: the number is read
    as a decimal and converted once. (``700 * 1e-3`` is 0.7000000000000001.) The control is that multiplying as floats
    does disagree on this property's draws."""
    forms = [f"{written:f}", f"{written:f}s", f"{written * 1000:f}ms", f"{written * 1_000_000:f}us"]
    assert len({seconds(form) for form in forms}) == 1, forms
    assert seconds(forms[0]) == float(written)
    assert seconds(f"{written / 60:.40f}min") == pytest.approx(float(written), rel=1e-15)


def test_the_case_that_found_it_and_the_control_that_floats_would_get_it_wrong():
    assert seconds("700ms") == seconds("0.7") == 0.7
    assert 700 * 1e-3 != 0.7, "the control: the product of the floats is not the float of the decimal"
    assert units.ms(700.0).value != 0.7
    assert seconds("2300ms") == 2.3 and 2300 * 1e-3 != 2.3


@pytest.mark.parametrize("text", ["", "s", "10x", "10 seconds", "ten", "1m", "1ms2", "5 S", "10s10", "--1s"])
def test_what_is_not_a_duration_is_none(text):
    assert seconds(text) is None


@pytest.mark.parametrize(("option", "code"), [("--duration", "run.duration"), ("--every", "run.every"),
                                              ("--stage-every", "run.stage-every")])
def test_a_refusal_has_its_options_code_and_names_the_forms(option, code):
    with pytest.raises(BuildRefused) as refused:
        option_seconds(option, "10x")
    (refusal,) = refused.value.refusals
    assert refusal.code == code and option in refusal.message and "10x" in refusal.message
    assert FORMS in refusal.action and all(form in refusal.action for form in ("10s", "500ms", "2min"))


def test_an_option_not_given_is_none_and_a_given_one_is_its_seconds():
    assert option_seconds("--every", None) is None
    assert option_seconds("--every", "20ms") == pytest.approx(0.02)
    assert math.isinf(option_seconds("--duration", "inf") or 0.0)


@pytest.mark.parametrize("text", ["1e400ms", "1e999999999s", "1e400"])
def test_a_number_past_the_float_s_range_is_infinite_as_it_was_not_an_error_or_a_hang(text):
    assert seconds(text) == math.inf and seconds("-" + text) == -math.inf


def test_a_number_below_the_float_s_range_is_zero():
    assert seconds("1e-400us") == 0.0
