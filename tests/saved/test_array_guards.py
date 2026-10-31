"""The saved form holds declarations, and an array quantity is a run's result: the three guards in ``saved/_document``
that refuse one (a quantity in a setting, a declared value, an operating condition), each beside its control. The build
refuses an array quantity first, so a guard is the document's own defence if one reaches it; each is reached directly.

The control of each: the same call with the guard blind to arrays (``numpy`` in the module made to hold no array
type, so nothing is one) writes the array into the document, so the refusal is the guard's and a test that passed
whatever the guard did would not be told from this one. And with a scalar and a tuple in the array's place, each writes
its value as it always did."""

import types
from typing import Any

import numpy as np
import pytest

import mime.saved._document as document
from mime.build import ValueDeclaration
from mime.saved import UnsavableError
from mime.units import LENGTH, VELOCITY, Condition, Quantity, m, mm_per_s

ARRAY = Quantity.of([1e-3, 2e-3], VELOCITY)
TUPLE = Quantity((1e-3, 2e-3), VELOCITY)
LENGTHS = Quantity.of([1.0, 2.0], LENGTH)


@pytest.fixture
def blind_to_arrays(monkeypatch: pytest.MonkeyPatch) -> None:
    """The module's ``np`` with no array type: ``isinstance(x, np.ndarray)`` is false of everything."""
    monkeypatch.setattr(document, "np", types.SimpleNamespace(ndarray=()))


def plain(value: Any) -> Any:
    return document._plain(value, "a setting")  # pyright: ignore[reportPrivateUsage]


def declared(value: Any) -> Any:
    return document._declared_doc(  # pyright: ignore[reportPrivateUsage]
        ValueDeclaration("probe.speed", value, "the test's"))


def conditions(value: Any, tolerance: Any) -> Any:
    return document.conditions_doc({"speed": Condition(value, tolerance)})


def test_a_quantity_in_a_setting_that_holds_an_array_is_refused_and_a_scalar_or_a_tuple_is_written():
    with pytest.raises(UnsavableError, match="an array quantity, a run's result"):
        plain(ARRAY)
    assert plain(mm_per_s(3.0)) == {"value": pytest.approx(3e-3), "unit": "m/s"}
    assert plain(TUPLE) == {"value": [1e-3, 2e-3], "unit": "m/s"}


def test_control_the_setting_guard_blind_to_arrays_writes_the_array(blind_to_arrays):
    assert isinstance(plain(ARRAY)["value"], np.ndarray)


def test_a_declared_value_that_is_an_array_quantity_is_refused_and_a_scalar_or_a_tuple_is_written():
    with pytest.raises(UnsavableError, match="array quantity, a run's result"):
        declared(ARRAY)
    assert declared(mm_per_s(3.0))[1]["unit"] == "m/s"
    assert declared(TUPLE)[1]["value"] == [1e-3, 2e-3]


def test_control_the_declared_value_guard_blind_to_arrays_writes_the_array(blind_to_arrays):
    assert isinstance(declared(ARRAY)[1]["value"], np.ndarray)


def test_a_condition_whose_value_or_tolerance_is_an_array_is_refused_and_a_scalar_one_is_written():
    for value, tolerance in ((LENGTHS, m(0.1)), (m(1.0), LENGTHS)):
        with pytest.raises(UnsavableError, match="not one value with its tolerance"):
            conditions(value, tolerance)
    assert conditions(m(1.0), m(0.1)) == {"speed": {"value": 1.0, "plus_minus": 0.1, "unit": "m"}}


def test_control_the_condition_guard_blind_to_arrays_writes_the_array(blind_to_arrays):
    assert isinstance(conditions(LENGTHS, m(0.1))["speed"]["value"], np.ndarray)
    assert isinstance(conditions(m(1.0), LENGTHS)["speed"]["plus_minus"], np.ndarray)
