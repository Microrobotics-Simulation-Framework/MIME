"""The fence on ``verify``'s tolerances (design #263, with #133's rule that nothing loosens unlogged): each tolerance a
save writes is pinned here against the measurement it was set from and the margin above it. Lowering one is free.
Raising one fails, unless the constant's comment in ``mime/saved/_scales.py`` names a design row and a measurement's
date, each other than the pin's: a new decision on a new measurement, after which the pin moves with it.

A tolerance governs only how another machine's rounding, or a GPU's with deterministic mode off, is judged: the same
environment stays bit for bit, CI's own bit tests are unchanged, and accuracy is the oracles'."""

import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from mime.saved import _scales


@dataclass(frozen=True)
class Pin:
    value: float
    measured: float
    margin: float
    row: str
    on: str
    #: The measurement as the comment writes it.
    said: str


#: float32: the T4's 1.07e-6 (the tube demo's least eigenvalue, against itself and its CPU); float64: the CPU's 5.5e-16
#: across 24, 4 and 2 usable CPUs. Both measured 2026-11-07, decided in design #263.
PINS = {"FLOAT32_RTOL": Pin(1e-5, measured=1.07e-6, margin=9.0, row="#263", on="2026-11-07", said="1.07e-6"),
        "FLOAT64_RTOL": Pin(2e-14, measured=5.5e-16, margin=36.0, row="#263", on="2026-11-07", said="5.5e-16")}
SOURCE = Path(_scales.__file__)


def comment_of(name: str) -> str:
    """The ``#:`` lines right above ``name = ...`` in the module."""
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    at = next(i for i, line in enumerate(lines) if line.startswith(f"{name} = "))
    above: list[str] = []
    for line in reversed(lines[:at]):
        if not line.startswith("#:"):
            break
        above.insert(0, line.removeprefix("#:").strip())
    return " ".join(above)


def ratchet_problem(name: str, value: float, comment: str) -> str | None:
    pin = PINS[name]
    if value <= pin.value:
        return None
    rows = set(re.findall(r"#\d+", comment)) - {pin.row}
    dates = set(re.findall(r"\d{4}-\d{2}-\d{2}", comment)) - {pin.on}
    if rows and dates:
        return None
    return (f"{name} is raised from {pin.value} to {value}: name the design row that decided it and the date of the new "
            f"measurement it rests on in its comment, then move this pin (design {pin.row}, measured {pin.on})")


@pytest.mark.parametrize("name", list(PINS))
def test_each_tolerance_holds_its_pin(name: str):
    assert ratchet_problem(name, getattr(_scales, name), comment_of(name)) is None


@pytest.mark.parametrize("name", list(PINS))
def test_each_pin_is_its_measurement_times_its_margin_and_the_comment_says_both(name: str):
    pin, comment = PINS[name], comment_of(name)
    assert pin.value >= pin.margin * pin.measured
    assert float(pin.said) == pin.measured
    assert pin.row in comment and pin.on in comment and pin.said in comment, comment


@pytest.mark.parametrize("name", list(PINS))
def test_control_a_raised_tolerance_whose_comment_is_unchanged_fails(name: str):
    assert ratchet_problem(name, PINS[name].value * 10, comment_of(name)) is not None


@pytest.mark.parametrize("name", list(PINS))
def test_a_raised_tolerance_whose_comment_names_a_new_row_and_measurement_passes_and_a_lowered_one_always(name: str):
    raised = comment_of(name) + " Raised by design #999 on a measurement of 2027-01-04."
    assert ratchet_problem(name, PINS[name].value * 10, raised) is None
    assert ratchet_problem(name, PINS[name].value / 10, "") is None
