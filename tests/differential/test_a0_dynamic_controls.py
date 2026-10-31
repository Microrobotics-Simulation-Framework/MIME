"""The controls of ``test_a0_dynamic.py``: a fault planted in each new part of A0 on its dynamic rungs (a gain of each
loop of its controller, and the joint loop's gravity term) at three times the comparison's resolution for the quantity
it moves (``a0_dynamic_plants.py``) fails the comparison, in both directions. A control that does not fail makes the
comparison's pass mean nothing; the rate at which a planted fault is caught is measured, over sizes from below the
resolution to ten times it, by ``measure_a0_dynamic_controls.py`` (``references/a0_dynamic_controls_measured.json``),
and the record is checked here against the constants the plants use. The swap test's controls (a step of lag removed
or added inside the compiled step) are in ``test_a0_swap.py``."""

import json
from pathlib import Path

import jax
import pytest
from a0_dynamic_compare import compare, failures, measure, table
from a0_dynamic_plants import KEY, PLANTS, RESPONSES

SIZE = 3.0
RECORD = json.loads((Path(__file__).parent / "references" / "a0_dynamic_controls_measured.json").read_text())


@pytest.fixture(autouse=True, scope="module")
def x64():
    """This module runs in float32, as the reference does."""
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", held)


CASES = [pytest.param(plant, sign, id=f"{plant.part.replace(' ', '-').replace(chr(39), '')}-{'plus' if sign > 0 else 'minus'}")
         for plant in PLANTS for sign in (1.0, -1.0)]


@pytest.mark.parametrize(("plant", "sign"), CASES)
def test_a_fault_planted_in_each_new_part_fails_the_comparison(plant, sign):
    """Held to the comparison's own rule (``failures``, the finding's allowance included), not the bare tolerances."""
    with plant.planted(SIZE, sign):
        found, run = measure(KEY, **plant.apply(SIZE, sign))
    rows = compare(found, KEY)
    print(f"\n{plant.part}: {plant.name}, {SIZE:g} x {plant.resolution:.3g}, sign {sign:+.0f}\n" + table(rows))
    assert plant.quantity in failures(rows, KEY, run)


def test_the_plants_resolutions_are_the_ones_the_record_measured():
    """The responses the plants' resolutions rest on are the record's (measured at a relative size of 1e-2, both
    signs), to 10 %; each plant gates the quantity the record found the most sensitive; and every fault above its
    resolution was caught by that quantity in every draw of the sweep."""
    sensitivities = RECORD["sensitivities"]
    for plant in PLANTS:
        measured = sensitivities[plant.part]
        assert plant.quantity == measured["sharpest_gate"], plant.part
        assert RESPONSES[plant.part] == pytest.approx(measured["response_per_unit"][plant.quantity]["0.01"], rel=0.1)
        found = RECORD["plants"][plant.part]
        above = [draw for draw in found["draws"] if draw["size_in_resolutions"] >= 1.5]
        assert len(above) >= 20, plant.part
        assert all(draw["quantity_missed"] for draw in above), plant.part
        assert found["resolution"] == pytest.approx(plant.resolution, rel=1e-12), plant.part
