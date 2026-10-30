"""The controls of ``test_a0.py``: a fault planted in each part the emulator was extended to (the arm's pose, the
dipole's moment and its position, the controller's latency and its law), at three times the comparison's resolution
for the quantity it moves (``a0_plants.py``), fails the comparison, in both directions. A control that does not fail
makes the comparison's pass mean nothing; the rate at which a planted fault is caught is measured, over sizes from
below the resolution to ten times it, by ``measure_a0_controls.py`` (``references/a0_controls_measured.json``)."""

import jax
import pytest
from a0_compare import compare, measure, missed, table
from a0_plants import PLANTS

SIZE = 3.0


@pytest.fixture(autouse=True, scope="module")
def x64():
    """This module runs in float32, as the reference does."""
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", held)


#: Each fault in both directions, the controller's latency (which only delays) in one.
CASES = [
    pytest.param(plant, sign, id=f"{plant.part.replace(' ', '-').replace(chr(39), '')}-{'plus' if sign > 0 else 'minus'}")
    for plant in PLANTS for sign in ((1.0,) if "latency" in plant.part else (1.0, -1.0))]


@pytest.mark.parametrize(("plant", "sign"), CASES)
def test_a_fault_planted_in_each_part_fails_the_comparison(plant, sign):
    rows = compare(measure("3hz", **plant.apply(SIZE, sign)), "3hz")
    print(f"\n{plant.part}: {plant.name}, {SIZE:g} x {plant.resolution}, sign {sign:+.0f}\n" + table(rows))
    assert plant.quantity in missed(rows)
