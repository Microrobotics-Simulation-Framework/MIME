"""The controls of ``test_a0.py``: a fault planted in each part the emulator was extended to (the arm's pose, the
dipole's moment and its position, the controller's latency and its law) and in the motor's speed and the gradient force
alone, at three times the comparison's resolution for the quantity it moves (``a0_plants.py``), fails the comparison,
in both directions. A control that does not fail makes the comparison's pass mean nothing; the rate at which a planted
fault is caught is measured, over sizes from below the resolution to ten times it, by ``measure_a0_controls.py``
(``references/a0_controls_measured.json``), and the record is checked here against the constants the plants use."""

import json
from pathlib import Path

import jax
import pytest
from a0_compare import compare, measure, missed, table
from a0_plants import GRADIENT_FORCE_RESOLUTION, MOMENT_LAG_SENSITIVITY, MOTOR_RESOLUTION, PLANTS

SIZE = 3.0
RECORD = json.loads((Path(__file__).parent / "references" / "a0_controls_measured.json").read_text())


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
    with plant.planted(SIZE, sign):
        rows = compare(measure("3hz", **plant.apply(SIZE, sign)), "3hz")
    print(f"\n{plant.part}: {plant.name}, {SIZE:g} x {plant.resolution}, sign {sign:+.0f}\n" + table(rows))
    assert plant.quantity in missed(rows)


def test_the_plants_resolutions_are_the_ones_the_record_measured():
    """The constants the plants' sizes rest on are declarations until a measurement fits them: the dipole moment's
    response of the lag (rad per unit of relative moment) and the gradient force's resolution are those the record
    measured, to 10 %; the record is of these plants (their resolutions as stated), and every fault above its
    resolution was caught by the quantity it moves most, in every draw."""
    sensitivities = RECORD["sensitivities"]
    assert MOMENT_LAG_SENSITIVITY == pytest.approx(sensitivities["moment"]["response_per_unit"]["0.001"], rel=0.1)
    assert GRADIENT_FORCE_RESOLUTION == pytest.approx(sensitivities["gradient_force"]["resolution_by_size"]["0.01"],
                                                      rel=0.1)
    motor = sensitivities["motor_speed"]["planted"]
    assert all(abs(motor[f"{sign}{MOTOR_RESOLUTION:g}"]["miss_in_tolerances"]) > 3 for sign in "+-"), motor
    assert {plant.part: plant.resolution for plant in PLANTS} == {
        part: found["resolution"] for part, found in RECORD["plants"].items()}
    whole = {plant.part for plant in PLANTS if not plant.sub_resolution}
    for part, found in RECORD["plants"].items():
        above = sum(1 for draw in found["draws"] if draw["size_in_resolutions"] >= 1.5 or part in whole)
        assert above >= 20, part
        assert found["caught_above_the_resolution"] == f"{above}/{above}", part
        assert found["quantity_missed_above"] == f"{above}/{above}", part
