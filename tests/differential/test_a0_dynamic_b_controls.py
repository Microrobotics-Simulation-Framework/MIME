"""The controls of the second declaration of A0 on its dynamic rungs (``test_a0_dynamic.py``'s gate), at 3 Hz and
10 Hz: each fault the declaration names (``a0_dynamic_b_plants.py``) fails the gate it must, here on every run, and
over 20 runs on the stress workflow. Three are the faults the review of PR #77 planted that passed the first
comparison's rule: the motor's angle offset at its start, the magnet turned in its mount, a downstream drift between the
rotor and the field. The other two hold the new gates' resolutions: the rotor's J 4 % large (the kick, which the first
declaration's transient tolerance let through) and the rotor loop's Ki (the integral, the absolute phase the loop
holds).

Each plant sits at least 1.5 times its gate's resolution as measured on 0.3 (the gate's tolerance over its response at
a second size; ``measure_a0_dynamic_b_controls.py``, ``references/a0_dynamic_b_controls_measured.json``), which the
last test holds to the record."""

import json
from pathlib import Path

import jax
import pytest
from a0_dynamic_b_plants import GATE, J_FACTOR, MOUNT_TURN, ANGLE_OFFSET, drifted_gates, ki_step, KI_SIZE, planted
from a0_dynamic_compare import FREQUENCY, SECOND, measure, second, second_failures, second_table, transient

RECORD_PATH = Path(__file__).parent / "references" / "a0_dynamic_b_controls_measured.json"


@pytest.fixture(autouse=True, scope="module")
def x64():
    """This module runs in float32, as the reference does."""
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", held)


@pytest.fixture(scope="module", params=list(FREQUENCY))
def unplanted(request):
    found, run = measure(request.param)
    return request.param, found, run


@pytest.mark.parametrize("key", list(FREQUENCY))
def test_the_motors_angle_offset_at_its_start_fails_the_initial_angle(key):
    """2^-14 rad on the motor's initial angle: the absolute phase is gated where it is set."""
    found, run = planted(key, "angle offset")
    gates = second(key, found, run)
    print(f"\n{key}, the angle offset {ANGLE_OFFSET:.3g} rad\n" + second_table(gates))
    assert GATE["angle offset"] in second_failures(gates)


@pytest.mark.parametrize("key", list(FREQUENCY))
def test_the_magnet_turned_in_its_mount_fails_its_mount_phase_and_the_fields_phase(key):
    """The magnet turned 1e-4 rad about the motor's axis, in its mount on the rotor: the first declaration's gates
    passed it all; here both the magnet's phase in its mount and the field's phase against the angle fail."""
    found, run = planted(key, "mount")
    gates = second(key, found, run)
    print(f"\n{key}, the magnet turned {MOUNT_TURN:g} rad in its mount\n" + second_table(gates))
    failed = second_failures(gates)
    assert {"magnet_mount_phase_rad", "field_phase_against_motor_angle_rad"} <= set(failed), failed


def test_a_downstream_drift_fails_the_fields_turning_against_the_angle(unplanted):
    """The field and the body turned by r t about the tube's axis, the motor's angle untouched (three field-turning
    tolerances): the field's turning against the motor's angle fails."""
    key, found, run = unplanted
    gates = drifted_gates(key, found, run)
    print(f"\n{key}, a downstream drift\n" + second_table(gates))
    assert GATE["drift"] in second_failures(gates)


@pytest.mark.parametrize("key", list(FREQUENCY))
def test_the_rotors_j_four_percent_large_fails_the_kick(key):
    """The first declaration's transient tolerance (2.8e-2) passed a 4 % error in the rotor's J; the second's, ten
    times the single-precision twin's gap, does not."""
    _, run = planted(key, "J")
    kick = transient(run)
    predicted = SECOND["transient"]["predicted"][key]
    misses = {name: kick[name] / predicted[name] - 1 for name in kick}
    print(f"\n{key}, J x {J_FACTOR:g}: {misses} against {SECOND['transient']['tolerance']:.3g}")
    assert abs(misses[GATE["J"]]) > SECOND["transient"]["tolerance"]


@pytest.mark.parametrize("sign", [1.0, -1.0], ids=["plus", "minus"])
@pytest.mark.parametrize("key", list(FREQUENCY))
def test_the_rotor_loops_ki_fails_its_integral(key, sign):
    """Ki scaled by three of the integral gate's resolutions either way: the angle the loop holds the rotor behind its
    command moves by three tolerances."""
    found, run = planted(key, "Ki", sign=sign)
    gates = second(key, found, run)
    print(f"\n{key}, Ki x (1 {sign:+.0f} {KI_SIZE:g} x {ki_step(key):.3g})\n" + second_table(gates))
    assert GATE["Ki"] in second_failures(gates)


def test_each_plant_is_above_its_gates_measured_resolution():
    """From the record: each gate's response measured on 0.3 at the plant's second size, its resolution (tolerance
    over response), and each plant at least 1.5 of them, at both frequencies; the gate held exactly (the angle at the
    start, tolerance zero) responds to its plant."""
    record = json.loads(RECORD_PATH.read_text())
    for key in FREQUENCY:
        plants = record["by_frequency"][key]["plants"]
        assert set(plants) == set(GATE), key
        for plant, found in plants.items():
            assert found["gate"] == GATE[plant], plant
            if found["tolerance"] == 0.0:
                assert found["plant_in_resolutions"] is None and found["response_per_declared_plant"] != 0.0, plant
            else:
                assert found["plant_in_resolutions"] >= 1.5, (key, plant, found["plant_in_resolutions"])
