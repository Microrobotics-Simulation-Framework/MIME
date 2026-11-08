"""The energy ledger on an apparatus (F0-1 M3, M6.2, M6.3), end to end on A0's closed loops.

**A0 on its dynamic rungs** (``tests/script_a0_dynamic.py``, the controller in the graph every step), with
``EnergyLedger``: the arm, the motor and the body are each metered (the joint, spinner and rigid body port kinds), each
step identity within its floor on every step, nothing unattributed on any of them (exactly zero), the books balancing
within the summed floors, and the motor's and the arm's laws booked under their own names: the controller's joint
torques and the motor's torque (supplied by the controller), the magnetic laws' reactions on the rotor and what the
motor hands the arm (its stator's reaction, its rotor's momentum), the arm's weight, damping and stops, the motor's
bearing and its rotor's inequality.

**Attribution's controls**, each caught on every drawn start: an edge carrying a millionth of the motor's reaction into
the arm's summed loads, of the force's reaction into the motor's summed loads, and of the controller's joint torques into
the arm's summed command, each added to the built graph, is unattributed work on its receiver, and the identities still
close (they book it, as unattributed).

**Outside**, from the graph: A0 with its kinematic arm puts the controller, the arm, the motor and the dipole outside,
and the magnetic laws are supplied by the dipole; on its dynamic rungs only the controller is outside, and it supplies
the commands. **The mount tree**, measured on the two mixed apparatus: a dynamic motor on a kinematic arm has the arm
outside and the motor and its magnet inside, the motor's exchange with its host supplied by the arm; a kinematic motor
on a dynamic arm takes no reaction, so its magnet and the arm, which then receives nothing, are outside.

**The field**: script A's uniform field is outside on UniformRotating 0.1.0 and 0.2.0 and on drives.Ramp's frequency,
the torque's paired work supplied by it, the identity closing; a ramp's supplied work is not a constant rate's."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from chain_cases import massive_rpm
from ledger_cases import in_dtype
from script_a import reproduction_arm
from script_a0 import a0
from script_a0_dynamic import a0_dynamic

import script_a as script_a_module
from mime import drives
from mime.build._maddening import run_scan
from mime.core import ledger
from mime.fields import KEPT as KEPT_FIELDS
from mime.observations import EnergyLedger
from mime.units import Hz, s

DTYPES = pytest.mark.parametrize("name", ["float32", "float64"])
#: A0's dynamic loop's run for the books: 100 steps of 0.5 ms, through the rotor's spin-up and the body's start.
STEPS = 100
RECEIVERS = ("umr", "rpm.arm", "rpm.motor")
#: The rows each receiver of A0's dynamic loop books, with each one's class.
ROWS = {
    "rpm.arm": {"controller": "unmetered", "rpm.motor.momentum": "workless", "rpm.motor.reaction": "unmetered",
                "rpm.arm.weight": "potential", "rpm.arm.damping": "dissipative", "rpm.arm.stops": "dissipative"},
    "rpm.motor": {"magnetic.force:umr": "potential", "magnetic.torque:umr": "potential", "controller": "unmetered",
                  "rpm.motor.bearing": "dissipative", "rpm.motor.inertia": "workless",
                  "rpm.motor.reaction": "unmetered"},
}


def floors(records: dict[str, Any]) -> float:
    return float(records["ledger_identity_worst"]) / (ledger.IDENTITY_ULPS * float(
        jnp.finfo(jnp.asarray(records["ledger_identity_worst"]).dtype).eps))


def metered(experiment):
    experiment.observe(EnergyLedger())
    return experiment.build()


@pytest.fixture(scope="module", params=["float32", "float64"])
def dynamic_loop(request):
    """A0's dynamic loop with the ledger, built and run once a dtype."""
    with in_dtype(request.param):
        built = metered(a0_dynamic())
        final = run_scan(built.graph, STEPS)
    return request.param, built, final


def test_each_part_of_the_apparatus_is_metered_and_books_its_laws_under_their_names(dynamic_loop):
    _, built, _ = dynamic_loop
    for name, rows in ROWS.items():
        held = ledger.deliveries(built.graph.get_node(name).constants())
        assert {d.law: d.energy_class for d in held} == rows, (name, held)
    commands = {d.supplied_by for name in ROWS for d in ledger.deliveries(built.graph.get_node(name).constants())
                if d.law == "controller"}
    assert commands == {"controller"}
    assert built.report.energy[-1] == "outside: controller (receives no delivery from the components)"


def test_the_dynamic_loop_s_books_close_on_every_receiver(dynamic_loop):
    dtype, built, final = dynamic_loop
    with in_dtype(dtype):
        for name in RECEIVERS:
            records = final[name]
            assert floors(records) <= 1.0, (name, floors(records))
            assert float(records["ledger_unattributed_worst"]) == 0.0, name
            assert abs(float(ledger.balance(records))) <= float(ledger.balance_floor(records)), name
            held = ledger.deliveries(built.graph.get_node(name).constants())
            gifts = np.asarray(records["ledger_gift_steps"])
            assert not np.any(gifts), (name, dict(zip(ledger.laws(held), gifts, strict=True)))


@contextmanager
def _planted(built, source: str, field: str, target: str, into: str, share: float, units: str) -> Iterator[Any]:
    graph = built.graph
    graph.add_edge(source, target, field, into, transform=lambda value: share * value, additive=True,
                   source_units=units, target_units=units)
    graph.compile()
    yield graph


#: The hidden edges: a millionth of what flows on a real edge, into the summed input of the receiver it reaches.
PLANTS = {
    "the motor's reaction into the arm's loads": ("rpm.motor", "reaction", "rpm.arm", "link_wrench", "N; N m"),
    "the force's reaction into the motor's loads": ("magnetic.force:umr.reaction", "link_wrench", "rpm.motor",
                                                    "link_wrench", "N; N m"),
    "the controller's torques into the arm's command": ("controller", "write_joint_torques", "rpm.arm",
                                                        "joint_torques", "N m"),
}
#: The drawn starts: the body's height above the tube's floor and its roll about the tube's axis (rad).
STARTS = ((0.0, 0.0), (2e-4, 0.7), (5e-4, -1.9), (1e-4, 2.8), (3e-4, -0.4))


@DTYPES
@pytest.mark.parametrize("plant", PLANTS)
def test_a_hidden_edge_on_an_apparatus_is_unattributed_work_on_every_drawn_start(name, plant):
    source, field, target, into, units = PLANTS[plant]
    with in_dtype(name):
        built = metered(a0_dynamic())
        start = dict(built.graph.get_node_state("umr"))
        caught = []
        with _planted(built, source, field, target, into, 1e-6, units) as graph:
            for height, roll in STARTS:
                pose = np.asarray(start["pose"], dtype=np.float64).copy()
                pose[2] += height
                pose[3:] = (math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0)
                graph.set_node_state("umr", {**start, "pose": jnp.asarray(pose, dtype=start["pose"].dtype)})
                final = run_scan(graph, 40)
                caught.append(float(final[target]["ledger_unattributed_worst"]) > 0.0)
                assert floors(final[target]) <= 1.0, "the identity books the hidden edge's work, and closes"
    assert all(caught), caught


def test_outside_on_a0_s_kinematic_loop_is_the_controller_the_arm_the_motor_and_the_dipole():
    built = metered(a0())
    boundary = built.report.energy
    assert boundary[-1] == ("outside: controller, drive:rpm.motor.velocity, rpm.arm, rpm.magnetic, rpm.motor (receives "
                            "no delivery from the components)")
    held = ledger.deliveries(built.graph.get_node("umr").constants())
    assert {d.law: d.supplied_by for d in held if d.law.startswith("magnetic.")} == {
        "magnetic.force:umr": "rpm.magnetic", "magnetic.torque:umr": "rpm.magnetic"}


def test_the_mount_tree_puts_a_kinematic_arm_carrying_a_dynamic_motor_outside_and_the_motor_inside():
    built = metered(massive_rpm(arm="kinematic"))
    assert built.report.energy[-1] == ("outside: drive:rpm.arm.joint_positions, drive:rpm.motor.torque, rpm.arm "
                                       "(receives no delivery from the components)")
    held = {d.law: d for d in ledger.deliveries(built.graph.get_node("rpm.motor").constants())}
    assert held["rpm.motor.inertia"].supplied_by == "rpm.arm" and held["rpm.motor.reaction"].supplied_by == "rpm.arm"
    assert held["rpm.motor.bearing"].supplied_by == ""
    body = {d.law: d.supplied_by for d in ledger.deliveries(built.graph.get_node("umr").constants())
            if d.law.startswith("magnetic.")}
    assert body == {"magnetic.force:umr": "", "magnetic.torque:umr": ""}, "the magnet's reactions reach the motor"


def test_the_mount_tree_puts_a_kinematic_motor_s_magnet_on_a_dynamic_arm_outside():
    built = metered(massive_rpm(motor="kinematic", density=None))
    assert built.report.energy[-1] == ("outside: drive:rpm.arm.joint_torques, drive:rpm.motor.velocity, rpm.arm, "
                                       "rpm.magnetic, rpm.motor (receives no delivery from the components)")
    body = {d.law: d.supplied_by for d in ledger.deliveries(built.graph.get_node("umr").constants())
            if d.law.startswith("magnetic.")}
    assert body == {"magnetic.force:umr": "rpm.magnetic", "magnetic.torque:umr": "rpm.magnetic"}


def _field_run(name: str, kept: bool, frequency: Any) -> Any:
    with in_dtype(name):
        if kept:
            held = script_a_module.UniformRotating
            script_a_module.UniformRotating = KEPT_FIELDS["uniform_rotating", "0.1.0"][0]
            try:
                experiment = reproduction_arm(frequency=frequency)
            finally:
                script_a_module.UniformRotating = held
        else:
            experiment = reproduction_arm(frequency=frequency)
        return metered(experiment).run(s(0.25))


@DTYPES
def test_the_uniform_field_is_outside_on_either_version_and_on_a_ramp(name):
    runs = {"0.1.0": _field_run(name, True, Hz(10.0)), "0.2.0": _field_run(name, False, Hz(10.0)),
            "ramp": _field_run(name, False, drives.Ramp(Hz(0.0), to=Hz(10.0), rate=Hz(40.0) / s(1.0)))}
    supplied = {}
    for version, result in runs.items():
        account = result.energy
        assert account is not None
        assert list(account.supplied) == ["field"], (version, account.text(details=True))
        assert account.unattributed == {"umr": 0.0} and floors(result.final["umr"]) <= 1.0, version
        supplied[version] = account.paired_supply["field"]
    print(f"\n{name}: the field's paired supply over 0.25 s: {supplied}")
    assert all(value > 0.0 for value in supplied.values()), supplied
    assert abs(supplied["ramp"] - supplied["0.2.0"]) > 0.1 * abs(supplied["0.2.0"]), supplied
