"""The energy ledger looks through a port's stage: a command a metered receiver pairs (a dynamic motor's torque) is
booked under its writer, the drive, as it is without the port, so the supply the books name is the drive's. The
stage carries a signal and is outside the ledger, as a drive and a controller are.

The control (``ledger_blind``): the look-through removed, the torque is booked under the stage, which supplies
nothing: the test must fail on every run."""

from __future__ import annotations

import pytest
from chain_cases import massive_rpm
from port_cases import planted

from mime import CommandPort
from mime.observations import EnergyLedger
from mime.units import ms


@pytest.fixture
def blind(monkeypatch: pytest.MonkeyPatch) -> None:
    if planted("ledger_blind"):
        monkeypatch.setattr("mime.build._ledger.writers_through", lambda edges: {})


def test_a_metered_receivers_command_is_booked_under_its_writer_through_a_port(blind):
    exp = massive_rpm("ported_ledger", ports={"motor.torque": CommandPort(latency=ms(1), reason="a test link")})
    exp.observe(EnergyLedger())
    built = exp.build()
    assert "ports:rpm" in built.graph.node_names
    lines = [line.strip() for line in built.report.energy]
    assert "drive:rpm.motor.torque: unmetered (torque); supplied by drive:rpm.motor.torque" in lines
    assert lines[-1].startswith("outside: drive:rpm.arm.joint_torques, drive:rpm.motor.torque")
