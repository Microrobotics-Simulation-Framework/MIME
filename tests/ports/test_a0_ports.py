"""Ports on A0 and on a closed loop: what a declared port moves, and a stage the schedule puts in the wrong place.

- **A one-step latency moves A0** (design section 9, control 2): a command port of one base step's latency on the arm's
  joints, which A0's controller writes inside the compiled step, changes the derived hash and moves the swim speed.
  The control (``latency_dropped``): the port's latency left out of its stage, which must fail the test on every run.
- **A stage after its receiver is refused** (control 3): a command stage that the order places after the coils it
  writes, inside the loop a controller closes through it, would deliver the command a step late; the lag accounting
  refuses that edge (``lag.unplanned``). The control (``stage_after_receiver``): the misplacement planted into the
  loop's build, which must fail the build's test on every run; and the refusal itself, named."""

from __future__ import annotations

import numpy as np
import pytest
from port_cases import fed_back_sphere, planted
from script_a0 import a0

import mime.build._construct as construct
from mime import CommandPort, Measurement
from mime.build import BuildRefused
from mime.units import ms

#: A0's port: one base step (0.5 ms) on the arm's joints.
A0_PORT = {"arm.joint_positions": CommandPort(latency=ms(0.5), reason="one step of the arm's link, for the control")}
LOOP_PORTS = {"currents": CommandPort(latency=ms(1), reason="a test link"),
              "tracker": Measurement(of="sphere.pose", reason="a test tracker")}


@pytest.fixture
def dropped_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    if planted("latency_dropped"):
        from mime.build import _ports

        command_port = _ports.command_port

        def without_latency(*args, **kwargs):
            import dataclasses

            plan = command_port(*args, **kwargs)
            return dataclasses.replace(plan, latency=0) if plan is not None else None

        monkeypatch.setattr(_ports, "command_port", without_latency)


def test_a_one_step_port_latency_on_a0_changes_its_program_and_moves_its_swim_speed(dropped_latency):
    runs = []
    for ports in (None, A0_PORT):
        built = a0("a0_ported", ports=ports).build()
        history = built.run(ms(20), history=True).history
        runs.append((built.report.derived_hash, np.asarray(history["swim_speed:umr"]["speed"])))
    (bare_hash, bare), (ported_hash, ported) = runs
    assert ported_hash != bare_hash
    scale = float(np.max(np.abs(bare)))
    assert float(np.max(np.abs(ported - bare))) > 64 * float(np.finfo(bare.dtype).eps) * scale


@pytest.fixture
def misplaced(monkeypatch: pytest.MonkeyPatch):
    """The order the loop is built in with the coils' command stage moved after the coils it writes."""
    order_of = construct.in_graph_order

    def after_receiver(controller, physics, edges, groups):
        order = order_of(controller, physics, edges, groups)
        order.remove("ports:coils")
        order.insert(order.index("coils.magnetic") + 1, "ports:coils")
        return order

    return lambda: monkeypatch.setattr(construct, "in_graph_order", after_receiver)


def test_the_ported_loop_builds_with_its_stage_between_writer_and_receiver(misplaced):
    if planted("stage_after_receiver"):
        misplaced()
    built = fed_back_sphere(LOOP_PORTS, mode="graph").build()
    order = list(built.graph.node_names)
    assert order.index("controller") < order.index("ports:coils") < order.index("coils.magnetic")


def test_a_stage_after_its_receiver_in_a_loop_is_refused_as_unplanned(misplaced):
    misplaced()
    with pytest.raises(BuildRefused) as refused:
        fed_back_sphere(LOOP_PORTS, mode="graph").build()
    unplanned = [r for r in refused.value.refusals if r.code == "lag.unplanned"]
    assert unplanned and "ports:coils.currents -> coils.magnetic.delivered_currents" in str(unplanned[0])
