"""``ctx.allocator``: the exact allocation through the controller's believed model, its bounds the commands' ports'.

- **It is the hand-written allocation**: on camera spike R2's sphere, :class:`believed_cases.BelievedPD` (the allocator)
  writes the currents ``allocation_cases.AllocatingPD`` writes by hand from the same rows, bounds and scales, bit for
  bit in float64; in float32 their first tick agrees within the allocation's rounding.
- **One function in both modes**: inside the compiled step and as a callback on the host, the first ticks' writes
  agree to rounding, and the callback's report says no derivative passes through it.
- **Bounds from the port**: a 3 A port holds every command within 3 A, the PD asking for more; with 25 A they pass 3 A
  (the bound is read, not a constant).
- **The slew box** (:func:`mime.control._models._slew_steps`): through a port slewing at 2 kA/s, for controller and port
  periods that divide each other and that do not, with latencies on both sides, the stage reaches every target before
  the next arrives, from the run's own port traffic: the box never asks for more than the stage delivers. The
  control, the box as the controller's period alone, is caught on every configuration where the two differ.
- **Where the box's assumption fails** (the stage not at ``last``): the first tick from a rest that is not the law's
  initial ``last``, and after a trip. The allocation asks for more than the stage delivers; the recorded traffic shows
  it, the build report says so beside the box, and the run flags a trip (a slew-limited delivery raises no flag).
- **Refusals, each with its own code and its control**: a command with no bounds (``allocation.bounds``), an objective
  naming no quantity of the model or a scale of another unit (``allocation.objective``). Each is raised inside
  ``law(ctx)`` and surfaces with its code; a law raising a plain error is still its entry's defect.
- **The hand-over**: commands no affine model takes (an arm's joint positions) and more commands than
  ``EXACT_COMMANDS``, which the exact path refused, take the iterative path, chosen when the law is made, and the
  report names the path and why."""

from __future__ import annotations

import contextlib
import os
import warnings
from collections.abc import Iterator
from typing import Any

import jax
import numpy as np
import pytest

from allocation_cases import allocating_sphere
from believed_cases import LIMIT, REASON, AllocatingArmProbe, AllocatingProbe, GlitchedPD, believed_sphere
from chain_cases import TRACKING_READS, kinematic_rpm

from mime.build import BuildRefused
from mime.control import _models
from mime.control._models import _slew_steps
from mime.experiment import Experiment
from mime.things import CommandPort
from mime.units import A, N, T, m, ms, rad, s

jnp: Any = jax.numpy
SLEW = 2000.0
#: The plant a control's detection rate is measured with (``AP3_PLANT``): ``slew``, the box as the controller's period
#: alone; ``nobox``, the box not there (the slew's steps infinite), caught by the allocation from rest;
#: ``defect``, a refusal raised inside ``law(ctx)`` made its entry's defect, as before AP3a.
PLANT = os.environ.get("AP3_PLANT", "")


@contextlib.contextmanager
def in_dtype(name: str) -> Iterator[None]:
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", name == "float64")
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", held)


def history(experiment: Experiment, steps: int = 30) -> dict[str, dict[str, np.ndarray]]:
    held = experiment.build().run(ms(steps), history=True).history
    return {node: {field: np.asarray(value) for field, value in fields.items()} for node, fields in held.items()}


def refused(experiment: Experiment) -> BuildRefused:
    with pytest.raises(BuildRefused) as caught:
        experiment.build()
    return caught.value


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_the_allocator_writes_the_hand_written_allocation(dtype):
    with in_dtype(dtype):
        through = history(believed_sphere(), 60)["controller"]["write_currents"]
        by_hand = history(allocating_sphere(), 60)["controller"]["write_currents"]
    if dtype == "float64":
        assert np.array_equal(through, by_hand)
    else:
        # the first tick, from the same state: the two order the rows' products otherwise
        assert np.max(np.abs(through[0] - by_hand[0])) <= 1e3 * float(np.finfo(np.float32).eps) * LIMIT
    assert np.max(np.abs(through)) == pytest.approx(LIMIT)


def test_the_allocator_is_one_function_in_both_modes():
    with in_dtype("float64"):
        graph = history(believed_sphere(), 15)["ports:coils"]["written_currents"]
        built = believed_sphere(mode="callback").build()
        callback = np.asarray(built.run(ms(15), history=True).history["ports:coils"]["written_currents"])
    assert np.max(np.abs(graph[:10] - callback[:10])) <= 1e3 * float(np.finfo(np.float64).eps) * LIMIT
    assert any("none pass through it in mode 2" in line for line in built.report.controller)


@pytest.mark.parametrize("limit, held", [(3.0, True), (LIMIT, False)], ids=["at_3_A", "control_at_25_A"])
def test_the_allocation_is_within_the_ports_bounds(limit, held):
    currents = history(believed_sphere(port=CommandPort(bounds=A(limit), reason=REASON)), 40)["controller"]["write_currents"]
    assert bool(np.max(np.abs(currents)) <= 3.0) is held
    assert np.max(np.abs(currents)) <= limit


# -- the slew box -------------------------------------------------------------------------------------------------

#: (controller period, port period, controller latency, port latency), base steps.
CONFIGURATIONS = [(3, 1, 1, 0), (3, 2, 1, 0), (2, 3, 0, 1), (4, 2, 1, 2), (5, 3, 2, 1), (2, 4, 1, 0)]


def slewed(controller: int, port: int, controller_latency: int, port_latency: int) -> Experiment:
    held = CommandPort(bounds=A(LIMIT), slew=A(SLEW) / s(1.0), rate=ms(port), latency=ms(port_latency) if port_latency
                       else None, reason=REASON)
    return believed_sphere(port=held, rate=ms(controller), latency=ms(controller_latency) if controller_latency else
                           ms(0))


def misses(history_: dict[str, dict[str, np.ndarray]], port: int, port_latency: int) -> float:
    """The most the stage's slewed value falls short of a target at the step before the next target arrives: the
    targets rebuilt from what the writer gave (the port's sample and delay of ``written``), the slewed value read."""
    stage = history_["ports:coils"]
    written, amplified = stage["written_currents"], stage["amplified_currents"]
    steps = np.arange(written.shape[0])
    sampled = written[port * (steps // port)]
    target = np.where((steps >= port_latency)[:, None], sampled[np.maximum(steps - port_latency, 0)], 0.0)
    target = np.clip(target, -LIMIT, LIMIT)
    change = np.flatnonzero(np.any(target[1:] != target[:-1], axis=1))
    return float(np.max(np.abs(amplified[change] - target[change]))) if change.size else 0.0


@pytest.mark.parametrize("configuration", CONFIGURATIONS, ids=[f"P{c[0]}_R{c[1]}_L{c[2]}_{c[3]}" for c in CONFIGURATIONS])
@pytest.mark.parametrize("planted", [False, True], ids=["as_built", "control_the_controllers_period"])
def test_the_slew_box_never_asks_for_more_than_the_stage_delivers(configuration, planted, monkeypatch):
    controller, port, controller_latency, port_latency = configuration
    if planted or PLANT == "slew":
        monkeypatch.setattr(_models, "_slew_steps", lambda period, rate: float(period))
    with in_dtype("float64"):
        found = misses(history(slewed(*configuration), 40), port, port_latency)
    rounding = 64 * float(np.finfo(np.float64).eps) * LIMIT
    differs = _slew_steps(controller, port) != controller
    assert bool(found <= rounding) is not (planted and differs), (configuration, found)


# -- where the box's assumption does not hold -------------------------------------------------------------------------
#
# The box is last +/- slew * dt * g, taken around where the stage stood at the write: it assumes the stage stood at
# ``last``. Two places it did not: the first tick from a rest that is not the law's initial ``last``, and after a trip
# latches the port to its rest. There the allocation asks for more than the stage delivers; what shows it is the
# port's recorded traffic (what was written against what was delivered), the report's line beside each boxed command,
# and, for a trip alone, the run's own flag. A slew-limited delivery raises no flag today.

RECORD = ["ports:coils.written_currents", "ports:coils.currents"]
REST = 10.0


def asked_and_delivered(experiment: Experiment, steps: int = 40) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    result = experiment.build().run(ms(steps), record=RECORD, history=True)
    stage = {field: np.asarray(value) for field, value in result.history["ports:coils"].items()}
    return result, stage["written_currents"], stage["currents"], stage["amplified_currents"]


@pytest.mark.parametrize("rest", [0.0, REST], ids=["control_the_stage_stands_at_the_laws_last", "the_stage_at_a_rest_elsewhere"])
def test_from_rest_the_box_asks_for_more_than_the_stage_delivers_where_the_stage_is_not_at_the_laws_last(rest, monkeypatch):
    """The law's first ``last`` is zero. A port resting at 0 A is where the law takes it; one resting at 10 A is not,
    and the commands asked for in the first writes are further from where the stage stands than it moves before the next:
    the shortfall the recorded traffic shows, against zero (to rounding) where the assumption holds."""
    if PLANT == "nobox":  # the control: no box at all, so the stage is asked for more than it delivers from its own rest
        monkeypatch.setattr(_models, "_slew_steps", lambda period, rate: float("inf"))
    held = CommandPort(bounds=A(LIMIT), slew=A(SLEW) / s(1.0), rate=ms(1), rest=A((rest, 0.0, 0.0)) if rest else None,
                       reason=REASON)
    experiment = believed_sphere(port=held, rate=ms(3), latency=ms(1))
    with in_dtype("float64"):
        result, written, delivered, amplified = asked_and_delivered(experiment)
        found = misses({"ports:coils": {"written_currents": written, "amplified_currents": amplified}}, 1, 0)
        report = "\n".join(experiment.build().report.controller)
    assert "taken to stand at the last command" in report
    assert PLANT == "nobox" or "currents: last +/- 6.0 per write" in report
    assert not [read for read in result.flags if read.was_raised]  # nothing is raised where the slew limits the stage
    rounding = 64 * float(np.finfo(np.float64).eps) * LIMIT
    if rest:
        assert found > 1.0, found  # asked and delivered, from the traffic: the stage is still on its way
        assert float(np.max(np.abs(written[1:] - delivered[1:]))) > 1.0
    else:
        assert found <= rounding, found


def test_after_a_trip_the_allocation_asks_for_more_than_the_stage_delivers_and_the_run_flags_the_trip():
    """A glitch beyond the port's bounds trips it; the stage then drives to its rest and stays, while the law, taking
    the box around the commands it allocated, goes on asking. The trip is flagged by the run (a warning); the
    shortfall is in the traffic. The control, the same law with no glitch, is delivered and raises nothing."""
    held = CommandPort(bounds=A(LIMIT), on_bound="trip", slew=A(SLEW) / s(1.0), rate=ms(1), reason=REASON)
    shortfalls = {}
    for glitch in (0.0, 30.0):
        experiment = believed_sphere(port=held, controller=GlitchedPD(glitch=glitch, glitch_at=4.0), rate=ms(3),
                                     latency=ms(1))
        with in_dtype("float64"), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result, written, delivered, _ = asked_and_delivered(experiment, 60)
        tripped = [read.path for read in result.flags if read.was_raised and read.path.endswith("tripped_currents")]
        shortfalls[glitch] = float(np.max(np.abs(np.clip(written[-1], -LIMIT, LIMIT) - delivered[-1])))
        assert bool(tripped) is bool(glitch), (glitch, tripped)
        assert any("tripped at its bounds" in str(warning.message) for warning in caught) is bool(glitch)
    assert shortfalls[0.0] <= 1e-9 and shortfalls[30.0] > 1.0, shortfalls


def test_the_slew_steps_are_the_controllers_period_where_one_period_divides_the_other():
    for controller in range(1, 9):
        for port in range(1, 9):
            held = _slew_steps(controller, port)
            if controller % port == 0 or port % controller == 0:
                assert held == controller
            else:
                assert 0 < held < controller


# -- refusals -----------------------------------------------------------------------------------------------------

def test_a_command_with_no_bounds_is_refused_with_its_code_and_its_control_builds(monkeypatch):
    if PLANT == "defect":
        from mime.control import _run

        monkeypatch.setattr(_run, "MimeError", type("NoMimeError", (Exception,), {}))
    caught = refused(believed_sphere(port=CommandPort(latency=ms(1), reason=REASON)))
    assert caught.codes() == ["allocation.bounds"] and "currents has no bounds" in str(caught)
    believed_sphere(port=CommandPort(bounds=A(LIMIT), latency=ms(1), reason=REASON)).build()


def test_commands_no_affine_model_takes_go_to_the_iterative_path_and_an_affine_ones_to_the_exact_one():
    experiment = kinematic_rpm(controlled=("arm.joint_positions",))
    experiment.set_controller(AllocatingArmProbe(apparatus="rpm", commands=("arm.joint_positions",),
                                                 objective=(("link_6.position", m(1e-3)),
                                                            ("link_6.orientation", rad(1e-2)))),
                              reads={"pose": TRACKING_READS["body_pose"]},
                              writes={"joint_positions": "rpm.arm.joint_positions"}, mode="graph")
    lines = "\n".join(experiment.build().report.controller)
    assert "iterative (its model is not affine in joint_positions), 6 commands of rpm.arm's model" in lines
    assert "exact, 3 commands" in "\n".join(believed_sphere(controller=AllocatingProbe()).build().report.controller)


def test_commands_naming_no_one_part_are_refused_and_a_parts_build():
    caught = refused(believed_sphere(controller=AllocatingProbe(commands=("torques",))))
    assert caught.codes() == ["allocation.model"] and "name no one provider or segment" in str(caught)
    believed_sphere(controller=AllocatingProbe()).build()


@pytest.mark.parametrize("objective", [(("flux", N(1e-3)),), (("force", A(1.0)),), (("force", N(-1.0)),)],
                         ids=["no_such_quantity", "another_unit", "not_above_zero"])
def test_an_objective_the_model_cannot_weigh_is_refused_and_its_control_builds(objective):
    assert refused(believed_sphere(controller=AllocatingProbe(objective=objective))).codes() == ["allocation.objective"]
    believed_sphere(controller=AllocatingProbe(objective=(("flux_density", T(1e-2)),))).build()


def test_more_commands_than_the_exact_path_takes_go_to_the_iterative_path_citing_its_constant(monkeypatch):
    lines = "\n".join(believed_sphere(controller=AllocatingProbe()).build().report.controller)
    assert "exact, 3 commands of coils.magnetic's model, affine in them, at most EXACT_COMMANDS = 8" in lines
    monkeypatch.setattr(_models, "EXACT_COMMANDS", 2)
    lines = "\n".join(believed_sphere(controller=AllocatingProbe()).build().report.controller)
    assert "iterative (3 commands, more than EXACT_COMMANDS = 2)" in lines


def test_a_law_raising_a_plain_error_is_still_its_entrys_defect():
    assert refused(believed_sphere(controller=AllocatingProbe(raising=True))).codes() == ["entry.defect"]
