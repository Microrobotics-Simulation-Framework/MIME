"""``ctx.allocator``: the exact allocation through the controller's believed model, its bounds the commands' ports'.

- **It is the hand-written allocation**: on camera spike R2's sphere, :class:`believed_cases.BelievedPD` (the allocator)
  writes the currents ``allocation_cases.AllocatingPD`` writes by hand from the same rows, bounds and scales, bit for
  bit in float64; in float32 their first tick agrees within the allocation's rounding.
- **One function in both modes**: inside the compiled step and as a callback on the host, the first ticks' writes
  agree to rounding, and the callback's report says no derivative passes through it.
- **Bounds from the port**: a 3 A port holds every command within 3 A, the PD asking for more; with 25 A they pass 3 A
  (the bound is read, not a constant).
- **The slew box** (:func:`mime.control.slew_steps`): through a port slewing at 2 kA/s, for controller and port
  periods that divide each other and that do not, with latencies on both sides, the stage reaches every target before
  the next arrives, from the run's own port traffic: the box never asks for more than the stage delivers. The
  control, the box as the controller's period alone, is caught on every configuration where the two differ.
- **Refusals, each with its own code and its control**: a command with no bounds (``allocation.bounds``), commands no
  affine model takes (``allocation.model``: an arm's joint positions), an objective naming no quantity of the model or
  a scale of another unit (``allocation.objective``), more commands than ``EXACT_COMMANDS`` (``allocation.exact-size``,
  citing the constant). Each is raised inside ``law(ctx)`` and surfaces with its code; a law raising a plain error is
  still its entry's defect."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from typing import Any

import jax
import numpy as np
import pytest

from allocation_cases import allocating_sphere
from believed_cases import LIMIT, REASON, AllocatingArmProbe, AllocatingProbe, believed_sphere
from chain_cases import TRACKING_READS, kinematic_rpm

from mime.build import BuildRefused
from mime.control import _models, slew_steps
from mime.experiment import Experiment
from mime.things import CommandPort
from mime.units import A, N, T, ms, s

jnp: Any = jax.numpy
SLEW = 2000.0
#: The plant a control's detection rate is measured with (``AP3_PLANT``): ``slew``, the box as the controller's period
#: alone; ``defect``, a refusal raised inside ``law(ctx)`` made its entry's defect, as before AP3a.
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
        monkeypatch.setattr(_models, "slew_steps", lambda period, rate: float(period))
    with in_dtype("float64"):
        found = misses(history(slewed(*configuration), 40), port, port_latency)
    rounding = 64 * float(np.finfo(np.float64).eps) * LIMIT
    differs = slew_steps(controller, port) != controller
    assert bool(found <= rounding) is not (planted and differs), (configuration, found)


def test_the_slew_steps_are_the_controllers_period_where_one_period_divides_the_other():
    for controller in range(1, 9):
        for port in range(1, 9):
            held = slew_steps(controller, port)
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


def test_commands_no_affine_model_takes_are_refused_and_an_affine_ones_build():
    experiment = kinematic_rpm(controlled=("arm.joint_positions",))
    experiment.set_controller(AllocatingArmProbe(apparatus="rpm", commands=("arm.joint_positions",),
                                                 objective=(("frame_poses", N(1.0)),)),
                              reads={"pose": TRACKING_READS["body_pose"]},
                              writes={"joint_positions": "rpm.arm.joint_positions"}, mode="graph")
    with pytest.raises(BuildRefused) as caught:
        experiment.build()
    assert "allocation.model" in caught.value.codes()
    believed_sphere(controller=AllocatingProbe()).build()


@pytest.mark.parametrize("objective", [(("flux", N(1e-3)),), (("force", A(1.0)),), (("force", N(-1.0)),)],
                         ids=["no_such_quantity", "another_unit", "not_above_zero"])
def test_an_objective_the_model_cannot_weigh_is_refused_and_its_control_builds(objective):
    assert refused(believed_sphere(controller=AllocatingProbe(objective=objective))).codes() == ["allocation.objective"]
    believed_sphere(controller=AllocatingProbe(objective=(("flux_density", T(1e-2)),))).build()


def test_more_commands_than_the_exact_path_takes_are_refused_citing_its_constant(monkeypatch):
    believed_sphere(controller=AllocatingProbe()).build()
    monkeypatch.setattr(_models, "EXACT_COMMANDS", 2)
    caught = refused(believed_sphere(controller=AllocatingProbe()))
    assert caught.codes() == ["allocation.exact-size"] and "EXACT_COMMANDS = 2" in str(caught)


def test_a_law_raising_a_plain_error_is_still_its_entrys_defect():
    assert refused(believed_sphere(controller=AllocatingProbe(raising=True))).codes() == ["entry.defect"]
