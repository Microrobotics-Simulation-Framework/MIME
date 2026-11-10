"""The flow's ledger flags (the ledger harness's second rung, ``mime.nodes.flow.fvm._flags``): ratios of a metered
step's terms, read from the node's declarations by ``mime.testing``'s harness, each quiet on correct runs and each
caught where its planted bug is. **Their rates are measured on Actions** over 200 fresh draws a dtype (the PR's table,
and each flag's ``evidence``); here a few draws of the same envelope (``flag_cases``), disjoint from the 200 the lines
were measured on.

- **Declared as measured**: an active flag carries a line in each dtype and a plant caught on every run with no false
  alarm, its run linked; the openings' two are pending and never counted as checks. The control: a flag declared a
  check with no measured rate, and one pending with evidence, are each named.
- **Quiet on correct runs**: each active flag's worst reading under its line, in both dtypes.
- **Each flag's plant caught** (the control): a solve stopped at a tolerance 1e4 looser (the stall), a convective term
  formed from the run's starting velocity (convection), the fluid's mask lost beyond the band (the leak). The pending
  openings' two are never counted."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from flag_cases import PLANTS, draw, readings

from mime.core.ledger import LedgerFlag
from mime.nodes.flow.fvm import DEFAULT_TOLERANCE, STALL_LINES, STALL_PER_TOLERANCE, ExplicitFlowNode, ImplicitFlowNode
from mime.nodes.flow.fvm._flags import FLAGS
from mime.testing import active_flags, flag_problems, judge_flags, ledger_closes

#: Each active flag's plant caught on every run (its rate in its evidence).
CAUGHT = {"stall": "solve_tolerance_1e4", "convection": "convection_stale", "leak": "leak_mask"}
#: CI's draws: outside the seeds the lines were measured on (0 to 199) and the rates' (1000 on).
SEEDS = (500, 501)


@pytest.mark.parametrize("node", [ImplicitFlowNode, ExplicitFlowNode])
def test_the_flags_are_declared_as_measured(node):
    assert flag_problems(node.LEDGER_FLAGS) == []


def test_the_pending_flags_are_never_counted_as_checks():
    node = ImplicitFlowNode.__new__(ImplicitFlowNode)
    report = judge_flags(node, {}, "float64")
    assert set(report.pending) == {"open_face_projection", "port_kinetic_flux"}
    assert {r.flag for r in report.readings} == set(CAUGHT) == {f.name for f in active_flags(node)}
    assert report.checks == len(CAUGHT)


def test_a_run_with_no_check_does_not_pass_and_a_flag_that_read_nothing_is_silent():
    """RK3 declares only the pending two: its report has no check and does not pass. A flag that read exactly zero at
    every judged step (convection's where the node's convection is off) is listed as silent, not counted; a flag with
    no line in the run's dtype fails, named."""
    explicit = judge_flags(ExplicitFlowNode.__new__(ExplicitFlowNode), {}, "float64")
    assert explicit.checks == 0 and not explicit.passed and "nothing was checked" in explicit.failures()[0]
    node = ImplicitFlowNode.__new__(ImplicitFlowNode)
    quiet = {flag: np.full(40, 1e-12) for flag in CAUGHT}
    report = judge_flags(node, {**quiet, "convection": np.zeros(40)}, "float64")
    assert report.silent == ("convection",) and report.checks == len(CAUGHT) - 1 and report.passed
    unlined = judge_flags(node, quiet, "float16")
    assert set(unlined.unlined) == set(CAUGHT) and not unlined.passed
    assert any("no measured line" in failure for failure in unlined.failures())


def test_control_a_flag_claimed_without_its_measurement_is_named():
    stall = ImplicitFlowNode.LEDGER_FLAGS[0]
    unproven = dataclasses.replace(stall, evidence=())
    assert any("coverage claimed" in problem for problem in flag_problems((unproven,)))
    missed = dataclasses.replace(stall, evidence=(dataclasses.replace(stall.evidence[0], caught=0),))
    assert any("coverage claimed" in problem for problem in flag_problems((missed,)))
    pending = dataclasses.replace(stall, pending="not yet")
    assert any("pending and carries evidence" in problem for problem in flag_problems((pending,)))
    lineless = dataclasses.replace(stall, line={"float64": 1e-5})
    assert any("no measured line in float32" in problem for problem in flag_problems((lineless,)))


@pytest.mark.parametrize("seed", SEEDS)
def test_each_active_flag_is_quiet_on_a_correct_run(dtype, seed):
    ended: list = []
    node, found = readings(draw(seed), dtype, ended=ended)
    report = judge_flags(node, found, dtype)
    assert report.checks == len(CAUGHT) and report.passed, report.failures()
    judged = ledger_closes(ended[0])
    assert judged.passed, judged.detail


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("flag", sorted(CAUGHT))
def test_control_each_flag_catches_its_plant(dtype, flag, seed, monkeypatch):
    plant = CAUGHT[flag]
    PLANTS[plant](monkeypatch.setattr)
    node, found = readings(draw(seed), dtype, plant)
    report = judge_flags(node, found, dtype)
    assert flag in {r.flag for r in report.readings if not r.passed}, report.readings


# -- the stall line: one number a dtype, in solve tolerances ---------------------------------------------------------
STALL = next(flag for flag in FLAGS if flag.name == "stall")
#: The node's own stall_flagged as a reading (1 where it is set), beside the harness's.
NODE_FLAG = LedgerFlag("node_flag", "stall_flagged", lambda node, p, before, after: after["stall_flagged"],
                       {"float64": 0.5, "float32": 0.5}, "")
#: Draws of the seeds the line was measured on whose reading at a tenfold looser tolerance is over the harness's line at
#: the default (3.3e-5 against 1e-5 in float64; 2.5e-2 against 5e-3 in float32): where the tying to the tolerance
#: decides the flag.
LOOSE_DRAWS = {"float64": 111, "float32": 159}
NOMINAL = {"float64": 1e-8, "float32": 1e-5}


def test_the_harnesss_line_is_the_nodes_line_at_the_default_tolerance():
    """One number a dtype: the harness's line is the node's, in tolerances, at the default tolerance."""
    assert dict(STALL.line) == STALL_LINES
    for name, line in STALL_LINES.items():
        assert line == STALL_PER_TOLERANCE[name] * DEFAULT_TOLERANCE[name] == STALL_PER_TOLERANCE[name] * NOMINAL[name]


@pytest.mark.parametrize("plant", [None, "solve_stops_short_1e4"])
def test_the_nodes_flag_is_the_harnesss_line_on_every_step_at_the_default_tolerance(dtype, plant, monkeypatch):
    """On a correct run neither flag is set; on a solve stopped short while the node declares its default tolerance
    both are, on the same steps."""
    if plant:
        PLANTS[plant](monkeypatch.setattr)
    _, found = readings(draw(SEEDS[0]), dtype, plant, flags=[STALL, NODE_FLAG])
    harness, node = np.abs(found["stall"]) > STALL_LINES[dtype], found["node_flag"] > 0.5
    assert np.array_equal(harness, node)
    assert bool(node.any()) == (plant is not None)


def test_a_looser_tolerance_is_judged_against_that_solve(dtype):
    """Control: the draw's reading at a tenfold looser tolerance is over the default tolerance's line (so a line that
    ignored the tolerance would flag a correct run that asked for a looser solve), and under its own."""
    loose = 10.0 * NOMINAL[dtype]
    node, found = readings(draw(LOOSE_DRAWS[dtype]), dtype, flags=[STALL, NODE_FLAG], settings={"tolerance": loose})
    reading = np.abs(found["stall"])
    assert reading.max() > STALL_LINES[dtype], "the control cannot discriminate: the reading is under the default line"
    assert reading.max() < STALL_PER_TOLERANCE[dtype] * loose
    assert not (found["node_flag"] > 0.5).any()
