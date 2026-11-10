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

import pytest
from flag_cases import PLANTS, draw, readings

from mime.nodes.flow.fvm import ExplicitFlowNode, ImplicitFlowNode
from mime.testing import active_flags, flag_problems, judge_flags

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
    node, found = readings(draw(seed), dtype)
    report = judge_flags(node, found, dtype)
    assert report.checks == len(CAUGHT) and report.passed, report.failures()


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("flag", sorted(CAUGHT))
def test_control_each_flag_catches_its_plant(dtype, flag, seed, monkeypatch):
    plant = CAUGHT[flag]
    PLANTS[plant](monkeypatch.setattr)
    node, found = readings(draw(seed), dtype, plant)
    report = judge_flags(node, found, dtype)
    assert flag in {r.flag for r in report.readings if not r.passed}, report.readings
