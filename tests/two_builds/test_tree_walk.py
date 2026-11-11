"""The tube tree's wall under the two-builds check: its distance is made in the program from the tree's leaves (the
union's candidates chosen there, where its check fails), and only how many faces they may run at is held at build, a
rung above the count the build's values choose (``mime.nodes.flow.fvm.plan_rung``), as the cell counts are held. So a
move of the radii keeps one program; the registry walk moves every float of ``tree_flow``'s document, the radii and the
period's length among them (``test_two_builds_walk.py``).

**The control:** the plan's size made the build's own count (the rung the identity), a move of the radii that changes
that count gives the two builds programs of two shapes; the move is a relative 1e-3, which the measured count moves by
(asserted), so the control discriminates. Float64 (the check refuses float32)."""

from __future__ import annotations

from cases import tree_flow

import mime.nodes.flow.fvm._node as flow_node
from mime.testing import two_builds_of

MOVE = 1e-3


def _plan_size(built) -> int:
    return built.graph.get_node("water.flow").plan_size()


def test_a_move_of_the_radii_keeps_the_trees_program():
    found = two_builds_of(tree_flow(), "vessel.radii", relative=MOVE, run=False)
    assert found.verdict == "quiet", found.lines


def test_control_with_the_plan_size_the_builds_own_count_the_same_move_changes_the_program(monkeypatch):
    monkeypatch.setattr(flow_node, "plan_rung", lambda count: count)
    experiment = tree_flow()
    found = two_builds_of(experiment, "vessel.radii", relative=MOVE, run=False)
    assert found.verdict == "shape", (found.verdict, found.lines)
