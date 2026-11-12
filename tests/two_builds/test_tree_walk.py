"""The tube tree's wall under the two-builds check: its distance is made in the program from the tree's leaves (the
union's candidates chosen there, where its check fails), and only how many faces they may run at is held at build, a
rung above the count the build's values choose (``mime.nodes.flow.fvm.plan_rung``), as the cell counts are held. The
registry walk moves every float of ``tree_flow``'s document by a relative 1e-6 in turn, the radii and the period's
length among them, and is quiet (``test_two_builds_walk.py``); there a move of 1e-6 changes no chosen face, so the walk
alone does not show the rung at work. Here the radii are moved as the walk moves them (a study over them builds each
value, one group a value: a kind's vector setting is built per value, so it never shares one program), by the least of
:data:`MOVES` that changes the build's chosen count and keeps the box's cells: the two builds' programs are one.

**The control:** the plan's size made the build's own count (the rung the identity), the same pair's programs differ
in a shape. So the quiet pair is the rung keeping one program, not a form that cannot see the plan. Float64 (the check
refuses float32)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import test_two_builds_walk as walking
from cases import CASES

import mime.nodes.flow.fvm._node as flow_node
from mime.testing import graph_program
from mime.testing._two_builds import programs_differ

TREE = next(case for case in CASES if case.name == "tree_flow")
#: The relative moves of the radii tried, least first.
MOVES = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)


def _node(graph: Any) -> Any:
    return graph.get_node("water.flow")


def _cells(graph: Any) -> tuple[int, int, int]:
    c = _node(graph).constants()
    return int(c["cells_x"]), int(c["cells_y"]), int(c["cells_z"])


def _pair(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Any, Any, float]:
    """The base build and the build with its radii moved by the least of :data:`MOVES` that changes the chosen count
    and keeps the cells."""
    document, load = walking.loader(TREE, tmp_path)
    (radii,) = [path for path in walking.numbers(document) if "radii" in map(str, path)]
    base = load(document, "base")
    for move in MOVES:
        monkeypatch.setattr(walking, "RELATIVE", move)
        other = load(walking.moved(document, radii), "moved")
        if _cells(other) == _cells(base) and _node(other).chosen_count() != _node(base).chosen_count():
            print(f"\nthe radii moved by {move:g}: chosen {_node(base).chosen_count()} and "
                  f"{_node(other).chosen_count()}, the plan's size {_node(base).plan_size()} and {_node(other).plan_size()}")
            return base, other, move
    pytest.fail(f"no move of {MOVES} changes the chosen count with the cells kept: the pair cannot discriminate")


def test_a_move_of_the_radii_that_changes_the_chosen_count_keeps_one_program(monkeypatch: pytest.MonkeyPatch,
                                                                              tmp_path: Path):
    base, other, _ = _pair(monkeypatch, tmp_path)
    assert _node(base).plan_size() == _node(other).plan_size()
    assert not programs_differ(graph_program(base), graph_program(other))


def test_control_with_the_plan_size_the_builds_own_count_the_same_pair_differs(monkeypatch: pytest.MonkeyPatch,
                                                                               tmp_path: Path):
    monkeypatch.setattr(flow_node, "plan_rung", lambda count: count)
    base, other, _ = _pair(monkeypatch, tmp_path)
    assert programs_differ(graph_program(base), graph_program(other))
