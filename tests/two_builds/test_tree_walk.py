"""The tube tree's wall under the two-builds check: its distance is made in the program from the tree's leaves (the
union's candidates chosen there, where its check fails), and only how many faces they may run at is held at build, a
rung above the count the build's values choose (``mime.nodes.flow.fvm.plan_rung``), as the cell counts are held. The
registry walk moves every float of ``tree_flow``'s document by a relative 1e-6 in turn, the radii and the period's
length among them, and is quiet (``test_two_builds_walk.py``). A study cannot move a tree's radii (a vector of the kind:
``study.held-geometry``), so the walk's document moves are how they are moved.

**The control:** the same walk with the plan's size the build's own count (the rung the identity): a move of the radii
changes the count, so the two builds' programs differ in a shape, and the walk fires on the radii. So the quiet walk is
the rung keeping one program, not a walk that cannot see the plan. Float64 (the check refuses float32)."""

from __future__ import annotations

from pathlib import Path

import pytest
import test_two_builds_walk as walking
from cases import CASES

import mime.nodes.flow.fvm._node as flow_node

TREE = next(case for case in CASES if case.name == "tree_flow")


def test_control_with_the_plan_size_the_builds_own_count_the_walk_fires_on_the_radii(monkeypatch: pytest.MonkeyPatch,
                                                                                       tmp_path: Path):
    monkeypatch.setattr(flow_node, "plan_rung", lambda count: count)
    rows = walking.walk(TREE, tmp_path)
    fired = {"/".join(map(str, row["path"])) for row in rows if row.get("how")}
    print(f"\nwith the plan's size exact, the walk fired on: {sorted(fired)}")
    assert any("/radii" in path for path in fired), fired
