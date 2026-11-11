"""The build report's line for each solving node (design row #357): ``solve arithmetic of <node>: as today``, for every
node whose class a registered solve site names. A test that reads the line, so that dropping it fails (the line was
untested before this: the report builder returning nothing for it passed every test)."""

from __future__ import annotations

import pytest
from near_field_cases import NODE, bodies, joint_node, mrs, spheres

from mime.build import _construct
from mime.core.solves import sites_of
from mime.nodes.hydro.near_field import JointDragNode
from mime.nodes.actuation._drive import ConstantNode


def solving_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("solve arithmetic of ")]


def built_text() -> tuple[object, str]:
    built = bodies("solve_report", spheres(0.3e-3, 2), mrs(0.6e-3)).build()
    return built, built.report.text(details=True)


def test_the_build_report_has_one_line_for_each_node_whose_class_a_solve_site_names():
    """The joint node and each rigid body (a six-unknown solve) are named; a node that solves nothing is not."""
    built, text = built_text()
    graph = built.graph  # type: ignore[attr-defined]
    assert isinstance(joint_node(built), JointDragNode)
    assert sorted(solving_lines(text)) == [f"solve arithmetic of {name}: as today" for name in (NODE, "s0", "s1")]
    assert sites_of(type(graph.get_node(NODE))) and not sites_of(ConstantNode)


def test_control_the_report_without_the_line_fails_the_test_above(monkeypatch: pytest.MonkeyPatch):
    """The line is read from the report's text: with the builder's function returning nothing, no line is there."""
    monkeypatch.setattr(_construct, "solving_lines", lambda graph: ())
    _, text = built_text()
    assert solving_lines(text) == []
