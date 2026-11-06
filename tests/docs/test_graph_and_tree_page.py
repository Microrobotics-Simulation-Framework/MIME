"""``docs/guide/graph_and_tree.md`` shows tutorial A0's tree and graph as the experiment prints them, and the page's blocks are
fragments (the check of executable docs runs only scripts), so this holds each to the experiment's own text: one changed
character in either fails it.

**What this bounds.** That the two blocks of the page are what ``exp.tree().text()`` and ``built.to_mermaid()`` give for
tutorial A0 as ``examples/tutorial_a0.py`` declares it. It does not bound the notebook table of the page, which records a
measurement made once, with the browser and the versions it names.

**Control.** The comparison is a function of the page's text, and a page with one character changed in a block fails it."""

from __future__ import annotations

import warnings

import pytest
from views_cases import ROOT, tutorial_a0

from mime.docs._executable import parse

PAGE = ROOT / "docs" / "guide" / "graph_and_tree.md"


def blocks_of(page: str) -> dict[str, str]:
    """The page's fenced blocks that hold the experiment's views, by their language: ``text`` is the tree and ``mermaid`` the
    graph."""
    return {block.language: block.text for block in parse(page).blocks if block.language in ("text", "mermaid")}


@pytest.fixture(scope="module")
def views() -> tuple[str, str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        experiment = tutorial_a0()
        return experiment.tree().text(), str(experiment.build().to_mermaid())


def differences(page: str, views: tuple[str, str]) -> list[str]:
    held = blocks_of(page)
    tree, graph = views
    return [name for name, text, shown in (("tree", tree, held.get("text")), ("graph", graph, held.get("mermaid")))
            if shown is None or shown.rstrip("\n") != text.rstrip("\n")]


def test_the_page_shows_tutorial_a0s_tree_and_graph_as_the_experiment_prints_them(views):
    page = PAGE.read_text(encoding="utf-8")
    assert sorted(blocks_of(page)) == ["mermaid", "text"]
    assert differences(page, views) == []


@pytest.mark.parametrize("which", ["tree", "graph"])
def test_control_a_block_with_one_character_changed_fails_the_comparison(views, which):
    page = PAGE.read_text(encoding="utf-8")
    old = "drag:umr  drag 0.4.0" if which == "tree" else 'n0["controller<br/>arm_tracking 0.1.0'
    assert old in page
    assert differences(page.replace(old, old.replace("0.", "1.", 1)), views) == [which]
