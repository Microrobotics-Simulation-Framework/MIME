"""The two views of an experiment: its compiled graph as Mermaid (``built.to_mermaid()``) and its topology as a tree
(``exp.tree()``).

**What this bounds.** That the Mermaid draws every node, edge, coupling group and external input of the graph the build
made, each node with the registered entry that built it; that the tree lists every thing, law, observation and the
controller the experiment declares, with what each binds; and that neither depends on the order a script made its
calls in. It says nothing of how a front end draws the Mermaid (``docs/guide/graph_and_tree.md`` reports that, measured).

**The oracles are not the products.** The Mermaid is read back by a parser written here from MADDENING's documented
shape (``n<i>["name<br/>..."]``, ``a -->|"f→g"| b``, ``subgraph``, ``x<i>[/"external: f"/]``) and compared with
MADDENING's own description of the graph (``built.topology()``, which reads ``GraphManager.to_dict``). The tree is
compared with phase 1's resolution of the same declarations (``built.plans``: its things, its entries with the names
each is bound to, its controller), which reaches them by another route than the tree does.

**A law bound to the wrong thing.** Script A, tutorial A0 and the full toy each declare one body, one medium and one
field, so a tree that printed the first body for every law would look right on them. The two-body toy (``views_cases``)
has two bodies, two media and two fields, bound across, and a chain, parts and fields declared out of alphabetical order,
so such a tree differs from the right one, and so does one that sorted the chain or left the parts in the order given.

**Controls.** Each check is a function of its inputs, and each control hands it the planted fault: every edge dropped in
turn, every node, every group, every external input, every law of the tree omitted in turn or bound to the wrong name.
Each fails on every run, since the faults are enumerated and not drawn. The order-dependence control restores the
dependence (the build skipping its canonical order, the tree not sorted) and shows that the property fails there."""

from __future__ import annotations

import base64
import copy
import hashlib
import html
import os
import re
import sys
import types
import warnings
from collections import Counter
from pathlib import Path

import pytest
from hypothesis import HealthCheck, find, given, settings
from hypothesis import strategies as st

import mime.experiment._mermaid_page as page_module
import mime.experiment._tree as tree_module
from views_cases import tutorial_a0, two_bodies, two_body_declarations
from mime.build import _testing
from mime.build._plan import Section
from mime.experiment import BuiltExperiment, Experiment, Mermaid
from mime.experiment._mermaid import POINTER, _escaped
from mime.experiment._mermaid_page import (CREDIT, MERMAID_CDN, MERMAID_SRI, MERMAID_VERSION, NOTE, OUTPUT_WARNING_BYTES,
                                           library_bytes)
from mime.testing import declaration_orders, declare_in_order
from mime.testing.toys import ToyDrag, full_declarations, toy_experiment

GOLDEN = Path(__file__).parent / "golden"
# Hypothesis's default example count, kept: each example builds a fresh graph.
PROPERTY = settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def script_a() -> Experiment:
    from script_a import script_a as declared

    return declared()


def full_toy() -> Experiment:
    return declare_in_order(lambda: toy_experiment("full_toy"), full_declarations())


MAKERS = {"script A": script_a, "tutorial A0": tutorial_a0, "full toy": full_toy, "two bodies": two_bodies}


@pytest.fixture(scope="module", params=list(MAKERS))
def experiment(request) -> Experiment:
    return MAKERS[request.param]()


@pytest.fixture(scope="module")
def built(experiment) -> BuiltExperiment:
    return experiment.build()


# --- the Mermaid read back -------------------------------------------------------------------------------------------


UNESCAPES = (("#quot;", '"'), ("#lt;", "<"), ("#gt;", ">"), ("#38;", "&"), ("#96;", "`"), ("#35;", "#"))
NODE = re.compile(r'^\s*(n\d+)\["(.*)"\]$')
EDGE = re.compile(r'^\s*(n\d+) (-->|-\.->)\|"(.*)"\| (n\d+)$')
EXTERNAL = re.compile(r'^\s*(x\d+)\[/"external: (.*)"/\]$')
FEED = re.compile(r"^\s*(x\d+) -\.-> (n\d+)$")
GROUP = re.compile(r'^\s*subgraph (g\d+)\["coupling group (.*)"\]$')


def unescaped(text: str) -> str:
    for code, character in UNESCAPES:
        text = text.replace(code, character)
    return text


def drawn(mermaid: str) -> dict[str, object]:
    """What a Mermaid text draws, read back: the nodes (each id's name and its label's other lines), the edges as
    ``(source, source field, target, target field, additive)``, the groups as the set of the nodes each holds, and the
    external inputs as ``(node, field)``. Written from MADDENING's documented shape, not from the code that wrote it."""
    names: dict[str, str] = {}
    labels: dict[str, list[str]] = {}
    in_group: dict[str, set[str]] = {}
    edges: list[tuple[str, str, str, str, bool]] = []
    externals: dict[str, str] = {}
    feeds: list[tuple[str, str]] = []
    current: str | None = None
    for line in mermaid.splitlines()[1:]:
        if (found := GROUP.match(line)) is not None:
            current = found.group(2)
            in_group[current] = set()
        elif line.strip() == "end":
            current = None
        elif (found := NODE.match(line)) is not None:
            parts = found.group(2).split("<br/>")
            names[found.group(1)] = unescaped(parts[0])
            labels[found.group(1)] = [unescaped(part) for part in parts[1:]]
            if current is not None:
                in_group[current].add(unescaped(parts[0]))
        elif (found := EDGE.match(line)) is not None:
            label = unescaped(found.group(3))
            additive = label.endswith(" (additive)")
            fields = label.removesuffix(" (additive)").split("→")
            edges.append((found.group(1), fields[0], found.group(4), fields[1], additive))
        elif (found := EXTERNAL.match(line)) is not None:
            externals[found.group(1)] = unescaped(found.group(2))
        elif (found := FEED.match(line)) is not None:
            feeds.append((found.group(1), found.group(2)))
        else:
            raise AssertionError(f"a line of the Mermaid this parser does not know: {line!r}")
    return {
        "nodes": names, "labels": labels,
        "edges": Counter((names.get(a, a), f, names.get(b, b), g, add) for a, f, b, g, add in edges),
        "groups": Counter(frozenset(members) for members in in_group.values()),
        "externals": Counter((names.get(node, node), externals[x]) for x, node in feeds)}


def expected(built: BuiltExperiment) -> dict[str, object]:
    """MADDENING's description of the compiled graph, in the shape ``drawn`` returns."""
    topology = built.topology()
    return {
        "nodes": sorted(node["name"] for node in topology["nodes"]),
        "edges": Counter((e["source_node"], e["source_field"], e["target_node"], e["target_field"], e["additive"])
                         for e in topology["edges"]),
        "groups": Counter(frozenset(group["nodes"]) for group in topology["coupling_groups"]),
        "externals": Counter((e["node"], e["field"]) for e in topology["external_inputs"])}


def differences(mermaid: str, built: BuiltExperiment) -> list[str]:
    """What the Mermaid does not draw of the graph, and what it draws that the graph does not hold; empty when it draws
    exactly the graph."""
    found, held = drawn(mermaid), expected(built)
    problems: list[str] = []
    if sorted(found["nodes"].values()) != held["nodes"]:  # type: ignore[union-attr]
        problems.append(f"nodes: drawn {sorted(found['nodes'].values())}, held {held['nodes']}")  # type: ignore[union-attr]
    for kind in ("edges", "groups", "externals"):
        lacking, extra = held[kind] - found[kind], found[kind] - held[kind]  # type: ignore[operator]
        problems += [f"{kind}: not drawn: {item}" for item in lacking]  # type: ignore[union-attr]
        problems += [f"{kind}: drawn but not in the graph: {item}" for item in extra]  # type: ignore[union-attr]
    return problems


def entry_problems(mermaid: str, built: BuiltExperiment) -> list[str]:
    """Each node's label must carry the registered entry that built it: the entry's name and version, as the registry
    holds them (not read from the build report the label was made from). A node no entry built has MADDENING's two
    lines."""
    found = drawn(mermaid)
    registered = {node: plan.registered.cls.label() for plan in built.plans.entries for node in plan.node_names()}
    problems: list[str] = []
    for ident, name in found["nodes"].items():  # type: ignore[union-attr]
        label = found["labels"][ident]  # type: ignore[index]
        if name in registered:
            if len(label) != 2 or not label[0].startswith(registered[name]):
                problems.append(f"{name}: label {label} does not start with the entry {registered[name]!r}")
        elif len(label) != 1:
            problems.append(f"{name}: no entry built it and the label is {label}")
    return problems


def test_the_mermaid_draws_every_node_edge_group_and_external_input_of_the_graph_and_nothing_else(built):
    mermaid = built.to_mermaid()
    assert differences(mermaid, built) == []
    held = expected(built)
    assert held["edges"] and held["nodes"], "the check judges a graph that has parts"


def test_every_node_is_labelled_with_the_registered_entry_that_built_it(built):
    assert entry_problems(built.to_mermaid(), built) == []
    assert any(built.plans.entries), "the check judges a graph that has entries"


def test_a_graph_with_a_coupling_group_is_among_those_judged():
    """The checks above judge groups only on a graph that has one: the full toy has a coupling group (script A and tutorial
    A0 have none, and say so)."""
    assert expected(full_toy().build())["groups"]


def lines_of(mermaid: str) -> list[str]:
    return mermaid.splitlines()


def without_line(mermaid: str, index: int) -> str:
    lines = lines_of(mermaid)
    return "\n".join(lines[:index] + lines[index + 1:]) + "\n"


def test_control_every_edge_dropped_in_turn_fails_the_check(built):
    mermaid = str(built.to_mermaid())
    edges = [i for i, line in enumerate(lines_of(mermaid)) if EDGE.match(line)]
    assert edges
    for index in edges:
        assert differences(without_line(mermaid, index), built), f"dropping line {index} was not found"


def test_control_every_node_dropped_in_turn_fails_the_check(built):
    mermaid = str(built.to_mermaid())
    nodes = [i for i, line in enumerate(lines_of(mermaid)) if NODE.match(line)]
    assert nodes
    for index in nodes:
        assert differences(without_line(mermaid, index), built), f"dropping line {index} was not found"


def test_control_the_group_dropped_fails_the_check():
    toy = full_toy().build()
    mermaid = str(toy.to_mermaid())
    lines = lines_of(mermaid)
    (start,) = [i for i, line in enumerate(lines) if GROUP.match(line)]
    flat = lines[:start] + [line for line in lines[start + 1:] if line.strip() != "end"]
    problems = differences("\n".join(flat) + "\n", toy)
    assert any(problem.startswith("groups: not drawn") for problem in problems), problems


def test_control_an_external_input_dropped_fails_the_check(built):
    """A graph the controller runs as a callback has external inputs; the graph of script A has none, and says so."""
    mermaid = str(built.to_mermaid())
    inputs = [i for i, line in enumerate(lines_of(mermaid)) if FEED.match(line)]
    for index in inputs:
        assert differences(without_line(mermaid, index), built)
    if not expected(built)["externals"]:
        assert not inputs


def test_control_a_callback_controllers_external_inputs_are_drawn_and_dropping_one_fails_the_check():
    built = tutorial_a0_callback().build()
    mermaid = str(built.to_mermaid())
    assert differences(mermaid, built) == []
    held = expected(built)["externals"]
    assert held, "the callback mode feeds the arm's joint positions from outside the graph"
    inputs = [i for i, line in enumerate(lines_of(mermaid)) if FEED.match(line)]
    assert len(inputs) == sum(held.values())  # type: ignore[union-attr]
    for index in inputs:
        assert differences(without_line(mermaid, index), built)


def tutorial_a0_callback() -> Experiment:
    """Tutorial A0 with its controller run as a callback on the host (``mode="callback"``), whose writes are external
    inputs of the graph."""
    experiment = tutorial_a0()
    controller = [d for d in experiment._declarations if type(d).__name__ == "ControllerDeclaration"]  # pyright: ignore[reportPrivateUsage]
    assert len(controller) == 1
    from dataclasses import replace

    index = experiment._declarations.index(controller[0])  # pyright: ignore[reportPrivateUsage]
    experiment._declarations[index] = replace(controller[0], mode="callback")  # pyright: ignore[reportPrivateUsage]
    return experiment


def test_control_madden_ings_own_mermaid_has_no_entry_labels_so_the_label_check_can_fail(built):
    assert entry_problems(str(built.graph.to_mermaid()), built), "MADDENING's text has no MIME entry in its labels"


def test_the_mermaid_is_maddenings_with_the_entry_added_and_no_other_line_changed(built):
    ours = lines_of(built.to_mermaid())
    theirs = lines_of(built.graph.to_mermaid())
    assert len(ours) == len(theirs)
    changed = [(a, b) for a, b in zip(ours, theirs, strict=True) if a != b]
    assert all(NODE.match(a) and NODE.match(b) for a, b in changed)
    assert all(a.replace(a.split("<br/>")[1] + "<br/>", "", 1) == b for a, b in changed)
    assert len(changed) == len(expected(built)["nodes"])  # type: ignore[arg-type]


def test_the_direction_is_maddenings_and_a_wrong_one_is_refused(built):
    assert lines_of(built.to_mermaid(direction="TB"))[0] == "flowchart TB"
    with pytest.raises(ValueError, match="direction"):
        built.to_mermaid(direction="up")


# --- the escaping is MADDENING's -------------------------------------------------------------------------------------


@given(st.text())
def test_the_escaping_is_the_one_maddening_uses_for_any_text(text):
    from maddening.core.inspection import _mermaid_text

    assert _escaped(text) == _mermaid_text(text)


def test_control_an_escaping_that_forgets_the_hash_differs_from_maddenings():
    from maddening.core.inspection import _mermaid_text

    assert _escaped("a#b") == _mermaid_text("a#b") == "a#35;b"
    assert "a#b".replace("#", "#") != _mermaid_text("a#b")


# --- the notebook's display ------------------------------------------------------------------------------------------
#
# What this bounds: the bundle a notebook gets is the same page in every front end (an ``<iframe srcdoc>`` that unescapes
# to the page, and the Mermaid text as ``text/plain``); the page holds the graph's text exactly once, the library either
# inlined (the shipped bytes, held to their pinned sha256) or from jsDelivr (held to its sha384, which is the sha384 of
# those same bytes), and loads nothing else; the same experiment gives the same page byte for byte. What it does not: that
# a front end draws it (the page was drawn, panned and zoomed by hand in headless Chrome, which the PR records, and the
# PR gives the cell the maintainer runs in Colab), and that the pinned hash is the file jsDelivr serves today (measured once, when the
# pin was written, with the command beside the constants).


MERMAID_SHA256 = "581ed7d74bd9048d0e3a91363927d72ef22942d7722546b27f7cc29e35390eb8"
SCRIPT_TAG = re.compile(r"<script\b[^>]*>")
URL = re.compile(r"https?://[^\s\"'<>)]+")
IFRAME_SRCDOC = re.compile(r'<iframe srcdoc="([^"]*)"')


@pytest.fixture
def shown(monkeypatch):
    """A stand-in for IPython's ``display``: the calls it received."""
    calls: list[tuple[object, dict]] = []
    package, module = types.ModuleType("IPython"), types.ModuleType("IPython.display")
    module.display = lambda obj, **kwargs: calls.append((obj, kwargs))  # type: ignore[attr-defined]
    package.display = module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "IPython", package)
    monkeypatch.setitem(sys.modules, "IPython.display", module)
    return calls


def page_of_frame(fragment: str) -> str:
    """The page a notebook's ``<iframe srcdoc>`` holds, unescaped as a browser reads the attribute."""
    (srcdoc,) = IFRAME_SRCDOC.findall(fragment)
    return html.unescape(srcdoc)


def element_text(page: str) -> str:
    (element,) = re.findall(r'<pre class="mermaid" id="source">(.*?)</pre>', page, re.DOTALL)
    return html.unescape(element)


def urls_outside_the_credit(page: str) -> list[str]:
    held = re.sub(r'src="data:[^"]+"', "", page).replace(CREDIT, "")
    return URL.findall(held)


def test_the_mermaid_is_a_string_and_a_notebook_shows_it_as_a_page_in_a_frame_with_the_text_as_the_fallback(built):
    mermaid = built.to_mermaid()
    assert isinstance(mermaid, str) and isinstance(mermaid, Mermaid)
    assert mermaid == str(mermaid) and mermaid.startswith("flowchart LR\n") and mermaid.endswith("\n")
    bundle = mermaid._repr_mimebundle_()
    assert set(bundle) == {"text/html", "text/plain"}
    assert bundle["text/plain"] == str(mermaid)
    fragment = bundle["text/html"]
    assert fragment == mermaid.frame() and len(IFRAME_SRCDOC.findall(fragment)) == 1
    assert page_of_frame(fragment) == mermaid.page(cdn=True), "a notebook's output loads the library from jsDelivr"
    assert 'sandbox="allow-scripts"' in fragment and "allowfullscreen" in fragment
    assert NOTE in html.unescape(fragment), "what shows where the frame is blank"
    assert repr(mermaid) == repr(str(mermaid))


def test_no_text_in_a_node_name_can_end_the_fence(built):
    """The offline text is a Markdown code block that ends at three backticks; MADDENING escapes a backtick in a name,
    so the diagram's own text cannot."""
    assert "`" not in built.to_mermaid()


def test_the_bundle_is_the_same_in_colab_as_anywhere_else(built, monkeypatch):
    mermaid = built.to_mermaid()
    outside = mermaid._repr_mimebundle_()
    monkeypatch.setitem(sys.modules, "google.colab", types.ModuleType("google.colab"))
    assert mermaid._repr_mimebundle_() == outside


def test_the_page_holds_the_graphs_text_exactly_once_escaped_for_the_frame(built):
    mermaid = built.to_mermaid()
    page = mermaid.page()
    assert element_text(page) == str(mermaid)
    assert page.count(html.escape(str(mermaid), quote=False)) == 1
    fragment = mermaid.frame()
    srcdoc = IFRAME_SRCDOC.findall(fragment)[0]
    assert '"' not in srcdoc and "<" not in srcdoc, "an attribute holds no raw quote or tag"
    assert html.unescape(srcdoc).count(html.escape(str(mermaid), quote=False)) == 1


def test_control_a_changed_character_of_the_graph_changes_the_page_and_the_text_it_holds(built):
    mermaid = built.to_mermaid()
    changed = Mermaid(mermaid.replace("flowchart LR", "flowchart TB", 1))
    assert changed != mermaid and changed.page() != mermaid.page() and changed.page(cdn=True) != mermaid.page(cdn=True)
    assert element_text(changed.page()) == str(changed) != str(mermaid)
    assert html.escape(str(mermaid), quote=False) not in changed.page()


def test_the_shipped_library_is_the_pinned_file_and_its_sha384_is_the_one_jsdelivrs_copy_is_held_to():
    data = library_bytes()
    assert len(data) == 3_572_661 and hashlib.sha256(data).hexdigest() == MERMAID_SHA256 == page_module.MERMAID_SHA256
    assert MERMAID_SRI == "sha384-" + base64.b64encode(hashlib.sha384(data).digest()).decode()


def test_the_page_sizes_the_docs_state_are_the_ones_measured(built):
    """``docs/guide/graph_and_tree.md`` and the CLI's README give the page's sizes: 12 kB from jsDelivr, 4.8 MB inlined, the
    library 3.6 MB; a test holds them, and the number allowlist's reasons name it."""
    mermaid = built.to_mermaid()
    assert 3_550_000 < len(library_bytes()) < 3_650_000
    assert 4_750_000 < len(mermaid.page()) < 4_850_000 and 4_750_000 < len(mermaid.frame(inline=True)) < 4_950_000
    assert 11_000 < len(mermaid.page(cdn=True)) < 14_000
    assert str(page_module.FRAME_HEIGHT) == "560" and "const READABLE = 0.75," in page_module.SCRIPT


def test_control_a_changed_byte_in_the_shipped_copy_fails_the_hash_test(monkeypatch):
    data = bytearray((Path(page_module.__file__).parent / "_mermaid_lib" / page_module.MERMAID_FILE).read_bytes())
    for index in (0, len(data) // 2, len(data) - 1):
        planted = bytearray(data)
        planted[index] ^= 1

        class Held:
            def joinpath(self, *names):
                return self

            def read_bytes(self, _planted=bytes(planted)):
                return _planted

        monkeypatch.setattr(page_module, "files", lambda package: Held())
        with pytest.raises(ValueError, match="not the library MIME pins"):
            library_bytes()
        assert hashlib.sha256(bytes(planted)).hexdigest() != MERMAID_SHA256


def test_the_inlined_page_carries_the_shipped_bytes_and_loads_nothing(built):
    page = built.to_mermaid().page()
    (encoded,) = re.findall(r'<script src="data:text/javascript;base64,([A-Za-z0-9+/=]+)"></script>', page)
    assert base64.b64decode(encoded) == library_bytes()
    assert urls_outside_the_credit(page) == []
    assert MERMAID_CDN not in page and "integrity" not in page
    assert len(SCRIPT_TAG.findall(page)) == 2, "the library, and the page's own script"


def test_the_jsdelivr_page_names_the_library_with_its_integrity_and_loads_nothing_else(built):
    page = built.to_mermaid().page(cdn=True)
    tags = SCRIPT_TAG.findall(page)
    assert tags == [f'<script src="{MERMAID_CDN}" integrity="{MERMAID_SRI}" crossorigin="anonymous">', "<script>"]
    assert re.fullmatch(r"https://cdn\.jsdelivr\.net/npm/mermaid@\d+\.\d+\.\d+/dist/mermaid\.min\.js", MERMAID_CDN)
    assert MERMAID_CDN.count(MERMAID_VERSION) == 1 and re.fullmatch(r"sha384-[A-Za-z0-9+/]{64}", MERMAID_SRI)
    assert urls_outside_the_credit(page) == [MERMAID_CDN] and "text/javascript;base64" not in page
    assert len(page) < 100_000


def test_control_a_url_planted_outside_the_credit_is_found(built):
    page = built.to_mermaid().page().replace("<body>", '<body><img src="https://example.org/x.png">')
    assert urls_outside_the_credit(page) == ["https://example.org/x.png"]


def test_the_page_draws_with_a_fixed_id_and_holds_nothing_that_varies(built):
    page = built.to_mermaid().page(cdn=True)
    assert 'mermaid.render("mime-graph"' in page and "deterministicIds: true" in page
    assert not re.search(r"Math\.random|Date\.now|new Date|crypto\.", page_module.SCRIPT)


def test_the_same_experiment_gives_the_same_page_byte_for_byte_and_another_experiment_another_page(experiment, built):
    first, second = built.to_mermaid(), experiment.build().to_mermaid()
    assert first.page() == second.page() and first.page(cdn=True) == second.page(cdn=True)
    assert first.frame() == second.frame() and first.frame(inline=True) == second.frame(inline=True)
    others = [make().build().to_mermaid() for make in MAKERS.values()]
    assert len({other.page(cdn=True) for other in others}) == len(MAKERS), "experiments that differ give different pages"


def test_save_writes_the_page_and_returns_the_path(built, tmp_path):
    mermaid = built.to_mermaid()
    written = mermaid.save(tmp_path / "graph.html")
    assert written == tmp_path / "graph.html" and written.read_text(encoding="utf-8") == mermaid.page()
    assert mermaid.save(tmp_path / "cdn.html", cdn=True).read_text(encoding="utf-8") == mermaid.page(cdn=True)


def test_the_page_escapes_what_would_end_the_element_or_open_a_tag():
    text = Mermaid('flowchart LR\n    n0["a</pre><script>x</script> & b"]\n')
    page = text.page(cdn=True)
    assert len(SCRIPT_TAG.findall(page)) == 2 and "<script>x" not in page
    assert element_text(page) == str(text)


def test_display_shows_what_returning_the_diagram_from_a_cell_shows(built, shown):
    mermaid = built.to_mermaid()
    mermaid.display()
    ((bundle, kwargs),) = shown
    assert kwargs == {"raw": True} and bundle == mermaid._repr_mimebundle_()


def test_display_inline_carries_the_library_and_warns_that_the_output_is_large(built, shown):
    mermaid = built.to_mermaid()
    with pytest.warns(UserWarning, match="MB as a notebook output"):
        mermaid.display(inline=True)
    ((bundle, _),) = shown
    page = page_of_frame(bundle["text/html"])
    assert page == mermaid.page() and MERMAID_CDN not in page
    assert len(bundle["text/html"]) > OUTPUT_WARNING_BYTES, "the warning is for an output over the rate limit"
    shown.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mermaid.display()
    assert len(shown[0][0]["text/html"]) < OUTPUT_WARNING_BYTES, "the control: the default output is far below it"


def test_offline_the_notebook_gets_the_text_and_the_pointer_and_no_script_in_colab_or_elsewhere(built, shown, monkeypatch):
    mermaid = built.to_mermaid()
    for colab in (False, True):
        if colab:
            monkeypatch.setitem(sys.modules, "google.colab", types.ModuleType("google.colab"))
        shown.clear()
        mermaid.display(offline=True)
        ((bundle, kwargs),) = shown
        assert kwargs == {"raw": True} and set(bundle) == {"text/markdown", "text/plain"}
        assert bundle["text/markdown"] == f"```text\n{mermaid}```\n\n*{POINTER}*\n"
        assert "<script" not in bundle["text/markdown"] and MERMAID_CDN not in bundle["text/markdown"]
        assert "```mermaid" not in bundle["text/markdown"], "offline shows the text, not a diagram to be drawn"


def test_display_without_ipython_says_what_to_do_instead(built, monkeypatch):
    monkeypatch.setitem(sys.modules, "IPython", None)
    monkeypatch.setitem(sys.modules, "IPython.display", None)
    with pytest.raises(RuntimeError, match="print"):
        built.to_mermaid().display()


# --- the tree --------------------------------------------------------------------------------------------------------


def rows(record: dict, key: str) -> dict[str, dict]:
    return {row.get("id", row.get("name")): row for row in record[key]}


def tree_problems(record: dict, built: BuiltExperiment) -> list[str]:
    """What the tree lacks of what phase 1 resolved, and what it lists that phase 1 did not."""
    plans = built.plans
    problems: list[str] = []
    for key, kind in (("structures", "structure"), ("media", "medium"), ("bodies", "body"), ("apparatus", "apparatus")):
        held = sorted(name for name, thing in plans.things.items() if thing.value == kind)
        listed = sorted(row["name"] for row in record[key])
        if held != listed:
            problems.append(f"{key}: the plan has {held}, the tree {listed}")
    for key, section in (("laws", Section.INTERACTION), ("observations", Section.OBSERVATION)):
        by_id = {plan.owner: plan for plan in plans.entries if plan.section is section}
        listed = rows(record, key)
        if sorted(by_id) != sorted(listed):
            problems.append(f"{key}: the plan has {sorted(by_id)}, the tree {sorted(listed)}")
        for ident, plan in by_id.items():
            row = listed.get(ident)
            if row is None:
                continue
            if (row["entry"]["id"], row["entry"]["version"]) != (plan.registered.name, plan.registered.version):
                problems.append(f"{ident}: entry {row['entry']} is not {plan.registered.name} {plan.registered.version}")
            binds = dict(row["binds" if key == "laws" else "reads"])
            for role, name in plan.bindings.items():
                if binds.get(role) != name:
                    problems.append(f"{ident}: {role} is bound to {binds.get(role)!r} in the tree, {name!r} in the plan")
    controller = plans.controller
    if (controller is None) != (record["controller"] is None):
        problems.append(f"controller: the plan has {controller and controller.owner}, the tree {record['controller']}")
    elif controller is not None:
        row = record["controller"]
        if (row["entry"]["id"], row["entry"]["version"]) != (controller.registered.name, controller.registered.version):
            problems.append(f"controller: entry {row['entry']}")
        if sorted(label for label, _ in row["reads"]) != sorted(read.label for read in controller.reads):
            problems.append("controller: the reads differ")
        if dict(row["writes"]) != {write.label: write.path for write in controller.writes}:
            problems.append("controller: the writes differ")
        if dict(row["reads"]) != {read.label: read.path for read in controller.reads}:
            problems.append("controller: the reads' paths differ")
    return problems


def test_the_tree_holds_every_declaration_the_plan_resolved(experiment, built):
    record = experiment.tree().record()
    assert tree_problems(record, built) == []
    assert record["laws"] and record["bodies"], "the check judges an experiment that declares a law and a body"


def test_the_tree_text_holds_every_name_and_entry_of_the_record(experiment):
    tree = experiment.tree()
    record = tree.record()
    text = tree.text()
    for key in ("structures", "media", "bodies", "apparatus"):
        for row in record[key]:
            assert f"\n    {row['name']}" in text
    for key in ("laws", "observations"):
        for row in record[key]:
            assert f"    {row['id']}  {row['entry']['id']} {row['entry']['version']}" in text
    assert text.splitlines()[0].startswith(f"experiment {experiment.name}  (profile ")
    assert text.endswith("\n") and str(tree) == text and repr(tree) == text


def test_control_every_law_omitted_in_turn_fails_the_check(experiment, built):
    record = experiment.tree().record()
    for key in ("laws", "observations"):
        for index in range(len(record[key])):
            planted = copy.deepcopy(record)
            del planted[key][index]
            assert tree_problems(planted, built), f"omitting {key}[{index}] was not found"


def test_control_every_law_bound_to_the_wrong_name_fails_the_check(experiment, built):
    record = experiment.tree().record()
    for key, binds in (("laws", "binds"), ("observations", "reads")):
        for index, row in enumerate(record[key]):
            for position in range(len(row[binds])):
                planted = copy.deepcopy(record)
                planted[key][index][binds][position][1] += "_other"
                assert tree_problems(planted, built), f"{key}[{index}] {binds}[{position}] wrong was not found"


def test_control_a_thing_a_controller_or_an_entry_version_changed_fails_the_check(experiment, built):
    record = experiment.tree().record()
    for key in ("structures", "media", "bodies", "apparatus"):
        if record[key]:
            planted = copy.deepcopy(record)
            planted[key].pop()
            assert tree_problems(planted, built), key
    planted = copy.deepcopy(record)
    planted["laws"][0]["entry"]["version"] = "9.9.9"
    assert tree_problems(planted, built)
    planted = copy.deepcopy(record)
    planted["controller"] = None if record["controller"] is not None else {"entry": {"id": "x", "version": "1"},
                                                                          "mode": "graph", "reads": [], "writes": []}
    assert tree_problems(planted, built)


def test_the_tree_of_tutorial_a0_has_the_controller_with_its_reads_and_writes():
    record = tutorial_a0().tree().record()
    controller = record["controller"]
    assert controller["entry"] == {"id": "arm_tracking", "version": "0.1.0"}
    assert controller["mode"] == "graph"
    assert controller["reads"] == [["body_pose", "umr.pose"], ["body_twist", "umr.twist"],
                                   ["joint_positions", "rpm.arm.joint_positions"]]
    assert controller["writes"] == [["joint_positions", "rpm.arm.joint_positions"]]
    (medium,) = record["media"]
    assert medium == {"name": "water", "fills": "tube.lumen", "flow": None}
    (apparatus,) = record["apparatus"]
    assert [segment["name"] for segment in apparatus["chain"]] == ["arm", "motor"], "the chain keeps its own order"
    assert apparatus["parts"][0]["mount"] == "motor.rotor" and apparatus["parts"][0]["magnet"]


def test_a_tree_can_be_hashed_and_equal_trees_hash_alike(experiment):
    """A frozen record over a dict has no hash of its own; the tree's is its text's."""
    assert hash(experiment.tree()) == hash(experiment.tree()) and experiment.tree() == experiment.tree()
    assert len({experiment.tree(), experiment.tree()}) == 1


def test_the_tree_needs_no_build_and_shows_an_experiment_the_build_would_refuse():
    experiment = toy_experiment("refused")
    experiment.attach(ToyDrag.Quadratic(), body="nobody")
    record = experiment.tree().record()
    assert record["laws"][0]["binds"] == [["body", "nobody"]]
    assert record["bodies"] == []


def test_the_two_body_tree_binds_each_law_and_observation_to_its_own_body_medium_and_field():
    """With one body, a tree that printed the first body for every ``body`` role would look right; with two, each law
    names the body, medium and field it was attached to (``zed`` to ``bath`` and ``push``, ``abe`` to ``oil`` and
    ``pull``). Written out here, not compared with the plan, so that a tree and a plan that were wrong the same way fail."""
    record = two_bodies().tree().record()
    assert {row["id"]: row["binds"] for row in record["laws"]} == {
        "toy.couple:abe": [["body", "abe"], ["medium", "oil"]],
        "toy.couple:zed": [["body", "zed"], ["medium", "bath"]],
        "toy.drag:zed": [["body", "zed"]],
        "toy.hold:abe": [["body", "abe"], ["field", "rig.pull"]],
        "toy.hold:zed": [["body", "zed"], ["field", "rig.push"]],
        "toy.spring:abe": [["body", "abe"]]}
    assert {row["id"]: row["reads"] for row in record["observations"]} == {
        "toy.recorder:abe": [["body", "abe"]], "toy.recorder:zed": [["body", "zed"]]}


def test_the_two_body_tree_keeps_a_chain_in_its_declared_order_and_sorts_parts_fields_and_the_things():
    """The chain is declared ``zeta`` then ``alpha`` (the second mounts on the first), the parts ``shell`` then ``cap``
    and ``yoke`` then ``cap``, the fields ``push`` then ``pull``, the bodies ``zed`` then ``abe``: the chain stays as
    declared, and the rest is sorted."""
    record = two_bodies().tree().record()
    (rig,) = record["apparatus"]
    assert [segment["name"] for segment in rig["chain"]] == ["zeta", "alpha"]
    assert [part["name"] for part in rig["parts"]] == ["cap", "yoke"]
    assert [each["name"] for each in rig["fields"]] == ["pull", "push"]
    assert [each["name"] for each in rig["commands"]] == ["alpha.velocity", "zeta.velocity"]
    assert [row["name"] for row in record["bodies"]] == ["abe", "zed"]
    assert [part["name"] for part in record["bodies"][1]["parts"]] == ["cap", "shell"]
    assert [row["name"] for row in record["media"]] == ["bath", "oil"]


def test_the_two_body_mermaid_wires_each_body_to_its_own_medium_and_field():
    mermaid = lines_of(two_bodies().build().to_mermaid())
    names = {found[1]: found[2].split("<br/>")[0] for line in mermaid if (found := NODE.match(line))}
    wired = {(names[found[1]], names[found[4]]) for line in mermaid if (found := EDGE.match(line))}
    right = {("zed", "bath.flow"), ("bath.flow", "zed"), ("abe", "oil.flow"), ("oil.flow", "abe"),
             ("rig.push", "toy.hold:zed"), ("toy.hold:zed", "zed"), ("rig.pull", "toy.hold:abe"), ("toy.hold:abe", "abe")}
    wrong = {("zed", "oil.flow"), ("oil.flow", "zed"), ("abe", "bath.flow"), ("bath.flow", "abe"),
             ("rig.push", "toy.hold:abe"), ("toy.hold:abe", "zed"), ("rig.pull", "toy.hold:zed"), ("toy.hold:zed", "abe")}
    assert right <= wired and not (wrong & wired)


# --- determinism -----------------------------------------------------------------------------------------------------


DECLARATIONS = full_declarations()


def views(order: list[int], declarations=DECLARATIONS, name: str = "full_toy") -> tuple[str, str, str]:
    experiment = declare_in_order(lambda: toy_experiment(name), declarations, order)
    return str(experiment.build().to_mermaid()), experiment.tree().text(), experiment.tree().json()


@pytest.fixture(scope="module")
def reference() -> tuple[str, str, str]:
    return views(list(range(len(DECLARATIONS))))


@PROPERTY
@given(order=declaration_orders(DECLARATIONS))
def test_any_order_of_the_calls_gives_the_same_mermaid_tree_and_data_byte_for_byte(order, reference):
    assert views(order) == reference


TWO_BODY_DECLARATIONS = two_body_declarations()


@pytest.fixture(scope="module")
def two_body_reference() -> tuple[str, str, str]:
    return views(list(range(len(TWO_BODY_DECLARATIONS))), TWO_BODY_DECLARATIONS, "two_bodies")


@PROPERTY
@given(order=declaration_orders(TWO_BODY_DECLARATIONS))
def test_any_order_of_the_calls_of_the_two_body_toy_gives_the_same_mermaid_tree_and_data_byte_for_byte(
        order, two_body_reference):
    """The toy with two bodies, two media and two fields, whose laws would each be bound to the other's thing by an order
    that mattered."""
    assert views(order, TWO_BODY_DECLARATIONS, "two_bodies") == two_body_reference


def test_control_with_the_canonical_order_skipped_hypothesis_finds_an_order_that_changes_the_mermaid(reference):
    with _testing.declaration_order():
        base = views(list(range(len(DECLARATIONS))))
        order = find(declaration_orders(DECLARATIONS), lambda found: views(found)[0] != base[0],
                     settings=settings(PROPERTY, database=None))
        assert views(order)[0] != base[0]


def test_control_with_the_tree_not_sorted_hypothesis_finds_an_order_that_changes_the_tree(reference, monkeypatch):
    monkeypatch.setattr(tree_module, "_sorted", lambda rows, key: rows)
    base = views(list(range(len(DECLARATIONS))))
    order = find(declaration_orders(DECLARATIONS), lambda found: views(found)[1] != base[1],
                 settings=settings(PROPERTY, database=None))
    assert views(order)[1] != base[1]


@pytest.mark.parametrize("name", ["script A", "tutorial A0"])
def test_the_declarations_reversed_give_the_same_mermaid_and_tree(name):
    """The extreme reorder of a real script: every declaration (things, values, laws, observations, the controller) in
    the reverse order. The graph is the canonical order's, so the Mermaid is the same."""
    experiment = MAKERS[name]()
    settings_, declarations = experiment._settings, list(experiment._declarations)  # pyright: ignore[reportPrivateUsage]
    reversed_ = Experiment._declared(settings_, declarations[::-1])  # pyright: ignore[reportPrivateUsage]
    assert reversed_._declarations != experiment._declarations  # pyright: ignore[reportPrivateUsage]
    assert reversed_.build().to_mermaid() == experiment.build().to_mermaid()
    assert reversed_.tree().text() == experiment.tree().text()
    assert reversed_.tree().json() == experiment.tree().json()


# --- the committed text ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["full toy", "script A", "two bodies"])
def test_the_mermaid_and_the_tree_match_their_committed_text(name):
    experiment = MAKERS[name]()
    stem = {"full toy": "full_toy", "script A": "script_a", "two bodies": "two_bodies"}[name]
    produced = {f"{stem}.mmd": str(experiment.build().to_mermaid()), f"{stem}.tree": experiment.tree().text()}
    if os.environ.get("MIME_REGENERATE_GOLDEN") == "1":
        for file, text in produced.items():
            (GOLDEN / file).write_text(text)
        pytest.fail("rewrote the golden Mermaid and tree; review the diff and rerun without MIME_REGENERATE_GOLDEN")
    for file, text in produced.items():
        assert text == (GOLDEN / file).read_text(), file
