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

import copy
import os
import re
from collections import Counter
from pathlib import Path

import pytest
from hypothesis import HealthCheck, find, given, settings
from hypothesis import strategies as st

import mime.experiment._tree as tree_module
from views_cases import tutorial_a0, two_bodies, two_body_declarations
from mime.build import _testing
from mime.build._plan import Section
from mime.experiment import BuiltExperiment, Experiment, Mermaid
from mime.experiment._mermaid import POINTER, _escaped
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


def test_the_mermaid_is_a_string_and_a_notebook_shows_it_as_a_fenced_diagram_with_the_text_as_the_fallback(built):
    mermaid = built.to_mermaid()
    assert isinstance(mermaid, str) and isinstance(mermaid, Mermaid)
    assert mermaid == str(mermaid) and mermaid.startswith("flowchart LR\n") and mermaid.endswith("\n")
    bundle = mermaid._repr_mimebundle_()
    assert set(bundle) == {"text/markdown", "text/plain"}
    assert bundle["text/plain"] == str(mermaid)
    assert bundle["text/markdown"] == f"```mermaid\n{mermaid}```\n\n*{POINTER}*\n"
    assert POINTER.count("https://mermaid.live") == 1
    assert "<script" not in bundle["text/markdown"], "no script is loaded to draw it"
    assert repr(mermaid) == repr(str(mermaid))


def test_no_text_in_a_node_name_can_end_the_fence(built):
    """The fence ends at three backticks; MADDENING escapes a backtick in a name, so the diagram's own text cannot."""
    assert "`" not in built.to_mermaid()


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
