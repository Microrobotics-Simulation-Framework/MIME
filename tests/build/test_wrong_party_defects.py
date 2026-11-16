"""A check of the built graph that faults a node is attributed to the code that has the flaw: the node's class, whatever
entry built it (G10).

Before, a shape, units or dtype that a node declared differently from the vocabulary's interface was refused as
``[internal] maddening.shape ... MIME defect: report it with this build report``, which sends a lab's author to report
the lab's own node to MIME. Now a node of a class outside MIME keeps its refusal's code, names the class and its package
as its owner, and has a ``do:`` that is the change the author makes, under a lab's entry or under MIME's own; a node of
one of MIME's classes is still MIME's defect under a lab's entry, which cannot edit it. All four combinations (a MIME
class or a lab's class, under a MIME entry or a lab's) are tested, so a rewording that blamed by the entry, or blamed
every package alike, fails one of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import pytest
from maddening.core.node import BoundaryInputSpec

from mime import Experiment
from mime.build import BuildRefused, InternalError, Refusal, RefusalClass
from mime.gradients import GradientStatus
from mime.registry import Delivery, Interaction, Read, RoleKind, register_interaction
from mime.build import _testing
from mime.testing.toys import LOAD, POSITION, ToySpring, ToySpringNode
from mime.testing.toys._entries import _respond
from mime.testing.toys._experiments import ADD_PROBE, toy_experiment
from mime.units import s
from mime.vocabulary import register_namespace

register_namespace("wrongparty", "0.0.1")


@register_interaction
@dataclass(frozen=True, kw_only=True)
class LabSpring(Interaction):
    """A lab's spring on the toy body, built of the toy's spring node: a lab's entry, its node MIME's class."""

    name = "wrongparty.spring"
    version = "0.0.1"
    differentiability = GradientStatus.EXACT
    builds = {"": (ToySpringNode,)}
    roles = {"body": RoleKind.BODY}
    reads = {"x": Read("body", POSITION.name)}
    delivers = {"load": Delivery("body", LOAD.name)}

    def build(self, ctx, b):
        node = b.add_response_node(ToySpringNode(ctx.node_name(), ctx.timestep))
        b.connect(b.read("x"), into=node["x"])
        b.connect(node["load"], into=b.target("load"))


OWNER = __name__.split(".", 1)[0]
MIME_ACTION = "MIME defect: report it with this build report."


def spec(shape=(3,), units="m", dtype=None):
    dtype = dtype or jnp.zeros(()).dtype
    return {"x": BoundaryInputSpec(shape=shape, dtype=dtype, default=jnp.zeros(shape, dtype=dtype),
                                   expected_units=units)}


def experiment(entry) -> Experiment:
    exp = toy_experiment("wrong_party")
    ADD_PROBE.apply(exp)
    exp.attach(entry(), body="probe")
    return exp


class LabNode(ToySpringNode):
    """A node of the lab's own, a class defined in the lab's package."""


@register_interaction
@dataclass(frozen=True, kw_only=True)
class LabOwnSpring(LabSpring):
    """The lab's spring built of the lab's own node class."""

    name = "wrongparty.own_spring"
    builds = {"": (LabNode,)}

    def build(self, ctx, b):
        node = b.add_response_node(LabNode(ctx.node_name(), ctx.timestep))
        b.connect(b.read("x"), into=node["x"])
        b.connect(node["load"], into=b.target("load"))


def mime_spring_of_the_labs_node(monkeypatch):
    """MIME's own entry, ``toy.spring``, building the lab's node class instead of MIME's."""
    def build(self, ctx, b):
        _respond(b, LabNode(ctx.node_name(), ctx.timestep, k=ctx.scalar("k"), x0=ctx.vector("x0")), "x", "load")

    monkeypatch.setattr(ToySpring, "build", build)
    monkeypatch.setattr(ToySpring, "builds", {"": (LabNode,)})
    return ToySpring


#: The four combinations: (the entry, a MIME class or the lab's, whether the lab's class is the one at fault).
COMBINATIONS = {
    "mime-class-under-mime-entry": (lambda monkeypatch: ToySpring, False),
    "mime-class-under-lab-entry": (lambda monkeypatch: LabSpring, False),
    "lab-class-under-lab-entry": (lambda monkeypatch: LabOwnSpring, True),
    "lab-class-under-mime-entry": (mime_spring_of_the_labs_node, True),
}


def refused(entry, monkeypatch, **flaw) -> list[Refusal]:
    monkeypatch.setattr(ToySpringNode, "boundary_input_spec", lambda self: spec(**flaw))
    with pytest.raises(BuildRefused) as error:
        experiment(entry).build()
    return list(error.value.refusals)


def assert_the_labs_to_fix(found: list[Refusal], cls: str = "LabNode"):
    for held in found:
        assert held.category is RefusalClass.INTERNAL and not held.mimes_defect and held.owner == OWNER, held
        assert f"({cls})" in held.message, "the node's class is named"
        assert "BoundaryInputSpec" in held.action or "boundary_input_spec" in held.action, held.action
        assert "report it to that package" in held.action and "report it with this build report" in held.action
        assert held.action.startswith(("make ", "give ")), "the author's change comes first"


def assert_mimes_defect(found: list[Refusal]):
    assert all(r.mimes_defect and r.owner is None and r.action == MIME_ACTION for r in found), found


@pytest.mark.parametrize("combination", COMBINATIONS)
@pytest.mark.parametrize(("flaw", "code"), [
    ({"shape": (4,)}, "maddening.shape"),
    ({"dtype": jnp.float16}, "maddening.dtype"),
    ({"units": "kg"}, "post-build.units"),
    ({"units": "kg"}, "maddening.units"),
    ({"units": ""}, "post-build.input-units"),
], ids=["shape", "dtype", "units-label", "units-warning", "no-units"])
def test_a_check_of_a_node_is_attributed_to_the_nodes_class_under_any_entry(combination, flaw, code, monkeypatch):
    entry, labs_class = COMBINATIONS[combination]
    found = [r for r in refused(entry(monkeypatch), monkeypatch, **flaw) if r.code == code]
    assert found, f"{code} is refused"
    if labs_class:
        assert_the_labs_to_fix(found)
    else:
        assert_mimes_defect(found)


def test_a_lab_s_node_under_a_lab_s_entry_names_the_entry_in_the_message(monkeypatch):
    found = [r for r in refused(LabOwnSpring, monkeypatch, shape=(4,)) if r.code == "maddening.shape"]
    assert found and all("wrongparty.own_spring" in r.message for r in found)


def test_a_node_that_does_not_keep_a_constructor_argument_is_the_labs_to_fix(monkeypatch):
    original = LabNode.__init__

    def lossy(self, name, timestep, scale=2.0):
        original(self, name, timestep)

    def build(self, ctx, b):
        node = b.add_response_node(LabNode(ctx.node_name(), ctx.timestep, scale=3.0))
        b.connect(b.read("x"), into=node["x"])
        b.connect(node["load"], into=b.target("load"))

    monkeypatch.setattr(LabNode, "__init__", lossy)
    monkeypatch.setattr(LabOwnSpring, "build", build)
    with pytest.raises(BuildRefused) as error:
        experiment(LabOwnSpring).build()
    (held,) = [r for r in error.value.refusals if r.owner == OWNER]
    assert held.code == "post-build.unsaved-arguments" and not held.mimes_defect
    assert "does not save the constructor argument scale" in held.message
    assert held.action.startswith("hand every constructor argument on to the base class's constructor")


def test_control_a_mime_node_that_does_not_keep_a_constructor_argument_under_a_lab_s_entry_is_mimes_defect(monkeypatch):
    original = ToySpringNode.__init__

    def lossy(self, name, timestep, scale=2.0):
        original(self, name, timestep)

    def build(self, ctx, b):
        node = b.add_response_node(ToySpringNode(ctx.node_name(), ctx.timestep, scale=3.0))
        b.connect(b.read("x"), into=node["x"])
        b.connect(node["load"], into=b.target("load"))

    monkeypatch.setattr(ToySpringNode, "__init__", lossy)
    monkeypatch.setattr(LabSpring, "build", build)
    with pytest.raises(BuildRefused) as error:
        experiment(LabSpring).build()
    found = [r for r in error.value.refusals if r.code == "post-build.unsaved-arguments"]
    assert found
    assert_mimes_defect(found)


def spring_of(graph) -> str:
    return next(name for name in graph.node_names if name != "probe")


def read_a_field_the_node_does_not_hold(graph, plans):
    spring = spring_of(graph)
    graph.remove_edge(spring, "probe", "load", "load")
    graph.add_edge(spring, "probe", "nofield", "load", additive=True, source_units="N", target_units="N")


def make_an_input_both_an_edge_and_an_external_input_target(graph, plans):
    graph.add_external_input(spring_of(graph), "x", shape=(3,), dtype=jnp.float64)


def validated(entry, mutation) -> list[Refusal]:
    with _testing.after_construction(mutation), pytest.raises(BuildRefused) as error:
        experiment(entry).build()
    return [r for r in error.value.refusals if r.code == "maddening.error"]


@pytest.mark.parametrize("combination", COMBINATIONS)
def test_an_edge_that_reads_a_field_a_node_does_not_hold_is_attributed_to_the_nodes_class(combination, monkeypatch):
    entry, labs_class = COMBINATIONS[combination]
    (held,) = [r for r in validated(entry(monkeypatch), read_a_field_the_node_does_not_hold)
               if "not in state of node" in r.message]
    if labs_class:
        assert not held.mimes_defect and held.owner == OWNER and "(LabNode)" in held.message
        assert held.action.startswith("hold the field the edge reads in the node's state")
        assert "report it to that package" in held.action
    else:
        assert_mimes_defect([held])


@pytest.mark.parametrize("combination", COMBINATIONS)
def test_the_other_errors_of_maddening_keep_their_text_and_stay_mimes_whatever_the_class(combination, monkeypatch):
    """Control: an input that is both an edge's target and an external input is a fault of the graph's wiring, which no
    node class decides. Only the message that fits (a field the node's state lacks) is reworded; this one keeps its text."""
    entry, _ = COMBINATIONS[combination]
    found = validated(entry(monkeypatch), make_an_input_both_an_edge_and_an_external_input_target)
    assert found and all("not in state of node" not in r.message for r in found)
    assert_mimes_defect(found)
    assert all(r.message.startswith("MADDENING's validate(): ERROR: ") for r in found)


def a_wrong_run_constant(monkeypatch):
    """A lab's node whose run-constants name a leaf it does not have, built into a graph."""
    monkeypatch.setattr(LabNode, "run_constants", lambda self, params: {"bogus": jnp.zeros(())}, raising=False)
    built = experiment(LabOwnSpring).build()
    lab = tuple(name for name in built.graph.node_names if isinstance(built.graph.get_node(name), LabNode))
    monkeypatch.setattr("mime.build._maddening.run_constant_nodes", lambda graph: lab)
    return built


def test_a_run_constant_that_is_no_leaf_of_a_lab_s_node_is_that_labs_to_fix(monkeypatch):
    from mime.build._maddening import graph_params, with_run_constants

    built = a_wrong_run_constant(monkeypatch)
    with pytest.raises(InternalError) as error:
        with_run_constants(built.graph, graph_params(built.graph))
    (held,) = error.value.refusals
    assert held.code == "run.run-constant" and held.owner == OWNER and not held.mimes_defect, held
    assert "LabNode" in held.message and "['bogus']" in held.message
    assert held.action.startswith("make each key run_constants() returns a key of the node's own parameters")
    assert held.why is not None and "same program" in held.why
