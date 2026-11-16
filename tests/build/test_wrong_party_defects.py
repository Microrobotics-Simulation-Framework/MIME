"""A check of the built graph that faults a node a lab's entry built is the lab's package's to fix, and says what to
change; the same check on a node of MIME's own entry is still MIME's defect (G10).

Before, a shape, units or dtype that a lab's node declared differently from the vocabulary's interface was refused as
``[internal] maddening.shape ... MIME defect: report it with this build report``, which sends a lab's author to report
the lab's own node to MIME. Now the refusal keeps its code, names the entry and its package as its owner, and its
``do:`` is the change the author makes. The controls plant the same flaw in a node of MIME's own toy entry: it stays
MIME's defect, so a rewording that blamed every package alike would fail them.
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
from mime.testing.toys import LOAD, POSITION, ToySpring, ToySpringNode
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


def spec(shape=(3,), units="m"):
    dtype = jnp.zeros(()).dtype
    return {"x": BoundaryInputSpec(shape=shape, dtype=dtype, default=jnp.zeros(shape, dtype=dtype),
                                   expected_units=units)}


def experiment(entry) -> Experiment:
    exp = toy_experiment("wrong_party")
    ADD_PROBE.apply(exp)
    exp.attach(entry(), body="probe")
    return exp


def refused(entry, monkeypatch, **flaw) -> list[Refusal]:
    monkeypatch.setattr(ToySpringNode, "boundary_input_spec", lambda self: spec(**flaw))
    with pytest.raises(BuildRefused) as error:
        experiment(entry).build()
    return list(error.value.refusals)


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


def labs(refusals: list[Refusal]) -> list[Refusal]:
    return [r for r in refusals if r.owner == __name__]


@pytest.mark.parametrize(("flaw", "code"), [
    ({"shape": (4,)}, "maddening.shape"),
    ({"units": "kg"}, "post-build.units"),
    ({"units": "kg"}, "maddening.units"),
    ({"units": ""}, "post-build.input-units"),
], ids=["shape", "units-label", "units-warning", "no-units"])
def test_a_check_of_a_node_the_labs_entry_built_is_the_labs_to_fix_and_says_what_to_change(flaw, code, monkeypatch):
    found = [r for r in labs(refused(LabSpring, monkeypatch, **flaw)) if r.code == code]
    assert found, "the lab's entry is named as the owner of the check it faults"
    for held in found:
        assert held.category is RefusalClass.INTERNAL and not held.mimes_defect
        assert "wrongparty.spring" in held.message and "BoundaryInputSpec" in held.action or "boundary_input_spec" in held.action
        assert "report it to that package" in held.action and "report it with this build report" in held.action
        assert held.action.startswith(("make ", "give ")), "the author's change comes first"


@pytest.mark.parametrize(("flaw", "code"), [
    ({"shape": (4,)}, "maddening.shape"),
    ({"units": "kg"}, "post-build.units"),
    ({"units": ""}, "post-build.input-units"),
], ids=["shape", "units-label", "no-units"])
def test_control_the_same_check_of_a_node_of_mimes_own_entry_stays_mimes_defect(flaw, code, monkeypatch):
    found = [r for r in refused(ToySpring, monkeypatch, **flaw) if r.code == code]
    assert found and all(r.mimes_defect and r.owner is None for r in found), found
    assert all(r.action == "MIME defect: report it with this build report." for r in found)


def test_a_node_that_does_not_keep_a_constructor_argument_is_the_labs_to_fix(monkeypatch):
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
    (held,) = labs(list(error.value.refusals))
    assert held.code == "post-build.unsaved-arguments" and not held.mimes_defect
    assert "does not save the constructor argument scale" in held.message
    assert held.action.startswith("hand every constructor argument on to the base class's constructor")


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
    assert held.code == "run.run-constant" and held.owner == __name__ and not held.mimes_defect, held
    assert "LabNode" in held.message and "['bogus']" in held.message
    assert held.action.startswith("make each key run_constants() returns a key of the node's own parameters")
    assert held.why is not None and "same program" in held.why
