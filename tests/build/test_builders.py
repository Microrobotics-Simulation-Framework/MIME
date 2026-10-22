"""The restricted builders: an entry that builds against its own plan is refused as a defect of that entry, named
with its module, after every entry has been asked."""

from dataclasses import dataclass

import pytest

from mime.build import BuildRefused, RefusalClass
from mime.gradients import GradientStatus
from mime.registry import Delivery, Interaction, Read, RoleKind, register_interaction
from mime.testing.toys import LOAD, POSITION, ToySpringNode, coupled_toy
from mime.vocabulary import register_namespace

register_namespace("building", "0.0.1")


@dataclass(frozen=True, kw_only=True)
class _Spring(Interaction):
    version = "0.0.1"
    differentiability = GradientStatus.EXACT
    roles = {"body": RoleKind.BODY}
    reads = {"x": Read("body", POSITION.name)}
    delivers = {"load": Delivery("body", LOAD.name)}


@register_interaction
@dataclass(frozen=True, kw_only=True)
class MisnamedNode(_Spring):
    name = "building.misnamed"

    def build(self, ctx, b):
        node = b.add_response_node(ToySpringNode("spring", ctx.timestep))
        b.connect(b.read("x"), into=node["x"])
        b.connect(node["load"], into=b.target("load"))


@register_interaction
@dataclass(frozen=True, kw_only=True)
class ReadsTwice(_Spring):
    name = "building.reads_twice"

    def build(self, ctx, b):
        node = b.add_response_node(ToySpringNode(ctx.node_name(), ctx.timestep))
        b.connect(b.read("x"), into=node["x"])
        b.connect(b.read("x"), into=node["x"])
        b.connect(node["load"], into=b.target("load"))


@register_interaction
@dataclass(frozen=True, kw_only=True)
class AddsNothing(_Spring):
    name = "building.adds_nothing"

    def build(self, ctx, b):
        return None


@register_interaction
@dataclass(frozen=True, kw_only=True)
class DeliversFromAnotherNode(_Spring):
    name = "building.delivers_from_another_node"

    def build(self, ctx, b):
        node = b.add_response_node(ToySpringNode(ctx.node_name(), ctx.timestep))
        b.connect(b.read("x"), into=node["x"])
        b.connect(node["load"], into=b.target("load"))
        b.connect(b.read("x"), into=b.target("load"))


@pytest.mark.parametrize(("entry", "problem"), [
    (MisnamedNode, "it named a node 'spring'; construct it with ctx.node_name('') = 'building.misnamed:probe'"),
    (ReadsTwice, "it connects 'x' 2 times, not once"),
    (AddsNothing, "it declares node '' and never adds it"),
    (DeliversFromAnotherNode, "read 'x' goes into TargetRef"),
])
def test_an_entry_building_against_its_plan_is_refused_as_its_defect(entry, problem):
    experiment = coupled_toy()
    experiment.attach(entry(), body="probe")
    with pytest.raises(BuildRefused) as refused:
        experiment.build()
    refusals = refused.value.refusals
    assert all(r.code == "entry.defect" and r.category is RefusalClass.INTERNAL for r in refusals)
    assert any(problem in r.message and "test_builders" in r.message for r in refusals)


def test_the_builders_set_each_edges_flag_from_its_target():
    built = coupled_toy().build()
    flags = {(e.source_node, e.target_node, e.target_field): e.additive for e in built.graph.edges}
    assert flags[("toy.spring:probe", "probe", "load")] is True
    assert flags[("bath.flow", "probe", "load")] is True
    assert flags[("probe", "toy.spring:probe", "x")] is False
