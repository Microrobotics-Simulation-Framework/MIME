"""A mutation per post-build check: each plants, between construction and the checks, the defect its check exists to
catch, and the build must refuse it with that check's code alone. The control: the same hook planting nothing
builds."""

import jax.numpy as jnp
import pytest
from maddening.core.node import SimulationNode

from mime.build import BuildRefused, RefusalClass, _testing
from mime.testing.toys import full_toy

SPRING_LOAD = ("toy.spring:probe", "probe", "load", "load")


class _KeepsAnArgumentOutsideParams(SimulationNode):
    def __init__(self, name, timestep, gain=1.0):
        super().__init__(name, timestep)
        self.gain = gain

    def initial_state(self):
        return {"x": jnp.zeros(3)}

    def update(self, state, boundary_inputs, dt):
        return state


def flip_an_additive_flag(graph, plans):
    graph.remove_edge(*SPRING_LOAD)
    graph.add_edge(*SPRING_LOAD, additive=False)


def add_a_second_edge_into_an_exclusive_target(graph, plans):
    graph.add_edge("toy.hold:probe", "probe", "hold_z", "hold_z")


def make_a_target_an_external_input(graph, plans):
    graph.add_external_input("probe", "load", shape=(3,), dtype=jnp.float64)


def put_a_transform_on_a_target_edge(graph, plans):
    graph.remove_edge(*SPRING_LOAD)
    graph.add_edge(*SPRING_LOAD, transform="negate", additive=True)


def add_a_node_that_drops_a_constructor_argument(graph, plans):
    graph.add_node(_KeepsAnArgumentOutsideParams("stray", plans.dt, gain=5.0))


MUTATIONS = {
    "post-build.additive": flip_an_additive_flag,
    "post-build.exclusive": add_a_second_edge_into_an_exclusive_target,
    "post-build.external-input": make_a_target_an_external_input,
    "post-build.transform": put_a_transform_on_a_target_edge,
    "post-build.unsaved-arguments": add_a_node_that_drops_a_constructor_argument,
}


def build_with(mutation):
    with _testing.after_construction(mutation):
        return full_toy().build()


def test_the_hook_planting_nothing_builds():
    built = build_with(lambda graph, plans: None)
    assert built.report.refusals == ()


@pytest.mark.parametrize("code", sorted(MUTATIONS))
def test_each_mutation_is_refused_by_its_check(code):
    with pytest.raises(BuildRefused) as refused:
        build_with(MUTATIONS[code])
    print(f"\n{MUTATIONS[code].__name__}:\n{refused.value}")
    assert refused.value.codes() == [code]
    (refusal,) = refused.value.refusals
    assert refusal.category is RefusalClass.INTERNAL
    assert refused.value.report is not None and refused.value.report.refusals == (refusal,)
