"""A mutation per post-build check: each plants, between construction and the checks, the defect its check exists to
catch, and the build must refuse it with that check's code alone. The control: the same hook planting nothing
builds. A wrong units label is refused twice, by MIME's check and by MADDENING's own units warning, which the build
turns into a refusal."""

import jax.numpy as jnp
import pytest
from maddening.core.node import BoundaryInputSpec, SimulationNode

from mime.build import BuildRefused, RefusalClass, _testing
from mime.testing.toys import full_toy

SPRING_LOAD = ("toy.spring:probe", "probe", "load", "load")
LOAD_UNITS = {"source_units": "N", "target_units": "N"}


class _KeepsAnArgumentOutsideParams(SimulationNode):
    def __init__(self, name, timestep, gain=1.0):
        super().__init__(name, timestep)
        self.gain = gain

    def initial_state(self):
        return {"x": jnp.zeros(3)}

    def update(self, state, boundary_inputs, dt):
        return state


class _UnlabelledInput(SimulationNode):
    def __init__(self, name, timestep):
        super().__init__(name, timestep)

    def initial_state(self):
        return {"x": jnp.zeros(3)}

    def boundary_input_spec(self):
        return {"q": BoundaryInputSpec(shape=(3,), default=jnp.zeros(3))}

    def update(self, state, boundary_inputs, dt):
        return state


def flip_an_additive_flag(graph, plans):
    graph.remove_edge(*SPRING_LOAD)
    graph.add_edge(*SPRING_LOAD, additive=False, **LOAD_UNITS)


def add_a_second_edge_into_an_exclusive_target(graph, plans):
    graph.add_edge("toy.hold:probe", "probe", "hold_z", "hold_z", source_units="m/s", target_units="m/s")


def make_a_target_an_external_input(graph, plans):
    graph.add_external_input("probe", "load", shape=(3,), dtype=jnp.float64)


def put_a_transform_on_a_target_edge(graph, plans):
    graph.remove_edge(*SPRING_LOAD)
    graph.add_edge(*SPRING_LOAD, transform="negate", additive=True, **LOAD_UNITS)


def add_a_node_that_drops_a_constructor_argument(graph, plans):
    graph.add_node(_KeepsAnArgumentOutsideParams("stray", plans.dt, gain=5.0))


def drop_an_edges_units_labels(graph, plans):
    graph.remove_edge(*SPRING_LOAD)
    graph.add_edge(*SPRING_LOAD, additive=True)


def add_a_node_whose_input_declares_no_units(graph, plans):
    graph.add_node(_UnlabelledInput("unlabelled", plans.dt))


def label_an_edge_with_other_units(graph, plans):
    graph.remove_edge(*SPRING_LOAD)
    graph.add_edge(*SPRING_LOAD, additive=True, source_units="mN", target_units="mN")


MUTATIONS = {
    "post-build.additive": (flip_an_additive_flag, ["post-build.additive"]),
    "post-build.exclusive": (add_a_second_edge_into_an_exclusive_target, ["post-build.exclusive"]),
    # MADDENING refuses this one too, at validate(): its error becomes a refusal beside MIME's.
    "post-build.external-input": (make_a_target_an_external_input, ["post-build.external-input", "maddening.error"]),
    "post-build.transform": (put_a_transform_on_a_target_edge, ["post-build.transform"]),
    "post-build.unsaved-arguments": (add_a_node_that_drops_a_constructor_argument, ["post-build.unsaved-arguments"]),
    "post-build.units (missing)": (drop_an_edges_units_labels, ["post-build.units"]),
    "post-build.input-units": (add_a_node_whose_input_declares_no_units, ["post-build.input-units"]),
    # MADDENING warns of each wrong end, and the build turns both warnings into refusals.
    "post-build.units (wrong)": (label_an_edge_with_other_units,
                                 ["post-build.units", "maddening.units", "maddening.units"]),
}


def build_with(mutation):
    with _testing.after_construction(mutation):
        return full_toy().build()


def test_the_hook_planting_nothing_builds():
    built = build_with(lambda graph, plans: None)
    assert built.report.refusals == ()


@pytest.mark.parametrize("check", sorted(MUTATIONS))
def test_each_mutation_is_refused_by_its_check(check):
    mutation, codes = MUTATIONS[check]
    with pytest.raises(BuildRefused) as refused:
        build_with(mutation)
    print(f"\n{mutation.__name__}:\n{refused.value}")
    assert refused.value.codes() == codes
    assert all(r.category is RefusalClass.INTERNAL for r in refused.value.refusals)
    assert refused.value.report is not None and refused.value.report.refusals == refused.value.refusals


def test_a_wrong_units_label_is_refused_naming_the_edge_and_both_units():
    with pytest.raises(BuildRefused) as refused:
        build_with(label_an_edge_with_other_units)
    mime, *maddening = refused.value.refusals
    assert "toy.spring:probe.load -> probe.load is labelled 'mN' -> 'mN', and its target takes 'N'" in mime.message
    assert any("target_units='mN' but node expects 'N'" in r.message for r in maddening)


def test_every_edge_of_a_built_toy_carries_its_targets_units():
    graph = full_toy().build().graph
    assert graph.edges
    for edge in graph.edges:
        expected = graph.get_node(edge.target_node).boundary_input_spec()[edge.target_field].expected_units
        assert edge.source_units == edge.target_units == expected, edge
