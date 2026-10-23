"""Pins of MADDENING 0.4.x behaviour that MIME's build works around. Each passes on 0.4.2; when one fails (the
MADDENING-``main`` lane shows it first), MADDENING has changed and the workaround named in the test may go."""

import jax.numpy as jnp
import numpy as np
import pytest
from maddening import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

DT = 0.02


class _Constant(SimulationNode):
    def __init__(self, name, timestep, value=1.0):
        super().__init__(name, timestep, value=value)

    def initial_state(self):
        return {"value": jnp.full((3,), self.params["value"])}

    def update(self, state, boundary_inputs, dt):
        return state


class _Receiver(SimulationNode):
    COUPLING_TYPE = "additive"

    def __init__(self, name, timestep):
        super().__init__(name, timestep)

    def initial_state(self):
        return {"seen": jnp.zeros(3), "x": jnp.asarray([0.3, -0.2, 0.1])}

    def boundary_input_spec(self):
        return {"load": BoundaryInputSpec(shape=(3,), default=jnp.zeros(3), coupling_type=self.COUPLING_TYPE)}

    def update(self, state, boundary_inputs, dt):
        load = boundary_inputs.get("load", jnp.zeros(3))
        return {"seen": load, "x": state["x"] + dt * jnp.sin(load + state["x"])}


def _delivered(edges: list[tuple[str, bool]]) -> float:
    graph = GraphManager()
    for name, value in (("one", 1.0), ("ten", 10.0), ("hundred", 100.0)):
        graph.add_node(_Constant(name, DT, value=value))
    graph.add_node(_Receiver("receiver", DT))
    for source, additive in edges:
        graph.add_edge(source, "receiver", "value", "load", additive=additive)
    graph.compile()
    graph.step()
    return float(graph.get_node_state("receiver")["seen"][0])


@pytest.mark.parametrize(("edges", "delivered"), [
    ([("one", True), ("ten", True), ("hundred", False)], 100.0),
    ([("hundred", False), ("one", True), ("ten", True)], 111.0),
    ([("one", True), ("hundred", False), ("ten", True)], 110.0),
])
def test_a_non_additive_edge_replaces_only_what_was_summed_before_it(edges, delivered):
    """MIME's builders set ``additive`` from the target's kind and its post-build check refuses a non-additive edge
    into a summed target, because MADDENING lets the mix through and its result depends on edge order. If this
    fails, MADDENING now enforces or refuses the mix, and the check may defer to it."""
    assert _delivered(edges) == delivered


def _history_with(coupling_type: str, sources: tuple[str, ...] = ("one", "ten")) -> dict:
    receiver = type(f"Receiver_{coupling_type}", (_Receiver,), {"COUPLING_TYPE": coupling_type})
    graph = GraphManager()
    graph.add_node(_Constant("one", DT, value=1.0))
    graph.add_node(_Constant("ten", DT, value=10.0))
    graph.add_node(receiver("receiver", DT))
    for source in sources:
        graph.add_edge(source, "receiver", "value", "load", additive=True)
    graph.compile()
    _, history = graph.run_scan_with_history(20)
    return {field: np.asarray(value) for field, value in history["receiver"].items()}


def test_a_targets_coupling_type_is_documentation_only():
    """MIME relies on nothing ``coupling_type`` says: the builders set each edge's flag. If this fails, MADDENING
    now reads the declared combination, and the builder's flag and its check may come to rest on it."""
    reference = _history_with("additive")
    last_edge_only = _history_with("additive", ("ten",))
    assert not np.array_equal(reference["seen"], last_edge_only["seen"]), "the comparison cannot see a replacement"
    for declared in ("replacive", "not-a-coupling-type"):
        other = _history_with(declared)
        assert all(np.array_equal(reference[f], other[f]) for f in reference), declared


def test_compile_refuses_an_edge_into_a_field_that_is_also_an_external_input():
    """MIME keeps its own check for this, which runs before ``compile()`` and names the target in the experiment's
    terms; this is MADDENING's refusal behind it. If this fails, MIME's check is the only one left."""
    graph = GraphManager()
    graph.add_node(_Constant("one", DT))
    graph.add_node(_Receiver("receiver", DT))
    graph.add_edge("one", "receiver", "value", "load", additive=True)
    graph.add_external_input("receiver", "load", shape=(3,))
    with pytest.raises(ValueError, match="an edge and a declared external input target the same field"):
        graph.compile()


def test_the_inputs_a_node_receives_resolve_through_the_public_interface_the_probe_uses():
    """The build's callback probe traces each node alone through MADDENING's public node interface: its state, the
    inputs ``resolve_boundary_inputs`` gives it, ``update`` and ``compute_boundary_fluxes``. This pins that a node
    receives there what its edges deliver, and that a node defining no fluxes returns none. If this fails, the probe's
    inputs no longer match what the compiled step gives the node."""
    graph = GraphManager()
    graph.add_node(_Constant("two", DT, value=2.0))
    graph.add_node(_Receiver("receiver", DT))
    graph.add_edge("two", "receiver", "value", "load", additive=True)
    graph.compile()
    delivered = graph.resolve_boundary_inputs("receiver")
    np.testing.assert_array_equal(np.asarray(delivered["load"]), np.full(3, 2.0))
    receiver = graph.get_node("receiver")
    assert receiver.compute_boundary_fluxes(graph.get_node_state("receiver"), delivered, DT) == {}
