"""The toy nodes' typed state: each node's TypedDicts name exactly the fields it holds and the inputs it declares.

The TypedDicts are static (the checker holds every view and every state built from them; ``tests/typing`` has the
controls), so this test holds them to what the nodes do at run time: a TypedDict that drifted from its node would
type-check code against fields the node does not have. Every toy node class must be listed here.
"""

import inspect

import jax.numpy as jnp
import pytest
from maddening.core.node import SimulationNode
from typing_extensions import NoExtraItems

from mime.core import node_fields, typed_view
from mime.testing.toys import _nodes as toys

DT = 0.02

#: Each node, constructed as an entry would, with its state TypedDict (None when every field is named at
#: construction), its inputs TypedDict, and the fields named at construction.
TYPED = [
    (toys.ToyBodyNode("b", DT), toys.ToyBodyState, toys.ToyBodyInputs, set()),
    (toys.ToyAnchoredBodyNode("a", DT), toys.ToyAnchoredBodyState, toys.ToyAnchoredBodyInputs, set()),
    (toys.ToyMediumNode("m", DT, bodies=("p", "q")), toys.ToyMediumState, toys.ToyMediumInputs,
     {"force_p", "force_q", "pos_p", "pos_q"}),
    (toys.ToySpringNode("s", DT), toys.ToyLoad, toys.ToySpringInputs, set()),
    (toys.ToyDragNode("d", DT), toys.ToyLoad, toys.ToyDragInputs, set()),
    (toys.ToyLiftNode("l", DT), toys.ToyLoad, toys.ToyLiftInputs, set()),
    (toys.ToyHoldNode("h", DT), toys.ToyHoldState, toys.ToyHoldInputs, set()),
    (toys.ToyFieldNode("f", DT, slots=("r",)), None, toys.ToyFieldInputs, {"push_r", "points_r"}),
    (toys.ToyRecorderNode("r", DT), toys.ToyRecorderState, toys.ToyRecorderInputs, set()),
]


def _declared(kind) -> set[str]:
    return set() if kind is None else set(kind.__annotations__)


def _extra(kind) -> bool:
    """Whether ``kind`` admits fields beyond those it names: ``extra_items``, or no TypedDict at all."""
    return kind is None or getattr(kind, "__extra_items__", NoExtraItems) is not NoExtraItems


def test_every_toy_node_class_has_its_typed_state_listed():
    concrete = {cls for _, cls in inspect.getmembers(toys, inspect.isclass)
                if issubclass(cls, SimulationNode) and cls.__module__ == toys.__name__
                and cls not in (toys.ToyNode, toys.ResponseNode)}
    assert concrete == {type(node) for node, *_ in TYPED}


def test_a_closed_typeddict_admits_no_other_field():
    assert getattr(toys.ToyBodyState, "__closed__", None) is True and not _extra(toys.ToyBodyState)
    assert _extra(toys.ToyMediumState)


@pytest.mark.parametrize(("node", "state", "inputs", "constructed"), TYPED, ids=lambda v: type(v).__name__)
def test_each_nodes_typeddicts_name_the_fields_it_holds_and_the_inputs_it_declares(node, state, inputs,
                                                                                  constructed):
    initial = node.initial_state()
    given = {name: jnp.zeros(spec.shape) for name, spec in node.boundary_input_spec().items()}
    stepped = node.update(initial, given, DT)
    for fields in (set(initial), set(stepped)):
        assert _declared(state) <= fields
        extra = fields - _declared(state)
        assert extra <= constructed and (not extra or _extra(state)), extra
    declared_inputs = set(node.boundary_input_spec())
    assert _declared(inputs) <= declared_inputs
    extra_inputs = declared_inputs - _declared(inputs)
    assert extra_inputs <= constructed and (not extra_inputs or _extra(inputs)), extra_inputs


def test_the_views_are_static_only():
    fields = {"x": jnp.zeros(3), "v": jnp.ones(3)}
    assert typed_view(toys.ToyBodyState, fields) is fields
    assert node_fields(toys.ToyBodyState(x=fields["x"], v=fields["v"])) == fields
