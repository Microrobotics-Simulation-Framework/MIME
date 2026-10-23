"""MIME's nodes' typed state: each node's TypedDicts name exactly the fields it holds and the inputs it declares, as
the toy nodes' do (``test_node_fields.py``). Every node class in ``mime.nodes`` must be listed here."""

import importlib
import inspect
import pkgutil

import jax.numpy as jnp
import pytest
from maddening.core.node import SimulationNode
from typing_extensions import NoExtraItems

import mime.nodes
from mime.nodes import actuation, body, magnetic

DT = 1e-3
TUBE = dict(surface_shape="cylinder", extent=1e-3, margin=1e-4, velocity_rule="remove_outward",
            surface_centre=(0.0, 0.0, 0.0), surface_axis=(1.0, 0.0, 0.0), surface_radius=3e-3)

TYPED = [
    (body.RigidInertialNode("b", DT, mass=1e-4, inertia=(1e-10,) * 3, initial_pose=(0, 0, 0, 1, 0, 0, 0),
                            initial_twist=(0.0,) * 6, **TUBE), body.RigidInertialState, body.RigidInertialInputs,
     set()),
    (body.GravityNode("g", DT, mass=1e-4, buoyant_volume=6e-8, fluid_density=1e3, gravity=(0, 0, -9.81),
                      centre_of_buoyancy=(0, 0, 1e-4)), body.Wrench, body.GravityInputs, set()),
    (body.SwimSpeedNode("s", DT, along=(1, 0, 0)), body.SwimSpeedState, body.SwimSpeedInputs, set()),
    (magnetic.MagneticTorqueNode("t", DT, moment=(0, 1e-3, 0)), body.Wrench, magnetic.MagneticTorqueInputs, set()),
    (actuation.UniformRotatingFieldNode("f", DT, slots=("r",), rotation_axis=(0, 0, 1), initial_direction=(1, 0, 0)),
     actuation.UniformRotatingFieldState, actuation.UniformRotatingFieldInputs, {"flux_density_r", "gradient_r"}),
    (actuation.ConstantNode("c", DT, value=10.0), actuation.Command, None, set()),
]


def _declared(kind) -> set[str]:
    return set() if kind is None else set(kind.__annotations__)


def _extra(kind) -> bool:
    return kind is None or getattr(kind, "__extra_items__", NoExtraItems) is not NoExtraItems


def _node_classes() -> set[type]:
    found: set[type] = set()
    for info in pkgutil.walk_packages(mime.nodes.__path__, "mime.nodes."):
        module = importlib.import_module(info.name)
        found |= {cls for _, cls in inspect.getmembers(module, inspect.isclass)
                  if issubclass(cls, SimulationNode) and cls.__module__ == module.__name__}
    return found


def test_every_mime_node_class_has_its_typed_state_listed():
    assert _node_classes() == {type(node) for node, *_ in TYPED}


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
    assert _declared(inputs) <= declared_inputs | _declared(inputs)
    assert declared_inputs <= _declared(inputs) | constructed, declared_inputs - _declared(inputs)


def test_every_input_of_every_mime_node_declares_its_units():
    for node, *_ in TYPED:
        for name, spec in node.boundary_input_spec().items():
            assert spec.expected_units, f"{type(node).__name__}.{name}"
